import argparse
import csv
import json
import os
from pathlib import Path
from itertools import permutations

import numpy as np
import torch
from PIL import Image
import nibabel as nib
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from scipy.ndimage import label
from tqdm import tqdm

from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.variants.MonaiPredictors import UNETRPredictor, UNetPlusPlusPredictor
from nnunetv2.inference.variants.UNeXtPredictor import UNeXtPredictor
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results
from nnunetv2.run.run_training import get_trainer_from_args
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_id", type=int, required=True)
    parser.add_argument("--test_dataset_id", type=int, required=True)
    parser.add_argument("--save_preds", type=str2bool, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--chk", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument("--plans", type=str, required=True)
    parser.add_argument("--trainer", type=str, default="nnUNetTrainer")
    parser.add_argument("--fold", type=str, default="all")
    parser.add_argument("--ensemble", type=str2bool, default=False)
    parser.add_argument("--cfg", type=str, default="2d")
    parser.add_argument("--results_csv", type=str, default="results.csv", help="CSV file to save results")
    parser.add_argument("--largest_component", type=str2bool, default=False)
    return parser.parse_args()


def get_dataset_name(nnunet_raw, dataset_id):
    dataset_id = str(dataset_id).zfill(3)
    dataset_names = [
        d
        for d in os.listdir(nnunet_raw)
        if d.startswith(f"Dataset{dataset_id}") and os.path.isdir(join(nnunet_raw, d))
    ]
    assert len(dataset_names) == 1, f"Found {len(dataset_names)} datasets with id {dataset_id}, expected 1"
    return dataset_names[0]


def load_dataset_json(nnunet_raw, dataset_name):
    with open(join(nnunet_raw, dataset_name, "dataset.json"), "r") as f:
        return json.load(f)


def get_img_ids_from_splits(nnunet_preprocessed, dataset_name, fold, split):
    assert split in ["Tr", "Val"], f"split must be either 'Tr' or 'Val', got {split}"
    split_key = "val" if split == "Val" else "train"

    with open(join(nnunet_preprocessed, dataset_name, "splits_final.json"), "r") as f:
        splits = json.load(f)

    if fold == "all":
        return splits[0]["train"] + splits[0]["val"]
    return splits[int(fold)][split_key]


def get_img_ids_from_val_folder(model_dir, fold):
    val_folder = join(model_dir, f"fold_{fold}", "validation")
    return [f.replace(".png", "") for f in os.listdir(val_folder) if f.endswith(".png")]


def prepare_input_img_paths(nnunet_raw, nnunet_preprocessed, dataset_name, fold, split, outdir=None, trainer=None, model_dir=None):
    dataset_json = load_dataset_json(nnunet_raw, dataset_name)
    if "PercentSplit" in trainer:
        img_ids = get_img_ids_from_val_folder(model_dir, fold)
    else:
        img_ids = get_img_ids_from_splits(nnunet_preprocessed, dataset_name, fold, split)

    img_paths = []
    out_paths = [] if outdir is not None else None
    for img_id in img_ids:
        img_channels = []
        for channel_id in dataset_json["channel_names"].keys():
            img_channel = join(
                nnunet_raw,
                dataset_name,
                "imagesTr",
                f"{img_id}_{str(channel_id).zfill(4)}{dataset_json['file_ending']}",
            )
            assert os.path.exists(img_channel), f"Image {img_channel} does not exist"
            img_channels.append(img_channel)
        img_paths.append(img_channels)
        if out_paths is not None:
            out_paths.append(join(outdir, img_id))
    return img_paths, out_paths


def get_predictor_class(trainer):
    if "UNeXtTrainer" in trainer:
        return UNeXtPredictor
    if trainer == "UNetPlusPlusTrainer":
        return UNetPlusPlusPredictor
    if trainer == "UNETRTrainer":
        return UNETRPredictor
    if trainer == "nnUNetTrainerLightMUNet":
        from nnunetv2.inference.variants.LightMUNetPredictor import LightMUNetPredictor

        return LightMUNetPredictor
    return nnUNetPredictor


def load_pretrained_predictor_and_minibatch(train_dataset_name, plans, trainer, cfg, fold, chk, device):
    fold_int = int(fold) if fold != "all" else 0
    nnunet_trainer = get_trainer_from_args(train_dataset_name, cfg, fold_int, trainer, plans, device=device)
    nnunet_trainer.enable_deep_supervision = False
    nnunet_trainer.initialize()
    mini_batch_size = int(nnunet_trainer.configuration_manager.batch_size)

    predictor_cls = get_predictor_class(trainer)
    predictor = predictor_cls(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    model_dir = join(nnUNet_results, train_dataset_name, f"{trainer}__{plans}__{cfg}")
    predictor.initialize_from_trained_model_folder(
        model_dir,
        use_folds=(fold_int,),
        checkpoint_name=chk,
    )
    return predictor, mini_batch_size


def find_largest_component_per_class(segmentation, num_classes):
    output = np.zeros_like(segmentation)
    for cls in range(1, num_classes):
        binary_mask = (segmentation == cls).astype(np.uint8)
        labeled_array, num_features = label(binary_mask)
        if num_features == 0:
            continue
        largest_label = max(range(1, num_features + 1), key=lambda x: np.sum(labeled_array == x))
        output[labeled_array == largest_label] = cls
    return output


def chunk_list(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def get_label_path_from_input(input_case, split):
    split_for_labels = "Tr" if split == "Val" else split
    first_channel_path = input_case[0]
    label_path = first_channel_path.replace(f"images{split_for_labels}", f"labels{split_for_labels}")
    label_path = label_path.replace("_0000.png", ".png")
    label_path = label_path.replace("_0000.nii.gz", ".nii.gz")
    label_path = label_path.replace("_0000.nii", ".nii")
    return label_path


def extract_image_id(input_case):
    p = Path(input_case[0]).name
    stem = p[:-7] if p.endswith(".nii.gz") else Path(p).stem
    if stem.endswith("_0000"):
        stem = stem[:-5]
    return stem


def load_label_array(label_path):
    if label_path.endswith(".nii.gz") or label_path.endswith(".nii"):
        arr = np.asarray(nib.load(label_path).get_fdata())
        # segmentation labels should be integer-valued
        return arr.astype(np.int32, copy=False)
    return np.array(Image.open(label_path))


def align_label_to_prediction_shape(label_arr, pred_arr):
    if label_arr.shape == pred_arr.shape:
        return label_arr

    # Common case for NIfTI labels: axis ordering differs (for example HWD vs DHW).
    if label_arr.ndim == pred_arr.ndim == 3:
        for perm in permutations(range(3)):
            cand = np.transpose(label_arr, perm)
            if cand.shape == pred_arr.shape:
                return cand

    return None


def _as_int_list(v):
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    return [int(v)]


def infer_multilabel_from_dataset_json(dataset_json):
    labels = dataset_json.get("labels", {})
    if not isinstance(labels, dict) or len(labels) == 0:
        return False

    non_bg_vals = [v for k, v in labels.items() if str(k).lower() != "background"]
    has_region_lists = any(isinstance(v, (list, tuple)) for v in non_bg_vals)
    has_regions_order = isinstance(dataset_json.get("regions_class_order", None), list)
    return bool(has_region_lists and has_regions_order)


def build_multilabel_region_defs(dataset_json):
    labels = dataset_json.get("labels", {})
    if not isinstance(labels, dict) or len(labels) == 0:
        return []

    non_bg_items = [(k, v) for k, v in labels.items() if str(k).lower() != "background"]
    if len(non_bg_items) == 0:
        return []

    regions_class_order = dataset_json.get("regions_class_order", None)
    region_defs = []

    if isinstance(regions_class_order, list) and len(regions_class_order) == len(non_bg_items):
        # Region-based nnU-Net setup with label-map output:
        # prediction is a single integer map (0/1/2/3...), so each region mask
        # must be reconstructed from its full label set (for example WT={1,2,3}).
        # regions_class_order defines region ordering, not singleton mask ids.
        for (name, target_v), _pred_id in zip(non_bg_items, regions_class_order):
            ids = _as_int_list(target_v)
            region_defs.append(
                {
                    "name": str(name),
                    "target_ids": ids,
                    "pred_ids": ids,
                }
            )
        return region_defs

    # Fallback: assume standard class ids and match pred ids to target ids.
    for name, target_v in non_bg_items:
        ids = _as_int_list(target_v)
        region_defs.append(
            {
                "name": str(name),
                "target_ids": ids,
                "pred_ids": ids,
            }
        )
    return region_defs


def convert_to_multilabel_tensors(pred_arr, label_arr, region_defs, include_background=True):
    pred_ch = []
    label_ch = []
    pred_any = np.zeros_like(pred_arr, dtype=bool)
    label_any = np.zeros_like(label_arr, dtype=bool)
    for rd in region_defs:
        p = np.isin(pred_arr, rd["pred_ids"])
        l = np.isin(label_arr, rd["target_ids"])
        pred_any |= p
        label_any |= l
        pred_ch.append(p.astype(np.float32))
        label_ch.append(l.astype(np.float32))

    if include_background:
        bg_pred = (~pred_any).astype(np.float32)
        bg_label = (~label_any).astype(np.float32)
        pred_ch = [bg_pred] + pred_ch
        label_ch = [bg_label] + label_ch

    pred_ml = np.stack(pred_ch, axis=0)
    label_ml = np.stack(label_ch, axis=0)
    pred_t = torch.tensor(pred_ml, dtype=torch.float32).unsqueeze(0)
    label_t = torch.tensor(label_ml, dtype=torch.float32).unsqueeze(0)
    return pred_t, label_t


def convert_to_multiclass_onehot_tensors(pred_arr, label_arr, num_classes):
    pred_ch = []
    label_ch = []
    for cls in range(num_classes):
        pred_ch.append((pred_arr == cls).astype(np.float32))
        label_ch.append((label_arr == cls).astype(np.float32))

    pred_oh = np.stack(pred_ch, axis=0)
    label_oh = np.stack(label_ch, axis=0)
    pred_t = torch.tensor(pred_oh, dtype=torch.float32).unsqueeze(0)
    label_t = torch.tensor(label_oh, dtype=torch.float32).unsqueeze(0)
    return pred_t, label_t


def get_foreground_class_infos(dataset_json, multi_label):
    if multi_label:
        return [{"index": i + 1, "name": rd["name"]} for i, rd in enumerate(build_multilabel_region_defs(dataset_json))]

    labels = dataset_json.get("labels", {})
    class_infos = []
    for name, value in labels.items():
        if str(name).lower() == "background":
            continue
        if isinstance(value, (list, tuple)):
            continue
        class_infos.append({"index": int(value), "name": str(name)})
    class_infos.sort(key=lambda x: x["index"])
    return class_infos


def metric_tensor_to_list(metric_value):
    arr = torch.as_tensor(metric_value).detach().cpu().numpy().astype(float).reshape(-1)
    return [float(x) for x in arr]


def finite_stats(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    mean = float(np.mean(arr))
    std = float(np.std(arr))  # population std to avoid NaN for n=1
    return mean, std


def build_table_row_values(per_class_metrics, dice_score, dice_std, hd95_score, hd95_std):
    preferred_class_order = ["Capsule", "Cortex", "Medulla", "CEC"]
    metrics_by_name = {}
    for class_name, class_store in per_class_metrics.items():
        metrics_by_name[class_name.lower()] = (class_name, class_store)

    def _fmt_pm(mean, std):
        if mean is None or std is None:
            return "-"
        return f"{float(mean):.2f}±{float(std):.2f}"

    row_values = []
    for preferred_name in preferred_class_order:
        match = metrics_by_name.get(preferred_name.lower())
        if match is None:
            row_values.extend(["-", "-"])
            continue

        _, class_store = match
        dice_score_cls, dice_std_cls = finite_stats(class_store["dice"])
        hd95_score_cls, hd95_std_cls = finite_stats(class_store["hd95"])
        row_values.append(_fmt_pm(dice_score_cls, dice_std_cls))
        row_values.append(_fmt_pm(hd95_score_cls, hd95_std_cls))

    row_values.append(_fmt_pm(dice_score, dice_std))
    row_values.append(_fmt_pm(hd95_score, hd95_std))
    return row_values


def evaluate_and_save_streaming(
    predictor,
    input_cases,
    output_targets,
    test_dataset_name,
    dataset_json,
    split,
    mini_batch_size,
    largest_component,
    num_classes,
    multi_label,
    results_csv_path,
    overwrite,
):
    # Reuse metric objects across all cases to avoid per-case re-instantiation overhead.
    d_img = DiceMetric(include_background=False, reduction="mean", ignore_empty=False)
    h_img = HausdorffDistanceMetric(
        include_background=False,
        reduction="mean",
        percentile=95,
    )
    m_img = SurfaceDistanceMetric(include_background=False, reduction="mean")
    d_img_per_class = DiceMetric(include_background=False, reduction="none", ignore_empty=False)
    h_img_per_class = HausdorffDistanceMetric(
        include_background=False,
        reduction="none",
        percentile=95,
    )
    m_img_per_class = SurfaceDistanceMetric(include_background=False, reduction="none")

    image_wise_csv_path = os.path.join(
        os.path.dirname(results_csv_path),
        f"image_wise_{os.path.basename(results_csv_path).replace('.csv', '')}_{test_dataset_name}.csv",
    )
    image_wise_csv_tmp_path = f"{image_wise_csv_path}.tmp"
    os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(image_wise_csv_path), exist_ok=True)

    region_defs = build_multilabel_region_defs(dataset_json) if multi_label else []
    class_infos = get_foreground_class_infos(dataset_json, multi_label)
    per_class_metrics = {
        ci["name"]: {"dice": [], "hd95": [], "masd": [], "index": ci["index"]} for ci in class_infos
    }
    if multi_label:
        if len(region_defs) == 0:
            raise RuntimeError("multi_label=True but no valid labels/regions found in dataset.json")
        region_info = ", ".join(
            [f"{r['name']} pred={r['pred_ids']} target={r['target_ids']}" for r in region_defs]
        )
        print(f"Multi-label evaluation enabled (include_background=False) with regions: {region_info}")

    def _finite_or_none(x):
        x = float(x)
        return x if np.isfinite(x) else None

    def _fmt_csv_value(x, nd=2):
        return "" if x is None else f"{float(x):.{nd}f}"

    with open(image_wise_csv_tmp_path, "w", newline="") as f_img:
        img_writer = csv.DictWriter(f_img, fieldnames=["image_id", "dice", "hd95", "masd"])
        img_writer.writeheader()

        n_total = 0
        dice_vals = []
        hd95_vals = []
        masd_vals = []
        metrics_pbar = tqdm(total=len(input_cases), desc="Metrics", unit="case", leave=True)
        try:
            for idx_chunk, case_chunk in enumerate(chunk_list(input_cases, mini_batch_size)):
                out_chunk = None if output_targets is None else output_targets[idx_chunk * mini_batch_size : idx_chunk * mini_batch_size + len(case_chunk)]

                preds = predictor.predict_from_files(
                    case_chunk,
                    out_chunk,
                    save_probabilities=False,
                    overwrite=overwrite,
                    num_processes_preprocessing=2,
                    num_processes_segmentation_export=2,
                    folder_with_segs_from_prev_stage=None,
                    num_parts=1,
                    part_id=0,
                )

                for case, pred in zip(case_chunk, preds):
                    if pred is None:
                        metrics_pbar.update(1)
                        continue
                    if largest_component:
                        pred = find_largest_component_per_class(pred, num_classes)

                    label_path = get_label_path_from_input(case, split)
                    if not os.path.exists(label_path):
                        metrics_pbar.update(1)
                        continue
                    label = load_label_array(label_path)

                    pred_arr = np.squeeze(np.asarray(pred))
                    label_arr = np.squeeze(np.asarray(label))

                    # Harmonize singleton dimensions (for example prediction as [1, H, W]).
                    if pred_arr.ndim == label_arr.ndim + 1 and pred_arr.shape[0] == 1:
                        pred_arr = pred_arr[0]
                    if label_arr.ndim == pred_arr.ndim + 1 and label_arr.shape[0] == 1:
                        label_arr = label_arr[0]

                    if pred_arr.ndim != label_arr.ndim:
                        print(
                            f"warning: skip {extract_image_id(case)} due to incompatible dims: "
                            f"pred {pred_arr.shape} vs label {label_arr.shape}"
                        )
                        metrics_pbar.update(1)
                        continue

                    if pred_arr.shape != label_arr.shape:
                        aligned_label = align_label_to_prediction_shape(label_arr, pred_arr)
                        if aligned_label is None:
                            print(
                                f"warning: skip {extract_image_id(case)} due to incompatible shapes: "
                                f"pred {pred_arr.shape} vs label {label_arr.shape}"
                            )
                            metrics_pbar.update(1)
                            continue
                        label_arr = aligned_label

                    if pred_arr.ndim not in (2, 3):
                        print(
                            f"warning: skip {extract_image_id(case)} due to unsupported dims: "
                            f"pred {pred_arr.shape} vs label {label_arr.shape}"
                        )
                        metrics_pbar.update(1)
                        continue

                    if multi_label:
                        pred_t, label_t = convert_to_multilabel_tensors(
                            pred_arr,
                            label_arr,
                            region_defs,
                            include_background=True,
                        )
                    else:
                        pred_t, label_t = convert_to_multiclass_onehot_tensors(
                            pred_arr,
                            label_arr,
                            num_classes,
                        )

                    # per-image metrics (saved immediately); metrics objects are reused.
                    d_img.reset()
                    h_img.reset()
                    m_img.reset()
                    d_img_per_class.reset()
                    h_img_per_class.reset()
                    m_img_per_class.reset()
                    d_img(pred_t, label_t)
                    h_img(pred_t, label_t)
                    m_img(pred_t, label_t)
                    d_img_per_class(pred_t, label_t)
                    h_img_per_class(pred_t, label_t)
                    m_img_per_class(pred_t, label_t)

                    dice_val = float(d_img.aggregate().item() * 100)
                    hd95_val = float(h_img.aggregate().item())
                    masd_val = float(m_img.aggregate().item())
                    dice_per_class = [v * 100 for v in metric_tensor_to_list(d_img_per_class.aggregate())]
                    hd95_per_class = metric_tensor_to_list(h_img_per_class.aggregate())
                    masd_per_class = metric_tensor_to_list(m_img_per_class.aggregate())
                    dice_val = _finite_or_none(dice_val)
                    hd95_val = _finite_or_none(hd95_val)
                    masd_val = _finite_or_none(masd_val)
                    for ci, dice_cls, hd95_cls, masd_cls in zip(
                        class_infos, dice_per_class, hd95_per_class, masd_per_class
                    ):
                        class_store = per_class_metrics[ci["name"]]
                        dice_cls = _finite_or_none(dice_cls)
                        hd95_cls = _finite_or_none(hd95_cls)
                        masd_cls = _finite_or_none(masd_cls)
                        if dice_cls is not None:
                            class_store["dice"].append(dice_cls)
                        if hd95_cls is not None:
                            class_store["hd95"].append(hd95_cls)
                        if masd_cls is not None:
                            class_store["masd"].append(masd_cls)
                    if dice_val is not None:
                        dice_vals.append(dice_val)
                    if hd95_val is not None:
                        hd95_vals.append(hd95_val)
                    if masd_val is not None:
                        masd_vals.append(masd_val)

                    img_writer.writerow(
                        {
                            "image_id": extract_image_id(case),
                            "dice": _fmt_csv_value(dice_val),
                            "hd95": _fmt_csv_value(hd95_val),
                            "masd": _fmt_csv_value(masd_val),
                        }
                    )
                    n_total += 1
                    metrics_pbar.update(1)
        finally:
            metrics_pbar.close()

    if n_total == 0:
        if os.path.exists(image_wise_csv_tmp_path):
            os.remove(image_wise_csv_tmp_path)
        print("No evaluable predictions found.")
        return
    os.replace(image_wise_csv_tmp_path, image_wise_csv_path)

    dice_score, dice_std = finite_stats(dice_vals)
    hd95_score, hd95_std = finite_stats(hd95_vals)
    masd_score, masd_std = finite_stats(masd_vals)

    print("\n")
    dice_msg = f"{dice_score:.2f}% ± {dice_std:.2f}%" if dice_score is not None else "n/a"
    hd95_msg = f"{hd95_score:.2f} ± {hd95_std:.2f}" if hd95_score is not None else "n/a"
    masd_msg = f"{masd_score:.2f} ± {masd_std:.2f}" if masd_score is not None else "n/a"
    print(f"Dice: {dice_msg}")
    print(f"HD95: {hd95_msg}")
    print(f"MASD: {masd_msg}")

    with open(results_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["test_dataset_name", "class_index", "class_name", "dice", "dice_std", "hd95", "hd95_std", "masd", "masd_std"],
        )
        writer.writeheader()

        for ci in class_infos:
            class_name = ci["name"]
            class_store = per_class_metrics[class_name]
            dice_score_cls, dice_std_cls = finite_stats(class_store["dice"])
            hd95_score_cls, hd95_std_cls = finite_stats(class_store["hd95"])
            masd_score_cls, masd_std_cls = finite_stats(class_store["masd"])
            writer.writerow(
                {
                    "test_dataset_name": test_dataset_name,
                    "class_index": ci["index"],
                    "class_name": class_name,
                    "dice": _fmt_csv_value(dice_score_cls),
                    "dice_std": _fmt_csv_value(dice_std_cls),
                    "hd95": _fmt_csv_value(hd95_score_cls),
                    "hd95_std": _fmt_csv_value(hd95_std_cls),
                    "masd": _fmt_csv_value(masd_score_cls),
                    "masd_std": _fmt_csv_value(masd_std_cls),
                }
            )

    table_row_txt_path = os.path.join(os.path.dirname(results_csv_path), f"{Path(results_csv_path).stem}_table_row.txt")
    table_row_values = build_table_row_values(per_class_metrics, dice_score, dice_std, hd95_score, hd95_std)
    table_row_text = ",".join(table_row_values)
    with open(table_row_txt_path, "w") as f:
        f.write(table_row_text + "\n")

    print(f"Per-class results saved to: {results_csv_path}")
    print(f"Table row saved to: {table_row_txt_path}")
    print(f"Table row: {table_row_text}")
    print(f"Image-wise results saved to: {image_wise_csv_path}\n")


def main():
    args = parse_args()

    train_dataset_id = args.train_dataset_id
    test_dataset_id = args.test_dataset_id
    split = args.split
    if split not in ["Tr", "Val", "Ts"]:
        raise ValueError(f"split must be either 'Tr', 'Val' or 'Ts', got {split}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_dataset_name = get_dataset_name(nnUNet_raw, train_dataset_id)
    test_dataset_name = get_dataset_name(nnUNet_raw, test_dataset_id)
    model_dir = join(nnUNet_results, train_dataset_name, f"{args.trainer}__{args.plans}__{args.cfg}")
    print("model_dir:", model_dir)

    eval_folds = [int(args.fold)]
    if train_dataset_name == test_dataset_name and split != "Ts" and args.fold == "all":
        eval_folds = [f for f in range(5) if os.path.exists(join(model_dir, f"fold_{f}"))]

    dataset_json = load_dataset_json(nnUNet_raw, test_dataset_name)
    num_classes = len(dataset_json["labels"])
    multi_label_mode = infer_multilabel_from_dataset_json(dataset_json)
    print(f"Evaluation mode: {'multi-label' if multi_label_mode else 'multi-class'}")

    for f in eval_folds:
        predictor, mini_batch_size = load_pretrained_predictor_and_minibatch(
            train_dataset_name=train_dataset_name,
            plans=args.plans,
            trainer=args.trainer,
            cfg=args.cfg,
            fold=f,
            chk=args.chk,
            device=device,
        )
        print(f"Using mini_batch_size={mini_batch_size} (from plans) for fold {f}")

        if split == "Ts":
            source_folder = join(nnUNet_raw, test_dataset_name, "imagesTs")
            assert os.path.exists(source_folder), f"Input directory {source_folder} does not exist"
            input_cases = create_lists_from_splitted_dataset_folder(source_folder, dataset_json["file_ending"])
            out_dir = join(model_dir, f"fold_{f}", "test", test_dataset_name, "preds") if args.save_preds else None
            output_targets = None if out_dir is None else [join(out_dir, extract_image_id(case)) for case in input_cases]
        elif split == "Val":
            out_dir = join(model_dir, f"fold_{f}", "test", test_dataset_name, "preds") if args.save_preds else None
            input_cases, output_targets = prepare_input_img_paths(
                nnUNet_raw,
                nnUNet_preprocessed,
                test_dataset_name,
                fold=str(f),
                split=split,
                outdir=out_dir,
                trainer=args.trainer,
                model_dir=model_dir,
            )
        else:
            out_dir = join(model_dir, f"fold_{f}", "validation") if args.save_preds else None
            input_cases, output_targets = prepare_input_img_paths(
                nnUNet_raw,
                nnUNet_preprocessed,
                train_dataset_name,
                fold=str(f),
                split=split,
                outdir=out_dir,
                trainer=args.trainer,
                model_dir=model_dir,
            )

        if not input_cases:
            print(f"No cases found for fold {f}, split {split}")
            continue

        results_csv_path = join(model_dir, f"fold_{f}", "test", args.results_csv)
        if args.largest_component:
            results_csv_path = results_csv_path.replace(".csv", "_largest_component.csv")

        evaluate_and_save_streaming(
            predictor=predictor,
            input_cases=input_cases,
            output_targets=output_targets,
            test_dataset_name=test_dataset_name,
            dataset_json=dataset_json,
            split=split,
            mini_batch_size=mini_batch_size,
            largest_component=args.largest_component,
            num_classes=num_classes,
            multi_label=multi_label_mode,
            results_csv_path=results_csv_path,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
