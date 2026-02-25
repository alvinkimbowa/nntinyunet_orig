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


def evaluate_and_save_streaming(
    predictor,
    input_cases,
    output_targets,
    test_dataset_name,
    split,
    mini_batch_size,
    largest_component,
    num_classes,
    results_csv_path,
    overwrite,
):
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95)
    masd_metric = SurfaceDistanceMetric(include_background=False, reduction="mean")

    image_wise_csv_path = os.path.join(
        os.path.dirname(results_csv_path),
        f"image_wise_{os.path.basename(results_csv_path).replace('.csv', '')}_{test_dataset_name}.csv",
    )
    os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(image_wise_csv_path), exist_ok=True)

    with open(image_wise_csv_path, "w", newline="") as f_img:
        img_writer = csv.DictWriter(f_img, fieldnames=["image_id", "dice", "hd95", "masd"])
        img_writer.writeheader()

        n_total = 0
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
                    continue
                if largest_component:
                    pred = find_largest_component_per_class(pred, num_classes)

                label_path = get_label_path_from_input(case, split)
                if not os.path.exists(label_path):
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
                    continue

                if pred_arr.shape != label_arr.shape:
                    aligned_label = align_label_to_prediction_shape(label_arr, pred_arr)
                    if aligned_label is None:
                        print(
                            f"warning: skip {extract_image_id(case)} due to incompatible shapes: "
                            f"pred {pred_arr.shape} vs label {label_arr.shape}"
                        )
                        continue
                    label_arr = aligned_label

                if pred_arr.ndim == 2:
                    pred_t = torch.tensor(pred_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    label_t = torch.tensor(label_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                elif pred_arr.ndim == 3:
                    pred_t = torch.tensor(pred_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    label_t = torch.tensor(label_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                else:
                    print(
                        f"warning: skip {extract_image_id(case)} due to unsupported dims: "
                        f"pred {pred_arr.shape} vs label {label_arr.shape}"
                    )
                    continue

                # update global metrics
                dice_metric(pred_t, label_t)
                hd95_metric(pred_t, label_t)
                masd_metric(pred_t, label_t)

                # per-image metrics (saved immediately)
                d_tmp = DiceMetric(include_background=False, reduction="mean")
                h_tmp = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95)
                m_tmp = SurfaceDistanceMetric(include_background=False, reduction="mean")
                d_tmp(pred_t, label_t)
                h_tmp(pred_t, label_t)
                m_tmp(pred_t, label_t)

                dice_val = float(d_tmp.aggregate().item() * 100)
                hd95_val = float(h_tmp.aggregate().item())
                masd_val = float(m_tmp.aggregate().item())

                img_writer.writerow(
                    {
                        "image_id": extract_image_id(case),
                        "dice": f"{dice_val:.2f}",
                        "hd95": f"{hd95_val:.2f}",
                        "masd": f"{masd_val:.2f}",
                    }
                )
                n_total += 1

    if n_total == 0:
        print("No evaluable predictions found.")
        return

    dice_score = float(dice_metric.aggregate().item() * 100)
    dice_std = float(dice_metric.get_buffer().std().item() * 100)
    hd95_score = float(hd95_metric.aggregate().item())
    hd95_std = float(hd95_metric.get_buffer().std().item())
    masd_score = float(masd_metric.aggregate().item())
    masd_std = float(masd_metric.get_buffer().std().item())

    print("\n")
    print(f"Dice: {dice_score:.2f}% ± {dice_std:.2f}%")
    print(f"HD95: {hd95_score:.2f} ± {hd95_std:.2f}")
    print(f"MASD: {masd_score:.2f} ± {masd_std:.2f}")

    csv_exists = os.path.exists(results_csv_path) and os.path.getsize(results_csv_path) > 0
    with open(results_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["test_dataset_name", "dice", "dice_std", "hd95", "hd95_std", "masd", "masd_std"],
        )
        if not csv_exists:
            writer.writeheader()
        writer.writerow(
            {
                "test_dataset_name": test_dataset_name,
                "dice": f"{dice_score:.2f}",
                "dice_std": f"{dice_std:.2f}",
                "hd95": f"{hd95_score:.2f}",
                "hd95_std": f"{hd95_std:.2f}",
                "masd": f"{masd_score:.2f}",
                "masd_std": f"{masd_std:.2f}",
            }
        )

    print(f"Results saved to: {results_csv_path}")
    print(f"Image-wise results saved to: {image_wise_csv_path}\n")


def main():
    args = parse_args()

    train_dataset_id = args.train_dataset_id
    test_dataset_id = args.test_dataset_id
    split = args.split
    if split not in ["Tr", "Val", "Ts"]:
        raise ValueError(f"split must be either 'Tr', 'Val' or 'Ts', got {split}")

    if args.gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", args.gpu)

    train_dataset_name = get_dataset_name(nnUNet_raw, train_dataset_id)
    test_dataset_name = get_dataset_name(nnUNet_raw, test_dataset_id)
    model_dir = join(nnUNet_results, train_dataset_name, f"{args.trainer}__{args.plans}__{args.cfg}")
    print("model_dir:", model_dir)

    eval_folds = [int(args.fold)]
    if train_dataset_name == test_dataset_name and split != "Ts" and args.fold == "all":
        eval_folds = [f for f in range(5) if os.path.exists(join(model_dir, f"fold_{f}"))]

    dataset_json = load_dataset_json(nnUNet_raw, test_dataset_name)
    num_classes = len(dataset_json["labels"])

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
            split=split,
            mini_batch_size=mini_batch_size,
            largest_component=args.largest_component,
            num_classes=num_classes,
            results_csv_path=results_csv_path,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
