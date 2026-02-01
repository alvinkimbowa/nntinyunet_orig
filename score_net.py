import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, InterpolationMode
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.run.run_training import get_trainer_from_args

from dataset import nnUNetDataset
from at_init_metrics import (
    naswot_score,
    naswot_module_contributions,
    aggregate_naswot_contributions,
    synflow_score,
    gradnorm_score,
    snip_score,
    jacobian_score,
    fisher_score,
)

nnUNet_raw = os.environ['nnUNet_raw']
nnUNet_results = os.environ['nnUNet_results']

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_dataset_name(dataset_id):
    dataset_id = str(dataset_id).zfill(3)
    dataset_name = [
        d for d in os.listdir(nnUNet_raw)
        if d.startswith(f"Dataset{dataset_id}") and os.path.isdir(join(nnUNet_raw, d))
    ]
    if len(dataset_name) != 1:
        raise RuntimeError(f"Found {len(dataset_name)} datasets with id {dataset_id}, expected 1")
    return dataset_name[0]


def get_num_input_channels(dataset_name):
    with open(join(nnUNet_raw, dataset_name, "dataset.json"), "r") as f:
        dataset_json = json.load(f)
    return len(dataset_json["channel_names"])


def load_nnunet_model(train_dataset_id, plans, trainer, cfg, fold, device):
    fold = fold if fold == "all" else int(fold)
    dataset_name = get_dataset_name(train_dataset_id)
    nnunet_trainer = get_trainer_from_args(dataset_name, cfg, fold, trainer, plans, device=device)
    nnunet_trainer.enable_deep_supervision = False
    nnunet_trainer.initialize()
    model = nnunet_trainer.network.to(device)
    with open("model.txt", "w") as f:
        f.write(str(model))
    loss_fn = nnunet_trainer.loss
    os.environ["nnUNet_n_proc_DA"] = "0"    # Use a single process for data augmentation to avoid wierd errors
    data_loader, val_loader = nnunet_trainer.get_dataloaders()
    return model, dataset_name, loss_fn, data_loader


class EncoderOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        if not hasattr(model, "encoder"):
            raise AttributeError("Model has no encoder attribute")

    def forward(self, x):
        out = self.model.encoder(x)
        if isinstance(out, (list, tuple)):
            return out[-1]
        return out

class ResizeTransform:
    def __init__(self, patch_size):
        self.patch_size = tuple(int(v) for v in patch_size)
        self._img_resize = Resize(self.patch_size, antialias=True)
        self._mask_resize = Resize(self.patch_size, interpolation=InterpolationMode.NEAREST)

    def __call__(self, image, mask):
        img = torch.as_tensor(image).permute(2, 0, 1).float()
        msk = torch.as_tensor(mask).permute(2, 0, 1).float()
        img = self._img_resize(img)
        msk = self._mask_resize(msk)
        img = img.permute(1, 2, 0).numpy()
        msk = msk.permute(1, 2, 0).numpy()
        return {"image": img, "mask": msk}

def main():
    parser = argparse.ArgumentParser(description="Compute NASWOT score for an nnUNet model")
    parser.add_argument("--train_dataset_id", type=int, required=True)
    parser.add_argument("--plans", type=str, required=True)
    parser.add_argument("--trainer", type=str, required=True)
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--fold", type=str, default="0")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--split", type=str, default="Tr", choices=["Tr", "Ts"])
    parser.add_argument("--split_type", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="results/naswot")
    parser.add_argument("--naswot_breakdown", action="store_true",
                        help="save per-module and aggregated NASWOT contributions")
    parser.add_argument("--save_batch_jacobian", action="store_true",
                        help="save per-batch jacobian with image ids")
    parser.add_argument("--encoder_only", action="store_true", help="compute NASWOT on encoder only")
    parser.add_argument(
        "--metrics",
        type=str,
        default="naswot,synflow,gradnorm,snip,jacobian,fisher",
        help="comma-separated list of metrics to compute",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu" if args.gpu < 0 else "cuda")
    metric_set = {m.strip().lower() for m in args.metrics.split(",") if m.strip()}

    model, dataset_name, loss_fn, data_loader = load_nnunet_model(
        args.train_dataset_id,
        args.plans,
        args.trainer,
        args.cfg,
        args.fold,
        device,
    )
    if args.encoder_only:
        model = EncoderOnly(model).to(device)

    naswot_scores = []
    synflow_scores = []
    gradnorm_scores = []
    snip_scores = []
    jacobian_scores = []
    fisher_scores = []
    breakdown_done = False
    batch_rows = []
    for i, batch in tqdm(enumerate(data_loader), total=args.batches):
        if i >= args.batches:
            break
        imgs = batch['data']
        targets = batch['target']
        meta = batch['keys']
        x = imgs.float().to(device)
        targets = targets.to(device)
        if "naswot" in metric_set:
            naswot_scores.append(naswot_score(model, x))
            if args.naswot_breakdown and not breakdown_done:
                breakdown_done = True
                module_scores = naswot_module_contributions(model, x)
                stage_scores = aggregate_naswot_contributions(module_scores, level="stage")
                block_scores = aggregate_naswot_contributions(module_scores, level="convblock")
        if "synflow" in metric_set:
            synflow_scores.append(
                synflow_score(model, (x.size(0),) + tuple(x.shape[1:]), device)
            )
        if "gradnorm" in metric_set:
            gradnorm_scores.append(gradnorm_score(model, x))
        if "snip" in metric_set:
            snip_scores.append(snip_score(model, x))
        if "fisher" in metric_set:
            fisher_scores.append(fisher_score(model, x))
        if "jacobian" in metric_set:
            jac = jacobian_score(model, x, targets, loss_fn)
            jacobian_scores.append(jac)
            if args.save_batch_jacobian:
                img_ids = meta["img_id"] if isinstance(meta, dict) else meta.get("img_id")
                if isinstance(img_ids, (list, tuple)):
                    img_ids = ";".join(img_ids)
                batch_rows.append(
                    {
                        "dataset": dataset_name,
                        "cfg": args.cfg,
                        "batch": i,
                        "jacobian": jac,
                        "img_ids": img_ids,
                        "seed": args.seed,
                    }
                )
    naswot_avg = float(np.nanmean(naswot_scores)) if naswot_scores else float("nan")
    synflow_avg = float(np.nanmean(synflow_scores)) if synflow_scores else float("nan")
    gradnorm_avg = float(np.nanmean(gradnorm_scores)) if gradnorm_scores else float("nan")
    snip_avg = float(np.nanmean(snip_scores)) if snip_scores else float("nan")
    jacobian_avg = float(np.nanmean(jacobian_scores)) if jacobian_scores else float("nan")
    fisher_avg = float(np.nanmean(fisher_scores)) if fisher_scores else float("nan")
    params = sum(p.numel() for p in model.parameters())
    line = (
        f"params={params} naswot={naswot_avg} synflow={synflow_avg} "
        f"gradnorm={gradnorm_avg} snip={snip_avg} "
        f"jacobian={jacobian_avg} fisher={fisher_avg}"
    )
    print(line)
    if args.encoder_only:
        out_file = join(args.out_dir, f"{dataset_name}_metrics_encoder_only_b{args.batches}.csv")
    else:
        out_file = join(args.out_dir, f"{dataset_name}_metrics_b{args.batches}.csv")
    print("out_file", out_file)
    need_header = not os.path.exists(out_file) or os.path.getsize(out_file) == 0
    with open(out_file, "a", encoding="utf-8") as f:
        if need_header:
            f.write("cfg,params,naswot,synflow,gradnorm,snip,jacobian,fisher\n")
        f.write(
            f"{args.cfg},{params},{naswot_avg},{synflow_avg},"
            f"{gradnorm_avg},{snip_avg},{jacobian_avg},{fisher_avg}\n"
        )
    
    if args.naswot_breakdown and breakdown_done:
        suffix = f"{dataset_name}_{args.cfg}_b{args.batches}"
        mod_path = join(args.out_dir, f"{suffix}_naswot_modules.csv")
        stage_path = join(args.out_dir, f"{suffix}_naswot_stages.csv")
        block_path = join(args.out_dir, f"{suffix}_naswot_blocks.csv")
        with open(mod_path, "w", encoding="utf-8") as f:
            f.write("module,logdet\n")
            for name, val in module_scores:
                f.write(f"{name},{val}\n")
        with open(stage_path, "w", encoding="utf-8") as f:
            f.write("stage,logdet\n")
            for name, val in stage_scores:
                f.write(f"{name},{val}\n")
        with open(block_path, "w", encoding="utf-8") as f:
            f.write("block,logdet\n")
            for name, val in block_scores:
                f.write(f"{name},{val}\n")

    if args.save_batch_jacobian and batch_rows:
        batch_path = join(args.out_dir, f"{dataset_name}_batch_jacobian_b{args.batches}.csv")
        need_header = not os.path.exists(batch_path) or os.path.getsize(batch_path) == 0
        with open(batch_path, "a", encoding="utf-8") as f:
            if need_header:
                f.write("dataset,cfg,batch,jacobian,img_ids,seed\n")
            for row in batch_rows:
                f.write(
                    f"{row['dataset']},{row['cfg']},{row['batch']},"
                    f"{row['jacobian']},{row['img_ids']},{row['seed']}\n"
                )


if __name__ == "__main__":
    main()
