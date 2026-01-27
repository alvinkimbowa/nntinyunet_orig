import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.paths import nnUNet_results, nnUNet_raw, nnUNet_preprocessed
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.variants.UNeXtPredictor import UNeXtPredictor
from nnunetv2.inference.variants.MonaiPredictors import UNetPlusPlusPredictor, UNETRPredictor
from nnunetv2.run.run_training import get_trainer_from_args

from dataset import nnUNetDataset


def _install_naswot_hooks(model, batch_size):
    handles = []
    K_accum = np.zeros((batch_size, batch_size), dtype=np.float32)

    def forward_hook(module, inp, _):
        try:
            if not getattr(module, "visited_backwards", False):
                return
            x = inp[0]
            x = x.view(x.size(0), -1)
            x = (x > 0).float()
            K = x @ x.t()
            K2 = (1.0 - x) @ (1.0 - x.t())
            K_accum[:] = K_accum + K.cpu().numpy() + K2.cpu().numpy()
        except Exception:
            pass

    def backward_hook(module, *_):
        module.visited_backwards = True

    for module in model.modules():
        if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
            if module.inplace:
                module.inplace = False
            module.visited_backwards = False
            handles.append(module.register_forward_hook(forward_hook))
            if hasattr(module, "register_full_backward_hook"):
                handles.append(module.register_full_backward_hook(backward_hook))
            else:
                handles.append(module.register_backward_hook(backward_hook))

    return handles, K_accum


def naswot_score(model, x):
    model.zero_grad(set_to_none=True)
    handles, K = _install_naswot_hooks(model, x.size(0))
    x = x.clone().requires_grad_(True)
    y = model(x)
    if isinstance(y, (tuple, list)):
        y = y[0]
    y.backward(torch.ones_like(y))
    _ = model(x.detach())
    for h in handles:
        h.remove()
    _, logdet = np.linalg.slogdet(K)
    return float(logdet)


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
    dataset_name = get_dataset_name(train_dataset_id)
    nnunet_trainer = get_trainer_from_args(dataset_name, cfg, fold, trainer, plans, device=device)
    nnunet_trainer.initialize()
    model = nnunet_trainer.network.to(device)
    with open("model.txt", "w") as f:
        f.write(str(model))
    return model, dataset_name, nnunet_trainer.batch_size

def load_nnunet_batch(dataset_name, input_channels, split, batch_size, fold, split_type):
    dataset = nnUNetDataset(
        dataset_name=dataset_name,
        input_channels=input_channels,
        split=split,
        fold=fold,
        split_type=split_type,
        transform=None,
        eval=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    imgs, _, _ = next(iter(loader))
    return imgs.float()


def main():
    parser = argparse.ArgumentParser(description="Compute NASWOT score for a trained nnUNet model")
    parser.add_argument("--train_dataset_id", type=int, required=True)
    parser.add_argument("--plans", type=str, required=True)
    parser.add_argument("--trainer", type=str, required=True)
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--fold", type=str, default="0")
    parser.add_argument("--chk", type=str, default="checkpoint_final.pth")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--split", type=str, default="Tr", choices=["Tr", "Ts"])
    parser.add_argument("--split_type", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out_csv", type=str, default="", help="append results to CSV file")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu" if args.gpu < 0 else f"cuda:{args.gpu}")

    model, dataset_name, batch_size = load_nnunet_model(
        args.train_dataset_id,
        args.plans,
        args.trainer,
        args.cfg,
        args.fold,
        device,
    )
    in_channels = get_num_input_channels(dataset_name)
    x = load_nnunet_batch(
        dataset_name,
        in_channels,
        args.split,
        batch_size // 2 if batch_size > 1 else 1,
        args.fold,
        args.split_type,
    ).to(device)

    scores = []
    for _ in range(args.batches):
        scores.append(naswot_score(model, x))
    avg = float(np.nanmean(scores))
    params = sum(p.numel() for p in model.parameters())
    line = f"params={params} naswot={avg}"
    print(line)
    if args.out_csv:
        need_header = not os.path.exists(args.out_csv) or os.path.getsize(args.out_csv) == 0
        with open(args.out_csv, "a", encoding="utf-8") as f:
            if need_header:
                f.write("cfg,params,naswot\n")
            f.write(f"{args.cfg},{params},{avg}\n")


if __name__ == "__main__":
    main()
