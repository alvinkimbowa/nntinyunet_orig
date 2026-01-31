#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import pandas as pd


def get_dataset_name(dataset_id, nnunet_raw):
    dataset_id = str(dataset_id).zfill(3)
    dataset_name = [
        d for d in os.listdir(nnunet_raw)
        if d.startswith(f"Dataset{dataset_id}") and os.path.isdir(os.path.join(nnunet_raw, d))
    ]
    if len(dataset_name) != 1:
        raise RuntimeError(f"Found {len(dataset_name)} datasets with id {dataset_id}, expected 1")
    return dataset_name[0]


def get_dice(results_dir, dataset_name, trainer, plans, cfg, fold=0, split="val"):
    model_dir = f"{results_dir}/{dataset_name}/{trainer}__{plans}__{cfg}"
    if split != "val":
        raise ValueError("Only split=val supported")
    metrics_file = f"{model_dir}/fold_{fold}/validation/summary.json"
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    return metrics["foreground_mean"]["Dice"] * 100


def main():
    parser = argparse.ArgumentParser(description="Analyze per-batch jacobian correlations vs Dice")
    parser.add_argument("--train_dataset_id", type=int, required=True)
    parser.add_argument("--results_dir", type=str, default="data/nnUNet_results/nnunet")
    parser.add_argument("--nas_dir", type=str, default="results/nas_metrics")
    parser.add_argument("--trainer", type=str, default="nnUNetTrainer_100epochs")
    parser.add_argument("--plans", type=str, default="nnUNetPlans")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--cfgs", nargs="+", default=[
        "2d_tiny1","2d_tiny2","2d_tiny4","2d_tiny8","2d_tiny16",
        "2d_tiny32","2d_tiny64","2d_tiny128","2d_tiny256","2d",
    ])
    args = parser.parse_args()

    nnunet_raw = os.environ["nnUNet_raw"]
    dataset_name = get_dataset_name(args.train_dataset_id, nnunet_raw)

    batch_path = Path(args.nas_dir) / f"{dataset_name}_batch_jacobian_b{args.batches}.csv"
    if not batch_path.exists():
        raise SystemExit(f"Missing {batch_path}. Run score_net.py with --save_batch_jacobian.")

    df = pd.read_csv(batch_path)
    df = df[df["cfg"].isin(args.cfgs)]

    dice_map = {cfg: get_dice(args.results_dir, dataset_name, args.trainer, args.plans, cfg, args.fold) for cfg in args.cfgs}
    dice_series = pd.Series(dice_map)

    rows = []
    for batch_id, g in df.groupby("batch"):
        if g["cfg"].duplicated().any():
            raise SystemExit(f"Duplicate cfgs found in batch {batch_id}.")
        jac = g.set_index("cfg")["jacobian"]
        jac = jac.reindex(args.cfgs)
        if jac.isna().any():
            continue
        img_ids = g["img_ids"].iloc[0]
        pearson = jac.corr(dice_series, method="pearson")
        spearman = jac.corr(dice_series, method="spearman")
        rows.append(
            {
                "batch": int(batch_id),
                "img_ids": img_ids,
                "pearson": pearson,
                "spearman": spearman,
            }
        )

    if not rows:
        raise SystemExit("No valid batches found. Check cfg coverage and batch_jacobian file contents.")
    out_df = pd.DataFrame(rows).sort_values("spearman", ascending=False)
    out_path = Path(args.nas_dir) / f"{dataset_name}_batch_jacobian_corr_b{args.batches}.csv"
    out_df.to_csv(out_path, index=False)
    print(out_df.head(10))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
