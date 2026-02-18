#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


def _stage_index(name):
    m = re.search(r"\.stages\.(\d+)$", str(name))
    return int(m.group(1)) if m else None


def resolve_dataset_name(dataset_id, results_dir):
    ds_prefix = f"Dataset{int(dataset_id):03d}_"

    # Prefer nnUNet_raw when available (authoritative mapping id -> name)
    nnunet_raw = os.environ.get("nnUNet_raw")
    if nnunet_raw:
        root = Path(nnunet_raw)
        if root.exists():
            matches = sorted(
                [p.name for p in root.iterdir() if p.is_dir() and p.name.startswith(ds_prefix)]
            )
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise SystemExit(
                    f"Multiple datasets found for id {dataset_id}: {matches}. "
                    "Pass --dataset explicitly."
                )

    # Fallback: infer from breakdown CSV filenames in results_dir
    results_root = Path(results_dir)
    matches = set()
    for p in results_root.glob(f"{ds_prefix}*_naswot_*.csv"):
        m = re.match(rf"^(Dataset{int(dataset_id):03d}_[^_]+)", p.name)
        if m:
            matches.add(m.group(1))
        else:
            # cfg can have underscores; split at first `_nnUNetTrainer` style chunk is not robust.
            # Use suffix token to trim from the right as a safer fallback.
            tail = "_naswot_"
            idx = p.name.find(tail)
            if idx > 0:
                prefix = p.name[:idx]
                # strip trailing `_..._b<batch>`
                b_idx = prefix.rfind("_b")
                if b_idx > 0:
                    candidate = prefix[: b_idx].split("_", 1)[0]
                    if candidate.startswith(ds_prefix):
                        matches.add(candidate)
    matches = sorted(matches)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(
            f"Could not resolve dataset name for id {dataset_id}. "
            "Pass --dataset explicitly or ensure nnUNet_raw/results files are available."
        )
    raise SystemExit(
        f"Ambiguous dataset id {dataset_id}; candidates: {matches}. "
        "Pass --dataset explicitly."
    )


def load_breakdown(results_dir, dataset, batches, kind):
    rows = []
    pattern = f"{dataset}_*_b{batches}_naswot_{kind}.csv"
    for path in sorted(Path(results_dir).glob(pattern)):
        cfg = path.name.split(f"{dataset}_")[1].split(f"_b{batches}")[0]
        df = pd.read_csv(path)
        col = "module" if kind == "modules" else "stage" if kind == "stages" else "block"
        for _, r in df.iterrows():
            rows.append({"cfg": cfg, col: r[col], "logdet": r["logdet"]})
    return rows


def _cfg_key(cfg):
    if "_tiny" in cfg:
        try:
            return (0, int(cfg.split("_tiny", 1)[1]))
        except ValueError:
            return (0, cfg)
    if cfg == "2d":
        return (1, 0)
    return (2, cfg)


def plot_kind(rows, kind, out_path):
    col = "module" if kind == "modules" else "stage" if kind == "stages" else "block"
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No rows to plot. Did you enable --naswot_breakdown?")
    # Tiny offset so decoder curves are visible when they exactly overlap encoder curves.
    if kind == "stages":
        is_decoder = df[col].astype(str).str.startswith("decoder.stages.")
        df.loc[is_decoder, "logdet"] = df.loc[is_decoder, "logdet"] + 0.25
    wide = pd.pivot_table(df, index="cfg", columns=col, values="logdet")
    wide = wide.reindex(sorted(wide.index, key=_cfg_key))
    if kind == "stages":
        fig, ax = plt.subplots()
        x_pos = list(range(len(wide.index)))
        enc_cols = sorted([c for c in wide.columns if str(c).startswith("encoder.stages.")])
        dec_cols = sorted([c for c in wide.columns if str(c).startswith("decoder.stages.")])
        other_cols = [c for c in wide.columns if c not in enc_cols and c not in dec_cols]
        max_dec_idx = max((_stage_index(c) for c in dec_cols if _stage_index(c) is not None), default=-1)

        # Decoder stages are reverse-ordered vs encoder stages; map to shared color ids.
        color_ids = set()
        for name in enc_cols:
            idx = _stage_index(name)
            if idx is not None:
                color_ids.add(idx)
        for name in dec_cols:
            didx = _stage_index(name)
            if didx is not None and max_dec_idx >= 0:
                color_ids.add(max_dec_idx - didx)
        color_ids = sorted(color_ids)
        cmap = plt.get_cmap("viridis")
        color_map = {cid: cmap(i / max(1, len(color_ids) - 1)) for i, cid in enumerate(color_ids)}

        for name in enc_cols:
            idx = _stage_index(name)
            color = color_map.get(idx, None)
            ax.plot(x_pos, wide[name].to_numpy(), marker="o", linewidth=1, color=color, label=name)
        for name in dec_cols:
            didx = _stage_index(name)
            paired_idx = max_dec_idx - didx if didx is not None and max_dec_idx >= 0 else None
            color = color_map.get(paired_idx, None)
            ax.plot(x_pos, wide[name].to_numpy(), marker="o", linewidth=1, color=color, linestyle="--", label=name)
        for name in other_cols:
            ax.plot(x_pos, wide[name].to_numpy(), marker="o", linewidth=1, label=name)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(list(wide.index), rotation=20, ha="right")
    else:
        ax = wide.plot(marker="o", linewidth=1, legend=True)
    ax.legend(ncol=3, fontsize=8)
    ax.set_title(f"NASWOT {kind} vs cfg")
    ax.set_xlabel("cfg")
    ax.set_ylabel("logdet")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot NASWOT breakdown across cfgs")
    parser.add_argument("--results_dir", type=str, default="results/nas_metrics")
    ds_group = parser.add_mutually_exclusive_group(required=True)
    ds_group.add_argument("--dataset", type=str, default="")
    ds_group.add_argument("--dataset_id", type=int, default=None)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--kind", type=str, choices=["stages", "blocks", "modules"], default="stages")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    dataset = args.dataset or resolve_dataset_name(args.dataset_id, args.results_dir)
    rows = load_breakdown(args.results_dir, dataset, args.batches, args.kind)
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(args.results_dir) / f"{dataset}_b{args.batches}_naswot_{args.kind}.png"
    plot_kind(rows, args.kind, out_path)


if __name__ == "__main__":
    main()
