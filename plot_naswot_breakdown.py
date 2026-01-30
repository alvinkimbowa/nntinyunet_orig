#!/usr/bin/env python3
import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


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
    wide = pd.pivot_table(df, index="cfg", columns=col, values="logdet")
    wide = wide.reindex(sorted(wide.index, key=_cfg_key))
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
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--kind", type=str, choices=["stages", "blocks", "modules"], default="stages")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    rows = load_breakdown(args.results_dir, args.dataset, args.batches, args.kind)
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(args.results_dir) / f"{args.dataset}_b{args.batches}_naswot_{args.kind}.png"
    plot_kind(rows, args.kind, out_path)


if __name__ == "__main__":
    main()
