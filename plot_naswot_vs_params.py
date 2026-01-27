#!/usr/bin/env python
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def parse_scores(path):
    params = []
    scores = []
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        if header[:3] != ["cfg", "params", "naswot"]:
            raise SystemExit(f"Unexpected CSV header in {path}")
        for line in f:
            if not line.strip():
                continue
            _, p, s = line.strip().split(",")[:3]
            params.append(int(p))
            scores.append(float(s))
    return params, scores


def main():
    parser = argparse.ArgumentParser(description="Plot NASWOT vs UNet parameter count")
    parser.add_argument("--in_file", type=str, default="", help="results file to plot")
    parser.add_argument("--results_dir", type=str, default="results/naswot", help="directory to search")
    parser.add_argument("--out_file", type=str, default="", help="output PNG path")
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=1)
    parser.add_argument("--num_stages", type=int, default=5)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if args.in_file:
        in_path = Path(args.in_file)
    else:
        fixed = results_dir / "naswot_unet_widths.csv"
        if fixed.exists() and fixed.stat().st_size > 0:
            in_path = fixed
        else:
            files = sorted(results_dir.glob("naswot_unet_widths_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
            in_path = next((p for p in files if p.stat().st_size > 0), None)
            if in_path is None:
                raise SystemExit(f"No non-empty results file found in {results_dir}")

    params, scores = parse_scores(in_path)
    if not params:
        raise SystemExit(f"No params parsed from {in_path}")

    if args.out_file:
        out_path = Path(args.out_file)
    else:
        out_path = results_dir / "naswot_vs_params.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    params_m = [p / 1_000_000 for p in params]
    plt.figure(figsize=(6, 4))
    plt.plot(params_m, scores, marker="o")
    plt.xlabel("Model parameters (millions)")
    plt.ylabel("NASWOT score")
    plt.title("NASWOT vs UNet size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Wrote {out_path} from {in_path.name}")


if __name__ == "__main__":
    main()
