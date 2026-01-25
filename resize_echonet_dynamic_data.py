#!/usr/bin/env python3
"""Resize nnUNet dataset images and labels in-place."""

from __future__ import annotations

import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def _process_one(task: tuple[Path, int, int]) -> None:
    path, size, resample = task
    with Image.open(path) as im:
        im = im.resize((size, size), resample)
        im.save(path)


def _resize_dir(dir_path: Path, size: int, resample: int, desc: str, workers: int) -> None:
    if not dir_path.exists():
        return
    paths = sorted(dir_path.glob("*.png"))
    tasks = [(p, size, resample) for p in paths]
    if workers < 1:
        workers = max(cpu_count() - 1, 1)
    with Pool(processes=workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(_process_one, tasks, chunksize=32),
            total=len(tasks),
            desc=desc,
        ):
            pass


def resize_images(root: Path, size: int, workers: int) -> None:
    for subdir in ("imagesTr", "imagesTs"):
        _resize_dir(root / subdir, size, Image.BILINEAR, f"resize {subdir}", workers)


def resize_labels(root: Path, size: int, workers: int) -> None:
    for subdir in ("labelsTr", "labelsTs"):
        _resize_dir(root / subdir, size, Image.NEAREST, f"resize {subdir}", workers)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resize nnUNet dataset images/labels in-place."
    )
    parser.add_argument(
        "--dataset",
        default="data/nnUNet_raw/Dataset302_EchoNet-Dynamic",
        help="Path to nnUNet dataset root.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Target size (square).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (0 = cpu_count()-1).",
    )
    args = parser.parse_args()

    root = Path(args.dataset)
    resize_images(root, args.size, args.workers)
    resize_labels(root, args.size, args.workers)


if __name__ == "__main__":
    main()
