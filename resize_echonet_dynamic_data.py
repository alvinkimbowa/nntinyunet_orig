#!/usr/bin/env python3
"""Resize nnUNet dataset images and labels into a new output dataset."""

from __future__ import annotations

import argparse
import shutil
from multiprocessing import Pool, cpu_count
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def _process_one(task: tuple[Path, Path, int, int]) -> None:
    src_path, dst_path, size, resample = task
    with Image.open(src_path) as im:
        im = im.resize((size, size), resample)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst_path)


def _resize_dir(
    src_dir: Path, dst_dir: Path, size: int, resample: int, desc: str, workers: int
) -> None:
    if not src_dir.exists():
        return
    paths = sorted(src_dir.glob("*.png"))
    tasks = [(p, dst_dir / p.name, size, resample) for p in paths]
    if workers < 1:
        workers = max(cpu_count() - 1, 1)
    with Pool(processes=workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(_process_one, tasks, chunksize=32),
            total=len(tasks),
            desc=desc,
        ):
            pass


def resize_images(src_root: Path, dst_root: Path, size: int, workers: int) -> None:
    for subdir in ("imagesTr", "imagesTs"):
        _resize_dir(
            src_root / subdir,
            dst_root / subdir,
            size,
            Image.BILINEAR,
            f"resize {subdir}",
            workers,
        )


def resize_labels(src_root: Path, dst_root: Path, size: int, workers: int) -> None:
    for subdir in ("labelsTr", "labelsTs"):
        _resize_dir(
            src_root / subdir,
            dst_root / subdir,
            size,
            Image.NEAREST,
            f"resize {subdir}",
            workers,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resize nnUNet dataset images/labels into a new output dataset."
    )
    parser.add_argument(
        "--dataset",
        default="data/nnUNet_raw/Dataset302_EchoNet-Dynamic",
        help="Path to input nnUNet dataset root.",
    )
    parser.add_argument(
        "--out",
        default="data/nnUNet_raw/Dataset303_echonet_dynamic_resized",
        help="Path to output nnUNet dataset root.",
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

    src_root = Path(args.dataset)
    dst_root = Path(args.out)
    resize_images(src_root, dst_root, args.size, args.workers)
    resize_labels(src_root, dst_root, args.size, args.workers)
    # copy the dataset.json file to the dst_root
    shutil.copy(src_root / "dataset.json", dst_root / "dataset.json")
    if (src_root / "splits_final.json").exists():
        shutil.copy(src_root / "splits_final.json", dst_root / "splits_final.json")

if __name__ == "__main__":
    main()
