#!/usr/bin/env python3
"""Convert BraTS2020 H5 slices to nnUNet PNG layout."""

from __future__ import annotations

import argparse
import json
import random
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


BRATS_RE = re.compile(r"^volume_(?P<vid>\d+)_slice_(?P<sid>\d+)\.h5$")


def _parse_ids(path: Path) -> tuple[str, str]:
    m = BRATS_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected BraTS filename: {path.name}")
    return m.group("vid"), m.group("sid")


def _norm_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    amin = float(np.min(arr))
    amax = float(np.max(arr))
    if amax <= amin:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = (arr - amin) / (amax - amin)
    out = (out * 255.0).clip(0, 255).astype(np.uint8)
    return out


def _process_one(task: tuple[Path, Path, Path, str]) -> tuple[str, str]:
    h5_path, dst_img_dir, dst_lbl_dir, split = task
    vol_id, slice_id = _parse_ids(h5_path)
    case_id = f"BRATS2020_v{int(vol_id):04d}_s{int(slice_id):04d}"

    with h5py.File(h5_path, "r") as f:
        image = np.asarray(f["image"])  # H,W,C
        mask = np.asarray(f["mask"])    # H,W,K

    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got {image.ndim} in {h5_path}")
    if mask.ndim != 3:
        raise ValueError(f"Expected mask ndim=3, got {mask.ndim} in {h5_path}")

    n_channels = image.shape[2]
    for c in range(n_channels):
        ch = _norm_to_uint8(image[:, :, c])
        Image.fromarray(ch, mode="L").save(dst_img_dir / f"{case_id}_{c:04d}.png")

    lbl = np.zeros(mask.shape[:2], dtype=np.uint8)
    max_k = min(mask.shape[2], 3)
    for k in range(max_k):
        lbl[mask[:, :, k] > 0] = k + 1
    Image.fromarray(lbl, mode="L").save(dst_lbl_dir / f"{case_id}.png")

    return split, case_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert BraTS2020 H5 slices to nnUNet Dataset311.")
    parser.add_argument(
        "--src",
        default="data/raw_data/BRATS2020/BraTS2020_training_data/content/data",
        help="Directory containing BraTS2020 H5 files.",
    )
    parser.add_argument(
        "--out",
        default="data/nnUNet_raw/Dataset311_BraTS2020",
        help="Output nnUNet dataset directory.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Volume-level train ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Volume-level val ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for volume split.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (0 = cpu_count()-1).",
    )
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Need 0 < train-ratio, val-ratio and train-ratio + val-ratio < 1")

    src_dir = Path(args.src)
    out_root = Path(args.out)
    images_tr = out_root / "imagesTr"
    labels_tr = out_root / "labelsTr"
    images_ts = out_root / "imagesTs"
    labels_ts = out_root / "labelsTs"
    for d in (images_tr, labels_tr, images_ts, labels_ts):
        d.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(src_dir.glob("volume_*_slice_*.h5"))
    if not h5_files:
        raise RuntimeError(f"No H5 files found under {src_dir}")

    vols = sorted({int(_parse_ids(p)[0]) for p in h5_files})
    rng = random.Random(args.seed)
    rng.shuffle(vols)
    n = len(vols)
    n_train = max(1, int(n * args.train_ratio))
    n_val = max(1, int(n * args.val_ratio))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    if n_train + n_val >= n:
        n_train = max(1, n - n_val - 1)

    train_vols = set(vols[:n_train])
    val_vols = set(vols[n_train:n_train + n_val])
    test_vols = set(vols[n_train + n_val:])

    tasks: list[tuple[Path, Path, Path, str]] = []
    for h5_path in h5_files:
        vol_id = int(_parse_ids(h5_path)[0])
        if vol_id in train_vols:
            tasks.append((h5_path, images_tr, labels_tr, "train"))
        elif vol_id in val_vols:
            tasks.append((h5_path, images_tr, labels_tr, "val"))
        else:
            tasks.append((h5_path, images_ts, labels_ts, "test"))

    workers = args.workers if args.workers > 0 else max(cpu_count() - 1, 1)
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    if workers == 1:
        iterator = map(_process_one, tasks)
    else:
        pool = Pool(processes=workers)
        iterator = pool.imap_unordered(_process_one, tasks, chunksize=16)
    try:
        for split, case_id in tqdm(
            iterator,
            total=len(tasks),
            desc="converting",
            unit="slice",
        ):
            if split == "train":
                train_ids.append(case_id)
            elif split == "val":
                val_ids.append(case_id)
            else:
                test_ids.append(case_id)
    finally:
        if workers != 1:
            pool.close()
            pool.join()

    splits = [{"train": sorted(train_ids), "val": sorted(val_ids)}]
    (out_root / "splits_final.json").write_text(json.dumps(splits, indent=4))

    dataset_json = {
        "channel_names": {"0": "flair", "1": "t1ce", "2": "t1", "3": "t2"},
        "labels": {"background": 0, "label_1": 1, "label_2": 2, "label_3": 3},
        "file_ending": ".png",
        "numTraining": len(train_ids) + len(val_ids),
    }
    (out_root / "dataset.json").write_text(json.dumps(dataset_json, indent=4))

    print(
        f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} "
        f"numTraining={len(train_ids) + len(val_ids)}"
    )


if __name__ == "__main__":
    main()
