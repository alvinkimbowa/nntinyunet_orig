#!/usr/bin/env python3
"""Convert ultrasound-nerve-segmentation dataset to nnUNet format."""

from __future__ import annotations

import argparse
import json
import random
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def _patient_id_from_stem(stem: str) -> str:
    # Examples: 10_123, 1_45
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected training image id format: {stem}")
    return parts[0]


def _process_train_one(task: tuple[Path, Path, Path, Path, str]) -> tuple[str, str]:
    img_path, mask_path, dst_img_path, dst_lbl_path, split = task

    with Image.open(img_path) as im:
        im = im.convert("L")
        im.save(dst_img_path)
        size = im.size

    with Image.open(mask_path) as mm:
        mm = mm.convert("L")
        if mm.size != size:
            mm = mm.resize(size, Image.NEAREST)
        arr = np.array(mm, dtype=np.uint8)
        arr = (arr > 0).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(dst_lbl_path)

    return split, dst_lbl_path.stem


def _process_test_one(task: tuple[Path, Path, Path]) -> str:
    img_path, dst_img_path, dst_lbl_path = task

    with Image.open(img_path) as im:
        im = im.convert("L")
        im.save(dst_img_path)
        size = im.size

    zeros = np.zeros((size[1], size[0]), dtype=np.uint8)
    Image.fromarray(zeros, mode="L").save(dst_lbl_path)
    return dst_lbl_path.stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ultrasound-nerve-segmentation to nnUNet Dataset312."
    )
    parser.add_argument(
        "--src",
        default="data/raw_data/ultrasound-nerve-segmentation",
        help="Dataset root containing train/ and test/ folders.",
    )
    parser.add_argument(
        "--out",
        default="data/nnUNet_raw/Dataset312_UltrasoundNerve",
        help="Output nnUNet dataset directory.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio (patient-level) from train split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (0 = cpu_count()-1).",
    )
    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in (0, 1)")

    src_root = Path(args.src)
    out_root = Path(args.out)
    images_tr = out_root / "imagesTr"
    labels_tr = out_root / "labelsTr"
    images_ts = out_root / "imagesTs"
    labels_ts = out_root / "labelsTs"
    for d in (images_tr, labels_tr, images_ts, labels_ts):
        d.mkdir(parents=True, exist_ok=True)

    train_dir = src_root / "train"
    test_dir = src_root / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError("Expected train/ and test/ under source root")

    train_imgs = sorted(p for p in train_dir.glob("*.tif") if not p.name.endswith("_mask.tif"))
    if not train_imgs:
        raise RuntimeError(f"No train image files found under {train_dir}")

    train_cases: list[tuple[str, Path, Path]] = []
    patient_ids: list[str] = []
    for img_path in train_imgs:
        stem = img_path.stem
        mask_path = train_dir / f"{stem}_mask.tif"
        if not mask_path.exists():
            continue
        pid = _patient_id_from_stem(stem)
        patient_ids.append(pid)
        train_cases.append((pid, img_path, mask_path))

    unique_pids = sorted(set(patient_ids))
    rng = random.Random(args.seed)
    rng.shuffle(unique_pids)
    n_val = max(1, int(len(unique_pids) * args.val_ratio))
    val_pids = set(unique_pids[:n_val])

    train_tasks: list[tuple[Path, Path, Path, Path, str]] = []
    for pid, img_path, mask_path in train_cases:
        case_id = f"UNS_{img_path.stem}"
        split = "val" if pid in val_pids else "train"
        train_tasks.append(
            (
                img_path,
                mask_path,
                images_tr / f"{case_id}_0000.tif",
                labels_tr / f"{case_id}.tif",
                split,
            )
        )

    test_imgs = sorted(p for p in test_dir.glob("*.tif"))
    test_tasks: list[tuple[Path, Path, Path]] = []
    for img_path in test_imgs:
        case_id = f"UNS_test_{img_path.stem}"
        test_tasks.append(
            (
                img_path,
                images_ts / f"{case_id}_0000.tif",
                labels_ts / f"{case_id}.tif",
            )
        )

    workers = args.workers if args.workers > 0 else max(cpu_count() - 1, 1)

    train_ids: list[str] = []
    val_ids: list[str] = []
    if workers == 1:
        train_iterator = map(_process_train_one, train_tasks)
    else:
        pool_train = Pool(processes=workers)
        train_iterator = pool_train.imap_unordered(_process_train_one, train_tasks, chunksize=32)
    try:
        for split, case_id in tqdm(
            train_iterator,
            total=len(train_tasks),
            desc="converting train",
            unit="img",
        ):
            if split == "train":
                train_ids.append(case_id)
            else:
                val_ids.append(case_id)
    finally:
        if workers != 1:
            pool_train.close()
            pool_train.join()

    test_ids: list[str] = []
    if workers == 1:
        test_iterator = map(_process_test_one, test_tasks)
    else:
        pool_test = Pool(processes=workers)
        test_iterator = pool_test.imap_unordered(_process_test_one, test_tasks, chunksize=32)
    try:
        for case_id in tqdm(
            test_iterator,
            total=len(test_tasks),
            desc="converting test",
            unit="img",
        ):
            test_ids.append(case_id)
    finally:
        if workers != 1:
            pool_test.close()
            pool_test.join()

    splits = [{"train": sorted(train_ids), "val": sorted(val_ids)}]
    (out_root / "splits_final.json").write_text(json.dumps(splits, indent=4))

    dataset_json = {
        "channel_names": {"0": "Gray"},
        "labels": {"background": 0, "nerve": 1},
        "file_ending": ".tif",
        "numTraining": len(train_ids) + len(val_ids),
    }
    (out_root / "dataset.json").write_text(json.dumps(dataset_json, indent=4))

    print(
        f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} "
        f"numTraining={len(train_ids) + len(val_ids)}"
    )


if __name__ == "__main__":
    main()
