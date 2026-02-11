#!/usr/bin/env python3
"""Convert FIVES vessel dataset to nnUNet format."""

from __future__ import annotations

import argparse
import json
import random
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def _process_one(
    task: tuple[Path, Path, Path, Path, Path, Path, str]
) -> tuple[str, str]:
    img_path, lbl_path, dst_r_path, dst_g_path, dst_b_path, dst_lbl_path, split = task

    with Image.open(img_path) as im:
        im = im.convert("RGB").resize((256, 256), Image.BILINEAR)
        r, g, b = im.split()
        r.save(dst_r_path)
        g.save(dst_g_path)
        b.save(dst_b_path)
        size = im.size

    with Image.open(lbl_path) as lb:
        lb = lb.convert("L").resize(size, Image.NEAREST)
        arr = np.array(lb, dtype=np.uint8)
        arr = (arr > 0).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(dst_lbl_path)

    return split, dst_lbl_path.stem


def _collect_pairs(img_dir: Path, lbl_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(img_dir.glob("*.png")):
        lbl_path = lbl_dir / img_path.name
        if not lbl_path.exists():
            raise FileNotFoundError(f"Missing label for {img_path.name}: {lbl_path}")
        pairs.append((img_path, lbl_path))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert FIVES to nnUNet Dataset309.")
    parser.add_argument(
        "--src",
        default="data/raw_data/FIVES A Fundus Image Dataset for AI-based Vessel Segmentation/FIVES A Fundus Image Dataset for AI-based Vessel Segmentation",
        help="FIVES base directory containing train/ and test/.",
    )
    parser.add_argument(
        "--out",
        default="data/nnUNet_raw/Dataset309_FIVES",
        help="Output nnUNet dataset directory.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio from train split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split.",
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

    train_pairs = _collect_pairs(src_root / "train" / "Original", src_root / "train" / "Ground truth")
    test_pairs = _collect_pairs(src_root / "test" / "Original", src_root / "test" / "Ground truth")

    rng = random.Random(args.seed)
    idxs = list(range(len(train_pairs)))
    rng.shuffle(idxs)
    n_val = max(1, int(len(idxs) * args.val_ratio))
    val_idx = set(idxs[:n_val])

    tasks: list[tuple[Path, Path, Path, Path, Path, Path, str]] = []
    for i, (img_path, lbl_path) in enumerate(train_pairs):
        case_id = f"FIVES_{img_path.stem}"
        split = "val" if i in val_idx else "train"
        tasks.append(
            (
                img_path,
                lbl_path,
                images_tr / f"{case_id}_0000.png",
                images_tr / f"{case_id}_0001.png",
                images_tr / f"{case_id}_0002.png",
                labels_tr / f"{case_id}.png",
                split,
            )
        )
    for img_path, lbl_path in test_pairs:
        case_id = f"FIVES_{img_path.stem}"
        tasks.append(
            (
                img_path,
                lbl_path,
                images_ts / f"{case_id}_0000.png",
                images_ts / f"{case_id}_0001.png",
                images_ts / f"{case_id}_0002.png",
                labels_ts / f"{case_id}.png",
                "test",
            )
        )

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
            unit="img",
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
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": {"background": 0, "vessel": 1},
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
