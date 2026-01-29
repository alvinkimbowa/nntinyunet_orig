#!/usr/bin/env python3
"""Convert BUSBRA combined dataset to nnUNet format using fold-1 split."""

from __future__ import annotations

import argparse
import csv
import json
import random
from multiprocessing import Pool, cpu_count
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def _is_nan(value: str | None) -> bool:
    if value is None:
        return True
    v = value.strip().lower()
    return v == "" or v == "nan"


def _mask_name(img_id: str) -> str:
    return f"mask_{img_id.replace('bus_', '', 1)}.png"


def _process_one(
    task: tuple[Path, Path, Path, Path, Path, Path, str]
) -> str:
    img_path, lbl_path, dst_r_path, dst_g_path, dst_b_path, dst_lbl_path, numbered = task
    with Image.open(img_path) as im:
        # Save single-channel if source is grayscale
        if im.mode in ("L", "I;16", "I"):
            im.convert("L").save(dst_r_path)
        else:
            im = im.convert("RGB")
            r, g, b = im.split()
            r.save(dst_r_path)
            g.save(dst_g_path)
            b.save(dst_b_path)
    with Image.open(lbl_path) as lb:
        lb = lb.convert("L").point(lambda p: 1 if p > 0 else 0, mode="L")
        lb.save(dst_lbl_path)
    return numbered


def convert_ids(
    img_ids: list[str],
    src_img_dir: Path,
    src_lbl_dir: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    seed: int,
    workers: int,
) -> list[str]:
    rng = random.Random(seed)
    ids_shuffled = list(img_ids)
    rng.shuffle(ids_shuffled)

    tasks: list[tuple[Path, Path, Path, Path, Path, Path, str]] = []
    for idx, img_id in enumerate(ids_shuffled, start=1):
        img_path = src_img_dir / f"{img_id}.png"
        lbl_path = src_lbl_dir / _mask_name(img_id)
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image for {img_id}: {img_path}")
        if not lbl_path.exists():
            raise FileNotFoundError(f"Missing label for {img_id}: {lbl_path}")

        numbered = f"{img_id}_{idx:03d}"
        dst_r_path = dst_img_dir / f"{numbered}_0000.png"
        dst_g_path = dst_img_dir / f"{numbered}_0001.png"
        dst_b_path = dst_img_dir / f"{numbered}_0002.png"
        dst_lbl_path = dst_lbl_dir / f"{numbered}.png"
        tasks.append(
            (img_path, lbl_path, dst_r_path, dst_g_path, dst_b_path, dst_lbl_path, numbered)
        )

    if workers < 1:
        workers = max(cpu_count() - 1, 1)

    ids: list[str] = []
    with Pool(processes=workers) as pool:
        for numbered in tqdm(
            pool.imap_unordered(_process_one, tasks, chunksize=8),
            total=len(tasks),
            desc=f"{src_img_dir.name} -> {dst_img_dir.name}",
            unit="img",
        ):
            ids.append(numbered)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert BUSBRA combined to nnUNet (fold-1 as test)."
    )
    parser.add_argument(
        "--csv",
        default="/home/ultrai/UltrAi/UNeXt/data/BUSBRA/5-fold-cv.csv",
        help="Path to BUSBRA 5-fold CSV file.",
    )
    parser.add_argument(
        "--images",
        default="/home/ultrai/UltrAi/UNeXt/data/BUSBRA/combined/images",
        help="Path to BUSBRA combined images directory.",
    )
    parser.add_argument(
        "--masks",
        default="/home/ultrai/UltrAi/UNeXt/data/BUSBRA/combined/masks",
        help="Path to BUSBRA combined masks directory.",
    )
    parser.add_argument(
        "--out",
        default="data/nnUNet_raw/Dataset301_busbra",
        help="Output nnUNet dataset directory.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Validation fold from CSV to use as test.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Train split ratio for remaining data (rest goes to val).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to shuffle before numbering.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (0 = cpu_count()-1).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    images_dir = Path(args.images)
    masks_dir = Path(args.masks)
    out_base = Path(args.out)

    images_tr = out_base / "imagesTr"
    labels_tr = out_base / "labelsTr"
    images_ts = out_base / "imagesTs"
    labels_ts = out_base / "labelsTs"
    for d in (images_tr, labels_tr, images_ts, labels_ts):
        d.mkdir(parents=True, exist_ok=True)

    fold_col = f"valid_{args.fold}"
    train_pool: list[str] = []
    test_raw: list[str] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row["ID"]
            if _is_nan(row.get(fold_col)):
                test_raw.append(img_id)
            else:
                train_pool.append(img_id)

    rng = random.Random(args.seed)
    rng.shuffle(train_pool)
    split_idx = int(len(train_pool) * args.train_ratio)
    train_raw = train_pool[:split_idx]
    val_raw = train_pool[split_idx:]

    train_ids = convert_ids(
        train_raw,
        images_dir,
        masks_dir,
        images_tr,
        labels_tr,
        args.seed,
        args.workers,
    )
    val_ids = convert_ids(
        val_raw,
        images_dir,
        masks_dir,
        images_tr,
        labels_tr,
        args.seed,
        args.workers,
    )
    test_ids = convert_ids(
        test_raw,
        images_dir,
        masks_dir,
        images_ts,
        labels_ts,
        args.seed,
        args.workers,
    )

    splits = [
        {
            "train": sorted(train_ids),
            "val": sorted(val_ids),
        }
    ]
    (out_base / "splits_final.json").write_text(json.dumps(splits, indent=4))

    dataset_json = {
        "channel_names": {"0": "Gray"},
        "labels": {"background": 0, "lesion": 1},
        "file_ending": ".png",
        "numTraining": len(train_ids) + len(val_ids),
    }
    (out_base / "dataset.json").write_text(json.dumps(dataset_json, indent=4))

    print(
        f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} numTraining={len(train_ids) + len(val_ids)}"
    )


if __name__ == "__main__":
    main()
