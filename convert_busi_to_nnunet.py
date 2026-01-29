#!/usr/bin/env python3
"""Convert BUSI dataset to nnUNet format (single combined dataset)."""

from __future__ import annotations

import argparse
import csv
import json
import re
import random
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def _sanitize_id(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_")


def _load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as m:
        m = m.convert("L")
        if m.size != size:
            m = m.resize(size, resample=Image.NEAREST)
        arr = np.array(m, dtype=np.uint8)
    return (arr > 0).astype(np.uint8)


def _process_one(task: tuple[Path, Path, list[Path], str, str]) -> dict:
    img_path, out_img, mask_paths, out_lbl, category = task
    with Image.open(img_path) as im:
        im = im.convert("L")
        im.save(out_img)
        size = im.size

    if mask_paths:
        masks = [_load_mask(p, size) for p in mask_paths]
        merged = np.clip(np.sum(masks, axis=0), 0, 1).astype(np.uint8)
    else:
        merged = np.zeros((size[1], size[0]), dtype=np.uint8)

    Image.fromarray(merged).convert("L").save(out_lbl)

    return {
        "id": out_lbl.stem,
        "category": category,
        "image_file": str(img_path),
        "mask_files": ";".join(str(p) for p in mask_paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert BUSI to nnUNet Dataset304.")
    parser.add_argument(
        "--src",
        default="/home/ultrai/UltrAi/nntinyunet/data/raw_data/Dataset_BUSI_with_GT",
        help="BUSI dataset root with category subfolders.",
    )
    parser.add_argument(
        "--out",
        default="data/nnUNet_raw/Dataset304_BUSI",
        help="Output nnUNet dataset directory.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["benign", "malignant", "normal"],
        help="Category subfolders to include.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (0 = cpu_count()-1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified split.",
    )
    args = parser.parse_args()

    src_root = Path(args.src)
    out_base = Path(args.out)
    images_tr = out_base / "imagesTr"
    labels_tr = out_base / "labelsTr"
    images_ts = out_base / "imagesTs"
    labels_ts = out_base / "labelsTs"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    images_ts.mkdir(parents=True, exist_ok=True)
    labels_ts.mkdir(parents=True, exist_ok=True)

    metadata_path = out_base / "metadata.csv"
    tasks: list[tuple[Path, Path, list[Path], str, str]] = []
    ids_by_cat: dict[str, list[str]] = {}
    for category in args.categories:
        cat_dir = src_root / category
        if not cat_dir.exists():
            raise FileNotFoundError(f"Missing category dir: {cat_dir}")

        img_files = sorted(p for p in cat_dir.glob("*.png") if "_mask" not in p.stem)
        for img_path in img_files:
            stem = img_path.stem
            safe_id = _sanitize_id(f"{category}_{stem}")
            out_img = images_tr / f"{safe_id}_0000.png"
            out_lbl = labels_tr / f"{safe_id}.png"
            mask_files = sorted(cat_dir.glob(f"{stem}_mask*.png"))
            tasks.append((img_path, out_img, mask_files, out_lbl, category))
            ids_by_cat.setdefault(category, []).append(safe_id)

    workers = args.workers
    if workers < 1:
        workers = max(cpu_count() - 1, 1)

    rows = []
    ids = []
    with Pool(processes=workers) as pool:
        for row in tqdm(
            pool.imap_unordered(_process_one, tasks, chunksize=8),
            total=len(tasks),
            desc="converting",
            unit="img",
        ):
            rows.append(row)
            ids.append(row["id"])

    with metadata_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "category", "image_file", "mask_files"])
        writer.writeheader()
        writer.writerows(rows)

    rng = random.Random(args.seed)
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    for category, cat_ids in ids_by_cat.items():
        cat_ids = list(cat_ids)
        rng.shuffle(cat_ids)
        n_cat = len(cat_ids)
        n_train = int(n_cat * 0.8)
        n_val = int(n_cat * 0.1)
        train_ids.extend(cat_ids[:n_train])
        val_ids.extend(cat_ids[n_train:n_train + n_val])
        test_ids.extend(cat_ids[n_train + n_val:])

    for tid in test_ids:
        src_img = images_tr / f"{tid}_0000.png"
        src_lbl = labels_tr / f"{tid}.png"
        dst_img = images_ts / f"{tid}_0000.png"
        dst_lbl = labels_ts / f"{tid}.png"
        if src_img.exists():
            src_img.replace(dst_img)
        if src_lbl.exists():
            src_lbl.replace(dst_lbl)

    splits = [{"train": sorted(train_ids), "val": sorted(val_ids)}]
    (out_base / "splits_final.json").write_text(json.dumps(splits, indent=4))

    dataset_json = {
        "channel_names": {"0": "Gray"},
        "labels": {"background": 0, "lesion": 1},
        "file_ending": ".png",
        "numTraining": len(train_ids) + len(val_ids),
    }
    (out_base / "dataset.json").write_text(json.dumps(dataset_json, indent=4))

    print(f"wrote {len(ids)} cases to {out_base}")


if __name__ == "__main__":
    main()
