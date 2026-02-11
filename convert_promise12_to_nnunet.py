#!/usr/bin/env python3
"""Convert PROMISE12 PNG slices to nnUNet format."""

from __future__ import annotations

import argparse
import json
import shutil
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def _process_one(task: tuple[Path, Path, Path, Path, str]) -> tuple[str, str]:
    img_path, lbl_path, dst_img_path, dst_lbl_path, split = task

    with Image.open(img_path) as im:
        im = im.convert("L")
        im.save(dst_img_path)
        size = im.size

    with Image.open(lbl_path) as lb:
        lb = lb.convert("L")
        if lb.size != size:
            lb = lb.resize(size, Image.NEAREST)
        arr = np.array(lb, dtype=np.uint8)
        arr = (arr > 0).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(dst_lbl_path)

    return split, dst_lbl_path.stem


def _collect_split_tasks(
    src_root: Path,
    split_name: str,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    out_split: str,
    case_prefix: str,
) -> list[tuple[Path, Path, Path, Path, str]]:
    img_dir = src_root / split_name / "image"
    lbl_dir = src_root / split_name / "mask"
    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError(f"Missing split folders under {src_root / split_name}")

    tasks: list[tuple[Path, Path, Path, Path, str]] = []
    for img_path in sorted(img_dir.glob("*.png")):
        lbl_path = lbl_dir / img_path.name
        if not lbl_path.exists():
            raise FileNotFoundError(f"Missing label for {img_path.name}: {lbl_path}")
        case_id = f"{case_prefix}{img_path.stem}"
        tasks.append(
            (
                img_path,
                lbl_path,
                dst_img_dir / f"{case_id}_0000.png",
                dst_lbl_dir / f"{case_id}.png",
                out_split,
            )
        )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PROMISE12 to nnUNet Dataset308.")
    parser.add_argument(
        "--src",
        default="data/raw_data/PROMISE12",
        help="PROMISE12 source directory.",
    )
    parser.add_argument(
        "--out",
        default="data/nnUNet_raw/Dataset308_PROMISE12",
        help="Output nnUNet dataset directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (0 = cpu_count()-1).",
    )
    args = parser.parse_args()

    src_root = Path(args.src)
    out_root = Path(args.out)
    images_tr = out_root / "imagesTr"
    labels_tr = out_root / "labelsTr"
    images_ts = out_root / "imagesTs"
    labels_ts = out_root / "labelsTs"
    for d in (images_tr, labels_tr, images_ts, labels_ts):
        d.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[Path, Path, Path, Path, str]] = []
    tasks += _collect_split_tasks(
        src_root, "train_data", images_tr, labels_tr, "train", "tr_"
    )
    tasks += _collect_split_tasks(
        src_root, "validation_data", images_tr, labels_tr, "val", "val_"
    )
    tasks += _collect_split_tasks(
        src_root, "test_data", images_ts, labels_ts, "test", "ts_"
    )

    workers = args.workers if args.workers > 0 else max(cpu_count() - 1, 1)
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    if workers == 1:
        iterator = map(_process_one, tasks)
    else:
        pool = Pool(processes=workers)
        iterator = pool.imap_unordered(_process_one, tasks, chunksize=32)
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
        "channel_names": {"0": "Gray"},
        "labels": {"background": 0, "prostate": 1},
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
