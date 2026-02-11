#!/usr/bin/env python3
"""Convert REFUGE dataset to nnUNet format (disc/cup segmentation)."""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def _process_one(
    task: tuple[Path, Path, Path, Path, Path, Path, Path, str]
) -> tuple[str, str]:
    img_path, disc_path, cup_path, dst_r_path, dst_g_path, dst_b_path, dst_lbl_path, split = task

    with Image.open(img_path) as im:
        im = im.convert("RGB").resize((256, 256), Image.BILINEAR)
        r, g, b = im.split()
        r.save(dst_r_path)
        g.save(dst_g_path)
        b.save(dst_b_path)
        size = im.size

    with Image.open(disc_path) as dm:
        dm = dm.convert("L")
        dm = dm.resize(size, Image.NEAREST)
        disc_arr = np.array(dm, dtype=np.uint8)
        disc = (disc_arr == 128)

    with Image.open(cup_path) as cm:
        cm = cm.convert("L")
        cm = cm.resize(size, Image.NEAREST)
        cup_arr = np.array(cm, dtype=np.uint8)
        cup = (cup_arr == 0)

    lbl = np.zeros((size[1], size[0]), dtype=np.uint8)
    lbl[disc] = 1
    lbl[cup] = 2
    Image.fromarray(lbl, mode="L").save(dst_lbl_path)

    return split, dst_lbl_path.stem


def _collect_cases(root: Path) -> list[tuple[Path, Path, Path, str]]:
    cases: list[tuple[Path, Path, Path, str]] = []
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cid = case_dir.name
        img_path = case_dir / f"{cid}.jpg"
        disc_path = case_dir / f"{cid}_disc.bmp"
        cup_path = case_dir / f"{cid}_cup.bmp"
        if not (img_path.exists() and disc_path.exists() and cup_path.exists()):
            continue
        cases.append((img_path, disc_path, cup_path, cid))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert REFUGE to nnUNet Dataset310.")
    parser.add_argument(
        "--src",
        default="data/raw_data/REFUGE",
        help="REFUGE source directory.",
    )
    parser.add_argument(
        "--out",
        default="data/nnUNet_raw/Dataset310_REFUGE",
        help="Output nnUNet dataset directory.",
    )
    parser.add_argument(
        "--train-split-dir",
        default="Training-400",
        help="Folder name under --src used as training split.",
    )
    parser.add_argument(
        "--val-split-dir",
        default="Validation-400",
        help="Folder name under --src used as validation split.",
    )
    parser.add_argument(
        "--test-split-dir",
        default="Test-400",
        help="Folder name under --src used as test split.",
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

    train_dir = src_root / args.train_split_dir
    val_dir = src_root / args.val_split_dir
    test_dir = src_root / args.test_split_dir

    train_cases = _collect_cases(train_dir)
    val_cases = _collect_cases(val_dir)
    test_cases = _collect_cases(test_dir)
    if not train_cases:
        raise RuntimeError(f"No cases found in REFUGE train folder: {train_dir}")
    if not val_cases:
        raise RuntimeError(f"No cases found in REFUGE val folder: {val_dir}")
    if not test_cases:
        raise RuntimeError(f"No cases found in REFUGE test folder: {test_dir}")

    tasks: list[tuple[Path, Path, Path, Path, Path, Path, Path, str]] = []
    for img_path, disc_path, cup_path, cid in train_cases:
        case_id = f"REFUGE_{cid}"
        tasks.append(
            (
                img_path,
                disc_path,
                cup_path,
                images_tr / f"{case_id}_0000.png",
                images_tr / f"{case_id}_0001.png",
                images_tr / f"{case_id}_0002.png",
                labels_tr / f"{case_id}.png",
                "train",
            )
        )

    for img_path, disc_path, cup_path, cid in val_cases:
        case_id = f"REFUGE_{cid}"
        tasks.append(
            (
                img_path,
                disc_path,
                cup_path,
                images_tr / f"{case_id}_0000.png",
                images_tr / f"{case_id}_0001.png",
                images_tr / f"{case_id}_0002.png",
                labels_tr / f"{case_id}.png",
                "val",
            )
        )

    for img_path, disc_path, cup_path, cid in test_cases:
        case_id = f"REFUGE_{cid}"
        tasks.append(
            (
                img_path,
                disc_path,
                cup_path,
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
        iterator = pool.imap_unordered(_process_one, tasks, chunksize=8)
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
        "labels": {"background": 0, "disc": 1, "cup": 2},
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
