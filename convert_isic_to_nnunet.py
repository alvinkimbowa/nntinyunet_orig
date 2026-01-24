#!/usr/bin/env python3
"""Convert ISIC 2018 dataset into nnUNet PNG layout with numbered IDs."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count

from PIL import Image
from tqdm import tqdm


def collect_jpgs(src_dir: Path) -> list[Path]:
    return sorted(p for p in src_dir.glob("*.jpg") if p.is_file())


def _process_one(
    task: tuple[Path, Path, Path, Path, Path, Path, str]
) -> str:
    img_path, lbl_path, dst_r_path, dst_g_path, dst_b_path, dst_lbl_path, numbered = task
    with Image.open(img_path) as im:
        im = im.convert("RGB").resize((256, 256), Image.BILINEAR)
        r, g, b = im.split()
        r.save(dst_r_path)
        g.save(dst_g_path)
        b.save(dst_b_path)
    with Image.open(lbl_path) as lb:
        lb = lb.resize((256, 256), Image.NEAREST)
        lb = lb.convert("L").point(lambda p: 1 if p > 0 else 0, mode="L")
        lb.save(dst_lbl_path)
    return numbered


def convert_split(
    src_img_dir: Path,
    src_lbl_dir: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    prefix: str,
    seed: int,
    workers: int,
) -> list[str]:
    """Convert a split and return list of new ids (without suffixes)."""
    rng = random.Random(seed)
    imgs = collect_jpgs(src_img_dir)
    rng.shuffle(imgs)

    tasks: list[tuple[Path, Path, Path, Path, Path, Path, str]] = []
    for idx, img_path in enumerate(imgs, start=1):
        orig_id = img_path.stem  # e.g., ISIC_0000000
        lbl_path = src_lbl_dir / f"{orig_id}_segmentation.png"
        if not lbl_path.exists():
            raise FileNotFoundError(f"Missing label for {orig_id}: {lbl_path}")

        numbered = f"{orig_id}_{idx:03d}"
        if prefix:
            numbered = f"{prefix}{numbered}"

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
        description="Convert ISIC2018 to nnUNet format with numbered ids."
    )
    parser.add_argument(
        "--src",
        default="/home/ultrai/UltrAi/UNeXt/data/isic2018",
        help="Path to ISIC2018 base directory.",
    )
    parser.add_argument(
        "--out",
        default="data/nnUNet_raw/Dataset300_isic2018",
        help="Output nnUNet dataset directory.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional prefix to add before each new id.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to shuffle each split before numbering.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (0 = cpu_count()-1).",
    )
    args = parser.parse_args()

    src_base = Path(args.src)
    out_base = Path(args.out)

    train_img_dir = src_base / "ISIC2018_Task1-2_Training_Input"
    train_lbl_dir = src_base / "ISIC2018_Task1_Training_GroundTruth"
    val_img_dir = src_base / "ISIC2018_Task1-2_Validation_Input"
    val_lbl_dir = src_base / "ISIC2018_Task1_Validation_GroundTruth"
    test_img_dir = src_base / "ISIC2018_Task1-2_Test_Input"
    test_lbl_dir = src_base / "ISIC2018_Task1_Test_GroundTruth"

    images_tr = out_base / "imagesTr"
    labels_tr = out_base / "labelsTr"
    images_ts = out_base / "imagesTs"
    labels_ts = out_base / "labelsTs"

    for d in (images_tr, labels_tr, images_ts, labels_ts):
        d.mkdir(parents=True, exist_ok=True)

    train_ids = convert_split(
        train_img_dir,
        train_lbl_dir,
        images_tr,
        labels_tr,
        args.prefix,
        args.seed,
        args.workers,
    )
    val_ids = convert_split(
        val_img_dir,
        val_lbl_dir,
        images_tr,
        labels_tr,
        args.prefix,
        args.seed,
        args.workers,
    )
    test_ids = convert_split(
        test_img_dir,
        test_lbl_dir,
        images_ts,
        labels_ts,
        args.prefix,
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
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": {"background": 0, "lesion": 1},
        "file_ending": ".png",
        "numTraining": len(train_ids) + len(val_ids),
    }
    (out_base / "dataset.json").write_text(json.dumps(dataset_json, indent=4))

    # Optional sanity print
    print(
        f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} numTraining={len(train_ids) + len(val_ids)}"
    )


if __name__ == "__main__":
    main()
