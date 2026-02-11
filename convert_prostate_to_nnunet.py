#!/usr/bin/env python3
"""Convert prostate domains A+B to nnUNet PNG layout with patient-level split."""

from __future__ import annotations

import argparse
import json
import random
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


PATIENT_RE = re.compile(r"^(?P<pid>[AB]_patient_\d+)_slice\d+_\d+_0000$")


def _extract_patient_id(img_path: Path) -> str:
    m = PATIENT_RE.match(img_path.stem)
    if not m:
        raise ValueError(f"Cannot parse patient id from filename: {img_path.name}")
    return m.group("pid")


def _case_id_from_img_name(img_name: str) -> str:
    if not img_name.endswith("_0000.png"):
        raise ValueError(f"Expected image name ending with _0000.png, got: {img_name}")
    return img_name[:-9]


def _process_one(task: tuple[Path, Path, Path, Path, str]) -> tuple[str, str]:
    img_path, lbl_path, dst_img_path, dst_lbl_path, split = task

    with Image.open(img_path) as im:
        im = im.convert("L")
        img_size = im.size
        im.save(dst_img_path)

    with Image.open(lbl_path) as lb:
        lb = lb.convert("L")
        if lb.size != img_size:
            lb = lb.resize(img_size, Image.NEAREST)
        arr = np.array(lb, dtype=np.uint8)
        arr = (arr > 0).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(dst_lbl_path)

    return split, dst_lbl_path.stem


def _collect_cases(src_root: Path, domains: list[str]) -> list[tuple[str, Path, Path]]:
    cases: list[tuple[str, Path, Path]] = []
    for domain in domains:
        img_dir = src_root / domain / "imagesTr"
        lbl_dir = src_root / domain / "labelsTr"
        if not img_dir.exists():
            raise FileNotFoundError(f"Missing imagesTr directory: {img_dir}")
        if not lbl_dir.exists():
            raise FileNotFoundError(f"Missing labelsTr directory: {lbl_dir}")

        for img_path in sorted(img_dir.glob("*_0000.png")):
            case_id = _case_id_from_img_name(img_path.name)
            lbl_path = lbl_dir / f"{case_id}.png"
            if not lbl_path.exists():
                raise FileNotFoundError(f"Missing label for {img_path.name}: {lbl_path}")
            patient_id = _extract_patient_id(img_path)
            cases.append((patient_id, img_path, lbl_path))
    return cases


def _split_patients(
    patient_ids: list[str], train_ratio: float, val_ratio: float, seed: int
) -> tuple[set[str], set[str], set[str]]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Require 0 < train-ratio, val-ratio and train-ratio + val-ratio < 1")

    rng = random.Random(seed)
    pids = sorted(set(patient_ids))
    rng.shuffle(pids)

    n = len(pids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    if n_train < 1:
        n_train = 1
    if n_val < 1:
        n_val = 1
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    if n_train + n_val >= n:
        n_train = max(1, n - n_val - 1)

    train_p = set(pids[:n_train])
    val_p = set(pids[n_train:n_train + n_val])
    test_p = set(pids[n_train + n_val:])
    return train_p, val_p, test_p


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert prostate A+B PNG slices into nnUNet dataset layout."
    )
    parser.add_argument(
        "--src",
        default="data/raw_data/prostate",
        help="Source prostate root containing A/ and B/ domain folders.",
    )
    parser.add_argument(
        "--out",
        default="data/nnUNet_raw/Dataset307_prostateAB",
        help="Output nnUNet dataset directory.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["A", "B"],
        help="Domain folders to include (for combining, keep A B).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Patient-level train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Patient-level validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for patient split.",
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

    cases = _collect_cases(src_root, args.domains)
    if not cases:
        raise RuntimeError(f"No *_0000.png cases found under {src_root}")

    patient_ids = [pid for pid, _, _ in cases]
    train_p, val_p, test_p = _split_patients(
        patient_ids, args.train_ratio, args.val_ratio, args.seed
    )

    tasks: list[tuple[Path, Path, Path, Path, str]] = []
    for patient_id, img_path, lbl_path in cases:
        case_id = _case_id_from_img_name(img_path.name)
        if patient_id in train_p or patient_id in val_p:
            split = "train" if patient_id in train_p else "val"
            dst_img = images_tr / f"{case_id}_0000.png"
            dst_lbl = labels_tr / f"{case_id}.png"
        else:
            split = "test"
            dst_img = images_ts / f"{case_id}_0000.png"
            dst_lbl = labels_ts / f"{case_id}.png"
        tasks.append((img_path, lbl_path, dst_img, dst_lbl, split))

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
