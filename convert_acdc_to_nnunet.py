#!/usr/bin/env python3
"""Convert ACDC dataset into nnUNet NIfTI layout with patient-level split."""

from __future__ import annotations

import argparse
import json
import random
from multiprocessing import Pool, cpu_count
from pathlib import Path
import shutil

from tqdm import tqdm


def _strip_nii_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def _process_one(task: tuple[Path, Path, Path, Path, str]) -> tuple[str, str]:
    img_path, lbl_path, dst_img_dir, dst_lbl_dir, split = task

    base_id = _strip_nii_suffix(img_path)
    dst_img_path = dst_img_dir / f"{base_id}_0000.nii.gz"
    dst_lbl_path = dst_lbl_dir / f"{base_id}.nii.gz"
    shutil.copy2(img_path, dst_img_path)
    shutil.copy2(lbl_path, dst_lbl_path)
    return split, base_id


def _collect_labeled_frames(root: Path) -> list[tuple[str, Path, Path]]:
    frames: list[tuple[str, Path, Path]] = []
    for patient_dir in sorted(p for p in root.glob("patient*") if p.is_dir()):
        patient_id = patient_dir.name
        for img_path in sorted(patient_dir.glob(f"{patient_id}_frame*.nii.gz")):
            if img_path.name.endswith("_gt.nii.gz"):
                continue
            lbl_path = img_path.with_name(f"{_strip_nii_suffix(img_path)}_gt.nii.gz")
            if not lbl_path.exists():
                continue
            frames.append((patient_id, img_path, lbl_path))
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ACDC to nnUNet NIfTI format with patient-level split."
    )
    parser.add_argument(
        "--src",
        default="data/raw_data/ACDC/database",
        help="Path to ACDC database directory containing training/testing folders.",
    )
    parser.add_argument(
        "--out",
        default="data/nnUNet_raw/Dataset306_ACDC",
        help="Output nnUNet dataset directory.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio (patient-level) from training split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for patient-level split.",
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

    src_base = Path(args.src)
    out_base = Path(args.out)

    training_root = src_base / "training"
    testing_root = src_base / "testing"

    images_tr = out_base / "imagesTr"
    labels_tr = out_base / "labelsTr"
    images_ts = out_base / "imagesTs"
    labels_ts = out_base / "labelsTs"
    for d in (images_tr, labels_tr, images_ts, labels_ts):
        d.mkdir(parents=True, exist_ok=True)

    training_frames = _collect_labeled_frames(training_root)
    testing_frames = _collect_labeled_frames(testing_root)

    patients = sorted({pid for pid, _, _ in training_frames})
    if not patients:
        raise RuntimeError(f"No labeled training frames found in {training_root}")

    rng = random.Random(args.seed)
    rng.shuffle(patients)

    split_idx = int(len(patients) * (1.0 - args.val_ratio))
    split_idx = min(max(split_idx, 1), len(patients) - 1)
    train_patients = set(patients[:split_idx])
    val_patients = set(patients[split_idx:])

    tasks: list[tuple[Path, Path, Path, Path, str]] = []
    for patient_id, img_path, lbl_path in training_frames:
        if patient_id in train_patients:
            tasks.append((img_path, lbl_path, images_tr, labels_tr, "train"))
        elif patient_id in val_patients:
            tasks.append((img_path, lbl_path, images_tr, labels_tr, "val"))

    for _, img_path, lbl_path in testing_frames:
        tasks.append((img_path, lbl_path, images_ts, labels_ts, "test"))

    if args.workers < 1:
        workers = max(cpu_count() - 1, 1)
    else:
        workers = args.workers

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []

    with Pool(processes=workers) as pool:
        for split, case_id in tqdm(
            pool.imap_unordered(_process_one, tasks, chunksize=2),
            total=len(tasks),
            desc="converting",
            unit="vol",
        ):
            if split == "train":
                train_ids.append(case_id)
            elif split == "val":
                val_ids.append(case_id)
            elif split == "test":
                test_ids.append(case_id)

    splits = [{"train": sorted(train_ids), "val": sorted(val_ids)}]
    (out_base / "splits_final.json").write_text(json.dumps(splits, indent=4))

    dataset_json = {
        "channel_names": {"0": "MRI"},
        "labels": {"background": 0, "RV": 1, "myocardium": 2, "LV": 3},
        "file_ending": ".nii.gz",
        "numTraining": len(train_ids) + len(val_ids),
    }
    (out_base / "dataset.json").write_text(json.dumps(dataset_json, indent=4))

    print(
        f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} "
        f"numTraining={len(train_ids) + len(val_ids)}"
    )


if __name__ == "__main__":
    main()
