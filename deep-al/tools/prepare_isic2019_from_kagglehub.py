#!/usr/bin/env python3
"""
Prepare ISIC2019 into ImageFolder layout used by deep-al:

  <output_root>/train/<class_name>/*.jpg
  <output_root>/test/<class_name>/*.jpg

Input expected from kagglehub dataset "salviohexia/isic-2019-skin-lesion-images-for-classification".
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_root",
        type=Path,
        required=True,
        help="Kagglehub dataset root path",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/home/jiayuan/TypiClust/data/isic2019"),
        help="Output directory for ImageFolder layout",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlink",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation split ratio when only class-folders are available",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for fallback split",
    )
    return parser.parse_args()


def find_one(root: Path, pattern: str) -> Path:
    matches = list(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Cannot find {pattern} under {root}")
    # Prefer shorter path (usually canonical file location)
    matches = sorted(matches, key=lambda p: len(str(p)))
    return matches[0]


def ensure_dirs(out_root: Path):
    for split in ["train", "test"]:
        for cls in CLASS_NAMES:
            (out_root / split / cls).mkdir(parents=True, exist_ok=True)


def clear_existing_split(out_root: Path):
    for split in ["train", "test"]:
        split_dir = out_root / split
        if not split_dir.exists():
            continue
        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            for p in cls_dir.iterdir():
                if p.is_symlink() or p.is_file():
                    p.unlink()


def get_label(row: dict) -> str:
    # The metadata usually has one-hot columns for class names.
    for c in CLASS_NAMES:
        v = row.get(c)
        if v is None:
            continue
        try:
            if int(float(v)) == 1:
                return c
        except Exception:
            pass
    raise ValueError(f"No class found for row image={row.get('image')}")


def link_or_copy(src: Path, dst: Path, do_copy: bool):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if do_copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def build_split(
    split_name: str,
    csv_path: Path,
    image_dir: Path,
    out_root: Path,
    do_copy: bool,
):
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    missing = 0
    total = 0
    for row in rows:
        image_id = row["image"]
        cls = get_label(row)
        src = image_dir / f"{image_id}.jpg"
        if not src.exists():
            # some packages might use .png
            src = image_dir / f"{image_id}.png"
        if not src.exists():
            missing += 1
            continue

        dst = out_root / split_name / cls / src.name
        link_or_copy(src, dst, do_copy)
        total += 1

    return total, missing


def main():
    args = parse_args()
    src_root = args.source_root.resolve()
    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    ensure_dirs(out_root)
    clear_existing_split(out_root)

    train_total = 0
    train_missing = 0
    test_total = 0
    test_missing = 0

    # Mode A: official package with explicit training/validation files.
    try:
        train_csv = find_one(src_root, "*Train_GroundTruth.csv")
        val_csv = find_one(src_root, "*Validation_GroundTruth.csv")
        train_img_dir = find_one(src_root, "*_Training_Input")
        val_img_dir = find_one(src_root, "*_Validation_Input")

        train_total, train_missing = build_split(
            "train", train_csv, train_img_dir, out_root, args.copy
        )
        test_total, test_missing = build_split(
            "test", val_csv, val_img_dir, out_root, args.copy
        )
    except FileNotFoundError:
        # Mode B: class-folders package, create deterministic per-class split.
        rng = random.Random(args.seed)
        for cls in CLASS_NAMES:
            cls_dir = src_root / cls
            if not cls_dir.exists():
                continue

            images = sorted(
                [
                    p
                    for p in cls_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ]
            )
            rng.shuffle(images)

            n_test = int(len(images) * args.val_ratio)
            # Ensure both splits exist when class has enough samples.
            if len(images) > 1:
                n_test = max(1, min(n_test, len(images) - 1))

            test_imgs = images[:n_test]
            train_imgs = images[n_test:]

            for src in train_imgs:
                dst = out_root / "train" / cls / src.name
                link_or_copy(src, dst, args.copy)
                train_total += 1

            for src in test_imgs:
                dst = out_root / "test" / cls / src.name
                link_or_copy(src, dst, args.copy)
                test_total += 1

    print("Prepared ISIC2019 dataset.")
    print(f"source_root: {src_root}")
    print(f"output_root: {out_root}")
    print(f"train linked/copied: {train_total}, missing: {train_missing}")
    print(f"test  linked/copied: {test_total}, missing: {test_missing}")


if __name__ == "__main__":
    main()
