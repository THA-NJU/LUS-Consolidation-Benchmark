#!/usr/bin/env python3
"""Count unique patients in each size/split from image filename prefixes."""

from __future__ import annotations

import argparse
from pathlib import Path


SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def count_one_split(images_dir: Path) -> tuple[int, int, list[str]]:
    """Return (patient_count, image_count, sorted_patient_ids)."""
    image_files = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    patient_ids = sorted({path.stem.split("_", 1)[0] for path in image_files})
    return len(patient_ids), len(image_files), patient_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count patients in train/val/test. A patient ID is the part of an "
            "image filename before the first underscore, e.g. p001."
        )
    )
    parser.add_argument(
        "data_root",
        type=Path,
        help="Dataset root containing Size_512 and Size_224_filtered.",
        default=Path("./datasets"),
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["Size_512", "Size_224_filtered"],
        help=(
            "Size directory names (default: Size_512 Size_224_filtered). "
            "For the filtered dataset, use: --sizes Size_512 Size_224_filtered"
        ),
    )
    parser.add_argument(
        "--show-ids",
        action="store_true",
        help="Also print the patient IDs found in each split.",
    )
    args = parser.parse_args()

    print(f"{'Size':<20} {'Split':<8} {'Patients':>10} {'Images':>10}")
    print("-" * 52)

    had_missing_dir = False
    for size in args.sizes:
        size_patient_ids: set[str] = set()

        for split in SPLITS:
            images_dir = args.data_root / size / split / "images"
            if not images_dir.is_dir():
                print(f"{size:<20} {split:<8} {'MISSING':>10} {'MISSING':>10}")
                print(f"  Directory not found: {images_dir}")
                had_missing_dir = True
                continue

            patient_count, image_count, patient_ids = count_one_split(images_dir)
            size_patient_ids.update(patient_ids)
            print(f"{size:<20} {split:<8} {patient_count:>10} {image_count:>10}")
            if args.show_ids:
                print("  " + ", ".join(patient_ids))

        print(f"{size:<20} {'union':<8} {len(size_patient_ids):>10} {'-':>10}")
        print()

    if had_missing_dir:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
