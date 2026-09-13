#!/usr/bin/env python3
"""Preflight checks for the Size_512 consolidation segmentation dataset.

Put this file in the APRIL-MedSeg repository root, edit DATA_ROOT below when
needed, then run:

    python check_consolidation_size512.py

The script verifies image/mask pairing, image sizes, mask values, positive and
negative case counts, and patient leakage across train/val/test.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


# =============================================================================
# User settings
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = (PROJECT_ROOT / "./datasets/Size_512").resolve()
REPORT_DIR = PROJECT_ROOT / "output" / "consolidation_size512_dataset_check"
EXPECTED_SIZE = (512, 512)
IMAGE_SUFFIX = ".png"
MASK_SUFFIX = ".png"
PATIENT_TOKEN_INDEX = 0  # pXXX_YYY_ZZZZ.png -> pXXX
SCAN_ALL_FILES = True
MAX_FILES_PER_SPLIT_WHEN_NOT_FULL = 1000
STOP_ON_PATIENT_LEAKAGE = True


def patient_id_from_name(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) <= PATIENT_TOKEN_INDEX:
        raise ValueError(f"Cannot parse patient id from filename: {filename}")
    return parts[PATIENT_TOKEN_INDEX]


def list_pairs(split_dir: Path) -> Tuple[List[Tuple[Path, Path]], List[str], List[str]]:
    image_dir = split_dir / "images"
    mask_dir = split_dir / "masks"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing mask directory: {mask_dir}")

    images = {p.stem: p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() == IMAGE_SUFFIX}
    masks = {p.stem: p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() == MASK_SUFFIX}
    common = sorted(images.keys() & masks.keys())
    image_only = sorted(images.keys() - masks.keys())
    mask_only = sorted(masks.keys() - images.keys())
    return [(images[k], masks[k]) for k in common], image_only, mask_only


def scan_split(split: str, split_dir: Path) -> Dict[str, object]:
    pairs, image_only, mask_only = list_pairs(split_dir)
    scan_pairs = pairs if SCAN_ALL_FILES else pairs[:MAX_FILES_PER_SPLIT_WHEN_NOT_FULL]

    image_sizes: Counter[str] = Counter()
    image_modes: Counter[str] = Counter()
    mask_sizes: Counter[str] = Counter()
    mask_modes: Counter[str] = Counter()
    mask_value_counter: Counter[int] = Counter()
    invalid_names: List[str] = []
    positive_cases = 0
    negative_cases = 0
    foreground_pixels = 0
    total_pixels = 0
    patient_ids: Set[str] = set()
    duplicate_stems: List[str] = []

    seen_stems: Set[str] = set()
    for image_path, mask_path in tqdm(scan_pairs, desc=f"scan {split}", unit="pair"):
        if image_path.stem in seen_stems:
            duplicate_stems.append(image_path.stem)
        seen_stems.add(image_path.stem)

        try:
            patient_ids.add(patient_id_from_name(image_path.name))
            parts = image_path.stem.split("_")
            if len(parts) != 3 or parts[2] != "0001":
                invalid_names.append(image_path.name)
        except ValueError:
            invalid_names.append(image_path.name)

        with Image.open(image_path) as im:
            image_sizes[str(im.size)] += 1
            image_modes[im.mode] += 1

        with Image.open(mask_path) as mm:
            mask_sizes[str(mm.size)] += 1
            mask_modes[mm.mode] += 1
            arr = np.asarray(mm)
            if arr.ndim == 3:
                arr = arr[..., 0]
            unique, counts = np.unique(arr, return_counts=True)
            for value, count in zip(unique.tolist(), counts.tolist()):
                mask_value_counter[int(value)] += int(count)
            positive = arr > 0
            fg = int(positive.sum())
            foreground_pixels += fg
            total_pixels += int(positive.size)
            if fg > 0:
                positive_cases += 1
            else:
                negative_cases += 1

    result: Dict[str, object] = {
        "split": split,
        "split_dir": str(split_dir),
        "paired_samples": len(pairs),
        "scanned_samples": len(scan_pairs),
        "image_without_mask": len(image_only),
        "mask_without_image": len(mask_only),
        "image_only_examples": image_only[:20],
        "mask_only_examples": mask_only[:20],
        "positive_cases": positive_cases,
        "negative_cases": negative_cases,
        "positive_case_fraction": positive_cases / max(len(scan_pairs), 1),
        "foreground_pixels": foreground_pixels,
        "total_pixels": total_pixels,
        "foreground_pixel_fraction": foreground_pixels / max(total_pixels, 1),
        "patient_count": len(patient_ids),
        "patient_ids": sorted(patient_ids),
        "image_sizes": dict(image_sizes),
        "mask_sizes": dict(mask_sizes),
        "image_modes": dict(image_modes),
        "mask_modes": dict(mask_modes),
        "mask_values": {str(k): v for k, v in sorted(mask_value_counter.items())},
        "invalid_name_count": len(invalid_names),
        "invalid_name_examples": invalid_names[:30],
        "duplicate_stem_count": len(duplicate_stems),
        "duplicate_stem_examples": duplicate_stems[:20],
    }
    return result


def main() -> None:
    print(f"DATA_ROOT: {DATA_ROOT}")
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(
            f"Size_512 dataset not found: {DATA_ROOT}\n"
            "Edit DATA_ROOT at the top of this script."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    split_results: Dict[str, Dict[str, object]] = {}
    for split in ("train", "val", "test"):
        split_results[split] = scan_split(split, DATA_ROOT / split)

    patient_sets = {
        split: set(result["patient_ids"]) for split, result in split_results.items()
    }
    overlaps = {
        "train_val": sorted(patient_sets["train"] & patient_sets["val"]),
        "train_test": sorted(patient_sets["train"] & patient_sets["test"]),
        "val_test": sorted(patient_sets["val"] & patient_sets["test"]),
    }

    report = {
        "data_root": str(DATA_ROOT),
        "expected_size": list(EXPECTED_SIZE),
        "splits": split_results,
        "patient_overlap": {k: {"count": len(v), "examples": v[:50]} for k, v in overlaps.items()},
    }

    with open(REPORT_DIR / "dataset_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(REPORT_DIR / "dataset_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split", "paired_samples", "patient_count", "positive_cases",
                "negative_cases", "positive_case_fraction", "foreground_pixel_fraction",
                "image_without_mask", "mask_without_image", "invalid_name_count",
            ],
        )
        writer.writeheader()
        for split in ("train", "val", "test"):
            r = split_results[split]
            writer.writerow({k: r[k] for k in writer.fieldnames})

    print("\n================ DATASET SUMMARY ================")
    for split in ("train", "val", "test"):
        r = split_results[split]
        print(
            f"{split:>5}: pairs={r['paired_samples']}, patients={r['patient_count']}, "
            f"positive={r['positive_cases']}, negative={r['negative_cases']}, "
            f"positive_fraction={r['positive_case_fraction']:.4f}, "
            f"fg_pixel_fraction={r['foreground_pixel_fraction']:.6f}"
        )
        print(f"       image_sizes={r['image_sizes']} mask_sizes={r['mask_sizes']}")
        print(f"       mask_values={list(r['mask_values'].keys())}")
        if r["image_without_mask"] or r["mask_without_image"]:
            print(
                f"       WARNING: image_without_mask={r['image_without_mask']}, "
                f"mask_without_image={r['mask_without_image']}"
            )
        if r["invalid_name_count"]:
            print(f"       WARNING: invalid filenames={r['invalid_name_count']}")

    print("\nPatient overlap:")
    for key, values in overlaps.items():
        print(f"  {key}: {len(values)}")

    bad_overlap = any(overlaps.values())
    print(f"\nReports written to: {REPORT_DIR}")
    if bad_overlap and STOP_ON_PATIENT_LEAKAGE:
        raise RuntimeError(
            "Patient leakage detected across train/val/test. "
            "Do not start the benchmark until the split is fixed."
        )


if __name__ == "__main__":
    main()
