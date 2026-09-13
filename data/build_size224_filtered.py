#!/usr/bin/env python3
"""Build a filtered 224x224 consolidation dataset from the current Size_512 dataset.

Pipeline
--------
1. Pool all paired 512x512 image/mask files from the existing train/val/test folders.
   The old split assignment is ignored.
2. For every source image, crop a 3x3 grid of 224x224 patches using start
   positions [0, 144, 288] on both axes (9 patches total, row-major order).
3. Classify every patch:
      class1_empty : no foreground pixel in the mask -> discard
      class2_edge  : foreground exists and touches the patch border -> discard
      class3_keep  : foreground exists and does not touch the patch border -> keep
4. Split patients with at least one kept patch into train/val/test = 8:1:1.
   Patient ID is the first filename field, e.g. p037 in p037_005_0001.png.
   All kept patches from the same patient always go to the same split.
5. Save only class3_keep image/mask pairs. Filenames are not re-indexed:
      p037_005_0001.png ... p037_005_0009.png
   Patch numbers preserve the original 1-9 row-major crop positions.
6. Write audit CSV/JSON files describing every candidate patch and the split.

Expected input layout
---------------------
Size_512/
  train/images/*.png
  train/masks/*.png
  val/images/*.png
  val/masks/*.png
  test/images/*.png
  test/masks/*.png

Expected source filename
------------------------
pxxx_xxx_0001.png
  first field  = patient ID
  second field = image index within patient
  third field  = 0001 for a 512x512 source image

The prepared Size_512 masks are assumed to be binary; any pixel > 0 is treated
as consolidation foreground.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


SOURCE_SPLITS: Tuple[str, ...] = ("train", "val", "test")
FINAL_SPLITS: Tuple[str, ...] = ("train", "val", "test")
SOURCE_NAME_RE = re.compile(
    r"^(p\d{3})_(\d{3})_(\d{3})\.png$",
    re.IGNORECASE,
)

PATCH_SIZE = 224
SOURCE_SIZE = 512
STRIDE = 144
PATCH_POSITIONS: Tuple[int, ...] = (0, STRIDE, SOURCE_SIZE - PATCH_SIZE)  # 0,144,288
PATCHES_PER_IMAGE = 9


@dataclass(frozen=True)
class SourcePair:
    source_split: str
    image_path: Path
    mask_path: Path
    filename: str
    patient_id: str
    image_index: str


@dataclass
class PatchRecord:
    source_split: str
    source_filename: str
    patient_id: str
    image_index: str
    patch_index: int
    output_filename: str
    left: int
    top: int
    foreground_pixels: int
    touches_edge: bool
    category: str
    final_split: str = ""


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_source = (
        script_dir / "./datasets/Size_512"
    ).resolve()

    parser = argparse.ArgumentParser(
        description=(
            "Crop Size_512 into 9 Size_224 patches, discard empty/edge-foreground "
            "patches, then re-split by patient at 8:1:1."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=default_source,
        help=f"Input Size_512 root (default: {default_source})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output root (default: Size_224_filtered next to Size_512)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for patient-level 8:1:1 split (default: 42)",
    )
    parser.add_argument(
        "--edge-width",
        type=int,
        default=1,
        help=(
            "Border width in pixels used for class2_edge. Default 1 means a "
            "foreground pixel on the outermost row/column causes rejection."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze/filter/split and write nothing.",
    )
    return parser.parse_args()


def png_map(directory: Path) -> Dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Required directory does not exist: {directory}")
    return {
        p.name: p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() == ".png"
    }


def parse_source_name(filename: str) -> Tuple[str, str]:
    match = SOURCE_NAME_RE.fullmatch(filename)
    if match is None:
        raise ValueError(
            f"Unsupported source filename {filename!r}; expected pxxx_xxx_xxx.png"
        )
    patient_id = match.group(1).lower()
    image_index = match.group(2)
    source_patch_index = match.group(3)
    if source_patch_index != "001":
        raise ValueError(
            f"Size_512 source must end in _001.png, got {filename!r}"
        )
    return patient_id, image_index


def check_512_size(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.load()
            size = image.size
    except Exception as exc:
        raise RuntimeError(f"Cannot read PNG: {path}") from exc
    if size != (SOURCE_SIZE, SOURCE_SIZE):
        raise ValueError(f"Expected 512x512 file, got {size}: {path}")


def collect_source_pairs(source_root: Path) -> List[SourcePair]:
    """Pool the old Size_512 train/val/test splits and ignore old split membership."""
    pairs: List[SourcePair] = []
    seen_filenames: Dict[str, Path] = {}
    seen_image_ids: Dict[Tuple[str, str], str] = {}

    for old_split in SOURCE_SPLITS:
        image_map = png_map(source_root / old_split / "images")
        mask_map = png_map(source_root / old_split / "masks")

        missing_masks = sorted(set(image_map) - set(mask_map))
        missing_images = sorted(set(mask_map) - set(image_map))
        if missing_masks or missing_images:
            raise RuntimeError(
                f"Unpaired files in old split={old_split}: "
                f"images_without_mask={missing_masks[:10]}, "
                f"masks_without_image={missing_images[:10]}"
            )

        for filename in sorted(image_map):
            if filename in seen_filenames:
                raise RuntimeError(
                    f"Duplicate source filename across old splits: {filename}; "
                    f"first={seen_filenames[filename]}, second={image_map[filename]}"
                )

            patient_id, image_index = parse_source_name(filename)
            image_id = (patient_id, image_index)
            if image_id in seen_image_ids:
                raise RuntimeError(
                    f"Duplicate patient/image ID {image_id}: "
                    f"{seen_image_ids[image_id]!r} and {filename!r}"
                )

            check_512_size(image_map[filename])
            check_512_size(mask_map[filename])

            seen_filenames[filename] = image_map[filename]
            seen_image_ids[image_id] = filename
            pairs.append(
                SourcePair(
                    source_split=old_split,
                    image_path=image_map[filename],
                    mask_path=mask_map[filename],
                    filename=filename,
                    patient_id=patient_id,
                    image_index=image_index,
                )
            )

    if not pairs:
        raise RuntimeError(f"No paired Size_512 PNG files found under {source_root}")
    return pairs


def patch_boxes() -> List[Tuple[int, int, int, int, int]]:
    """Return (patch_index, left, top, right, bottom) in row-major order."""
    boxes: List[Tuple[int, int, int, int, int]] = []
    patch_index = 1
    for top in PATCH_POSITIONS:
        for left in PATCH_POSITIONS:
            boxes.append(
                (patch_index, left, top, left + PATCH_SIZE, top + PATCH_SIZE)
            )
            patch_index += 1
    if len(boxes) != PATCHES_PER_IMAGE:
        raise AssertionError(f"Expected 9 crop boxes, got {len(boxes)}")
    return boxes


def foreground_array(mask_patch: Image.Image) -> np.ndarray:
    arr = np.asarray(mask_patch)
    if arr.ndim == 2:
        return arr > 0
    if arr.ndim == 3:
        return np.any(arr > 0, axis=-1)
    raise ValueError(f"Unsupported mask shape: {arr.shape}")


def foreground_touches_edge(fg: np.ndarray, edge_width: int) -> bool:
    if edge_width <= 0:
        raise ValueError(f"edge_width must be >= 1, got {edge_width}")
    if fg.ndim != 2:
        raise ValueError(f"Foreground array must be 2D, got shape={fg.shape}")
    h, w = fg.shape
    if edge_width * 2 > min(h, w):
        raise ValueError(
            f"edge_width={edge_width} is too large for patch shape {fg.shape}"
        )
    return bool(
        fg[:edge_width, :].any()
        or fg[-edge_width:, :].any()
        or fg[:, :edge_width].any()
        or fg[:, -edge_width:].any()
    )


def analyze_patches(
    pairs: Sequence[SourcePair],
    edge_width: int,
) -> List[PatchRecord]:
    records: List[PatchRecord] = []
    boxes = patch_boxes()

    for i, pair in enumerate(pairs, start=1):
        with Image.open(pair.mask_path) as mask_file:
            mask = mask_file.copy()

        for patch_index, left, top, right, bottom in boxes:
            mask_patch = mask.crop((left, top, right, bottom))
            fg = foreground_array(mask_patch)
            fg_pixels = int(fg.sum())

            if fg_pixels == 0:
                touches = False
                category = "class1_empty"
            else:
                touches = foreground_touches_edge(fg, edge_width=edge_width)
                category = "class2_edge" if touches else "class3_keep"

            output_filename = (
                f"{pair.patient_id}_{pair.image_index}_{patch_index:04d}.png"
            )
            records.append(
                PatchRecord(
                    source_split=pair.source_split,
                    source_filename=pair.filename,
                    patient_id=pair.patient_id,
                    image_index=pair.image_index,
                    patch_index=patch_index,
                    output_filename=output_filename,
                    left=left,
                    top=top,
                    foreground_pixels=fg_pixels,
                    touches_edge=touches,
                    category=category,
                )
            )

        if i % 100 == 0 or i == len(pairs):
            print(f"Analyzed source images: {i}/{len(pairs)}", flush=True)

    expected = len(pairs) * PATCHES_PER_IMAGE
    if len(records) != expected:
        raise AssertionError(f"Expected {expected} patch records, got {len(records)}")
    return records


def proportional_counts(n: int) -> Tuple[int, int, int]:
    """Nearest integer 8:1:1 allocation using the largest-remainder method."""
    if n < 3:
        raise ValueError(
            f"Need at least 3 patients with kept patches for train/val/test, got {n}"
        )

    ratios = (0.8, 0.1, 0.1)
    raw = [n * r for r in ratios]
    counts = [int(x) for x in raw]
    remainder = n - sum(counts)
    order = sorted(
        range(3),
        key=lambda idx: (raw[idx] - counts[idx], -idx),
        reverse=True,
    )
    for idx in order[:remainder]:
        counts[idx] += 1

    # For realistic benchmark patient counts this naturally yields non-empty
    # val/test. Guard tiny edge cases explicitly.
    if counts[1] == 0 or counts[2] == 0:
        if n < 10:
            counts = [n - 2, 1, 1]
        else:
            raise AssertionError(f"Unexpected empty validation/test split: {counts}")
    return counts[0], counts[1], counts[2]


def assign_patient_splits(
    records: Sequence[PatchRecord],
    seed: int,
) -> Dict[str, str]:
    kept_patients = sorted(
        {record.patient_id for record in records if record.category == "class3_keep"}
    )
    rng = random.Random(seed)
    rng.shuffle(kept_patients)

    n_train, n_val, n_test = proportional_counts(len(kept_patients))
    train_patients = kept_patients[:n_train]
    val_patients = kept_patients[n_train : n_train + n_val]
    test_patients = kept_patients[n_train + n_val :]

    if len(test_patients) != n_test:
        raise AssertionError(
            f"Patient allocation mismatch: expected test={n_test}, got {len(test_patients)}"
        )

    mapping: Dict[str, str] = {}
    for patient_id in train_patients:
        mapping[patient_id] = "train"
    for patient_id in val_patients:
        mapping[patient_id] = "val"
    for patient_id in test_patients:
        mapping[patient_id] = "test"

    if len(mapping) != len(kept_patients):
        raise AssertionError("A patient was assigned more than once or not assigned")

    for record in records:
        if record.category == "class3_keep":
            record.final_split = mapping[record.patient_id]

    return mapping


def prepare_output(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        has_content = any(output_root.iterdir())
        if has_content and not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_root}. "
                "Use a new directory or pass --overwrite."
            )
        if overwrite:
            shutil.rmtree(output_root)

    for split in FINAL_SPLITS:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "masks").mkdir(parents=True, exist_ok=True)


def pair_lookup(pairs: Sequence[SourcePair]) -> Dict[str, SourcePair]:
    lookup = {pair.filename: pair for pair in pairs}
    if len(lookup) != len(pairs):
        raise RuntimeError("Duplicate source filenames encountered")
    return lookup


def save_kept_patches(
    output_root: Path,
    pairs: Sequence[SourcePair],
    records: Sequence[PatchRecord],
) -> Counter:
    lookup = pair_lookup(pairs)
    by_source: Dict[str, List[PatchRecord]] = defaultdict(list)
    for record in records:
        if record.category == "class3_keep":
            by_source[record.source_filename].append(record)

    saved: Counter = Counter()
    processed_sources = 0
    for source_filename in sorted(by_source):
        pair = lookup[source_filename]
        with Image.open(pair.image_path) as image_file, Image.open(pair.mask_path) as mask_file:
            image = image_file.copy()
            mask = mask_file.copy()

        for record in sorted(by_source[source_filename], key=lambda r: r.patch_index):
            box = (
                record.left,
                record.top,
                record.left + PATCH_SIZE,
                record.top + PATCH_SIZE,
            )
            image_patch = image.crop(box)
            mask_patch = mask.crop(box)
            if image_patch.size != (PATCH_SIZE, PATCH_SIZE):
                raise RuntimeError(
                    f"Bad image crop size for {source_filename}/{record.patch_index}: "
                    f"{image_patch.size}"
                )
            if mask_patch.size != (PATCH_SIZE, PATCH_SIZE):
                raise RuntimeError(
                    f"Bad mask crop size for {source_filename}/{record.patch_index}: "
                    f"{mask_patch.size}"
                )

            split = record.final_split
            if split not in FINAL_SPLITS:
                raise RuntimeError(f"Missing final split for kept patch: {record}")

            image_out = output_root / split / "images" / record.output_filename
            mask_out = output_root / split / "masks" / record.output_filename
            image_patch.save(image_out)
            mask_patch.save(mask_out)
            saved[split] += 1

        processed_sources += 1
        if processed_sources % 100 == 0 or processed_sources == len(by_source):
            print(
                f"Saved kept patches from source images: "
                f"{processed_sources}/{len(by_source)}",
                flush=True,
            )

    return saved


def write_patch_audit(path: Path, records: Sequence[PatchRecord]) -> None:
    fieldnames = [
        "source_split",
        "source_filename",
        "patient_id",
        "image_index",
        "patch_index",
        "output_filename",
        "left",
        "top",
        "foreground_pixels",
        "touches_edge",
        "category",
        "final_split",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_patient_split_csv(
    path: Path,
    pairs: Sequence[SourcePair],
    records: Sequence[PatchRecord],
    patient_split: Dict[str, str],
) -> None:
    source_image_counts: Counter = Counter(pair.patient_id for pair in pairs)
    kept_patch_counts: Counter = Counter(
        record.patient_id for record in records if record.category == "class3_keep"
    )
    class1_counts: Counter = Counter(
        record.patient_id for record in records if record.category == "class1_empty"
    )
    class2_counts: Counter = Counter(
        record.patient_id for record in records if record.category == "class2_edge"
    )

    fieldnames = [
        "patient_id",
        "split",
        "source_images",
        "kept_patches",
        "class1_empty_patches",
        "class2_edge_patches",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for patient_id in sorted(patient_split):
            writer.writerow(
                {
                    "patient_id": patient_id,
                    "split": patient_split[patient_id],
                    "source_images": source_image_counts[patient_id],
                    "kept_patches": kept_patch_counts[patient_id],
                    "class1_empty_patches": class1_counts[patient_id],
                    "class2_edge_patches": class2_counts[patient_id],
                }
            )


def verify_output(
    output_root: Path,
    records: Sequence[PatchRecord],
    patient_split: Dict[str, str],
) -> None:
    expected: Dict[str, set[str]] = {split: set() for split in FINAL_SPLITS}
    for record in records:
        if record.category != "class3_keep":
            continue
        expected[record.final_split].add(record.output_filename)

    observed_patient_splits: Dict[str, set[str]] = defaultdict(set)
    for split in FINAL_SPLITS:
        image_names = {
            p.name for p in (output_root / split / "images").glob("*.png") if p.is_file()
        }
        mask_names = {
            p.name for p in (output_root / split / "masks").glob("*.png") if p.is_file()
        }
        if image_names != mask_names:
            raise RuntimeError(
                f"Image/mask output mismatch in split={split}: "
                f"images={len(image_names)}, masks={len(mask_names)}"
            )
        if image_names != expected[split]:
            missing = sorted(expected[split] - image_names)[:10]
            extra = sorted(image_names - expected[split])[:10]
            raise RuntimeError(
                f"Output verification failed in split={split}: "
                f"expected={len(expected[split])}, observed={len(image_names)}, "
                f"missing_examples={missing}, extra_examples={extra}"
            )

        for name in image_names:
            match = SOURCE_NAME_RE.fullmatch(name)
            if match is None:
                raise RuntimeError(f"Unexpected output filename: {name}")
            patient_id = match.group(1).lower()
            observed_patient_splits[patient_id].add(split)

    leaking = {
        patient: sorted(splits)
        for patient, splits in observed_patient_splits.items()
        if len(splits) != 1
    }
    if leaking:
        raise RuntimeError(f"Patient leakage detected: {list(leaking.items())[:10]}")

    for patient_id, split in patient_split.items():
        observed = observed_patient_splits.get(patient_id, set())
        if observed != {split}:
            raise RuntimeError(
                f"Patient split verification failed for {patient_id}: "
                f"planned={split}, observed={sorted(observed)}"
            )


def summarize(
    source_root: Path,
    output_root: Path,
    pairs: Sequence[SourcePair],
    records: Sequence[PatchRecord],
    patient_split: Dict[str, str],
    seed: int,
    edge_width: int,
    saved_counts: Counter | None,
) -> Dict[str, object]:
    category_counts = Counter(record.category for record in records)
    source_patient_ids = {pair.patient_id for pair in pairs}
    kept_patient_ids = set(patient_split)

    patient_counts = Counter(patient_split.values())
    patch_counts = Counter(
        record.final_split
        for record in records
        if record.category == "class3_keep"
    )

    return {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "source_size": SOURCE_SIZE,
        "patch_size": PATCH_SIZE,
        "stride": STRIDE,
        "x_positions": list(PATCH_POSITIONS),
        "y_positions": list(PATCH_POSITIONS),
        "patches_per_source_image": PATCHES_PER_IMAGE,
        "n_source_images": len(pairs),
        "n_source_patients": len(source_patient_ids),
        "n_patients_with_kept_patches": len(kept_patient_ids),
        "n_patients_removed_entirely": len(source_patient_ids - kept_patient_ids),
        "candidate_patches": len(records),
        "category_counts": dict(category_counts),
        "filter_definition": {
            "class1_empty": "foreground pixel count == 0",
            "class2_edge": (
                f"foreground exists and touches the outer {edge_width} pixel(s) "
                "of any patch side"
            ),
            "class3_keep": "foreground exists and does not touch the defined border",
            "foreground": "mask pixel > 0",
        },
        "patient_split": {
            "seed": seed,
            "target_ratio": {"train": 0.8, "val": 0.1, "test": 0.1},
            "patient_counts": {
                split: int(patient_counts.get(split, 0)) for split in FINAL_SPLITS
            },
            "kept_patch_counts": {
                split: int(patch_counts.get(split, 0)) for split in FINAL_SPLITS
            },
        },
        "saved_patch_counts": (
            {
                split: int(saved_counts.get(split, 0)) for split in FINAL_SPLITS
            }
            if saved_counts is not None
            else None
        ),
        "naming": (
            "pPATIENT_IMAGE_PATCH.png; PATCH remains 0001-0009 row-major and is not re-indexed after filtering"
        ),
        "old_size512_split_membership": "ignored; all source pairs are pooled before the new patient-level split",
    }


def print_summary(summary: Dict[str, object]) -> None:
    print("\n=== Size224 filtered dataset plan ===")
    print(f"Source images: {summary['n_source_images']}")
    print(f"Source patients: {summary['n_source_patients']}")
    print(f"Candidate patches: {summary['candidate_patches']}")
    print(f"Category counts: {summary['category_counts']}")
    print(
        "Patients with kept patches: "
        f"{summary['n_patients_with_kept_patches']} "
        f"(removed entirely: {summary['n_patients_removed_entirely']})"
    )
    split = summary["patient_split"]
    print(f"Patient counts: {split['patient_counts']}")
    print(f"Kept patch counts: {split['kept_patch_counts']}")


def main() -> int:
    args = parse_args()
    if args.edge_width < 1:
        raise ValueError("--edge-width must be >= 1")

    source_root = args.source.expanduser().resolve()
    output_root = (
        args.output.expanduser().resolve()
        if args.output is not None
        else source_root.parent / "Size_224_filtered"
    )

    if source_root == output_root:
        raise ValueError("Source and output directories must be different")

    print(f"Source root: {source_root}")
    print(f"Output root: {output_root}")
    print(f"Patch positions: {PATCH_POSITIONS} x {PATCH_POSITIONS}")
    print(f"Patient split seed: {args.seed}")
    print(f"Edge width: {args.edge_width} pixel(s)")

    pairs = collect_source_pairs(source_root)
    records = analyze_patches(pairs, edge_width=args.edge_width)
    patient_split = assign_patient_splits(records, seed=args.seed)

    planned_summary = summarize(
        source_root=source_root,
        output_root=output_root,
        pairs=pairs,
        records=records,
        patient_split=patient_split,
        seed=args.seed,
        edge_width=args.edge_width,
        saved_counts=None,
    )
    print_summary(planned_summary)

    if args.dry_run:
        print("\nDry run complete; no files were written.")
        return 0

    prepare_output(output_root, overwrite=args.overwrite)
    saved_counts = save_kept_patches(output_root, pairs, records)
    verify_output(output_root, records, patient_split)

    write_patch_audit(output_root / "patch_filter_audit.csv", records)
    write_patient_split_csv(
        output_root / "patient_split.csv",
        pairs=pairs,
        records=records,
        patient_split=patient_split,
    )

    final_summary = summarize(
        source_root=source_root,
        output_root=output_root,
        pairs=pairs,
        records=records,
        patient_split=patient_split,
        seed=args.seed,
        edge_width=args.edge_width,
        saved_counts=saved_counts,
    )
    with (output_root / "conversion_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(final_summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print_summary(final_summary)
    print("\nOutput verification passed: image/mask pairs match and no patient leaks across splits.")
    print(f"Audit CSV: {output_root / 'patch_filter_audit.csv'}")
    print(f"Patient split CSV: {output_root / 'patient_split.csv'}")
    print(f"Summary JSON: {output_root / 'conversion_summary.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)