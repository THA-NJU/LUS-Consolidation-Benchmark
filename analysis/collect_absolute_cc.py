#!/usr/bin/env python3
"""
Compute per-model absolute connected-component count errors from *_cases.csv.

Expected use:
    cd <repository-root>
    python analysis/collect_absolute_cc.py \
        --root Evaluation \
        --output absolute_cc_summary.csv

The input files are:
    train_cases.csv
    val_cases.csv
    test_cases.csv

For every evaluated image, the script uses the first available definition:
    1. abs_cc_delta
    2. abs(cc_delta)
    3. abs(gt_cc - pred_cc)

If cc_delta is absent, the script falls back to:
    abs_cc_delta = abs(gt_cc - pred_cc)

The output contains one row per model and image size for all 36 models in the
Evaluation tree. Mean and standard deviation use all finite per-image values
and population standard deviation (ddof=0), matching the evaluation JSON files.

Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SPLITS: Tuple[Tuple[str, str], ...] = (
    ("train", "Train"),
    ("val", "Validation"),
    ("test", "Test"),
)
DEFAULT_SIZES: Tuple[int, ...] = (512, 224)


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    directory_aliases: Tuple[str, ...]


# This is the frozen 36-model inventory. Keep this list aligned with
# collect_evaluation_summaries.py. Each model is
# emitted with Size 512 first and Size 224 second by DEFAULT_SIZES.
MODEL_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec("Unet", ("monai_unet", "unet")),
    ModelSpec("Vnet", ("monai_vnet", "vnet")),
    ModelSpec(
        "Attention U-net",
        ("monai_attention_unet", "attention_unet", "attention_u_net"),
    ),
    ModelSpec(
        "Unet++",
        ("monai_unetplusplus", "unetplusplus", "unet_plus_plus", "unetpp"),
    ),
    ModelSpec("MedNeXt", ("mednext",)),
    ModelSpec("nnU-net", ("nnunet_2d", "nnunet2d", "nnunet")),
    ModelSpec("AAUnet", ("aau_net", "aaunet", "aau_net_2d")),
    ModelSpec("FPN-ResNet34", ("fpn_resnet34", "resnet34_fpn")),
    ModelSpec(
        "DeepLabV3+-ResNet34",
        ("deeplabv3plus_resnet34", "resnet34_dlv3", "resnet_dlv3"),
    ),
    ModelSpec("Swin-Unet", ("swinunet", "swin_unet")),
    ModelSpec("nnFormer", ("nnformer_2d", "nnformer2d", "nnformer")),
    ModelSpec(
        "PVTv2-b2-EMCAD",
        ("pvtb2_emcad", "pvtv2_b2_emcad", "pvt2_b2_emcad"),
    ),
    ModelSpec("SEPNet", ("sepnet", "sep_net")),
    ModelSpec("NuLite", ("nulite", "nu_lite")),
    ModelSpec("Mamba-Unet", ("mamba_unet", "mambaunet")),
    ModelSpec("VM-Unet V2", ("vm_unet_v2", "vmunetv2", "vm_unetv2")),
    ModelSpec("nn-Mamba", ("nnmamba_2d", "nnmamba2d", "nn_mamba")),
    ModelSpec("Swin Umamba", ("swin_umamba", "swinumamba")),
    ModelSpec("Rolling-UNet", ("rolling_unet", "rollingunet")),
    ModelSpec("U-KAN", ("ukan", "u_kan")),
    ModelSpec("xLSTM-Unet", ("xlstm_unet_bot", "xlstm_unet", "xlstmunet")),
    ModelSpec("S2DENet", ("s2denet", "s2de_net")),
    ModelSpec("U-RWKV", ("u_rwkv", "urwkv")),
    ModelSpec("RWKV-Unet", ("rwkv_unet", "rwkvunet")),
    ModelSpec("SAM2", ("sam2", "sam_2")),
    ModelSpec("MedSAM", ("medsam", "med_sam")),
    ModelSpec("SAMUS", ("samus", "sam_us")),
    ModelSpec("SAM3", ("sam3", "sam_3")),
    ModelSpec("USFM transfer", ("usfm_transfer", "usfm")),
    ModelSpec("YOLO11s-seg", ("yolo11s_seg", "yolo11s-seg")),
    ModelSpec("YOLO11m-seg", ("yolo11m_seg", "yolo11m-seg")),
    ModelSpec("YOLO11l-seg", ("yolo11l_seg", "yolo11l-seg")),
    ModelSpec("YOLO26s-sem", ("yolo26s_sem", "yolo26s-sem")),
    ModelSpec("YOLO26m-sem", ("yolo26m_sem", "yolo26m-sem")),
    ModelSpec("YOLO26l-sem", ("yolo26l_sem", "yolo26l-sem")),
    ModelSpec("SegFormer-B2", ("segformer_b2", "segformer")),
)

EXPECTED_MODEL_COUNT = 36
EXCLUDED_RESULT_DIRS = {"_torch_extensions", "worker_logs"}

if len(MODEL_SPECS) != EXPECTED_MODEL_COUNT:
    raise RuntimeError(
        f"Internal model inventory error: expected {EXPECTED_MODEL_COUNT}, "
        f"found {len(MODEL_SPECS)}"
    )


OUTPUT_COLUMNS: Tuple[str, ...] = (
    "Model",
    "Size",
    "Train-Mean-|Delta-CC|",
    "Train-Std-|Delta-CC|",
    "Train-Valid-Count",
    "Validation-Mean-|Delta-CC|",
    "Validation-Std-|Delta-CC|",
    "Validation-Valid-Count",
    "Test-Mean-|Delta-CC|",
    "Test-Std-|Delta-CC|",
    "Test-Valid-Count",
)


def normalized(text: object) -> str:
    return str(text or "").strip().casefold().replace("-", "_")


def parse_finite(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def size_is_in_path(path: Path, root: Path, size: int) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    return str(size) in parts or f"size_{size}" in {
        normalized(part) for part in parts
    }


def alias_is_in_path(path: Path, root: Path, aliases: Sequence[str]) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    normalized_parts = {normalized(part) for part in parts}
    return any(normalized(alias) in normalized_parts for alias in aliases)


def discover_case_files(root: Path) -> Mapping[str, List[Path]]:
    found: Dict[str, List[Path]] = {}
    for split, _ in SPLITS:
        found[split] = sorted(
            path.resolve()
            for path in root.rglob(f"{split}_cases.csv")
            if path.is_file()
            and not any(
                normalized(part) in EXCLUDED_RESULT_DIRS
                for part in path.relative_to(root).parts
            )
        )
    return found


def select_case_file(
    candidates: Sequence[Path],
    root: Path,
    spec: ModelSpec,
    size: int,
    split: str,
) -> Optional[Path]:
    matches = [
        path
        for path in candidates
        if size_is_in_path(path, root, size)
        and alias_is_in_path(path, root, spec.directory_aliases)
    ]
    if not matches:
        return None
    if len(matches) > 1:
        choices = "\n    ".join(str(path) for path in matches)
        raise RuntimeError(
            f"Multiple {split}_cases.csv files match "
            f"{spec.display_name} size {size}:\n    {choices}\n"
            "Remove duplicate result directories or use a narrower --root."
        )
    return matches[0]


def row_abs_cc_delta(row: Mapping[str, str], path: Path, row_number: int) -> float:
    abs_cc_delta = parse_finite(row.get("abs_cc_delta"))
    if abs_cc_delta is not None:
        return abs(abs_cc_delta)

    cc_delta = parse_finite(row.get("cc_delta"))
    if cc_delta is not None:
        return abs(cc_delta)

    gt_cc = parse_finite(row.get("gt_cc"))
    pred_cc = parse_finite(row.get("pred_cc"))
    if gt_cc is not None and pred_cc is not None:
        return abs(gt_cc - pred_cc)

    raise ValueError(
        f"{path}: row {row_number} has neither a valid abs_cc_delta, a valid "
        "cc_delta, nor valid gt_cc/pred_cc values"
    )


def read_abs_cc_values(path: Path) -> List[float]:
    values: List[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV has no header")

        normalized_to_original = {
            normalized(name): name for name in reader.fieldnames
        }
        has_abs_delta = "abs_cc_delta" in normalized_to_original
        has_delta = "cc_delta" in normalized_to_original
        has_counts = (
            "gt_cc" in normalized_to_original
            and "pred_cc" in normalized_to_original
        )
        if not has_abs_delta and not has_delta and not has_counts:
            raise ValueError(
                f"{path}: expected abs_cc_delta, cc_delta, or both gt_cc and "
                f"pred_cc; "
                f"found columns: {reader.fieldnames}"
            )

        for row_number, raw_row in enumerate(reader, start=2):
            row = {
                normalized(key): value
                for key, value in raw_row.items()
                if key is not None
            }
            values.append(row_abs_cc_delta(row, path, row_number))
    return values


def mean_std_count(values: Sequence[float]) -> Tuple[float, float, int]:
    if not values:
        return math.nan, math.nan, 0
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    return float(mean), float(std), len(values)


def format_number(value: float, digits: int) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def build_rows(
    root: Path,
    sizes: Sequence[int],
    missing_policy: str,
    digits: int,
) -> Tuple[List[Dict[str, object]], List[str]]:
    discovered = discover_case_files(root)
    output_rows: List[Dict[str, object]] = []
    warnings: List[str] = []

    for spec in MODEL_SPECS:
        for size in sizes:
            row: Dict[str, object] = {"Model": spec.display_name, "Size": size}
            found_any = False

            for split, split_label in SPLITS:
                path = select_case_file(
                    discovered[split],
                    root,
                    spec,
                    size,
                    split,
                )
                if path is None:
                    message = (
                        f"Missing {split}_cases.csv for "
                        f"{spec.display_name} size {size}"
                    )
                    if missing_policy == "error":
                        raise FileNotFoundError(message)
                    warnings.append(message)
                    mean, std, count = math.nan, math.nan, 0
                else:
                    values = read_abs_cc_values(path)
                    mean, std, count = mean_std_count(values)
                    found_any = True

                row[f"{split_label}-Mean-|Delta-CC|"] = format_number(
                    mean, digits
                )
                row[f"{split_label}-Std-|Delta-CC|"] = format_number(std, digits)
                row[f"{split_label}-Valid-Count"] = count if count else ""

            if missing_policy != "skip" or found_any:
                output_rows.append(row)

    return output_rows, warnings


def write_output(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_sizes(values: Sequence[str]) -> Tuple[int, ...]:
    sizes: List[int] = []
    for value in values:
        size = int(value)
        if size <= 0:
            raise argparse.ArgumentTypeError("sizes must be positive integers")
        if size not in sizes:
            sizes.append(size)
    return tuple(sizes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute mean/std of per-image absolute connected-component "
            "count error from train/val/test_cases.csv."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="APRIL/result root to scan recursively (default: current directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("absolute_cc_summary.csv"),
        help="output CSV path (default: absolute_cc_summary.csv)",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=[str(size) for size in DEFAULT_SIZES],
        metavar="N",
        help="size order within each model (default: 512 224)",
    )
    parser.add_argument(
        "--missing-policy",
        choices=("blank", "skip", "error"),
        default="blank",
        help=(
            "blank: keep missing rows/cells; skip: omit models with no split "
            "files; error: stop at first missing file (default: blank)"
        ),
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=6,
        help="decimal places for mean/std (default: 6)",
    )
    args = parser.parse_args()
    args.sizes = parse_sizes(args.sizes)
    if args.digits < 0:
        parser.error("--digits must be >= 0")
    return args


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: root directory does not exist: {root}", file=sys.stderr)
        return 2

    try:
        rows, warnings = build_rows(
            root=root,
            sizes=args.sizes,
            missing_policy=args.missing_policy,
            digits=args.digits,
        )
        output = args.output.expanduser()
        if not output.is_absolute():
            output = (Path.cwd() / output).resolve()
        write_output(output, rows)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(rows)} rows to: {output}")
    if warnings:
        print(
            f"Warning: {len(warnings)} expected split files were not found. "
            "Their output cells were left blank.",
            file=sys.stderr,
        )
        for warning in warnings[:20]:
            print(f"  - {warning}", file=sys.stderr)
        if len(warnings) > 20:
            print(
                f"  ... and {len(warnings) - 20} more",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
