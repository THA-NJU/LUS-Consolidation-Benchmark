#!/usr/bin/env python3
"""
Aggregate difficult test cases across segmentation models.

For every image and image size, this script reads all discovered
``test_cases.csv`` files, counts how many models have ``dice < threshold``,
and computes the population mean/std of Dice across the available models.

Typical use
-----------
    python analysis/collect_hard_test_cases.py \
        --root ./Evaluation \
        --output ./Evaluation/hard_test_cases.csv

The output is one CSV. Results from Size 512 and Size 224 are kept separate by
the ``size`` column even when their case names are identical.

Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


CASE_COLUMN_ALIASES: Tuple[str, ...] = (
    "case_id",
    "filename",
    "image_name",
    "name",
    "case_name",
    "file_name",
    "image",
    "path",
    "image_path",
)
DICE_COLUMN_ALIASES: Tuple[str, ...] = (
    "dice",
    "dice_score",
    "dice_coefficient",
)
SIZE_COLUMN_ALIASES: Tuple[str, ...] = (
    "size",
    "image_size",
    "input_size",
    "resolution",
)
EXCLUDED_DIRS = {"torch_extensions", "worker_logs", "pycache"}
DEFAULT_SIZE_ORDER = {512: 0, 224: 1}


@dataclass(frozen=True)
class ModelCases:
    model: str
    size: int
    path: Path
    dice_by_case: Mapping[str, float]


def normalize_column(value: object) -> str:
    text = str(value or "").strip().casefold()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def choose_column(
    fieldnames: Sequence[str], aliases: Sequence[str]
) -> Optional[str]:
    normalized = {normalize_column(name): name for name in fieldnames}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def parse_finite_float(value: object) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def parse_size_value(value: object) -> Optional[int]:
    text = str(value or "").strip().casefold()
    if not text:
        return None
    matches = re.findall(r"(?<!\d)(224|512)(?!\d)", text)
    if len(set(matches)) == 1:
        return int(matches[0])
    return None


def infer_size_from_path(path: Path, root: Path) -> Optional[int]:
    try:
        # Include root.name so ``--root Evaluation/224`` and roots such as
        # ``usfm_finetune_224`` still carry usable size evidence.
        parts = (root.name, *path.relative_to(root).parts[:-1])
    except ValueError:
        parts = path.parts[:-1]
    found: List[int] = []
    for part in parts:
        size = parse_size_value(part)
        if size is not None:
            found.append(size)
    unique = set(found)
    if len(unique) == 1:
        return found[-1]
    return None


def normalize_case_id(value: object) -> str:
    """Keep the evaluator's case identity, but normalize path separators."""
    case_id = str(value or "").strip().replace("\\", "/")
    while case_id.startswith("./"):
        case_id = case_id[2:]
    return case_id


def display_filename(case_id: str) -> str:
    return case_id.rsplit("/", 1)[-1]


def discover_test_case_files(root: Path) -> List[Path]:
    return sorted(
        path.resolve()
        for path in root.rglob("test_cases.csv")
        if path.is_file()
        and not any(
            normalize_column(part) in EXCLUDED_DIRS
            for part in path.relative_to(root).parts
        )
    )


def infer_model_name(path: Path) -> str:
    # Under the benchmark convention, test_cases.csv is stored directly in the
    # model result directory. If the direct parent is only ``224``/``512``, use
    # its parent. A user can override this with --model-parent-level.
    direct = normalize_column(path.parent.name)
    if re.fullmatch(r"(?:size_)?(?:224|512)", direct):
        return path.parent.parent.name
    return path.parent.name


def ancestor_name(path: Path, level: int) -> str:
    parent = path.parent
    for _ in range(level - 1):
        parent = parent.parent
    return parent.name


def inspect_csv_columns(path: Path) -> Tuple[List[str], Optional[str], Optional[str], Optional[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV has no header")
        fields = list(reader.fieldnames)
    return (
        fields,
        choose_column(fields, CASE_COLUMN_ALIASES),
        choose_column(fields, DICE_COLUMN_ALIASES),
        choose_column(fields, SIZE_COLUMN_ALIASES),
    )


def read_model_cases(
    path: Path,
    root: Path,
    forced_size: Optional[int],
    default_size: Optional[int],
    model_parent_level: int,
) -> ModelCases:
    fields, case_col, dice_col, size_col = inspect_csv_columns(path)
    if case_col is None:
        raise ValueError(
            f"{path}: no case-name column found; columns={fields}; expected one "
            f"of {CASE_COLUMN_ALIASES}"
        )
    if dice_col is None:
        raise ValueError(
            f"{path}: no Dice column found; columns={fields}; expected one of "
            f"{DICE_COLUMN_ALIASES}"
        )

    path_size = infer_size_from_path(path, root)
    model = (
        infer_model_name(path)
        if model_parent_level == 1
        else ancestor_name(path, model_parent_level)
    )
    values: Dict[str, float] = {}
    csv_sizes: set[int] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            case_id = normalize_case_id(row.get(case_col))
            if not case_id:
                raise ValueError(f"{path}: row {row_number} has an empty case name")

            dice = parse_finite_float(row.get(dice_col))
            if dice is None:
                raise ValueError(
                    f"{path}: row {row_number} has invalid Dice "
                    f"{row.get(dice_col)!r}"
                )
            if not 0.0 <= dice <= 1.0:
                raise ValueError(
                    f"{path}: row {row_number} Dice={dice} is outside [0, 1]"
                )
            if case_id in values:
                raise ValueError(
                    f"{path}: duplicate case identifier {case_id!r}; use a unique "
                    "case_id column (including the tile index for tiled data)"
                )
            values[case_id] = dice

            if size_col is not None:
                row_size = parse_size_value(row.get(size_col))
                if row_size is not None:
                    csv_sizes.add(row_size)

    if not values:
        raise ValueError(f"{path}: contains no test cases")
    if len(csv_sizes) > 1:
        raise ValueError(f"{path}: contains multiple image sizes: {sorted(csv_sizes)}")
    csv_size = next(iter(csv_sizes), None)

    direct_evidence = [size for size in (csv_size, path_size) if size is not None]
    if forced_size is not None and any(
        size != forced_size for size in direct_evidence
    ):
        raise ValueError(
            f"{path}: discovered size evidence {direct_evidence}, which conflicts "
            f"with selected --size {forced_size}"
        )
    candidates = [
        size for size in (forced_size, csv_size, path_size, default_size)
        if size is not None
    ]
    if forced_size is not None:
        size = forced_size
    elif candidates and len(set(candidates)) == 1:
        size = candidates[0]
    elif not candidates:
        raise ValueError(
            f"{path}: cannot infer 224/512 size from path or CSV; pass "
            "--default-size 224 (or 512)"
        )
    else:
        raise ValueError(
            f"{path}: conflicting size evidence {candidates}; use a narrower "
            "--root or an explicit --size"
        )

    return ModelCases(model=model, size=size, path=path, dice_by_case=values)


def validate_models(models: Sequence[ModelCases]) -> None:
    seen: Dict[Tuple[int, str], Path] = {}
    for item in models:
        key = (item.size, normalize_column(item.model))
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(
                f"Duplicate result for model={item.model!r}, size={item.size}:\n"
                f"  {previous}\n  {item.path}\n"
                "Use a narrower --root or move obsolete/duplicate result trees."
            )
        seen[key] = item.path


def aggregate(
    models: Sequence[ModelCases], threshold: float, only_hard: bool
) -> List[Dict[str, object]]:
    model_names_by_size: Dict[int, List[str]] = defaultdict(list)
    observations: Dict[Tuple[int, str], List[Tuple[str, float]]] = defaultdict(list)

    for item in models:
        model_names_by_size[item.size].append(item.model)
        for case_id, dice in item.dice_by_case.items():
            observations[(item.size, case_id)].append((item.model, dice))

    rows: List[Dict[str, object]] = []
    for (size, case_id), case_values in observations.items():
        case_values = sorted(case_values, key=lambda pair: pair[0].casefold())
        dice_values = [value for _, value in case_values]
        low_models = [model for model, value in case_values if value < threshold]
        if only_hard and not low_models:
            continue

        evaluated_names = {model for model, _ in case_values}
        all_names = set(model_names_by_size[size])
        missing_models = sorted(all_names - evaluated_names, key=str.casefold)
        rows.append(
            {
                "size": size,
                "case_id": case_id,
                "filename": display_filename(case_id),
                "dice_below_threshold_count": len(low_models),
                "models_evaluated": len(dice_values),
                "total_models_for_size": len(all_names),
                "mean_dice": statistics.fmean(dice_values),
                "dice_std": statistics.pstdev(dice_values),
                "low_dice_models": ";".join(low_models),
                "missing_models": ";".join(missing_models),
            }
        )

    rows.sort(
        key=lambda row: (
            -int(row["dice_below_threshold_count"]),
            float(row["mean_dice"]),
            DEFAULT_SIZE_ORDER.get(int(row["size"]), 99),
            str(row["case_id"]).casefold(),
        )
    )
    return rows


def write_output(path: Path, rows: Sequence[Mapping[str, object]], digits: int) -> None:
    columns = (
        "size",
        "case_id",
        "filename",
        "dice_below_threshold_count",
        "models_evaluated",
        "total_models_for_size",
        "mean_dice",
        "dice_std",
        "low_dice_models",
        "missing_models",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for raw in rows:
            row = dict(raw)
            row["mean_dice"] = f"{float(row['mean_dice']):.{digits}f}"
            row["dice_std"] = f"{float(row['dice_std']):.{digits}f}"
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count per-image Dice failures across model test_cases.csv files "
            "and compute cross-model Dice mean/population std."
        )
    )
    parser.add_argument(
        "--root", type=Path, default=Path("Evaluation"),
        help="Result root scanned recursively (default: Evaluation)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("hard_test_cases.csv"),
        help="Output CSV path (default: hard_test_cases.csv)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="A case is difficult for one model when Dice is strictly below this value (default: 0.3)",
    )
    parser.add_argument(
        "--size", type=int, choices=(224, 512), default=None,
        help="Only aggregate this image size; unresolved files are assigned this size",
    )
    parser.add_argument(
        "--default-size", type=int, choices=(224, 512), default=None,
        help="Fallback size only when neither path nor CSV contains 224/512",
    )
    parser.add_argument(
        "--model-parent-level", type=int, default=1,
        help="Model directory level above test_cases.csv: 1=direct parent (default)",
    )
    parser.add_argument(
        "--digits", type=int, default=6,
        help="Decimal places for mean_dice and dice_std (default: 6)",
    )
    parser.add_argument(
        "--include-non-hard", action="store_true",
        help="Also output images for which no model has Dice below threshold",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: result root does not exist or is not a directory: {root}", file=sys.stderr)
        return 2
    if not 0.0 <= args.threshold <= 1.0:
        print("ERROR: --threshold must be within [0, 1]", file=sys.stderr)
        return 2
    if args.model_parent_level < 1:
        print("ERROR: --model-parent-level must be at least 1", file=sys.stderr)
        return 2
    if args.digits < 0:
        print("ERROR: --digits must be non-negative", file=sys.stderr)
        return 2

    paths = discover_test_case_files(root)
    if args.size is not None:
        paths = [
            path for path in paths
            if infer_size_from_path(path, root) in (None, args.size)
        ]
    if not paths:
        print(f"ERROR: no test_cases.csv found below {root}", file=sys.stderr)
        return 2

    try:
        models = [
            read_model_cases(
                path=path,
                root=root,
                forced_size=args.size,
                default_size=args.default_size,
                model_parent_level=args.model_parent_level,
            )
            for path in paths
        ]
        validate_models(models)
        rows = aggregate(
            models=models,
            threshold=args.threshold,
            only_hard=not args.include_non_hard,
        )
        write_output(output, rows, args.digits)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    counts_by_size: Dict[int, int] = defaultdict(int)
    cases_by_size: Dict[int, set[str]] = defaultdict(set)
    hard_by_size: Dict[int, int] = defaultdict(int)
    for item in models:
        counts_by_size[item.size] += 1
        cases_by_size[item.size].update(item.dice_by_case)
    for row in rows:
        hard_by_size[int(row["size"])] += 1

    print(f"Discovered {len(models)} model test files below: {root}")
    for size in sorted(counts_by_size, key=lambda x: DEFAULT_SIZE_ORDER.get(x, 99)):
        print(
            f"  Size {size}: models={counts_by_size[size]}, "
            f"unique_cases={len(cases_by_size[size])}, "
            f"rows_written={hard_by_size[size]}"
        )
    print(f"Difficult criterion: dice < {args.threshold:g}")
    print(f"Population Dice std: ddof=0")
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
