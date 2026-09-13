#!/usr/bin/env python3
"""Recursively collect segmentation evaluation summaries into split CSV files.

Typical use
-----------
    cd <repository-root>
    python analysis/collect_evaluation_summaries.py \
        --evaluation-root Evaluation \
        --output-dir Evaluation \
        --absolute-cc absolute_cc_summary.csv

The script discovers all models dynamically, canonicalizes known model names,
and writes them in the benchmark's prescribed model order.  Within each model,
Size 512 is written before Size 224.
It accepts these layouts (and combinations of them):

    Evaluation/<family>/<size>/<model>/train_summary.json
    Evaluation/<family>/<size>/<model>/train.json
    Evaluation/<family>/<size>/<model>/train/summary.json

Both flat and nested JSON are supported, for example either
``dice_mean`` or ``{"dice": {"mean": ...}}``.  Three UTF-8-with-BOM files
are written: ``train.csv``, ``val.csv`` and ``test.csv``.

Older summaries may not contain per-image absolute connected-component error.
For those rows, Mean/Std-|Delta-CC| are filled from absolute_cc_summary.csv when
available; otherwise those cells remain blank.  Signed Delta-CC and absolute
|Delta-CC| are intentionally kept as separate columns.

Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SPLITS = ("train", "val", "test")

OUTPUT_COLUMNS = (
    "Model",
    "Size",
    "Mean-dice",
    "Std-dice",
    "Mean-IoU",
    "Std-IoU",
    "Mean-Recall",
    "Std-Recall",
    "Mean-Precision",
    "Std-Precision",
    "HD-count-delta",
    "Mean-HD95",
    "Std-HD95",
    "Mean-HD",
    "Std-HD",
    "Mean-Delta-CC",
    "Std-Delta-CC",
    "Mean-Abs-Delta-CC",
    "Std-Abs-Delta-CC",
    "Mean-Efficiency-ms/image",
    "Std-Efficiency-ms/image",
)


def normalize_key(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(
        r"\|\s*(?:delta|Δ|δ)[\s_-]*cc\s*\|",
        " abs_delta_cc ",
        text,
        flags=re.IGNORECASE,
    )
    text = text.casefold()
    text = text.replace("δ", "delta").replace("Δ", "delta")
    text = text.replace("|", "_")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def parse_number(value: object) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip().replace(",", "")
        if not text or text.casefold() in {
            "nan", "none", "null", "na", "n/a", "-", "--"
        }:
            return None
        if text.endswith("%"):
            text = text[:-1].strip()
        try:
            number = float(text)
        except ValueError:
            return None
    return number if math.isfinite(number) else None


def flatten_numeric_json(
    value: object,
    prefix: Tuple[str, ...] = (),
) -> Dict[str, float]:
    """Flatten nested numeric leaves while preserving their full key paths."""
    output: Dict[str, float] = {}
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = normalize_key(raw_key)
            if key:
                output.update(flatten_numeric_json(child, prefix + (key,)))
    elif isinstance(value, list):
        # A one-element summary list is a common legacy serialization.
        if len(value) == 1:
            output.update(flatten_numeric_json(value[0], prefix))
    else:
        number = parse_number(value)
        if number is not None and prefix:
            output["_".join(prefix)] = number
    return output


def nested_string(payload: Mapping[str, Any], aliases: Iterable[str]) -> str:
    wanted = {normalize_key(alias) for alias in aliases}
    for key, value in payload.items():
        if normalize_key(key) in wanted and isinstance(value, str):
            return value.strip()
    for container_name in ("metadata", "meta", "config"):
        container = payload.get(container_name)
        if isinstance(container, Mapping):
            for key, value in container.items():
                if normalize_key(key) in wanted and isinstance(value, str):
                    return value.strip()
    return ""


def nested_int(payload: Mapping[str, Any], aliases: Iterable[str]) -> Optional[int]:
    wanted = {normalize_key(alias) for alias in aliases}
    for key, value in payload.items():
        if normalize_key(key) in wanted:
            number = parse_number(value)
            if number is not None and number.is_integer():
                return int(number)
    return None


def canonical_split(value: object) -> Optional[str]:
    key = normalize_key(value)
    if key == "validation":
        return "val"
    return key if key in SPLITS else None


def split_from_path(path: Path) -> Optional[str]:
    stem = normalize_key(path.stem)
    match = re.fullmatch(r"(train|val|validation|test)(?:_summary)?", stem)
    if match:
        return canonical_split(match.group(1))
    if stem == "summary":
        for part in reversed(path.parent.parts):
            split = canonical_split(part)
            if split is not None:
                return split
    return None


def read_summary(path: Path, path_split: str) -> Tuple[Dict[str, float], str, str, Optional[int]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if isinstance(payload, list) and len(payload) == 1:
        payload = payload[0]
    if not isinstance(payload, Mapping):
        raise ValueError("expected a JSON object or a one-object JSON list")

    recorded_split = canonical_split(nested_string(payload, ("split", "phase")))
    if recorded_split is not None and recorded_split != path_split:
        raise ValueError(
            f"path says split={path_split!r}, JSON says split={recorded_split!r}"
        )
    model = nested_string(payload, ("model", "model_name", "architecture"))
    size = nested_int(payload, ("size", "image_size", "input_size"))
    return flatten_numeric_json(payload), recorded_split or path_split, model, size


def size_from_part(part: str) -> Optional[int]:
    key = normalize_key(part)
    match = re.fullmatch(
        r"(?:size_?)?(\d{2,4})(?:x\1)?(?:_filtered)?",
        key,
    )
    if not match:
        return None
    size = int(match.group(1))
    return size if 16 <= size <= 8192 else None


GENERIC_DIRS = {
    "evaluation", "evaluations", "eval", "result", "results", "output",
    "outputs", "summary", "summaries", "checkpoint", "checkpoints",
    "weight", "weights", "pth", "runs", "run", "final", "best",
    "train", "val", "validation", "test",
}


def infer_model_size(path: Path, root: Path) -> Tuple[str, int]:
    try:
        parts = list(path.relative_to(root).parts[:-1])
    except ValueError:
        parts = list(path.parts[:-1])

    # A trailing split directory belongs to the file layout, not the model name.
    while parts and canonical_split(parts[-1]) is not None:
        parts.pop()

    sized = [(index, size_from_part(part)) for index, part in enumerate(parts)]
    sized = [(index, size) for index, size in sized if size is not None]
    if not sized:
        raise ValueError("cannot infer image size from the directory path")
    size_index, size = sized[-1]
    assert size is not None

    def meaningful(part: str) -> bool:
        key = normalize_key(part)
        return bool(key) and key not in GENERIC_DIRS and size_from_part(part) is None

    after = [part for part in parts[size_index + 1 :] if meaningful(part)]
    before = [part for part in parts[:size_index] if meaningful(part)]
    if after:
        model = after[0]
    elif before:
        model = before[-1]
    else:
        raise ValueError("cannot infer model name from the directory path")
    return model, size


def alias_candidates(alias: str) -> Tuple[str, ...]:
    base = normalize_key(alias)
    choices = [base]
    for split in SPLITS:
        choices.extend((f"{split}_{base}", f"{split}_metrics_{base}"))
    choices.append(f"metrics_{base}")
    return tuple(choices)


def find_value(values: Mapping[str, float], aliases: Iterable[str]) -> Optional[float]:
    """Prefer exact aliases, then accept an unambiguous nested suffix match."""
    candidates: List[str] = []
    for alias in aliases:
        candidates.extend(alias_candidates(alias))
    for candidate in candidates:
        if candidate in values:
            return values[candidate]

    matches: List[float] = []
    normalized_aliases = tuple(normalize_key(alias) for alias in aliases)
    for key, value in values.items():
        if any(key.endswith("_" + alias) for alias in normalized_aliases):
            matches.append(value)
    if not matches:
        return None
    first = matches[0]
    if all(math.isclose(item, first, rel_tol=1e-12, abs_tol=1e-12) for item in matches[1:]):
        return first
    return None


METRIC_ALIASES: Mapping[Tuple[str, str], Tuple[str, ...]] = {
    ("dice", "mean"): ("dice_mean", "mean_dice", "dice_avg", "average_dice"),
    ("dice", "std"): ("dice_std", "std_dice", "dice_stdev"),
    ("iou", "mean"): ("iou_mean", "mean_iou", "jaccard_mean", "mean_jaccard"),
    ("iou", "std"): ("iou_std", "std_iou", "jaccard_std", "std_jaccard"),
    ("hd", "mean"): ("hd_mean", "mean_hd", "hausdorff_mean", "mean_hausdorff"),
    ("hd", "std"): ("hd_std", "std_hd", "hausdorff_std", "std_hausdorff"),
    ("hd95", "mean"): ("hd95_mean", "mean_hd95", "hausdorff95_mean"),
    ("hd95", "std"): ("hd95_std", "std_hd95", "hausdorff95_std"),
    ("precision", "mean"): ("precision_mean", "mean_precision", "ppv_mean"),
    ("precision", "std"): ("precision_std", "std_precision", "ppv_std"),
    ("recall", "mean"): ("recall_mean", "mean_recall", "sensitivity_mean"),
    ("recall", "std"): ("recall_std", "std_recall", "sensitivity_std"),
    ("delta_cc", "mean"): (
        "cc_delta_mean", "mean_cc_delta", "delta_cc_mean", "mean_delta_cc",
        "connected_component_delta_mean",
    ),
    ("delta_cc", "std"): (
        "cc_delta_std", "std_cc_delta", "delta_cc_std", "std_delta_cc",
        "connected_component_delta_std",
    ),
    ("abs_delta_cc", "mean"): (
        "abs_cc_delta_mean", "mean_abs_cc_delta", "abs_delta_cc_mean",
        "mean_abs_delta_cc", "absolute_cc_delta_mean",
        "mean_absolute_cc_delta", "mean_abs_delta_cc_abs",
    ),
    ("abs_delta_cc", "std"): (
        "abs_cc_delta_std", "std_abs_cc_delta", "abs_delta_cc_std",
        "std_abs_delta_cc", "absolute_cc_delta_std", "std_absolute_cc_delta",
    ),
    ("efficiency", "mean"): (
        "efficiency_ms_per_image_mean", "mean_efficiency_ms_per_image",
        "inference_time_ms_mean", "mean_inference_time_ms",
        "efficiency_ms_image", "efficiency_mean", "mean_efficiency",
    ),
    ("efficiency", "std"): (
        "efficiency_ms_per_image_std", "std_efficiency_ms_per_image",
        "inference_time_ms_std", "std_inference_time_ms",
        "efficiency_std", "std_efficiency",
    ),
}

EMPTY_PREDICTION_COUNT_ALIASES = (
    "empty_prediction_count", "pred_empty_count", "empty_pred_count",
    "number_of_empty_predictions", "num_empty_predictions",
)

HD_COUNT_DELTA_ALIASES = (
    "hd_count_delta", "distance_count_delta", "valid_hd_count_delta",
)

HD_VALID_COUNT_ALIASES = (
    "hd_valid_count", "hd95_valid_count", "distance_valid_count",
)

EVALUATED_COUNT_ALIASES = (
    "evaluated_gt_nonempty_count", "evaluated_count", "case_count",
    "num_cases", "num_images",
)


def metric(values: Mapping[str, float], name: str, stat: str) -> Optional[float]:
    # Never let abs_cc_delta_mean satisfy the signed cc_delta_mean suffix.
    if name == "delta_cc":
        values = {
            key: value
            for key, value in values.items()
            if not re.search(r"(?:^|_)(?:abs|absolute)(?:_|$)", key)
        }
    return find_value(values, METRIC_ALIASES[(name, stat)])


def hd_count_delta(values: Mapping[str, float]) -> Optional[float]:
    """Return the number of non-empty-GT cases with an empty prediction.

    New summaries expose ``empty_prediction_count`` directly.  In older
    summaries, HD/HD95 is non-finite for an empty prediction, so the same count
    is reconstructed as evaluated_count - hd_valid_count.  A legacy explicit
    ``hd_count_delta`` is accepted as a final fallback.
    """
    empty_count = find_value(values, EMPTY_PREDICTION_COUNT_ALIASES)
    if empty_count is not None:
        return max(0.0, empty_count)

    evaluated_count = find_value(values, EVALUATED_COUNT_ALIASES)
    valid_count = find_value(values, HD_VALID_COUNT_ALIASES)
    if evaluated_count is not None and valid_count is not None:
        return max(0.0, evaluated_count - valid_count)

    explicit = find_value(values, HD_COUNT_DELTA_ALIASES)
    return abs(explicit) if explicit is not None else None


# Display names and order are the final benchmark table protocol.  Aliases are
# exact normalized names; substring matching is intentionally avoided so that
# e.g. Unet cannot accidentally match Mamba-Unet or RWKV-Unet.
MODEL_SPECS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Unet", ("monai_unet", "unet")),
    ("Vnet", ("monai_vnet", "vnet")),
    ("Attention U-net", ("monai_attention_unet", "attention_unet", "attention_u_net", "attentionunet")),
    ("Unet++", ("monai_unetplusplus", "unetplusplus", "unet_plus_plus", "unetpp")),
    ("MedNeXt", ("mednext", "med_next")),
    ("nnU-net", ("nnunet_2d", "nnunet2d", "nnunet", "nn_u_net", "nn_u_net_2d")),
    ("AAUnet", ("aau_net", "aaunet", "aau_net_2d")),
    ("FPN-ResNet34", ("fpn_resnet34", "resnet34_fpn")),
    ("DeepLabV3+-ResNet34", ("deeplabv3plus_resnet34", "resnet34_dlv3", "resnet_dlv3")),
    ("Swin-Unet", ("swinunet", "swin_unet")),
    ("nnFormer", ("nnformer_2d", "nnformer2d", "nnformer")),
    ("PVTv2-b2-EMCAD", ("pvtb2_emcad", "pvtv2_b2_emcad", "pvt2_b2_emcad", "pvtv2b2emcad")),
    ("SEPNet", ("sepnet", "sep_net")),
    ("NuLite", ("nulite", "nu_lite")),
    ("Mamba-Unet", ("mamba_unet", "mambaunet")),
    ("VM-Unet V2", ("vm_unet_v2", "vmunetv2", "vm_unetv2")),
    ("nn-Mamba", ("nnmamba_2d", "nnmamba2d", "nn_mamba", "nnmamba")),
    ("Swin Umamba", ("swin_umamba", "swinumamba")),
    ("Rolling-UNet", ("rolling_unet", "rollingunet")),
    ("U-KAN", ("u_kan", "ukan")),
    ("xLSTM-Unet", ("xlstm_unet_bot", "xlstm_unet", "xlstmunet")),
    ("S2DENet", ("s2denet", "s2de_net")),
    ("U-RWKV", ("u_rwkv", "urwkv")),
    ("RWKV-Unet", ("rwkv_unet", "rwkvunet")),
    ("SAM2", ("sam2", "sam_2")),
    ("MedSAM", ("medsam", "med_sam")),
    ("SAMUS", ("samus", "sam_us")),
    ("SAM3", ("sam3", "sam_3")),
    ("USFM transfer", ("usfm_transfer", "usfm")),
    ("YOLO11s-seg", ("yolo11s_seg", "yolo11s")),
    ("YOLO11m-seg", ("yolo11m_seg", "yolo11m")),
    ("YOLO11l-seg", ("yolo11l_seg", "yolo11l")),
    ("YOLO26s-sem", ("yolo26s_sem", "yolo26s")),
    ("YOLO26m-sem", ("yolo26m_sem", "yolo26m")),
    ("YOLO26l-sem", ("yolo26l_sem", "yolo26l")),
    ("SegFormer-B2", ("segformer", "segformer_b2", "segformerb2")),
)

def compact_model_name(name: object) -> str:
    text = str(name or "").casefold()
    text = text.replace("++", "plusplus").replace("+", "plus")
    return re.sub(r"[^a-z0-9]", "", text)


ALIAS_TO_CANONICAL: Dict[str, str] = {}
CANONICAL_TO_DISPLAY: Dict[str, str] = {}
MODEL_ORDER: Dict[str, int] = {}
for order, (display_name, aliases) in enumerate(MODEL_SPECS):
    canonical = compact_model_name(display_name)
    CANONICAL_TO_DISPLAY[canonical] = display_name
    MODEL_ORDER[canonical] = order
    for alias in (display_name, *aliases):
        compact = compact_model_name(alias)
        previous = ALIAS_TO_CANONICAL.get(compact)
        if previous is not None and previous != canonical:
            raise RuntimeError(f"model alias collision: {alias!r}")
        ALIAS_TO_CANONICAL[compact] = canonical


def model_key(name: object) -> str:
    compact = compact_model_name(name)
    return ALIAS_TO_CANONICAL.get(compact, compact)


def model_display(name: object) -> str:
    """Return the agreed table label, preserving unknown model names."""
    return CANONICAL_TO_DISPLAY.get(model_key(name), str(name).strip())


def record_sort_key(record: "SummaryRecord") -> Tuple[int, int, str, int, int]:
    """Known model order first; within a model, 512 precedes 224."""
    canonical = model_key(record.model)
    size_rank = {512: 0, 224: 1}.get(record.size, 2)
    if canonical in MODEL_ORDER:
        return 0, MODEL_ORDER[canonical], "", size_rank, record.size
    # Unknown models are retained after the prescribed list and grouped by
    # their own names, still with 512 before 224.
    return 1, 0, model_display(record.model).casefold(), size_rank, record.size


@dataclass
class SummaryRecord:
    split: str
    model: str
    size: int
    path: Path
    values: Dict[str, float]


def is_summary_candidate(path: Path) -> bool:
    if path.suffix.casefold() != ".json":
        return False
    stem = normalize_key(path.stem)
    if re.fullmatch(r"(train|val|validation|test)(?:_summary)?", stem):
        return True
    return stem == "summary" and any(
        canonical_split(part) is not None for part in path.parent.parts
    )


def discover_summaries(root: Path) -> Tuple[List[SummaryRecord], List[str]]:
    records: List[SummaryRecord] = []
    warnings: List[str] = []
    for path in sorted(root.rglob("*.json")):
        if not is_summary_candidate(path):
            continue
        path_split = split_from_path(path)
        if path_split is None:
            continue
        try:
            values, split, json_model, json_size = read_summary(path, path_split)
            path_model, path_size = infer_model_size(path, root)
            model = json_model or path_model
            size = json_size or path_size
            if json_size is not None and json_size != path_size:
                warnings.append(
                    f"size mismatch: using JSON size {json_size} for {path} "
                    f"(path suggested {path_size})"
                )
            records.append(SummaryRecord(split, model, size, path.resolve(), values))
        except Exception as error:
            warnings.append(f"ignored {path}: {type(error).__name__}: {error}")
    return records, warnings


def choose_records(records: Sequence[SummaryRecord], strict: bool) -> Tuple[List[SummaryRecord], List[str]]:
    grouped: Dict[Tuple[str, str, int], List[SummaryRecord]] = {}
    for record in records:
        grouped.setdefault((record.split, model_key(record.model), record.size), []).append(record)

    chosen: List[SummaryRecord] = []
    warnings: List[str] = []
    for key, group in sorted(grouped.items()):
        if len(group) == 1:
            chosen.append(group[0])
            continue
        paths = "\n      ".join(str(item.path) for item in group)
        if strict:
            raise RuntimeError(
                f"duplicate summaries for split/model/size {key}:\n      {paths}"
            )
        newest = max(group, key=lambda item: (item.path.stat().st_mtime, str(item.path)))
        chosen.append(newest)
        warnings.append(
            f"duplicate {key}: using newest {newest.path}; candidates:\n      {paths}"
        )
    return chosen, warnings


AbsCcKey = Tuple[str, int, str]
AbsCcValue = Tuple[Optional[float], Optional[float]]


def normalized_csv_row(row: Mapping[str, str]) -> Dict[str, str]:
    return {normalize_key(key): value for key, value in row.items() if key is not None}


def first_row_value(row: Mapping[str, str], names: Iterable[str]) -> str:
    for name in names:
        value = row.get(normalize_key(name), "").strip()
        if value:
            return value
    return ""


def parse_abs_cc_csv(path: Path) -> Dict[AbsCcKey, AbsCcValue]:
    output: Dict[AbsCcKey, AbsCcValue] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        for line_number, raw in enumerate(reader, start=2):
            row = normalized_csv_row(raw)
            model = first_row_value(row, ("model", "model_name", "architecture"))
            size_value = parse_number(first_row_value(row, ("size", "image_size", "input_size")))
            if not model or size_value is None or not size_value.is_integer():
                continue
            size = int(size_value)
            row_split = canonical_split(first_row_value(row, ("split", "phase")))

            if row_split is not None:
                mean = parse_number(first_row_value(row, (
                    "abs_cc_delta_mean", "mean_abs_cc_delta", "abs_delta_cc_mean",
                    "mean_abs_delta_cc", "mean_abs_delta_cc_abs",
                    "mean_delta_cc", "mean_abs_delta_cc_value",
                )))
                std = parse_number(first_row_value(row, (
                    "abs_cc_delta_std", "std_abs_cc_delta", "abs_delta_cc_std",
                    "std_abs_delta_cc", "std_delta_cc",
                )))
                if mean is not None or std is not None:
                    output[(model_key(model), size, row_split)] = (mean, std)
                continue

            # Wide format emitted by collect_absolute_cc_from_cases.py:
            # Train-Mean-|Delta-CC|, Validation-Mean-|Delta-CC|, ...
            for split in SPLITS:
                prefixes = (split,) if split != "val" else ("val", "validation")
                mean_names: List[str] = []
                std_names: List[str] = []
                for prefix in prefixes:
                    mean_names.extend((
                        f"{prefix}_mean_abs_delta_cc_abs",
                        f"{prefix}_mean_abs_delta_cc",
                        f"{prefix}_mean_delta_cc",
                        f"{prefix}_abs_cc_delta_mean",
                    ))
                    std_names.extend((
                        f"{prefix}_std_abs_delta_cc_abs",
                        f"{prefix}_std_abs_delta_cc",
                        f"{prefix}_std_delta_cc",
                        f"{prefix}_abs_cc_delta_std",
                    ))
                mean = parse_number(first_row_value(row, mean_names))
                std = parse_number(first_row_value(row, std_names))
                if mean is not None or std is not None:
                    output[(model_key(model), size, split)] = (mean, std)
    return output


def discover_abs_cc_files(root: Path, explicit: Sequence[Path]) -> List[Path]:
    if explicit:
        return [path.expanduser().resolve() for path in explicit]
    candidates = set(root.rglob("absolute_cc_summary.csv"))
    for base in (root, root.parent):
        candidate = base / "absolute_cc_summary.csv"
        if candidate.is_file():
            candidates.add(candidate)
    return sorted(path.resolve() for path in candidates if path.is_file())


def load_abs_cc(files: Sequence[Path], strict: bool) -> Tuple[Dict[AbsCcKey, AbsCcValue], List[str]]:
    combined: Dict[AbsCcKey, AbsCcValue] = {}
    warnings: List[str] = []
    for path in files:
        if not path.is_file():
            message = f"absolute CC file not found: {path}"
            if strict:
                raise FileNotFoundError(message)
            warnings.append(message)
            continue
        try:
            parsed = parse_abs_cc_csv(path)
        except Exception as error:
            message = f"failed to parse {path}: {type(error).__name__}: {error}"
            if strict:
                raise RuntimeError(message) from error
            warnings.append(message)
            continue
        for key, value in parsed.items():
            previous = combined.get(key)
            if previous is not None and previous != value:
                message = f"conflicting absolute CC values for {key}; using value from {path}"
                if strict:
                    raise RuntimeError(message)
                warnings.append(message)
            combined[key] = value
    return combined, warnings


def scaled_score(value: Optional[float], mode: str, reference_values: Sequence[Optional[float]]) -> Optional[float]:
    if value is None:
        return None
    if mode == "raw":
        return value
    if mode == "percent":
        finite = [abs(item) for item in reference_values if item is not None]
        return value * 100.0 if finite and max(finite) <= 1.000001 else value
    if mode == "fraction":
        finite = [abs(item) for item in reference_values if item is not None]
        return value / 100.0 if finite and max(finite) > 1.000001 else value
    raise ValueError(mode)


def format_cell(value: Optional[float], digits: int) -> object:
    if value is None or not math.isfinite(value):
        return ""
    result = round(float(value), digits)
    return 0.0 if result == 0 else result


def output_row(
    record: SummaryRecord,
    abs_cc: Mapping[AbsCcKey, AbsCcValue],
    digits: int,
    score_output: str,
) -> Dict[str, object]:
    values = record.values
    score_pairs = {
        "Mean-dice": metric(values, "dice", "mean"),
        "Std-dice": metric(values, "dice", "std"),
        "Mean-IoU": metric(values, "iou", "mean"),
        "Std-IoU": metric(values, "iou", "std"),
        "Mean-Precision": metric(values, "precision", "mean"),
        "Std-Precision": metric(values, "precision", "std"),
        "Mean-Recall": metric(values, "recall", "mean"),
        "Std-Recall": metric(values, "recall", "std"),
    }
    score_reference = list(score_pairs.values())

    abs_mean = metric(values, "abs_delta_cc", "mean")
    abs_std = metric(values, "abs_delta_cc", "std")
    if abs_mean is None or abs_std is None:
        fallback = abs_cc.get((model_key(record.model), record.size, record.split))
        if fallback is not None:
            if abs_mean is None:
                abs_mean = fallback[0]
            if abs_std is None:
                abs_std = fallback[1]

    row: Dict[str, object] = {column: "" for column in OUTPUT_COLUMNS}
    row.update({"Model": model_display(record.model), "Size": record.size})
    for column, value in score_pairs.items():
        row[column] = format_cell(
            scaled_score(value, score_output, score_reference), digits
        )
    direct = {
        "HD-count-delta": hd_count_delta(values),
        "Mean-HD95": metric(values, "hd95", "mean"),
        "Std-HD95": metric(values, "hd95", "std"),
        "Mean-HD": metric(values, "hd", "mean"),
        "Std-HD": metric(values, "hd", "std"),
        "Mean-Delta-CC": metric(values, "delta_cc", "mean"),
        "Std-Delta-CC": metric(values, "delta_cc", "std"),
        "Mean-Abs-Delta-CC": abs_mean,
        "Std-Abs-Delta-CC": abs_std,
        "Mean-Efficiency-ms/image": metric(values, "efficiency", "mean"),
        "Std-Efficiency-ms/image": metric(values, "efficiency", "std"),
    }
    for column, value in direct.items():
        row[column] = format_cell(value, digits)
    return row


def write_split_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively collect all Evaluation summary JSON files into train/val/test CSVs."
    )
    parser.add_argument(
        "--evaluation-root", type=Path, default=Path("Evaluation"),
        help="Root directory searched recursively (default: Evaluation).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for train.csv, val.csv and test.csv (default: evaluation root).",
    )
    parser.add_argument(
        "--absolute-cc", type=Path, action="append", default=[],
        help=(
            "Optional absolute_cc_summary.csv; repeat for multiple files. "
            "Without this option, files with that name are auto-discovered."
        ),
    )
    parser.add_argument(
        "--score-output", choices=("raw", "fraction", "percent"), default="raw",
        help="Keep scores as stored, or normalize Dice/IoU/Precision/Recall (default: raw).",
    )
    parser.add_argument(
        "--round-digits", type=int, default=6,
        help="Decimal places in output CSVs (default: 6).",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Stop on duplicate/bad summaries or bad absolute-CC files.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = args.evaluation_root.expanduser().resolve()
    output_dir = (args.output_dir or root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"evaluation root does not exist: {root}")
    if args.round_digits < 0:
        raise ValueError("--round-digits must be non-negative")

    discovered, warnings = discover_summaries(root)
    records, duplicate_warnings = choose_records(discovered, args.strict)
    warnings.extend(duplicate_warnings)

    abs_files = discover_abs_cc_files(root, args.absolute_cc)
    abs_cc, abs_warnings = load_abs_cc(abs_files, args.strict)
    warnings.extend(abs_warnings)

    counts: Dict[str, int] = {}
    missing_abs = 0
    for split in SPLITS:
        split_records = sorted(
            (record for record in records if record.split == split),
            key=record_sort_key,
        )
        rows = [
            output_row(record, abs_cc, args.round_digits, args.score_output)
            for record in split_records
        ]
        missing_abs += sum(row["Mean-Abs-Delta-CC"] == "" for row in rows)
        output_path = output_dir / f"{split}.csv"
        write_split_csv(output_path, rows)
        counts[split] = len(rows)
        print(f"Wrote {output_path} ({len(rows)} row(s))")

    print(
        f"Discovered {len(discovered)} summary file(s); selected {len(records)} "
        f"unique model/size/split result(s)."
    )
    if abs_files:
        print(f"Loaded absolute CC fallback data from {len(abs_files)} file(s).")
    else:
        print("No absolute_cc_summary.csv found; unavailable |Delta-CC| cells were left blank.")
    if missing_abs:
        print(f"Left Mean-Abs-Delta-CC blank in {missing_abs} output row(s).")
    if warnings:
        print("\nWarnings:", file=sys.stderr)
        for warning in warnings:
            print(f"  - {warning}", file=sys.stderr)

    if not records:
        print("No supported summary JSON files were found.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
