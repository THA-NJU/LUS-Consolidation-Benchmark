#!/usr/bin/env python3
"""Unified evaluator for models trained by autodl_easy_models_benchmark.py.

The evaluator imports the original training script so model construction,
preprocessing, output unwrapping, AMP dtype, and checkpoint conventions stay
identical to training.

Default scope:
  - family: CNN
  - models: aau_net, nnunet_2d
  - sizes: 224, 512
  - splits: train, val, test

Evaluation protocol:
  - fixed probability threshold 0.5;
  - 224 samples are independent images: no stitching and no output resize;
  - GT-empty samples are excluded from case metrics and macro mean/std;
  - GT-nonempty/pred-empty samples receive Dice/IoU/Precision/Recall = 0,
    while HD/HD95 equal the track side length and enter distance mean/std;
  - HD and HD95 are symmetric Euclidean surface distances in pixels;
  - connected components use 8-connectivity and
    cc_delta = gt_components - pred_components;
  - all reported segmentation metrics are per-image macro statistics;
  - efficiency excludes disk I/O, DataLoader, host-to-device transfer, metric
    computation, CSV output, and visualization. It includes model forward,
    softmax, and thresholding;
  - no prediction-mask images are saved. Each split saves only one top-5 and
    one bottom-5 Dice montage, with GT-empty samples excluded.

Expected weight layout:
  ../Pth/<family>/<size>/<model>/*best*.pth

Expected dataset layout:
  <data_root>/<split>/images/*.png
  <data_root>/<split>/masks/*.png

Run this file inside the same APRIL-MedSeg environment and repository as the
training script.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# =============================================================================
# USER SETTINGS
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

# Leave as None to auto-discover autodl_easy_models_benchmark*.py beside this
# evaluator. Set an explicit path if the training script is elsewhere.
TRAINING_SCRIPT: Optional[Path] = None

WEIGHTS_ROOT = Path("../Pth")
EVALUATION_ROOT = Path("../Evaluation/autodl_easy_models")

RUN_SIZES = [224, 512]
FAMILIES_TO_EVALUATE = ["CNN"]
SPLITS = ["train", "val", "test"]
GPU_ID = 0

THRESHOLD = 0.50
NUM_WORKERS = 4
PIN_MEMORY = True
AUTO_TUNE_BATCH = True
BATCH_TUNE_WARMUP_STEPS = 3
BATCH_TUNE_TIMED_STEPS = 10
MAX_BATCH_BY_SIZE = {224: 256, 512: 64}
VISUAL_CASES = 5
VISUAL_ALPHA = 0.45

# Only models built by the supplied training script are listed here.
MODEL_FAMILY = {
    "nnunet_2d": "CNN",
    "aau_net": "CNN",
    "swinunet": "Transformer",
    "nnformer_2d": "Transformer",
    "sepnet": "Transformer",
    "nulite": "Transformer",
    "ukan": "other",
    "xlstm_unet_bot": "other",
}

METRICS = ("dice", "iou", "hd", "hd95", "precision", "recall", "cc_delta")
CC_STRUCTURE_8 = np.ones((3, 3), dtype=np.uint8)


@dataclass
class VisualCase:
    case_name: str
    dice: float
    image_path: Path
    gt: np.ndarray
    pred: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-script",
        type=Path,
        default=TRAINING_SCRIPT,
        help="Original autodl_easy_models_benchmark.py used for training.",
    )
    parser.add_argument("--weights-root", type=Path, default=WEIGHTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=EVALUATION_ROOT)
    parser.add_argument("--gpu", type=int, default=GPU_ID)
    parser.add_argument("--sizes", type=int, nargs="+", default=RUN_SIZES, choices=(224, 512))
    parser.add_argument("--families", nargs="+", default=FAMILIES_TO_EVALUATE)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional exact model-name filter. By default all models in selected families are used.",
    )
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=("train", "val", "test"))
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--no-auto-batch", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def discover_training_script(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        path = resolve_path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Training script not found: {path}")
        return path

    candidates = sorted(
        p
        for p in PROJECT_ROOT.glob("autodl_easy_models_benchmark*.py")
        if p.name != Path(__file__).name
    )
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(
            "Could not auto-discover autodl_easy_models_benchmark*.py beside "
            "the evaluator. Pass --training-script /absolute/path/to/script.py"
        )
    names = "\n  ".join(str(p) for p in candidates)
    raise RuntimeError(
        "Several training scripts matched; choose the exact one with "
        f"--training-script:\n  {names}"
    )


def import_training_module(path: Path):
    spec = importlib.util.spec_from_file_location("lus_training_source", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = (
        "DATA_ROOTS",
        "MODEL_SPECS",
        "ConsolidationDataset",
        "build_april_model",
        "get_model_spec",
        "resolve_amp_dtype",
        "autocast_context",
        "unwrap_primary_output",
        "set_task_context",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            f"Training script {path} lacks required definitions: {missing}"
        )
    return module


def select_models(training, families: Sequence[str], explicit: Optional[Sequence[str]]) -> List[str]:
    family_lookup = {name.lower(): name for name in set(MODEL_FAMILY.values())}
    normalized_families = []
    for family in families:
        key = family.lower()
        if key not in family_lookup:
            raise KeyError(
                f"Unknown family {family!r}; choices={sorted(family_lookup.values())}"
            )
        normalized_families.append(family_lookup[key])

    supported = [
        name
        for name, family in MODEL_FAMILY.items()
        if family in normalized_families and name in training.MODEL_SPECS
    ]
    if explicit is not None:
        requested = list(dict.fromkeys(explicit))
        unknown = [name for name in requested if name not in training.MODEL_SPECS]
        if unknown:
            raise KeyError(f"Models absent from the training script: {unknown}")
        wrong_family = [
            name
            for name in requested
            if MODEL_FAMILY.get(name) not in normalized_families
        ]
        if wrong_family:
            raise ValueError(
                f"Models do not belong to selected families {normalized_families}: "
                f"{wrong_family}"
            )
        supported = requested
    if not supported:
        raise RuntimeError("No models selected")
    return supported


def find_best_checkpoint(weights_root: Path, family: str, size: int, model_name: str) -> Path:
    model_dir = weights_root / family / str(size) / model_name
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Weight directory not found: {model_dir}")

    exact = model_dir / "best_model.pth"
    if exact.is_file():
        return exact.resolve()

    candidates = sorted(
        p for p in model_dir.iterdir()
        if p.is_file() and "best" in p.name.lower() and p.suffix.lower() == ".pth"
    )
    if not candidates:
        raise FileNotFoundError(
            f"No .pth checkpoint containing 'best' was found in {model_dir}"
        )
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise RuntimeError(
            f"Ambiguous best checkpoint in {model_dir}: {names}. "
            "Keep one matching file or name the intended checkpoint best_model.pth."
        )
    return candidates[0].resolve()


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, Mapping):
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    else:
        state = checkpoint
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint does not contain a state_dict: {checkpoint_path}")

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        # Compatibility with checkpoints saved from DataParallel.
        if state and all(str(k).startswith("module.") for k in state):
            stripped = {str(k)[7:]: v for k, v in state.items()}
            model.load_state_dict(stripped, strict=True)
        else:
            raise
    return dict(checkpoint) if isinstance(checkpoint, Mapping) else {}


def make_eval_dataset(training, data_root: Path, split: str, image_size: int):
    # train=False is deliberate: train-set evaluation must not apply random augmentation.
    return training.ConsolidationDataset(
        data_root / split,
        train=False,
        image_size=image_size,
        max_samples=None,
    )


def make_loader(dataset, batch_size: int, workers: int, pin_memory: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(workers),
        pin_memory=bool(pin_memory),
        persistent_workers=int(workers) > 0,
    )


def model_probabilities(training, model, images, amp_dtype):
    with training.autocast_context(
        bool(training.USE_AMP and images.device.type == "cuda"),
        amp_dtype,
    ):
        logits = training.unwrap_primary_output(model(images))
        if logits.ndim != 4:
            raise RuntimeError(f"Expected BCHW logits, got shape={tuple(logits.shape)}")
        if logits.shape[1] == 2:
            return torch.softmax(logits.float(), dim=1)[:, 1]
        if logits.shape[1] == 1:
            return torch.sigmoid(logits.float())[:, 0]
        raise RuntimeError(
            f"Expected one or two output channels, got shape={tuple(logits.shape)}"
        )


@torch.inference_mode()
def benchmark_batch_size(
    training,
    model: torch.nn.Module,
    sample: torch.Tensor,
    amp_dtype: torch.dtype,
    candidates: Sequence[int],
    device: torch.device,
    threshold: float,
) -> Tuple[int, List[Dict[str, object]]]:
    if device.type != "cuda":
        return int(candidates[0]), []

    rows: List[Dict[str, object]] = []
    model.eval()
    for batch_size in candidates:
        try:
            torch.cuda.empty_cache()
            images = sample.repeat(int(batch_size), 1, 1, 1).to(device)
            torch.cuda.reset_peak_memory_stats(device)

            for _ in range(BATCH_TUNE_WARMUP_STEPS):
                probs = model_probabilities(training, model, images, amp_dtype)
                _ = probs >= float(threshold)
            torch.cuda.synchronize(device)

            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            for _ in range(BATCH_TUNE_TIMED_STEPS):
                probs = model_probabilities(training, model, images, amp_dtype)
                _ = probs >= float(threshold)
            ender.record()
            ender.synchronize()

            elapsed_ms = float(starter.elapsed_time(ender))
            images_seen = int(batch_size) * BATCH_TUNE_TIMED_STEPS
            images_per_s = images_seen / max(elapsed_ms / 1000.0, 1e-12)
            peak_gb = float(torch.cuda.max_memory_allocated(device)) / (1024 ** 3)
            rows.append(
                {
                    "batch_size": int(batch_size),
                    "status": "ok",
                    "images_per_s": images_per_s,
                    "ms_per_image": elapsed_ms / images_seen,
                    "peak_memory_gb": peak_gb,
                }
            )
            del images, probs
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            rows.append(
                {
                    "batch_size": int(batch_size),
                    "status": "oom",
                    "images_per_s": math.nan,
                    "ms_per_image": math.nan,
                    "peak_memory_gb": math.nan,
                }
            )
            torch.cuda.empty_cache()
            break

    valid = [row for row in rows if row["status"] == "ok"]
    if not valid:
        raise RuntimeError("Every candidate batch size ran out of GPU memory")
    best = max(valid, key=lambda row: float(row["images_per_s"]))
    return int(best["batch_size"]), rows


@torch.inference_mode()
def warmup_model(
    training,
    model: torch.nn.Module,
    sample: torch.Tensor,
    batch_size: int,
    amp_dtype: torch.dtype,
    threshold: float,
    device: torch.device,
) -> None:
    """Warm kernels at the selected batch size without adding to efficiency."""
    images = sample.repeat(int(batch_size), 1, 1, 1).to(device)
    for _ in range(BATCH_TUNE_WARMUP_STEPS):
        probs = model_probabilities(training, model, images, amp_dtype)
        _ = probs >= float(threshold)
    torch.cuda.synchronize(device)
    del images, probs


def batch_candidates(start: int, maximum: int) -> List[int]:
    start = max(1, int(start))
    maximum = max(start, int(maximum))
    values: List[int] = []
    value = start
    while value <= maximum:
        values.append(value)
        value *= 2
    if values[-1] != maximum and maximum > values[-1]:
        values.append(maximum)
    return values


def connected_components(mask: np.ndarray) -> int:
    _, count = ndimage.label(mask.astype(bool), structure=CC_STRUCTURE_8)
    return int(count)


def surface_distances(gt: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    if not gt.any():
        return math.nan, math.nan
    if not pred.any():
        penalty = float(max(gt.shape))
        return penalty, penalty

    gt_surface = gt ^ ndimage.binary_erosion(
        gt, structure=CC_STRUCTURE_8, border_value=0
    )
    pred_surface = pred ^ ndimage.binary_erosion(
        pred, structure=CC_STRUCTURE_8, border_value=0
    )
    distance_to_gt = ndimage.distance_transform_edt(~gt_surface)
    distance_to_pred = ndimage.distance_transform_edt(~pred_surface)
    distances = np.concatenate(
        (distance_to_pred[gt_surface], distance_to_gt[pred_surface])
    ).astype(np.float64, copy=False)
    if distances.size == 0:
        return math.nan, math.nan
    return float(np.max(distances)), float(np.percentile(distances, 95))


def compute_case_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, object]:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    gt_pixels = int(gt.sum())
    pred_pixels = int(pred.sum())
    if gt_pixels == 0:
        raise ValueError("GT-empty cases must be filtered before metric computation")

    tp = int(np.logical_and(gt, pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())
    pred_empty = pred_pixels == 0

    dice_den = 2 * tp + fp + fn
    iou_den = tp + fp + fn
    dice = 0.0 if pred_empty else (2.0 * tp / max(dice_den, 1))
    iou = 0.0 if pred_empty else (tp / max(iou_den, 1))
    precision = 0.0 if pred_empty else (tp / max(tp + fp, 1))
    recall = 0.0 if pred_empty else (tp / max(tp + fn, 1))
    hd, hd95 = surface_distances(gt, pred)
    gt_cc = connected_components(gt)
    pred_cc = connected_components(pred)

    return {
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": float(dice),
        "iou": float(iou),
        "hd": float(hd),
        "hd95": float(hd95),
        "precision": float(precision),
        "recall": float(recall),
        "gt_cc": gt_cc,
        "pred_cc": pred_cc,
        "cc_delta": int(gt_cc - pred_cc),
        "pred_empty": bool(pred_empty),
    }


def finite_values(rows: Sequence[Mapping[str, object]], key: str) -> np.ndarray:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def population_mean_std(values: np.ndarray) -> Tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    return float(np.mean(values)), float(np.std(values, ddof=0))


def summarize_split(
    split: str,
    case_rows: Sequence[Mapping[str, object]],
    total_samples: int,
    gt_empty_count: int,
    efficiency_rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "split": split,
        "total_samples": int(total_samples),
        "gt_empty_excluded_count": int(gt_empty_count),
        "evaluated_gt_nonempty_count": int(len(case_rows)),
        "empty_prediction_count": int(sum(bool(row["pred_empty"]) for row in case_rows)),
        "empty_prediction_rate": (
            float(sum(bool(row["pred_empty"]) for row in case_rows)) / len(case_rows)
            if case_rows else math.nan
        ),
    }
    for metric in METRICS:
        values = finite_values(case_rows, metric)
        mean, std = population_mean_std(values)
        result[f"{metric}_mean"] = mean
        result[f"{metric}_std"] = std
        result[f"{metric}_valid_count"] = int(values.size)

    times_s = np.asarray(
        [float(row["elapsed_s"]) for row in efficiency_rows], dtype=np.float64
    )
    batch_sizes = np.asarray(
        [int(row["batch_size"]) for row in efficiency_rows], dtype=np.int64
    )
    ms_per_image = np.asarray(
        [float(row["ms_per_image"]) for row in efficiency_rows], dtype=np.float64
    )
    images_per_s = np.asarray(
        [float(row["images_per_s"]) for row in efficiency_rows], dtype=np.float64
    )
    ms_mean, ms_std = population_mean_std(ms_per_image)
    ips_mean, ips_std = population_mean_std(images_per_s)
    total_timed_images = int(batch_sizes.sum()) if batch_sizes.size else 0
    total_inference_s = float(times_s.sum()) if times_s.size else 0.0
    result.update(
        {
            "efficiency_batch_count": int(len(efficiency_rows)),
            "efficiency_timed_images": total_timed_images,
            "efficiency_total_inference_s": total_inference_s,
            "efficiency_ms_per_image_mean": ms_mean,
            "efficiency_ms_per_image_std": ms_std,
            "efficiency_images_per_s_mean": ips_mean,
            "efficiency_images_per_s_std": ips_std,
            "efficiency_images_per_s_aggregate": (
                total_timed_images / total_inference_s
                if total_inference_s > 0 else math.nan
            ),
        }
    )
    return result


def update_visual_cases(
    top_cases: List[VisualCase],
    bottom_cases: List[VisualCase],
    candidate: VisualCase,
    limit: int,
) -> None:
    top_cases.append(candidate)
    top_cases.sort(key=lambda item: (-item.dice, item.case_name))
    del top_cases[limit:]

    bottom_cases.append(candidate)
    bottom_cases.sort(key=lambda item: (item.dice, item.case_name))
    del bottom_cases[limit:]


def overlay_mask(image: Image.Image, mask: np.ndarray, color: Tuple[int, int, int]) -> Image.Image:
    base = image.convert("RGB")
    color_image = Image.new("RGB", base.size, color)
    alpha = Image.fromarray(
        np.uint8(mask.astype(bool) * round(255 * VISUAL_ALPHA)), mode="L"
    )
    return Image.composite(color_image, base, alpha)


def save_visual_montage(
    path: Path,
    cases: Sequence[VisualCase],
    image_size: int,
    title: str,
) -> None:
    if not cases:
        return
    header_h = 34
    row_label_h = 24
    panel = int(image_size)
    canvas = Image.new(
        "RGB",
        (3 * panel, header_h + len(cases) * (row_label_h + panel)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((6, 5), f"{title} | Original | GT overlay (green) | Prediction overlay (red)", fill="black", font=font)

    for row_index, case in enumerate(cases):
        y = header_h + row_index * (row_label_h + panel)
        draw.text(
            (6, y + 4),
            f"{case.case_name} | Dice={case.dice:.6f}",
            fill="black",
            font=font,
        )
        with Image.open(case.image_path) as handle:
            original = handle.convert("L")
        if original.size != (panel, panel):
            original = original.resize((panel, panel), Image.Resampling.BILINEAR)
        original_rgb = original.convert("RGB")
        gt_overlay = overlay_mask(original_rgb, case.gt, (0, 255, 0))
        pred_overlay = overlay_mask(original_rgb, case.pred, (255, 0, 0))
        panel_y = y + row_label_h
        canvas.paste(original_rgb, (0, panel_y))
        canvas.paste(gt_overlay, (panel, panel_y))
        canvas.paste(pred_overlay, (2 * panel, panel_y))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (Path,)):
        return str(value)
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, ensure_ascii=False, indent=2)


@torch.inference_mode()
def evaluate_split(
    training,
    model: torch.nn.Module,
    dataset,
    split: str,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    threshold: float,
    workers: int,
    output_dir: Path,
) -> Dict[str, object]:
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        workers=workers,
        pin_memory=PIN_MEMORY and device.type == "cuda",
    )
    image_path_by_name = {image.name: image for image, _ in dataset.samples}

    case_rows: List[Dict[str, object]] = []
    efficiency_rows: List[Dict[str, object]] = []
    top_cases: List[VisualCase] = []
    bottom_cases: List[VisualCase] = []
    gt_empty_count = 0
    model.eval()

    for batch_index, batch in enumerate(
        tqdm(loader, desc=f"{split} evaluation", unit="batch")
    ):
        # Host-to-device transfer occurs before timing by protocol.
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["label"]
        case_names = list(batch["case_name"])
        patient_ids = list(batch["patient_id"])

        if device.type == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            probs = model_probabilities(training, model, images, amp_dtype)
            predictions = probs >= float(threshold)
            ender.record()
            ender.synchronize()
            elapsed_s = float(starter.elapsed_time(ender)) / 1000.0
        else:
            raise RuntimeError("CUDA is required for the agreed efficiency protocol")

        current_bs = int(images.shape[0])
        efficiency_rows.append(
            {
                "batch_index": batch_index,
                "batch_size": current_bs,
                "elapsed_s": elapsed_s,
                "ms_per_image": elapsed_s * 1000.0 / current_bs,
                "images_per_s": current_bs / max(elapsed_s, 1e-12),
            }
        )

        # Metric computation begins only after inference timing has ended.
        pred_np = predictions.detach().cpu().numpy().astype(bool)
        gt_np = targets.numpy().astype(bool)
        for item_index, (case_name, patient_id) in enumerate(zip(case_names, patient_ids)):
            gt = gt_np[item_index]
            pred = pred_np[item_index]
            if not gt.any():
                gt_empty_count += 1
                continue

            metrics = compute_case_metrics(gt, pred)
            row = {
                "case_name": case_name,
                "patient_id": patient_id,
                **metrics,
            }
            case_rows.append(row)
            candidate = VisualCase(
                case_name=case_name,
                dice=float(metrics["dice"]),
                image_path=image_path_by_name[case_name],
                gt=gt.astype(np.uint8, copy=True),
                pred=pred.astype(np.uint8, copy=True),
            )
            update_visual_cases(
                top_cases, bottom_cases, candidate, limit=VISUAL_CASES
            )

        del images, probs, predictions

    summary = summarize_split(
        split=split,
        case_rows=case_rows,
        total_samples=len(dataset),
        gt_empty_count=gt_empty_count,
        efficiency_rows=efficiency_rows,
    )
    summary["batch_size"] = int(batch_size)
    summary["threshold"] = float(threshold)
    summary["efficiency_timing_scope"] = (
        "forward+softmax+threshold; excludes disk_io+dataloader+h2d+metrics+outputs"
    )
    summary["std_definition"] = "population_std_ddof_0"

    write_csv(output_dir / f"{split}_cases.csv", case_rows)
    write_csv(output_dir / f"{split}_efficiency_batches.csv", efficiency_rows)
    write_json(output_dir / f"{split}_summary.json", summary)
    save_visual_montage(
        output_dir / f"{split}_top5_dice.png",
        top_cases,
        image_size=dataset.image_size,
        title=f"{split} top-{len(top_cases)} Dice",
    )
    save_visual_montage(
        output_dir / f"{split}_bottom5_dice.png",
        bottom_cases,
        image_size=dataset.image_size,
        title=f"{split} bottom-{len(bottom_cases)} Dice",
    )
    return summary


def evaluate_one_model(
    training,
    training_script: Path,
    weights_root: Path,
    output_root: Path,
    model_name: str,
    size: int,
    splits: Sequence[str],
    device: torch.device,
    workers: int,
    threshold: float,
    auto_tune: bool,
) -> List[Dict[str, object]]:
    family = MODEL_FAMILY[model_name]
    data_root = resolve_path(Path(training.DATA_ROOTS[int(size)]))
    training.set_task_context(size, data_root, output_root)
    training.ensure_dataset_layout(data_root)
    training.seed_everything(int(training.SEED))

    checkpoint_path = find_best_checkpoint(
        weights_root, family, size, model_name
    )
    model, model_config = training.build_april_model(model_name)
    model = model.to(device)
    checkpoint = load_model_checkpoint(model, checkpoint_path, device)
    spec = training.get_model_spec(model_name)
    amp_dtype = training.resolve_amp_dtype(
        spec.get("amp_dtype", training.DEFAULT_AMP_DTYPE), device
    )
    parameter_count = int(sum(p.numel() for p in model.parameters()))

    datasets = {
        split: make_eval_dataset(training, data_root, split, size)
        for split in splits
    }
    if not datasets:
        raise RuntimeError("No splits selected")
    first_dataset = next(iter(datasets.values()))
    sample = first_dataset[0]["image"].unsqueeze(0)
    training_batch = int(spec["batch_size"])
    tune_rows: List[Dict[str, object]] = []
    if auto_tune:
        candidates = batch_candidates(
            training_batch,
            MAX_BATCH_BY_SIZE[int(size)],
        )
        eval_batch, tune_rows = benchmark_batch_size(
            training,
            model,
            sample,
            amp_dtype,
            candidates,
            device,
            threshold,
        )
    else:
        eval_batch = training_batch
    warmup_model(
        training,
        model,
        sample,
        eval_batch,
        amp_dtype,
        threshold,
        device,
    )

    model_output = output_root / family / str(size) / model_name
    model_output.mkdir(parents=True, exist_ok=True)
    write_csv(model_output / "batch_tuning.csv", tune_rows)
    settings = {
        "training_script": str(training_script),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_score": checkpoint.get("best_score"),
        "family": family,
        "model": model_name,
        "size": int(size),
        "data_root": str(data_root),
        "splits": list(splits),
        "threshold": float(threshold),
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "training_batch_size": training_batch,
        "evaluation_batch_size": int(eval_batch),
        "auto_tune_batch": bool(auto_tune),
        "parameter_count": parameter_count,
        "model_config": model_config,
        "protocol": {
            "gt_empty": "excluded",
            "gt_nonempty_pred_empty_overlap_metrics": 0,
            "gt_nonempty_pred_empty_hd_hd95": "track_side_length_included",
            "connected_components": "8-connectivity",
            "cc_delta": "gt_cc-pred_cc",
            "distance": "symmetric Euclidean surface distance in pixels",
            "aggregation": "per-image macro mean/std; no global metrics",
            "std": "population ddof=0",
            "prediction_masks_saved": False,
            "visualizations": "top5 and bottom5 Dice per split; GT-empty excluded",
        },
    }
    write_json(model_output / "evaluation_settings.json", settings)

    summaries: List[Dict[str, object]] = []
    for split in splits:
        summary = evaluate_split(
            training=training,
            model=model,
            dataset=datasets[split],
            split=split,
            batch_size=eval_batch,
            device=device,
            amp_dtype=amp_dtype,
            threshold=threshold,
            workers=workers,
            output_dir=model_output,
        )
        summary = {
            "family": family,
            "size": int(size),
            "model": model_name,
            "checkpoint": str(checkpoint_path),
            "amp_dtype": str(amp_dtype).replace("torch.", ""),
            "parameter_count": parameter_count,
            **summary,
        }
        summaries.append(summary)
    write_csv(model_output / "summary.csv", summaries)

    del datasets, model
    gc.collect()
    torch.cuda.empty_cache()
    return summaries


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(
            f"Invalid GPU {args.gpu}; visible CUDA devices={torch.cuda.device_count()}"
        )
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(f"Threshold must be in [0,1], got {args.threshold}")

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    training_script = discover_training_script(args.training_script)
    training = import_training_module(training_script)
    weights_root = resolve_path(args.weights_root)
    output_root = resolve_path(args.output_root)
    models = select_models(training, args.families, args.models)

    print("=" * 88)
    print(f"Training script: {training_script}")
    print(f"Weights root:    {weights_root}")
    print(f"Output root:     {output_root}")
    print(f"Device:          {device} ({torch.cuda.get_device_name(args.gpu)})")
    print(f"Models:          {models}")
    print(f"Sizes:           {args.sizes}")
    print(f"Splits:          {args.splits}")
    print(f"Threshold:       {args.threshold:.2f}")
    print("=" * 88)

    all_summaries: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    for size in args.sizes:
        for model_name in models:
            print("\n" + "#" * 88)
            print(
                f"EVALUATE family={MODEL_FAMILY[model_name]} "
                f"size={size} model={model_name}"
            )
            print("#" * 88)
            try:
                summaries = evaluate_one_model(
                    training=training,
                    training_script=training_script,
                    weights_root=weights_root,
                    output_root=output_root,
                    model_name=model_name,
                    size=int(size),
                    splits=args.splits,
                    device=device,
                    workers=int(args.workers),
                    threshold=float(args.threshold),
                    auto_tune=not args.no_auto_batch,
                )
                all_summaries.extend(summaries)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                failure = {
                    "family": MODEL_FAMILY.get(model_name),
                    "size": int(size),
                    "model": model_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(failure)
                print(f"[FAILED] {failure}")
                gc.collect()
                torch.cuda.empty_cache()

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "evaluation_summary.csv", all_summaries)
    write_csv(output_root / "evaluation_failures.csv", failures)
    write_json(
        output_root / "evaluation_report.json",
        {
            "completed_model_split_rows": len(all_summaries),
            "failed_model_tasks": len(failures),
            "summaries": all_summaries,
            "failures": failures,
        },
    )
    print("\n" + "=" * 88)
    print(
        f"Completed summary rows={len(all_summaries)}, "
        f"failed model tasks={len(failures)}"
    )
    print(f"Summary: {output_root / 'evaluation_summary.csv'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
