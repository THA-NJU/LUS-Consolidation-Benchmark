#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared native-resolution evaluation utilities for the LUS benchmark."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CASE_FIELDS = [
    "filename", "metric_included", "gt_pixels", "pred_pixels",
    "dice", "iou", "hd", "hd95", "precision", "recall",
    "gt_cc", "pred_cc", "cc_delta", "abs_cc_delta", "inference_time_ms",
]
METRIC_NAMES = [
    "dice", "iou", "hd", "hd95", "precision", "recall",
    "cc_delta", "abs_cc_delta", "inference_time_ms",
]


def read_gray(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Cannot read image: {path}")
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.dtype == np.uint16:
        arr = np.rint(arr.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        lo, hi = float(np.min(arr)), float(np.max(arr))
        if hi <= lo:
            arr = np.zeros(arr.shape, dtype=np.uint8)
        else:
            arr = np.rint((arr.astype(np.float32) - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return arr


def read_binary_mask(path: Path, label_values: Sequence[int] = (1,)) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Cannot read mask: {path}")
    if arr.ndim == 3:
        arr = arr[..., 0]
    return np.isin(arr, np.asarray(label_values)).astype(np.uint8)


def find_mask(mask_dir: Path, image_path: Path) -> Path:
    direct = mask_dir / image_path.name
    if direct.is_file():
        return direct
    matches = [p for p in mask_dir.glob(image_path.stem + ".*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one mask for {image_path.name} in {mask_dir}; found {matches}")
    return matches[0]


def list_pairs(data_root: Path, split: str) -> List[Tuple[Path, Path]]:
    image_dir = data_root / split / "images"
    mask_dir = data_root / split / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Incomplete split layout: {image_dir} and {mask_dir}")
    images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")
    return [(p, find_mask(mask_dir, p)) for p in images]


def _surface(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    return np.logical_xor(mask, binary_erosion(mask, structure=structure, border_value=0))


def _hd_hd95(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    h, w = gt.shape
    side_length = float(max(h, w))
    if not gt.any():
        return (0.0, 0.0) if not pred.any() else (side_length, side_length)
    if not pred.any():
        return side_length, side_length
    ps, gs = _surface(pred.astype(bool)), _surface(gt.astype(bool))
    d_to_gt = distance_transform_edt(~gs)[ps]
    d_to_pred = distance_transform_edt(~ps)[gs]
    distances = np.concatenate([d_to_gt, d_to_pred]).astype(np.float64)
    return float(distances.max()), float(np.percentile(distances, 95))


def _cc(mask: np.ndarray) -> int:
    n, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return int(n - 1)


def compute_case(filename: str, pred: np.ndarray, gt: np.ndarray, inference_time_ms: float) -> Dict[str, object]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    gt_pixels, pred_pixels = int(gt.sum()), int(pred.sum())
    if gt_pixels == 0:
        return {
            "filename": filename, "metric_included": 0, "gt_pixels": 0,
            "pred_pixels": pred_pixels, **{k: float("nan") for k in METRIC_NAMES[:-1]},
            "inference_time_ms": float(inference_time_ms),
        }
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    dice = 2.0 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    hd, hd95 = _hd_hd95(pred, gt)
    gt_cc, pred_cc = _cc(gt), _cc(pred)
    cc_delta = gt_cc - pred_cc
    return {
        "filename": filename, "metric_included": 1,
        "gt_pixels": gt_pixels, "pred_pixels": pred_pixels,
        "dice": dice, "iou": iou, "hd": hd, "hd95": hd95,
        "precision": precision, "recall": recall,
        "gt_cc": gt_cc, "pred_cc": pred_cc, "cc_delta": cc_delta,
        "abs_cc_delta": abs(cc_delta), "inference_time_ms": float(inference_time_ms),
    }


def write_cases(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CASE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[Mapping[str, object]], split: str) -> Dict[str, object]:
    included = [r for r in rows if int(r["metric_included"]) == 1]
    summary: Dict[str, object] = {
        "split": split,
        "total_samples": len(rows),
        "gt_empty_excluded_count": len(rows) - len(included),
        "evaluated_gt_nonempty_count": len(included),
        "empty_prediction_count": sum(int(r["pred_pixels"]) == 0 for r in included),
    }
    summary["empty_prediction_rate"] = summary["empty_prediction_count"] / max(len(included), 1)
    for name in METRIC_NAMES:
        source = rows if name == "inference_time_ms" else included
        values = np.asarray([float(r[name]) for r in source], dtype=np.float64)
        values = values[np.isfinite(values)]
        summary[f"{name}_mean"] = float(values.mean()) if values.size else float("nan")
        summary[f"{name}_std"] = float(values.std(ddof=0)) if values.size else float("nan")
        summary[f"{name}_valid_count"] = int(values.size)
    return summary


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=True)


def overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)
    rgb[mask.astype(bool)] = 0.45 * rgb[mask.astype(bool)] + 0.55 * np.asarray(color, dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def save_ranked_figure(path: Path, ranked: Sequence[Mapping[str, object]], cache: Mapping[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], title: str) -> None:
    if not ranked:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(ranked), 3, figsize=(10, 3.2 * len(ranked)), squeeze=False)
    for row_idx, row in enumerate(ranked):
        name = str(row["filename"])
        image, gt, pred = cache[name]
        panels = [image, overlay(image, gt, (0, 255, 0)), overlay(image, pred, (255, 0, 0))]
        subtitles = [name, "GT (green)", f"Pred (red), Dice={float(row['dice']):.4f}"]
        for col, (panel, subtitle) in enumerate(zip(panels, subtitles)):
            axes[row_idx, col].imshow(panel, cmap="gray" if col == 0 else None)
            axes[row_idx, col].set_title(subtitle, fontsize=9)
            axes[row_idx, col].axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_top_bottom(output_dir: Path, split: str, rows: Sequence[Mapping[str, object]], cache: Mapping[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
    included = sorted((r for r in rows if int(r["metric_included"]) == 1), key=lambda r: float(r["dice"]))
    bottom = included[:5]
    top = list(reversed(included[-5:]))
    save_ranked_figure(output_dir / f"{split}_top5_dice.png", top, cache, f"{split}: Top-5 Dice")
    save_ranked_figure(output_dir / f"{split}_bottom5_dice.png", bottom, cache, f"{split}: Bottom-5 Dice")


def write_summary_csv(path: Path, model: str, size: int, summaries: Mapping[str, Mapping[str, object]]) -> None:
    fields = ["model", "size", "split", "Mean-dice", "Std-dice", "Mean-iou", "Std-iou", "Mean-HD", "Std-HD", "Mean-HD95", "Std-HD95", "Mean-Precision", "Std-Precision", "Mean-Recall", "Std-Recall", "Mean-Delta-CC", "Std-Delta-CC", "Mean-Abs-Delta-CC", "Std-Abs-Delta-CC", "Efficiency-ms-image", "N"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for split in ("train", "val", "test"):
            s = summaries[split]
            writer.writerow({
                "model": model, "size": size, "split": split,
                "Mean-dice": s["dice_mean"], "Std-dice": s["dice_std"],
                "Mean-iou": s["iou_mean"], "Std-iou": s["iou_std"],
                "Mean-HD": s["hd_mean"], "Std-HD": s["hd_std"],
                "Mean-HD95": s["hd95_mean"], "Std-HD95": s["hd95_std"],
                "Mean-Precision": s["precision_mean"], "Std-Precision": s["precision_std"],
                "Mean-Recall": s["recall_mean"], "Std-Recall": s["recall_std"],
                "Mean-Delta-CC": s["cc_delta_mean"], "Std-Delta-CC": s["cc_delta_std"],
                "Mean-Abs-Delta-CC": s["abs_cc_delta_mean"], "Std-Abs-Delta-CC": s["abs_cc_delta_std"],
                "Efficiency-ms-image": s["inference_time_ms_mean"], "N": s["evaluated_gt_nonempty_count"],
            })
