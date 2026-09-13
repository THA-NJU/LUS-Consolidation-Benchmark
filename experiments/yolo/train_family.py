#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train and evaluate the YOLO family for the LUS consolidation benchmark.

Supported checkpoints
---------------------
Instance segmentation (polygon labels; predicted instances are merged):
    yolo11s-seg.pt, yolo11m-seg.pt, yolo11l-seg.pt

Semantic segmentation (PNG class masks):
    yolo26m-sem.pt, yolo26l-sem.pt

The script intentionally keeps these two output mechanisms separate.  For
both, every epoch is ranked by the foreground per-image Mean Dice on the
native-resolution validation split.  This Mean Dice is also the Ultralytics
``fitness`` value, so ``best.pt`` and patience=15 early stopping use the same
benchmark metric rather than mask mAP/mIoU.

After training finishes (early stopping or epoch 600), the selected checkpoint
is evaluated once on train -> val -> test.  It writes *_cases.csv,
*_summary.json, top/bottom figures, summary.csv and evaluation_settings.json.

This is a single-GPU script.  Put it next to the local pretrained checkpoint or
pass --weights.  It never downloads a missing weight unless --allow-download
is explicitly supplied.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import shutil
import time
import warnings
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


SUPPORTED_MODELS = {
    "yolo11s-seg": "instance",
    "yolo11m-seg": "instance",
    "yolo11l-seg": "instance",
    "yolo26m-sem": "semantic",
    "yolo26l-sem": "semantic",
}

# Consumer RTX 4060 Ti 16 GB starting points. Ultralytics uses nbs below to
# accumulate gradients to the benchmark's nominal/effective batch size.
MICRO_BATCH = {
    ("yolo11s-seg", 512): 4,
    ("yolo11s-seg", 224): 16,
    ("yolo11m-seg", 512): 2,
    ("yolo11m-seg", 224): 8,
    ("yolo11l-seg", 512): 1,
    ("yolo11l-seg", 224): 4,
    ("yolo26m-sem", 512): 2,
    ("yolo26m-sem", 224): 8,
    ("yolo26l-sem", 512): 1,
    ("yolo26l-sem", 224): 4,
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")
CASE_FIELDS = [
    "filename", "metric_included", "gt_pixels", "pred_pixels",
    "dice", "iou", "hd", "hd95", "precision", "recall",
    "gt_cc", "pred_cc", "cc_delta", "abs_cc_delta", "inference_time_ms",
]

# This is also the exact summary.csv order requested for the benchmark.
SUMMARY_COLUMNS = [
    "Model", "Size", "Split",
    "Mean-dice", "Std-dice",
    "Mean-IoU", "Std-IoU",
    "Mean-Recall", "Std-Recall",
    "Mean-Precision", "Std-Precision",
    "HD-count-delta", "empty_prediction_count",
    "Mean-HD95", "Std-HD95",
    "Mean-HD", "Std-HD",
    "Mean-Delta-CC", "Std-Delta-CC",
    "Mean-Abs-Delta-CC", "Std-Abs-Delta-CC",
    "Mean-Efficiency-ms/image", "Std-Efficiency-ms/image",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="YOLO11-seg/YOLO26-sem native 224/512 Mean-Dice benchmark trainer.",
    )
    parser.add_argument("--model", required=True, choices=tuple(SUPPORTED_MODELS))
    parser.add_argument("--size", required=True, type=int, choices=(224, 512))
    parser.add_argument("--weights", type=Path, default=None,
                        help="Local pretrained .pt; defaults to ./<model>.pt")
    parser.add_argument("--allow-download", action="store_true",
                        help="Allow Ultralytics to download the official checkpoint if local weights are absent")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Defaults to Size_512 or Size_224_filtered under --data-base-root")
    parser.add_argument("--data-base-root", type=Path,
                        default=Path("./datasets"))
    parser.add_argument("--cache-root", type=Path, default=Path("./_yolo_family_cache"))
    parser.add_argument("--run-root", type=Path, default=Path("./sota_yolo_family_runs"))
    # The documented launch directory is <repository-root>, whose existing
    # benchmark summaries live under ./Evaluation.
    parser.add_argument("--evaluation-root", type=Path, default=Path("./Evaluation/YOLO"))
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--labels", type=int, nargs="+", default=[1],
                        help="Pixel values treated as consolidation foreground")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override the safe per-model micro-batch")
    parser.add_argument("--effective-batch-size", type=int, default=None,
                        help="Defaults to 4 for 512 and 16 for 224 (Ultralytics nbs)")
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instance-conf", type=float, default=0.25,
                        help="Fixed confidence for YOLO11 instance predictions in both val selection and final eval")
    parser.add_argument("--instance-iou", type=float, default=0.70,
                        help="Fixed NMS IoU for YOLO11 instance predictions")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def set_fast_reproducible_seed(seed: int) -> None:
    """Seed stochastic sources without forcing slow deterministic CUDA kernels."""
    random.seed(seed)
    try:
        import numpy as np
        import torch
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=True)


def find_mask(mask_dir: Path, image_path: Path) -> Path:
    candidates = [
        mask_dir / image_path.name,
        mask_dir / f"{image_path.stem}_mask{image_path.suffix}",
        mask_dir / f"{image_path.stem}.png",
        mask_dir / f"{image_path.stem}_mask.png",
    ]
    found: List[Path] = []
    for candidate in candidates:
        if candidate.is_file() and candidate not in found:
            found.append(candidate)
    if len(found) != 1:
        raise FileNotFoundError(
            f"Expected exactly one mask for {image_path.name} in {mask_dir}; found {found}"
        )
    return found[0]


def read_gray(path: Path):
    import cv2
    import numpy as np
    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise RuntimeError(f"Cannot read image: {path}")
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    if array.dtype == np.uint16:
        array = np.rint(array.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        low, high = float(array.min()), float(array.max())
        if high <= low:
            array = np.zeros(array.shape, dtype=np.uint8)
        else:
            array = np.rint((array.astype(np.float32) - low) / (high - low) * 255.0).astype(np.uint8)
    return array


def read_binary_mask(path: Path, labels: Sequence[int]):
    import cv2
    import numpy as np
    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise RuntimeError(f"Cannot read mask: {path}")
    if array.ndim == 3:
        array = array[..., 0]
    return np.isin(array, np.asarray(labels)).astype(np.uint8)


def list_pairs(data_root: Path, split: str, size: int) -> List[Tuple[Path, Path]]:
    image_dir = data_root / split / "images"
    mask_dir = data_root / split / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Incomplete split: {image_dir} and {mask_dir}")
    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")
    pairs = []
    for image_path in images:
        mask_path = find_mask(mask_dir, image_path)
        image = read_gray(image_path)
        mask = read_binary_mask(mask_path, (1,))  # shape check only; real labels checked below
        if image.shape != (size, size) or mask.shape != (size, size):
            raise ValueError(
                f"Expected {size}x{size}: image={image.shape}, mask={mask.shape}, file={image_path}"
            )
        pairs.append((image_path, mask_path))
    return pairs


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)


def mask_to_polygons(mask) -> List[List[Tuple[float, float]]]:
    """Convert each external connected foreground contour to a YOLO polygon."""
    import cv2
    height, width = mask.shape
    contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygons: List[List[Tuple[float, float]]] = []
    for contour in contours:
        if contour.shape[0] < 3 or cv2.contourArea(contour) <= 0:
            continue
        # Keep boundary detail; simplify only extremely dense contours.
        if contour.shape[0] > 1000:
            contour = cv2.approxPolyDP(contour, epsilon=0.25, closed=True)
        points = contour.reshape(-1, 2)
        if points.shape[0] < 3:
            continue
        polygon = [
            (min(max(float(x) / width, 0.0), 1.0), min(max(float(y) / height, 0.0), 1.0))
            for x, y in points
        ]
        polygons.append(polygon)
    return polygons


def polygon_text(polygons: Sequence[Sequence[Tuple[float, float]]]) -> str:
    lines = []
    for polygon in polygons:
        coordinates = " ".join(f"{coordinate:.8f}" for point in polygon for coordinate in point)
        lines.append(f"0 {coordinates}")
    return "\n".join(lines) + ("\n" if lines else "")


def rasterize_polygons(polygons: Sequence[Sequence[Tuple[float, float]]], size: int):
    import cv2
    import numpy as np
    canvas = np.zeros((size, size), dtype=np.uint8)
    for polygon in polygons:
        points = np.asarray([
            [min(int(round(x * size)), size - 1), min(int(round(y * size)), size - 1)]
            for x, y in polygon
        ], dtype=np.int32)
        if len(points) >= 3:
            cv2.fillPoly(canvas, [points], 1)
    return canvas


def prepare_cache(
    task_kind: str,
    model_name: str,
    size: int,
    data_root: Path,
    cache_base: Path,
    pairs_by_split: Mapping[str, Sequence[Tuple[Path, Path]]],
    labels: Sequence[int],
    rebuild: bool,
) -> Tuple[Path, Dict[str, object]]:
    import cv2
    import numpy as np
    import yaml
    from tqdm import tqdm

    cache_dir = cache_base / f"size{size}" / task_kind
    manifest_path = cache_dir / "manifest.json"
    signature = {
        "version": 1,
        "task_kind": task_kind,
        "source": str(data_root.resolve()),
        "size": size,
        "labels": list(labels),
        "no_sampling_or_balancing": True,
    }
    if rebuild and cache_dir.exists():
        shutil.rmtree(cache_dir)
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        yaml_path = cache_dir / "dataset.yaml"
        if manifest.get("signature") == signature and yaml_path.is_file():
            log(f"Reusing {task_kind} cache: {cache_dir}")
            return yaml_path, manifest
        shutil.rmtree(cache_dir)

    split_stats: Dict[str, object] = {}
    for split, pairs in pairs_by_split.items():
        image_out = cache_dir / "images" / split
        target_out = cache_dir / ("masks" if task_kind == "semantic" else "labels") / split
        positive = 0
        conversion_dice: List[float] = []
        for image_path, mask_path in tqdm(pairs, desc=f"Cache {model_name} {size} {split}"):
            link_or_copy(image_path, image_out / image_path.name)
            binary = read_binary_mask(mask_path, labels)
            positive += int(binary.any())
            if task_kind == "semantic":
                destination = target_out / f"{image_path.stem}.png"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists() and not cv2.imwrite(str(destination), binary.astype(np.uint8)):
                    raise RuntimeError(f"Cannot write semantic mask: {destination}")
            else:
                polygons = mask_to_polygons(binary)
                destination = target_out / f"{image_path.stem}.txt"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(polygon_text(polygons), encoding="utf-8")
                if binary.any():
                    reconstructed = rasterize_polygons(polygons, size)
                    intersection = int(np.logical_and(binary, reconstructed).sum())
                    conversion_dice.append(
                        2.0 * intersection / max(int(binary.sum()) + int(reconstructed.sum()), 1)
                    )
        stats: Dict[str, object] = {
            "samples": len(pairs), "positive": positive, "negative": len(pairs) - positive,
        }
        if conversion_dice:
            stats.update({
                "polygon_conversion_dice_mean": float(np.mean(conversion_dice)),
                "polygon_conversion_dice_min": float(np.min(conversion_dice)),
            })
            if float(np.min(conversion_dice)) < 0.95:
                warnings.warn(
                    f"{split}: minimum polygon conversion Dice is {min(conversion_dice):.4f}; "
                    "inspect masks with holes or very thin components."
                )
        split_stats[split] = stats

    if task_kind == "semantic":
        dataset_yaml = {
            "path": str(cache_dir.resolve()),
            "train": "images/train", "val": "images/val", "test": "images/test",
            "masks_dir": "masks", "names": {0: "background", 1: "consolidation"},
        }
    else:
        dataset_yaml = {
            "path": str(cache_dir.resolve()),
            "train": "images/train", "val": "images/val", "test": "images/test",
            "names": {0: "consolidation"},
        }
    yaml_path = cache_dir / "dataset.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dataset_yaml, handle, sort_keys=False, allow_unicode=True)
    manifest = {"signature": signature, "split_stats": split_stats, "dataset_yaml": dataset_yaml}
    write_json(manifest_path, manifest)
    return yaml_path, manifest


def _dice_from_unions(pred_union, gt_union) -> float:
    import torch
    pred_union = pred_union.bool()
    gt_union = gt_union.bool()
    intersection = torch.logical_and(pred_union, gt_union).sum(dtype=torch.float64)
    denominator = pred_union.sum(dtype=torch.float64) + gt_union.sum(dtype=torch.float64)
    return float((2.0 * intersection / denominator.clamp_min(1.0)).detach().cpu())


def build_instance_trainer():
    """Build a validator/trainer whose fitness is merged-mask validation Dice."""
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from ultralytics.models.yolo.segment import SegmentationTrainer, SegmentationValidator

    class BenchmarkInstanceDiceValidator(SegmentationValidator):
        def init_metrics(self, model) -> None:
            super().init_metrics(model)
            self.benchmark_dice: List[float] = []
            self.benchmark_empty_predictions = 0

        def _process_batch(self, preds, batch):
            native_stats = super()._process_batch(preds, batch)
            gt_masks = batch.get("masks")
            if gt_masks is None:
                return native_stats
            if gt_masks.ndim == 2:
                gt_masks = gt_masks[None]
            gt_union = gt_masks.bool().any(dim=0) if gt_masks.shape[0] else torch.zeros(
                gt_masks.shape[-2:], dtype=torch.bool, device=gt_masks.device
            )
            if not bool(gt_union.any()):
                return native_stats  # same benchmark rule: exclude empty-GT cases

            pred_masks = preds.get("masks")
            pred_cls = preds.get("cls")
            if pred_masks is None or pred_masks.shape[0] == 0:
                pred_union = torch.zeros_like(gt_union)
            else:
                if pred_masks.ndim == 2:
                    pred_masks = pred_masks[None]
                if pred_cls is not None and pred_cls.numel() == pred_masks.shape[0]:
                    pred_masks = pred_masks[pred_cls.to(torch.int64) == 0]
                if pred_masks.shape[0] == 0:
                    pred_union = torch.zeros_like(gt_union)
                else:
                    if tuple(pred_masks.shape[-2:]) != tuple(gt_union.shape[-2:]):
                        pred_masks = functional.interpolate(
                            pred_masks[:, None].float(), size=gt_union.shape[-2:], mode="nearest"
                        )[:, 0]
                    pred_union = (pred_masks > 0.5).any(dim=0)
            self.benchmark_empty_predictions += int(not bool(pred_union.any()))
            self.benchmark_dice.append(_dice_from_unions(pred_union, gt_union))
            return native_stats

        def get_stats(self) -> Dict[str, object]:
            stats = super().get_stats()
            values = np.asarray(self.benchmark_dice, dtype=np.float64)
            if values.size == 0:
                raise RuntimeError("Validation produced no non-empty-GT cases for Mean Dice")
            mean_dice = float(values.mean())
            stats["benchmark/val_mean_dice"] = mean_dice
            stats["benchmark/val_std_dice"] = float(values.std(ddof=0))
            stats["benchmark/val_dice_n"] = int(values.size)
            stats["benchmark/val_empty_prediction_count"] = int(self.benchmark_empty_predictions)
            stats["fitness"] = mean_dice
            return stats

    class BenchmarkInstanceDiceTrainer(SegmentationTrainer):
        def get_validator(self):
            return BenchmarkInstanceDiceValidator(
                self.test_loader, save_dir=self.save_dir,
                args=copy(self.args), _callbacks=self.callbacks,
            )

    return BenchmarkInstanceDiceTrainer


def build_semantic_trainer():
    """Build a validator/trainer whose fitness is foreground semantic Dice."""
    import numpy as np
    import torch
    try:
        from ultralytics.models.yolo.semantic import (
            SemanticSegmentationTrainer, SemanticSegmentationValidator,
        )
    except ImportError as error:
        raise RuntimeError(
            "This Ultralytics installation has no YOLO semantic module. Install the same "
            "current Ultralytics release that can load yolo26m-sem.pt/yolo26l-sem.pt."
        ) from error

    class BenchmarkSemanticDiceValidator(SemanticSegmentationValidator):
        def init_metrics(self, model) -> None:
            super().init_metrics(model)
            self.benchmark_dice: List[float] = []
            self.benchmark_empty_predictions = 0

        def update_metrics(self, preds, batch) -> None:
            super().update_metrics(preds, batch)
            targets = batch["semantic_mask"]
            if targets.ndim == 4 and targets.shape[1] == 1:
                targets = targets[:, 0]
            if preds.ndim == 4 and preds.shape[1] == 1:
                preds = preds[:, 0]
            if tuple(preds.shape) != tuple(targets.shape):
                raise RuntimeError(f"Semantic prediction/target mismatch: {preds.shape} != {targets.shape}")
            pred_fg = preds == 1
            target_fg = targets == 1
            for pred_one, target_one in zip(pred_fg, target_fg):
                if not bool(target_one.any()):
                    continue
                self.benchmark_empty_predictions += int(not bool(pred_one.any()))
                self.benchmark_dice.append(_dice_from_unions(pred_one, target_one))

        def get_stats(self) -> Dict[str, object]:
            stats = super().get_stats()
            values = np.asarray(self.benchmark_dice, dtype=np.float64)
            if values.size == 0:
                raise RuntimeError("Validation produced no non-empty-GT cases for Mean Dice")
            mean_dice = float(values.mean())
            stats["benchmark/val_mean_dice"] = mean_dice
            stats["benchmark/val_std_dice"] = float(values.std(ddof=0))
            stats["benchmark/val_dice_n"] = int(values.size)
            stats["benchmark/val_empty_prediction_count"] = int(self.benchmark_empty_predictions)
            stats["fitness"] = mean_dice
            return stats

    class BenchmarkSemanticDiceTrainer(SemanticSegmentationTrainer):
        def get_validator(self):
            return BenchmarkSemanticDiceValidator(
                self.test_loader, save_dir=self.save_dir,
                args=copy(self.args), _callbacks=self.callbacks,
            )

    return BenchmarkSemanticDiceTrainer


def resolve_weights(args: argparse.Namespace) -> str:
    candidate = args.weights or Path(f"./{args.model}.pt")
    candidate = candidate.expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    if args.allow_download:
        log(f"Local checkpoint absent; allowing official download of {args.model}.pt")
        return f"{args.model}.pt"
    raise FileNotFoundError(
        f"Pretrained checkpoint not found: {candidate.resolve()}\n"
        f"Put {args.model}.pt in the current directory, pass --weights, or explicitly use --allow-download."
    )


def semantic_prediction(result, size: int):
    import cv2
    import numpy as np
    import torch
    semantic_mask = getattr(result, "semantic_mask", None)
    if semantic_mask is None:
        raise RuntimeError("result.semantic_mask is unavailable; check the Ultralytics/YOLO26 semantic version")
    data = semantic_mask.data
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    class_map = np.squeeze(np.asarray(data))
    if class_map.ndim != 2:
        raise ValueError(f"Unexpected semantic class map: {class_map.shape}")
    if class_map.shape != (size, size):
        class_map = cv2.resize(class_map.astype(np.int32), (size, size), interpolation=cv2.INTER_NEAREST)
    return (class_map == 1).astype(np.uint8)


def instance_prediction(result, size: int, conf: float):
    import cv2
    import numpy as np
    import torch
    output = np.zeros((size, size), dtype=np.uint8)
    masks_obj = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks_obj is None or masks_obj.data is None or len(masks_obj.data) == 0:
        return output
    masks = masks_obj.data
    if torch.is_tensor(masks):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks)
    keep = np.ones(masks.shape[0], dtype=bool)
    if boxes is not None and len(boxes) == masks.shape[0]:
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        confidences = boxes.conf.detach().cpu().numpy()
        keep = (classes == 0) & (confidences >= conf)
    selected = masks[keep]
    if selected.size:
        merged = np.any(selected > 0.5, axis=0).astype(np.uint8)
        if merged.shape != (size, size):
            merged = cv2.resize(merged, (size, size), interpolation=cv2.INTER_NEAREST)
        output = merged
    return output


def _surface(mask):
    import numpy as np
    from scipy.ndimage import binary_erosion
    structure = np.ones((3, 3), dtype=bool)
    return np.logical_xor(mask, binary_erosion(mask, structure=structure, border_value=0))


def hd_hd95(pred, gt) -> Tuple[float, float]:
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    height, width = gt.shape
    side_length = float(max(height, width))
    if not gt.any():
        return ((0.0, 0.0) if not pred.any() else (side_length, side_length))
    if not pred.any():
        return side_length, side_length
    pred_surface, gt_surface = _surface(pred.astype(bool)), _surface(gt.astype(bool))
    distances = np.concatenate([
        distance_transform_edt(~gt_surface)[pred_surface],
        distance_transform_edt(~pred_surface)[gt_surface],
    ]).astype(np.float64)
    return float(distances.max()), float(np.percentile(distances, 95))


def connected_components(mask) -> int:
    import cv2
    count, _ = cv2.connectedComponents(mask.astype("uint8"), connectivity=8)
    return int(count - 1)


def compute_case(filename: str, pred, gt, inference_ms: float) -> Dict[str, object]:
    import numpy as np
    pred, gt = pred.astype(bool), gt.astype(bool)
    gt_pixels, pred_pixels = int(gt.sum()), int(pred.sum())
    if gt_pixels == 0:
        return {
            "filename": filename, "metric_included": 0,
            "gt_pixels": 0, "pred_pixels": pred_pixels,
            "dice": float("nan"), "iou": float("nan"),
            "hd": float("nan"), "hd95": float("nan"),
            "precision": float("nan"), "recall": float("nan"),
            "gt_cc": float("nan"), "pred_cc": float("nan"),
            "cc_delta": float("nan"), "abs_cc_delta": float("nan"),
            "inference_time_ms": float(inference_ms),
        }
    true_positive = int(np.logical_and(pred, gt).sum())
    false_positive = int(np.logical_and(pred, ~gt).sum())
    false_negative = int(np.logical_and(~pred, gt).sum())
    dice = 2.0 * true_positive / max(2 * true_positive + false_positive + false_negative, 1)
    iou = true_positive / max(true_positive + false_positive + false_negative, 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    hd, hd95 = hd_hd95(pred, gt)
    gt_cc, pred_cc = connected_components(gt), connected_components(pred)
    delta = gt_cc - pred_cc
    return {
        "filename": filename, "metric_included": 1,
        "gt_pixels": gt_pixels, "pred_pixels": pred_pixels,
        "dice": dice, "iou": iou, "hd": hd, "hd95": hd95,
        "precision": precision, "recall": recall,
        "gt_cc": gt_cc, "pred_cc": pred_cc, "cc_delta": delta,
        "abs_cc_delta": abs(delta), "inference_time_ms": float(inference_ms),
    }


def summarize(rows: Sequence[Mapping[str, object]], split: str) -> Dict[str, object]:
    import numpy as np
    included = [row for row in rows if int(row["metric_included"]) == 1]

    def stats(name: str, source: Sequence[Mapping[str, object]]) -> Tuple[float, float, int]:
        values = np.asarray([float(row[name]) for row in source], dtype=np.float64)
        values = values[np.isfinite(values)]
        if not values.size:
            return float("nan"), float("nan"), 0
        return float(values.mean()), float(values.std(ddof=0)), int(values.size)

    output: Dict[str, object] = {
        "split": split,
        "total_samples": len(rows),
        "gt_empty_excluded_count": len(rows) - len(included),
        "evaluated_gt_nonempty_count": len(included),
    }
    # Insert metrics in the benchmark's requested presentation order.
    for metric in ("dice", "iou", "recall", "precision"):
        mean, std, count = stats(metric, included)
        output[f"{metric}_mean"] = mean
        output[f"{metric}_std"] = std
        output[f"{metric}_valid_count"] = count
    output["empty_prediction_count"] = sum(int(row["pred_pixels"]) == 0 for row in included)
    output["empty_prediction_rate"] = output["empty_prediction_count"] / max(len(included), 1)
    for metric in ("hd95", "hd", "cc_delta", "abs_cc_delta"):
        mean, std, count = stats(metric, included)
        output[f"{metric}_mean"] = mean
        output[f"{metric}_std"] = std
        output[f"{metric}_valid_count"] = count
    mean, std, count = stats("inference_time_ms", rows)
    output["inference_time_ms_mean"] = mean
    output["inference_time_ms_std"] = std
    output["inference_time_ms_valid_count"] = count
    return output


def write_cases(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CASE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summary_row(model_name: str, size: int, split: str, summary: Mapping[str, object]) -> Dict[str, object]:
    return {
        "Model": model_name, "Size": size, "Split": split,
        "Mean-dice": summary["dice_mean"], "Std-dice": summary["dice_std"],
        "Mean-IoU": summary["iou_mean"], "Std-IoU": summary["iou_std"],
        "Mean-Recall": summary["recall_mean"], "Std-Recall": summary["recall_std"],
        "Mean-Precision": summary["precision_mean"], "Std-Precision": summary["precision_std"],
        "HD-count-delta": summary["evaluated_gt_nonempty_count"] - summary["hd_valid_count"],
        "empty_prediction_count": summary["empty_prediction_count"],
        "Mean-HD95": summary["hd95_mean"], "Std-HD95": summary["hd95_std"],
        "Mean-HD": summary["hd_mean"], "Std-HD": summary["hd_std"],
        "Mean-Delta-CC": summary["cc_delta_mean"], "Std-Delta-CC": summary["cc_delta_std"],
        "Mean-Abs-Delta-CC": summary["abs_cc_delta_mean"],
        "Std-Abs-Delta-CC": summary["abs_cc_delta_std"],
        "Mean-Efficiency-ms/image": summary["inference_time_ms_mean"],
        "Std-Efficiency-ms/image": summary["inference_time_ms_std"],
    }


def save_summary_csv(path: Path, model_name: str, size: int, summaries: Mapping[str, Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for split in SPLITS:
            writer.writerow(summary_row(model_name, size, split, summaries[split]))


def overlay(image, mask, color: Tuple[int, int, int]):
    import cv2
    import numpy as np
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)
    chosen = mask.astype(bool)
    rgb[chosen] = 0.45 * rgb[chosen] + 0.55 * np.asarray(color, dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def save_ranked_figure(path: Path, entries, title: str) -> None:
    if not entries:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(len(entries), 3, figsize=(10, 3.2 * len(entries)), squeeze=False)
    for index, (_, row, image, gt, pred) in enumerate(entries):
        panels = [image, overlay(image, gt, (0, 255, 0)), overlay(image, pred, (255, 0, 0))]
        subtitles = [row["filename"], "GT (green)", f"Pred (red), Dice={float(row['dice']):.4f}"]
        for column, (panel, subtitle) in enumerate(zip(panels, subtitles)):
            axes[index, column].imshow(panel, cmap="gray" if column == 0 else None)
            axes[index, column].set_title(subtitle, fontsize=9)
            axes[index, column].axis("off")
    figure.suptitle(title, fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def evaluate_split(
    model,
    task_kind: str,
    pairs: Sequence[Tuple[Path, Path]],
    split: str,
    size: int,
    labels: Sequence[int],
    batch_size: int,
    device: str,
    instance_conf: float,
    instance_iou: float,
    output_dir: Path,
    save_figures: bool,
) -> Dict[str, object]:
    import numpy as np
    import torch
    from tqdm import tqdm

    if not pairs:
        raise RuntimeError(f"Empty split: {split}")
    predict_kwargs = dict(imgsz=size, device=device, verbose=False, stream=False)
    if task_kind == "instance":
        predict_kwargs.update(conf=instance_conf, iou=instance_iou, retina_masks=True)
    for _ in range(2):
        model.predict(source=str(pairs[0][0]), batch=1, **predict_kwargs)

    rows: List[Dict[str, object]] = []
    batch_rows: List[Dict[str, object]] = []
    top: List[Tuple[float, Mapping[str, object], object, object, object]] = []
    bottom: List[Tuple[float, Mapping[str, object], object, object, object]] = []
    for batch_index, start_index in enumerate(
        tqdm(range(0, len(pairs), batch_size), desc=f"{split} evaluation"), 1
    ):
        chunk = pairs[start_index:start_index + batch_size]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        results = model.predict(
            source=[str(image_path) for image_path, _ in chunk],
            batch=min(batch_size, len(chunk)), **predict_kwargs,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - start) * 1000.0
        if len(results) != len(chunk):
            raise RuntimeError(f"Prediction count mismatch: {len(results)} != {len(chunk)}")
        reported = [float(getattr(result, "speed", {}).get("inference", float("nan"))) for result in results]
        finite = [value for value in reported if np.isfinite(value)]
        fallback = wall_ms / len(chunk)
        per_case_times = [value if np.isfinite(value) else fallback for value in reported]
        batch_rows.append({
            "batch_index": batch_index, "batch_size": len(chunk),
            "predict_wall_time_ms": wall_ms,
            "reported_inference_mean_ms_per_image": float(np.mean(finite)) if finite else fallback,
        })
        for (image_path, mask_path), result, inference_ms in zip(chunk, results, per_case_times):
            if task_kind == "semantic":
                pred = semantic_prediction(result, size)
            else:
                pred = instance_prediction(result, size, instance_conf)
            gt = read_binary_mask(mask_path, labels)
            row = compute_case(image_path.name, pred, gt, inference_ms)
            rows.append(row)
            if save_figures and int(row["metric_included"]) == 1:
                payload = (float(row["dice"]), row, read_gray(image_path), gt, pred)
                top.append(payload); top.sort(key=lambda item: item[0], reverse=True); del top[5:]
                bottom.append(payload); bottom.sort(key=lambda item: item[0]); del bottom[5:]

    write_cases(output_dir / f"{split}_cases.csv", rows)
    with (output_dir / f"{split}_efficiency_batches.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fields = ["batch_index", "batch_size", "predict_wall_time_ms", "reported_inference_mean_ms_per_image"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(batch_rows)
    summary = summarize(rows, split)
    write_json(output_dir / f"{split}_summary.json", summary)
    if save_figures:
        save_ranked_figure(output_dir / f"{split}_top5_dice.png", top, f"{split}: Top-5 Dice")
        save_ranked_figure(output_dir / f"{split}_bottom5_dice.png", bottom, f"{split}: Bottom-5 Dice")
    return summary


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    set_fast_reproducible_seed(args.seed)

    import torch
    import ultralytics
    from ultralytics import YOLO

    task_kind = SUPPORTED_MODELS[args.model]
    data_root = args.data_root or (
        args.data_base_root / ("Size_224_filtered" if args.size == 224 else "Size_512")
    )
    data_root = data_root.expanduser().resolve()
    micro_batch = args.batch_size or MICRO_BATCH[(args.model, args.size)]
    effective_batch = args.effective_batch_size or (16 if args.size == 224 else 4)
    eval_batch = args.eval_batch_size or micro_batch
    weights = resolve_weights(args)

    log(f"Model={args.model}, task={task_kind}, size={args.size}")
    log(f"micro_batch={micro_batch}, effective_batch(nbs)={effective_batch}, eval_batch={eval_batch}")
    pairs_by_split = {
        split: list_pairs(data_root, split, args.size) for split in SPLITS
    }
    for split, pairs in pairs_by_split.items():
        positives = sum(int(read_binary_mask(mask, args.labels).any()) for _, mask in pairs)
        log(f"[{split}] samples={len(pairs)}, positive={positives}, negative={len(pairs)-positives}")

    yaml_path, cache_manifest = prepare_cache(
        task_kind, args.model, args.size, data_root, args.cache_root.expanduser().resolve(),
        pairs_by_split, args.labels, args.rebuild_cache,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_root.expanduser().resolve() / args.model / f"size{args.size}" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    evaluation_dir = args.evaluation_root.expanduser().resolve() / str(args.size) / args.model.replace("-", "_")
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    preflight = YOLO(weights)
    total_parameters = int(sum(parameter.numel() for parameter in preflight.model.parameters()))
    del preflight
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    run_config = {
        "protocol_version": "YOLO_FAMILY_MEAN_DICE_EVAL_FINAL_V1",
        "model": args.model, "task_kind": task_kind, "size": args.size,
        "weights": weights, "data_root": str(data_root), "dataset_yaml": str(yaml_path),
        "run_dir": str(run_dir), "evaluation_dir": str(evaluation_dir),
        "ultralytics_version": ultralytics.__version__, "torch_version": torch.__version__,
        "parameters_total": total_parameters,
        "epochs": args.epochs, "patience": args.patience,
        "warmup_epochs": args.warmup_epochs, "seed": args.seed,
        "optimizer": "AdamW", "learning_rate": args.lr, "weight_decay": args.weight_decay,
        "micro_batch_size": micro_batch, "effective_batch_size_nbs": effective_batch,
        "selection_metric": "native-resolution validation foreground per-image Mean Dice",
        "fixed_semantic_rule": "two-class argmax (foreground probability >= 0.5 equivalent)",
        "fixed_instance_conf": args.instance_conf if task_kind == "instance" else None,
        "fixed_instance_nms_iou": args.instance_iou if task_kind == "instance" else None,
        "gt_empty_metric_rule": "excluded",
        "sampling": "full traversal, shuffle train only, no replacement, no balancing",
        "cache_manifest": cache_manifest,
    }
    write_json(run_dir / "run_config.json", run_config)

    trainer_class = build_semantic_trainer() if task_kind == "semantic" else build_instance_trainer()
    model = YOLO(weights)
    train_kwargs: Dict[str, object] = {
        "task": "semantic" if task_kind == "semantic" else "segment",
        "data": str(yaml_path), "epochs": args.epochs, "imgsz": args.size,
        "batch": micro_batch, "nbs": effective_batch, "device": args.device,
        "workers": args.workers, "project": str(run_dir), "name": "ultralytics_train",
        "exist_ok": True, "pretrained": True, "trainer": trainer_class,
        "optimizer": "AdamW", "lr0": args.lr, "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs, "patience": args.patience,
        "val": True, "save": True, "save_period": -1, "cache": False,
        "amp": not args.no_amp, "seed": args.seed,
        # Seeded but fast, matching the other benchmark models; not bitwise deterministic.
        "deterministic": False,
        "rect": False, "multi_scale": 0.0, "cos_lr": True,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.08,
        "degrees": 0.0, "translate": 0.03, "scale": 0.08,
        "shear": 0.0, "perspective": 0.0,
        "flipud": 0.0, "fliplr": 0.50,
        "mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0,
        "plots": True, "verbose": True,
    }
    if task_kind == "instance":
        train_kwargs.update(conf=args.instance_conf, iou=args.instance_iou, overlap_mask=True)
    model.train(**train_kwargs)

    trainer = getattr(model, "trainer", None)
    train_output = (
        Path(trainer.save_dir) if trainer is not None and getattr(trainer, "save_dir", None)
        else run_dir / "ultralytics_train"
    )
    best_checkpoint = train_output / "weights" / "best.pt"
    if not best_checkpoint.is_file():
        raise FileNotFoundError(f"Mean-Dice-selected best.pt not found: {best_checkpoint}")
    selected_checkpoint = run_dir / f"best_by_val_mean_dice_{args.model}_size{args.size}.pt"
    shutil.copy2(best_checkpoint, selected_checkpoint)
    completed_epochs = int(getattr(trainer, "epoch", -1)) + 1 if trainer is not None else None
    stopped_early = completed_epochs is not None and completed_epochs < args.epochs
    write_json(run_dir / "training_completion.json", {
        "completed_epochs": completed_epochs, "configured_epochs": args.epochs,
        "stopped_early": stopped_early, "patience": args.patience,
        "best_checkpoint": str(selected_checkpoint),
        "best_fitness_val_mean_dice": float(getattr(trainer, "best_fitness", float("nan"))) if trainer else None,
    })
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # One and only one post-training evaluation pass over train -> val -> test.
    selected_model = YOLO(str(selected_checkpoint))
    summaries: Dict[str, Mapping[str, object]] = {}
    for split in SPLITS:
        summaries[split] = evaluate_split(
            selected_model, task_kind, pairs_by_split[split], split, args.size,
            args.labels, eval_batch, args.device, args.instance_conf, args.instance_iou,
            evaluation_dir, not args.skip_figures,
        )
    save_summary_csv(evaluation_dir / "summary.csv", args.model, args.size, summaries)
    write_json(evaluation_dir / "evaluation_settings.json", {
        **run_config,
        "checkpoint": str(selected_checkpoint),
        "evaluation_order": list(SPLITS),
        "summary_csv_column_order": SUMMARY_COLUMNS,
        "prediction_rule": (
            "semantic class map == 1"
            if task_kind == "semantic"
            else f"merge all class-0 instance masks after conf>={args.instance_conf}, NMS IoU={args.instance_iou}"
        ),
        "cc_delta": "gt_cc - pred_cc", "abs_cc_delta": "abs(gt_cc - pred_cc)",
        "connected_components": "8-neighborhood", "hd_units": "pixels",
        "empty_prediction_hd_penalty": "track side length (512 or 224 pixels)",
        "efficiency_scope": "Ultralytics result.speed['inference']; two warmups and metric time excluded",
    })
    write_json(run_dir / "final_evaluation_locations.json", {
        "checkpoint": str(selected_checkpoint), "evaluation_dir": str(evaluation_dir),
        "train_summary": str(evaluation_dir / "train_summary.json"),
        "val_summary": str(evaluation_dir / "val_summary.json"),
        "test_summary": str(evaluation_dir / "test_summary.json"),
        "summary_csv": str(evaluation_dir / "summary.csv"),
    })
    log(f"Finished training and train/val/test evaluation: {args.model} size={args.size}")
    log(f"Best checkpoint: {selected_checkpoint}")
    log(f"Evaluation: {evaluation_dir}")


if __name__ == "__main__":
    main()
