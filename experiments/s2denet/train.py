#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train and evaluate the official S2DENet on the LUS consolidation benchmark.

The upstream repository currently publishes the model implementation only. This
script supplies the benchmark-specific dataset, optimization, early stopping,
checkpointing, resume support, and train/val/test evaluation pipeline.

Protocol defaults:
  * native 512 or native 224 inputs (two independent tracks)
  * full split traversal; no resampling or class balancing
  * AdamW, lr=1e-4, weight_decay=1e-4
  * 600 epochs, 10-epoch linear warm-up, cosine decay, patience=15
  * effective batch: 4 (512) or 16 (224)
  * checkpoint selection: validation per-case mean Dice at threshold 0.5
  * masks in prepared patch datasets: 0=background, 1=consolidation
  * final evaluation order: train -> val -> test

The official model already applies sigmoid to both mask and edge outputs, so
this script deliberately uses probability-domain BCE rather than
BCEWithLogitsLoss. Probability-domain BCE is evaluated in an explicit FP32,
autocast-disabled region because PyTorch rejects BCELoss under AMP.

S2DENet's official DiffSA uses ``F.normalize(..., eps=1e-12)``. That epsilon
underflows to zero in FP16 and can produce NaN for an all-zero attention
vector, so CUDA AMP deliberately uses BF16 (whose exponent range safely
represents 1e-12) rather than FP16.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")
EXPECTED_COUNTS = {
    512: {"train": 16131, "val": 2111, "test": 1539},
    224: {"train": 26017, "val": 2796, "test": 3557},
}
CASE_FIELDS = [
    "filename", "metric_included", "gt_pixels", "pred_pixels",
    "dice", "iou", "recall", "precision", "hd95", "hd",
    "gt_cc", "pred_cc", "cc_delta", "abs_cc_delta",
    "inference_time_ms",
]
SUMMARY_FIELDS = [
    "model", "size", "split",
    "dice_mean", "dice_std",
    "iou_mean", "iou_std",
    "recall_mean", "recall_std",
    "precision_mean", "precision_std",
    "hd_count_delta", "empty_prediction_count",
    "hd95_mean", "hd95_std",
    "hd_mean", "hd_std",
    "cc_delta_mean", "cc_delta_std",
    "abs_cc_delta_mean", "abs_cc_delta_std",
    "efficiency_mean_ms", "efficiency_std_ms",
    "evaluated_gt_nonempty_count", "total_samples",
]


@dataclass
class RunConfig:
    model: str
    size: int
    data_root: str
    repo_root: str
    output_dir: str
    epochs: int
    warmup_epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    min_lr_ratio: float
    effective_batch_size: int
    micro_batch_size: int
    grad_accum_steps: int
    workers: int
    seed: int
    threshold: float
    edge_loss_weight: float
    hflip_prob: float
    image_mean: float
    image_std: float
    grad_clip: float
    amp: bool
    amp_dtype: str
    device: str
    official_commit: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train official S2DENet on native 512 or 224 consolidation patches.",
    )
    parser.add_argument("--size", type=int, choices=(512, 224), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("./third_party/S2DENet"))
    parser.add_argument("--output-root", type=Path, default=Path("./Evaluation/S2DENet"))
    parser.add_argument("--device", default="0", help="CUDA index such as 0, or cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument(
        "--micro-batch", type=int, default=0,
        help="0 uses benchmark batch (512:4, 224:16); smaller divisors use accumulation",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--edge-loss-weight", type=float, default=0.2)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--image-mean", type=float, default=0.100638)
    parser.add_argument("--image-std", type=float, default=0.145868)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--resume", nargs="?", const="auto", default=None,
        help="Resume from a checkpoint path; --resume alone uses output last.pt",
    )
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="Validate files and run one forward/loss pass without training",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(spec: str) -> torch.device:
    if spec.lower() == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    try:
        index = int(spec)
    except ValueError as exc:
        raise ValueError("--device must be a CUDA integer index or 'cpu'") from exc
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {index} unavailable; count={torch.cuda.device_count()}")
    torch.cuda.set_device(index)
    return torch.device(f"cuda:{index}")


def resolve_amp(device: torch.device, no_amp: bool) -> Tuple[bool, torch.dtype]:
    """Use BF16 AMP only; FP16 is numerically unsafe for official DiffSA."""
    enabled = device.type == "cuda" and not no_amp
    if enabled and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "S2DENet AMP requires CUDA BF16 support because official DiffSA uses "
            "F.normalize(eps=1e-12), which is unsafe in FP16. Re-run with --no-amp "
            "for FP32 on a GPU without BF16 support."
        )
    return enabled, torch.bfloat16


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype == np.uint16:
        image = np.rint(image.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        lo, hi = float(image.min()), float(image.max())
        if hi <= lo:
            image = np.zeros(image.shape, dtype=np.uint8)
        else:
            image = np.rint((image.astype(np.float32) - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return image


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Cannot read mask: {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask == 1).astype(np.uint8)


def find_mask(mask_dir: Path, image_path: Path) -> Path:
    direct = mask_dir / image_path.name
    if direct.is_file():
        return direct
    suffixed = mask_dir / f"{image_path.stem}_mask{image_path.suffix}"
    if suffixed.is_file():
        return suffixed
    candidates = sorted(
        p for p in mask_dir.glob(f"{image_path.stem}*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one mask for {image_path.name} in {mask_dir}, found {candidates}"
        )
    return candidates[0]


def list_pairs(data_root: Path, split: str, size: int) -> List[Tuple[Path, Path]]:
    image_dir = data_root / split / "images"
    mask_dir = data_root / split / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Incomplete split layout: {image_dir} and {mask_dir}")
    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No supported images found in {image_dir}")
    pairs = [(p, find_mask(mask_dir, p)) for p in images]
    for image_path, mask_path in pairs[: min(16, len(pairs))]:
        image, mask = read_gray(image_path), read_mask(mask_path)
        if image.shape != (size, size) or mask.shape != (size, size):
            raise ValueError(
                f"Native-size protocol expected {(size, size)}, got image={image.shape}, "
                f"mask={mask.shape}: {image_path.name}"
            )
    expected = EXPECTED_COUNTS[size][split]
    status = "OK" if len(pairs) == expected else f"WARNING expected {expected}"
    log(f"[{split}] paired={len(pairs)} ({status})")
    return pairs


class ConsolidationDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[Path, Path]],
        size: int,
        training: bool,
        hflip_prob: float,
        image_mean: float,
        image_std: float,
    ) -> None:
        self.pairs = list(pairs)
        self.size = size
        self.training = training
        self.hflip_prob = hflip_prob
        self.image_mean = image_mean
        self.image_std = image_std

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]
        image, mask = read_gray(image_path), read_mask(mask_path)
        if image.shape != (self.size, self.size) or mask.shape != (self.size, self.size):
            raise ValueError(f"Shape mismatch at {image_path}: {image.shape}, {mask.shape}")
        if self.training and random.random() < self.hflip_prob:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
        image = np.ascontiguousarray(image, dtype=np.float32) / 255.0
        image = (image - self.image_mean) / self.image_std
        image = np.repeat(image[None, ...], 3, axis=0)
        mask = np.ascontiguousarray(mask[None, ...], dtype=np.float32)
        return torch.from_numpy(image), torch.from_numpy(mask), image_path.name


def make_loader(
    pairs: Sequence[Tuple[Path, Path]],
    size: int,
    training: bool,
    batch_size: int,
    workers: int,
    seed: int,
    hflip_prob: float,
    image_mean: float,
    image_std: float,
) -> DataLoader:
    dataset = ConsolidationDataset(
        pairs, size, training, hflip_prob, image_mean, image_std
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def import_official_model(repo_root: Path):
    repo_root = repo_root.expanduser().resolve()
    model_file = repo_root / "s2denet" / "model.py"
    if not model_file.is_file():
        raise FileNotFoundError(
            f"Official S2DENet checkout not found at {repo_root}. "
            "Clone https://github.com/PXinTao/S2DENet.git first."
        )
    sys.path.insert(0, str(repo_root))
    from s2denet import S2DENet  # pylint: disable=import-outside-toplevel

    return S2DENet


def model_forward(model: nn.Module, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    output = model(images)
    if not isinstance(output, (tuple, list)) or len(output) != 2:
        raise RuntimeError("Official S2DENet must return (mask_probability, edge_probability)")
    mask_prob, edge_prob = output
    if mask_prob.shape[-2:] != images.shape[-2:]:
        mask_prob = F.interpolate(mask_prob, size=images.shape[-2:], mode="bilinear", align_corners=False)
    if edge_prob.shape[-2:] != images.shape[-2:]:
        edge_prob = F.interpolate(edge_prob, size=images.shape[-2:], mode="bilinear", align_corners=False)
    return mask_prob, edge_prob


def boundary_target(mask: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded > 0).to(mask.dtype)


def probability_bce(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # S2DENet returns sigmoid probabilities rather than logits. PyTorch blocks
    # BCELoss inside autocast even if its inputs are manually cast to float32,
    # so the BCE operation itself must run in a nested autocast-disabled block.
    # Gradients still propagate through the FP32 cast to the AMP model forward.
    with torch.autocast(device_type=probability.device.type, enabled=False):
        probability_fp32 = probability.float().clamp(1e-6, 1.0 - 1e-6)
        target_fp32 = target.float()
        return F.binary_cross_entropy(probability_fp32, target_fp32)


def validate_probability_output(name: str, probability: torch.Tensor) -> None:
    """Validate the upstream sigmoid-output contract once during preflight."""
    detached = probability.detach().float()
    if not torch.isfinite(detached).all().item():
        raise FloatingPointError(f"{name} contains NaN or Inf during preflight")
    minimum = float(detached.amin())
    maximum = float(detached.amax())
    tolerance = 1e-5
    if minimum < -tolerance or maximum > 1.0 + tolerance:
        raise RuntimeError(
            f"{name} must contain sigmoid probabilities in [0, 1], "
            f"but observed range [{minimum:.6g}, {maximum:.6g}]"
        )


def soft_dice_loss(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability, target = probability.float(), target.float()
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def composite_loss(
    mask_prob: torch.Tensor,
    edge_prob: torch.Tensor,
    target: torch.Tensor,
    edge_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    mask_loss = probability_bce(mask_prob, target) + soft_dice_loss(mask_prob, target)
    edge_gt = boundary_target(target)
    edge_loss = probability_bce(edge_prob, edge_gt) + soft_dice_loss(edge_prob, edge_gt)
    total = mask_loss + edge_weight * edge_loss
    return total, {
        "mask_loss": float(mask_loss.detach()),
        "edge_loss": float(edge_loss.detach()),
    }


def batch_positive_dice(
    probability: torch.Tensor, target: torch.Tensor, threshold: float
) -> List[float]:
    pred = probability >= threshold
    gt = target >= 0.5
    pred_count = pred.flatten(1).sum(1).to(torch.float64)
    gt_count = gt.flatten(1).sum(1).to(torch.float64)
    intersection = (pred & gt).flatten(1).sum(1).to(torch.float64)
    valid = gt_count > 0
    scores = 2.0 * intersection[valid] / (pred_count[valid] + gt_count[valid]).clamp_min(1.0)
    return [float(value) for value in scores.detach().cpu().tolist()]


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float,
):
    def factor(epoch_index: int) -> float:
        if warmup_epochs > 0 and epoch_index < warmup_epochs:
            return float(epoch_index + 1) / float(warmup_epochs)
        span = max(epochs - warmup_epochs, 1)
        progress = min(max((epoch_index - warmup_epochs) / span, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_dice: float,
    bad_epochs: int,
    config: RunConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_mean_dice": best_dice,
            "bad_epochs": bad_epochs,
            "config": asdict(config),
        },
        path,
    )


def write_history(path: Path, history: Sequence[Mapping[str, object]]) -> None:
    if not history:
        return
    fields = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    grad_accum_steps: int,
    edge_weight: float,
    grad_clip: float,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_samples = 0.0, 0
    for step, (images, targets, _) in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            mask_prob, edge_prob = model_forward(model, images)
            loss, _ = composite_loss(mask_prob, edge_prob, targets, edge_weight)
            scaled_loss = loss / grad_accum_steps
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at step {step}: {float(loss)}")
        scaler.scale(scaled_loss).backward()
        should_step = step % grad_accum_steps == 0 or step == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        batch_n = images.shape[0]
        total_loss += float(loss.detach()) * batch_n
        total_samples += batch_n
    return total_loss / max(total_samples, 1)


@torch.inference_mode()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    edge_weight: float,
    threshold: float,
) -> Dict[str, float]:
    model.eval()
    losses, loss_n, dice_scores = 0.0, 0, []
    for images, targets, _ in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            mask_prob, edge_prob = model_forward(model, images)
            loss, _ = composite_loss(mask_prob, edge_prob, targets, edge_weight)
        batch_n = images.shape[0]
        losses += float(loss) * batch_n
        loss_n += batch_n
        dice_scores.extend(batch_positive_dice(mask_prob, targets, threshold))
    if not dice_scores:
        raise RuntimeError("Validation split contains no non-empty consolidation masks")
    values = np.asarray(dice_scores, dtype=np.float64)
    return {
        "loss": losses / max(loss_n, 1),
        "dice_mean": float(values.mean()),
        "dice_std": float(values.std(ddof=0)),
        "dice_n": int(values.size),
    }


def mask_surface(mask: np.ndarray) -> np.ndarray:
    return np.logical_xor(
        mask,
        binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0),
    )


def hd_metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    side_length = float(max(gt.shape))
    if not pred.any():
        return side_length, side_length
    pred_surface, gt_surface = mask_surface(pred), mask_surface(gt)
    d1 = distance_transform_edt(~gt_surface)[pred_surface]
    d2 = distance_transform_edt(~pred_surface)[gt_surface]
    distances = np.concatenate([d1, d2]).astype(np.float64)
    return float(distances.max()), float(np.percentile(distances, 95))


def component_count(mask: np.ndarray) -> int:
    count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return int(count - 1)


def compute_case(
    filename: str,
    pred: np.ndarray,
    gt: np.ndarray,
    inference_time_ms: float,
) -> Dict[str, object]:
    pred, gt = pred.astype(bool), gt.astype(bool)
    gt_pixels, pred_pixels = int(gt.sum()), int(pred.sum())
    if gt_pixels == 0:
        row: Dict[str, object] = {
            "filename": filename,
            "metric_included": 0,
            "gt_pixels": 0,
            "pred_pixels": pred_pixels,
            "inference_time_ms": float(inference_time_ms),
        }
        for key in CASE_FIELDS:
            row.setdefault(key, float("nan"))
        return row
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    dice = 2.0 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    hd, hd95 = hd_metrics(pred, gt)
    gt_cc, pred_cc = component_count(gt), component_count(pred)
    cc_delta = gt_cc - pred_cc
    return {
        "filename": filename,
        "metric_included": 1,
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
        "dice": dice,
        "iou": iou,
        "recall": recall,
        "precision": precision,
        "hd95": hd95,
        "hd": hd,
        "gt_cc": gt_cc,
        "pred_cc": pred_cc,
        "cc_delta": cc_delta,
        "abs_cc_delta": abs(cc_delta),
        "inference_time_ms": float(inference_time_ms),
    }


def summarize_cases(rows: Sequence[Mapping[str, object]], split: str) -> Dict[str, object]:
    included = [row for row in rows if int(row["metric_included"]) == 1]
    summary: Dict[str, object] = {
        "split": split,
        "total_samples": len(rows),
        "evaluated_gt_nonempty_count": len(included),
        "gt_empty_excluded_count": len(rows) - len(included),
        "empty_prediction_count": sum(int(row["pred_pixels"]) == 0 for row in included),
    }
    summary["empty_prediction_rate"] = summary["empty_prediction_count"] / max(len(included), 1)
    for metric in (
        "dice", "iou", "recall", "precision", "hd95", "hd",
        "cc_delta", "abs_cc_delta",
    ):
        values = np.asarray([float(row[metric]) for row in included], dtype=np.float64)
        values = values[np.isfinite(values)]
        summary[f"{metric}_mean"] = float(values.mean()) if values.size else float("nan")
        summary[f"{metric}_std"] = float(values.std(ddof=0)) if values.size else float("nan")
        summary[f"{metric}_valid_count"] = int(values.size)
    times = np.asarray([float(row["inference_time_ms"]) for row in rows], dtype=np.float64)
    times = times[np.isfinite(times)]
    summary["inference_time_ms_mean"] = float(times.mean()) if times.size else float("nan")
    summary["inference_time_ms_std"] = float(times.std(ddof=0)) if times.size else float("nan")
    summary["inference_time_ms_valid_count"] = int(times.size)
    return summary


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)


def write_case_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CASE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)
    selected = mask.astype(bool)
    rgb[selected] = 0.45 * rgb[selected] + 0.55 * np.asarray(color, dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def save_gallery(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    pair_by_name: Mapping[str, Tuple[Path, Path]],
    prediction_by_name: Mapping[str, np.ndarray],
    title: str,
) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 3.1 * len(rows)), squeeze=False)
    for row_index, row in enumerate(rows):
        name = str(row["filename"])
        image_path, mask_path = pair_by_name[name]
        image, gt, pred = read_gray(image_path), read_mask(mask_path), prediction_by_name[name]
        panels = [image, overlay(image, gt, (0, 255, 0)), overlay(image, pred, (255, 0, 0))]
        subtitles = [name, "GT (green)", f"Prediction (red), Dice={float(row['dice']):.4f}"]
        for column, (panel, subtitle) in enumerate(zip(panels, subtitles)):
            axes[row_index, column].imshow(panel, cmap="gray" if column == 0 else None)
            axes[row_index, column].set_title(subtitle, fontsize=9)
            axes[row_index, column].axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def predict_gallery_masks(
    model: nn.Module,
    names: Sequence[str],
    pair_by_name: Mapping[str, Tuple[Path, Path]],
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    threshold: float,
    image_mean: float,
    image_std: float,
) -> Dict[str, np.ndarray]:
    predictions: Dict[str, np.ndarray] = {}
    for name in names:
        image = read_gray(pair_by_name[name][0]).astype(np.float32) / 255.0
        image = (image - image_mean) / image_std
        tensor = torch.from_numpy(
            np.ascontiguousarray(np.repeat(image[None, ...], 3, axis=0))
        ).unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            probability, _ = model_forward(model, tensor)
        predictions[name] = (probability[0, 0] >= threshold).to(torch.uint8).cpu().numpy()
    return predictions


@torch.inference_mode()
def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    pairs: Sequence[Tuple[Path, Path]],
    split: str,
    output_dir: Path,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    threshold: float,
    image_mean: float,
    image_std: float,
) -> Dict[str, object]:
    model.eval()
    pair_by_name = {image.name: (image, mask) for image, mask in pairs}
    if len(pair_by_name) != len(pairs):
        raise RuntimeError(f"Duplicate basenames detected in {split}")

    # Warm up kernels; this pass is intentionally excluded from timing.
    first_images, _, _ = next(iter(loader))
    first_images = first_images[:1].to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        model_forward(model, first_images)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    rows: List[Dict[str, object]] = []
    for images, targets, filenames in tqdm(loader, desc=f"evaluate {split}"):
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            probabilities, _ = model_forward(model, images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_per_image_ms = (time.perf_counter() - started) * 1000.0 / images.shape[0]
        batch_pred = (probabilities >= threshold).to(torch.uint8).cpu().numpy()[:, 0]
        batch_gt = targets.numpy()[:, 0].astype(np.uint8)
        for name, pred, gt in zip(filenames, batch_pred, batch_gt):
            rows.append(compute_case(name, pred, gt, elapsed_per_image_ms))

    summary = summarize_cases(rows, split)
    write_case_csv(output_dir / f"{split}_cases.csv", rows)
    write_json(output_dir / f"{split}_summary.json", summary)

    included = sorted(
        (row for row in rows if int(row["metric_included"]) == 1),
        key=lambda row: float(row["dice"]),
    )
    bottom = included[:5]
    top = list(reversed(included[-5:]))
    gallery_names = list(dict.fromkeys(str(row["filename"]) for row in top + bottom))
    gallery_predictions = predict_gallery_masks(
        model, gallery_names, pair_by_name, device, amp_enabled, amp_dtype, threshold,
        image_mean, image_std,
    )
    save_gallery(
        output_dir / f"{split}_top5_dice.png", top, pair_by_name, gallery_predictions,
        f"S2DENet {split}: Top-5 Dice",
    )
    save_gallery(
        output_dir / f"{split}_bottom5_dice.png", bottom, pair_by_name, gallery_predictions,
        f"S2DENet {split}: Bottom-5 Dice",
    )
    return summary


def write_ordered_summary(
    path: Path,
    size: int,
    summaries: Mapping[str, Mapping[str, object]],
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for split in SPLITS:
            summary = summaries[split]
            writer.writerow(
                {
                    "model": "S2DENet",
                    "size": size,
                    "split": split,
                    "dice_mean": summary["dice_mean"],
                    "dice_std": summary["dice_std"],
                    "iou_mean": summary["iou_mean"],
                    "iou_std": summary["iou_std"],
                    "recall_mean": summary["recall_mean"],
                    "recall_std": summary["recall_std"],
                    "precision_mean": summary["precision_mean"],
                    "precision_std": summary["precision_std"],
                    "hd_count_delta": summary["evaluated_gt_nonempty_count"] - summary["hd_valid_count"],
                    "empty_prediction_count": summary["empty_prediction_count"],
                    "hd95_mean": summary["hd95_mean"],
                    "hd95_std": summary["hd95_std"],
                    "hd_mean": summary["hd_mean"],
                    "hd_std": summary["hd_std"],
                    "cc_delta_mean": summary["cc_delta_mean"],
                    "cc_delta_std": summary["cc_delta_std"],
                    "abs_cc_delta_mean": summary["abs_cc_delta_mean"],
                    "abs_cc_delta_std": summary["abs_cc_delta_std"],
                    "efficiency_mean_ms": summary["inference_time_ms_mean"],
                    "efficiency_std_ms": summary["inference_time_ms_std"],
                    "evaluated_gt_nonempty_count": summary["evaluated_gt_nonempty_count"],
                    "total_samples": summary["total_samples"],
                }
            )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must lie strictly between 0 and 1")
    if args.image_std <= 0:
        raise ValueError("--image-std must be positive")
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("--epochs and --patience must be positive")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    data_root = args.data_root.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    output_dir = (args.output_root.expanduser().resolve() / str(args.size) / "s2denet")
    output_dir.mkdir(parents=True, exist_ok=True)

    effective_batch = 4 if args.size == 512 else 16
    micro_batch = args.micro_batch or effective_batch
    if micro_batch < 1 or micro_batch > effective_batch or effective_batch % micro_batch != 0:
        raise ValueError(
            f"--micro-batch must be a positive divisor of effective batch {effective_batch}"
        )
    grad_accum = effective_batch // micro_batch
    amp_enabled, amp_dtype = resolve_amp(device, args.no_amp)

    pairs = {split: list_pairs(data_root, split, args.size) for split in SPLITS}
    loaders = {
        split: make_loader(
            pairs[split], args.size, split == "train", micro_batch,
            args.workers, args.seed, args.hflip_prob if split == "train" else 0.0,
            args.image_mean, args.image_std,
        )
        for split in SPLITS
    }

    S2DENet = import_official_model(repo_root)
    model = S2DENet(in_channels=8, num_class=1, return_edge=True).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    try:
        import subprocess

        official_commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        official_commit = "unknown"

    config = RunConfig(
        model="S2DENet",
        size=args.size,
        data_root=str(data_root),
        repo_root=str(repo_root),
        output_dir=str(output_dir),
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        min_lr_ratio=args.min_lr_ratio,
        effective_batch_size=effective_batch,
        micro_batch_size=micro_batch,
        grad_accum_steps=grad_accum,
        workers=args.workers,
        seed=args.seed,
        threshold=args.threshold,
        edge_loss_weight=args.edge_loss_weight,
        hflip_prob=args.hflip_prob,
        image_mean=args.image_mean,
        image_std=args.image_std,
        grad_clip=args.grad_clip,
        amp=amp_enabled,
        amp_dtype="bfloat16" if amp_enabled else "float32",
        device=str(device),
        official_commit=official_commit,
    )
    settings = asdict(config)
    settings.update(
        {
            "parameters_total": total_parameters,
            "parameters_trainable": trainable_parameters,
            "split_counts": {split: len(pairs[split]) for split in SPLITS},
            "checkpoint_metric": "validation per-case mean Dice over GT-nonempty cases",
            "metric_threshold": args.threshold,
            "mask_label_values": [1],
            "normalization": f"grayscale/255, then (x-{args.image_mean})/{args.image_std}, repeated to 3 channels",
            "loss": "mask(BCE+soft-Dice) + edge_loss_weight * edge(BCE+soft-Dice)",
        }
    )
    write_json(output_dir / "training_settings.json", settings)
    log(json.dumps(settings, indent=2, ensure_ascii=False))

    # A real-size forward and loss check catches clone/API/CUDA problems before epoch 1.
    images, targets, _ = next(iter(loaders["train"]))
    images, targets = images[:1].to(device), targets[:1].to(device)
    model.eval()
    if not torch.isfinite(images).all().item():
        raise FloatingPointError("Preflight input contains NaN or Inf after normalization")
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
    ):
        mask_prob, edge_prob = model_forward(model, images)
        validate_probability_output("mask_probability", mask_prob)
        validate_probability_output("edge_probability", edge_prob)
        preflight_loss, _ = composite_loss(
            mask_prob, edge_prob, targets, args.edge_loss_weight
        )
    log(
        f"[PREFLIGHT OK] input={tuple(images.shape)} mask={tuple(mask_prob.shape)} "
        f"edge={tuple(edge_prob.shape)} loss={float(preflight_loss):.6f} "
        f"params={total_parameters:,}"
    )
    if args.preflight_only:
        return

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = make_scheduler(
        optimizer, args.epochs, args.warmup_epochs, args.min_lr_ratio
    )
    # BF16 does not need loss scaling. Keep the common scaler call path disabled.
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    start_epoch, best_dice, bad_epochs = 1, -math.inf, 0
    history: List[Dict[str, object]] = []
    best_path, last_path = output_dir / "best.pt", output_dir / "last.pt"
    if args.resume is not None:
        resume_path = last_path if args.resume == "auto" else Path(args.resume).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_dice = float(checkpoint["best_val_mean_dice"])
        bad_epochs = int(checkpoint["bad_epochs"])
        history_path = output_dir / "history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        log(f"[RESUME] {resume_path} -> epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_loss = train_epoch(
            model, loaders["train"], optimizer, scaler, device, amp_enabled, amp_dtype,
            grad_accum, args.edge_loss_weight, args.grad_clip,
        )
        val_stats = validate_epoch(
            model, loaders["val"], device, amp_enabled, amp_dtype,
            args.edge_loss_weight, args.threshold,
        )
        improved = val_stats["dice_mean"] > best_dice + 1e-12
        if improved:
            best_dice = val_stats["dice_mean"]
            bad_epochs = 0
        else:
            bad_epochs += 1

        record = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_stats["loss"],
            "val_dice_mean": val_stats["dice_mean"],
            "val_dice_std": val_stats["dice_std"],
            "val_dice_n": val_stats["dice_n"],
            "best_val_dice": best_dice,
            "improved": int(improved),
            "bad_epochs": bad_epochs,
        }
        history.append(record)
        write_json(output_dir / "history.json", history)
        write_history(output_dir / "history.csv", history)
        if improved:
            save_checkpoint(
                best_path, model, optimizer, scheduler, scaler,
                epoch, best_dice, bad_epochs, config,
            )
        scheduler.step()
        save_checkpoint(
            last_path, model, optimizer, scheduler, scaler,
            epoch, best_dice, bad_epochs, config,
        )
        log(
            f"Epoch {epoch:03d}/{args.epochs} lr={current_lr:.3e} "
            f"train_loss={train_loss:.6f} val_loss={val_stats['loss']:.6f} "
            f"val_dice={val_stats['dice_mean']:.6f}±{val_stats['dice_std']:.6f} "
            f"best={best_dice:.6f} bad_epochs={bad_epochs}/{args.patience}"
        )
        if bad_epochs >= args.patience:
            log(f"[EARLY STOP] no validation Mean Dice improvement for {args.patience} epochs")
            break

    if not best_path.is_file():
        raise RuntimeError("Training ended without best.pt")
    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    log(
        f"[BEST] epoch={best_checkpoint['epoch']} "
        f"val_mean_dice={best_checkpoint['best_val_mean_dice']:.6f}"
    )

    # Evaluation loaders never shuffle or augment and may use the full effective batch.
    eval_loaders = {
        split: make_loader(
            pairs[split], args.size, False, effective_batch,
            args.workers, args.seed, 0.0, args.image_mean, args.image_std,
        )
        for split in SPLITS
    }
    summaries: Dict[str, Dict[str, object]] = {}
    for split in SPLITS:
        summaries[split] = evaluate_split(
            model, eval_loaders[split], pairs[split], split, output_dir,
            device, amp_enabled, amp_dtype, args.threshold, args.image_mean, args.image_std,
        )
        log(
            f"[{split}] Dice={summaries[split]['dice_mean']:.6f} "
            f"IoU={summaries[split]['iou_mean']:.6f} "
            f"empty_pred={summaries[split]['empty_prediction_count']}"
        )
    write_ordered_summary(output_dir / "summary.csv", args.size, summaries)
    write_json(
        output_dir / "evaluation_settings.json",
        {
            **settings,
            "best_epoch": int(best_checkpoint["epoch"]),
            "best_val_mean_dice": float(best_checkpoint["best_val_mean_dice"]),
            "evaluation_order": list(SPLITS),
            "summary_column_order": SUMMARY_FIELDS,
        },
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"[DONE] Results saved to {output_dir}")


if __name__ == "__main__":
    main()
