#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fine-tune the legacy USFM joint model for binary consolidation at Size_224.

This script deliberately reuses the model/checkpoint/evaluation implementation
from ``evaluate_zeroshot.py``. Keep both files in this directory and run them
with the original USFM environment.

Transfer protocol
-----------------
* Source checkpoint: old 3-class joint model (0 background, 1 B-line,
  2 consolidation), normally usfm_weights/best_usfm_decoder.pth.
* Target model: 2 classes (0 background/non-consolidation, 1 consolidation).
* The target head is initialized from source rows [0, 2], so the pretrained
  consolidation head is retained and the obsolete B-line output is removed.
* Default training mode is decoder-only; the whole HVIT encoder stays frozen.
* Model selection uses validation macro Dice at the fixed binary threshold 0.5.
* The test split is evaluated once, after reloading best.pth.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def import_evaluation_module():
    """Load the matching USFM evaluation implementation from this directory."""
    candidates = (
        "evaluate_zeroshot",
    )
    errors: List[str] = []
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception as exc:  # retain exact causes for an actionable error
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise SystemExit(
        "Cannot import the required USFM evaluation script. Put "
        "evaluate_zeroshot.py in the same directory.\n"
        + "\n".join(errors)
    )


ev = import_evaluation_module()

DEFAULT_CHECKPOINT = Path("./pretrained/usfm/best_usfm_decoder.pth")
DEFAULT_DATA_ROOT = Path(
    "./data/Size_224_filtered"
)
DEFAULT_OUTPUT_ROOT = Path("./outputs/usfm_transfer_224")
EXPECTED_COUNTS = {"train": 26017, "val": 2796, "test": 3557}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--decoder-lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.4)
    parser.add_argument("--dice-weight", type=float, default=0.6)
    parser.add_argument("--foreground-weight", type=float, default=5.0)
    parser.add_argument(
        "--train-mode",
        choices=("decoder", "last2", "last4", "all"),
        default="decoder",
        help="Default decoder keeps the whole encoder frozen. Other modes are ablations.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--horizontal-flip", type=float, default=0.5)
    parser.add_argument("--visual-cases", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-count-check", action="store_true")
    parser.add_argument("--allow-nonbinary-mask-values", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("--epochs and --patience must be >= 1")
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("--warmup-epochs must be in [0, epochs)")
    if args.batch_size < 1 or args.grad_accum < 1 or args.workers < 0:
        raise ValueError("batch size/grad accumulation must be >=1 and workers >=0")
    if args.decoder_lr <= 0 or args.encoder_lr <= 0:
        raise ValueError("learning rates must be positive")
    if not 0 < args.min_lr_ratio <= 1:
        raise ValueError("--min-lr-ratio must be in (0,1]")
    if args.ce_weight < 0 or args.dice_weight < 0 or args.ce_weight + args.dice_weight <= 0:
        raise ValueError("loss weights must be nonnegative and not both zero")
    if args.foreground_weight <= 0 or args.grad_clip <= 0:
        raise ValueError("foreground weight and grad clip must be positive")
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be in (0,1)")
    if not 0 <= args.horizontal_flip <= 1:
        raise ValueError("--horizontal-flip must be in [0,1]")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class TrainDataset(ev.BenchmarkDataset):
    """Size-224 dataset with synchronized horizontal flipping."""

    def __init__(
        self,
        root: Path,
        split: str,
        allow_nonbinary_masks: bool,
        horizontal_flip: float = 0.0,
    ) -> None:
        super().__init__(root, split, allow_nonbinary_masks)
        self.horizontal_flip = float(horizontal_flip)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_path, mask_path = self.pairs[index]
        rgb = ev.read_rgb(image_path)
        mask = ev.read_binary_mask(mask_path, self.allow_nonbinary_masks)
        if rgb.shape[:2] != mask.shape:
            raise ValueError(
                f"Image/mask mismatch for {image_path.name}: {rgb.shape[:2]} vs {mask.shape}"
            )
        if random.random() < self.horizontal_flip:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        if rgb.shape[:2] != (224, 224):
            rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        image = rgb.astype(np.float32) / 255.0
        image = (image - ev.IMAGENET_MEAN[None, None, :]) / ev.IMAGENET_STD[None, None, :]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask.astype(np.uint8)),
            "name": image_path.name,
            "image_path": str(image_path),
        }


def transfer_joint_checkpoint_to_binary(
    checkpoint_path: Path,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Strictly load the legacy joint model, then map head rows 0/2 to 0/1."""
    checkpoint = ev.torch_load_cpu(checkpoint_path)
    raw_state, state_container = ev.extract_state_dict(checkpoint)
    normalized = ev.strip_uniform_prefixes(raw_state)
    source_classes, source_output_key = ev.infer_output_classes(normalized)
    if source_classes != 3:
        raise RuntimeError(
            f"Expected the old 3-class [background,B-line,consolidation] checkpoint, "
            f"but found {source_classes} output classes in {source_output_key}."
        )

    source_model = ev.USFMModel(num_classes=3)
    key_transform, _, loaded_output_key = ev.load_checkpoint_strict(
        source_model, raw_state, checkpoint_path
    )
    target_model = ev.USFMModel(num_classes=2)
    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    adapted: List[str] = []
    copied = 0

    for key, target_value in target_state.items():
        if key not in source_state:
            raise RuntimeError(f"Target key missing from source model: {key}")
        source_value = source_state[key]
        if source_value.shape == target_value.shape:
            target_state[key] = source_value.detach().clone()
            copied += 1
            continue
        if key.endswith("decode_head.conv_seg.weight"):
            if source_value.ndim != 4 or source_value.shape[0] != 3 or target_value.shape[0] != 2:
                raise RuntimeError(f"Unexpected output weight shapes: {source_value.shape} -> {target_value.shape}")
            target_state[key] = source_value[[0, 2]].detach().clone()
            adapted.append(key + ": rows [0,2] -> [0,1]")
            continue
        if key.endswith("decode_head.conv_seg.bias"):
            if source_value.ndim != 1 or source_value.shape[0] != 3 or target_value.shape[0] != 2:
                raise RuntimeError(f"Unexpected output bias shapes: {source_value.shape} -> {target_value.shape}")
            target_state[key] = source_value[[0, 2]].detach().clone()
            adapted.append(key + ": rows [0,2] -> [0,1]")
            continue
        raise RuntimeError(
            f"Unexpected non-head shape mismatch for {key}: "
            f"{tuple(source_value.shape)} -> {tuple(target_value.shape)}"
        )

    target_model.load_state_dict(target_state, strict=True)
    del source_model
    metadata = {
        "checkpoint_state_container": state_container,
        "checkpoint_key_transform": key_transform,
        "source_output_key": loaded_output_key or source_output_key,
        "source_classes": 3,
        "target_classes": 2,
        "copied_tensor_count": copied,
        "adapted_tensors": adapted,
        "source_class_mapping": {"0": "background", "1": "B-line", "2": "consolidation"},
        "target_class_mapping": {"0": "background/non-consolidation", "1": "consolidation"},
    }
    return target_model, metadata


def configure_trainable(model: nn.Module, train_mode: str) -> Dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    decoder = model.mmseg_model.decode_head
    for parameter in decoder.parameters():
        parameter.requires_grad = True

    selected_blocks: List[int] = []
    if train_mode in ("last2", "last4", "all"):
        if train_mode == "all":
            for parameter in model.mmseg_model.backbone.parameters():
                parameter.requires_grad = True
            selected_blocks = list(range(12))
        else:
            count = 2 if train_mode == "last2" else 4
            selected_blocks = list(range(12 - count, 12))
            matched = 0
            markers = tuple(f".blocks.{index}." for index in selected_blocks)
            for name, parameter in model.named_parameters():
                if "backbone" in name and any(marker in name for marker in markers):
                    parameter.requires_grad = True
                    matched += 1
            if matched == 0:
                raise RuntimeError(
                    "Could not find HVIT block parameter names for partial unfreezing. "
                    "Use --train-mode decoder, or inspect model.named_parameters()."
                )

    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    decoder_names = [name for name in trainable_names if "decode_head" in name]
    encoder_names = [name for name in trainable_names if "backbone" in name]
    if not decoder_names:
        raise RuntimeError("No decoder parameters were made trainable")
    return {
        "selected_encoder_blocks": selected_blocks,
        "trainable_tensor_count": len(trainable_names),
        "decoder_trainable_tensor_count": len(decoder_names),
        "encoder_trainable_tensor_count": len(encoder_names),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameter_count": sum(p.numel() for p in model.parameters()),
    }


def enforce_module_modes(model: nn.Module, train_mode: str) -> None:
    """Keep frozen encoder deterministic while decoder remains in training mode."""
    model.train()
    backbone = model.mmseg_model.backbone
    if train_mode == "decoder":
        backbone.eval()
    elif train_mode in ("last2", "last4"):
        backbone.eval()
        count = 2 if train_mode == "last2" else 4
        blocks = getattr(backbone, "blocks", None)
        if blocks is not None:
            for block in list(blocks)[-count:]:
                block.train()


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    decoder_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and "decode_head" in name
    ]
    encoder_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and "backbone" in name
    ]
    groups: List[Dict[str, Any]] = []
    if decoder_params:
        groups.append({"params": decoder_params, "lr": args.decoder_lr, "base_lr": args.decoder_lr})
    if encoder_params:
        groups.append({"params": encoder_params, "lr": args.encoder_lr, "base_lr": args.encoder_lr})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def lr_factor(epoch: int, args: argparse.Namespace) -> float:
    if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
        return float(epoch + 1) / float(args.warmup_epochs)
    denominator = max(args.epochs - args.warmup_epochs - 1, 1)
    progress = min(max((epoch - args.warmup_epochs) / denominator, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine


def set_epoch_lrs(optimizer: torch.optim.Optimizer, epoch: int, args: argparse.Namespace) -> None:
    factor = lr_factor(epoch, args)
    for group in optimizer.param_groups:
        group["lr"] = float(group["base_lr"]) * factor


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    probability = torch.softmax(logits.float(), dim=1)[:, 1]
    foreground = target.float()
    axes = tuple(range(1, probability.ndim))
    intersection = (probability * foreground).sum(dim=axes)
    denominator = probability.sum(dim=axes) + foreground.sum(dim=axes)
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice.mean()


def combined_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ce = F.cross_entropy(logits.float(), target.long(), weight=class_weights)
    dice = soft_dice_loss(logits, target)
    total_weight = args.ce_weight + args.dice_weight
    loss = (args.ce_weight * ce + args.dice_weight * dice) / total_weight
    return loss, ce.detach(), dice.detach()


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    class_weights: torch.Tensor,
    args: argparse.Namespace,
    epoch: int,
) -> Dict[str, float]:
    enforce_module_modes(model, args.train_mode)
    optimizer.zero_grad(set_to_none=True)
    sums = {"loss": 0.0, "ce": 0.0, "dice_loss": 0.0}
    samples = 0
    progress = tqdm(loader, desc=f"train {epoch + 1}/{args.epochs}", dynamic_ncols=True)

    for step, batch in enumerate(progress):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True).long()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(images)
            if logits.shape[-2:] != targets.shape[-2:]:
                logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
            loss, ce, dice = combined_loss(logits, targets, class_weights, args)
            scaled_loss = loss / args.grad_accum
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at epoch={epoch + 1}, step={step + 1}")
        scaler.scale(scaled_loss).backward()

        should_step = (step + 1) % args.grad_accum == 0 or (step + 1) == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_size = images.shape[0]
        samples += batch_size
        sums["loss"] += float(loss.detach()) * batch_size
        sums["ce"] += float(ce) * batch_size
        sums["dice_loss"] += float(dice) * batch_size
        progress.set_postfix(loss=f"{sums['loss'] / max(samples, 1):.4f}")
    return {key: value / max(samples, 1) for key, value in sums.items()}


@torch.inference_mode()
def validate_macro_dice(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    threshold: float,
    description: str,
) -> Dict[str, float]:
    model.eval()
    dice_values: List[torch.Tensor] = []
    loss_values: List[float] = []
    for batch in tqdm(loader, desc=description, dynamic_ncols=True):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True).bool()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(images)
            if logits.shape[-2:] != targets.shape[-2:]:
                logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
        predictions = probabilities >= threshold
        axes = tuple(range(1, predictions.ndim))
        tp = torch.logical_and(predictions, targets).sum(dim=axes).float()
        fp = torch.logical_and(predictions, ~targets).sum(dim=axes).float()
        fn = torch.logical_and(~predictions, targets).sum(dim=axes).float()
        dice_values.append((2.0 * tp / torch.clamp(2.0 * tp + fp + fn, min=1.0)).cpu())
        loss_values.append(float(soft_dice_loss(logits, targets.long())) * images.shape[0])
    values = torch.cat(dice_values).numpy().astype(np.float64)
    return {
        "dice_mean": float(values.mean()),
        "dice_std": float(values.std(ddof=0)),
        "soft_dice_loss": float(sum(loss_values) / max(len(loader.dataset), 1)),
        "n": int(values.size),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    best_val_dice: float,
    bad_epochs: int,
    args: argparse.Namespace,
    transfer_metadata: Mapping[str, Any],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": int(epoch),
        "best_val_dice": float(best_val_dice),
        "bad_epochs": int(bad_epochs),
        "num_classes": 2,
        "consolidation_channel": 1,
        "train_mode": args.train_mode,
        "source_checkpoint": str(args.checkpoint.expanduser().resolve()),
        "transfer_metadata": dict(transfer_metadata),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    torch.save(payload, path)


def load_training_state(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
) -> Tuple[int, float, int]:
    checkpoint = ev.torch_load_cpu(path)
    state, _ = ev.extract_state_dict(checkpoint)
    model.load_state_dict(ev.strip_uniform_prefixes(state), strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    parameter_device = next(model.parameters()).device
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(parameter_device)
    if checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint.get("best_val_dice", -1.0)),
        int(checkpoint.get("bad_epochs", 0)),
    )


def write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_history(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def check_counts(datasets: Mapping[str, Any], skip: bool) -> None:
    for split, dataset in datasets.items():
        actual = len(dataset)
        expected = EXPECTED_COUNTS[split]
        print(f"[Size_224][{split}] paired={actual}, expected={expected}")
        if actual != expected and not skip:
            raise RuntimeError(
                f"Expected {expected} samples for Size_224_filtered/{split}, got {actual}. "
                "Use the correct patient-wise benchmark split or pass --skip-count-check intentionally."
            )


def make_loader(
    dataset: Any,
    batch_size: int,
    shuffle: bool,
    workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker if workers > 0 else None,
        generator=generator,
        drop_last=False,
    )


def make_final_record(
    best_path: Path,
    args: argparse.Namespace,
    summary: Mapping[str, Any],
    best_val_dice: float,
) -> Dict[str, Any]:
    return {
        "model": "USFM_HVIT_UPerHead_binary_finetune",
        "size": 224,
        "split": "test",
        "checkpoint": str(best_path),
        "source_checkpoint": str(args.checkpoint.expanduser().resolve()),
        "train_mode": args.train_mode,
        "best_val_dice": best_val_dice,
        "input_size": 224,
        "native_eval_size": 224,
        "decode_mode": "threshold",
        "consolidation_channel": 1,
        "threshold": args.threshold,
        "Mean-dice": summary["dice_mean"],
        "Std-dice": summary["dice_std"],
        "Mean-iou": summary["iou_mean"],
        "Std-iou": summary["iou_std"],
        "Mean-Precision": summary["precision_mean"],
        "Std-Precision": summary["precision_std"],
        "Mean-Recall": summary["recall_mean"],
        "Std-Recall": summary["recall_std"],
        "Mean-Specificity": summary["specificity_mean"],
        "Std-Specificity": summary["specificity_std"],
        "empty_prediction_count": summary["empty_prediction_count"],
        "Mean-HD": summary["hd_mean"],
        "Std-HD": summary["hd_std"],
        "Mean-HD95": summary["hd95_mean"],
        "Std-HD95": summary["hd95_std"],
        "HD-valid-count": summary["hd_valid_count"],
        "Mean-Delta-CC": summary["cc_delta_mean"],
        "Std-Delta-CC": summary["cc_delta_std"],
        "Mean-Abs-Delta-CC": summary["abs_cc_delta_mean"],
        "Std-Abs-Delta-CC": summary["abs_cc_delta_std"],
        "Efficiency-ms-image": summary["inference_time_ms_mean"],
        "Efficiency-std": summary["inference_time_ms_std"],
        "N": summary["evaluated_gt_nonempty_count"],
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)

    checkpoint_path = args.checkpoint.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {checkpoint_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    best_path = output_root / "best.pth"
    last_path = output_root / "last.pth"
    history_path = output_root / "training_history.csv"

    model, transfer_metadata = transfer_joint_checkpoint_to_binary(checkpoint_path)
    trainable_metadata = configure_trainable(model, args.train_mode)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    scaler = make_grad_scaler(amp_enabled and amp_dtype == torch.float16)
    model.to(device)
    optimizer = build_optimizer(model, args)
    class_weights = torch.tensor([1.0, args.foreground_weight], device=device)

    datasets = {
        "train": TrainDataset(
            data_root, "train", args.allow_nonbinary_mask_values, args.horizontal_flip
        ),
        "val": TrainDataset(data_root, "val", args.allow_nonbinary_mask_values, 0.0),
        "test": TrainDataset(data_root, "test", args.allow_nonbinary_mask_values, 0.0),
    }
    check_counts(datasets, args.skip_count_check)
    loaders = {
        "train": make_loader(datasets["train"], args.batch_size, True, args.workers, device, args.seed),
        "val": make_loader(datasets["val"], args.batch_size, False, args.workers, device, args.seed + 1),
        "test": make_loader(datasets["test"], args.batch_size, False, args.workers, device, args.seed + 2),
    }

    start_epoch, best_val_dice, bad_epochs = 0, -1.0, 0
    history: List[Dict[str, Any]] = []
    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"--resume requested, but last checkpoint is missing: {last_path}")
        start_epoch, best_val_dice, bad_epochs = load_training_state(
            last_path, model, optimizer, scaler
        )
        history = read_history(history_path)
        print(f"[RESUME] epoch={start_epoch + 1}, best_val_dice={best_val_dice:.6f}, bad_epochs={bad_epochs}")
    elif best_path.exists() or last_path.exists():
        raise FileExistsError(
            f"Output already contains checkpoints: {output_root}. "
            "Use --resume or choose a new --output-root."
        )

    settings = {
        "model": "USFM HVITBackbone4Seg + UPerHead",
        "task": "binary consolidation segmentation, Size_224_filtered",
        "source_checkpoint": str(checkpoint_path),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "transfer": transfer_metadata,
        "trainable": trainable_metadata,
        "input_size": [224, 224],
        "normalization": "ImageNet; identical to zero-shot evaluation",
        "source_classes": "0 background, 1 B-line, 2 consolidation",
        "target_classes": "0 background/non-consolidation, 1 consolidation",
        "head_initialization": "source output rows [0,2] copied to target rows [0,1]",
        "model_selection": "validation per-image macro Dice, foreground probability >= 0.5",
        "test_policy": "test evaluated once after early stopping and best reload",
        "sampling": "complete dataset; shuffle train only; no balanced sampler; drop_last=False",
        "augmentation": f"synchronized horizontal flip p={args.horizontal_flip}; train only",
        "loss": f"{args.ce_weight} CrossEntropy + {args.dice_weight} SoftDice; foreground CE weight={args.foreground_weight}",
        "optimizer": "AdamW",
        "schedule": "linear warmup then cosine decay",
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "decoder_lr": args.decoder_lr,
        "encoder_lr": args.encoder_lr if args.train_mode != "decoder" else None,
        "min_lr_ratio": args.min_lr_ratio,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "threshold": args.threshold,
        "amp": amp_enabled,
        "amp_dtype": args.amp_dtype,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "command": " ".join(sys.argv),
    }
    ev.write_json(output_root / "training_settings.json", settings)

    print("=" * 96)
    print(f"torch={torch.__version__}, python={platform.python_version()}, device={device}")
    print(f"source={checkpoint_path}")
    print(f"data={data_root}")
    print(f"output={output_root}")
    print(f"train_mode={args.train_mode}; trainable={trainable_metadata['trainable_parameter_count']:,}"
          f"/{trainable_metadata['total_parameter_count']:,}")
    print(f"epochs={args.epochs}, warmup={args.warmup_epochs}, patience={args.patience}, "
          f"batch={args.batch_size}, grad_accum={args.grad_accum}")
    print("target=2 classes; source head rows [background, consolidation] retained")
    print("=" * 96)

    for epoch in range(start_epoch, args.epochs):
        set_epoch_lrs(optimizer, epoch, args)
        train_stats = train_one_epoch(
            model, loaders["train"], optimizer, scaler, device, amp_enabled,
            amp_dtype, class_weights, args, epoch,
        )
        val_stats = validate_macro_dice(
            model, loaders["val"], device, amp_enabled, amp_dtype,
            args.threshold, f"val {epoch + 1}/{args.epochs}",
        )
        current_lr = max(float(group["lr"]) for group in optimizer.param_groups)
        improved = val_stats["dice_mean"] > best_val_dice + 1e-8
        if improved:
            best_val_dice = val_stats["dice_mean"]
            bad_epochs = 0
        else:
            bad_epochs += 1

        row = {
            "epoch": epoch + 1,
            "lr_max": current_lr,
            "train_loss": train_stats["loss"],
            "train_ce": train_stats["ce"],
            "train_dice_loss": train_stats["dice_loss"],
            "val_dice_mean": val_stats["dice_mean"],
            "val_dice_std": val_stats["dice_std"],
            "val_soft_dice_loss": val_stats["soft_dice_loss"],
            "best_val_dice": best_val_dice,
            "bad_epochs": bad_epochs,
            "improved": int(improved),
        }
        history.append(row)
        write_history(history_path, history)
        save_checkpoint(
            last_path, model, optimizer, scaler, epoch, best_val_dice,
            bad_epochs, args, transfer_metadata,
        )
        if improved:
            save_checkpoint(
                best_path, model, optimizer, scaler, epoch, best_val_dice,
                bad_epochs, args, transfer_metadata,
            )
        print(
            f"[epoch {epoch + 1:03d}] train_loss={train_stats['loss']:.6f} "
            f"val_dice={val_stats['dice_mean']:.6f} best={best_val_dice:.6f} "
            f"bad_epochs={bad_epochs}/{args.patience}"
        )
        if bad_epochs >= args.patience:
            print(f"[EARLY STOP] no validation Dice improvement for {args.patience} epochs")
            break

    if not best_path.is_file():
        raise RuntimeError("Training ended without creating best.pth")

    best_checkpoint = ev.torch_load_cpu(best_path)
    best_state, _ = ev.extract_state_dict(best_checkpoint)
    model.load_state_dict(ev.strip_uniform_prefixes(best_state), strict=True)
    model.to(device).eval()

    test_rows = ev.evaluate(
        model, loaders["test"], device, amp_enabled, amp_dtype,
        "threshold", 1, args.threshold, "USFM fine-tuned Size_224 test",
    )
    test_summary = ev.summarize(test_rows)
    ev.write_cases(output_root / "test_cases.csv", test_rows)
    ev.write_json(output_root / "test_summary.json", test_summary)

    ranked = sorted(
        (row for row in test_rows if int(row["metric_included"]) == 1),
        key=lambda row: float(row["dice"]),
    )
    index_by_name = {
        image_path.name: index
        for index, (image_path, _) in enumerate(datasets["test"].pairs)
    }
    visual_n = min(args.visual_cases, len(ranked))
    ev.save_ranked_visuals(
        output_root / f"test_bottom{visual_n}_dice.png",
        f"USFM fine-tuned Size_224: Bottom-{visual_n}",
        ranked[:visual_n], model, datasets["test"], index_by_name, device,
        amp_enabled, amp_dtype, "threshold", 1, args.threshold,
    )
    ev.save_ranked_visuals(
        output_root / f"test_top{visual_n}_dice.png",
        f"USFM fine-tuned Size_224: Top-{visual_n}",
        list(reversed(ranked[-visual_n:])), model, datasets["test"], index_by_name,
        device, amp_enabled, amp_dtype, "threshold", 1, args.threshold,
    )

    record = make_final_record(best_path, args, test_summary, best_val_dice)
    summary_path = output_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record.keys()))
        writer.writeheader()
        writer.writerow(record)
    print(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=True))
    print(f"\n[DONE] best={best_path}")
    print(f"[DONE] summary={summary_path}")


if __name__ == "__main__":
    main()
