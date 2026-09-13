#!/usr/bin/env python3
"""SAM2.1 benchmark for binary lung-ultrasound consolidation segmentation.

Protocol implemented here
-------------------------
* The prepared Size_512 masks are binary; foreground values default to ``1``.
* Samples with fewer than ``--min-mask-pixels`` foreground pixels are excluded.
* Every retained training image is visited once per epoch; there is no
  positive/negative balancing and no ``samples_per_epoch`` resampling.
* The model receives a fixed full-image box. The box is identical for every
  sample and contains no information derived from the ground-truth mask.
* SAM2 keeps its native 1024x1024 model input. Dice is measured after resizing
  predictions back to the original whole-image mask size (normally 512x512).
* The best checkpoint is selected by mean per-image validation Dice at a fixed
  probability threshold (default 0.5).

This file is intended to sit in the APRIL project root beside the existing
benchmark scripts. It imports the official SAM2 package installed in the
``sam_bench`` Conda environment; it does not modify APRIL internals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SAM_IMAGE_MEAN = (0.485, 0.456, 0.406)
SAM_IMAGE_STD = (0.229, 0.224, 0.225)


def parse_int_values(text: str) -> tuple[int, ...]:
    values = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one mask value is required.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("./datasets/Size_512"),
        help="Directory containing train/val/test, each with images and masks.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("pretrained/sam2/sam2.1_hiera_base_plus.pt"),
    )
    parser.add_argument(
        "--model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="A timestamped directory is created when this is omitted.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume a checkpoint produced by this script.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate --resume on val and test without further training.",
    )

    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--train-scope",
        choices=("decoder", "image_decoder"),
        default="decoder",
        help="decoder freezes the image encoder; image_decoder also fine-tunes it.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("auto", "bf16", "fp16", "none"),
        default="auto",
    )

    parser.add_argument("--model-input-size", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--foreground-values", type=parse_int_values, default=(1,))
    parser.add_argument("--min-mask-pixels", type=int, default=100)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--iou-weight", type=float, default=0.1)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")

    # These limits are for the initial local smoke test only. Zero means all.
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError("batch-size and grad-accum must both be positive.")
    if args.epochs < 1 and not args.eval_only:
        raise ValueError("epochs must be positive.")
    if args.patience < 1:
        raise ValueError("patience must be positive.")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    if args.model_input_size != 1024:
        raise ValueError(
            "SAM2.1 checkpoints use a native 1024 input. Keep --model-input-size 1024."
        )
    if args.eval_only and args.resume is None:
        raise ValueError("--eval-only requires --resume.")


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def list_image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def resolve_pairs(split_root: Path) -> list[tuple[Path, Path]]:
    image_dir = split_root / "images"
    mask_dir = split_root / "masks"
    images = list_image_files(image_dir)
    masks = list_image_files(mask_dir)

    masks_by_name = {p.name: p for p in masks}
    masks_by_stem: dict[str, list[Path]] = {}
    for path in masks:
        masks_by_stem.setdefault(path.stem, []).append(path)

    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for image_path in images:
        mask_path = masks_by_name.get(image_path.name)
        if mask_path is None:
            candidates = masks_by_stem.get(image_path.stem, [])
            if len(candidates) == 1:
                mask_path = candidates[0]
        if mask_path is None:
            missing.append(image_path.name)
        else:
            pairs.append((image_path, mask_path))

    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(
            f"{len(missing)} images in {image_dir} have no unique matching mask. "
            f"Examples: {preview}"
        )
    if not pairs:
        raise RuntimeError(f"No image-mask pairs found below {split_root}")
    return pairs


def load_binary_mask(path: Path, foreground_values: Sequence[int]) -> torch.Tensor:
    with Image.open(path) as mask_image:
        array = np.asarray(mask_image)
    if array.ndim == 3:
        array = array[..., 0]
    binary = np.isin(array, np.asarray(foreground_values)).astype(np.float32)
    return torch.from_numpy(binary).unsqueeze(0)


class ConsolidationDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        split_root: Path,
        foreground_values: Sequence[int],
        min_mask_pixels: int,
        model_input_size: int,
        augment: bool,
        max_samples: int = 0,
    ) -> None:
        self.split_root = split_root
        self.foreground_values = tuple(foreground_values)
        self.model_input_size = model_input_size
        self.augment = augment

        all_pairs = resolve_pairs(split_root)
        retained: list[tuple[Path, Path]] = []
        excluded = 0
        for image_path, mask_path in tqdm(
            all_pairs, desc=f"Scanning {split_root.name} masks", leave=False
        ):
            mask = load_binary_mask(mask_path, self.foreground_values)
            if int(mask.sum().item()) >= min_mask_pixels:
                retained.append((image_path, mask_path))
            else:
                excluded += 1

        if max_samples > 0:
            retained = retained[:max_samples]
        if not retained:
            raise RuntimeError(
                f"No samples remain in {split_root} after min_mask_pixels={min_mask_pixels}. "
                f"Check --foreground-values (currently {self.foreground_values})."
            )

        self.pairs = retained
        self.total_pairs = len(all_pairs)
        self.excluded_pairs = excluded

    def __len__(self) -> int:
        return len(self.pairs)

    def _augment_pair(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        angle = random.uniform(-8.0, 8.0)
        max_dx = round(0.03 * image.shape[-1])
        max_dy = round(0.03 * image.shape[-2])
        translate = (
            random.randint(-max_dx, max_dx),
            random.randint(-max_dy, max_dy),
        )
        scale = random.uniform(0.95, 1.05)
        image = TF.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=(0.0, 0.0),
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        mask = TF.affine(
            mask,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=(0.0, 0.0),
            interpolation=InterpolationMode.NEAREST,
            fill=0.0,
        )

        if random.random() < 0.3:
            image = TF.adjust_brightness(image, random.uniform(0.85, 1.15))
        if random.random() < 0.3:
            image = TF.adjust_contrast(image, random.uniform(0.85, 1.15))
        return image.clamp_(0.0, 1.0), (mask > 0.5).float()

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as image_pil:
            image = TF.pil_to_tensor(image_pil.convert("L")).float().div_(255.0)
        mask = load_binary_mask(mask_path, self.foreground_values)

        if image.shape[-2:] != mask.shape[-2:]:
            raise RuntimeError(
                f"Image/mask shape mismatch for {image_path.name}: "
                f"{tuple(image.shape[-2:])} vs {tuple(mask.shape[-2:])}"
            )
        if self.augment:
            image, mask = self._augment_pair(image, mask)

        image = TF.resize(
            image,
            [self.model_input_size, self.model_input_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        image = image.repeat(3, 1, 1)
        image = TF.normalize(image, mean=SAM_IMAGE_MEAN, std=SAM_IMAGE_STD)
        return {
            "image": image,
            "mask": mask,
            "name": image_path.name,
        }


class SAM2FixedFullBox(nn.Module):
    """Differentiable SAM2 image forward pass with one non-informative box."""

    def __init__(self, sam_model: nn.Module) -> None:
        super().__init__()
        self.sam = sam_model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        backbone_out = self.sam.forward_image(images)
        feature_maps = backbone_out["backbone_fpn"][-self.sam.num_feature_levels :]
        backbone_features = feature_maps[-1]

        # Match SAM2ImagePredictor's single-image feature preparation.
        if self.sam.directly_add_no_mem_embed:
            no_mem = self.sam.no_mem_embed.permute(0, 2, 1).unsqueeze(-1)
            backbone_features = backbone_features + no_mem

        high_res_features = (
            feature_maps[:-1] if self.sam.use_high_res_features_in_sam else None
        )

        batch_size = images.shape[0]
        edge = float(self.sam.image_size - 1)
        box_corners = images.new_tensor([[0.0, 0.0], [edge, edge]])
        point_coords = box_corners.unsqueeze(0).expand(batch_size, -1, -1)
        # SAM encodes a box as its top-left and bottom-right corners, labels 2 and 3.
        point_labels = torch.tensor(
            [2, 3], dtype=torch.int32, device=images.device
        ).unsqueeze(0).expand(batch_size, -1)

        sparse_embeddings, dense_embeddings = self.sam.sam_prompt_encoder(
            points=(point_coords, point_labels), boxes=None, masks=None
        )
        low_res_logits, iou_predictions, _, _ = self.sam.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        logits = F.interpolate(
            low_res_logits.float(),
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits, iou_predictions[:, :1]


def configure_trainable_parameters(model: SAM2FixedFullBox, scope: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    modules: list[nn.Module] = [
        model.sam.sam_prompt_encoder,
        model.sam.sam_mask_decoder,
    ]
    if scope == "image_decoder":
        modules.append(model.sam.image_encoder)

    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True


def set_training_mode(model: SAM2FixedFullBox, scope: str) -> None:
    model.train()
    # Unused video-memory modules should never update internal train-time state.
    model.sam.memory_attention.eval()
    model.sam.memory_encoder.eval()
    if scope == "decoder":
        model.sam.image_encoder.eval()


def trainable_parameter_summary(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid()
    dims = tuple(range(1, probabilities.ndim))
    intersection = (probabilities * target).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def hard_iou_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = logits.detach() >= 0.0
    truth = target >= 0.5
    dims = tuple(range(1, prediction.ndim))
    intersection = (prediction & truth).sum(dim=dims).float()
    union = (prediction | truth).sum(dim=dims).float()
    return (intersection + 1.0) / (union + 1.0)


def compute_loss(
    logits: torch.Tensor,
    iou_predictions: torch.Tensor,
    target_native: torch.Tensor,
    bce_weight: float,
    dice_weight: float,
    iou_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    target = F.interpolate(target_native, size=logits.shape[-2:], mode="nearest")
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = soft_dice_loss(logits, target)
    true_iou = hard_iou_from_logits(logits, target)
    iou_loss = F.mse_loss(iou_predictions.flatten(), true_iou)
    total = bce_weight * bce + dice_weight * dice + iou_weight * iou_loss
    parts = {
        "loss": float(total.detach()),
        "bce": float(bce.detach()),
        "dice_loss": float(dice.detach()),
        "iou_loss": float(iou_loss.detach()),
    }
    return total, parts


@dataclass(frozen=True)
class AmpConfig:
    enabled: bool
    dtype: torch.dtype | None
    name: str


def choose_amp_config(requested: str) -> AmpConfig:
    if requested == "none":
        return AmpConfig(False, None, "none")
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("--amp-dtype bf16 was requested but this GPU lacks BF16 support.")
        return AmpConfig(True, torch.bfloat16, "bf16")
    if requested == "fp16":
        return AmpConfig(True, torch.float16, "fp16")
    if torch.cuda.is_bf16_supported():
        return AmpConfig(True, torch.bfloat16, "bf16")
    return AmpConfig(True, torch.float16, "fp16")


def autocast_context(amp: AmpConfig):
    if not amp.enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp.dtype)


def make_grad_scaler(amp: AmpConfig) -> torch.amp.GradScaler:
    # BF16 has enough exponent range and does not require gradient scaling.
    return torch.amp.GradScaler("cuda", enabled=amp.enabled and amp.dtype == torch.float16)


def per_image_dice(
    logits: torch.Tensor, target_native: torch.Tensor, threshold: float
) -> torch.Tensor:
    resized_logits = F.interpolate(
        logits.float(),
        size=target_native.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    prediction = resized_logits.sigmoid() >= threshold
    truth = target_native >= 0.5
    dims = tuple(range(1, prediction.ndim))
    intersection = (prediction & truth).sum(dim=dims).float()
    denominator = prediction.sum(dim=dims).float() + truth.sum(dim=dims).float()
    return (2.0 * intersection + 1e-7) / (denominator + 1e-7)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def evaluate(
    model: SAM2FixedFullBox,
    loader: DataLoader,
    device: torch.device,
    amp: AmpConfig,
    threshold: float,
    split: str,
    collect_cases: bool,
) -> tuple[float, list[dict[str, Any]]]:
    model.eval()
    dice_values: list[float] = []
    case_rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Evaluating {split}", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with autocast_context(amp):
                logits, _ = model(images)
            batch_dice = per_image_dice(logits, masks, threshold)
            dice_values.extend(float(x) for x in batch_dice.cpu())

            if collect_cases:
                resized_logits = F.interpolate(
                    logits.float(),
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                predictions = resized_logits.sigmoid() >= threshold
                for idx, name in enumerate(batch["name"]):
                    case_rows.append(
                        {
                            "split": split,
                            "filename": name,
                            "dice": float(batch_dice[idx].cpu()),
                            "gt_pixels": int((masks[idx] >= 0.5).sum().cpu()),
                            "pred_pixels": int(predictions[idx].sum().cpu()),
                            "threshold": threshold,
                        }
                    )

    if not dice_values:
        raise RuntimeError(f"No samples were evaluated for split '{split}'.")
    return float(np.mean(dice_values)), case_rows


def train_one_epoch(
    model: SAM2FixedFullBox,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: AmpConfig,
    args: argparse.Namespace,
) -> dict[str, float]:
    set_training_mode(model, args.train_scope)
    optimizer.zero_grad(set_to_none=True)
    sums = {"loss": 0.0, "bce": 0.0, "dice_loss": 0.0, "iou_loss": 0.0}
    sample_count = 0

    progress = tqdm(loader, desc="Training", leave=False)
    for step, batch in enumerate(progress):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        batch_size = images.shape[0]

        with autocast_context(amp):
            logits, iou_predictions = model(images)
            loss, parts = compute_loss(
                logits,
                iou_predictions,
                masks,
                bce_weight=args.bce_weight,
                dice_weight=args.dice_weight,
                iou_weight=args.iou_weight,
            )
            scaled_loss = loss / args.grad_accum

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at training step {step}: {float(loss.detach())}"
            )

        scaler.scale(scaled_loss).backward()
        should_step = (step + 1) % args.grad_accum == 0 or (step + 1) == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), args.grad_clip
            )
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Non-finite gradient norm at training step {step}: {float(grad_norm)}"
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        for key, value in parts.items():
            sums[key] += value * batch_size
        sample_count += batch_size
        progress.set_postfix(loss=f"{parts['loss']:.4f}")

    return {key: value / sample_count for key, value in sums.items()}


def lr_factor(epoch_index: int, warmup_epochs: int, total_epochs: int) -> float:
    if warmup_epochs > 0 and epoch_index < warmup_epochs:
        return float(epoch_index + 1) / float(warmup_epochs)
    remaining = max(total_epochs - warmup_epochs, 1)
    progress = min(max((epoch_index - warmup_epochs) / remaining, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(
    model: SAM2FixedFullBox,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val_dice: float,
    patience_counter: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    # Keep "model" compatible with official build_sam2 checkpoint loading.
    return {
        "model": model.sam.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_val_dice": best_val_dice,
        "patience_counter": patience_counter,
        "args": json_safe(vars(args)),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }


def load_resume_checkpoint(
    path: Path,
    model: SAM2FixedFullBox,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise RuntimeError(f"Checkpoint lacks a 'model' state: {path}")
    model.sam.load_state_dict(payload["model"], strict=True)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    return payload


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    fieldnames = [
        "epoch",
        "lr",
        "train_loss",
        "train_bce",
        "train_dice_loss",
        "train_iou_loss",
        "val_whole_image_dice_mean",
        "best_val_dice",
        "patience_counter",
        "epoch_seconds",
    ]
    write_csv(path, history, fieldnames)


def build_datasets(args: argparse.Namespace) -> tuple[ConsolidationDataset, ...]:
    common = dict(
        foreground_values=args.foreground_values,
        min_mask_pixels=args.min_mask_pixels,
        model_input_size=args.model_input_size,
    )
    train_dataset = ConsolidationDataset(
        args.data_root / "train",
        augment=not args.no_augment,
        max_samples=args.max_train_samples,
        **common,
    )
    val_dataset = ConsolidationDataset(
        args.data_root / "val",
        augment=False,
        max_samples=args.max_val_samples,
        **common,
    )
    test_dataset = ConsolidationDataset(
        args.data_root / "test",
        augment=False,
        max_samples=args.max_test_samples,
        **common,
    )
    return train_dataset, val_dataset, test_dataset


def print_dataset_summary(name: str, dataset: ConsolidationDataset) -> None:
    print(
        f"{name}: retained={len(dataset)}, original={dataset.total_pairs}, "
        f"excluded_below_min_pixels={dataset.excluded_pairs}"
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed, args.deterministic)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SAM2 training, but torch.cuda.is_available() is False.")
    device = torch.device("cuda:0")
    amp = choose_amp_config(args.amp_dtype)

    args.data_root = args.data_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.resume is not None:
        args.resume = args.resume.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Official SAM2 checkpoint not found: {args.checkpoint}")

    if args.output_dir is None:
        if args.resume is not None:
            output_dir = args.resume.parent
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"benchmark_sam2_consolidation_size512_{timestamp}")
    else:
        output_dir = args.output_dir.expanduser()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"AMP: {amp.name}")
    print(f"Data root: {args.data_root}")
    print(f"Output: {output_dir}")
    print("Prompt protocol: fixed full-image box; no GT-derived prompt")

    train_dataset, val_dataset, test_dataset = build_datasets(args)
    print_dataset_summary("train", train_dataset)
    print_dataset_summary("val", val_dataset)
    print_dataset_summary("test", test_dataset)

    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed + 1,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed + 2,
    )

    from sam2.build_sam import build_sam2

    sam_model = build_sam2(
        args.model_cfg,
        str(args.checkpoint),
        device=device,
        mode="train",
        apply_postprocessing=False,
    )
    model = SAM2FixedFullBox(sam_model).to(device)
    configure_trainable_parameters(model, args.train_scope)
    trainable, total = trainable_parameter_summary(model)
    print(
        f"Parameters: trainable={trainable / 1e6:.2f} M / total={total / 1e6:.2f} M "
        f"({100.0 * trainable / total:.2f}%)"
    )
    print(
        f"Batch: physical={args.batch_size}, accumulation={args.grad_accum}, "
        f"effective={args.batch_size * args.grad_accum}"
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: lr_factor(epoch, args.warmup_epochs, args.epochs),
    )
    scaler = make_grad_scaler(amp)

    start_epoch = 0
    best_val_dice = -math.inf
    patience_counter = 0
    if args.resume is not None:
        resume_payload = load_resume_checkpoint(
            args.resume,
            model,
            None if args.eval_only else optimizer,
            None if args.eval_only else scheduler,
            None if args.eval_only else scaler,
        )
        start_epoch = int(resume_payload.get("epoch", -1)) + 1
        best_val_dice = float(resume_payload.get("best_val_dice", -math.inf))
        patience_counter = int(resume_payload.get("patience_counter", 0))
        print(
            f"Loaded resume checkpoint: epoch={start_epoch}, "
            f"best_val_dice={best_val_dice:.6f}"
        )

    if args.eval_only:
        val_dice, _ = evaluate(
            model, val_loader, device, amp, args.threshold, "val", collect_cases=False
        )
        test_dice, test_rows = evaluate(
            model, test_loader, device, amp, args.threshold, "test", collect_cases=True
        )
        write_csv(
            output_dir / "test_cases.csv",
            test_rows,
            ("split", "filename", "dice", "gt_pixels", "pred_pixels", "threshold"),
        )
        result = {
            "model": "sam2.1_hiera_base_plus",
            "mode": "eval_only",
            "checkpoint": str(args.resume),
            "val_whole_image_dice_mean": val_dice,
            "test_whole_image_dice_mean": test_dice,
            "threshold": args.threshold,
            "test_samples": len(test_dataset),
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Validation whole-image Dice: {val_dice:.6f}")
        print(f"Test whole-image Dice: {test_dice:.6f}")
        return

    history_path = output_dir / "history.csv"
    history: list[dict[str, Any]] = []
    if start_epoch > 0 and history_path.is_file():
        with history_path.open("r", newline="", encoding="utf-8") as handle:
            history.extend(csv.DictReader(handle))

    best_path = output_dir / "best_model.pth"
    last_path = output_dir / "last_model.pth"
    training_started = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.time()
        current_lr = float(optimizer.param_groups[0]["lr"])
        print(f"\nEpoch {epoch + 1}/{args.epochs} | lr={current_lr:.8g}")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, amp, args
        )
        val_dice, _ = evaluate(
            model, val_loader, device, amp, args.threshold, "val", collect_cases=False
        )

        improved = val_dice > best_val_dice
        if improved:
            best_val_dice = val_dice
            patience_counter = 0
        else:
            patience_counter += 1

        scheduler.step()
        payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_val_dice,
            patience_counter,
            args,
        )
        atomic_torch_save(payload, last_path)
        if improved:
            atomic_torch_save(payload, best_path)

        epoch_seconds = time.time() - epoch_started
        history.append(
            {
                "epoch": epoch + 1,
                "lr": current_lr,
                "train_loss": train_metrics["loss"],
                "train_bce": train_metrics["bce"],
                "train_dice_loss": train_metrics["dice_loss"],
                "train_iou_loss": train_metrics["iou_loss"],
                "val_whole_image_dice_mean": val_dice,
                "best_val_dice": best_val_dice,
                "patience_counter": patience_counter,
                "epoch_seconds": epoch_seconds,
            }
        )
        write_history(history_path, history)

        print(
            f"train_loss={train_metrics['loss']:.6f} | "
            f"val_whole_image_dice={val_dice:.6f} | "
            f"best={best_val_dice:.6f} | patience={patience_counter}/{args.patience} | "
            f"time={epoch_seconds / 60.0:.1f} min"
        )
        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    if not best_path.is_file():
        raise RuntimeError("Training ended without producing best_model.pth")

    best_payload = load_resume_checkpoint(best_path, model)
    best_epoch = int(best_payload.get("epoch", -1)) + 1
    val_dice, _ = evaluate(
        model, val_loader, device, amp, args.threshold, "val", collect_cases=False
    )
    test_dice, test_rows = evaluate(
        model, test_loader, device, amp, args.threshold, "test", collect_cases=True
    )
    write_csv(
        output_dir / "test_cases.csv",
        test_rows,
        ("split", "filename", "dice", "gt_pixels", "pred_pixels", "threshold"),
    )

    summary_row = {
        "model": "sam2.1_hiera_base_plus",
        "train_scope": args.train_scope,
        "prompt_protocol": "fixed_full_image_box_no_gt",
        "best_epoch": best_epoch,
        "val_whole_image_dice_mean": val_dice,
        "test_whole_image_dice_mean": test_dice,
        "threshold": args.threshold,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "physical_batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "amp_dtype": amp.name,
    }
    write_csv(
        output_dir / "benchmark_summary.csv",
        [summary_row],
        tuple(summary_row.keys()),
    )

    result = {
        **summary_row,
        "best_model": str(best_path),
        "last_model": str(last_path),
        "training_seconds": time.time() - training_started,
        "args": json_safe(vars(args)),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Validation whole-image Dice: {val_dice:.6f}")
    print(f"Test whole-image Dice: {test_dice:.6f}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()