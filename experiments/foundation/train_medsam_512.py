#!/usr/bin/env python3
"""MedSAM ViT-B benchmark for binary lung-ultrasound consolidation.

The script is intended to sit in the APRIL project root. It uses the official
MedSAM source tree and official ``medsam_vit_b.pth`` initialization, while
retaining the Size_512 benchmark protocol used for SAM2:

* binary prepared masks use foreground value 1;
* samples below the configured foreground-pixel minimum are excluded;
* every retained training image is visited exactly once per epoch;
* every image receives the same full-image box [0, 0, 1024, 1024];
* no prompt is derived from the ground-truth mask;
* best checkpoint selection uses mean per-image whole-image validation Dice;
* validation and test predictions use a fixed probability threshold of 0.5.

MedSAM's official inference preprocessing resizes to 1024 and performs
per-image min-max normalization to [0, 1]. The same preprocessing is used here.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import random
import sys
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


def parse_int_values(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
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
        default=Path("pretrained/medsam/medsam_vit_b.pth"),
        help="Official MedSAM ViT-B initialization checkpoint.",
    )
    parser.add_argument(
        "--medsam-source-dir",
        type=Path,
        default=Path("~/third_party/MedSAM"),
        help="Official bowang-lab/MedSAM checkout containing segment_anything.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="A timestamped output directory is used when omitted.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume a best_model.pth or last_model.pth made by this script.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate --resume on validation and test without training.",
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
        help=(
            "decoder trains only the mask decoder; image_decoder follows the "
            "original MedSAM scope and also fine-tunes the ViT image encoder."
        ),
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
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")

    # Smoke-test limits. Zero means the complete retained split.
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError("batch-size and grad-accum must be positive.")
    if args.epochs < 1 and not args.eval_only:
        raise ValueError("epochs must be positive.")
    if args.patience < 1:
        raise ValueError("patience must be positive.")
    if args.warmup_epochs < 0:
        raise ValueError("warmup-epochs cannot be negative.")
    if args.min_mask_pixels < 0:
        raise ValueError("min-mask-pixels cannot be negative.")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    if args.model_input_size != 1024:
        raise ValueError("MedSAM ViT-B requires --model-input-size 1024.")
    if args.eval_only and args.resume is None:
        raise ValueError("--eval-only requires --resume.")
    if args.bce_weight < 0.0 or args.dice_weight < 0.0:
        raise ValueError("Loss weights cannot be negative.")
    if args.bce_weight + args.dice_weight <= 0.0:
        raise ValueError("At least one loss weight must be positive.")


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
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def list_image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def resolve_pairs(split_root: Path) -> list[tuple[Path, Path]]:
    image_dir = split_root / "images"
    mask_dir = split_root / "masks"
    images = list_image_files(image_dir)
    masks = list_image_files(mask_dir)

    masks_by_name = {path.name: path for path in masks}
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
            f"{len(missing)} images in {image_dir} lack a unique matching mask. "
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
                f"No samples remain in {split_root} after "
                f"min_mask_pixels={min_mask_pixels}. Check --foreground-values "
                f"(currently {self.foreground_values})."
            )

        self.pairs = retained
        self.total_pairs = len(all_pairs)
        self.excluded_pairs = excluded

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _augment_pair(
        image: torch.Tensor, mask: torch.Tensor
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
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        # Official MedSAM inference performs per-image min-max normalization and
        # feeds [0,1] pixels directly to the image encoder.
        minimum = image.amin()
        maximum = image.amax()
        image = (image - minimum) / (maximum - minimum).clamp_min(1e-8)
        image = image.repeat(3, 1, 1)
        return {"image": image, "mask": mask, "name": image_path.name}


def import_medsam_registry(source_dir: Path):
    package_dir = source_dir / "segment_anything"
    if not (package_dir / "build_sam.py").is_file():
        raise FileNotFoundError(
            f"MedSAM source is incomplete: {package_dir / 'build_sam.py'}"
        )
    sys.path.insert(0, str(source_dir))
    from segment_anything import sam_model_registry

    imported_path = Path(inspect.getfile(sys.modules["segment_anything"])).resolve()
    if source_dir not in imported_path.parents:
        raise RuntimeError(
            f"segment_anything was imported from {imported_path}, not {source_dir}."
        )
    return sam_model_registry, imported_path


class MedSAMFixedFullBox(nn.Module):
    """Differentiable MedSAM forward with one identical, non-GT box."""

    def __init__(self, sam_model: nn.Module, model_input_size: int = 1024) -> None:
        super().__init__()
        self.sam = sam_model
        self.model_input_size = model_input_size

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_embeddings = self.sam.image_encoder(images)
        batch_size = images.shape[0]
        edge = float(self.model_input_size)
        boxes = images.new_tensor([0.0, 0.0, edge, edge])
        boxes = boxes.view(1, 1, 4).expand(batch_size, -1, -1)

        # The prompt encoder is always frozen, matching official MedSAM training.
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None, boxes=boxes, masks=None
            )
        low_res_logits, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return low_res_logits, iou_predictions[:, :1]


def configure_trainable_parameters(model: MedSAMFixedFullBox, scope: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.sam.mask_decoder.parameters():
        parameter.requires_grad = True
    if scope == "image_decoder":
        for parameter in model.sam.image_encoder.parameters():
            parameter.requires_grad = True


def set_training_mode(model: MedSAMFixedFullBox, scope: str) -> None:
    model.train()
    model.sam.prompt_encoder.eval()
    if scope == "decoder":
        model.sam.image_encoder.eval()


def trainable_parameter_summary(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable, total


def resize_logits_to_target(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return F.interpolate(
        logits.float(), size=target.shape[-2:], mode="bilinear", align_corners=False
    )


def medsam_soft_dice_loss(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Dice loss with squared predictions, as used by official MedSAM."""
    probabilities = logits.sigmoid()
    dims = tuple(range(1, probabilities.ndim))
    intersection = (probabilities * target).sum(dim=dims)
    denominator = probabilities.square().sum(dim=dims) + target.square().sum(dim=dims)
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def compute_loss(
    low_res_logits: torch.Tensor,
    target_native: torch.Tensor,
    bce_weight: float,
    dice_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = resize_logits_to_target(low_res_logits, target_native)
    bce = F.binary_cross_entropy_with_logits(logits, target_native)
    dice = medsam_soft_dice_loss(logits, target_native)
    total = bce_weight * bce + dice_weight * dice
    return total, {
        "loss": float(total.detach()),
        "bce": float(bce.detach()),
        "dice_loss": float(dice.detach()),
    }


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
            raise RuntimeError("BF16 was requested but this GPU does not support it.")
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
    return torch.amp.GradScaler(
        "cuda", enabled=amp.enabled and amp.dtype == torch.float16
    )


def per_image_dice(
    low_res_logits: torch.Tensor, target_native: torch.Tensor, threshold: float
) -> torch.Tensor:
    logits = resize_logits_to_target(low_res_logits, target_native)
    prediction = logits.sigmoid() >= threshold
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
    model: MedSAMFixedFullBox,
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
                low_res_logits, _ = model(images)
            batch_dice = per_image_dice(low_res_logits, masks, threshold)
            dice_values.extend(float(value) for value in batch_dice.cpu())

            if collect_cases:
                logits = resize_logits_to_target(low_res_logits, masks)
                predictions = logits.sigmoid() >= threshold
                for index, name in enumerate(batch["name"]):
                    case_rows.append(
                        {
                            "split": split,
                            "filename": name,
                            "dice": float(batch_dice[index].cpu()),
                            "gt_pixels": int((masks[index] >= 0.5).sum().cpu()),
                            "pred_pixels": int(predictions[index].sum().cpu()),
                            "threshold": threshold,
                        }
                    )

    if not dice_values:
        raise RuntimeError(f"No samples were evaluated for split '{split}'.")
    return float(np.mean(dice_values)), case_rows


def train_one_epoch(
    model: MedSAMFixedFullBox,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: AmpConfig,
    args: argparse.Namespace,
) -> dict[str, float]:
    set_training_mode(model, args.train_scope)
    optimizer.zero_grad(set_to_none=True)
    sums = {"loss": 0.0, "bce": 0.0, "dice_loss": 0.0}
    sample_count = 0

    progress = tqdm(loader, desc="Training", leave=False)
    for step, batch in enumerate(progress):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        batch_size = images.shape[0]

        with autocast_context(amp):
            low_res_logits, _ = model(images)
            loss, parts = compute_loss(
                low_res_logits,
                masks,
                bce_weight=args.bce_weight,
                dice_weight=args.dice_weight,
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
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                args.grad_clip,
            )
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Non-finite gradient norm at step {step}: {float(grad_norm)}"
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        for key, value in parts.items():
            sums[key] += value * batch_size
        sample_count += batch_size
        progress.set_postfix(loss=f"{parts['loss']:.4f}")

    if sample_count == 0:
        raise RuntimeError("Training loader produced no samples.")
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
    model: MedSAMFixedFullBox,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val_dice: float,
    patience_counter: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        # Raw SAM state keys preserve compatibility with the MedSAM architecture.
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
    model: MedSAMFixedFullBox,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError(
            f"{path} is not a checkpoint created by this benchmark script."
        )
    model.sam.load_state_dict(payload["model"], strict=True)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    return payload


def write_csv(
    path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]
) -> None:
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
        "val_whole_image_dice_mean",
        "best_val_dice",
        "patience_counter",
        "epoch_seconds",
    ]
    write_csv(path, history, fieldnames)


def build_datasets(args: argparse.Namespace) -> tuple[ConsolidationDataset, ...]:
    common = {
        "foreground_values": args.foreground_values,
        "min_mask_pixels": args.min_mask_pixels,
        "model_input_size": args.model_input_size,
    }
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
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is False.")
    device = torch.device("cuda:0")
    amp = choose_amp_config(args.amp_dtype)

    args.data_root = args.data_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.medsam_source_dir = args.medsam_source_dir.expanduser().resolve()
    if args.resume is not None:
        args.resume = args.resume.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Official MedSAM checkpoint not found: {args.checkpoint}")

    if args.output_dir is None:
        if args.resume is not None:
            output_dir = args.resume.parent
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"benchmark_medsam_consolidation_size512_{timestamp}")
    else:
        output_dir = args.output_dir.expanduser()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"AMP: {amp.name}")
    print(f"Data root: {args.data_root}")
    print(f"Output: {output_dir}")
    print("Initialization: official medsam_vit_b.pth")
    print("Prompt protocol: fixed full-image box; no GT-derived prompt")
    print("Preprocessing: 1024 resize + per-image min-max [0,1]")

    train_dataset, val_dataset, test_dataset = build_datasets(args)
    print_dataset_summary("train", train_dataset)
    print_dataset_summary("val", val_dataset)
    print_dataset_summary("test", test_dataset)

    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.num_workers, args.seed
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, False, args.num_workers, args.seed + 1
    )
    test_loader = make_loader(
        test_dataset, args.batch_size, False, args.num_workers, args.seed + 2
    )

    registry, imported_path = import_medsam_registry(args.medsam_source_dir)
    print(f"MedSAM source import: {imported_path}")
    sam_model = registry["vit_b"](checkpoint=str(args.checkpoint))
    model = MedSAMFixedFullBox(sam_model, args.model_input_size).to(device)
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
    if args.train_scope == "image_decoder" and args.lr >= 1e-4:
        print(
            "WARNING: image_decoder with lr >= 1e-4 may be unstable for local "
            "fine-tuning; --lr 1e-5 is recommended."
        )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
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
        saved_args = resume_payload.get("args", {})
        saved_scope = saved_args.get("train_scope") if isinstance(saved_args, dict) else None
        if saved_scope is not None and saved_scope != args.train_scope and not args.eval_only:
            raise RuntimeError(
                f"Resume scope mismatch: checkpoint={saved_scope}, "
                f"command={args.train_scope}."
            )
        start_epoch = int(resume_payload.get("epoch", -1)) + 1
        best_val_dice = float(resume_payload.get("best_val_dice", -math.inf))
        patience_counter = int(resume_payload.get("patience_counter", 0))
        print(
            f"Loaded checkpoint: next_epoch={start_epoch + 1}, "
            f"best_val_dice={best_val_dice:.6f}"
        )

    case_fields = (
        "split",
        "filename",
        "dice",
        "gt_pixels",
        "pred_pixels",
        "threshold",
    )
    if args.eval_only:
        val_dice, _ = evaluate(
            model, val_loader, device, amp, args.threshold, "val", False
        )
        test_dice, test_rows = evaluate(
            model, test_loader, device, amp, args.threshold, "test", True
        )
        write_csv(output_dir / "test_cases.csv", test_rows, case_fields)
        result = {
            "model": "medsam_vit_b",
            "mode": "eval_only",
            "checkpoint": str(args.resume),
            "prompt_protocol": "fixed_full_image_box_no_gt",
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
            model, val_loader, device, amp, args.threshold, "val", False
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
            f"best={best_val_dice:.6f} | "
            f"patience={patience_counter}/{args.patience} | "
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
        model, val_loader, device, amp, args.threshold, "val", False
    )
    test_dice, test_rows = evaluate(
        model, test_loader, device, amp, args.threshold, "test", True
    )
    write_csv(output_dir / "test_cases.csv", test_rows, case_fields)

    summary_row = {
        "model": "medsam_vit_b",
        "initialization": "official_medsam_vit_b",
        "train_scope": args.train_scope,
        "prompt_protocol": "fixed_full_image_box_no_gt",
        "preprocessing": "resize1024_per_image_minmax_0_1",
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
