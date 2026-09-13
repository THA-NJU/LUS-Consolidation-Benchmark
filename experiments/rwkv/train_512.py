#!/usr/bin/env python3
"""Single-GPU APRIL RWKV benchmark for Size_512 consolidation masks.

Models:
  - rwkv_unet: RWKV-UNet (Jiang et al., 2025), variant B
  - u_rwkv: U-RWKV (MICCAI 2025 implementation in APRIL)

The script uses APRIL only for model construction and its WKV CUDA kernel.
Dataset loading, training, checkpoint selection, and evaluation follow the
project's finalized Size_512 protocol.

Version 3 aligns APRIL's U-RWKV SpatialMix with the public U-RWKV reference:
``spatial_decay / T`` and ``spatial_first / T`` are passed to WKV, and
LayerNorm is applied to the WKV output before gating. It also caps both stable
norm recoveries and genuine non-finite FP32 retries. No sample is silently
skipped.
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
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "rwkv_unet": {
        "architecture": "rwkv_unet",
        "arch_params": {"variant": "b"},
        "batch_size": 1,
        "grad_accum": 4,
        "kernel_tmax": 8192,
    },
    "u_rwkv": {
        "architecture": "u_rwkv",
        "arch_params": {
            "embed_dims": [64, 128, 256, 512],
            "depths": [2, 2, 2, 2],
            "shift_pixel": 1,
            "se_ratio": 0.25,
            "deep_supervision": False,
        },
        "batch_size": 1,
        "grad_accum": 4,
        "kernel_tmax": 65536,
    },
}

PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "epochs": 1,
        "warmup_epochs": 0,
        "patience": 1,
        "max_train_samples": 2,
        "max_val_samples": 2,
        "max_test_samples": 5,
    },
    "formal": {
        "epochs": 600,
        "warmup_epochs": 10,
        "patience": 15,
        "max_train_samples": None,
        "max_val_samples": None,
        "max_test_samples": None,
    },
}

IMAGE_SIZE = 512
NUM_CLASSES = 2
FIXED_THRESHOLD = 0.5
CE_WEIGHT = 0.40
DICE_WEIGHT = 0.60
FOREGROUND_CE_WEIGHT = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--april-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="APRIL-MedSeg root; defaults to this script's directory.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Defaults to ./datasets/Size_512.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Explicit output root. Required to resume an earlier run.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=["rwkv_unet"],
    )
    parser.add_argument("--run-mode", choices=tuple(PRESETS), default="formal")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument(
        "--max-fp32-retries-per-epoch",
        type=int,
        default=3,
        help=(
            "Maximum accumulation groups per epoch that may be recomputed "
            "without autocast after genuinely non-finite gradients. "
            "Samples are never skipped."
        ),
    )
    parser.add_argument(
        "--max-stable-norm-recoveries-per-epoch",
        type=int,
        default=3,
        help=(
            "Maximum finite-gradient groups per epoch that may use the "
            "overflow-safe but mathematically equivalent L2 clipping path. "
            "The next event aborts training instead of masking systematic "
            "gradient explosion."
        ),
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-foreground-pixels", type=int, default=100)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict[str, Any], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return result.stdout.strip()
    except Exception as exc:
        return f"unavailable ({type(exc).__name__}: {exc})"


def normalize_mask_key(path: Path) -> str:
    stem = path.stem
    return stem[:-5] if stem.endswith("_mask") else stem


class ConsolidationDataset(Dataset):
    def __init__(
        self,
        split_root: Path,
        *,
        augment: bool,
        min_foreground_pixels: int,
        max_samples: int | None,
    ) -> None:
        self.split_root = split_root
        self.augment = augment
        image_dir = split_root / "images"
        mask_dir = split_root / "masks"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(
                f"Expected images/ and masks/ under {split_root}"
            )

        image_map = {path.stem: path for path in image_dir.glob("*.png")}
        mask_map: dict[str, Path] = {}
        duplicate_mask_keys: list[str] = []
        for path in mask_dir.glob("*.png"):
            key = normalize_mask_key(path)
            if key in mask_map:
                duplicate_mask_keys.append(key)
            mask_map[key] = path
        if duplicate_mask_keys:
            raise RuntimeError(
                f"Duplicate normalized mask names in {mask_dir}: "
                f"{duplicate_mask_keys[:10]}"
            )

        missing_masks = sorted(image_map.keys() - mask_map.keys())
        extra_masks = sorted(mask_map.keys() - image_map.keys())
        if missing_masks or extra_masks:
            raise RuntimeError(
                f"Unpaired PNG files in {split_root}: "
                f"images_without_mask={len(missing_masks)}, "
                f"masks_without_image={len(extra_masks)}; "
                f"examples={missing_masks[:3] + extra_masks[:3]}"
            )

        retained: list[tuple[Path, Path, int]] = []
        excluded = 0
        for key in tqdm(
            sorted(image_map),
            desc=f"scan {split_root.name} masks",
            unit="mask",
            leave=False,
        ):
            mask_path = mask_map[key]
            with Image.open(mask_path) as mask_image:
                mask_array = np.asarray(mask_image.convert("L"))
            foreground_pixels = int(np.count_nonzero(mask_array > 0))
            if foreground_pixels < min_foreground_pixels:
                excluded += 1
                continue
            retained.append((image_map[key], mask_path, foreground_pixels))
            if max_samples is not None and len(retained) >= max_samples:
                break

        if not retained:
            raise RuntimeError(f"No retained samples in {split_root}")
        self.samples = retained
        self.excluded = excluded

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def augment_pair(
        image: Image.Image, mask: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)

        if random.random() < 0.35:
            angle = random.uniform(-8.0, 8.0)
            max_shift = round(0.03 * IMAGE_SIZE)
            translate = (
                random.randint(-max_shift, max_shift),
                random.randint(-max_shift, max_shift),
            )
            scale = random.uniform(0.95, 1.05)
            image = TF.affine(
                image,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=(0.0, 0.0),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            mask = TF.affine(
                mask,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=(0.0, 0.0),
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )

        if random.random() < 0.35:
            image = ImageEnhance.Brightness(image).enhance(
                random.uniform(0.90, 1.10)
            )
            image = ImageEnhance.Contrast(image).enhance(
                random.uniform(0.90, 1.10)
            )
            gamma = random.uniform(0.85, 1.20)
            array = np.asarray(image, dtype=np.float32) / 255.0
            array = np.power(np.clip(array, 0.0, 1.0), gamma)
            image = Image.fromarray(
                np.uint8(np.clip(array * 255.0, 0.0, 255.0)),
                mode="L",
            )
        if random.random() < 0.10:
            image = image.filter(
                ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.0))
            )
        return image, mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, mask_path, foreground_pixels = self.samples[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("L")
        with Image.open(mask_path) as mask_file:
            mask = mask_file.convert("L")

        target_size = (IMAGE_SIZE, IMAGE_SIZE)
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.BILINEAR)
        if mask.size != target_size:
            mask = mask.resize(target_size, Image.Resampling.NEAREST)
        if self.augment:
            image, mask = self.augment_pair(image, mask)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        if self.augment and random.random() < 0.15:
            noise = np.random.normal(
                0.0, 0.02, image_array.shape
            ).astype(np.float32)
            image_array = image_array + noise
        image_array = np.clip(image_array, 0.0, 1.0)
        image_array = np.repeat(image_array[None, ...], 3, axis=0)
        mask_array = (np.asarray(mask) > 0).astype(np.int64)

        return {
            "image": torch.from_numpy(
                np.ascontiguousarray(image_array)
            ).float(),
            "target": torch.from_numpy(
                np.ascontiguousarray(mask_array)
            ).long(),
            "name": image_path.name,
            "foreground_pixels": foreground_pixels,
        }


class ForegroundDiceLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probability = torch.softmax(logits.float(), dim=1)[:, 1]
        target_float = target.float()
        intersection = (probability * target_float).sum(dim=(1, 2))
        denominator = (
            probability.sum(dim=(1, 2)) + target_float.sum(dim=(1, 2))
        )
        dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
        return 1.0 - dice.mean()


class ConsolidationLoss(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        class_weight = torch.tensor(
            [1.0, FOREGROUND_CE_WEIGHT],
            dtype=torch.float32,
            device=device,
        )
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weight)
        self.dice = ForegroundDiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            CE_WEIGHT * self.cross_entropy(logits.float(), target)
            + DICE_WEIGHT * self.dice(logits, target)
        )


def primary_logits(output: Any, target_size: tuple[int, int]) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        if not output:
            raise RuntimeError("Model returned an empty output sequence")
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError(f"Unsupported model output type: {type(output)!r}")
    if output.ndim != 4:
        raise RuntimeError(f"Expected BCHW logits, got {tuple(output.shape)}")
    if output.shape[-2:] != target_size:
        output = F.interpolate(
            output,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
    return output


@dataclass
class MetricAccumulator:
    threshold: float = FIXED_THRESHOLD
    count: int = 0
    dice_sum: float = 0.0
    iou_sum: float = 0.0
    global_tp: int = 0
    global_fp: int = 0
    global_fn: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)

    def update(
        self,
        probabilities: torch.Tensor,
        target: torch.Tensor,
        names: Iterable[str],
        store_rows: bool,
    ) -> None:
        prediction = probabilities >= self.threshold
        target_bool = target.bool()
        batch_size = target.shape[0]
        prediction_flat = prediction.reshape(batch_size, -1)
        target_flat = target_bool.reshape(batch_size, -1)
        tp = (prediction_flat & target_flat).sum(dim=1)
        fp = (prediction_flat & ~target_flat).sum(dim=1)
        fn = (~prediction_flat & target_flat).sum(dim=1)
        denominator = 2 * tp + fp + fn
        union = tp + fp + fn
        dice = torch.where(
            denominator > 0,
            2.0 * tp.double() / denominator.clamp_min(1).double(),
            torch.ones_like(denominator, dtype=torch.float64),
        )
        iou = torch.where(
            union > 0,
            tp.double() / union.clamp_min(1).double(),
            torch.ones_like(union, dtype=torch.float64),
        )
        self.count += batch_size
        self.dice_sum += float(dice.sum().item())
        self.iou_sum += float(iou.sum().item())
        self.global_tp += int(tp.sum().item())
        self.global_fp += int(fp.sum().item())
        self.global_fn += int(fn.sum().item())

        if store_rows:
            name_list = list(names)
            for index in range(batch_size):
                self.rows.append(
                    {
                        "case_name": name_list[index],
                        "dice": float(dice[index].item()),
                        "iou": float(iou[index].item()),
                        "tp": int(tp[index].item()),
                        "fp": int(fp[index].item()),
                        "fn": int(fn[index].item()),
                        "gt_pixels": int(target_flat[index].sum().item()),
                        "pred_pixels": int(prediction_flat[index].sum().item()),
                        "threshold": self.threshold,
                    }
                )

    def summary(self) -> dict[str, Any]:
        precision = self.global_tp / max(self.global_tp + self.global_fp, 1)
        recall = self.global_tp / max(self.global_tp + self.global_fn, 1)
        global_dice = (
            2 * self.global_tp
            / max(2 * self.global_tp + self.global_fp + self.global_fn, 1)
        )
        global_iou = self.global_tp / max(
            self.global_tp + self.global_fp + self.global_fn, 1
        )
        return {
            "threshold": self.threshold,
            "image_count": self.count,
            "whole_image_dice_mean": self.dice_sum / max(self.count, 1),
            "whole_image_iou_mean": self.iou_sum / max(self.count, 1),
            "global_dice": global_dice,
            "global_iou": global_iou,
            "precision": precision,
            "recall": recall,
            "global_tp": self.global_tp,
            "global_fp": self.global_fp,
            "global_fn": self.global_fn,
        }


def amp_dtype_from_name(name: str, device: torch.device) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU/PyTorch build does not support BF16")
        return torch.bfloat16
    return torch.float16


def autocast_context(dtype: torch.dtype, device: torch.device):
    if device.type != "cuda" or dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_grad_scaler(dtype: torch.dtype):
    enabled = dtype == torch.float16
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
) -> LambdaLR:
    def multiplier(epoch_index: int) -> float:
        if warmup_epochs > 0 and epoch_index < warmup_epochs:
            return max((epoch_index + 1) / warmup_epochs, 1e-3)
        remaining = max(epochs - warmup_epochs, 1)
        progress = min(
            max((epoch_index - warmup_epochs) / remaining, 0.0),
            1.0,
        )
        return max(0.5 * (1.0 + math.cos(math.pi * progress)), 0.01)

    return LambdaLR(optimizer, lr_lambda=multiplier)


def apply_reference_aligned_u_rwkv_patch() -> str:
    """Align APRIL's U-RWKV SpatialMix with the public U-RWKV code.

    APRIL's reimplementation passes raw ``spatial_decay`` and
    ``spatial_first`` into a WKV sequence whose first stage has T=65536 for a
    512x512 image. The public implementation divides both by T and applies
    LayerNorm to the WKV output, not to k before WKV. The missing scaling
    causes resolution-dependent gradients and is especially unstable at 512.

    This runtime patch is deliberately local to the special-architecture
    ``u_rwkv`` model. It does not modify RWKV-UNet or files under the APRIL
    checkout.
    """
    from medseg.models.networks.rwkv import u_rwkv as u_rwkv_module

    spatial_mix_class = u_rwkv_module.SpatialMix
    patch_id = "public_u_rwkv_spatialmix_decay_first_div_T_post_wkv_ln_v1"
    if getattr(spatial_mix_class, "_size512_reference_patch", None) == patch_id:
        return patch_id

    def reference_aligned_forward(
        self: nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch, tokens, channels = x.shape
        if self.shift_pixel > 0:
            shifted = u_rwkv_module.q_shift(x, self.shift_pixel)
        else:
            shifted = x

        xk = x * self.spatial_mix_k + shifted * (1 - self.spatial_mix_k)
        xv = x * self.spatial_mix_v + shifted * (1 - self.spatial_mix_v)
        xr = x * self.spatial_mix_r + shifted * (1 - self.spatial_mix_r)

        key = self.key(xk)
        value = self.value(xv)
        receptance = self.receptance(xr)
        gate = torch.sigmoid(receptance)

        rwkv = u_rwkv_module.wkv_pytorch(
            batch,
            tokens,
            channels,
            self.spatial_decay.float() / tokens,
            self.spatial_first.float() / tokens,
            key.float(),
            value.float(),
        ).to(x.dtype)
        if self.key_norm is not None:
            rwkv = self.key_norm(rwkv)
        return self.output(gate * rwkv)

    spatial_mix_class.forward = reference_aligned_forward
    spatial_mix_class._size512_reference_patch = patch_id
    return patch_id


def model_config(model_name: str) -> dict[str, Any]:
    spec = MODEL_SPECS[model_name]
    return {
        "model": {
            "architecture": spec["architecture"],
            "num_classes": NUM_CLASSES,
            "img_size": IMAGE_SIZE,
            "encoder": {"in_channels": 3, "pretrained": False},
            "arch_params": dict(spec["arch_params"]),
        }
    }


def require_wkv_cuda(model_name: str) -> None:
    from torch.utils.cpp_extension import CUDA_HOME
    from medseg.kernels.wkv import (
        get_load_error,
        is_cuda_available,
        load_wkv_cuda,
    )

    if CUDA_HOME is None or shutil.which("nvcc") is None:
        raise RuntimeError(
            "RWKV requires a local CUDA toolkit/NVCC. "
            "Run check_rwkv_bench_env_v1.py first."
        )
    t_max = int(MODEL_SPECS[model_name]["kernel_tmax"])
    print(f"Loading WKV CUDA kernel (Tmax={t_max}) ...")
    op = load_wkv_cuda(t_max=t_max, force=True, verbose=True)
    if op is None or not is_cuda_available():
        raise RuntimeError(
            f"WKV CUDA kernel unavailable; refusing PyTorch fallback: "
            f"{get_load_error()!r}"
        )
    print("WKV CUDA kernel: OK")


def build_dataloaders(
    data_root: Path,
    *,
    augment: bool,
    min_foreground_pixels: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
    max_test_samples: int | None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    train_dataset = ConsolidationDataset(
        data_root / "train",
        augment=augment,
        min_foreground_pixels=min_foreground_pixels,
        max_samples=max_train_samples,
    )
    val_dataset = ConsolidationDataset(
        data_root / "val",
        augment=False,
        min_foreground_pixels=min_foreground_pixels,
        max_samples=max_val_samples,
    )
    test_dataset = ConsolidationDataset(
        data_root / "test",
        augment=False,
        min_foreground_pixels=min_foreground_pixels,
        max_samples=max_test_samples,
    )
    generator = torch.Generator().manual_seed(seed)
    common = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
        "worker_init_fn": worker_init_fn,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        sampler=None,
        drop_last=False,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    counts = {
        "train": len(train_dataset),
        "val": len(val_dataset),
        "test": len(test_dataset),
        "excluded_below_min_pixels": {
            "train": train_dataset.excluded,
            "val": val_dataset.excluded,
            "test": test_dataset.excluded,
        },
    }
    return train_loader, val_loader, test_loader, counts


def inspect_nonfinite_gradients(
    model: nn.Module,
) -> tuple[list[str], int]:
    """Return parameter names and element count for NaN/Inf gradients."""
    names: list[str] = []
    element_count = 0
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        finite = torch.isfinite(gradient)
        if bool(finite.all().item()):
            continue
        names.append(name)
        element_count += int((~finite).sum().item())
    return names, element_count


def largest_gradient_statistics(
    model: nn.Module,
    limit: int = 10,
) -> list[str]:
    """Return compact max-absolute-gradient diagnostics for exceptional steps."""
    rows: list[tuple[float, str]] = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        finite_gradient = torch.nan_to_num(
            gradient.detach().float(),
            nan=0.0,
            posinf=torch.finfo(torch.float32).max,
            neginf=-torch.finfo(torch.float32).max,
        )
        max_abs = float(finite_gradient.abs().max().item())
        rows.append((max_abs, name))
    rows.sort(key=lambda item: item[0], reverse=True)
    return [f"{name}:{max_abs:.9g}" for max_abs, name in rows[:limit]]


def stable_clip_grad_norm_(
    model: nn.Module,
    max_norm: float,
) -> float:
    """Clip finite gradients using a scaled L2 norm that cannot overflow.

    ``torch.nn.utils.clip_grad_norm_`` computes the total norm in the
    gradients' dtype. A very large but still finite FP32 gradient can therefore
    produce an infinite norm. Scaling by the global maximum before summing
    squares preserves the direction and avoids that false non-finite result.
    This slower path is used only after the ordinary norm reports NaN/Inf.
    """
    gradients = [
        parameter.grad.detach()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        return 0.0

    finite_flags = torch.stack(
        [torch.isfinite(gradient).all() for gradient in gradients]
    )
    if not bool(finite_flags.all().item()):
        return float("nan")

    maxima = torch.stack(
        [gradient.abs().max().float() for gradient in gradients]
    )
    max_abs_tensor = maxima.max()
    max_abs = float(max_abs_tensor.item())
    if max_abs == 0.0:
        return 0.0

    scaled_square_sums = torch.stack(
        [
            (gradient.float() / max_abs_tensor).square().sum()
            for gradient in gradients
        ]
    )
    scaled_square_sum = float(scaled_square_sums.sum().item())
    total_norm = max_abs * math.sqrt(max(scaled_square_sum, 0.0))
    if not math.isfinite(total_norm):
        return total_norm

    clip_coefficient = min(max_norm / (total_norm + 1e-6), 1.0)
    for gradient in gradients:
        gradient.mul_(clip_coefficient)
    return total_norm


def replay_accumulation_group_fp32(
    model: nn.Module,
    criterion: nn.Module,
    replay_group: Sequence[tuple[torch.Tensor, torch.Tensor]],
    grad_clip: float,
) -> tuple[float, list[str], int]:
    """Recompute one unchanged accumulation group without autocast."""
    model.train()
    group_size = len(replay_group)
    if group_size < 1:
        raise RuntimeError("Cannot replay an empty accumulation group")

    for image, target in replay_group:
        logits = primary_logits(model(image), target.shape[-2:])
        full_loss = criterion(logits, target)
        if not torch.isfinite(full_loss):
            raise FloatingPointError(
                "FP32 retry produced a non-finite loss: "
                f"{float(full_loss.item())}"
            )
        (full_loss / group_size).backward()

    nonfinite_names, nonfinite_elements = inspect_nonfinite_gradients(model)
    if nonfinite_names:
        return float("nan"), nonfinite_names, nonfinite_elements
    grad_norm = stable_clip_grad_norm_(model, grad_clip)
    return grad_norm, nonfinite_names, nonfinite_elements


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    amp_dtype: torch.dtype,
    grad_accum: int,
    grad_clip: float,
    epoch: int,
    *,
    diagnostic_path: Path,
    max_fp32_retries: int,
    max_stable_norm_recoveries: int,
) -> tuple[float, dict[str, int]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_samples = 0
    stable_norm_recoveries = 0
    fp32_retries = 0
    progress = tqdm(loader, desc=f"train epoch {epoch}", unit="batch")
    accumulation_group_size = grad_accum
    replay_group: list[tuple[torch.Tensor, torch.Tensor]] = []
    replay_names: list[str] = []
    replay_losses: list[float] = []
    diagnostic_fields = [
        "epoch",
        "step",
        "action",
        "sample_names",
        "sample_losses",
        "learning_rate",
        "ordinary_grad_norm",
        "stable_or_retry_grad_norm",
        "nonfinite_parameter_count",
        "nonfinite_element_count",
        "nonfinite_parameters",
        "largest_gradient_parameters",
    ]
    for step, batch in enumerate(progress):
        if step % grad_accum == 0:
            accumulation_group_size = min(grad_accum, len(loader) - step)
            replay_group = []
            replay_names = []
            replay_losses = []
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        replay_group.append((image.detach(), target.detach()))
        names = batch["name"]
        if isinstance(names, str):
            replay_names.append(names)
        else:
            replay_names.extend(str(name) for name in names)
        with autocast_context(amp_dtype, device):
            logits = primary_logits(model(image), target.shape[-2:])
            full_loss = criterion(logits, target)
            loss = full_loss / accumulation_group_size
        if not torch.isfinite(full_loss):
            raise FloatingPointError(
                f"Non-finite loss at epoch={epoch}, step={step}: "
                f"{full_loss.item()}"
            )
        replay_losses.append(float(full_loss.item()))
        scaler.scale(loss).backward()

        should_step = (
            (step + 1) % grad_accum == 0 or step + 1 == len(loader)
        )
        if should_step:
            scaler.unscale_(optimizer)
            try:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                    error_if_nonfinite=True,
                )
                grad_norm = float(grad_norm_tensor.item())
                scaler.step(optimizer)
                scaler.update()
            except RuntimeError as ordinary_norm_error:
                nonfinite_names, nonfinite_elements = (
                    inspect_nonfinite_gradients(model)
                )
                largest_gradients = largest_gradient_statistics(model)
                ordinary_grad_norm = "non-finite"

                if not nonfinite_names:
                    grad_norm = stable_clip_grad_norm_(model, grad_clip)
                    if not math.isfinite(grad_norm):
                        raise FloatingPointError(
                            "Gradient elements are finite, but even the stable "
                            f"norm is non-finite at epoch={epoch}, step={step}"
                        ) from ordinary_norm_error
                    if (
                        stable_norm_recoveries
                        >= max_stable_norm_recoveries
                    ):
                        optimizer.zero_grad(set_to_none=True)
                        append_csv(
                            diagnostic_path,
                            {
                                "epoch": epoch,
                                "step": step,
                                "action": "stable_norm_limit_exceeded",
                                "sample_names": "|".join(replay_names),
                                "sample_losses": "|".join(
                                    f"{value:.9g}"
                                    for value in replay_losses
                                ),
                                "learning_rate": optimizer.param_groups[0][
                                    "lr"
                                ],
                                "ordinary_grad_norm": ordinary_grad_norm,
                                "stable_or_retry_grad_norm": grad_norm,
                                "nonfinite_parameter_count": 0,
                                "nonfinite_element_count": 0,
                                "nonfinite_parameters": "",
                                "largest_gradient_parameters": "|".join(
                                    largest_gradients
                                ),
                            },
                            diagnostic_fields,
                        )
                        raise FloatingPointError(
                            "Exceeded stable-norm recovery limit at "
                            f"epoch={epoch}, step={step}; "
                            f"stable_grad_norm={grad_norm:.9g}, "
                            f"largest_gradients={largest_gradients}, "
                            f"samples={replay_names}"
                        ) from ordinary_norm_error
                    stable_norm_recoveries += 1
                    optimizer.step()
                    scaler.update()
                    action = "stable_norm_clip"
                else:
                    if fp32_retries >= max_fp32_retries:
                        raise FloatingPointError(
                            "Exceeded FP32 retry limit at "
                            f"epoch={epoch}, step={step}; "
                            f"nonfinite_parameters={nonfinite_names[:20]}, "
                            f"samples={replay_names}"
                        ) from ordinary_norm_error
                    optimizer.zero_grad(set_to_none=True)
                    grad_norm, retry_nonfinite_names, retry_nonfinite_elements = (
                        replay_accumulation_group_fp32(
                            model,
                            criterion,
                            replay_group,
                            grad_clip,
                        )
                    )
                    if retry_nonfinite_names or not math.isfinite(grad_norm):
                        retry_largest_gradients = (
                            largest_gradient_statistics(model)
                        )
                        optimizer.zero_grad(set_to_none=True)
                        append_csv(
                            diagnostic_path,
                            {
                                "epoch": epoch,
                                "step": step,
                                "action": "fp32_retry_failed",
                                "sample_names": "|".join(replay_names),
                                "sample_losses": "|".join(
                                    f"{value:.9g}" for value in replay_losses
                                ),
                                "learning_rate": optimizer.param_groups[0][
                                    "lr"
                                ],
                                "ordinary_grad_norm": ordinary_grad_norm,
                                "stable_or_retry_grad_norm": grad_norm,
                                "nonfinite_parameter_count": len(
                                    retry_nonfinite_names
                                ),
                                "nonfinite_element_count": (
                                    retry_nonfinite_elements
                                ),
                                "nonfinite_parameters": "|".join(
                                    retry_nonfinite_names[:50]
                                ),
                                "largest_gradient_parameters": "|".join(
                                    retry_largest_gradients
                                ),
                            },
                            diagnostic_fields,
                        )
                        raise FloatingPointError(
                            "FP32 retry still produced non-finite gradients at "
                            f"epoch={epoch}, step={step}; "
                            f"parameters={retry_nonfinite_names[:20]}, "
                            f"samples={replay_names}"
                        ) from ordinary_norm_error
                    fp32_retries += 1
                    optimizer.step()
                    scaler.update()
                    action = "fp32_retry_succeeded"

                append_csv(
                    diagnostic_path,
                    {
                        "epoch": epoch,
                        "step": step,
                        "action": action,
                        "sample_names": "|".join(replay_names),
                        "sample_losses": "|".join(
                            f"{value:.9g}" for value in replay_losses
                        ),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "ordinary_grad_norm": ordinary_grad_norm,
                        "stable_or_retry_grad_norm": grad_norm,
                        "nonfinite_parameter_count": len(nonfinite_names),
                        "nonfinite_element_count": nonfinite_elements,
                        "nonfinite_parameters": "|".join(
                            nonfinite_names[:50]
                        ),
                        "largest_gradient_parameters": "|".join(
                            largest_gradients
                        ),
                    },
                    diagnostic_fields,
                )
                print(
                    "\nRecovered non-finite ordinary gradient norm: "
                    f"epoch={epoch}, step={step}, action={action}, "
                    f"grad_norm={grad_norm:.6g}, samples={replay_names}"
                )
            optimizer.zero_grad(set_to_none=True)
            replay_group = []
            replay_names = []
            replay_losses = []

        batch_size = image.shape[0]
        total_loss += float(full_loss.item()) * batch_size
        total_samples += batch_size
        progress.set_postfix(
            loss=f"{total_loss / max(total_samples, 1):.5f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )
    return total_loss / max(total_samples, 1), {
        "stable_norm_recoveries": stable_norm_recoveries,
        "fp32_retries": fp32_retries,
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    *,
    store_rows: bool,
    description: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    accumulator = MetricAccumulator()
    for batch in tqdm(loader, desc=description, unit="batch", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        with autocast_context(amp_dtype, device):
            logits = primary_logits(model(image), target.shape[-2:])
        probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
        accumulator.update(
            probabilities,
            target,
            batch["name"],
            store_rows,
        )
    return accumulator.summary(), accumulator.rows


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: Any,
    *,
    epoch: int,
    best_epoch: int,
    best_score: float,
    model_name: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "model_name": model_name,
        "model_config": model_config(model_name),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "settings": settings,
    }


def load_checkpoint(
    path: Path,
    model: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: LambdaLR | None = None,
    scaler: Any = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in payload:
        scaler.load_state_dict(payload["scaler_state_dict"])
    return payload


def dice_bin_rows(case_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = [0] * 10
    for row in case_rows:
        dice = min(max(float(row["dice"]), 0.0), 1.0)
        index = min(int(dice * 10), 9)
        counts[index] += 1
    total = max(len(case_rows), 1)
    return [
        {
            "dice_bin": f"{index * 10:02d}-{(index + 1) * 10:02d}%",
            "count": count,
            "fraction": count / total,
        }
        for index, count in enumerate(counts)
    ]


def train_model(
    model_name: str,
    *,
    args: argparse.Namespace,
    preset: dict[str, Any],
    output_root: Path,
    data_root: Path,
    device: torch.device,
    amp_dtype: torch.dtype,
    april_commit: str,
) -> dict[str, Any]:
    from medseg.model_builder import build_model

    spec = dict(MODEL_SPECS[model_name])
    batch_size = args.batch_size or int(spec["batch_size"])
    grad_accum = args.grad_accum or int(spec["grad_accum"])
    model_dir = output_root / model_name
    result_path = model_dir / "result.json"
    if args.skip_completed and result_path.is_file():
        with result_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    model_dir.mkdir(parents=True, exist_ok=True)

    require_wkv_cuda(model_name)
    u_rwkv_patch_id = None
    if model_name == "u_rwkv":
        u_rwkv_patch_id = apply_reference_aligned_u_rwkv_patch()
        print(f"U-RWKV reference-alignment patch: {u_rwkv_patch_id}")
    train_loader, val_loader, test_loader, counts = build_dataloaders(
        data_root,
        augment=not args.no_augment,
        min_foreground_pixels=args.min_foreground_pixels,
        batch_size=batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_train_samples=preset["max_train_samples"],
        max_val_samples=preset["max_val_samples"],
        max_test_samples=preset["max_test_samples"],
    )
    model = build_model(model_config(model_name)).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    criterion = ConsolidationLoss(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = make_scheduler(
        optimizer,
        int(preset["epochs"]),
        int(preset["warmup_epochs"]),
    )
    scaler = make_grad_scaler(amp_dtype)

    settings = {
        "model": model_name,
        "architecture": spec["architecture"],
        "arch_params": spec["arch_params"],
        "pretrained": False,
        "april_commit": april_commit,
        "data_root": str(data_root),
        "output_root": str(output_root),
        "image_size": IMAGE_SIZE,
        "run_mode": args.run_mode,
        "epochs": preset["epochs"],
        "warmup_epochs": preset["warmup_epochs"],
        "patience": preset["patience"],
        "samples_per_epoch": None,
        "balanced_sampler": False,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "nonfinite_gradient_policy": (
            "stable norm for finite gradients; otherwise replay the same "
            "accumulation group once in FP32; never skip samples"
        ),
        "max_fp32_retries_per_epoch": args.max_fp32_retries_per_epoch,
        "max_stable_norm_recoveries_per_epoch": (
            args.max_stable_norm_recoveries_per_epoch
        ),
        "u_rwkv_reference_alignment_patch": u_rwkv_patch_id,
        "u_rwkv_implementation_note": (
            "APRIL reimplementation with public-reference SpatialMix "
            "decay/T, first/T, and post-WKV LayerNorm alignment"
            if model_name == "u_rwkv"
            else None
        ),
        "amp_dtype": args.amp_dtype,
        "fixed_threshold": FIXED_THRESHOLD,
        "checkpoint_metric": "val_whole_image_dice_mean",
        "min_foreground_pixels": args.min_foreground_pixels,
        "augmentation": not args.no_augment,
        "dataset_counts": counts,
        "seed": args.seed,
    }
    save_json(model_dir / "run_settings.json", settings)

    print("\n" + "=" * 88)
    print(f"MODEL: {model_name}")
    print(
        f"parameters={parameter_count / 1e6:.2f} M, "
        f"trainable={trainable_count / 1e6:.2f} M"
    )
    print(
        f"dataset train/val/test={counts['train']}/{counts['val']}/"
        f"{counts['test']}"
    )
    print(
        f"batch={batch_size}, grad_accum={grad_accum}, "
        f"effective_batch={batch_size * grad_accum}, "
        f"amp={args.amp_dtype}, lr={args.lr:.2e}"
    )
    print("sampler=shuffle without replacement; samples_per_epoch=None")
    print("checkpoint metric=mean per-image validation Dice at threshold 0.5")

    best_path = model_dir / "best_model.pth"
    last_path = model_dir / "last_model.pth"
    history_path = model_dir / "history.csv"
    start_epoch = 1
    best_epoch = 0
    best_score = -float("inf")
    no_improvement = 0
    if args.resume and not last_path.is_file():
        raise FileNotFoundError(
            f"--resume was requested, but checkpoint is missing: {last_path}"
        )
    if args.resume:
        payload = load_checkpoint(
            last_path, model, device, optimizer, scheduler, scaler
        )
        if model_name == "u_rwkv":
            checkpoint_patch_id = payload.get("settings", {}).get(
                "u_rwkv_reference_alignment_patch"
            )
            if checkpoint_patch_id != u_rwkv_patch_id:
                raise RuntimeError(
                    "Refusing to resume U-RWKV from a checkpoint created "
                    "before the reference-alignment patch. Start a new v3 "
                    "output directory and train from epoch 1. "
                    f"checkpoint_patch={checkpoint_patch_id!r}, "
                    f"required_patch={u_rwkv_patch_id!r}"
                )
        start_epoch = int(payload["epoch"]) + 1
        best_epoch = int(payload.get("best_epoch", 0))
        best_score = float(payload.get("best_score", -float("inf")))
        no_improvement = max(0, int(payload["epoch"]) - best_epoch)
        print(
            f"resumed from epoch {payload['epoch']}; "
            f"best_epoch={best_epoch}, best_score={best_score:.6f}"
        )

    history_fields = [
        "epoch",
        "train_loss",
        "val_whole_image_dice_mean",
        "val_whole_image_iou_mean",
        "val_global_dice",
        "val_precision",
        "val_recall",
        "lr",
        "seconds",
        "peak_memory_gib",
        "stable_norm_recoveries",
        "fp32_retries",
        "best_epoch",
        "best_score",
    ]
    train_start = time.time()
    peak_memory_gib = 0.0
    for epoch in range(start_epoch, int(preset["epochs"]) + 1):
        epoch_start = time.time()
        torch.cuda.reset_peak_memory_stats(device)
        train_loss, recovery_counts = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp_dtype,
            grad_accum,
            args.grad_clip,
            epoch,
            diagnostic_path=model_dir / "nonfinite_gradient_events.csv",
            max_fp32_retries=args.max_fp32_retries_per_epoch,
            max_stable_norm_recoveries=(
                args.max_stable_norm_recoveries_per_epoch
            ),
        )
        val_metrics, _ = evaluate(
            model,
            val_loader,
            device,
            amp_dtype,
            store_rows=False,
            description=f"val epoch {epoch}",
        )
        score = float(val_metrics["whole_image_dice_mean"])
        if not math.isfinite(score):
            raise FloatingPointError(f"Non-finite validation Dice: {score}")
        if score > best_score + 1e-4:
            best_score = score
            best_epoch = epoch
            no_improvement = 0
            torch.save(
                checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_score=best_score,
                    model_name=model_name,
                    settings=settings,
                ),
                best_path,
            )
            print(f"New best: epoch={epoch}, val_dice={best_score:.6f}")
        else:
            no_improvement += 1
        scheduler.step()
        torch.save(
            checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=epoch,
                best_epoch=best_epoch,
                best_score=best_score,
                model_name=model_name,
                settings=settings,
            ),
            last_path,
        )
        epoch_peak = torch.cuda.max_memory_allocated(device) / (1024**3)
        peak_memory_gib = max(peak_memory_gib, epoch_peak)
        elapsed = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_whole_image_dice_mean": val_metrics[
                "whole_image_dice_mean"
            ],
            "val_whole_image_iou_mean": val_metrics[
                "whole_image_iou_mean"
            ],
            "val_global_dice": val_metrics["global_dice"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
            "peak_memory_gib": epoch_peak,
            "stable_norm_recoveries": recovery_counts[
                "stable_norm_recoveries"
            ],
            "fp32_retries": recovery_counts["fp32_retries"],
            "best_epoch": best_epoch,
            "best_score": best_score,
        }
        append_csv(history_path, row, history_fields)
        print(
            f"Epoch {epoch:03d}/{preset['epochs']} | "
            f"loss={train_loss:.5f} | "
            f"val_dice={score:.5f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"{elapsed:.1f}s | peak={epoch_peak:.2f} GiB | "
            f"no_improve={no_improvement}/{preset['patience']}"
        )
        if (
            int(preset["patience"]) > 0
            and no_improvement >= int(preset["patience"])
        ):
            print(f"Early stopping at epoch {epoch}")
            break

    if not best_path.is_file():
        raise RuntimeError("No best checkpoint was saved")
    load_checkpoint(best_path, model, device)
    val_metrics, val_rows = evaluate(
        model,
        val_loader,
        device,
        amp_dtype,
        store_rows=True,
        description="final val",
    )
    test_metrics, test_rows = evaluate(
        model,
        test_loader,
        device,
        amp_dtype,
        store_rows=True,
        description="test",
    )
    write_csv(model_dir / "val_cases.csv", val_rows)
    write_csv(model_dir / "test_cases.csv", test_rows)
    write_csv(model_dir / "test_dice_bins.csv", dice_bin_rows(test_rows))

    result = {
        "model": model_name,
        "architecture": spec["architecture"],
        "best_epoch": best_epoch,
        "best_val_whole_image_dice_mean": best_score,
        "final_val": val_metrics,
        "test": test_metrics,
        "threshold": FIXED_THRESHOLD,
        "dataset_counts": counts,
        "parameters_total": parameter_count,
        "parameters_trainable": trainable_count,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "amp_dtype": args.amp_dtype,
        "peak_memory_gib": peak_memory_gib,
        "total_seconds": time.time() - train_start,
        "output_dir": str(model_dir),
    }
    save_json(result_path, result)
    print(
        f"FINAL {model_name}: "
        f"test_dice={test_metrics['whole_image_dice_mean']:.6f}, "
        f"global_dice={test_metrics['global_dice']:.6f}, "
        f"precision={test_metrics['precision']:.6f}, "
        f"recall={test_metrics['recall']:.6f}"
    )
    return result


def summary_rows(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        test = result["test"]
        rows.append(
            {
                "model": result["model"],
                "best_epoch": result["best_epoch"],
                "val_whole_image_dice_mean": result[
                    "best_val_whole_image_dice_mean"
                ],
                "test_whole_image_dice_mean": test[
                    "whole_image_dice_mean"
                ],
                "test_whole_image_iou_mean": test["whole_image_iou_mean"],
                "test_global_dice": test["global_dice"],
                "test_precision": test["precision"],
                "test_recall": test["recall"],
                "parameters_m": result["parameters_total"] / 1e6,
                "peak_memory_gib": result["peak_memory_gib"],
                "hours": result["total_seconds"] / 3600.0,
                "output_dir": result["output_dir"],
            }
        )
    rows.sort(
        key=lambda row: float(row["test_whole_image_dice_mean"]),
        reverse=True,
    )
    return rows


def main() -> None:
    args = parse_args()
    if args.max_fp32_retries_per_epoch < 0:
        raise ValueError("--max-fp32-retries-per-epoch must be >= 0")
    if args.max_stable_norm_recoveries_per_epoch < 0:
        raise ValueError(
            "--max-stable-norm-recoveries-per-epoch must be >= 0"
        )
    april_root = args.april_root.expanduser().resolve()
    if not (april_root / "medseg" / "model_builder.py").is_file():
        raise FileNotFoundError(
            f"Not an APRIL-MedSeg root: {april_root}"
        )
    sys.path.insert(0, str(april_root))
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(
            f"Invalid --gpu {args.gpu}; visible GPUs={torch.cuda.device_count()}"
        )
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    capability = torch.cuda.get_device_capability(device)
    os.environ.setdefault(
        "TORCH_CUDA_ARCH_LIST", f"{capability[0]}.{capability[1]}"
    )
    os.environ.setdefault("MAX_JOBS", "4")

    seed_everything(args.seed, args.deterministic)
    amp_dtype = amp_dtype_from_name(args.amp_dtype, device)
    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root is not None
        else (
            april_root
            / "./datasets/Size_512"
        ).resolve()
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (
            april_root
            / "output"
            / f"consolidation_size512_rwkv_benchmark_{timestamp}"
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)
    if args.resume and args.output_dir is None:
        raise ValueError("--resume requires an explicit --output-dir")

    preset = dict(PRESETS[args.run_mode])
    for key in ("epochs", "warmup_epochs", "patience"):
        value = getattr(args, key)
        if value is not None:
            preset[key] = value
    for key in (
        "max_train_samples",
        "max_val_samples",
        "max_test_samples",
    ):
        value = getattr(args, key)
        if value is not None:
            preset[key] = value

    april_commit = command_output(
        ["git", "-C", str(april_root), "rev-parse", "--short", "HEAD"]
    )
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"compute capability: {capability[0]}.{capability[1]}")
    print(f"APRIL root: {april_root}")
    print(f"APRIL commit: {april_commit}")
    print(f"data root: {data_root}")
    print(f"output root: {output_root}")
    print(f"models: {args.models}")
    print(f"run mode: {args.run_mode}, preset={preset}")

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for model_name in args.models:
        try:
            results.append(
                train_model(
                    model_name,
                    args=args,
                    preset=preset,
                    output_root=output_root,
                    data_root=data_root,
                    device=device,
                    amp_dtype=amp_dtype,
                    april_commit=april_commit,
                )
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            failure = {
                "model": model_name,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            save_json(output_root / model_name / "failure.json", failure)
            if args.fail_fast:
                raise
            traceback.print_exc()
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    if results:
        write_csv(output_root / "benchmark_summary.csv", summary_rows(results))
    save_json(
        output_root / "run_report.json",
        {"results": results, "failures": failures},
    )
    print(
        f"Completed={len(results)}, failed={len(failures)}, "
        f"output={output_root}"
    )
    for failure in failures:
        print(f"FAILED {failure['model']}: {failure['error']}")


if __name__ == "__main__":
    main()
