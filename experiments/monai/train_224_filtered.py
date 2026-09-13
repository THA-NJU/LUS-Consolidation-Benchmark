#!/usr/bin/env python3
"""
Unified Size_224 patch-level consolidation benchmark for APRIL-MedSeg.

FINAL Size_224 protocol
-----------------------
Dataset:
  ./datasets/Size_224_filtered
  train/images, train/masks
  val/images,   val/masks
  test/images,  test/masks

Every retained 224x224 patch is a benchmark sample.
There is NO reconstruction to 512x512.

Checkpoint selection:
  fixed threshold = 0.5
  best checkpoint = highest validation mean per-patch Dice

Final test:
  load the validation-selected best checkpoint and evaluate directly on all
  test 224x224 patches. Report:
    mean Dice, mean IoU,
    global Dice, global IoU,
    micro precision, recall, specificity,
    macro precision, recall, specificity,
    TP, FP, FN, TN,
    Dice quartiles and Dice-bin histogram.

Training:
  formal: 600 epochs maximum, warmup 10, patience 15
  every formal epoch traverses every training patch exactly once
  shuffle=True, no replacement sampler, no positive/negative balancing
  CE + Dice loss (0.4 / 0.6), foreground CE weight=5
  effective batch size=4 by default (batch=1, grad_accum=4)
  fixed seed=42

Families:
  MONAI:
    monai_unet
    monai_unetplusplus
    monai_attention_unet
    monai_vnet

  Mamba/SSM:
    swin_umamba
    mamba_unet
    vm_unet_v2
    nnmamba_2d

DCM Net is intentionally not part of the default benchmark because the prior
Size_512 run was unstable. It can be added later only if the benchmark policy
is explicitly changed.

Use ``tools/train.py`` as the public entrypoint; it selects this backend and
passes the requested model, track, dataset root, and run mode.
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
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm


# =============================================================================
# Fixed benchmark protocol
# =============================================================================

IMAGE_SIZE = 224
NUM_CLASSES = 2
IN_CHANNELS = 3  # replicated grayscale for a common loader
FIXED_THRESHOLD = 0.50

CE_WEIGHT = 0.40
DICE_WEIGHT = 0.60
FOREGROUND_CE_WEIGHT = 5.0

AUG_HORIZONTAL_FLIP_P = 0.50
AUG_AFFINE_P = 0.35
AUG_MAX_ROTATE_DEG = 8.0
AUG_MAX_TRANSLATE_FRACTION = 0.03
AUG_SCALE_RANGE = (0.95, 1.05)
AUG_INTENSITY_P = 0.35
AUG_GAMMA_RANGE = (0.85, 1.20)
AUG_BRIGHTNESS_RANGE = (0.90, 1.10)
AUG_CONTRAST_RANGE = (0.90, 1.10)
AUG_NOISE_P = 0.15
AUG_NOISE_STD = 0.02
AUG_BLUR_P = 0.10

EARLY_STOPPING_MIN_DELTA = 1e-4
MAX_NONFINITE_BATCHES_PER_EPOCH = 2

PRESETS: Dict[str, Dict[str, Optional[int]]] = {
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

MONAI_MODELS = (
    "monai_unet",
    "monai_unetplusplus",
    "monai_attention_unet",
    "monai_vnet",
)

MAMBA_MODELS = (
    "swin_umamba",
    "mamba_unet",
    "vm_unet_v2",
    "nnmamba_2d",
)

ALL_MODELS = MONAI_MODELS + MAMBA_MODELS


MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    # ------------------------------- MONAI -----------------------------------
    "monai_unet": {
        "family": "monai",
        "architecture": "benchmark_monai_unet",
        "arch_params": {
            "channels": [32, 64, 128, 256, 512],
            "strides": [2, 2, 2, 2],
            "num_res_units": 2,
        },
        "batch_size": 1,
        "grad_accum": 4,
        "lr": 1e-4,
        "grad_clip_norm": 0.5,
        "amp_dtype": "bf16",
    },
    "monai_unetplusplus": {
        "family": "monai",
        "architecture": "benchmark_monai_unetplusplus",
        "arch_params": {
            "features": [32, 32, 64, 128, 256, 32],
            "deep_supervision": False,
        },
        "batch_size": 1,
        "grad_accum": 4,
        "lr": 1e-4,
        "grad_clip_norm": 0.5,
        "amp_dtype": "bf16",
    },
    "monai_attention_unet": {
        "family": "monai",
        "architecture": "benchmark_monai_attention_unet",
        "arch_params": {
            "channels": [32, 64, 128, 256, 512],
            "strides": [2, 2, 2, 2],
        },
        "batch_size": 1,
        "grad_accum": 4,
        "lr": 1e-4,
        "grad_clip_norm": 0.5,
        "amp_dtype": "bf16",
    },
    "monai_vnet": {
        "family": "monai",
        "architecture": "benchmark_monai_vnet",
        "arch_params": {
            "dropout_probability": 0.2,
        },
        "batch_size": 1,
        "grad_accum": 4,
        "lr": 1e-4,
        "grad_clip_norm": 0.5,
        "amp_dtype": "bf16",
    },

    # ----------------------------- Mamba/SSM ---------------------------------
    "swin_umamba": {
        "family": "mamba",
        "architecture": "swin_umamba",
        "arch_params": {
            "feat_size": [48, 96, 192, 384, 768],
            "depths": [2, 2, 2, 2],
            "d_state": 16,
            "drop_path_rate": 0.2,
            "deep_supervision": False,
        },
        "batch_size": 1,
        "grad_accum": 4,
        "lr": 2e-5,
        "grad_clip_norm": 0.5,
        "amp_dtype": "bf16",
    },
    "mamba_unet": {
        "family": "mamba",
        "architecture": "mamba_unet",
        "arch_params": {
            "embed_dim": 96,
            "depths": [2, 2, 2, 2],
            "d_state": 16,
            "drop_path_rate": 0.1,
            "deep_supervision": False,
        },
        "batch_size": 1,
        "grad_accum": 4,
        "lr": 5e-5,
        "grad_clip_norm": 0.5,
        "amp_dtype": "bf16",
    },
    "vm_unet_v2": {
        "family": "mamba",
        "architecture": "vm_unet_v2",
        "arch_params": {
            "embed_dim": 64,
            "depths": [2, 2, 6, 2],
            "mid_channel": 32,
            "drop_path_rate": 0.2,
            "deep_supervision": True,
        },
        "batch_size": 16,
        "grad_accum": 4,
        "lr": 5e-5,
        "grad_clip_norm": 0.5,
        "amp_dtype": "bf16",
    },
    "nnmamba_2d": {
        "family": "mamba",
        "architecture": "nnmamba_2d",
        "arch_params": {
            "channels": 32,
            "blocks": 3,
            "deep_supervision": False,
        },
        "batch_size": 16,
        "grad_accum": 4,
        "lr": 5e-5,
        "grad_clip_norm": 0.5,
        "amp_dtype": "bf16",
    },
}


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_data_root = (
        script_dir / "./datasets/Size_224_filtered"
    ).resolve()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Size_224 patch-level consolidation benchmark.",
    )
    parser.add_argument(
        "--april-root",
        type=Path,
        default=script_dir,
        help="APRIL-MedSeg root. Required for Mamba family.",
    )
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--family",
        choices=("monai", "mamba", "all"),
        default="monai",
        help="Used only when --models is not supplied.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=None,
        help="Explicit model list; overrides --family.",
    )
    parser.add_argument("--run-mode", choices=tuple(PRESETS), default="formal")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument(
        "--amp-dtype",
        choices=("bf16", "fp16", "fp32"),
        default=None,
        help="Override per-model default precision.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip a model if result.json already exists.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def resolve_models(args: argparse.Namespace) -> List[str]:
    if args.models:
        return list(args.models)
    if args.family == "monai":
        return list(MONAI_MODELS)
    if args.family == "mamba":
        return list(MAMBA_MODELS)
    return list(ALL_MODELS)


# =============================================================================
# Reproducibility and utilities
# =============================================================================

def seed_everything(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def make_worker_init_fn(base_seed: int):
    def _worker_init(worker_id: int) -> None:
        worker_seed = (base_seed + worker_id) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
    return _worker_init


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_csv(path: Path, row: Dict[str, Any], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def ensure_dataset_layout(root: Path) -> None:
    missing = []
    for split in ("train", "val", "test"):
        for leaf in ("images", "masks"):
            path = root / split / leaf
            if not path.is_dir():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Dataset layout is incomplete:\n  " + "\n  ".join(missing)
        )


def patient_id_from_name(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) != 3 or not parts[0].lower().startswith("p"):
        raise ValueError(
            f"Expected pxxx_xxx_xxxx.png naming, got {filename!r}"
        )
    return parts[0].lower()


def audit_patient_split(data_root: Path) -> Dict[str, Any]:
    patient_sets: Dict[str, set[str]] = {}
    sample_sets: Dict[str, set[str]] = {}
    counts: Dict[str, int] = {}

    for split in ("train", "val", "test"):
        image_dir = data_root / split / "images"
        mask_dir = data_root / split / "masks"

        image_names = {
            p.name for p in image_dir.glob("*.png") if p.is_file()
        }
        mask_names = {
            p.name for p in mask_dir.glob("*.png") if p.is_file()
        }
        if image_names != mask_names:
            raise RuntimeError(
                f"Image/mask mismatch for {split}: "
                f"images={len(image_names)}, masks={len(mask_names)}, "
                f"missing_masks={sorted(image_names-mask_names)[:10]}, "
                f"missing_images={sorted(mask_names-image_names)[:10]}"
            )
        if not image_names:
            raise RuntimeError(f"No samples found in split={split}")

        sample_sets[split] = image_names
        patient_sets[split] = {
            patient_id_from_name(name) for name in image_names
        }
        counts[split] = len(image_names)

    for i, a in enumerate(("train", "val", "test")):
        for b in ("train", "val", "test")[i + 1:]:
            sample_overlap = sample_sets[a] & sample_sets[b]
            patient_overlap = patient_sets[a] & patient_sets[b]
            if sample_overlap:
                raise RuntimeError(
                    f"Sample leakage between {a}/{b}: "
                    f"{sorted(sample_overlap)[:10]}"
                )
            if patient_overlap:
                raise RuntimeError(
                    f"Patient leakage between {a}/{b}: "
                    f"{sorted(patient_overlap)[:10]}"
                )

    return {
        "sample_counts": counts,
        "patient_counts": {
            split: len(patient_sets[split])
            for split in ("train", "val", "test")
        },
        "patient_overlap": False,
    }


# =============================================================================
# Dataset and augmentation
# =============================================================================

class Consolidation224Dataset(Dataset):
    """
    The final Size_224_filtered dataset contains only retained class3 patches.
    Every mask must contain foreground; there is no secondary area filtering.
    """

    def __init__(
        self,
        split_root: Path,
        *,
        train: bool,
        augment: bool,
        max_samples: Optional[int],
        seed: int,
    ) -> None:
        self.split_root = Path(split_root)
        self.image_dir = self.split_root / "images"
        self.mask_dir = self.split_root / "masks"
        self.train = bool(train)
        self.augment = bool(augment and train)
        self.seed = int(seed)

        image_map = {
            p.name: p
            for p in sorted(self.image_dir.glob("*.png"))
            if p.is_file()
        }
        mask_map = {
            p.name: p
            for p in sorted(self.mask_dir.glob("*.png"))
            if p.is_file()
        }

        if not image_map:
            raise RuntimeError(f"No PNG files under {self.image_dir}")
        if image_map.keys() != mask_map.keys():
            raise RuntimeError(
                f"Image/mask mismatch in {split_root}: "
                f"images={len(image_map)}, masks={len(mask_map)}"
            )

        names = sorted(image_map)
        if max_samples is not None:
            # Deterministic smoke subset, not random sampling.
            names = names[: int(max_samples)]

        self.samples: List[Tuple[Path, Path, str]] = [
            (image_map[name], mask_map[name], name)
            for name in names
        ]

        self._validate_samples()

    def _validate_samples(self) -> None:
        bad_size: List[str] = []
        empty_masks: List[str] = []

        for image_path, mask_path, name in self.samples:
            with Image.open(image_path) as img:
                if img.size != (IMAGE_SIZE, IMAGE_SIZE):
                    bad_size.append(
                        f"{name}: image={img.size}"
                    )

            with Image.open(mask_path) as mask:
                if mask.size != (IMAGE_SIZE, IMAGE_SIZE):
                    bad_size.append(
                        f"{name}: mask={mask.size}"
                    )
                mask_arr = np.asarray(mask)
                if mask_arr.ndim == 3:
                    mask_arr = mask_arr[..., 0]
                if not np.any(mask_arr > 0):
                    empty_masks.append(name)

        if bad_size:
            raise RuntimeError(
                "Non-224 files found; examples: " + ", ".join(bad_size[:10])
            )
        if empty_masks:
            raise RuntimeError(
                "Size_224_filtered must contain only foreground-positive "
                "class3 patches, but empty masks were found; examples: "
                + ", ".join(empty_masks[:10])
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _augment_pair(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        if random.random() < AUG_HORIZONTAL_FLIP_P:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < AUG_AFFINE_P:
            angle = random.uniform(
                -AUG_MAX_ROTATE_DEG, AUG_MAX_ROTATE_DEG
            )
            max_shift = int(round(
                IMAGE_SIZE * AUG_MAX_TRANSLATE_FRACTION
            ))
            translate = [
                random.randint(-max_shift, max_shift),
                random.randint(-max_shift, max_shift),
            ]
            scale = random.uniform(*AUG_SCALE_RANGE)

            image = TF.affine(
                image,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            mask = TF.affine(
                mask,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )

        if random.random() < AUG_INTENSITY_P:
            gamma = random.uniform(*AUG_GAMMA_RANGE)
            image = TF.adjust_gamma(image, gamma=gamma)

            brightness = random.uniform(*AUG_BRIGHTNESS_RANGE)
            image = ImageEnhance.Brightness(image).enhance(brightness)

            contrast = random.uniform(*AUG_CONTRAST_RANGE)
            image = ImageEnhance.Contrast(image).enhance(contrast)

        if random.random() < AUG_BLUR_P:
            image = image.filter(
                ImageFilter.GaussianBlur(
                    radius=random.uniform(0.1, 0.8)
                )
            )

        return image, mask

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_path, mask_path, name = self.samples[index]

        with Image.open(image_path) as f:
            image = f.convert("L")
        with Image.open(mask_path) as f:
            mask = f.convert("L")

        if self.augment:
            image, mask = self._augment_pair(image, mask)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = (np.asarray(mask) > 0).astype(np.int64)

        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        if self.augment and random.random() < AUG_NOISE_P:
            noise = torch.randn_like(image_tensor) * AUG_NOISE_STD
            image_tensor = torch.clamp(image_tensor + noise, 0.0, 1.0)

        # Common input contract with the APRIL models.
        image_tensor = image_tensor.repeat(3, 1, 1)

        return {
            "image": image_tensor,
            "mask": torch.from_numpy(mask_np),
            "name": name,
        }


def build_dataloaders(
    data_root: Path,
    *,
    batch_size: int,
    num_workers: int,
    augment: bool,
    seed: int,
    max_train_samples: Optional[int],
    max_val_samples: Optional[int],
    max_test_samples: Optional[int],
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    train_set = Consolidation224Dataset(
        data_root / "train",
        train=True,
        augment=augment,
        max_samples=max_train_samples,
        seed=seed,
    )
    val_set = Consolidation224Dataset(
        data_root / "val",
        train=False,
        augment=False,
        max_samples=max_val_samples,
        seed=seed,
    )
    test_set = Consolidation224Dataset(
        data_root / "test",
        train=False,
        augment=False,
        max_samples=max_test_samples,
        seed=seed,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    common = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": True,
        "worker_init_fn": make_worker_init_fn(seed),
    }
    if num_workers > 0:
        common["persistent_workers"] = True

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        drop_last=False,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        drop_last=False,
        **common,
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        drop_last=False,
        **common,
    )

    return train_loader, val_loader, test_loader, {
        "train": len(train_set),
        "val": len(val_set),
        "test": len(test_set),
    }


# =============================================================================
# Models
# =============================================================================

class _ReplicatedGrayInput(nn.Module):
    """Restore native one-channel MONAI input from replicated grayscale."""

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, image: torch.Tensor):
        if image.ndim != 4:
            raise RuntimeError(
                f"Expected BCHW input, got {tuple(image.shape)}"
            )
        if image.shape[1] == 3:
            image = image.mean(dim=1, keepdim=True)
        elif image.shape[1] != 1:
            raise RuntimeError(
                f"Expected C=1 or C=3, got C={image.shape[1]}"
            )
        return self.network(image)


def build_monai_model(model_name: str) -> nn.Module:
    spec = MODEL_SPECS[model_name]
    params = dict(spec["arch_params"])

    if model_name == "monai_unet":
        from monai.networks.nets import UNet
        network = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=NUM_CLASSES,
            channels=tuple(params["channels"]),
            strides=tuple(params["strides"]),
            num_res_units=int(params["num_res_units"]),
        )
        return _ReplicatedGrayInput(network)

    if model_name == "monai_unetplusplus":
        try:
            from monai.networks.nets import BasicUNetPlusPlus
        except ImportError as exc:
            raise RuntimeError(
                "MONAI BasicUNetPlusPlus is unavailable. Upgrade MONAI in "
                "the current environment."
            ) from exc

        network = BasicUNetPlusPlus(
            spatial_dims=2,
            in_channels=1,
            out_channels=NUM_CLASSES,
            features=tuple(params["features"]),
            deep_supervision=False,
        )
        return _ReplicatedGrayInput(network)

    if model_name == "monai_attention_unet":
        from monai.networks.nets import AttentionUnet
        network = AttentionUnet(
            spatial_dims=2,
            in_channels=1,
            out_channels=NUM_CLASSES,
            channels=tuple(params["channels"]),
            strides=tuple(params["strides"]),
        )
        return _ReplicatedGrayInput(network)

    if model_name == "monai_vnet":
        from monai.networks.nets import VNet
        dropout = float(params["dropout_probability"])
        try:
            network = VNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=NUM_CLASSES,
                dropout_prob_down=dropout,
                dropout_prob_up=(dropout, dropout),
                dropout_dim=2,
            )
        except TypeError:
            network = VNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=NUM_CLASSES,
                dropout_prob=dropout,
                dropout_dim=2,
            )
        return _ReplicatedGrayInput(network)

    raise KeyError(model_name)


def dependency_preflight_mamba(model_name: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(f"{model_name} requires CUDA")
    try:
        import selective_scan_cuda  # noqa: F401
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            f"{model_name} requires a CUDA-enabled mamba_ssm build and "
            "selective_scan_cuda. Use the APRIL Mamba environment that was "
            "used for the Size_512 benchmark."
        ) from exc


def build_mamba_model(
    model_name: str,
    april_root: Path,
) -> nn.Module:
    dependency_preflight_mamba(model_name)

    if not (april_root / "medseg" / "model_builder.py").is_file():
        raise FileNotFoundError(
            f"Not an APRIL-MedSeg root: {april_root}"
        )
    if str(april_root) not in sys.path:
        sys.path.insert(0, str(april_root))

    from medseg.model_builder import build_model

    spec = MODEL_SPECS[model_name]
    model_cfg = {
        "model": {
            "architecture": spec["architecture"],
            "num_classes": NUM_CLASSES,
            "img_size": IMAGE_SIZE,
            "encoder": {
                "in_channels": IN_CHANNELS,
                "pretrained": False,
            },
            "arch_params": dict(spec["arch_params"]),
        }
    }
    return build_model(model_cfg)


def build_model(
    model_name: str,
    april_root: Path,
) -> nn.Module:
    family = MODEL_SPECS[model_name]["family"]
    if family == "monai":
        return build_monai_model(model_name)
    if family == "mamba":
        return build_mamba_model(model_name, april_root)
    raise KeyError(f"Unknown family for {model_name}")


def unwrap_primary_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output

    if isinstance(output, dict):
        for key in ("out", "logits", "pred", "prediction"):
            value = output.get(key)
            if torch.is_tensor(value):
                return value
        for value in output.values():
            if torch.is_tensor(value):
                return value
        raise RuntimeError(
            f"Model returned dict without tensor output: {output.keys()}"
        )

    if isinstance(output, (list, tuple)):
        if not output:
            raise RuntimeError("Model returned an empty sequence")
        # MONAI BasicUNetPlusPlus and some deep-supervision networks may return
        # a sequence. The first/final-resolution output is used for the common
        # single-output benchmark loss.
        for value in output:
            if torch.is_tensor(value):
                return value
        raise RuntimeError("Model output sequence contains no tensor")

    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def normalize_logits(logits: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    if logits.ndim != 4:
        raise RuntimeError(
            f"Expected BCHW logits, got {tuple(logits.shape)}"
        )

    # Binary one-channel models are converted to equivalent two-class logits.
    if logits.shape[1] == 1:
        logits = torch.cat([-logits, logits], dim=1)

    if logits.shape[1] != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} output channels, got {logits.shape[1]}"
        )

    if tuple(logits.shape[-2:]) != tuple(target_hw):
        logits = F.interpolate(
            logits,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
    return logits


# =============================================================================
# Loss, AMP and optimizer
# =============================================================================

def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=1)[:, 1]
    gt = (target == 1).float()

    dims = tuple(range(1, probs.ndim))
    intersection = (probs * gt).sum(dim=dims)
    denominator = probs.sum(dim=dims) + gt.sum(dim=dims)

    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    return 1.0 - dice.mean()


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ce_class_weights: torch.Tensor,
) -> torch.Tensor:
    ce = F.cross_entropy(
        logits.float(),
        target,
        weight=ce_class_weights,
    )
    dice = soft_dice_loss(logits, target)
    return CE_WEIGHT * ce + DICE_WEIGHT * dice


def resolve_amp_dtype(name: str, device: torch.device) -> torch.dtype:
    name = str(name).lower()
    if device.type != "cuda" or name == "fp32":
        return torch.float32
    if name == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print("[WARN] BF16 unsupported; falling back to FP32.")
        return torch.float32
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
        return max(
            0.5 * (1.0 + math.cos(math.pi * progress)),
            0.01,
        )

    return LambdaLR(optimizer, lr_lambda=multiplier)


def gradients_are_finite(model: nn.Module) -> bool:
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            return False
    return True


# =============================================================================
# Metrics
# =============================================================================

def safe_div(num: float, den: float, empty_value: float = 0.0) -> float:
    if den <= 0:
        return float(empty_value)
    return float(num / den)


@dataclass
class PatchMetricAccumulator:
    threshold: float = FIXED_THRESHOLD
    count: int = 0

    dice_sum: float = 0.0
    iou_sum: float = 0.0
    precision_sum: float = 0.0
    recall_sum: float = 0.0
    specificity_sum: float = 0.0
    accuracy_sum: float = 0.0

    global_tp: int = 0
    global_fp: int = 0
    global_fn: int = 0
    global_tn: int = 0

    dice_values: List[float] = field(default_factory=list)
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        probability: torch.Tensor,
        target: torch.Tensor,
        name: str,
        store_row: bool,
    ) -> None:
        pred = probability >= self.threshold
        gt = target > 0

        tp = int((pred & gt).sum().item())
        fp = int((pred & ~gt).sum().item())
        fn = int((~pred & gt).sum().item())
        tn = int((~pred & ~gt).sum().item())

        dice = safe_div(2 * tp, 2 * tp + fp + fn, empty_value=1.0)
        iou = safe_div(tp, tp + fp + fn, empty_value=1.0)

        # All final 224 GT masks are positive. Precision may have zero predicted
        # positives; in that case precision is 0. Recall is always defined.
        precision = safe_div(tp, tp + fp, empty_value=0.0)
        recall = safe_div(tp, tp + fn, empty_value=0.0)
        specificity = safe_div(tn, tn + fp, empty_value=0.0)
        accuracy = safe_div(tp + tn, tp + fp + fn + tn, empty_value=0.0)

        self.count += 1
        self.dice_sum += dice
        self.iou_sum += iou
        self.precision_sum += precision
        self.recall_sum += recall
        self.specificity_sum += specificity
        self.accuracy_sum += accuracy

        self.global_tp += tp
        self.global_fp += fp
        self.global_fn += fn
        self.global_tn += tn

        self.dice_values.append(dice)

        if store_row:
            self.rows.append({
                "name": name,
                "dice": dice,
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "accuracy": accuracy,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "gt_foreground_pixels": tp + fn,
                "pred_foreground_pixels": tp + fp,
            })

    def compute(self) -> Dict[str, Any]:
        if self.count <= 0:
            raise RuntimeError("No samples were accumulated")

        tp = self.global_tp
        fp = self.global_fp
        fn = self.global_fn
        tn = self.global_tn

        global_dice = safe_div(
            2 * tp, 2 * tp + fp + fn, empty_value=1.0
        )
        global_iou = safe_div(
            tp, tp + fp + fn, empty_value=1.0
        )
        precision_micro = safe_div(
            tp, tp + fp, empty_value=0.0
        )
        recall_micro = safe_div(
            tp, tp + fn, empty_value=0.0
        )
        specificity_micro = safe_div(
            tn, tn + fp, empty_value=0.0
        )
        accuracy_micro = safe_div(
            tp + tn, tp + fp + fn + tn, empty_value=0.0
        )

        dice_arr = np.asarray(self.dice_values, dtype=np.float64)

        return {
            "num_images": self.count,
            "mean_dice": self.dice_sum / self.count,
            "mean_iou": self.iou_sum / self.count,
            "precision_macro": self.precision_sum / self.count,
            "recall_macro": self.recall_sum / self.count,
            "specificity_macro": self.specificity_sum / self.count,
            "accuracy_macro": self.accuracy_sum / self.count,
            "global_dice": global_dice,
            "global_iou": global_iou,
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "specificity_micro": specificity_micro,
            "accuracy_micro": accuracy_micro,
            "empty_fp_rate": 0.0,
            "num_pos_images": self.count,
            "num_empty_images": 0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "dice_p25": float(np.percentile(dice_arr, 25)),
            "dice_median": float(np.percentile(dice_arr, 50)),
            "dice_p75": float(np.percentile(dice_arr, 75)),
            "threshold": self.threshold,
        }


def dice_bin_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bins = [(i / 10.0, (i + 1) / 10.0) for i in range(10)]
    counts = [0] * 10

    for row in rows:
        dice = float(row["dice"])
        index = min(int(dice * 10.0), 9)
        counts[index] += 1

    total = max(len(rows), 1)
    output: List[Dict[str, Any]] = []
    for index, (low, high) in enumerate(bins):
        output.append({
            "bin": f"{int(low*100):02d}-{int(high*100):02d}%",
            "lower_inclusive": low,
            "upper": high,
            "count": counts[index],
            "fraction": counts[index] / total,
        })
    return output


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    *,
    store_rows: bool,
    description: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()
    accumulator = PatchMetricAccumulator(threshold=FIXED_THRESHOLD)

    for batch in tqdm(loader, desc=description, leave=False):
        images = batch["image"].to(
            device, non_blocking=True
        )
        masks = batch["mask"].to(
            device, non_blocking=True
        )
        names = batch["name"]

        with autocast_context(amp_dtype, device):
            raw_output = model(images)
            logits = unwrap_primary_output(raw_output)
            logits = normalize_logits(
                logits, tuple(masks.shape[-2:])
            )

        probabilities = torch.softmax(
            logits.float(), dim=1
        )[:, 1]

        for index, name in enumerate(names):
            accumulator.add(
                probabilities[index].cpu(),
                masks[index].cpu(),
                str(name),
                store_row=store_rows,
            )

    return accumulator.compute(), accumulator.rows


# =============================================================================
# Checkpoint IO
# =============================================================================

def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: Any,
    epoch: int,
    best_score: float,
    best_epoch: int,
    model_name: str,
    model_spec: Dict[str, Any],
) -> None:
    payload = {
        "epoch": int(epoch),
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "model_name": model_name,
        "model_spec": model_spec,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
    }
    torch.save(payload, path)


def load_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: Any,
    device: torch.device,
) -> Tuple[int, float, int]:
    payload = torch.load(
        path, map_location=device, weights_only=False
    )
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    scaler.load_state_dict(payload.get("scaler_state", {}))

    return (
        int(payload["epoch"]) + 1,
        float(payload.get("best_score", -float("inf"))),
        int(payload.get("best_epoch", 0)),
    )


def load_best_checkpoint(
    path: Path,
    model: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    payload = torch.load(
        path, map_location=device, weights_only=False
    )
    model.load_state_dict(payload["model_state"])
    return payload


# =============================================================================
# Train one model
# =============================================================================

def resolve_model_runtime(
    model_name: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    spec = MODEL_SPECS[model_name]

    return {
        "batch_size": (
            int(args.batch_size)
            if args.batch_size is not None
            else int(spec["batch_size"])
        ),
        "grad_accum": (
            int(args.grad_accum)
            if args.grad_accum is not None
            else int(spec["grad_accum"])
        ),
        "lr": (
            float(args.lr)
            if args.lr is not None
            else float(spec["lr"])
        ),
        "grad_clip_norm": (
            float(args.grad_clip)
            if args.grad_clip is not None
            else float(spec["grad_clip_norm"])
        ),
        "amp_dtype": (
            str(args.amp_dtype)
            if args.amp_dtype is not None
            else str(spec["amp_dtype"])
        ),
    }


def train_one_model(
    model_name: str,
    *,
    args: argparse.Namespace,
    preset: Dict[str, Optional[int]],
    output_root: Path,
    data_root: Path,
    april_root: Path,
    device: torch.device,
) -> Dict[str, Any]:
    spec = MODEL_SPECS[model_name]
    runtime = resolve_model_runtime(model_name, args)

    model_dir = output_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    result_path = model_dir / "result.json"
    if args.skip_completed and result_path.is_file():
        with result_path.open("r", encoding="utf-8") as f:
            print(f"[SKIP] {model_name}: existing result.json")
            return json.load(f)

    train_loader, val_loader, test_loader, counts = build_dataloaders(
        data_root,
        batch_size=runtime["batch_size"],
        num_workers=args.num_workers,
        augment=not args.no_augment,
        seed=args.seed,
        max_train_samples=preset["max_train_samples"],
        max_val_samples=preset["max_val_samples"],
        max_test_samples=preset["max_test_samples"],
    )

    model = build_model(model_name, april_root).to(device)
    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    amp_dtype = resolve_amp_dtype(
        runtime["amp_dtype"], device
    )

    optimizer = AdamW(
        model.parameters(),
        lr=runtime["lr"],
        weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(
        optimizer,
        epochs=int(preset["epochs"]),
        warmup_epochs=int(preset["warmup_epochs"]),
    )
    scaler = make_grad_scaler(amp_dtype)

    ce_class_weights = torch.tensor(
        [1.0, FOREGROUND_CE_WEIGHT],
        dtype=torch.float32,
        device=device,
    )

    best_path = model_dir / "best_model.pth"
    last_path = model_dir / "last_model.pth"
    history_path = model_dir / "history.csv"

    if history_path.exists() and not args.resume:
        history_path.unlink()

    start_epoch = 1
    best_score = -float("inf")
    best_epoch = 0

    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError(
                f"--resume requested but missing {last_path}"
            )
        start_epoch, best_score, best_epoch = load_training_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        print(
            f"[RESUME] {model_name}: epoch={start_epoch}, "
            f"best={best_score:.6f}@{best_epoch}"
        )

    history_fields = [
        "epoch",
        "train_loss",
        "nonfinite_batches",
        "val_mean_dice",
        "val_mean_iou",
        "val_global_dice",
        "val_global_iou",
        "val_precision_micro",
        "val_recall_micro",
        "val_specificity_micro",
        "val_accuracy_micro",
        "lr",
        "seconds",
        "peak_memory_gib",
        "best_score",
        "best_epoch",
        "no_improvement",
    ]

    print("\n" + "=" * 96)
    print(f"MODEL: {model_name}")
    print(f"family={spec['family']} architecture={spec['architecture']}")
    print(f"samples={counts}")
    print(
        f"batch={runtime['batch_size']} "
        f"grad_accum={runtime['grad_accum']} "
        f"effective_batch="
        f"{runtime['batch_size'] * runtime['grad_accum']}"
    )
    print(
        f"lr={runtime['lr']:.3e} "
        f"amp={runtime['amp_dtype']} "
        f"grad_clip={runtime['grad_clip_norm']}"
    )
    print(
        "CHECKPOINT POLICY: highest val mean Dice over 224 patches; "
        f"threshold={FIXED_THRESHOLD}"
    )
    print("=" * 96)

    train_start = time.time()
    no_improvement = 0
    peak_memory_gib = 0.0

    for epoch in range(start_epoch, int(preset["epochs"]) + 1):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        valid_microbatches = 0
        nonfinite_batches = 0

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        progress = tqdm(
            train_loader,
            desc=f"{model_name} epoch {epoch}",
            leave=False,
        )

        for batch_index, batch in enumerate(progress, start=1):
            images = batch["image"].to(
                device, non_blocking=True
            )
            masks = batch["mask"].to(
                device, non_blocking=True
            )

            with autocast_context(amp_dtype, device):
                raw_output = model(images)
                logits = unwrap_primary_output(raw_output)
                logits = normalize_logits(
                    logits, tuple(masks.shape[-2:])
                )
                loss = segmentation_loss(
                    logits, masks, ce_class_weights
                )

            if not torch.isfinite(loss):
                nonfinite_batches += 1
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"[WARN] non-finite loss: model={model_name}, "
                    f"epoch={epoch}, batch={batch_index}"
                )
                if (
                    nonfinite_batches
                    > MAX_NONFINITE_BATCHES_PER_EPOCH
                ):
                    raise FloatingPointError(
                        f"{model_name}: more than "
                        f"{MAX_NONFINITE_BATCHES_PER_EPOCH} non-finite "
                        f"batches in epoch {epoch}"
                    )
                continue

            scaled_loss = loss / runtime["grad_accum"]
            scaler.scale(scaled_loss).backward()

            total_loss += float(loss.detach().item())
            valid_microbatches += 1

            is_step = (
                batch_index % runtime["grad_accum"] == 0
                or batch_index == len(train_loader)
            )
            if not is_step:
                continue

            if scaler.is_enabled():
                scaler.unscale_(optimizer)

            if not gradients_are_finite(model):
                nonfinite_batches += 1
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"[WARN] non-finite gradients: model={model_name}, "
                    f"epoch={epoch}, batch={batch_index}"
                )
                if (
                    nonfinite_batches
                    > MAX_NONFINITE_BATCHES_PER_EPOCH
                ):
                    raise FloatingPointError(
                        f"{model_name}: more than "
                        f"{MAX_NONFINITE_BATCHES_PER_EPOCH} non-finite "
                        f"gradient groups in epoch {epoch}"
                    )
                continue

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=runtime["grad_clip_norm"],
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(
                loss=f"{float(loss.detach().item()):.4f}"
            )

        if valid_microbatches == 0:
            raise RuntimeError(
                f"No valid training batches in epoch {epoch}"
            )

        train_loss = total_loss / valid_microbatches

        val_metrics, _ = evaluate(
            model,
            val_loader,
            device,
            amp_dtype,
            store_rows=False,
            description=f"{model_name} val",
        )

        score = float(val_metrics["mean_dice"])
        improved = score > best_score + EARLY_STOPPING_MIN_DELTA

        if improved:
            best_score = score
            best_epoch = epoch
            no_improvement = 0
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_score=best_score,
                best_epoch=best_epoch,
                model_name=model_name,
                model_spec=spec,
            )
        else:
            no_improvement += 1

        scheduler.step()

        # Save the resumable checkpoint after scheduler.step(), so resuming at
        # epoch+1 restores exactly the LR state that a continuous run would use.
        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_score=best_score,
            best_epoch=best_epoch,
            model_name=model_name,
            model_spec=spec,
        )

        epoch_peak = (
            torch.cuda.max_memory_allocated(device) / (1024**3)
            if device.type == "cuda"
            else 0.0
        )
        peak_memory_gib = max(peak_memory_gib, epoch_peak)

        elapsed = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "nonfinite_batches": nonfinite_batches,
            "val_mean_dice": val_metrics["mean_dice"],
            "val_mean_iou": val_metrics["mean_iou"],
            "val_global_dice": val_metrics["global_dice"],
            "val_global_iou": val_metrics["global_iou"],
            "val_precision_micro": val_metrics["precision_micro"],
            "val_recall_micro": val_metrics["recall_micro"],
            "val_specificity_micro": val_metrics["specificity_micro"],
            "val_accuracy_micro": val_metrics["accuracy_micro"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
            "peak_memory_gib": epoch_peak,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "no_improvement": no_improvement,
        }
        append_csv(history_path, row, history_fields)

        print(
            f"Epoch {epoch:03d}/{int(preset['epochs'])} | "
            f"loss={train_loss:.5f} | "
            f"val_mean_dice={score:.5f} | "
            f"val_mean_iou={float(val_metrics['mean_iou']):.5f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"{elapsed:.1f}s | peak={epoch_peak:.2f} GiB | "
            f"best={best_score:.5f}@{best_epoch} | "
            f"no_improve={no_improvement}/{int(preset['patience'])}"
        )

        if (
            int(preset["patience"]) > 0
            and no_improvement >= int(preset["patience"])
        ):
            print(
                f"Early stopping at epoch {epoch}: "
                f"no validation mean-Dice improvement for "
                f"{no_improvement} epochs."
            )
            break

    if not best_path.is_file():
        raise RuntimeError(
            f"No best checkpoint saved for {model_name}"
        )

    best_payload = load_best_checkpoint(
        best_path, model, device
    )

    final_val_metrics, val_rows = evaluate(
        model,
        val_loader,
        device,
        amp_dtype,
        store_rows=True,
        description=f"{model_name} final val",
    )
    test_metrics, test_rows = evaluate(
        model,
        test_loader,
        device,
        amp_dtype,
        store_rows=True,
        description=f"{model_name} test",
    )

    write_csv(model_dir / "val_cases.csv", val_rows)
    write_csv(model_dir / "test_cases.csv", test_rows)
    write_csv(
        model_dir / "test_dice_bins.csv",
        dice_bin_rows(test_rows),
    )

    result: Dict[str, Any] = {
        "model": model_name,
        "family": spec["family"],
        "architecture": spec["architecture"],
        "image_size": IMAGE_SIZE,
        "evaluation_unit": "individual_224_patch",
        "checkpoint_selection_metric": "val_mean_dice",
        "threshold": FIXED_THRESHOLD,
        "best_epoch": int(best_payload["best_epoch"]),
        "best_val_mean_dice": float(best_payload["best_score"]),
        "final_val": final_val_metrics,
        "test": test_metrics,
        "dataset_counts": counts,
        "parameters_total": params_total,
        "parameters_trainable": params_trainable,
        "batch_size": runtime["batch_size"],
        "grad_accum": runtime["grad_accum"],
        "effective_batch_size": (
            runtime["batch_size"] * runtime["grad_accum"]
        ),
        "lr": runtime["lr"],
        "weight_decay": args.weight_decay,
        "grad_clip_norm": runtime["grad_clip_norm"],
        "amp_dtype": runtime["amp_dtype"],
        "epochs_max": int(preset["epochs"]),
        "warmup_epochs": int(preset["warmup_epochs"]),
        "patience": int(preset["patience"]),
        "full_train_traversal": (
            preset["max_train_samples"] is None
        ),
        "replacement_sampling": False,
        "positive_negative_balancing": False,
        "peak_memory_gib": peak_memory_gib,
        "total_seconds": time.time() - train_start,
        "output_dir": str(model_dir),
    }
    save_json(result_path, result)

    print(
        f"FINAL {model_name}: "
        f"test_mean_dice={test_metrics['mean_dice']:.6f}, "
        f"test_mean_iou={test_metrics['mean_iou']:.6f}, "
        f"global_dice={test_metrics['global_dice']:.6f}, "
        f"precision={test_metrics['precision_micro']:.6f}, "
        f"recall={test_metrics['recall_micro']:.6f}, "
        f"specificity={test_metrics['specificity_micro']:.6f}, "
        f"accuracy={test_metrics['accuracy_micro']:.6f}"
    )

    return result


def summary_rows(
    results: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for result in results:
        test = result["test"]
        rows.append({
            "model": result["model"],
            "family": result["family"],
            "best_epoch": result["best_epoch"],
            "best_val_mean_dice": result["best_val_mean_dice"],
            "test_mean_dice": test["mean_dice"],
            "test_mean_iou": test["mean_iou"],
            "test_global_dice": test["global_dice"],
            "test_global_iou": test["global_iou"],
            "test_precision_micro": test["precision_micro"],
            "test_recall_micro": test["recall_micro"],
            "test_specificity_micro": test["specificity_micro"],
            "test_accuracy_micro": test["accuracy_micro"],
            "test_precision_macro": test["precision_macro"],
            "test_recall_macro": test["recall_macro"],
            "test_specificity_macro": test["specificity_macro"],
            "test_accuracy_macro": test["accuracy_macro"],
            "test_empty_fp_rate": test["empty_fp_rate"],
            "test_dice_p25": test["dice_p25"],
            "test_dice_median": test["dice_median"],
            "test_dice_p75": test["dice_p75"],
            "tp": test["tp"],
            "fp": test["fp"],
            "fn": test["fn"],
            "tn": test["tn"],
            "parameters_m": result["parameters_total"] / 1e6,
            "peak_memory_gib": result["peak_memory_gib"],
            "hours": result["total_seconds"] / 3600.0,
            "output_dir": result["output_dir"],
        })

    rows.sort(
        key=lambda row: float(row["test_mean_dice"]),
        reverse=True,
    )
    return rows


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    models = resolve_models(args)
    data_root = args.data_root.expanduser().resolve()
    april_root = args.april_root.expanduser().resolve()

    ensure_dataset_layout(data_root)
    split_audit = audit_patient_split(data_root)

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(
            f"Invalid --gpu {args.gpu}; "
            f"visible GPUs={torch.cuda.device_count()}"
        )

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    seed_everything(args.seed, args.deterministic)

    preset = dict(PRESETS[args.run_mode])
    for key in ("epochs", "warmup_epochs", "patience"):
        override = getattr(args, key)
        if override is not None:
            preset[key] = int(override)

    for key in (
        "max_train_samples",
        "max_val_samples",
        "max_test_samples",
    ):
        override = getattr(args, key)
        if override is not None:
            preset[key] = int(override)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (
            april_root
            / "output"
            / f"consolidation_size224_patch_benchmark_{timestamp}"
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)

    if args.resume and args.output_dir is None:
        raise ValueError(
            "--resume requires an explicit --output-dir"
        )

    save_json(
        output_root / "benchmark_protocol.json",
        {
            "image_size": IMAGE_SIZE,
            "evaluation_unit": "individual_224_patch",
            "data_root": str(data_root),
            "models": models,
            "run_mode": args.run_mode,
            "preset": preset,
            "seed": args.seed,
            "fixed_threshold": FIXED_THRESHOLD,
            "checkpoint_selection_metric": "val_mean_dice",
            "formal_epoch_policy": (
                "full traversal once per epoch, shuffle=True, "
                "drop_last=False, no replacement"
            ),
            "positive_negative_balancing": False,
            "split_audit": split_audit,
        },
    )

    print("=" * 96)
    print("FINAL SIZE_224 PATCH BENCHMARK")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"APRIL root: {april_root}")
    print(f"data root: {data_root}")
    print(f"output root: {output_root}")
    print(f"models: {models}")
    print(f"run mode: {args.run_mode}, preset={preset}")
    print(f"split audit: {split_audit}")
    print(
        "BEST MODEL: maximum validation mean Dice over retained "
        "224x224 patches."
    )
    print(
        "TEST: metrics are computed directly over retained "
        "224x224 test patches; no 512 reconstruction."
    )
    print("=" * 96)

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    for model_name in models:
        try:
            result = train_one_model(
                model_name,
                args=args,
                preset=preset,
                output_root=output_root,
                data_root=data_root,
                april_root=april_root,
                device=device,
            )
            results.append(result)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            failure = {
                "model": model_name,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            save_json(
                output_root / model_name / "failure.json",
                failure,
            )
            print(
                f"\n[FAILED] {model_name}: {failure['error']}"
            )
            traceback.print_exc()
            if args.fail_fast:
                raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if results:
        write_csv(
            output_root / "benchmark_summary.csv",
            summary_rows(results),
        )

    save_json(
        output_root / "run_report.json",
        {
            "results": results,
            "failures": failures,
        },
    )

    print("\n" + "=" * 96)
    print(
        f"Completed={len(results)}, failed={len(failures)}, "
        f"output={output_root}"
    )
    if results:
        print(
            f"Summary: {output_root / 'benchmark_summary.csv'}"
        )
    for failure in failures:
        print(
            f"FAILED {failure['model']}: {failure['error']}"
        )


if __name__ == "__main__":
    main()
