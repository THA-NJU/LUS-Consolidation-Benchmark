#!/usr/bin/env python3
"""AutoDL two-GPU benchmark for easy APRIL-MedSeg architectures.

This file is intended to be copied into a freshly extracted APRIL-MedSeg
repository. It provides its own dataset, augmentation, training loop, fixed-threshold
validation/test evaluation, timestamped output directories, and optional
2-GPU task scheduling. It does not rely on the custom scripts created in prior
local runs.

Target dataset layout for each size:
  DATA_ROOTS[512]/train/images/*.png, train/masks/*.png, val/..., test/...
  DATA_ROOTS[224]/train/images/*.png, train/masks/*.png, val/..., test/...

Default selected models are low-dependency compared with Mamba/RWKV/SAM:
  nnunet_2d, aau_net, swinunet, nnformer_2d, sepnet, nulite, ukan,
  xlstm_unet_bot.

Edit only the USER SETTINGS section before running.
"""

from __future__ import annotations

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
import multiprocessing as mp
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm


# =============================================================================
# USER SETTINGS: edit here, no command-line arguments are required
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

# 你在 AutoDL 上解压/挂载数据后，只需要改这里。
# 每个目录必须包含 train/val/test，并且每个 split 下有 images/ 和 masks/。
DATA_ROOTS = {
    512: Path("./data/Size_512"),
    224: Path("./data/Size_224_filtered"),
}
RUN_SIZES = [512, 224]

# 每次运行默认创建新的时间戳目录，避免覆盖。需要断点续训同一次实验时，
# 手动固定 RUN_ID，例如 RUN_ID = "20260720_autodl_easy_v1"。
RUN_ID = os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_BASE = PROJECT_ROOT / "output" / f"autodl_easy_models_benchmark_{RUN_ID}"
# The following globals are set per task by set_task_context().
IMAGE_SIZE = RUN_SIZES[0]
DATA_ROOT = DATA_ROOTS[IMAGE_SIZE]
OUTPUT_ROOT = OUTPUT_BASE / f"size{IMAGE_SIZE}"

# AutoDL 两张 A800 通常可见为 cuda:0 和 cuda:1。
# 设置 GPU_IDS=[0] 则单卡顺序跑；设置 [0, 1] 则一个脚本内双卡并行调度任务。
GPU_IDS = [0]
RUN_MODE = "formal"  # 先 smoke；成功后改为 formal。

# 低环境复杂度优先选择：CNN / Transformer / KAN / xLSTM。
# 不放 U-RWKV/RWKV-UNet/SAM 系列，因为它们需要额外 CUDA kernel、prompt、预训练权重或特殊输入流程。
MODELS_TO_RUN = [
    # "nnunet_2d",       # nnU-Net 2D plain conv UNet
    # "aau_net",         # Adaptive Attention U-Net, ultrasound-friendly CNN
    "swinunet",        # Swin-UNet
    "nnformer_2d",     # 2D nnFormer-style Transformer
    # "sepnet",          # PVTv2-based polyp segmentation network; may try downloading PVT weights then fallback
    # "nulite",          # lightweight FastViT-style model; pretrained disabled by this script
    # "ukan",            # U-KAN
    "xlstm_unet_bot",  # xLSTM-UNet bottleneck variant, pure torch/einops path
]

NUM_CLASSES = 2  # 0 background, 1 consolidation
IN_CHANNELS = 3  # grayscale is replicated to three channels
PRETRAINED = False  # fair architecture comparison: train every model from scratch where supported
SEED = 42
DETERMINISTIC = False
USE_AMP = True
NUM_WORKERS = 4
PIN_MEMORY = True

# Fair training objective for all architectures.
CE_WEIGHT = 0.40
DICE_WEIGHT = 0.60
FOREGROUND_CE_WEIGHT = 5.0
GRAD_CLIP_NORM = 0.5
DEFAULT_AMP_DTYPE = "bf16"  # bf16 is appropriate for A800; falls back to fp32 if unsupported.
MAX_NONFINITE_BATCHES_PER_EPOCH = 20
ABORT_ON_NONFINITE = False

# Image-level sampling. When both positive and negative images exist, each
# epoch samples approximately this fraction of positive images.
BALANCE_POSITIVE_IMAGES = False
TARGET_POSITIVE_IMAGE_FRACTION = 0.50

# Input scaling: fixed [0,1], never per-image min-max normalization.
NORMALIZE_INPUT = False
INPUT_MEAN = 0.100638
INPUT_STD = 0.145868

# Lung-ultrasound-safe augmentation. No vertical flip / 90-degree rotation.
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

# Validation/checkpoint policy.
FIXED_VALIDATION_THRESHOLD = 0.50
CHECKPOINT_SELECTION_METRIC = "dice_positive"
EARLY_STOPPING_MIN_DELTA = 1e-4

# Resume/overwrite behavior within the same RUN_ID.
RESUME_FROM_LAST = True
SKIP_COMPLETED_MODELS = True
FAIL_FAST = False

# Formal mode keeps a fixed compute budget per epoch for fairness and time control.
# Set samples_per_epoch=None if you want a true full pass over every training image per epoch.
RUN_PRESETS = {
    "smoke": {
        "epochs": 1,
        "samples_per_epoch": None,
        "patience": 1,
        "val_interval": 1,
        "max_val_samples": 2,
        "max_test_samples": 5,
        "warmup_epochs": 0,
    },
    "formal": {
        "epochs": 600,
        "samples_per_epoch": None,
        "patience": 15,
        "val_interval": 1,
        "max_val_samples": None,
        "max_test_samples": None,
        "warmup_epochs": 10,
    },
}

# Per-size defaults: A800 usually has enough memory. If a model OOMs, lower the corresponding batch.
SIZE_DEFAULTS = {
    512: {"batch_size": 4, "grad_accum": 1, "lr": 1e-4},
    224: {"batch_size": 16, "grad_accum": 1, "lr": 1e-4},
}

MODEL_SPECS = {
    "nnunet_2d": {
        "architecture": "nnunet_2d",
        "arch_params": {"base_features": 32, "num_stages": 6, "max_features": 512},
    },
    "aau_net": {
        "architecture": "aau_net",
        "arch_params": {},
        "override_by_size": {512: {"batch_size": 2, "grad_accum": 2}},
    },
    "swinunet": {
        "architecture": "swinunet",
        "arch_params": {},
        "override_by_size": {512: {"batch_size": 2, "grad_accum": 2}},
    },
    "nnformer_2d": {
        "architecture": "nnformer_2d",
        "arch_params": {"embed_dim": 48, "drop_path_rate": 0.1},
        "override_by_size": {512: {"batch_size": 2, "grad_accum": 2}},
    },
    "sepnet": {
        "architecture": "sepnet",
        "arch_params": {"mid_channels": 128},
        "override_by_size": {512: {"batch_size": 2, "grad_accum": 2}},
    },
    "nulite": {
        "architecture": "nulite",
        "arch_params": {"backbone": "fastvit_t8", "drop_rate": 0.0, "embed_dim": 32},
        "override_by_size": {512: {"batch_size": 4, "grad_accum": 1}},
    },
    "ukan": {
        "architecture": "ukan",
        "arch_params": {"embed_dims": [128, 160, 256], "drop_rate": 0.0, "drop_path_rate": 0.1, "deep_supervision": False},
        "override_by_size": {512: {"batch_size": 2, "grad_accum": 2}},
    },
    "xlstm_unet_bot": {
        "architecture": "xlstm_unet_bot",
        "arch_params": {"features": [32, 64, 128, 256, 320], "deep_supervision": False},
        "override_by_size": {512: {"batch_size": 2, "grad_accum": 2}},
    },
}

# =============================================================================
# Utilities
# =============================================================================
def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def patient_id_from_name(filename: str) -> str:
    return Path(filename).stem.split("_")[0]


def ensure_dataset_layout(root: Path) -> None:
    missing = []
    for split in ("train", "val", "test"):
        for sub in ("images", "masks"):
            p = root / split / sub
            if not p.is_dir():
                missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            "Dataset layout is incomplete:\n  " + "\n  ".join(missing)
        )


def unwrap_primary_output(output: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
    if isinstance(output, (list, tuple)):
        if not output:
            raise RuntimeError("Model returned an empty output list")
        return output[0]
    return output


def safe_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_csv(path: Path, row: Dict[str, object], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def dependency_preflight(model_name: str) -> None:
    """Fail early only for intentionally excluded complex families.

    The default selected models are designed to run without mamba-ssm, RWKV
    custom kernels, or SAM prompt-specific packages. We still guard against
    accidentally adding those models to MODELS_TO_RUN.
    """
    complex_models = {
        "u_rwkv", "rwkv_unet", "swin_umamba", "mamba_unet", "vm_unet_v2",
        "nnmamba_2d", "dcm_net", "sam2", "medsam", "samus", "sam_b",
        "sam_l", "mobile_sam", "sam_med2d", "medical_sam_adapter", "samed",
    }
    if model_name in complex_models:
        raise RuntimeError(
            f"{model_name} is intentionally excluded from this easy AutoDL script. "
            "Run it in a separate environment/script because it needs Mamba/RWKV/SAM-specific setup."
        )



# =============================================================================
# Dataset and augmentation
# =============================================================================
class ConsolidationDataset(Dataset):
    """Explicit split dataset; every file in the chosen split is used once."""

    def __init__(
        self,
        split_root: Path,
        train: bool,
        image_size: int = 512,
        max_samples: Optional[int] = None,
    ) -> None:
        self.split_root = Path(split_root)
        self.image_dir = self.split_root / "images"
        self.mask_dir = self.split_root / "masks"
        self.train = train
        self.image_size = int(image_size)

        image_map = {p.stem: p for p in self.image_dir.glob("*.png")}
        mask_map = {p.stem: p for p in self.mask_dir.glob("*.png")}
        common = sorted(image_map.keys() & mask_map.keys())
        if not common:
            raise RuntimeError(f"No paired PNG samples found under {self.split_root}")
        missing_mask = sorted(image_map.keys() - mask_map.keys())
        missing_image = sorted(mask_map.keys() - image_map.keys())
        if missing_mask or missing_image:
            raise RuntimeError(
                f"Unpaired files in {self.split_root}: "
                f"images_without_mask={len(missing_mask)}, masks_without_image={len(missing_image)}"
            )

        self.samples = [(image_map[k], mask_map[k]) for k in common]
        if max_samples is not None:
            self.samples = self.samples[: int(max_samples)]
        self._positive_flags: Optional[List[bool]] = None

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def case_names(self) -> List[str]:
        return [img.name for img, _ in self.samples]

    def positive_flags(self, cache_path: Optional[Path] = None) -> List[bool]:
        if self._positive_flags is not None:
            return self._positive_flags

        cached: Dict[str, bool] = {}
        if cache_path is not None and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    cached = {str(k): bool(v) for k, v in raw.items()}
            except Exception:
                cached = {}

        changed = False
        flags: List[bool] = []
        for _, mask_path in tqdm(self.samples, desc="scan train masks", unit="mask"):
            key = mask_path.name
            if key in cached:
                flags.append(cached[key])
                continue
            with Image.open(mask_path) as mask:
                arr = np.asarray(mask)
                if arr.ndim == 3:
                    arr = arr[..., 0]
                flag = bool(np.any(arr > 0))
            cached[key] = flag
            flags.append(flag)
            changed = True

        if cache_path is not None and (changed or not cache_path.exists()):
            save_json(cache_path, cached)
        self._positive_flags = flags
        return flags

    def _paired_spatial_augmentation(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < AUG_HORIZONTAL_FLIP_P:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)

        if random.random() < AUG_AFFINE_P:
            angle = random.uniform(-AUG_MAX_ROTATE_DEG, AUG_MAX_ROTATE_DEG)
            max_shift = int(round(self.image_size * AUG_MAX_TRANSLATE_FRACTION))
            translate = (
                random.randint(-max_shift, max_shift),
                random.randint(-max_shift, max_shift),
            )
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
        return image, mask

    @staticmethod
    def _intensity_augmentation(image: Image.Image) -> Image.Image:
        if random.random() < AUG_INTENSITY_P:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(*AUG_BRIGHTNESS_RANGE))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(*AUG_CONTRAST_RANGE))
            gamma = random.uniform(*AUG_GAMMA_RANGE)
            arr = np.asarray(image, dtype=np.float32) / 255.0
            arr = np.power(np.clip(arr, 0.0, 1.0), gamma)
            image = Image.fromarray(np.uint8(np.clip(arr * 255.0, 0, 255)), mode="L")
        if random.random() < AUG_BLUR_P:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.0)))
        return image

    def __getitem__(self, index: int) -> Dict[str, object]:
        image_path, mask_path = self.samples[index]
        with Image.open(image_path) as im:
            image = im.convert("L")
        with Image.open(mask_path) as mm:
            mask = mm.convert("L")

        target_size = (self.image_size, self.image_size)
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.BILINEAR)
        if mask.size != target_size:
            mask = mask.resize(target_size, Image.Resampling.NEAREST)

        if self.train:
            image, mask = self._paired_spatial_augmentation(image, mask)
            image = self._intensity_augmentation(image)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        if self.train and random.random() < AUG_NOISE_P:
            image_np = image_np + np.random.normal(0.0, AUG_NOISE_STD, image_np.shape).astype(np.float32)
        image_np = np.clip(image_np, 0.0, 1.0)
        if NORMALIZE_INPUT:
            image_np = (image_np - INPUT_MEAN) / max(INPUT_STD, 1e-8)
        image_np = np.repeat(image_np[None, ...], IN_CHANNELS, axis=0)

        mask_np = np.asarray(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        mask_np = (mask_np > 0).astype(np.int64)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image_np)).float(),
            "label": torch.from_numpy(np.ascontiguousarray(mask_np)).long(),
            "case_name": image_path.name,
            "patient_id": patient_id_from_name(image_path.name),
        }


def check_patient_leakage(datasets: Dict[str, ConsolidationDataset]) -> None:
    patient_sets = {
        split: {patient_id_from_name(name) for name in ds.case_names}
        for split, ds in datasets.items()
    }
    overlaps = {
        "train-val": patient_sets["train"] & patient_sets["val"],
        "train-test": patient_sets["train"] & patient_sets["test"],
        "val-test": patient_sets["val"] & patient_sets["test"],
    }
    bad = {k: v for k, v in overlaps.items() if v}
    if bad:
        details = ", ".join(f"{k}={len(v)}" for k, v in bad.items())
        raise RuntimeError(f"Patient leakage detected: {details}")


def build_train_sampler(dataset: ConsolidationDataset, output_root: Path, samples_per_epoch: Optional[int]):
    if not BALANCE_POSITIVE_IMAGES:
        return None
    flags = np.asarray(dataset.positive_flags(output_root / "train_positive_flags.json"), dtype=bool)
    n_pos = int(flags.sum())
    n_neg = int((~flags).sum())
    if n_pos == 0 or n_neg == 0:
        print(f"Balanced sampler disabled: positive={n_pos}, negative={n_neg}")
        return None

    p_pos = float(TARGET_POSITIVE_IMAGE_FRACTION)
    weights = np.where(flags, p_pos / n_pos, (1.0 - p_pos) / n_neg).astype(np.float64)
    n_samples = len(dataset) if samples_per_epoch is None else int(samples_per_epoch)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    print(
        f"Weighted sampler: train={len(dataset)}, positive={n_pos}, negative={n_neg}, "
        f"target_positive_fraction={p_pos:.2f}, samples_per_epoch={n_samples}"
    )
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=n_samples,
        replacement=True,
        generator=generator,
    )


# =============================================================================
# Loss and metrics
# =============================================================================
class ForegroundDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)[:, 1]
        target_f = target.float()
        intersection = (probs * target_f).sum(dim=(1, 2))
        denominator = probs.sum(dim=(1, 2)) + target_f.sum(dim=(1, 2))
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class ConsolidationLoss(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        class_weight = torch.tensor([1.0, FOREGROUND_CE_WEIGHT], dtype=torch.float32, device=device)
        self.ce = nn.CrossEntropyLoss(weight=class_weight)
        self.dice = ForegroundDiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return CE_WEIGHT * self.ce(logits, target) + DICE_WEIGHT * self.dice(logits, target)


@dataclass
class MetricAccumulator:
    threshold: float
    n: int = 0
    dice_all_sum: float = 0.0
    iou_all_sum: float = 0.0
    positive_n: int = 0
    dice_positive_sum: float = 0.0
    iou_positive_sum: float = 0.0
    nonempty_union_n: int = 0
    dice_nonempty_union_sum: float = 0.0
    negative_n: int = 0
    negative_correct_n: int = 0
    global_tp: int = 0
    global_fp: int = 0
    global_fn: int = 0
    case_rows: List[Dict[str, object]] = field(default_factory=list)

    def update(
        self,
        probs: torch.Tensor,
        target: torch.Tensor,
        case_names: Optional[Sequence[str]] = None,
        patient_ids: Optional[Sequence[str]] = None,
        store_cases: bool = False,
    ) -> None:
        pred = probs >= self.threshold
        truth = target.bool()
        dims = (1, 2)
        tp = (pred & truth).sum(dim=dims).to(torch.int64)
        pred_sum = pred.sum(dim=dims).to(torch.int64)
        target_sum = truth.sum(dim=dims).to(torch.int64)
        fp = pred_sum - tp
        fn = target_sum - tp
        denom = pred_sum + target_sum
        both_empty = denom == 0
        dice = torch.where(
            both_empty,
            torch.ones_like(denom, dtype=torch.float64),
            (2.0 * tp).to(torch.float64) / denom.clamp_min(1).to(torch.float64),
        )
        union = pred_sum + target_sum - tp
        iou = torch.where(
            union == 0,
            torch.ones_like(union, dtype=torch.float64),
            tp.to(torch.float64) / union.clamp_min(1).to(torch.float64),
        )
        target_positive = target_sum > 0
        pred_positive = pred_sum > 0
        nonempty_union = target_positive | pred_positive
        negative = ~target_positive

        self.n += int(target.shape[0])
        self.dice_all_sum += float(dice.sum().item())
        self.iou_all_sum += float(iou.sum().item())
        self.positive_n += int(target_positive.sum().item())
        self.dice_positive_sum += float(dice[target_positive].sum().item())
        self.iou_positive_sum += float(iou[target_positive].sum().item())
        self.nonempty_union_n += int(nonempty_union.sum().item())
        self.dice_nonempty_union_sum += float(dice[nonempty_union].sum().item())
        self.negative_n += int(negative.sum().item())
        self.negative_correct_n += int((negative & ~pred_positive).sum().item())
        self.global_tp += int(tp.sum().item())
        self.global_fp += int(fp.sum().item())
        self.global_fn += int(fn.sum().item())

        if store_cases:
            names = list(case_names or [f"case_{i}" for i in range(target.shape[0])])
            patients = list(patient_ids or [patient_id_from_name(n) for n in names])
            for i in range(target.shape[0]):
                self.case_rows.append(
                    {
                        "case_name": names[i],
                        "patient_id": patients[i],
                        "threshold": self.threshold,
                        "target_positive": int(target_positive[i].item()),
                        "pred_positive": int(pred_positive[i].item()),
                        "target_pixels": int(target_sum[i].item()),
                        "pred_pixels": int(pred_sum[i].item()),
                        "tp": int(tp[i].item()),
                        "fp": int(fp[i].item()),
                        "fn": int(fn[i].item()),
                        "dice": float(dice[i].item()),
                        "iou": float(iou[i].item()),
                    }
                )

    def summary(self) -> Dict[str, float | int]:
        precision = self.global_tp / max(self.global_tp + self.global_fp, 1)
        recall = self.global_tp / max(self.global_tp + self.global_fn, 1)
        global_dice = (2 * self.global_tp) / max(2 * self.global_tp + self.global_fp + self.global_fn, 1)
        global_iou = self.global_tp / max(self.global_tp + self.global_fp + self.global_fn, 1)
        return {
            "threshold": float(self.threshold),
            "count": self.n,
            "positive_count": self.positive_n,
            "negative_count": self.negative_n,
            "dice_all": self.dice_all_sum / max(self.n, 1),
            "dice_positive": self.dice_positive_sum / max(self.positive_n, 1),
            "dice_nonempty_union": self.dice_nonempty_union_sum / max(self.nonempty_union_n, 1),
            "iou_all": self.iou_all_sum / max(self.n, 1),
            "iou_positive": self.iou_positive_sum / max(self.positive_n, 1),
            "negative_empty_accuracy": self.negative_correct_n / max(self.negative_n, 1),
            "negative_false_positive_rate": 1.0 - self.negative_correct_n / max(self.negative_n, 1),
            "global_dice": global_dice,
            "global_iou": global_iou,
            "precision": precision,
            "recall": recall,
            "global_tp": self.global_tp,
            "global_fp": self.global_fp,
            "global_fn": self.global_fn,
        }


# =============================================================================
# Model, optimization, and training
# =============================================================================
def get_model_spec(model_name: str) -> Dict[str, object]:
    if model_name not in MODEL_SPECS:
        raise KeyError(f"Unknown model spec: {model_name}")
    raw = dict(MODEL_SPECS[model_name])
    merged = dict(SIZE_DEFAULTS.get(int(IMAGE_SIZE), {}))
    merged.update({k: v for k, v in raw.items() if k != "override_by_size"})
    override = dict(raw.get("override_by_size", {}).get(int(IMAGE_SIZE), {}))
    merged.update(override)
    merged.setdefault("amp_dtype", DEFAULT_AMP_DTYPE)
    merged.setdefault("grad_clip_norm", GRAD_CLIP_NORM)
    merged.setdefault("warmup_epochs", RUN_PRESETS[RUN_MODE]["warmup_epochs"])
    return merged


def build_april_model(model_name: str) -> Tuple[nn.Module, Dict[str, object]]:
    dependency_preflight(model_name)
    spec = get_model_spec(model_name)

    from medseg.model_builder import build_model

    model_cfg = {
        "model": {
            "architecture": spec["architecture"],
            "num_classes": NUM_CLASSES,
            "img_size": IMAGE_SIZE,
            "encoder": {
                "in_channels": IN_CHANNELS,
                "pretrained": PRETRAINED,
            },
            "arch_params": dict(spec.get("arch_params", {})),
        }
    }
    model = build_model(model_cfg)
    return model, model_cfg


def make_scheduler(optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int) -> LambdaLR:
    min_ratio = 1e-6 / max(float(optimizer.param_groups[0]["lr"]), 1e-12)

    def multiplier(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return max((epoch + 1) / warmup_epochs, 1e-3)
        denom = max(epochs - warmup_epochs - 1, 1)
        progress = min(max((epoch - warmup_epochs) / denom, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, multiplier)


def resolve_amp_dtype(name: str, device: torch.device) -> torch.dtype:
    """Resolve per-model autocast dtype with a safe fallback."""
    name = str(name).lower()
    if device.type != "cuda" or name == "fp32":
        return torch.float32
    if name == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print("[WARN] BF16 is not supported on this GPU; falling back to FP32.")
        return torch.float32
    return torch.float16


def make_grad_scaler(enabled: bool, amp_dtype: torch.dtype):
    # Gradient scaling is useful for FP16 but unnecessary for BF16/FP32.
    use_scaler = bool(enabled and amp_dtype == torch.float16)
    try:
        return torch.amp.GradScaler("cuda", enabled=use_scaler)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=use_scaler)


def autocast_context(enabled: bool, amp_dtype: torch.dtype):
    if not enabled or amp_dtype == torch.float32:
        return torch.autocast(device_type="cuda", enabled=False)
    return torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)


def _reduce_scaler_after_nonfinite(scaler) -> None:
    """Reduce FP16 scale after a non-finite batch without touching weights."""
    if not scaler.is_enabled():
        return
    try:
        old_scale = float(scaler.get_scale())
        scaler.update(new_scale=max(old_scale * 0.5, 1.0))
    except TypeError:
        scaler.update()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    epoch: int,
    grad_accum: int,
    amp_dtype: torch.dtype,
    grad_clip_norm: float,
) -> Tuple[float, int, int]:
    """Train one epoch and explicitly reject non-finite logits/loss/gradients."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    n_batches = 0
    nonfinite_batches = 0
    skipped_steps = 0
    accumulated_finite = 0
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False, unit="batch")

    for step, batch in enumerate(progress):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast_context(USE_AMP and device.type == "cuda", amp_dtype):
            logits = unwrap_primary_output(model(images))

        # Compute CE + Dice in FP32 even when the backbone uses BF16/FP16.
        logits_f32 = logits.float()
        if not torch.isfinite(logits_f32).all():
            nonfinite_batches += 1
            skipped_steps += 1
            optimizer.zero_grad(set_to_none=True)
            accumulated_finite = 0
            _reduce_scaler_after_nonfinite(scaler)
            print(f"[WARN] epoch={epoch} step={step + 1}: non-finite logits; batch skipped")
            if ABORT_ON_NONFINITE or nonfinite_batches > MAX_NONFINITE_BATCHES_PER_EPOCH:
                raise FloatingPointError(
                    f"Non-finite logits detected at epoch={epoch}, step={step + 1}. "
                    "The current epoch was aborted before saving a checkpoint."
                )
            continue

        loss_full = criterion(logits_f32, labels)
        if not torch.isfinite(loss_full):
            nonfinite_batches += 1
            skipped_steps += 1
            optimizer.zero_grad(set_to_none=True)
            accumulated_finite = 0
            _reduce_scaler_after_nonfinite(scaler)
            print(f"[WARN] epoch={epoch} step={step + 1}: non-finite loss; batch skipped")
            if ABORT_ON_NONFINITE or nonfinite_batches > MAX_NONFINITE_BATCHES_PER_EPOCH:
                raise FloatingPointError(
                    f"Non-finite loss detected at epoch={epoch}, step={step + 1}. "
                    "The current epoch was aborted before saving a checkpoint."
                )
            continue

        loss = loss_full / grad_accum
        scaler.scale(loss).backward()
        accumulated_finite += 1

        should_step = (accumulated_finite >= grad_accum) or (step + 1 == len(loader))
        if should_step:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            if not torch.isfinite(torch.as_tensor(grad_norm, device=device)):
                nonfinite_batches += 1
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                accumulated_finite = 0
                _reduce_scaler_after_nonfinite(scaler)
                print(f"[WARN] epoch={epoch} step={step + 1}: non-finite gradient; update skipped")
                if ABORT_ON_NONFINITE or nonfinite_batches > MAX_NONFINITE_BATCHES_PER_EPOCH:
                    raise FloatingPointError(
                        f"Non-finite gradient detected at epoch={epoch}, step={step + 1}. "
                        "The current epoch was aborted before saving a checkpoint."
                    )
                continue

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accumulated_finite = 0

        total_loss += float(loss_full.item())
        n_batches += 1
        progress.set_postfix(
            loss=f"{total_loss / max(n_batches, 1):.4f}",
            bad=nonfinite_batches,
        )

    if n_batches == 0:
        raise FloatingPointError(f"Epoch {epoch} contained no finite training batches.")
    return total_loss / n_batches, nonfinite_batches, skipped_steps


@torch.inference_mode()
def evaluate_single_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    store_cases: bool = False,
) -> Tuple[Dict[str, float | int], List[Dict[str, object]]]:
    model.eval()
    acc = MetricAccumulator(threshold=float(threshold))
    for batch in tqdm(loader, desc=f"eval t={threshold:.2f}", leave=False, unit="batch"):
        images = batch["image"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        with autocast_context(False, torch.float32):
            logits = unwrap_primary_output(model(images))
            probs = torch.softmax(logits.float(), dim=1)[:, 1]
        acc.update(
            probs,
            target,
            case_names=batch.get("case_name"),
            patient_ids=batch.get("patient_id"),
            store_cases=store_cases,
        )
    return acc.summary(), acc.case_rows


def write_rows_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler,
    epoch: int,
    best_score: float,
    model_name: str,
    model_cfg: Dict[str, object],
) -> None:
    payload = {
        "epoch": epoch,
        "model_name": model_name,
        "model_config": model_cfg,
        "model_state_dict": safe_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_score": best_score,
        "seed": SEED,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    scaler=None,
) -> Dict[str, object]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        except Exception:
            pass
    return ckpt


def build_dataloaders(
    model_name: str,
    spec: Dict[str, object],
    preset: Dict[str, object],
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    train_ds = ConsolidationDataset(DATA_ROOT / "train", train=True, image_size=IMAGE_SIZE)
    val_ds = ConsolidationDataset(
        DATA_ROOT / "val",
        train=False,
        image_size=IMAGE_SIZE,
        max_samples=preset["max_val_samples"],
    )
    test_ds = ConsolidationDataset(
        DATA_ROOT / "test",
        train=False,
        image_size=IMAGE_SIZE,
        max_samples=preset["max_test_samples"],
    )
    check_patient_leakage({"train": train_ds, "val": val_ds, "test": test_ds})

    sampler = build_train_sampler(train_ds, OUTPUT_ROOT / model_name / RUN_MODE, preset["samples_per_epoch"])
    generator = torch.Generator().manual_seed(SEED)
    common_kwargs = {
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY and torch.cuda.is_available(),
        "worker_init_fn": worker_init_fn,
        "persistent_workers": NUM_WORKERS > 0,
    }
    train_loader = DataLoader(
        train_ds,
        batch_size=int(spec["batch_size"]),
        sampler=sampler,
        shuffle=sampler is None,
        drop_last=True,
        generator=generator,
        **common_kwargs,
    )
    eval_bs = max(1, int(spec["batch_size"]))
    val_loader = DataLoader(val_ds, batch_size=eval_bs, shuffle=False, drop_last=False, **common_kwargs)
    test_loader = DataLoader(test_ds, batch_size=eval_bs, shuffle=False, drop_last=False, **common_kwargs)
    counts = {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)}
    return train_loader, val_loader, test_loader, counts


def train_and_evaluate_one(model_name: str, device: torch.device, preset: Dict[str, object]) -> Dict[str, object]:
    spec = get_model_spec(model_name)
    model_dir = OUTPUT_ROOT / model_name / RUN_MODE
    result_path = model_dir / "result.json"
    if SKIP_COMPLETED_MODELS and result_path.exists():
        print(f"[SKIP] {model_name}: completed result exists at {result_path}")
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)

    model_dir.mkdir(parents=True, exist_ok=True)
    save_json(model_dir / "run_settings.json", {
        "run_mode": RUN_MODE,
        "model_name": model_name,
        "model_spec": spec,
        "preset": preset,
        "data_root": str(DATA_ROOT),
        "image_size": IMAGE_SIZE,
        "seed": SEED,
        "pretrained": PRETRAINED,
        "numerical_stability": {
            "default_amp_dtype": DEFAULT_AMP_DTYPE,
            "max_nonfinite_batches_per_epoch": MAX_NONFINITE_BATCHES_PER_EPOCH,
            "abort_on_nonfinite": ABORT_ON_NONFINITE,
        },
        "loss": {
            "ce_weight": CE_WEIGHT,
            "dice_weight": DICE_WEIGHT,
            "foreground_ce_weight": FOREGROUND_CE_WEIGHT,
        },
    })

    print("\n" + "=" * 88)
    print(f"MODEL: {model_name} | size={IMAGE_SIZE} | mode={RUN_MODE} | device={device}")
    print("=" * 88)

    train_loader, val_loader, test_loader, dataset_counts = build_dataloaders(model_name, spec, preset)
    model, model_cfg = build_april_model(model_name)
    model = model.to(device)
    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: total={params_total/1e6:.2f}M, trainable={params_trainable/1e6:.2f}M")
    print(
        f"Batch={spec['batch_size']}, grad_accum={spec['grad_accum']}, "
        f"effective_batch={int(spec['batch_size']) * int(spec['grad_accum'])}"
    )

    criterion = ConsolidationLoss(device)
    amp_dtype = resolve_amp_dtype(spec.get("amp_dtype", DEFAULT_AMP_DTYPE), device)
    grad_clip_norm = float(spec.get("grad_clip_norm", GRAD_CLIP_NORM))
    warmup_epochs = int(spec.get("warmup_epochs", preset["warmup_epochs"]))
    print(
        f"Numerics: amp_dtype={str(amp_dtype).replace('torch.', '')}, "
        f"grad_clip={grad_clip_norm}, warmup_epochs={warmup_epochs}"
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(spec["lr"]),
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-6,
    )
    scheduler = make_scheduler(
        optimizer,
        epochs=int(preset["epochs"]),
        warmup_epochs=warmup_epochs,
    )
    scaler = make_grad_scaler(USE_AMP and device.type == "cuda", amp_dtype)

    best_path = model_dir / "best_model.pth"
    last_path = model_dir / "last_model.pth"
    history_path = model_dir / "history.csv"
    start_epoch = 1
    best_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    if RESUME_FROM_LAST and last_path.exists():
        ckpt = load_checkpoint(last_path, model, device, optimizer, scheduler, scaler)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_score = float(ckpt.get("best_score", -float("inf")))
        best_epoch = int(ckpt.get("best_epoch", 0))
        # Validation runs every epoch in formal mode, so this exactly restores
        # the early-stopping counter after an interrupted run.
        epochs_without_improvement = max(0, int(ckpt.get("epoch", 0)) - best_epoch)
        print(
            f"Resumed from epoch {start_epoch}; best_score={best_score:.6f}; "
            f"no_improve={epochs_without_improvement}/{int(preset['patience'])}"
        )

    history_fields = [
        "epoch", "train_loss", "nonfinite_batches", "skipped_steps", "lr", "seconds", "threshold",
        "dice_all", "dice_positive", "dice_nonempty_union",
        "iou_all", "iou_positive", "negative_empty_accuracy",
        "negative_false_positive_rate", "global_dice", "global_iou",
        "precision", "recall", "best_score", "best_epoch",
    ]

    train_start = time.time()
    for epoch in range(start_epoch, int(preset["epochs"]) + 1):
        epoch_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_loss, nonfinite_batches, skipped_steps = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch,
            int(spec["grad_accum"]),
            amp_dtype,
            grad_clip_norm,
        )
        scheduler.step()

        val_metrics: Dict[str, float | int] = {
            "threshold": FIXED_VALIDATION_THRESHOLD,
            "dice_all": float("nan"),
            "dice_positive": float("nan"),
            "dice_nonempty_union": float("nan"),
            "iou_all": float("nan"),
            "iou_positive": float("nan"),
            "negative_empty_accuracy": float("nan"),
            "negative_false_positive_rate": float("nan"),
            "global_dice": float("nan"),
            "global_iou": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
        }
        if epoch % int(preset["val_interval"]) == 0:
            val_metrics, _ = evaluate_single_threshold(
                model,
                val_loader,
                device,
                FIXED_VALIDATION_THRESHOLD,
                store_cases=False,
            )
            score = float(val_metrics[CHECKPOINT_SELECTION_METRIC])
            if score > best_score + EARLY_STOPPING_MIN_DELTA:
                best_score = score
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_score,
                    model_name,
                    model_cfg,
                )
                print(
                    f"New best: epoch={epoch}, {CHECKPOINT_SELECTION_METRIC}={best_score:.6f}, "
                    f"dice_all={float(val_metrics['dice_all']):.6f}"
                )
            else:
                epochs_without_improvement += 1

        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_score,
            model_name,
            model_cfg,
        )
        # Add convenience metadata not required for loading.
        try:
            ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
            ckpt["best_epoch"] = best_epoch
            torch.save(ckpt, last_path)
        except Exception:
            pass

        elapsed = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "nonfinite_batches": nonfinite_batches,
            "skipped_steps": skipped_steps,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
            **{k: val_metrics.get(k, "") for k in history_fields if k in val_metrics},
            "best_score": best_score,
            "best_epoch": best_epoch,
        }
        append_csv(history_path, row, history_fields)
        peak_gb = (
            torch.cuda.max_memory_allocated(device) / (1024**3)
            if device.type == "cuda" else 0.0
        )
        print(
            f"Epoch {epoch:03d}/{int(preset['epochs'])} | loss={train_loss:.5f} | "
            f"val_dice_pos={float(val_metrics['dice_positive']):.5f} | "
            f"val_dice_all={float(val_metrics['dice_all']):.5f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s | peak={peak_gb:.2f}GB"
        )

        if (
            RUN_MODE == "formal"
            and int(preset["patience"]) > 0
            and epochs_without_improvement >= int(preset["patience"])
        ):
            print(
                f"Early stopping at epoch {epoch}; no improvement for "
                f"{epochs_without_improvement} validations."
            )
            break

    if not best_path.exists():
        # This can happen only if validation produced NaN in an unusual dataset.
        shutil.copy2(last_path, best_path)
    load_checkpoint(best_path, model, device)

    fixed_val_metrics, _ = evaluate_single_threshold(
        model,
        val_loader,
        device,
        FIXED_VALIDATION_THRESHOLD,
        store_cases=False,
    )

    test_metrics, test_cases = evaluate_single_threshold(
        model,
        test_loader,
        device,
        FIXED_VALIDATION_THRESHOLD,
        store_cases=True,
    )
    write_rows_csv(model_dir / "test_cases.csv", test_cases)

    total_seconds = time.time() - train_start
    peak_memory_gb = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
        if device.type == "cuda" else 0.0
    )
    result: Dict[str, object] = {
        "model": model_name,
        "image_size": IMAGE_SIZE,
        "run_id": RUN_ID,
        "architecture": spec["architecture"],
        "run_mode": RUN_MODE,
        "best_epoch": best_epoch,
        "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
        "best_fixed_threshold_val_score": best_score,
        "threshold_policy": "fixed",
        "threshold": FIXED_VALIDATION_THRESHOLD,
        "val_at_fixed_threshold": fixed_val_metrics,
        "test": test_metrics,
        "dataset_counts": dataset_counts,
        "parameters_total": params_total,
        "parameters_trainable": params_trainable,
        "batch_size": int(spec["batch_size"]),
        "grad_accum": int(spec["grad_accum"]),
        "effective_batch_size": int(spec["batch_size"]) * int(spec["grad_accum"]),
        "pretrained": PRETRAINED,
        "total_seconds": total_seconds,
        "peak_memory_gb": peak_memory_gb,
        "output_dir": str(model_dir),
    }
    save_json(result_path, result)
    print(
        f"FINAL size={IMAGE_SIZE} {model_name}: threshold={FIXED_VALIDATION_THRESHOLD:.2f}, "
        f"test_dice_positive={float(test_metrics['dice_positive']):.6f}, "
        f"test_dice_all={float(test_metrics['dice_all']):.6f}, "
        f"precision={float(test_metrics['precision']):.6f}, "
        f"recall={float(test_metrics['recall']):.6f}"
    )
    return result


def write_aggregate_summary(results: Sequence[Dict[str, object]]) -> Path:
    path = OUTPUT_ROOT / f"benchmark_summary_{RUN_MODE}.csv"
    rows: List[Dict[str, object]] = []
    for result in results:
        test = dict(result.get("test", {}))
        val = dict(result.get("val_at_fixed_threshold", {}))
        rows.append(
            {
                "model": result.get("model"),
                "image_size": result.get("image_size"),
                "architecture": result.get("architecture"),
                "run_mode": result.get("run_mode"),
                "best_epoch": result.get("best_epoch"),
                "threshold": result.get("threshold"),
                "val_dice_positive": val.get("dice_positive"),
                "val_dice_all": val.get("dice_all"),
                "test_dice_positive": test.get("dice_positive"),
                "test_dice_all": test.get("dice_all"),
                "test_dice_nonempty_union": test.get("dice_nonempty_union"),
                "test_global_dice": test.get("global_dice"),
                "test_iou_positive": test.get("iou_positive"),
                "test_precision": test.get("precision"),
                "test_recall": test.get("recall"),
                "test_negative_empty_accuracy": test.get("negative_empty_accuracy"),
                "test_negative_false_positive_rate": test.get("negative_false_positive_rate"),
                "parameters_m": float(result.get("parameters_total", 0)) / 1e6,
                "peak_memory_gb": result.get("peak_memory_gb"),
                "total_hours": float(result.get("total_seconds", 0)) / 3600.0,
                "output_dir": result.get("output_dir"),
            }
        )
    rows.sort(
        key=lambda r: float(r["test_dice_positive"])
        if r.get("test_dice_positive") is not None else -1.0,
        reverse=True,
    )
    write_rows_csv(path, rows)
    return path


def set_task_context(image_size: int, data_root: str | Path, output_base: str | Path) -> None:
    global IMAGE_SIZE, DATA_ROOT, OUTPUT_ROOT
    IMAGE_SIZE = int(image_size)
    DATA_ROOT = Path(data_root).expanduser().resolve()
    OUTPUT_ROOT = Path(output_base).expanduser().resolve() / f"size{IMAGE_SIZE}"


def _run_one_task(gpu_id: int, image_size: int, data_root: str, model_name: str, preset: Dict[str, object], output_base: str) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, str]]]:
    try:
        set_task_context(image_size, data_root, output_base)
        ensure_dataset_layout(DATA_ROOT)
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        seed_everything(SEED + int(image_size) + gpu_id)
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        print("\n" + "#" * 96)
        print(f"TASK START | gpu={gpu_id} | size={IMAGE_SIZE} | model={model_name}")
        print(f"DATA_ROOT={DATA_ROOT}")
        print(f"OUTPUT_ROOT={OUTPUT_ROOT}")
        print("#" * 96)
        result = train_and_evaluate_one(model_name, device, preset)
        return result, None
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(f"\n[FAILED] size={image_size} model={model_name}: {message}")
        traceback.print_exc()
        try:
            set_task_context(image_size, data_root, output_base)
            save_json(OUTPUT_ROOT / model_name / RUN_MODE / "failure.json", {
                "image_size": image_size,
                "model": model_name,
                "error": message,
            })
        except Exception:
            pass
        if FAIL_FAST:
            raise
        return None, {"image_size": str(image_size), "model": model_name, "error": message}
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _gpu_worker(gpu_id: int, task_queue, result_queue, preset: Dict[str, object], output_base: str) -> None:
    while True:
        task = task_queue.get()
        if task is None:
            break
        image_size, data_root, model_name = task
        result, failure = _run_one_task(gpu_id, image_size, data_root, model_name, preset, output_base)
        result_queue.put((result, failure))


def write_global_summary(output_base: Path, results: Sequence[Dict[str, object]], failures: Sequence[Dict[str, str]]) -> Path:
    path = output_base / f"benchmark_summary_all_{RUN_MODE}.csv"
    rows: List[Dict[str, object]] = []
    for result in results:
        test = dict(result.get("test", {}))
        val = dict(result.get("val_at_fixed_threshold", {}))
        rows.append({
            "run_id": result.get("run_id"),
            "image_size": result.get("image_size"),
            "model": result.get("model"),
            "architecture": result.get("architecture"),
            "run_mode": result.get("run_mode"),
            "best_epoch": result.get("best_epoch"),
            "threshold": result.get("threshold"),
            "val_dice_positive": val.get("dice_positive"),
            "val_dice_all": val.get("dice_all"),
            "test_dice_positive": test.get("dice_positive"),
            "test_dice_all": test.get("dice_all"),
            "test_dice_nonempty_union": test.get("dice_nonempty_union"),
            "test_global_dice": test.get("global_dice"),
            "test_iou_positive": test.get("iou_positive"),
            "test_precision": test.get("precision"),
            "test_recall": test.get("recall"),
            "test_negative_empty_accuracy": test.get("negative_empty_accuracy"),
            "test_negative_false_positive_rate": test.get("negative_false_positive_rate"),
            "parameters_m": float(result.get("parameters_total", 0)) / 1e6,
            "peak_memory_gb": result.get("peak_memory_gb"),
            "total_hours": float(result.get("total_seconds", 0)) / 3600.0,
            "output_dir": result.get("output_dir"),
        })
    rows.sort(
        key=lambda r: (int(r.get("image_size") or 0), float(r.get("test_dice_positive") or -1.0)),
        reverse=True,
    )
    write_rows_csv(path, rows)
    save_json(output_base / f"run_report_all_{RUN_MODE}.json", {
        "run_id": RUN_ID,
        "run_mode": RUN_MODE,
        "results": results,
        "failures": list(failures),
        "summary_csv": str(path),
    })
    return path


def main() -> None:
    if RUN_MODE not in RUN_PRESETS:
        raise ValueError(f"RUN_MODE must be one of {list(RUN_PRESETS)}, got {RUN_MODE!r}")
    unknown = [m for m in MODELS_TO_RUN if m not in MODEL_SPECS]
    if unknown:
        raise KeyError(f"Unknown models in MODELS_TO_RUN: {unknown}")
    for size in RUN_SIZES:
        if int(size) not in DATA_ROOTS:
            raise KeyError(f"DATA_ROOTS is missing size {size}")
        ensure_dataset_layout(Path(DATA_ROOTS[int(size)]).expanduser().resolve())

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this benchmark")
    visible = torch.cuda.device_count()
    bad = [g for g in GPU_IDS if g < 0 or g >= visible]
    if bad:
        raise ValueError(f"Invalid GPU_IDS={bad}; visible CUDA devices={visible}")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    print(f"RUN_ID: {RUN_ID}")
    print(f"OUTPUT_BASE: {OUTPUT_BASE}")
    print(f"RUN_MODE: {RUN_MODE}")
    print(f"RUN_SIZES: {RUN_SIZES}")
    print(f"MODELS: {MODELS_TO_RUN}")
    print(f"GPU_IDS: {GPU_IDS}")
    print(
        f"FORMAL POLICY: max_epochs={RUN_PRESETS['formal']['epochs']}, "
        f"patience={RUN_PRESETS['formal']['patience']}, "
        f"samples_per_epoch={RUN_PRESETS['formal']['samples_per_epoch']}"
    )

    preset = dict(RUN_PRESETS[RUN_MODE])
    tasks = []
    for size in RUN_SIZES:
        root = str(Path(DATA_ROOTS[int(size)]).expanduser().resolve())
        for model_name in MODELS_TO_RUN:
            tasks.append((int(size), root, model_name))

    results: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []

    if len(GPU_IDS) <= 1:
        gpu = int(GPU_IDS[0])
        for image_size, data_root, model_name in tasks:
            result, failure = _run_one_task(gpu, image_size, data_root, model_name, preset, str(OUTPUT_BASE))
            if result is not None:
                results.append(result)
            if failure is not None:
                failures.append(failure)
    else:
        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()
        for task in tasks:
            task_queue.put(task)
        for _ in GPU_IDS:
            task_queue.put(None)

        procs = []
        for gpu in GPU_IDS:
            p = ctx.Process(
                target=_gpu_worker,
                args=(int(gpu), task_queue, result_queue, preset, str(OUTPUT_BASE)),
                daemon=False,
            )
            p.start()
            procs.append(p)

        remaining = len(tasks)
        while remaining > 0:
            result, failure = result_queue.get()
            remaining -= 1
            if result is not None:
                results.append(result)
            if failure is not None:
                failures.append(failure)

        for p in procs:
            p.join()
            if p.exitcode not in (0, None):
                failures.append({"image_size": "unknown", "model": "worker", "error": f"worker exitcode={p.exitcode}"})

    summary_path = write_global_summary(OUTPUT_BASE, results, failures)
    print("\n" + "=" * 88)
    print(f"Completed tasks: {len(results)}; failed tasks: {len(failures)}")
    print(f"Summary: {summary_path}")
    if failures:
        for item in failures:
            print(f"  FAILED size={item.get('image_size')} model={item.get('model')}: {item.get('error')}")


if __name__ == "__main__":
    main()
