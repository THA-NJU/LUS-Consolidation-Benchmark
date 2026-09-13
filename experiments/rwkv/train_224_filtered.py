#!/usr/bin/env python3
"""
Local Size_224 RWKV benchmark adapted from the retained Size_512 RWKV backend.

Models:
  - rwkv_unet : RWKV-UNet variant B
  - u_rwkv    : U-RWKV

This script deliberately reuses the reviewed Size_512 RWKV implementation for:
  - APRIL model construction
  - U-RWKV public-reference SpatialMix alignment patch
  - WKV CUDA loading
  - CE + Dice loss
  - optimizer/scheduler AMP helpers
  - stable gradient-norm recovery
  - same-accumulation-group FP32 replay

What changes for Size_224:
  1) IMAGE_SIZE = 224.
  2) Data root points to Size_224_filtered.
  3) No min-foreground-pixel >=100 filter. Every retained patch must simply
     contain foreground, because the dataset builder has already performed the
     empty-patch and edge-touching filtering.
  4) No 512 reconstruction. Every retained 224x224 patch is one evaluation
     sample.
  5) Best checkpoint = highest mean Dice across validation 224 patches at
     threshold 0.5.
  6) Test metrics are computed directly across test 224 patches.

IMPORTANT:
  Run RWKV-UNet and U-RWKV in SEPARATE Python processes. Their native WKV
  libraries register the same PyTorch namespace. The companion shell launcher
  does this automatically.

Examples:
  python experiments/rwkv/train_224_filtered.py \
      --model rwkv_unet --run-mode smoke --gpu 0

  python experiments/rwkv/train_224_filtered.py \
      --model u_rwkv --run-mode formal --gpu 0
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


IMAGE_SIZE = 224
NUM_CLASSES = 2
FIXED_THRESHOLD = 0.50
IN_CHANNELS = 3

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "rwkv_unet": {
        "architecture": "rwkv_unet",
        "arch_params": {"variant": "b"},
        # Keep the local Size_512 effective batch unchanged by default.
        "batch_size": 32,
        "grad_accum": 4,
        # Size_224 first RWKV sequence is 28x28=784; 1024 is sufficient.
        "kernel_tmax": 1024,
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
        "batch_size": 32,
        "grad_accum": 4,
        # WKV follows the 2x stem: 112x112=12544 tokens.
        "kernel_tmax": 16384,
    },
}

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


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_data = (
        script_dir
        / "./datasets/Size_224_filtered"
    ).resolve()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--april-root",
        type=Path,
        default=script_dir,
        help="APRIL root containing the bundled medseg/ subset.",
    )
    parser.add_argument(
        "--base512-script",
        type=Path,
        default=None,
        help=(
            "Path to the retained Size_512 RWKV backend. "
            "The unified dispatcher supplies experiments/rwkv/train_512.py."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=default_data)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_SPECS),
        default="rwkv_unet",
        help="Run exactly one RWKV model per Python process.",
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
        "--amp-dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
    )
    parser.add_argument(
        "--max-fp32-retries-per-epoch",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--max-stable-norm-recoveries-per-epoch",
        type=int,
        default=3,
    )

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def seed_everything(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: Dict[str, Any], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def patient_id(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) != 3 or not parts[0].lower().startswith("p"):
        raise ValueError(f"Unexpected patch filename: {filename}")
    return parts[0].lower()


def validate_dataset_root(root: Path) -> Dict[str, Any]:
    patient_sets: Dict[str, set[str]] = {}
    sample_sets: Dict[str, set[str]] = {}
    counts: Dict[str, int] = {}

    for split in ("train", "val", "test"):
        image_dir = root / split / "images"
        mask_dir = root / split / "masks"

        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(
                f"Missing {image_dir} or {mask_dir}"
            )

        image_names = {p.name for p in image_dir.glob("*.png")}
        mask_names = {p.name for p in mask_dir.glob("*.png")}

        if image_names != mask_names:
            raise RuntimeError(
                f"{split} image/mask mismatch: "
                f"images={len(image_names)}, masks={len(mask_names)}"
            )
        if not image_names:
            raise RuntimeError(f"No PNG samples in split={split}")

        counts[split] = len(image_names)
        sample_sets[split] = image_names
        patient_sets[split] = {patient_id(x) for x in image_names}

    splits = ("train", "val", "test")
    for i, left in enumerate(splits):
        for right in splits[i + 1:]:
            sample_overlap = sample_sets[left] & sample_sets[right]
            patient_overlap = patient_sets[left] & patient_sets[right]
            if sample_overlap:
                raise RuntimeError(
                    f"Sample leakage {left}/{right}: "
                    f"{sorted(sample_overlap)[:10]}"
                )
            if patient_overlap:
                raise RuntimeError(
                    f"Patient leakage {left}/{right}: "
                    f"{sorted(patient_overlap)[:10]}"
                )

    return {
        "sample_counts": counts,
        "patient_counts": {
            split: len(patient_sets[split]) for split in splits
        },
        "patient_overlap": False,
    }


def load_base512_module(
    april_root: Path,
    base512_script: Optional[Path],
):
    path = (
        base512_script.expanduser().resolve()
        if base512_script is not None
        else Path(__file__).resolve().parent / "train_512.py"
    )

    if not path.is_file():
        raise FileNotFoundError(
            "The verified Size_512 RWKV v3 script is required so that the "
            "Size_224 run reuses exactly the same U-RWKV correction and "
            f"non-finite-gradient recovery logic. Missing: {path}"
        )

    module_name = "_rwkv_size512_v3_base"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base script: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Switch the proven training helpers into Size_224 mode.
    module.IMAGE_SIZE = IMAGE_SIZE
    module.FIXED_THRESHOLD = FIXED_THRESHOLD

    return module, path


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Filtered224Dataset(Dataset):
    """
    Every file in Size_224_filtered is a retained positive patch.

    This class does NOT perform another lesion-area threshold. It rejects an
    empty mask because that indicates the prepared dataset violates the final
    Size_224 protocol.
    """

    def __init__(
        self,
        split_root: Path,
        *,
        augment: bool,
        max_samples: Optional[int],
        base_module,
    ) -> None:
        self.split_root = Path(split_root)
        self.image_dir = self.split_root / "images"
        self.mask_dir = self.split_root / "masks"
        self.augment = bool(augment)
        self.base = base_module

        image_map = {
            p.name: p for p in sorted(self.image_dir.glob("*.png"))
        }
        mask_map = {
            p.name: p for p in sorted(self.mask_dir.glob("*.png"))
        }

        if image_map.keys() != mask_map.keys():
            raise RuntimeError(
                f"Image/mask mismatch under {split_root}"
            )

        names = sorted(image_map)
        if max_samples is not None:
            names = names[: int(max_samples)]

        self.samples: List[Tuple[Path, Path, str]] = [
            (image_map[name], mask_map[name], name) for name in names
        ]

        if not self.samples:
            raise RuntimeError(f"No samples in {split_root}")

        bad: List[str] = []
        empty: List[str] = []

        for image_path, mask_path, name in self.samples:
            with Image.open(image_path) as image:
                if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                    bad.append(f"{name}: image={image.size}")

            with Image.open(mask_path) as mask:
                if mask.size != (IMAGE_SIZE, IMAGE_SIZE):
                    bad.append(f"{name}: mask={mask.size}")
                arr = np.asarray(mask.convert("L"))
                if not np.any(arr > 0):
                    empty.append(name)

        if bad:
            raise RuntimeError(
                "Non-224 samples found: " + ", ".join(bad[:10])
            )
        if empty:
            raise RuntimeError(
                "Size_224_filtered contains empty masks, which should already "
                "have been removed. Examples: "
                + ", ".join(empty[:10])
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_path, mask_path, name = self.samples[index]

        with Image.open(image_path) as f:
            image = f.convert("L")
        with Image.open(mask_path) as f:
            mask = f.convert("L")

        # Reuse the exact augmentation from the verified Size_512 v3 script.
        if self.augment:
            image, mask = self.base.ConsolidationDataset.augment_pair(
                image, mask
            )

        image_arr = np.asarray(image, dtype=np.float32) / 255.0

        # Match the Size_512 v3 noise augmentation.
        if self.augment and random.random() < 0.15:
            noise = np.random.normal(
                0.0, 0.02, image_arr.shape
            ).astype(np.float32)
            image_arr = image_arr + noise

        image_arr = np.clip(image_arr, 0.0, 1.0)
        image_arr = np.repeat(image_arr[None, ...], 3, axis=0)
        target_arr = (np.asarray(mask) > 0).astype(np.int64)

        return {
            "image": torch.from_numpy(
                np.ascontiguousarray(image_arr)
            ).float(),
            "target": torch.from_numpy(
                np.ascontiguousarray(target_arr)
            ).long(),
            "name": name,
        }


def build_loaders(
    data_root: Path,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    augment: bool,
    max_train_samples: Optional[int],
    max_val_samples: Optional[int],
    max_test_samples: Optional[int],
    base_module,
):
    train_set = Filtered224Dataset(
        data_root / "train",
        augment=augment,
        max_samples=max_train_samples,
        base_module=base_module,
    )
    val_set = Filtered224Dataset(
        data_root / "val",
        augment=False,
        max_samples=max_val_samples,
        base_module=base_module,
    )
    test_set = Filtered224Dataset(
        data_root / "test",
        augment=False,
        max_samples=max_test_samples,
        base_module=base_module,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
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


# ---------------------------------------------------------------------------
# Metrics: direct 224-patch evaluation
# ---------------------------------------------------------------------------

def safe_div(num: float, den: float, empty: float = 0.0) -> float:
    if den <= 0:
        return float(empty)
    return float(num / den)


@dataclass
class PatchMetrics:
    threshold: float = FIXED_THRESHOLD
    count: int = 0

    dice_sum: float = 0.0
    iou_sum: float = 0.0
    precision_sum: float = 0.0
    recall_sum: float = 0.0
    specificity_sum: float = 0.0
    accuracy_sum: float = 0.0

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    dice_values: List[float] = field(default_factory=list)
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        probability: torch.Tensor,
        target: torch.Tensor,
        name: str,
        *,
        store_row: bool,
    ) -> None:
        pred = probability >= self.threshold
        gt = target > 0

        tp = int((pred & gt).sum().item())
        fp = int((pred & ~gt).sum().item())
        fn = int((~pred & gt).sum().item())
        tn = int((~pred & ~gt).sum().item())

        dice = safe_div(2 * tp, 2 * tp + fp + fn, 1.0)
        iou = safe_div(tp, tp + fp + fn, 1.0)
        precision = safe_div(tp, tp + fp, 0.0)
        recall = safe_div(tp, tp + fn, 0.0)
        specificity = safe_div(tn, tn + fp, 0.0)
        accuracy = safe_div(tp + tn, tp + fp + fn + tn, 0.0)

        self.count += 1
        self.dice_sum += dice
        self.iou_sum += iou
        self.precision_sum += precision
        self.recall_sum += recall
        self.specificity_sum += specificity
        self.accuracy_sum += accuracy

        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
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
        if self.count == 0:
            raise RuntimeError("No evaluation samples")

        values = np.asarray(self.dice_values, dtype=np.float64)

        return {
            "num_images": self.count,
            "mean_dice": self.dice_sum / self.count,
            "mean_iou": self.iou_sum / self.count,
            "precision_macro": self.precision_sum / self.count,
            "recall_macro": self.recall_sum / self.count,
            "specificity_macro": self.specificity_sum / self.count,
            "accuracy_macro": self.accuracy_sum / self.count,

            "global_dice": safe_div(
                2 * self.tp,
                2 * self.tp + self.fp + self.fn,
                1.0,
            ),
            "global_iou": safe_div(
                self.tp,
                self.tp + self.fp + self.fn,
                1.0,
            ),
            "precision_micro": safe_div(
                self.tp,
                self.tp + self.fp,
                0.0,
            ),
            "recall_micro": safe_div(
                self.tp,
                self.tp + self.fn,
                0.0,
            ),
            "specificity_micro": safe_div(
                self.tn,
                self.tn + self.fp,
                0.0,
            ),
            "accuracy_micro": safe_div(
                self.tp + self.tn,
                self.tp + self.fp + self.fn + self.tn,
                0.0,
            ),

            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,

            "dice_p25": float(np.percentile(values, 25)),
            "dice_median": float(np.percentile(values, 50)),
            "dice_p75": float(np.percentile(values, 75)),
            "threshold": self.threshold,
        }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    base_module,
    *,
    store_rows: bool,
    description: str,
):
    model.eval()
    accumulator = PatchMetrics()

    for batch in tqdm(loader, desc=description, leave=False):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        names = batch["name"]

        with base_module.autocast_context(amp_dtype, device):
            logits = base_module.primary_logits(
                model(image),
                target.shape[-2:],
            )

        probability = torch.softmax(logits.float(), dim=1)[:, 1]

        for i, name in enumerate(names):
            accumulator.add(
                probability[i].cpu(),
                target[i].cpu(),
                str(name),
                store_row=store_rows,
            )

    return accumulator.compute(), accumulator.rows


def dice_bins(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts = [0] * 10
    for row in rows:
        index = min(int(float(row["dice"]) * 10), 9)
        counts[index] += 1

    total = max(len(rows), 1)
    result = []
    for index, count in enumerate(counts):
        result.append({
            "bin": f"{index*10:02d}-{(index+1)*10:02d}%",
            "count": count,
            "fraction": count / total,
        })
    return result


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def _check_rwkv_environment(
    model_name: str,
    april_root: Path,
) -> None:
    """Check prerequisites without loading or registering WKV."""
    if not torch.cuda.is_available():
        raise RuntimeError(f"{model_name} requires a CUDA GPU")

    required = [
        april_root / "medseg" / "kernels" / "wkv" / "__init__.py",
        april_root / "medseg" / "kernels" / "wkv" / "wkv_op.cpp",
        april_root / "medseg" / "kernels" / "wkv" / "wkv_cuda.cu",
        april_root / "medseg" / "models" / "networks" / "rwkv" / "rwkv_unet.py",
        april_root / "medseg" / "models" / "networks" / "rwkv" / "u_rwkv.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "APRIL checkout is missing RWKV/WKV files:\n  "
            + "\n  ".join(missing)
        )

    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None or shutil.which("nvcc") is None:
        raise RuntimeError("RWKV WKV requires a local CUDA toolkit/NVCC.")
    if (
        shutil.which("ninja") is None
        and importlib.util.find_spec("ninja") is None
    ):
        raise RuntimeError(
            "RWKV WKV requires Ninja. Install with: python -m pip install ninja"
        )

    capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    os.environ.setdefault(
        "TORCH_CUDA_ARCH_LIST",
        f"{capability[0]}.{capability[1]}",
    )
    os.environ.setdefault("MAX_JOBS", "4")

    print(
        "[RWKV] environment OK; benchmark does not preload WKV. "
        "APRIL model code owns WKV registration."
    )
    print(
        f"[RWKV] model={model_name}, torch={torch.__version__}, "
        f"torch_cuda={torch.version.cuda}, CUDA_HOME={CUDA_HOME}"
    )
    print(
        f"[RWKV] GPU={torch.cuda.get_device_name(torch.cuda.current_device())}, "
        f"compute_capability={capability[0]}.{capability[1]}"
    )


def prepare_model(
    model_name: str,
    *,
    april_root: Path,
    base_module,
):
    if str(april_root) not in sys.path:
        sys.path.insert(0, str(april_root))

    raw_spec = MODEL_SPECS[model_name]
    base_module.IMAGE_SIZE = IMAGE_SIZE

    # Critical fix:
    # Do not call base_module.require_wkv_cuda() here.
    # APRIL's RWKV model implementation will load/register WKV itself.
    _check_rwkv_environment(model_name, april_root)

    patch_id = None
    if model_name == "u_rwkv":
        patch_id = base_module.apply_reference_aligned_u_rwkv_patch()
        print(f"U-RWKV reference-alignment patch: {patch_id}")

    from medseg.model_builder import build_model

    model_cfg = {
        "model": {
            "architecture": raw_spec["architecture"],
            "num_classes": NUM_CLASSES,
            "img_size": IMAGE_SIZE,
            "encoder": {
                "in_channels": IN_CHANNELS,
                "pretrained": False,
            },
            "arch_params": dict(raw_spec["arch_params"]),
        }
    }

    print(
        "[RWKV] building APRIL model; WKV registration must occur only "
        "inside the APRIL model implementation."
    )
    model = build_model(model_cfg)
    return model, model_cfg, patch_id


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    *,
    epoch: int,
    best_epoch: int,
    best_score: float,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "settings": settings,
    }


def load_checkpoint(
    path: Path,
    model: nn.Module,
    device: torch.device,
):
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    return payload


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> Dict[str, Any]:
    april_root = args.april_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()

    split_audit = validate_dataset_root(data_root)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(
            f"Invalid --gpu {args.gpu}; visible GPUs={torch.cuda.device_count()}"
        )

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    seed_everything(args.seed, args.deterministic)

    base, base_path = load_base512_module(
        april_root,
        args.base512_script,
    )

    preset = dict(PRESETS[args.run_mode])
    for key in ("epochs", "warmup_epochs", "patience"):
        value = getattr(args, key)
        if value is not None:
            preset[key] = int(value)

    for key in (
        "max_train_samples",
        "max_val_samples",
        "max_test_samples",
    ):
        value = getattr(args, key)
        if value is not None:
            preset[key] = int(value)

    model_spec = MODEL_SPECS[args.model]
    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(model_spec["batch_size"])
    )
    grad_accum = (
        int(args.grad_accum)
        if args.grad_accum is not None
        else int(model_spec["grad_accum"])
    )

    if batch_size < 1 or grad_accum < 1:
        raise ValueError("batch-size and grad-accum must be >=1")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else april_root
        / "output"
        / f"consolidation_size224_rwkv_{args.model}_{timestamp}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    if args.resume and args.output_dir is None:
        raise ValueError("--resume requires explicit --output-dir")

    result_path = output_root / "result.json"
    if args.skip_completed and result_path.is_file():
        with result_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    train_loader, val_loader, test_loader, counts = build_loaders(
        data_root,
        batch_size=batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        augment=not args.no_augment,
        max_train_samples=preset["max_train_samples"],
        max_val_samples=preset["max_val_samples"],
        max_test_samples=preset["max_test_samples"],
        base_module=base,
    )

    model, model_cfg, patch_id = prepare_model(
        args.model,
        april_root=april_root,
        base_module=base,
    )
    model = model.to(device)

    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    criterion = base.ConsolidationLoss(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = base.make_scheduler(
        optimizer,
        int(preset["epochs"]),
        int(preset["warmup_epochs"]),
    )
    amp_dtype = base.amp_dtype_from_name(args.amp_dtype, device)
    scaler = base.make_grad_scaler(amp_dtype)

    settings = {
        "model": args.model,
        "architecture": model_spec["architecture"],
        "arch_params": model_spec["arch_params"],
        "image_size": IMAGE_SIZE,
        "evaluation_unit": "individual_224_patch",
        "data_root": str(data_root),
        "base512_script": str(base_path),
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
        "amp_dtype": args.amp_dtype,
        "fixed_threshold": FIXED_THRESHOLD,
        "checkpoint_metric": "val_mean_dice_224",
        "kernel_tmax": "APRIL model default; no benchmark-side reload",
        "u_rwkv_reference_alignment_patch": patch_id,
        "max_fp32_retries_per_epoch": args.max_fp32_retries_per_epoch,
        "max_stable_norm_recoveries_per_epoch": (
            args.max_stable_norm_recoveries_per_epoch
        ),
        "dataset_counts": counts,
        "split_audit": split_audit,
        "seed": args.seed,
    }
    save_json(output_root / "run_settings.json", settings)

    best_path = output_root / "best_model.pth"
    last_path = output_root / "last_model.pth"
    history_path = output_root / "history.csv"

    history_fields = [
        "epoch",
        "train_loss",
        "val_mean_dice",
        "val_mean_iou",
        "val_global_dice",
        "val_global_iou",
        "val_precision_micro",
        "val_recall_micro",
        "val_specificity_micro",
        "lr",
        "seconds",
        "peak_memory_gib",
        "stable_norm_recoveries",
        "fp32_retries",
        "best_epoch",
        "best_score",
        "no_improvement",
    ]

    start_epoch = 1
    best_epoch = 0
    best_score = -float("inf")
    no_improvement = 0

    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError(
                f"--resume requested but missing {last_path}"
            )
        payload = load_checkpoint(last_path, model, device)
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        scaler.load_state_dict(payload.get("scaler_state", {}))

        start_epoch = int(payload["epoch"]) + 1
        best_epoch = int(payload.get("best_epoch", 0))
        best_score = float(payload.get("best_score", -float("inf")))

        if history_path.is_file():
            # Preserve prior no-improvement state approximately from best epoch.
            no_improvement = max(start_epoch - 1 - best_epoch, 0)

    print("=" * 96)
    print("LOCAL RWKV SIZE_224 BENCHMARK")
    print(f"model={args.model}")
    print(f"GPU={torch.cuda.get_device_name(device)}")
    print(f"data={data_root}")
    print(f"base512={base_path}")
    print(f"samples={counts}")
    print(f"patients={split_audit['patient_counts']}")
    print(
        f"batch={batch_size}, grad_accum={grad_accum}, "
        f"effective_batch={batch_size * grad_accum}"
    )
    print(
        "WKV loading=APRIL model-owned (no benchmark preload), "
        f"amp={args.amp_dtype}, lr={args.lr}"
    )
    print(
        "BEST POLICY: maximum validation mean Dice directly over retained "
        "224x224 patches at threshold 0.5."
    )
    print("NO 512 reconstruction. NO lesion-area >=100 secondary filter.")
    print("=" * 96)

    train_start = time.time()
    peak_memory = 0.0

    for epoch in range(start_epoch, int(preset["epochs"]) + 1):
        epoch_start = time.time()
        torch.cuda.reset_peak_memory_stats(device)

        train_loss, recoveries = base.train_one_epoch(
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
            diagnostic_path=output_root / "nonfinite_gradient_events.csv",
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
            base,
            store_rows=False,
            description=f"val epoch {epoch}",
        )

        score = float(val_metrics["mean_dice"])
        if not math.isfinite(score):
            raise FloatingPointError(
                f"Non-finite validation mean Dice: {score}"
            )

        improved = score > best_score + 1e-4

        if improved:
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
                    settings=settings,
                ),
                best_path,
            )
            print(
                f"New best: epoch={epoch}, "
                f"val_mean_dice={best_score:.6f}"
            )
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
                settings=settings,
            ),
            last_path,
        )

        epoch_peak = (
            torch.cuda.max_memory_allocated(device) / (1024**3)
        )
        peak_memory = max(peak_memory, epoch_peak)
        elapsed = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mean_dice": val_metrics["mean_dice"],
            "val_mean_iou": val_metrics["mean_iou"],
            "val_global_dice": val_metrics["global_dice"],
            "val_global_iou": val_metrics["global_iou"],
            "val_precision_micro": val_metrics["precision_micro"],
            "val_recall_micro": val_metrics["recall_micro"],
            "val_specificity_micro": val_metrics["specificity_micro"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
            "peak_memory_gib": epoch_peak,
            "stable_norm_recoveries": recoveries[
                "stable_norm_recoveries"
            ],
            "fp32_retries": recoveries["fp32_retries"],
            "best_epoch": best_epoch,
            "best_score": best_score,
            "no_improvement": no_improvement,
        }
        append_csv(history_path, row, history_fields)

        print(
            f"Epoch {epoch:03d}/{int(preset['epochs'])} | "
            f"loss={train_loss:.5f} | "
            f"val_mean_dice={score:.5f} | "
            f"val_mean_iou={val_metrics['mean_iou']:.5f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"peak={epoch_peak:.2f} GiB | "
            f"best={best_score:.5f}@{best_epoch} | "
            f"no_improve={no_improvement}/{int(preset['patience'])}"
        )

        if (
            int(preset["patience"]) > 0
            and no_improvement >= int(preset["patience"])
        ):
            print(f"Early stopping at epoch {epoch}")
            break

    if not best_path.is_file():
        raise RuntimeError("No best checkpoint saved")

    best_payload = load_checkpoint(best_path, model, device)

    val_metrics, val_rows = evaluate(
        model,
        val_loader,
        device,
        amp_dtype,
        base,
        store_rows=True,
        description="final val",
    )
    test_metrics, test_rows = evaluate(
        model,
        test_loader,
        device,
        amp_dtype,
        base,
        store_rows=True,
        description="test",
    )

    write_csv(output_root / "val_cases.csv", val_rows)
    write_csv(output_root / "test_cases.csv", test_rows)
    write_csv(output_root / "test_dice_bins.csv", dice_bins(test_rows))

    result = {
        "model": args.model,
        "architecture": model_spec["architecture"],
        "image_size": IMAGE_SIZE,
        "evaluation_unit": "individual_224_patch",
        "checkpoint_selection_metric": "val_mean_dice",
        "best_epoch": int(best_payload["best_epoch"]),
        "best_val_mean_dice": float(best_payload["best_score"]),
        "final_val": val_metrics,
        "test": test_metrics,
        "threshold": FIXED_THRESHOLD,
        "dataset_counts": counts,
        "parameters_total": params_total,
        "parameters_trainable": params_trainable,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "kernel_tmax": "APRIL model default; no benchmark-side reload",
        "amp_dtype": args.amp_dtype,
        "peak_memory_gib": peak_memory,
        "total_seconds": time.time() - train_start,
        "output_dir": str(output_root),
    }

    save_json(result_path, result)

    print(
        f"FINAL {args.model}: "
        f"test_mean_dice={test_metrics['mean_dice']:.6f}, "
        f"test_mean_iou={test_metrics['mean_iou']:.6f}, "
        f"global_dice={test_metrics['global_dice']:.6f}, "
        f"global_iou={test_metrics['global_iou']:.6f}, "
        f"precision={test_metrics['precision_micro']:.6f}, "
        f"recall={test_metrics['recall_micro']:.6f}, "
        f"specificity={test_metrics['specificity_micro']:.6f}"
    )

    return result


def main() -> None:
    args = parse_args()
    try:
        train(args)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
