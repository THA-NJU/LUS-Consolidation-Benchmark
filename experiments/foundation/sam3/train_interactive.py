#!/usr/bin/env python3
"""Fine-tune standard Meta SAM 3 on the Size_512 consolidation benchmark.

Benchmark protocol
------------------
* Prepared masks are binary; foreground defaults to value 1.
* Samples with fewer than ``--min-mask-pixels`` foreground pixels are excluded.
* Every retained training image is visited exactly once per epoch.
* Every image receives the same full-image box; the prompt contains no
  ground-truth-derived location information.
* SAM 3 keeps its native 1008 x 1008 interactive-image input.
* The best checkpoint is chosen only by mean per-image validation Dice at the
  fixed probability threshold (default 0.5).
* Test metrics are computed once after loading the best validation checkpoint.

This script uses the standard ``sam3.pt`` checkpoint and the official SAM 3
source tree.  It deliberately sets ``load_from_HF=False`` and never downloads
weights.  By default only the interactive prompt encoder and mask decoder are
trained, matching the decoder-only protocol used by the existing SAM2
benchmark.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
SAM3_IMAGE_MEAN = (0.5, 0.5, 0.5)
SAM3_IMAGE_STD = (0.5, 0.5, 0.5)
SAM3_NATIVE_SIZE = 1008
SAM3_SOURCE_REFERENCE = "46957e47805eaa273f4aa7bbbd25a88bca9108ce"
PATIENT_RE = re.compile(r"^(p\d+)(?:_|$)", flags=re.IGNORECASE)


def parse_int_values(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one foreground value is required.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run-mode", choices=("smoke", "formal"), default="formal")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("./datasets/Size_512"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("pretrained/sam3/sam3.pt"),
        help="Local standard SAM 3 checkpoint. SAM 3.1 multiplex is not used here.",
    )
    parser.add_argument(
        "--sam3-source-dir",
        type=Path,
        default=Path("./.external/sam3"),
        help="Official facebookresearch/sam3 source checkout.",
    )
    parser.add_argument(
        "--allow-source-mismatch",
        action="store_true",
        help="Allow a SAM3 source revision other than the frozen benchmark commit.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--train-scope",
        choices=("decoder", "image_decoder"),
        default="decoder",
        help="decoder is the benchmark default; image_decoder also trains the ViT.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("auto", "bf16", "fp16", "none"),
        default="auto",
    )

    parser.add_argument("--model-input-size", type=int, default=SAM3_NATIVE_SIZE)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--foreground-values", type=parse_int_values, default=(1,))
    parser.add_argument("--min-mask-pixels", type=int, default=100)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--iou-weight", type=float, default=0.1)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def apply_run_mode_defaults(args: argparse.Namespace) -> None:
    if args.run_mode == "smoke":
        defaults = {
            "epochs": 1,
            "patience": 2,
            "warmup_epochs": 0,
            "grad_accum": 1,
            "num_workers": 0,
            "max_train_samples": 4,
            "max_val_samples": 2,
            "max_test_samples": 2,
        }
    else:
        defaults = {
            "epochs": 600,
            "patience": 15,
            "warmup_epochs": 10,
            "grad_accum": 4,
            "num_workers": 4,
            "max_train_samples": 0,
            "max_val_samples": 0,
            "max_test_samples": 0,
        }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError("batch-size and grad-accum must both be positive")
    if args.epochs < 1 and not args.eval_only:
        raise ValueError("epochs must be positive")
    if args.patience < 1:
        raise ValueError("patience must be positive")
    if args.model_input_size != SAM3_NATIVE_SIZE:
        raise ValueError(
            f"Standard SAM 3 interactive checkpoints use {SAM3_NATIVE_SIZE}; "
            "do not change --model-input-size."
        )
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if args.eval_only and args.resume is None:
        raise ValueError("--eval-only requires --resume")
    if "multiplex" in args.checkpoint.name.lower():
        raise ValueError(
            "This trainer targets standard sam3.pt, not a SAM 3.1 multiplex checkpoint."
        )


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
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def normalized_mask_key(path: Path) -> str:
    stem = path.stem
    if stem.lower().endswith("_mask"):
        stem = stem[:-5]
    return stem


@dataclass(frozen=True)
class PairRecord:
    sample_id: str
    image_path: Path
    mask_path: Path


def resolve_pairs(split_root: Path) -> list[PairRecord]:
    image_paths = list_image_files(split_root / "images")
    mask_paths = list_image_files(split_root / "masks")

    images: dict[str, Path] = {}
    masks: dict[str, Path] = {}
    for path in image_paths:
        key = path.stem
        if key in images:
            raise RuntimeError(f"Duplicate image sample ID '{key}' in {split_root}")
        images[key] = path
    for path in mask_paths:
        key = normalized_mask_key(path)
        if key in masks:
            raise RuntimeError(f"Duplicate mask sample ID '{key}' in {split_root}")
        masks[key] = path

    missing_masks = sorted(images.keys() - masks.keys())
    extra_masks = sorted(masks.keys() - images.keys())
    if missing_masks or extra_masks:
        raise RuntimeError(
            f"Unpaired files in {split_root}: "
            f"missing_masks={missing_masks[:8]}, extra_masks={extra_masks[:8]}"
        )
    if not images:
        raise RuntimeError(f"No image-mask pairs found in {split_root}")
    return [
        PairRecord(key, images[key], masks[key])
        for key in sorted(images)
    ]


def patient_id(sample_id: str) -> str:
    match = PATIENT_RE.match(sample_id)
    if match is None:
        raise RuntimeError(
            f"Cannot extract patient ID from '{sample_id}'. Expected pXXX_... naming."
        )
    return match.group(1).lower()


def audit_and_write_split_manifest(
    data_root: Path,
    records_by_split: dict[str, list[PairRecord]],
    output_dir: Path,
) -> str:
    sample_sets = {
        split: {record.sample_id for record in records}
        for split, records in records_by_split.items()
    }
    patient_sets = {
        split: {patient_id(record.sample_id) for record in records}
        for split, records in records_by_split.items()
    }
    split_names = ("train", "val", "test")
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            sample_overlap = sorted(sample_sets[left] & sample_sets[right])
            patient_overlap = sorted(patient_sets[left] & patient_sets[right])
            if sample_overlap or patient_overlap:
                raise RuntimeError(
                    f"Split leakage between {left} and {right}: "
                    f"sample_overlap={sample_overlap[:8]}, "
                    f"patient_overlap={patient_overlap[:8]}"
                )

    rows: list[dict[str, str]] = []
    digest = hashlib.sha256()
    for split in split_names:
        for record in records_by_split[split]:
            row = {
                "split": split,
                "sample_id": record.sample_id,
                "patient_id": patient_id(record.sample_id),
                "image": str(record.image_path.relative_to(data_root)),
                "mask": str(record.mask_path.relative_to(data_root)),
            }
            rows.append(row)
            digest.update(
                (
                    f"{row['split']}\t{row['sample_id']}\t{row['patient_id']}\t"
                    f"{row['image']}\t{row['mask']}\n"
                ).encode("utf-8")
            )
    write_csv(
        output_dir / "split_manifest.csv",
        rows,
        ("split", "sample_id", "patient_id", "image", "mask"),
    )
    split_hash = digest.hexdigest()
    (output_dir / "split_manifest.sha256").write_text(
        split_hash + "\n", encoding="utf-8"
    )
    return split_hash


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
        records: Sequence[PairRecord],
        foreground_values: Sequence[int],
        min_mask_pixels: int,
        model_input_size: int,
        augment: bool,
        max_samples: int,
        split: str,
    ) -> None:
        self.foreground_values = tuple(foreground_values)
        self.model_input_size = model_input_size
        self.augment = augment
        self.split = split
        retained: list[PairRecord] = []
        excluded = 0
        for record in tqdm(records, desc=f"Scanning {split} masks", leave=False):
            mask = load_binary_mask(record.mask_path, self.foreground_values)
            if int(mask.sum().item()) >= min_mask_pixels:
                retained.append(record)
            else:
                excluded += 1
        if max_samples > 0:
            retained = retained[:max_samples]
        if not retained:
            raise RuntimeError(
                f"No {split} samples remain after min_mask_pixels={min_mask_pixels}"
            )
        self.records = retained
        self.total_pairs = len(records)
        self.excluded_pairs = excluded

    def __len__(self) -> int:
        return len(self.records)

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
            image = TF.adjust_brightness(image, random.uniform(0.85, 1.15))
        if random.random() < 0.3:
            image = TF.adjust_contrast(image, random.uniform(0.85, 1.15))
        return image.clamp_(0.0, 1.0), (mask > 0.5).float()

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.image_path) as image_pil:
            image = TF.pil_to_tensor(image_pil.convert("L")).float().div_(255.0)
        mask = load_binary_mask(record.mask_path, self.foreground_values)
        if image.shape[-2:] != mask.shape[-2:]:
            raise RuntimeError(
                f"Image/mask shape mismatch for {record.sample_id}: "
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
        image = TF.normalize(
            image.repeat(3, 1, 1),
            mean=SAM3_IMAGE_MEAN,
            std=SAM3_IMAGE_STD,
        )
        return {"image": image, "mask": mask, "name": record.image_path.name}


class SAM3FixedFullBox(nn.Module):
    """Differentiable official SAM 3 interactive path with a fixed full box."""

    def __init__(
        self,
        vision_trunk: nn.Module,
        interactive_neck: nn.ModuleList,
        prompt_encoder: nn.Module,
        mask_decoder: nn.Module,
        no_mem_embed: torch.Tensor,
        image_size: int,
    ) -> None:
        super().__init__()
        self.vision_trunk = vision_trunk
        self.interactive_neck = interactive_neck
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.no_mem_embed = nn.Parameter(
            no_mem_embed.detach().clone(), requires_grad=False
        )
        self.image_size = int(image_size)
        self.train_vision = False

    def _interactive_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        context = nullcontext() if self.train_vision else torch.no_grad()
        with context:
            trunk_outputs = self.vision_trunk(images)
            trunk_feature = trunk_outputs[-1]
            features = [neck(trunk_feature) for neck in self.interactive_neck]
            # Official SAM3VLBackbone uses scalp=1.
            features = features[:-1]
        if len(features) != 3:
            raise RuntimeError(
                f"Expected three SAM 3 interactive feature levels, got {len(features)}"
            )
        return features

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._interactive_features(images)
        high_res_features = [
            self.mask_decoder.conv_s0(features[0]),
            self.mask_decoder.conv_s1(features[1]),
        ]
        no_mem = self.no_mem_embed.permute(1, 2, 0).unsqueeze(-1)
        image_embedding = features[2] + no_mem

        batch_size = images.shape[0]
        edge = float(self.image_size)
        corners = images.new_tensor([[0.0, 0.0], [edge, edge]])
        point_coords = corners.unsqueeze(0).expand(batch_size, -1, -1)
        point_labels = torch.tensor(
            [2, 3], dtype=torch.int32, device=images.device
        ).unsqueeze(0).expand(batch_size, -1)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(point_coords, point_labels), boxes=None, masks=None
        )
        low_res_logits, iou_predictions, _, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
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


def source_commit(source_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_sam3_subset(
    source_dir: Path,
    checkpoint: Path,
    device: torch.device,
    image_size: int,
) -> tuple[SAM3FixedFullBox, str]:
    model_builder_path = source_dir / "sam3" / "model_builder.py"
    if not model_builder_path.is_file():
        raise FileNotFoundError(
            f"Official SAM 3 source not found at {source_dir}. "
            "Expected sam3/model_builder.py."
        )
    sys.path.insert(0, str(source_dir))
    from sam3.model_builder import build_sam3_image_model

    bpe_path = source_dir / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if not bpe_path.is_file():
        raise FileNotFoundError(f"SAM 3 BPE asset not found: {bpe_path}")
    print("Loading official SAM 3 checkpoint on CPU...")
    base_model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        device="cpu",
        eval_mode=False,
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        enable_segmentation=False,
        enable_inst_interactivity=True,
        compile=False,
    )
    if base_model.inst_interactive_predictor is None:
        raise RuntimeError("SAM 3 instance-interactive predictor was not constructed")
    dual_neck = base_model.backbone.vision_backbone
    if dual_neck.sam2_convs is None:
        raise RuntimeError("SAM 3 checkpoint/source lacks the interactive SAM neck")
    tracker = base_model.inst_interactive_predictor.model
    if int(tracker.image_size) != image_size:
        raise RuntimeError(
            f"Checkpoint interactive image size is {tracker.image_size}, expected {image_size}"
        )
    model = SAM3FixedFullBox(
        vision_trunk=dual_neck.trunk,
        interactive_neck=dual_neck.sam2_convs,
        prompt_encoder=tracker.sam_prompt_encoder,
        mask_decoder=tracker.sam_mask_decoder,
        no_mem_embed=tracker.no_mem_embed,
        image_size=image_size,
    )
    commit = source_commit(source_dir)
    del base_model, dual_neck, tracker
    gc.collect()
    model = model.to(device)
    return model, commit


def configure_trainable_parameters(model: SAM3FixedFullBox, scope: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    modules: list[nn.Module] = [model.prompt_encoder, model.mask_decoder]
    model.train_vision = scope == "image_decoder"
    if model.train_vision:
        modules.extend([model.vision_trunk, model.interactive_neck])
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True


def set_training_mode(model: SAM3FixedFullBox, scope: str) -> None:
    model.eval()
    model.prompt_encoder.train()
    model.mask_decoder.train()
    if scope == "image_decoder":
        model.vision_trunk.train()
        model.interactive_neck.train()


def trainable_parameter_summary(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return trainable, total


def model_delta_state(model: SAM3FixedFullBox, scope: str) -> dict[str, Any]:
    delta = {
        "prompt_encoder": model.prompt_encoder.state_dict(),
        "mask_decoder": model.mask_decoder.state_dict(),
    }
    if scope == "image_decoder":
        delta["vision_trunk"] = model.vision_trunk.state_dict()
        delta["interactive_neck"] = model.interactive_neck.state_dict()
    return delta


def load_model_delta(
    model: SAM3FixedFullBox, delta: dict[str, Any], scope: str
) -> None:
    required = {"prompt_encoder", "mask_decoder"}
    if scope == "image_decoder":
        required.update({"vision_trunk", "interactive_neck"})
    missing = sorted(required - delta.keys())
    if missing:
        raise RuntimeError(f"Checkpoint model delta is missing: {missing}")
    model.prompt_encoder.load_state_dict(delta["prompt_encoder"], strict=True)
    model.mask_decoder.load_state_dict(delta["mask_decoder"], strict=True)
    if scope == "image_decoder":
        model.vision_trunk.load_state_dict(delta["vision_trunk"], strict=True)
        model.interactive_neck.load_state_dict(
            delta["interactive_neck"], strict=True
        )


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid()
    dims = tuple(range(1, probabilities.ndim))
    intersection = (probabilities * target).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def hard_iou_from_logits(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
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
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    target = F.interpolate(target_native, size=logits.shape[-2:], mode="nearest")
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = soft_dice_loss(logits, target)
    true_iou = hard_iou_from_logits(logits, target)
    iou_loss = F.mse_loss(iou_predictions.flatten(), true_iou)
    total = (
        args.bce_weight * bce
        + args.dice_weight * dice
        + args.iou_weight * iou_loss
    )
    return total, {
        "loss": float(total.detach()),
        "bce": float(bce.detach()),
        "dice_loss": float(dice.detach()),
        "iou_loss": float(iou_loss.detach()),
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
            raise RuntimeError("This GPU does not support requested BF16")
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


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def evaluate(
    model: SAM3FixedFullBox,
    loader: DataLoader,
    device: torch.device,
    amp: AmpConfig,
    threshold: float,
    split: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    cases: list[dict[str, Any]] = []
    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    dice_values: list[float] = []
    iou_values: list[float] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    specificity_values: list[float] = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Evaluating {split}", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with autocast_context(amp):
                logits, _ = model(images)
            logits = F.interpolate(
                logits.float(),
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            predictions = logits.sigmoid() >= threshold
            truth = masks >= 0.5
            for index, name in enumerate(batch["name"]):
                pred = predictions[index]
                gt = truth[index]
                tp = int((pred & gt).sum().item())
                fp = int((pred & ~gt).sum().item())
                fn = int((~pred & gt).sum().item())
                tn = int((~pred & ~gt).sum().item())
                for key, value in (("tp", tp), ("fp", fp), ("fn", fn), ("tn", tn)):
                    totals[key] += value
                dice = safe_divide(2 * tp, 2 * tp + fp + fn)
                iou = safe_divide(tp, tp + fp + fn)
                precision = safe_divide(tp, tp + fp)
                recall = safe_divide(tp, tp + fn)
                specificity = safe_divide(tn, tn + fp)
                dice_values.append(dice)
                iou_values.append(iou)
                precision_values.append(precision)
                recall_values.append(recall)
                specificity_values.append(specificity)
                cases.append(
                    {
                        "split": split,
                        "filename": name,
                        "dice": dice,
                        "iou": iou,
                        "precision": precision,
                        "recall": recall,
                        "specificity": specificity,
                        "gt_pixels": tp + fn,
                        "pred_pixels": tp + fp,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "tn": tn,
                        "threshold": threshold,
                    }
                )
    if not cases:
        raise RuntimeError(f"No samples were evaluated for split '{split}'")
    tp, fp, fn, tn = (totals[key] for key in ("tp", "fp", "fn", "tn"))
    summary = {
        "whole_image_dice_mean": float(np.mean(dice_values)),
        "whole_image_iou_mean": float(np.mean(iou_values)),
        "precision_macro": float(np.mean(precision_values)),
        "recall_macro": float(np.mean(recall_values)),
        "specificity_macro": float(np.mean(specificity_values)),
        "global_dice": safe_divide(2 * tp, 2 * tp + fp + fn),
        "global_iou": safe_divide(tp, tp + fp + fn),
        "precision_micro": safe_divide(tp, tp + fp),
        "recall_micro": safe_divide(tp, tp + fn),
        "specificity_micro": safe_divide(tn, tn + fp),
        "accuracy_micro": safe_divide(tp + tn, tp + fp + fn + tn),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "num_images": float(len(cases)),
    }
    return summary, cases


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


def train_one_epoch(
    model: SAM3FixedFullBox,
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
            loss, parts = compute_loss(logits, iou_predictions, masks, args)
            scaled_loss = loss / args.grad_accum
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at step={step}: {float(loss.detach())}"
            )
        scaler.scale(scaled_loss).backward()
        should_step = (step + 1) % args.grad_accum == 0 or step + 1 == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                args.grad_clip,
                error_if_nonfinite=True,
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"Non-finite gradient norm at step={step}: {float(grad_norm)}"
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


def checkpoint_payload(
    model: SAM3FixedFullBox,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val_dice: float,
    patience_counter: int,
    split_hash: str,
    source_revision: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "format": "sam3_size512_trainable_delta_v1",
        "model_delta": model_delta_state(model, args.train_scope),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_val_dice": best_val_dice,
        "patience_counter": patience_counter,
        "split_manifest_sha256": split_hash,
        "sam3_source_revision": source_revision,
        "base_checkpoint_name": args.checkpoint.name,
        "base_checkpoint_size_bytes": args.checkpoint.stat().st_size,
        "train_scope": args.train_scope,
        "args": json_safe(vars(args)),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }


def load_resume_checkpoint(
    path: Path,
    model: SAM3FixedFullBox,
    args: argparse.Namespace,
    split_hash: str,
    source_revision: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "sam3_size512_trainable_delta_v1":
        raise RuntimeError(f"Unsupported SAM 3 checkpoint format: {path}")
    if payload.get("train_scope") != args.train_scope:
        raise RuntimeError(
            f"Resume train_scope={payload.get('train_scope')} differs from "
            f"requested {args.train_scope}"
        )
    if payload.get("split_manifest_sha256") != split_hash:
        raise RuntimeError(
            "Current train/val/test manifest differs from the resume checkpoint"
        )
    saved_size = int(payload.get("base_checkpoint_size_bytes", -1))
    if saved_size != args.checkpoint.stat().st_size:
        raise RuntimeError("The local base SAM 3 checkpoint size has changed")
    saved_revision = payload.get("sam3_source_revision", "unknown")
    if (
        saved_revision != "unknown"
        and source_revision != "unknown"
        and saved_revision != source_revision
    ):
        raise RuntimeError(
            f"SAM 3 source revision changed: {saved_revision} -> {source_revision}"
        )
    load_model_delta(model, payload["model_delta"], args.train_scope)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None:
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
    write_csv(
        path,
        history,
        (
            "epoch",
            "lr",
            "train_loss",
            "train_bce",
            "train_dice_loss",
            "train_iou_loss",
            "val_whole_image_dice_mean",
            "val_whole_image_iou_mean",
            "best_val_dice",
            "patience_counter",
            "epoch_seconds",
        ),
    )


def write_dice_bins(path: Path, cases: Sequence[dict[str, Any]]) -> None:
    rows = []
    for lower in range(0, 100, 10):
        upper = lower + 10
        if upper == 100:
            count = sum(
                lower / 100.0 <= float(row["dice"]) <= 1.0 for row in cases
            )
        else:
            count = sum(
                lower / 100.0
                <= float(row["dice"])
                < upper / 100.0
                for row in cases
            )
        rows.append(
            {
                "bin": f"{lower:02d}-{upper:03d}%",
                "lower": lower / 100.0,
                "upper": upper / 100.0,
                "count": count,
            }
        )
    write_csv(path, rows, ("bin", "lower", "upper", "count"))


def make_datasets(
    args: argparse.Namespace,
    records_by_split: dict[str, list[PairRecord]],
) -> tuple[ConsolidationDataset, ConsolidationDataset, ConsolidationDataset]:
    common = {
        "foreground_values": args.foreground_values,
        "min_mask_pixels": args.min_mask_pixels,
        "model_input_size": args.model_input_size,
    }
    train = ConsolidationDataset(
        records_by_split["train"],
        augment=not args.no_augment,
        max_samples=args.max_train_samples,
        split="train",
        **common,
    )
    val = ConsolidationDataset(
        records_by_split["val"],
        augment=False,
        max_samples=args.max_val_samples,
        split="val",
        **common,
    )
    test = ConsolidationDataset(
        records_by_split["test"],
        augment=False,
        max_samples=args.max_test_samples,
        split="test",
        **common,
    )
    return train, val, test


def print_dataset_summary(name: str, dataset: ConsolidationDataset) -> None:
    print(
        f"{name}: retained={len(dataset)}, original={dataset.total_pairs}, "
        f"excluded_below_min_pixels={dataset.excluded_pairs}"
    )


def prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def main() -> None:
    args = parse_args()
    apply_run_mode_defaults(args)
    validate_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    seed_everything(args.seed, args.deterministic)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is False")
    device = torch.device("cuda:0")
    amp = choose_amp_config(args.amp_dtype)

    args.data_root = args.data_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.sam3_source_dir = args.sam3_source_dir.expanduser().resolve()
    if args.resume is not None:
        args.resume = args.resume.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Local SAM 3 checkpoint not found: {args.checkpoint}")

    if args.output_dir is None:
        if args.resume is not None:
            output_dir = args.resume.parent
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            size_label = "224" if "224" in args.data_root.name else "512"
            output_dir = Path(f"benchmark_sam3_consolidation_size{size_label}_{timestamp}")
    else:
        output_dir = args.output_dir.expanduser()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"AMP: {amp.name}")
    print(f"Data root: {args.data_root}")
    print(f"SAM 3 source: {args.sam3_source_dir}")
    print(f"Base checkpoint: {args.checkpoint}")
    print(f"Output: {output_dir}")
    print("Prompt: fixed full-image box; no GT-derived location information")

    records_by_split = {
        split: resolve_pairs(args.data_root / split)
        for split in ("train", "val", "test")
    }
    split_hash = audit_and_write_split_manifest(
        args.data_root, records_by_split, output_dir
    )
    train_dataset, val_dataset, test_dataset = make_datasets(
        args, records_by_split
    )
    print_dataset_summary("train", train_dataset)
    print_dataset_summary("val", val_dataset)
    print_dataset_summary("test", test_dataset)
    print(f"Split manifest SHA256: {split_hash}")

    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.num_workers, args.seed
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, False, args.num_workers, args.seed + 1
    )
    test_loader = make_loader(
        test_dataset, args.batch_size, False, args.num_workers, args.seed + 2
    )

    model, source_revision = build_sam3_subset(
        args.sam3_source_dir,
        args.checkpoint,
        device,
        args.model_input_size,
    )
    configure_trainable_parameters(model, args.train_scope)
    trainable, total = trainable_parameter_summary(model)
    print(f"SAM 3 source revision: {source_revision}")
    if source_revision != SAM3_SOURCE_REFERENCE:
        message = (
            f"SAM3 source revision {source_revision!r} differs from frozen "
            f"benchmark commit {SAM3_SOURCE_REFERENCE}."
        )
        if not args.allow_source_mismatch:
            raise RuntimeError(message + " Use --allow-source-mismatch only for a declared non-reference run.")
        print("WARNING: " + message)
    print(
        f"Parameters resident in trainer: trainable={trainable / 1e6:.2f} M / "
        f"total={total / 1e6:.2f} M ({100.0 * trainable / total:.2f}%)"
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
        lr_lambda=lambda epoch: lr_factor(
            epoch, args.warmup_epochs, args.epochs
        ),
    )
    scaler = make_grad_scaler(amp)

    start_epoch = 0
    best_val_dice = -math.inf
    patience_counter = 0
    if args.resume is not None:
        payload = load_resume_checkpoint(
            args.resume,
            model,
            args,
            split_hash,
            source_revision,
            None if args.eval_only else optimizer,
            None if args.eval_only else scheduler,
            None if args.eval_only else scaler,
        )
        start_epoch = int(payload["epoch"]) + 1
        best_val_dice = float(payload["best_val_dice"])
        patience_counter = int(payload["patience_counter"])
        print(
            f"Loaded {args.resume}: next_epoch={start_epoch + 1}, "
            f"best_val_dice={best_val_dice:.6f}"
        )

    if args.eval_only:
        val_metrics, val_cases = evaluate(
            model, val_loader, device, amp, args.threshold, "val"
        )
        test_metrics, test_cases = evaluate(
            model, test_loader, device, amp, args.threshold, "test"
        )
        write_csv(output_dir / "val_cases.csv", val_cases, tuple(val_cases[0]))
        write_csv(output_dir / "test_cases.csv", test_cases, tuple(test_cases[0]))
        write_dice_bins(output_dir / "test_dice_bins.csv", test_cases)
        result = {
            "model": "sam3_standard_interactive",
            "mode": "eval_only",
            **prefixed_metrics("val", val_metrics),
            **prefixed_metrics("test", test_metrics),
            "threshold": args.threshold,
            "checkpoint": str(args.resume),
            "split_manifest_sha256": split_hash,
        }
        (output_dir / "eval_result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"Validation Mean Dice: {val_metrics['whole_image_dice_mean']:.6f}"
        )
        print(f"Test Mean Dice: {test_metrics['whole_image_dice_mean']:.6f}")
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
        val_metrics, _ = evaluate(
            model, val_loader, device, amp, args.threshold, "val"
        )
        val_dice = val_metrics["whole_image_dice_mean"]
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
            split_hash,
            source_revision,
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
                "val_whole_image_iou_mean": val_metrics[
                    "whole_image_iou_mean"
                ],
                "best_val_dice": best_val_dice,
                "patience_counter": patience_counter,
                "epoch_seconds": epoch_seconds,
            }
        )
        write_history(history_path, history)
        print(
            f"train_loss={train_metrics['loss']:.6f} | "
            f"val_mean_dice={val_dice:.6f} | best={best_val_dice:.6f} | "
            f"patience={patience_counter}/{args.patience} | "
            f"time={epoch_seconds / 60.0:.1f} min"
        )
        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    if not best_path.is_file():
        raise RuntimeError("Training ended without best_model.pth")
    best_payload = load_resume_checkpoint(
        best_path, model, args, split_hash, source_revision
    )
    best_epoch = int(best_payload["epoch"]) + 1
    val_metrics, val_cases = evaluate(
        model, val_loader, device, amp, args.threshold, "val"
    )
    test_metrics, test_cases = evaluate(
        model, test_loader, device, amp, args.threshold, "test"
    )
    write_csv(output_dir / "val_cases.csv", val_cases, tuple(val_cases[0]))
    write_csv(output_dir / "test_cases.csv", test_cases, tuple(test_cases[0]))
    write_dice_bins(output_dir / "test_dice_bins.csv", test_cases)

    summary = {
        "model": "sam3_standard_interactive",
        "train_scope": args.train_scope,
        "prompt_protocol": "fixed_full_image_box_no_gt",
        "best_epoch": best_epoch,
        **prefixed_metrics("val", val_metrics),
        **prefixed_metrics("test", test_metrics),
        "threshold": args.threshold,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "physical_batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "amp_dtype": amp.name,
        "split_manifest_sha256": split_hash,
        "sam3_source_revision": source_revision,
    }
    write_csv(
        output_dir / "benchmark_summary.csv", [summary], tuple(summary.keys())
    )
    result = {
        **summary,
        "best_model": str(best_path),
        "last_model": str(last_path),
        "base_checkpoint": str(args.checkpoint),
        "training_seconds": time.time() - training_started,
        "args": json_safe(vars(args)),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(
        f"Validation Mean Dice: {val_metrics['whole_image_dice_mean']:.6f}"
    )
    print(f"Test Mean Dice: {test_metrics['whole_image_dice_mean']:.6f}")
    print(f"Test Mean IoU: {test_metrics['whole_image_iou_mean']:.6f}")
    print(f"Test Precision micro: {test_metrics['precision_micro']:.6f}")
    print(f"Test Recall micro: {test_metrics['recall_micro']:.6f}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
