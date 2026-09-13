#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate MONAI UNet, Attention U-Net, and VNet checkpoints on Size_512.

This script is intended for the final lung-ultrasound consolidation benchmark.
It:
  * strictly reloads the original MONAI model weights;
  * infers the supported one-channel or legacy RGB model variant;
  * restores checkpoint normalization metadata when available;
  * excludes test masks with fewer than 30 foreground pixels by default;
  * uses a fixed probability threshold (default: 0.5);
  * reports per-image and pooled pixel-level segmentation metrics.

Expected dataset layout:

    Size_512/
        test/
            images/p001_001_0001.png
            masks/p001_001_0001_mask.png

Prepared patch masks are binary: 0=background, 1=consolidation.

Typical use:

    python size512_monai_checkpoint_evaluator.py \
      --data-root ./datasets/Size_512 \
      --checkpoint-dir /path/to/the/three/best/checkpoints

Explicit checkpoint paths can be supplied with --unet, --attention-unet,
and --vnet. Explicit paths take precedence over discovery in --checkpoint-dir.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import platform
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import cv2
    import monai
    import numpy as np
    import torch
    import torch.nn as nn
    from monai.networks.nets import AttentionUnet, UNet, VNet
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit(
        "Missing evaluation dependency. Activate the environment used to train "
        "the MONAI baselines and ensure torch, monai, opencv-python-headless, "
        "numpy, and tqdm are installed.\n"
        f"Original import error: {exc}"
    ) from exc


MODEL_ORDER = ("monai_unet", "monai_attention_unet", "monai_vnet")
DISPLAY_NAMES = {
    "monai_unet": "UNet",
    "monai_attention_unet": "Attention U-Net",
    "monai_vnet": "VNet",
}

DEFAULT_GRAY_MEAN = 0.10063751267950466
DEFAULT_GRAY_STD = 0.14586819260714984
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TestRecord:
    image_path: Path
    mask_path: Path
    sample_id: str
    gt_pixels: int
    height: int
    width: int


@dataclass(frozen=True)
class PreprocessConfig:
    mode: str
    input_channels: int
    gray_mean: Optional[float]
    gray_std: Optional[float]
    source: str
    warning: Optional[str]


class RGBToVNet2D(nn.Module):
    """Legacy three-channel VNet wrapper used by the earlier benchmark."""

    def __init__(self) -> None:
        super().__init__()
        self.rgb_to_4 = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.vnet = make_vnet(in_channels=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vnet(self.rgb_to_4(x))


def make_vnet(in_channels: int) -> nn.Module:
    """Build VNet compatibly across the MONAI versions used by the project."""
    try:
        return VNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=1,
            dropout_prob_down=0.2,
            dropout_prob_up=(0.2, 0.2),
            dropout_dim=2,
        )
    except TypeError:
        return VNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=1,
            dropout_prob=0.2,
            dropout_dim=2,
        )


def build_model(model_name: str, variant: str) -> Tuple[nn.Module, int]:
    if model_name == "monai_unet":
        input_channels = int(variant)
        return (
            UNet(
                spatial_dims=2,
                in_channels=input_channels,
                out_channels=1,
                channels=(32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            ),
            input_channels,
        )

    if model_name == "monai_attention_unet":
        input_channels = int(variant)
        return (
            AttentionUnet(
                spatial_dims=2,
                in_channels=input_channels,
                out_channels=1,
                channels=(32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2),
            ),
            input_channels,
        )

    if model_name == "monai_vnet":
        if variant == "1":
            return make_vnet(in_channels=1), 1
        if variant == "rgb_adapter":
            return RGBToVNet2D(), 3

    raise ValueError(f"Unsupported model/variant: {model_name}/{variant}")


def model_variants(model_name: str, metadata: Mapping[str, Any]) -> List[str]:
    if model_name == "monai_vnet":
        variants = ["1", "rgb_adapter"]
    else:
        variants = ["1", "3"]

    metadata_channels = to_optional_int(metadata.get("input_channels"))
    if metadata_channels in (1, 3):
        preferred = "rgb_adapter" if model_name == "monai_vnet" and metadata_channels == 3 else str(metadata_channels)
        if preferred in variants:
            variants.remove(preferred)
            variants.insert(0, preferred)
    return variants


def torch_load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def is_tensor_state_dict(obj: Any) -> bool:
    return (
        isinstance(obj, Mapping)
        and bool(obj)
        and all(isinstance(k, str) for k in obj.keys())
        and all(torch.is_tensor(v) or isinstance(v, nn.Parameter) for v in obj.values())
    )


def extract_state_dict_and_metadata(payload: Any) -> Tuple[Mapping[str, torch.Tensor], Dict[str, Any], str]:
    if isinstance(payload, nn.Module):
        return payload.state_dict(), {}, "serialized_module"

    if is_tensor_state_dict(payload):
        return payload, {}, "raw_state_dict"

    if not isinstance(payload, Mapping):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload).__name__}")

    state_keys = (
        "state_dict",
        "model_state",
        "model_state_dict",
        "network_state_dict",
        "weights",
        "model",
        "net",
    )
    for key in state_keys:
        candidate = payload.get(key)
        if isinstance(candidate, nn.Module):
            candidate = candidate.state_dict()
        if is_tensor_state_dict(candidate):
            metadata = {
                str(k): json_safe_scalar_or_container(v)
                for k, v in payload.items()
                if k != key and not is_tensor_state_dict(v)
            }
            return candidate, metadata, key

    raise KeyError(
        "No model state dictionary found. Supported keys: "
        + ", ".join(state_keys)
    )


def json_safe_scalar_or_container(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(k): json_safe_scalar_or_container(v)
            for k, v in value.items()
            if not torch.is_tensor(v)
        }
    if isinstance(value, (list, tuple)):
        return [json_safe_scalar_or_container(v) for v in value if not torch.is_tensor(v)]
    return repr(value)


def state_dict_candidates(
    state_dict: Mapping[str, torch.Tensor],
) -> Iterable[Tuple[str, Dict[str, torch.Tensor]]]:
    """Yield safe prefix variants; strict loading decides which one is valid."""
    original = dict(state_dict)
    yielded_signatures = set()

    queue: List[Tuple[str, Dict[str, torch.Tensor]]] = [("unchanged", original)]
    prefixes = ("module.", "_orig_mod.", "model.", "net.")

    while queue:
        label, candidate = queue.pop(0)
        signature = tuple(candidate.keys())
        if signature in yielded_signatures:
            continue
        yielded_signatures.add(signature)
        yield label, candidate

        for prefix in prefixes:
            if candidate and all(key.startswith(prefix) for key in candidate):
                stripped = {key[len(prefix):]: value for key, value in candidate.items()}
                queue.append((f"{label}|strip:{prefix}", stripped))


def validate_checkpoint_model_name(model_name: str, metadata: Mapping[str, Any], path: Path) -> None:
    saved = metadata.get("model_name")
    if not isinstance(saved, str):
        return
    normalized = saved.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "monai_unet": {"unet", "monai_unet"},
        "monai_attention_unet": {
            "attention_unet",
            "attentionunet",
            "attention_u_net",
            "monai_attention_unet",
        },
        "monai_vnet": {"vnet", "monai_vnet"},
    }
    if normalized not in aliases[model_name]:
        raise ValueError(
            f"Checkpoint {path} says model_name={saved!r}, but it was assigned "
            f"to {DISPLAY_NAMES[model_name]}."
        )


def load_model_strict(
    model_name: str,
    checkpoint_path: Path,
) -> Tuple[nn.Module, int, Dict[str, Any], Dict[str, Any]]:
    payload = torch_load_checkpoint(checkpoint_path)
    state_dict, metadata, payload_key = extract_state_dict_and_metadata(payload)
    validate_checkpoint_model_name(model_name, metadata, checkpoint_path)

    attempts: List[str] = []
    for variant in model_variants(model_name, metadata):
        for prefix_label, candidate_state in state_dict_candidates(state_dict):
            model, input_channels = build_model(model_name, variant)
            try:
                model.load_state_dict(candidate_state, strict=True)
            except RuntimeError as exc:
                first_line = str(exc).splitlines()[0]
                attempts.append(f"variant={variant}, {prefix_label}: {first_line}")
                del model
                continue

            load_info = {
                "payload_key": payload_key,
                "model_variant": variant,
                "input_channels": input_channels,
                "state_prefix_strategy": prefix_label,
                "strict": True,
                "num_state_tensors": len(candidate_state),
            }
            return model, input_channels, metadata, load_info

    preview = "\n  - ".join(attempts[:8])
    raise RuntimeError(
        f"Strict checkpoint loading failed for {checkpoint_path}.\n"
        "This normally means that the weight was trained with a different "
        "model definition or MONAI version. No partial load was performed.\n"
        f"Attempts:\n  - {preview}"
    )


def to_optional_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def to_optional_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def checkpoint_metric(
    metadata: Mapping[str, Any],
    direct_keys: Sequence[str],
    nested_keys: Sequence[str],
) -> Optional[float]:
    for key in direct_keys:
        value = to_optional_float(metadata.get(key))
        if value is not None:
            return value
    val_metrics = metadata.get("val_metrics")
    if isinstance(val_metrics, Mapping):
        for key in nested_keys:
            value = to_optional_float(val_metrics.get(key))
            if value is not None:
                return value
    return None


def resolve_preprocessing(
    args: argparse.Namespace,
    input_channels: int,
    metadata: Mapping[str, Any],
) -> PreprocessConfig:
    requested = args.normalization
    saved_mode = str(metadata.get("normalization_mode", "")).strip().lower()
    saved_mean = to_optional_float(
        metadata.get("gray_mean", metadata.get("image_mean"))
    )
    saved_std = to_optional_float(
        metadata.get("gray_std", metadata.get("image_std"))
    )

    warning: Optional[str] = None
    source = "command_line"

    if requested == "auto":
        if saved_mode == "none":
            mode = "none"
            source = "checkpoint"
        elif saved_mean is not None and saved_std is not None and saved_std > 0:
            mode = "fixed-gray"
            source = f"checkpoint:{saved_mode or 'stored_stats'}"
        elif input_channels == 3:
            mode = "imagenet"
            source = "legacy_rgb_fallback"
            warning = (
                "Checkpoint has no usable normalization metadata. The strictly "
                "matched three-channel legacy MONAI model therefore uses the "
                "ImageNet normalization from the original RGB training script."
            )
        else:
            mode = "fixed-gray"
            source = "project_fixed_gray_fallback"
            warning = (
                "Checkpoint has no usable normalization metadata. Falling back "
                f"to the project Size_512 grayscale statistics: mean={DEFAULT_GRAY_MEAN:.12f}, "
                f"std={DEFAULT_GRAY_STD:.12f}. Use --normalization/--gray-mean/"
                "--gray-std if this weight came from a different run."
            )
    else:
        mode = requested

    if mode == "imagenet" and input_channels != 3:
        raise ValueError("ImageNet normalization requires a three-channel model.")

    gray_mean: Optional[float] = None
    gray_std: Optional[float] = None
    if mode == "fixed-gray":
        gray_mean = (
            args.gray_mean
            if args.gray_mean is not None
            else saved_mean
            if saved_mean is not None
            else DEFAULT_GRAY_MEAN
        )
        gray_std = (
            args.gray_std
            if args.gray_std is not None
            else saved_std
            if saved_std is not None
            else DEFAULT_GRAY_STD
        )
        if not math.isfinite(gray_mean) or not math.isfinite(gray_std) or gray_std <= 0:
            raise ValueError(f"Invalid grayscale normalization: mean={gray_mean}, std={gray_std}")

    return PreprocessConfig(
        mode=mode,
        input_channels=input_channels,
        gray_mean=gray_mean,
        gray_std=gray_std,
        source=source,
        warning=warning,
    )


def parse_foreground_values(text: str) -> Tuple[int, ...]:
    try:
        values = tuple(sorted({int(part.strip()) for part in text.split(",") if part.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--foreground-values must be comma-separated integers, for example: 1"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("--foreground-values cannot be empty")
    return values


def find_mask_path(mask_dir: Path, image_path: Path) -> Path:
    candidates = (
        mask_dir / f"{image_path.stem}_mask.png",
        mask_dir / image_path.name,
    )
    existing = [path for path in candidates if path.is_file()]
    if len(existing) == 1:
        return existing[0]
    if not existing:
        raise FileNotFoundError(
            f"No mask found for {image_path.name}. Tried: "
            + ", ".join(str(path) for path in candidates)
        )
    raise RuntimeError(
        f"Ambiguous masks for {image_path.name}: "
        + ", ".join(str(path) for path in existing)
    )


def read_gray(path: Path, flag: int) -> np.ndarray:
    array = cv2.imread(str(path), flag)
    if array is None:
        raise RuntimeError(f"OpenCV failed to read: {path}")
    return array


def scan_test_records(
    data_root: Path,
    expected_size: int,
    foreground_values: Sequence[int],
    require_positive: bool,
    minimum_gt_pixels: int,
    max_samples: Optional[int],
) -> Tuple[List[TestRecord], Dict[str, Any]]:
    image_dir = data_root / "test" / "images"
    mask_dir = data_root / "test" / "masks"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Test image directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Test mask directory not found: {mask_dir}")

    image_paths = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"
    )
    if not image_paths:
        raise RuntimeError(f"No PNG test images found in {image_dir}")

    records: List[TestRecord] = []
    excluded_records: List[Dict[str, Any]] = []
    empty_ids_before_filter: List[str] = []
    unexpected_patch_ids: List[str] = []
    mask_unique_values = set()
    gt_pixels_total_before_filter = 0
    gt_pixels_total = 0

    for image_path in tqdm(image_paths, desc="Validate test pairs", leave=False):
        mask_path = find_mask_path(mask_dir, image_path)
        image = read_gray(image_path, cv2.IMREAD_GRAYSCALE)
        mask = read_gray(mask_path, cv2.IMREAD_GRAYSCALE)
        if image.shape != (expected_size, expected_size):
            raise ValueError(
                f"Unexpected image shape {image.shape} for {image_path}; "
                f"expected {(expected_size, expected_size)}. No hidden resize is performed."
            )
        if mask.shape != (expected_size, expected_size):
            raise ValueError(
                f"Unexpected mask shape {mask.shape} for {mask_path}; "
                f"expected {(expected_size, expected_size)}."
            )

        unique = np.unique(mask)
        mask_unique_values.update(int(v) for v in unique)
        binary = np.isin(mask, foreground_values)
        gt_pixels = int(binary.sum())
        sample_id = image_path.stem

        if gt_pixels == 0:
            empty_ids_before_filter.append(sample_id)
        if not sample_id.endswith("_0001"):
            unexpected_patch_ids.append(sample_id)

        gt_pixels_total_before_filter += gt_pixels
        if gt_pixels < minimum_gt_pixels:
            excluded_records.append(
                {
                    "sample_id": sample_id,
                    "gt_pixels": gt_pixels,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "reason": f"gt_pixels < {minimum_gt_pixels}",
                }
            )
            continue

        records.append(
            TestRecord(
                image_path=image_path,
                mask_path=mask_path,
                sample_id=sample_id,
                gt_pixels=gt_pixels,
                height=expected_size,
                width=expected_size,
            )
        )
        gt_pixels_total += gt_pixels

    eligible_before_max = len(records)
    if max_samples is not None:
        records = records[:max_samples]
        gt_pixels_total = sum(record.gt_pixels for record in records)

    if not records:
        raise RuntimeError(
            "No test images remain after applying "
            f"--min-gt-pixels {minimum_gt_pixels}."
        )

    retained_empty_ids = [record.sample_id for record in records if record.gt_pixels == 0]
    if require_positive and retained_empty_ids:
        raise ValueError(
            f"Size_512 is expected to be positive-only, but {len(retained_empty_ids)} empty-GT "
            f"test images remain after filtering. First examples: {retained_empty_ids[:10]}. "
            "Use --no-require-positive-test only when intentionally evaluating a mixed test set."
        )
    allowed_mask_values = {0, *foreground_values}
    unexpected_mask_values = sorted(mask_unique_values - allowed_mask_values)
    if unexpected_mask_values:
        raise ValueError(
            "Prepared Size_512 masks must already be binary, but unexpected label "
            f"value(s) {unexpected_mask_values} were found. Observed values: "
            f"{sorted(mask_unique_values)}. Do not evaluate the original three-class "
            "moe_bcdata masks with this script."
        )

    stats = {
        "num_images_found": len(image_paths),
        "num_images_eligible_before_max_samples": eligible_before_max,
        "num_images": len(records),
        "num_positive_gt": len(records) - len(retained_empty_ids),
        "num_empty_gt": len(retained_empty_ids),
        "num_empty_gt_before_filter": len(empty_ids_before_filter),
        "min_gt_pixels": minimum_gt_pixels,
        "num_excluded_below_min_gt_pixels": len(excluded_records),
        "num_excluded_small_positive_gt": sum(
            1 for item in excluded_records if 0 < int(item["gt_pixels"]) < minimum_gt_pixels
        ),
        "excluded_below_min_gt_pixels": excluded_records,
        "num_truncated_by_max_samples": eligible_before_max - len(records),
        "gt_pixels_total": gt_pixels_total,
        "gt_pixels_total_before_filter": gt_pixels_total_before_filter,
        "mask_unique_values": sorted(mask_unique_values),
        "unexpected_patch_id_count": len(unexpected_patch_ids),
        "max_samples": max_samples,
    }
    return records, stats


class Size512TestDataset(Dataset):
    def __init__(
        self,
        records: Sequence[TestRecord],
        preprocessing: PreprocessConfig,
        foreground_values: Sequence[int],
    ) -> None:
        self.records = list(records)
        self.preprocessing = preprocessing
        self.foreground_values = tuple(foreground_values)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[index]
        mask = read_gray(record.mask_path, cv2.IMREAD_GRAYSCALE)

        if self.preprocessing.input_channels == 1:
            image = read_gray(record.image_path, cv2.IMREAD_GRAYSCALE)
            image_t = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        elif self.preprocessing.input_channels == 3:
            image_bgr = read_gray(record.image_path, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_t = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        else:
            raise ValueError(
                f"Unsupported input channel count: {self.preprocessing.input_channels}"
            )

        if self.preprocessing.mode == "fixed-gray":
            assert self.preprocessing.gray_mean is not None
            assert self.preprocessing.gray_std is not None
            image_t = (image_t - self.preprocessing.gray_mean) / self.preprocessing.gray_std
        elif self.preprocessing.mode == "imagenet":
            mean = torch.tensor(IMAGENET_MEAN, dtype=image_t.dtype).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD, dtype=image_t.dtype).view(3, 1, 1)
            image_t = (image_t - mean) / std
        elif self.preprocessing.mode != "none":
            raise ValueError(f"Unknown preprocessing mode: {self.preprocessing.mode}")

        mask_binary = np.isin(mask, self.foreground_values).astype(np.uint8)
        mask_t = torch.from_numpy(mask_binary).unsqueeze(0)
        return image_t, mask_t, record.sample_id


def safe_div(numerator: float, denominator: float, zero_value: float = 0.0) -> float:
    return float(numerator / denominator) if denominator > 0 else float(zero_value)


def confusion_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    pred_positive = tp + fp
    gt_positive = tp + fn
    gt_negative = tn + fp
    total = tp + fp + fn + tn

    dice = safe_div(2 * tp, 2 * tp + fp + fn, zero_value=1.0)
    iou = safe_div(tp, tp + fp + fn, zero_value=1.0)
    precision = safe_div(tp, pred_positive, zero_value=0.0)
    recall = safe_div(tp, gt_positive, zero_value=0.0)
    specificity = safe_div(tn, gt_negative, zero_value=0.0)
    npv = safe_div(tn, tn + fn, zero_value=0.0)
    accuracy = safe_div(tp + tn, total, zero_value=0.0)
    balanced_accuracy = 0.5 * (recall + specificity)
    f2 = safe_div(5 * precision * recall, 4 * precision + recall, zero_value=0.0)

    mcc_denominator = math.sqrt(
        float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    )
    mcc = safe_div(tp * tn - fp * fn, mcc_denominator, zero_value=0.0)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "npv": npv,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f2": f2,
        "mcc": mcc,
    }


def extract_logits(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)) and output and torch.is_tensor(output[0]):
        return output[0]
    if isinstance(output, Mapping):
        for key in ("logits", "out", "pred", "prediction", "mask_logits"):
            value = output.get(key)
            if torch.is_tensor(value):
                return value
    raise TypeError(f"Unsupported model output type: {type(output).__name__}")


def resolve_amp_dtype(device: torch.device, requested: str) -> Tuple[Optional[torch.dtype], str]:
    if device.type != "cuda" or requested == "float32":
        return None, "float32"
    if requested == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested, but the current CUDA device does not support it.")
        return torch.bfloat16, "bfloat16"
    if requested == "float16":
        return torch.float16, "float16"
    if requested == "auto":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bfloat16"
        return torch.float16, "float16"
    raise ValueError(f"Unknown AMP dtype: {requested}")


def autocast_context(device: torch.device, dtype: Optional[torch.dtype]):
    if device.type == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def evaluate_one_model(
    model_name: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    threshold: float,
    checkpoint_path: Path,
    metadata: Mapping[str, Any],
    preprocessing: PreprocessConfig,
    load_info: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    model = model.to(device)
    model.eval()
    per_image_rows: List[Dict[str, Any]] = []
    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    with torch.inference_mode():
        for images, masks, sample_ids in tqdm(
            loader,
            desc=f"Test {DISPLAY_NAMES[model_name]}",
            leave=True,
        ):
            images = images.to(device, non_blocking=True)
            with autocast_context(device, amp_dtype):
                logits = extract_logits(model(images))
            if logits.ndim == 3:
                logits = logits.unsqueeze(1)
            if logits.ndim != 4 or logits.shape[1] != 1:
                raise ValueError(
                    f"{DISPLAY_NAMES[model_name]} returned shape {tuple(logits.shape)}; "
                    "expected [B, 1, H, W]."
                )
            if logits.shape[-2:] != masks.shape[-2:]:
                raise ValueError(
                    f"Prediction shape {tuple(logits.shape[-2:])} differs from GT "
                    f"shape {tuple(masks.shape[-2:])}; no hidden interpolation is performed."
                )

            probabilities = torch.sigmoid(logits).float().cpu()
            predictions = probabilities >= threshold
            targets = masks.bool()

            for batch_index, sample_id in enumerate(sample_ids):
                pred = predictions[batch_index, 0].numpy()
                gt = targets[batch_index, 0].numpy()
                tp = int(np.logical_and(pred, gt).sum())
                fp = int(np.logical_and(pred, ~gt).sum())
                fn = int(np.logical_and(~pred, gt).sum())
                tn = int(np.logical_and(~pred, ~gt).sum())
                metrics = confusion_metrics(tp, fp, fn, tn)
                totals["tp"] += tp
                totals["fp"] += fp
                totals["fn"] += fn
                totals["tn"] += tn

                row: Dict[str, Any] = {
                    "model": model_name,
                    "display_name": DISPLAY_NAMES[model_name],
                    "sample_id": sample_id,
                    "threshold": threshold,
                    "gt_pixels": tp + fn,
                    "pred_pixels": tp + fp,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                }
                row.update(metrics)
                per_image_rows.append(row)

    if not per_image_rows:
        raise RuntimeError("No test predictions were produced.")

    micro = confusion_metrics(**totals)
    positive_rows = [row for row in per_image_rows if row["gt_pixels"] > 0]
    empty_rows = [row for row in per_image_rows if row["gt_pixels"] == 0]
    dice_values = np.asarray([row["dice"] for row in per_image_rows], dtype=np.float64)

    summary: Dict[str, Any] = {
        "model": model_name,
        "display_name": DISPLAY_NAMES[model_name],
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": metadata.get("epoch"),
        "checkpoint_val_dice_positive": checkpoint_metric(
            metadata,
            direct_keys=("best_val_dice_positive", "best_val_dice_pos", "val_dice_positive"),
            nested_keys=("dice_positive", "dice_pos"),
        ),
        "checkpoint_val_dice_all": checkpoint_metric(
            metadata,
            direct_keys=("best_val_dice_all", "val_dice_all"),
            nested_keys=("dice_all",),
        ),
        "threshold": threshold,
        "input_channels": preprocessing.input_channels,
        "normalization": preprocessing.mode,
        "normalization_source": preprocessing.source,
        "gray_mean": preprocessing.gray_mean,
        "gray_std": preprocessing.gray_std,
        "num_images": len(per_image_rows),
        "num_positive_gt": len(positive_rows),
        "num_empty_gt": len(empty_rows),
        "gt_pixels_total": totals["tp"] + totals["fn"],
        "pred_pixels_total": totals["tp"] + totals["fp"],
        "dice_mean_all": float(dice_values.mean()),
        "dice_mean_positive": (
            float(np.mean([row["dice"] for row in positive_rows]))
            if positive_rows
            else None
        ),
        "dice_median": float(np.median(dice_values)),
        "dice_q25": float(np.percentile(dice_values, 25)),
        "dice_q75": float(np.percentile(dice_values, 75)),
        "iou_macro": float(np.mean([row["iou"] for row in per_image_rows])),
        "precision_macro": float(np.mean([row["precision"] for row in per_image_rows])),
        "recall_macro": float(np.mean([row["recall"] for row in per_image_rows])),
        "specificity_macro": float(np.mean([row["specificity"] for row in per_image_rows])),
        "accuracy_macro": float(np.mean([row["accuracy"] for row in per_image_rows])),
        "balanced_accuracy_macro": float(
            np.mean([row["balanced_accuracy"] for row in per_image_rows])
        ),
        "empty_gt_false_positive_rate": (
            float(np.mean([row["pred_pixels"] > 0 for row in empty_rows]))
            if empty_rows
            else None
        ),
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
        "tn": totals["tn"],
        "dice_micro": micro["dice"],
        "iou_micro": micro["iou"],
        "precision_micro": micro["precision"],
        "recall_micro": micro["recall"],
        "specificity_micro": micro["specificity"],
        "npv_micro": micro["npv"],
        "accuracy_micro": micro["accuracy"],
        "balanced_accuracy_micro": micro["balanced_accuracy"],
        "f2_micro": micro["f2"],
        "mcc_micro": micro["mcc"],
        "load_info": dict(load_info),
        "preprocessing_warning": preprocessing.warning,
    }
    return per_image_rows, summary


def discover_checkpoints(args: argparse.Namespace) -> Dict[str, Path]:
    explicit = {
        "monai_unet": args.unet,
        "monai_attention_unet": args.attention_unet,
        "monai_vnet": args.vnet,
    }
    result: Dict[str, Path] = {}
    for model_name, path in explicit.items():
        if path is not None:
            resolved = path.expanduser().resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {resolved}")
            result[model_name] = resolved

    missing = [name for name in MODEL_ORDER if name not in result]
    if not missing:
        return result
    if args.checkpoint_dir is None:
        raise ValueError(
            "Checkpoint paths are incomplete. Supply --checkpoint-dir or all of "
            "--unet, --attention-unet, and --vnet."
        )

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    files = sorted(
        path
        for path in checkpoint_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".pth", ".pt"}
    )

    def belongs(path: Path, model_name: str) -> bool:
        name = path.name.lower().replace("-", "_")
        if model_name == "monai_attention_unet":
            return "attention" in name and "unet" in name
        if model_name == "monai_vnet":
            return "vnet" in name
        return "unet" in name and "attention" not in name and "vnet" not in name

    for model_name in missing:
        matches = [path for path in files if belongs(path, model_name)]
        if len(matches) != 1:
            match_text = ", ".join(path.name for path in matches) if matches else "none"
            raise RuntimeError(
                f"Expected exactly one {DISPLAY_NAMES[model_name]} checkpoint in "
                f"{checkpoint_dir}, found {len(matches)}: {match_text}. "
                "Pass its exact path explicitly."
            )
        result[model_name] = matches[0]
    return result


def resolve_device(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {text!r} requested, but CUDA is unavailable.")
    return device


def ensure_new_output_dir(path: Optional[Path]) -> Path:
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(f"size512_monai_checkpoint_eval_{stamp}")
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {path}. "
            "Choose a new --output-dir; existing benchmark results are not overwritten."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8f}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def sanitize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return sanitize_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


SUMMARY_FIELDS = (
    "model",
    "display_name",
    "checkpoint",
    "checkpoint_epoch",
    "checkpoint_val_dice_positive",
    "checkpoint_val_dice_all",
    "threshold",
    "input_channels",
    "normalization",
    "normalization_source",
    "gray_mean",
    "gray_std",
    "num_images",
    "num_positive_gt",
    "num_empty_gt",
    "gt_pixels_total",
    "pred_pixels_total",
    "dice_mean_all",
    "dice_mean_positive",
    "dice_median",
    "dice_q25",
    "dice_q75",
    "dice_micro",
    "iou_macro",
    "iou_micro",
    "precision_macro",
    "precision_micro",
    "recall_macro",
    "recall_micro",
    "specificity_macro",
    "specificity_micro",
    "accuracy_macro",
    "accuracy_micro",
    "balanced_accuracy_macro",
    "balanced_accuracy_micro",
    "npv_micro",
    "f2_micro",
    "mcc_micro",
    "empty_gt_false_positive_rate",
    "tp",
    "fp",
    "fn",
    "tn",
    "preprocessing_warning",
    "load_info",
)

PER_IMAGE_FIELDS = (
    "model",
    "display_name",
    "sample_id",
    "threshold",
    "gt_pixels",
    "pred_pixels",
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
    "npv",
    "accuracy",
    "balanced_accuracy",
    "f2",
    "mcc",
    "tp",
    "fp",
    "fn",
    "tn",
)


def format_optional(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown_report(
    path: Path,
    summaries: Sequence[Mapping[str, Any]],
    dataset_stats: Mapping[str, Any],
    args: argparse.Namespace,
    amp_name: str,
) -> None:
    lines = [
        "# Size_512 MONAI checkpoint test report",
        "",
        f"- Dataset: `{args.data_root.expanduser().resolve()}`",
        f"- Test images found: {dataset_stats['num_images_found']}",
        f"- Minimum included GT pixels: {dataset_stats['min_gt_pixels']}",
        f"- Images excluded below threshold: {dataset_stats['num_excluded_below_min_gt_pixels']}",
        f"- Images evaluated: {dataset_stats['num_images']}",
        f"- Positive-GT images: {dataset_stats['num_positive_gt']}",
        f"- Empty-GT images: {dataset_stats['num_empty_gt']}",
        f"- Foreground mask value(s): {list(args.foreground_values)}",
        f"- Probability threshold: {args.threshold}",
        f"- Inference precision: {amp_name}",
        "",
        "## Main results",
        "",
        "| Model | Dice mean | Dice median | Dice micro | IoU micro | Precision micro | Recall micro | Specificity micro |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {display_name} | {dice_mean_all} | {dice_median} | {dice_micro} | "
            "{iou_micro} | {precision_micro} | {recall_micro} | {specificity_micro} |".format(
                display_name=row["display_name"],
                **{key: format_optional(row.get(key)) for key in (
                    "dice_mean_all",
                    "dice_median",
                    "dice_micro",
                    "iou_micro",
                    "precision_micro",
                    "recall_micro",
                    "specificity_micro",
                )},
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `dice_mean_all` is the arithmetic mean of per-image Dice and is the main Size_512 reporting metric.",
            "- `dice_mean_positive` uses only images with non-empty GT. It should equal `dice_mean_all` for the final positive-only Size_512 test set.",
            "- `micro` metrics pool TP/FP/FN/TN over all test pixels; this matches the earlier benchmark's pooled precision/recall calculation.",
            "- `macro` metrics average the corresponding per-image values.",
            "- Accuracy and specificity can appear high because background pixels dominate; interpret them together with Dice, IoU, precision, and recall.",
            "- Empty/empty Dice and IoU are defined as 1. Undefined precision/recall divisions are defined as 0. The final positive-only test set does not use the empty-GT convention.",
            "",
            "## Test-set filtering",
            "",
            f"- Samples with fewer than {dataset_stats['min_gt_pixels']} foreground pixels were excluded before inference.",
            f"- Excluded samples: {dataset_stats['num_excluded_below_min_gt_pixels']}.",
            f"- Evaluated samples: {dataset_stats['num_images']}.",
            "",
        ]
    )
    excluded = dataset_stats.get("excluded_below_min_gt_pixels", [])
    if excluded:
        lines.extend(
            [
                "| Excluded sample | GT pixels | Reason |",
                "|---|---:|---|",
            ]
        )
        for item in excluded:
            lines.append(
                f"| {item['sample_id']} | {item['gt_pixels']} | {item['reason']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Checkpoint loading and preprocessing",
            "",
        ]
    )
    for row in summaries:
        lines.append(
            f"- **{row['display_name']}**: strict load; input channels={row['input_channels']}; "
            f"normalization={row['normalization']} ({row['normalization_source']}); "
            f"checkpoint=`{row['checkpoint']}`."
        )
        if row.get("preprocessing_warning"):
            lines.append(f"  - Warning: {row['preprocessing_warning']}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_console_summary(summaries: Sequence[Mapping[str, Any]]) -> None:
    print("\nFinal test results")
    print(
        f"{'Model':<18} {'DiceMean':>10} {'DiceMicro':>10} "
        f"{'IoU':>9} {'Precision':>10} {'Recall':>9} {'Specificity':>12}"
    )
    print("-" * 84)
    for row in summaries:
        print(
            f"{row['display_name']:<18} "
            f"{format_optional(row['dice_mean_all']):>10} "
            f"{format_optional(row['dice_micro']):>10} "
            f"{format_optional(row['iou_micro']):>9} "
            f"{format_optional(row['precision_micro']):>10} "
            f"{format_optional(row['recall_micro']):>9} "
            f"{format_optional(row['specificity_micro']):>12}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Strictly reload MONAI UNet/Attention U-Net/VNet best checkpoints "
            "and evaluate them on Size_512 after filtering tiny-GT test samples."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("./datasets/Size_512"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory containing exactly one checkpoint for each of the three models.",
    )
    parser.add_argument("--unet", type=Path, default=None, help="Exact UNet checkpoint path.")
    parser.add_argument(
        "--attention-unet",
        dest="attention_unet",
        type=Path,
        default=None,
        help="Exact Attention U-Net checkpoint path.",
    )
    parser.add_argument("--vnet", type=Path, default=None, help="Exact VNet checkpoint path.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--foreground-values",
        type=parse_foreground_values,
        default=(1,),
        help="Comma-separated foreground label values in prepared masks.",
    )
    parser.add_argument(
        "--normalization",
        choices=("auto", "none", "fixed-gray", "imagenet"),
        default="auto",
        help="Auto restores checkpoint metadata and supports the legacy RGB fallback.",
    )
    parser.add_argument("--gray-mean", type=float, default=None)
    parser.add_argument("--gray-std", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="Examples: auto, cuda, cuda:0, cpu")
    parser.add_argument(
        "--amp-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--require-positive-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require every final Size_512 test mask to contain consolidation.",
    )
    parser.add_argument(
        "--min-gt-pixels",
        type=int,
        default=30,
        help=(
            "Exclude test samples whose foreground GT pixel count is below this "
            "threshold before inference. Use 0 to disable filtering."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Debug only: evaluate the first N sorted test files.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when one model fails instead of preserving successful results.",
    )
    args = parser.parse_args()

    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be strictly between 0 and 1")
    if args.image_size <= 0:
        parser.error("--image-size must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.min_gt_pixels < 0:
        parser.error("--min-gt-pixels cannot be negative")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if (args.gray_mean is None) != (args.gray_std is None):
        parser.error("--gray-mean and --gray-std must be supplied together")
    return args


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    amp_dtype, amp_name = resolve_amp_dtype(device, args.amp_dtype)
    checkpoints = discover_checkpoints(args)
    output_dir = ensure_new_output_dir(args.output_dir)

    records, dataset_stats = scan_test_records(
        data_root=args.data_root.expanduser().resolve(),
        expected_size=args.image_size,
        foreground_values=args.foreground_values,
        require_positive=args.require_positive_test,
        minimum_gt_pixels=args.min_gt_pixels,
        max_samples=args.max_samples,
    )

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Inference precision: {amp_name}")
    print(f"Test images found: {dataset_stats['num_images_found']}")
    print(
        f"Excluded (GT pixels < {dataset_stats['min_gt_pixels']}): "
        f"{dataset_stats['num_excluded_below_min_gt_pixels']}"
    )
    if dataset_stats["excluded_below_min_gt_pixels"]:
        excluded_preview = [
            (item["sample_id"], item["gt_pixels"])
            for item in dataset_stats["excluded_below_min_gt_pixels"][:10]
        ]
        print(f"Excluded examples: {excluded_preview}")
    print(f"Test images to evaluate: {dataset_stats['num_images']}")
    print(f"Output: {output_dir}")

    excluded_rows = dataset_stats["excluded_below_min_gt_pixels"]
    if excluded_rows:
        write_csv(
            output_dir / "excluded_test_samples.csv",
            excluded_rows,
            ("sample_id", "gt_pixels", "reason", "image_path", "mask_path"),
        )

    all_per_image: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    load_reports: Dict[str, Any] = {}

    for model_name in MODEL_ORDER:
        checkpoint_path = checkpoints[model_name]
        model: Optional[nn.Module] = None
        dataset: Optional[Size512TestDataset] = None
        loader: Optional[DataLoader] = None
        print("\n" + "=" * 80)
        print(f"{DISPLAY_NAMES[model_name]}: {checkpoint_path}")
        print("=" * 80)
        try:
            model, input_channels, metadata, load_info = load_model_strict(
                model_name=model_name,
                checkpoint_path=checkpoint_path,
            )
            preprocessing = resolve_preprocessing(
                args=args,
                input_channels=input_channels,
                metadata=metadata,
            )
            print(
                f"Strict load: OK | channels={input_channels} | "
                f"normalization={preprocessing.mode} ({preprocessing.source})"
            )
            if preprocessing.warning:
                print(f"WARNING: {preprocessing.warning}")

            dataset = Size512TestDataset(
                records=records,
                preprocessing=preprocessing,
                foreground_values=args.foreground_values,
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                persistent_workers=(args.num_workers > 0),
                drop_last=False,
            )
            per_image_rows, summary = evaluate_one_model(
                model_name=model_name,
                model=model,
                loader=loader,
                device=device,
                amp_dtype=amp_dtype,
                threshold=args.threshold,
                checkpoint_path=checkpoint_path,
                metadata=metadata,
                preprocessing=preprocessing,
                load_info=load_info,
            )
            all_per_image.extend(per_image_rows)
            summaries.append(summary)
            load_reports[model_name] = {
                "checkpoint": str(checkpoint_path),
                "checkpoint_metadata": metadata,
                "load_info": load_info,
                "preprocessing": preprocessing.__dict__,
            }
            print(
                f"Done | DiceMean={summary['dice_mean_all']:.4f} | "
                f"PrecisionMicro={summary['precision_micro']:.4f} | "
                f"RecallMicro={summary['recall_micro']:.4f}"
            )
        except Exception as exc:
            error = {
                "model": model_name,
                "checkpoint": str(checkpoint_path),
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            errors.append(error)
            print(
                f"ERROR: {DISPLAY_NAMES[model_name]} failed: "
                f"{error['error_type']}: {error['message']}",
                file=sys.stderr,
            )
            if args.fail_fast:
                raise
        finally:
            del loader, dataset, model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if all_per_image:
        write_csv(output_dir / "per_image_metrics.csv", all_per_image, PER_IMAGE_FIELDS)
    if summaries:
        write_csv(output_dir / "benchmark_summary.csv", summaries, SUMMARY_FIELDS)
        write_markdown_report(
            output_dir / "test_report.md",
            summaries=summaries,
            dataset_stats=dataset_stats,
            args=args,
            amp_name=amp_name,
        )
        print_console_summary(summaries)

    run_config = {
        "created_at": datetime.now().astimezone().isoformat(),
        "script": str(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "monai": monai.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "amp_dtype": amp_name,
        "arguments": vars(args),
        "dataset_stats": dataset_stats,
        "checkpoints": {key: str(value) for key, value in checkpoints.items()},
        "load_reports": load_reports,
        "errors": errors,
    }
    write_json(output_dir / "results.json", run_config)
    write_json(output_dir / "run_config.json", {
        key: value for key, value in run_config.items() if key != "load_reports"
    })

    print(f"\nSaved results to: {output_dir}")
    for filename in (
        "benchmark_summary.csv",
        "per_image_metrics.csv",
        "test_report.md",
        "excluded_test_samples.csv",
        "results.json",
        "run_config.json",
    ):
        path = output_dir / filename
        if path.exists():
            print(f"  - {path}")

    if errors:
        raise SystemExit(
            f"{len(errors)} model(s) failed. Successful model results were preserved; "
            "see results.json for exact errors."
        )
    if len(summaries) != len(MODEL_ORDER):
        raise SystemExit("Evaluation did not complete for all three models.")


if __name__ == "__main__":
    main()