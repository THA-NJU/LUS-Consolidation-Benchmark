#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zero-shot transfer evaluation for legacy ResNet34 FPN/DeepLabV3+ weights.

The old checkpoints were trained for joint B-line/consolidation segmentation.
This script does NOT fine-tune them.  It loads each checkpoint strictly, extracts
the consolidation prediction, and evaluates it on the current positive-only
binary consolidation benchmark at native 512 and 224 resolution.

Expected dataset layout:
    <data-root>/<split>/images/*
    <data-root>/<split>/masks/*

Prepared benchmark masks use 0=background and 1=consolidation.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import platform
import time
try:
    from importlib.metadata import PackageNotFoundError, version as package_version
except ImportError:  # Python 3.7 compatibility
    from importlib_metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:
    raise SystemExit(
        "segmentation_models_pytorch is required. Run this script with the old "
        "ML environment that created the checkpoints, or install it there."
    ) from exc


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_DATA_ROOTS = {
    512: Path("./data/Size_512"),
    224: Path("./data/Size_224_filtered"),
}
DEFAULT_WEIGHTS_ROOT = Path("./pretrained/legacy_resnet")
DEFAULT_OUTPUT_ROOT = Path("./outputs/evaluation/legacy_resnet")
CHECKPOINTS = {
    "fpn_resnet34": "FPN_resnet34_best.pth",
    "deeplabv3plus_resnet34": "DeepLabV3Plus_resnet34_best.pth",
}
EXPECTED_COUNTS = {
    512: {"test": 1539},
    224: {"test": 3557},
}


class BenchmarkDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        input_channels: int,
        normalization: str,
        dataset_mean: float,
        dataset_std: float,
    ) -> None:
        self.data_root = data_root
        self.split = split
        self.image_dir = data_root / split / "images"
        self.mask_dir = data_root / split / "masks"
        self.input_channels = input_channels
        self.normalization = normalization
        self.dataset_mean = float(dataset_mean)
        self.dataset_std = float(dataset_std)

        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {self.image_dir} and {self.mask_dir}"
            )
        self.pairs = self._make_pairs()

    def _make_pairs(self) -> List[Tuple[Path, Path]]:
        images = sorted(
            p for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise RuntimeError(f"No images found in {self.image_dir}")
        pairs: List[Tuple[Path, Path]] = []
        for image_path in images:
            exact = self.mask_dir / image_path.name
            if exact.is_file():
                mask_path = exact
            else:
                matches = sorted(
                    p for p in self.mask_dir.glob(image_path.stem + ".*")
                    if p.suffix.lower() in IMAGE_EXTENSIONS
                )
                if len(matches) != 1:
                    raise FileNotFoundError(
                        f"Expected one mask for {image_path.name}; found {matches}"
                    )
                mask_path = matches[0]
            pairs.append((image_path, mask_path))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def read_gray(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Cannot read image: {path}")
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        elif image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)
            lo, hi = float(image.min()), float(image.max())
            image = np.zeros_like(image) if hi <= lo else (image - lo) / (hi - lo)
        return image

    @staticmethod
    def read_mask(path: Path) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Cannot read mask: {path}")
        if mask.ndim == 3:
            mask = mask[..., 0]
        # Current prepared datasets are binary: consolidation foreground == 1.
        return (mask == 1).astype(np.uint8)

    def _normalize(self, gray: np.ndarray) -> np.ndarray:
        if self.input_channels == 1:
            image = gray[None, ...]
            if self.normalization == "imagenet":
                # A one-channel old checkpoint cannot literally use RGB ImageNet
                # normalization; use the average statistics and record this choice.
                image = (image - np.mean([0.485, 0.456, 0.406])) / np.mean(
                    [0.229, 0.224, 0.225]
                )
            elif self.normalization == "dataset":
                image = (image - self.dataset_mean) / self.dataset_std
            return image.astype(np.float32)

        image = np.repeat(gray[None, ...], self.input_channels, axis=0)
        if self.normalization == "imagenet":
            if self.input_channels != 3:
                raise ValueError(
                    "ImageNet normalization is defined for 3-channel checkpoints; "
                    f"checkpoint expects {self.input_channels} channels"
                )
            mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
            std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
            image = (image - mean) / std
        elif self.normalization == "dataset":
            image = (image - self.dataset_mean) / self.dataset_std
        return image.astype(np.float32)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_path, mask_path = self.pairs[index]
        gray = self.read_gray(image_path)
        mask = self.read_mask(mask_path)
        if gray.shape != mask.shape:
            raise ValueError(
                f"Image/mask shape mismatch for {image_path.name}: "
                f"{gray.shape} versus {mask.shape}"
            )
        return {
            "image": torch.from_numpy(self._normalize(gray)),
            "mask": torch.from_numpy(mask),
            "name": image_path.name,
            "image_path": str(image_path),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", type=Path, default=DEFAULT_WEIGHTS_ROOT)
    parser.add_argument("--data-root-512", type=Path, default=DEFAULT_DATA_ROOTS[512])
    parser.add_argument("--data-root-224", type=Path, default=DEFAULT_DATA_ROOTS[224])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sizes", type=int, nargs="+", choices=(224, 512), default=[512, 224])
    parser.add_argument(
        "--models", nargs="+", choices=tuple(CHECKPOINTS), default=list(CHECKPOINTS)
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size-512", type=int, default=4)
    parser.add_argument("--batch-size-224", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--normalization",
        choices=("imagenet", "dataset", "none"),
        default="imagenet",
        help="Must match old checkpoint training. SMP ResNet34 commonly used imagenet.",
    )
    parser.add_argument("--dataset-mean", type=float, default=0.100638)
    parser.add_argument("--dataset-std", type=float, default=0.145868)
    parser.add_argument(
        "--output-mode",
        choices=("auto", "sigmoid", "softmax_argmax", "softmax_threshold"),
        default="auto",
    )
    parser.add_argument(
        "--consolidation-channel",
        type=int,
        default=None,
        help="Default: 0 for one output, 1 for two outputs, 2 for three outputs.",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--visual-cases", type=int, default=5)
    parser.add_argument("--skip-count-check", action="store_true")
    return parser.parse_args()


def torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint: Any) -> Tuple[MutableMapping[str, torch.Tensor], str, Mapping[str, Any]]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must be a mapping, got {type(checkpoint).__name__}")
    tensor_ratio = sum(torch.is_tensor(v) for v in checkpoint.values()) / max(len(checkpoint), 1)
    if tensor_ratio > 0.8:
        return dict(checkpoint), "root", checkpoint
    for key in ("state_dict", "model_state_dict", "model", "net", "network", "weights"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping) and value and all(
            isinstance(k, str) for k in value.keys()
        ):
            tensors = {k: v for k, v in value.items() if torch.is_tensor(v)}
            if tensors:
                return tensors, key, checkpoint
    raise KeyError(
        "Could not find a state_dict. Top-level keys: "
        + ", ".join(map(str, list(checkpoint.keys())[:30]))
    )


def strip_common_wrappers(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    result = dict(state)
    wrappers = ("module.", "model.", "net.", "network.")
    changed = True
    while changed and result:
        changed = False
        for prefix in wrappers:
            if all(key.startswith(prefix) for key in result):
                result = {key[len(prefix):]: value for key, value in result.items()}
                changed = True
                break
    return result


def infer_model_dimensions(state: Mapping[str, torch.Tensor]) -> Tuple[int, int, str, str]:
    input_candidates = [
        (key, value) for key, value in state.items()
        if torch.is_tensor(value)
        and value.ndim == 4
        and (key.endswith("encoder.conv1.weight") or key.endswith("conv1.weight"))
    ]
    if not input_candidates:
        raise KeyError("Cannot infer input channels: encoder conv1 weight was not found")
    input_key, input_weight = input_candidates[0]
    input_channels = int(input_weight.shape[1])

    output_candidates = [
        (key, value) for key, value in state.items()
        if torch.is_tensor(value)
        and value.ndim in (1, 4)
        and "segmentation_head" in key
        and (key.endswith("weight") or key.endswith("bias"))
    ]
    if not output_candidates:
        raise KeyError("Cannot infer output channels: segmentation_head was not found")
    # The final segmentation convolution/bias has the smallest plausible output
    # channel count. Prefer a weight tensor over bias for a more useful audit key.
    output_candidates.sort(key=lambda item: (int(item[1].shape[0]), item[1].ndim != 4))
    output_key, output_weight = output_candidates[0]
    output_channels = int(output_weight.shape[0])
    if not 1 <= output_channels <= 32:
        raise ValueError(
            f"Implausible output channel count {output_channels} inferred from {output_key}"
        )
    return input_channels, output_channels, input_key, output_key


def resolve_smp_architecture(architecture: str):
    """Resolve an SMP model class across public and package-internal layouts."""
    public_class = getattr(smp, architecture, None)
    if public_class is not None:
        return public_class, f"segmentation_models_pytorch.{architecture}"

    module_candidates = {
        "FPN": (
            "segmentation_models_pytorch.decoders.fpn.model",
            "segmentation_models_pytorch.fpn.model",
        ),
        "DeepLabV3Plus": (
            "segmentation_models_pytorch.decoders.deeplabv3.model",
            "segmentation_models_pytorch.deeplabv3.model",
        ),
    }
    if architecture not in module_candidates:
        raise KeyError(architecture)

    failures: List[str] = []
    for module_name in module_candidates[architecture]:
        try:
            module = importlib.import_module(module_name)
            model_class = getattr(module, architecture)
            return model_class, f"{module_name}.{architecture}"
        except (ImportError, AttributeError) as exc:
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")

    module_origin = getattr(smp, "__file__", None)
    module_paths = list(getattr(smp, "__path__", []))
    visible_names = sorted(name for name in dir(smp) if not name.startswith("_"))
    raise RuntimeError(
        "Cannot resolve the requested architecture from the imported "
        "segmentation_models_pytorch module.\n"
        f"architecture={architecture}\n"
        f"smp.__file__={module_origin!r}\n"
        f"smp.__path__={module_paths!r}\n"
        f"visible public names={visible_names[:80]}\n"
        "fallback import failures:\n  - " + "\n  - ".join(failures) + "\n"
        "A normal segmentation-models-pytorch installation provides FPN and "
        "DeepLabV3Plus. If smp.__file__ points inside the legacy environment, a local "
        "file or directory named segmentation_models_pytorch is shadowing the "
        "installed package; rename that local item and rerun. Otherwise reinstall "
        "the package in this exact venv."
    )


def build_model(
    architecture: str,
    input_channels: int,
    output_channels: int,
) -> Tuple[torch.nn.Module, str]:
    kwargs = dict(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=input_channels,
        classes=output_channels,
        activation=None,
    )
    model_class, constructor_source = resolve_smp_architecture(architecture)
    return model_class(**kwargs), constructor_source


def load_state_strict(model: torch.nn.Module, state: Mapping[str, torch.Tensor], path: Path) -> None:
    model_keys = set(model.state_dict())
    candidates: List[Dict[str, torch.Tensor]] = [dict(state), strip_common_wrappers(state)]
    # Some training wrappers prefix only the actual SMP model keys.
    for prefix in ("model.", "module.", "net.", "network.", "segmentation_model."):
        reduced = {
            key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)
        }
        if reduced:
            candidates.append(reduced)
    best = max(candidates, key=lambda candidate: len(model_keys.intersection(candidate)))
    missing = sorted(model_keys - set(best))
    unexpected = sorted(set(best) - model_keys)
    if missing or unexpected:
        message = [
            f"Checkpoint is not strictly compatible with standard SMP model: {path}",
            f"matched={len(model_keys.intersection(best))}/{len(model_keys)}",
            f"missing({len(missing)}): {missing[:15]}",
            f"unexpected({len(unexpected)}): {unexpected[:15]}",
            "Use the original ML training script/model constructor if custom decoder settings were used.",
        ]
        raise RuntimeError("\n".join(message))
    model.load_state_dict(best, strict=True)


def resolve_decode(
    output_channels: int,
    requested_mode: str,
    requested_channel: Optional[int],
) -> Tuple[str, int]:
    if requested_channel is None:
        if output_channels == 1:
            channel = 0
        elif output_channels == 2:
            channel = 1  # expected old order: B-line, consolidation
        elif output_channels == 3:
            channel = 2  # expected old order: background, B-line, consolidation
        else:
            raise ValueError(
                f"Cannot safely infer consolidation channel for {output_channels} outputs; "
                "pass --consolidation-channel explicitly"
            )
    else:
        channel = requested_channel
    if not 0 <= channel < output_channels:
        raise ValueError(
            f"consolidation channel {channel} is invalid for {output_channels} outputs"
        )

    if requested_mode == "auto":
        mode = "softmax_argmax" if output_channels >= 3 else "sigmoid"
    else:
        mode = requested_mode
    if output_channels == 1 and mode.startswith("softmax"):
        raise ValueError("Softmax cannot decode a one-channel output")
    return mode, channel


def extract_logits(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        logits = output
    elif isinstance(output, Mapping):
        for key in ("out", "logits", "mask", "masks", "pred"):
            if key in output and torch.is_tensor(output[key]):
                logits = output[key]
                break
        else:
            raise TypeError(f"No tensor output in mapping keys {list(output)}")
    elif isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        logits = output[0]
    else:
        raise TypeError(f"Unsupported model output: {type(output).__name__}")
    if logits.ndim != 4:
        raise ValueError(f"Expected BCHW logits, got {tuple(logits.shape)}")
    return logits


def decode_prediction(
    logits: torch.Tensor,
    mode: str,
    channel: int,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mode == "sigmoid":
        probability = torch.sigmoid(logits[:, channel])
        prediction = probability >= threshold
    elif mode == "softmax_argmax":
        probabilities = torch.softmax(logits, dim=1)
        probability = probabilities[:, channel]
        prediction = logits.argmax(dim=1) == channel
    elif mode == "softmax_threshold":
        probability = torch.softmax(logits, dim=1)[:, channel]
        prediction = probability >= threshold
    else:
        raise KeyError(mode)
    return probability, prediction


def surface(mask: np.ndarray) -> np.ndarray:
    return np.logical_xor(
        mask, binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    )


def hd_metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    side_length = float(max(gt.shape))
    if not gt.any():
        return (0.0, 0.0) if not pred.any() else (side_length, side_length)
    if not pred.any():
        # Include empty predictions in HD/HD95 aggregates using the track side.
        return side_length, side_length
    pred_surface, gt_surface = surface(pred.astype(bool)), surface(gt.astype(bool))
    distances = np.concatenate(
        [
            distance_transform_edt(~gt_surface)[pred_surface],
            distance_transform_edt(~pred_surface)[gt_surface],
        ]
    ).astype(np.float64)
    return float(distances.max()), float(np.percentile(distances, 95))


def connected_components(mask: np.ndarray) -> int:
    count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return int(count - 1)


def case_metrics(
    name: str,
    pred: np.ndarray,
    gt: np.ndarray,
    inference_ms: float,
    image_path: str,
) -> Dict[str, Any]:
    pred, gt = pred.astype(bool), gt.astype(bool)
    gt_pixels, pred_pixels = int(gt.sum()), int(pred.sum())
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    dice = 2.0 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    hd, hd95 = hd_metrics(pred, gt)
    gt_cc, pred_cc = connected_components(gt), connected_components(pred)
    return {
        "filename": name,
        "image_path": image_path,
        "metric_included": int(gt_pixels > 0),
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
        "dice": dice,
        "iou": iou,
        "hd": hd,
        "hd95": hd95,
        "precision": precision,
        "recall": recall,
        "gt_cc": gt_cc,
        "pred_cc": pred_cc,
        "cc_delta": gt_cc - pred_cc,
        "abs_cc_delta": abs(gt_cc - pred_cc),
        "inference_time_ms": float(inference_ms),
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    included = [row for row in rows if int(row["metric_included"]) == 1]
    result: Dict[str, Any] = {
        "total_samples": len(rows),
        "evaluated_gt_nonempty_count": len(included),
        "gt_empty_excluded_count": len(rows) - len(included),
        "empty_prediction_count": sum(int(row["pred_pixels"]) == 0 for row in included),
    }
    result["empty_prediction_rate"] = result["empty_prediction_count"] / max(len(included), 1)
    for metric in (
        "dice", "iou", "recall", "precision", "hd95", "hd",
        "cc_delta", "abs_cc_delta", "inference_time_ms",
    ):
        source = rows if metric == "inference_time_ms" else included
        values = np.asarray([float(row[metric]) for row in source], dtype=np.float64)
        values = values[np.isfinite(values)]
        result[f"{metric}_mean"] = float(values.mean()) if values.size else float("nan")
        result[f"{metric}_std"] = float(values.std(ddof=0)) if values.size else float("nan")
        result[f"{metric}_valid_count"] = int(values.size)
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=True)


def write_cases(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "filename", "metric_included", "gt_pixels", "pred_pixels", "dice", "iou",
        "hd", "hd95", "precision", "recall", "gt_cc", "pred_cc", "cc_delta",
        "abs_cc_delta", "inference_time_ms",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def overlay(gray: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    base = np.repeat((gray[..., None] * 255.0).clip(0, 255), 3, axis=2)
    selected = mask.astype(bool)
    base[selected] = 0.45 * base[selected] + 0.55 * np.asarray(color)
    return base.astype(np.uint8)


@torch.inference_mode()
def predict_one_for_visual(
    model: torch.nn.Module,
    dataset: BenchmarkDataset,
    index: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    mode: str,
    channel: int,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    item = dataset[index]
    tensor = item["image"].unsqueeze(0).to(device)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        logits = extract_logits(model(tensor))
        if tuple(logits.shape[-2:]) != tuple(tensor.shape[-2:]):
            logits = F.interpolate(
                logits,
                size=tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        _, pred = decode_prediction(logits.float(), mode, channel, threshold)
    gray = dataset.read_gray(Path(item["image_path"]))
    return gray, item["mask"].numpy(), pred[0].cpu().numpy().astype(np.uint8)


def save_ranked_visuals(
    output_path: Path,
    title: str,
    ranked: Sequence[Mapping[str, Any]],
    model: torch.nn.Module,
    dataset: BenchmarkDataset,
    index_by_name: Mapping[str, int],
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    mode: str,
    channel: int,
    threshold: float,
) -> None:
    if not ranked:
        return
    fig, axes = plt.subplots(len(ranked), 3, figsize=(10, 3.1 * len(ranked)), squeeze=False)
    for row_index, row in enumerate(ranked):
        name = str(row["filename"])
        gray, gt, pred = predict_one_for_visual(
            model, dataset, index_by_name[name], device, amp_enabled, amp_dtype,
            mode, channel, threshold,
        )
        panels = [gray, overlay(gray, gt, (0, 255, 0)), overlay(gray, pred, (255, 0, 0))]
        labels = [name, "GT: consolidation (green)", f"Prediction (red), Dice={row['dice']:.4f}"]
        for column, (panel, label) in enumerate(zip(panels, labels)):
            axes[row_index, column].imshow(panel, cmap="gray" if column == 0 else None)
            axes[row_index, column].set_title(label, fontsize=9)
            axes[row_index, column].axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    mode: str,
    channel: int,
    threshold: float,
    description: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    model.eval()
    for batch in tqdm(loader, desc=description, dynamic_ncols=True):
        images = batch["image"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = extract_logits(model(images))
            if tuple(logits.shape[-2:]) != tuple(images.shape[-2:]):
                logits = F.interpolate(
                    logits,
                    size=images.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            probabilities, predictions = decode_prediction(
                logits.float(), mode, channel, threshold
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        per_image_ms = elapsed_ms / images.shape[0]

        if not torch.isfinite(probabilities).all():
            raise FloatingPointError("Prediction contains NaN or Inf")
        preds_np = predictions.cpu().numpy().astype(np.uint8)
        masks_np = batch["mask"].numpy().astype(np.uint8)
        for index in range(images.shape[0]):
            rows.append(
                case_metrics(
                    batch["name"][index], preds_np[index], masks_np[index],
                    per_image_ms, batch["image_path"][index],
                )
            )
    return rows


def append_master_summary(path: Path, record: Mapping[str, Any]) -> None:
    fields = [
        "model", "architecture", "size", "split", "checkpoint", "input_channels",
        "output_channels", "decode_mode", "consolidation_channel", "normalization",
        "threshold", "Mean-dice", "Std-dice", "Mean-iou", "Std-iou",
        "Mean-Recall", "Std-Recall", "Mean-Precision", "Std-Precision",
        "empty_prediction_count", "Mean-HD95", "Std-HD95", "Mean-HD", "Std-HD",
        "HD-valid-count",
        "Mean-Delta-CC", "Std-Delta-CC", "Mean-Abs-Delta-CC", "Std-Abs-Delta-CC",
        "Efficiency-ms-image", "Efficiency-std", "N",
    ]
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.dataset_std <= 0:
        raise ValueError("--dataset-std must be positive")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    data_roots = {512: args.data_root_512.expanduser().resolve(), 224: args.data_root_224.expanduser().resolve()}
    weights_root = args.weights_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    master_summary = output_root / "summary.csv"
    if master_summary.exists():
        master_summary.unlink()

    smp_version = getattr(smp, "__version__", None)
    if smp_version is None:
        try:
            smp_version = package_version("segmentation-models-pytorch")
        except PackageNotFoundError:
            smp_version = "unknown"
    smp_origin = getattr(smp, "__file__", None)
    if smp_origin is None:
        smp_origin = list(getattr(smp, "__path__", []))
    print(
        f"torch={torch.__version__}, smp={smp_version}, device={device}\n"
        f"smp_origin={smp_origin}"
    )
    for model_name in args.models:
        architecture = "FPN" if model_name == "fpn_resnet34" else "DeepLabV3Plus"
        checkpoint_path = (weights_root / CHECKPOINTS[model_name]).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch_load_cpu(checkpoint_path)
        raw_state, state_container, metadata = extract_state_dict(checkpoint)
        state = strip_common_wrappers(raw_state)
        input_channels, output_channels, input_key, output_key = infer_model_dimensions(state)
        mode, channel = resolve_decode(
            output_channels, args.output_mode, args.consolidation_channel
        )
        model, constructor_source = build_model(
            architecture, input_channels, output_channels
        )
        load_state_strict(model, state, checkpoint_path)
        model.to(device).eval()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        print("\n" + "=" * 88)
        print(f"model={model_name}, architecture={architecture}")
        print(f"constructor={constructor_source}")
        print(f"checkpoint={checkpoint_path}")
        print(f"state_container={state_container}")
        print(f"input_channels={input_channels} ({input_key})")
        print(f"output_channels={output_channels} ({output_key})")
        print(f"decode_mode={mode}, consolidation_channel={channel}")
        print(f"normalization={args.normalization}, parameters={parameter_count:,}")
        print("=" * 88)

        for size in args.sizes:
            dataset = BenchmarkDataset(
                data_roots[size], args.split, input_channels, args.normalization,
                args.dataset_mean, args.dataset_std,
            )
            expected = EXPECTED_COUNTS.get(size, {}).get(args.split)
            print(f"[{model_name}][{size}][{args.split}] paired={len(dataset)}")
            if expected is not None and len(dataset) != expected and not args.skip_count_check:
                raise RuntimeError(
                    f"Expected {expected} cases for Size_{size}/{args.split}, got {len(dataset)}. "
                    "Use the correct benchmark root, or pass --skip-count-check intentionally."
                )
            batch_size = args.batch_size_512 if size == 512 else args.batch_size_224
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.workers > 0,
            )

            # Preflight catches spatial/output/AMP incompatibility before the full run.
            sample = next(iter(loader))
            sample_images = sample["image"].to(device, non_blocking=True)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                sample_logits = extract_logits(model(sample_images))
            if tuple(sample_logits.shape[-2:]) != tuple(sample_images.shape[-2:]):
                sample_logits = F.interpolate(
                    sample_logits, size=sample_images.shape[-2:], mode="bilinear", align_corners=False
                )
            if sample_logits.shape[1] != output_channels or not torch.isfinite(sample_logits).all():
                raise RuntimeError(
                    f"Preflight failed: output={tuple(sample_logits.shape)}, finite="
                    f"{bool(torch.isfinite(sample_logits).all())}"
                )
            del sample, sample_images, sample_logits

            rows = evaluate(
                model, loader, device, amp_enabled, amp_dtype, mode, channel,
                args.threshold, f"{model_name} Size_{size} {args.split}",
            )
            summary = summarize(rows)
            run_dir = output_root / str(size) / model_name
            run_dir.mkdir(parents=True, exist_ok=True)
            write_cases(run_dir / f"{args.split}_cases.csv", rows)
            write_json(run_dir / f"{args.split}_summary.json", summary)

            settings = {
                "model": model_name,
                "architecture": architecture,
                "encoder": "resnet34",
                "size": size,
                "split": args.split,
                "data_root": str(data_roots[size]),
                "checkpoint": str(checkpoint_path),
                "checkpoint_state_container": state_container,
                "input_channels": input_channels,
                "output_channels": output_channels,
                "input_weight_key": input_key,
                "output_weight_key": output_key,
                "decode_mode": mode,
                "consolidation_channel": channel,
                "channel_assumption": (
                    "single output = consolidation" if output_channels == 1 else
                    "[B-line, consolidation]" if output_channels == 2 else
                    "[background, B-line, consolidation]" if output_channels == 3 else
                    "explicit user-supplied mapping"
                ),
                "normalization": args.normalization,
                "dataset_mean": args.dataset_mean if args.normalization == "dataset" else None,
                "dataset_std": args.dataset_std if args.normalization == "dataset" else None,
                "threshold": args.threshold,
                "amp": amp_enabled,
                "amp_dtype": args.amp_dtype,
                "batch_size": batch_size,
                "workers": args.workers,
                "parameters_total": parameter_count,
                "torch_version": torch.__version__,
                "smp_version": smp_version,
                "smp_origin": str(smp_origin),
                "model_constructor": constructor_source,
                "python_version": platform.python_version(),
                "metric_protocol": {
                    "aggregation": "per-image macro mean/std over GT-nonempty cases",
                    "mask_label_values": [1],
                    "hd_empty_prediction": "track side length; included in HD/HD95 mean and population std",
                    "connected_components": "8-connectivity; delta = GT CC - predicted CC",
                    "efficiency": "forward plus decode/threshold; excludes data loading and metrics",
                },
            }
            write_json(run_dir / "evaluation_settings.json", settings)

            ranked = sorted(
                (row for row in rows if int(row["metric_included"]) == 1),
                key=lambda row: float(row["dice"]),
            )
            index_by_name = {
                image_path.name: index for index, (image_path, _) in enumerate(dataset.pairs)
            }
            visual_n = min(args.visual_cases, len(ranked))
            save_ranked_visuals(
                run_dir / f"{args.split}_bottom{visual_n}_dice.png",
                f"{model_name} Size_{size} {args.split}: Bottom-{visual_n}",
                ranked[:visual_n], model, dataset, index_by_name, device, amp_enabled,
                amp_dtype, mode, channel, args.threshold,
            )
            save_ranked_visuals(
                run_dir / f"{args.split}_top{visual_n}_dice.png",
                f"{model_name} Size_{size} {args.split}: Top-{visual_n}",
                list(reversed(ranked[-visual_n:])), model, dataset, index_by_name,
                device, amp_enabled, amp_dtype, mode, channel, args.threshold,
            )

            record = {
                "model": model_name,
                "architecture": architecture,
                "size": size,
                "split": args.split,
                "checkpoint": str(checkpoint_path),
                "input_channels": input_channels,
                "output_channels": output_channels,
                "decode_mode": mode,
                "consolidation_channel": channel,
                "normalization": args.normalization,
                "threshold": args.threshold,
                "Mean-dice": summary["dice_mean"],
                "Std-dice": summary["dice_std"],
                "Mean-iou": summary["iou_mean"],
                "Std-iou": summary["iou_std"],
                "Mean-Recall": summary["recall_mean"],
                "Std-Recall": summary["recall_std"],
                "Mean-Precision": summary["precision_mean"],
                "Std-Precision": summary["precision_std"],
                "empty_prediction_count": summary["empty_prediction_count"],
                "Mean-HD95": summary["hd95_mean"],
                "Std-HD95": summary["hd95_std"],
                "Mean-HD": summary["hd_mean"],
                "Std-HD": summary["hd_std"],
                "HD-valid-count": summary["hd_valid_count"],
                "Mean-Delta-CC": summary["cc_delta_mean"],
                "Std-Delta-CC": summary["cc_delta_std"],
                "Mean-Abs-Delta-CC": summary["abs_cc_delta_mean"],
                "Std-Abs-Delta-CC": summary["abs_cc_delta_std"],
                "Efficiency-ms-image": summary["inference_time_ms_mean"],
                "Efficiency-std": summary["inference_time_ms_std"],
                "N": summary["evaluated_gt_nonempty_count"],
            }
            append_master_summary(master_summary, record)
            print(json.dumps(record, ensure_ascii=False, indent=2))

        del model, checkpoint, raw_state, state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n[DONE] summary: {master_summary}")


if __name__ == "__main__":
    main()
