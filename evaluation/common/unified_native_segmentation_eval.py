#!/usr/bin/env python3
"""Shared native-resolution evaluation engine for the LUS benchmark.

This module is used by the MONAI and Mamba family entry scripts.  It contains
only family-independent dataset loading, checkpoint loading, metrics,
efficiency measurement, visualisation, and report writing.

Protocol implemented here:
  * evaluate Size_224 patches and Size_512 images independently at native size;
  * fixed foreground probability threshold (normally 0.5);
  * exclude GT-empty cases from all per-case metric aggregates;
  * for GT-nonempty/pred-empty cases, overlap/precision/recall are 0 and
    HD/HD95 equal the track side length and enter distance aggregates;
  * HD/HD95 are symmetric Euclidean surface distances in pixels;
  * connected components use 8-connectivity;
  * cc_delta = gt_components - predicted_components;
  * abs_cc_delta = abs(gt_components - predicted_components), computed per image;
  * report per-image macro mean and population standard deviation only;
  * time GPU forward + probability conversion + thresholding, excluding I/O,
    DataLoader work, host-to-device transfer, metrics, CSV, and visualisation;
  * do not save all prediction masks; save only top/bottom Dice montages.
"""

from __future__ import annotations

import csv
import gc
import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


METRICS = (
    "dice",
    "iou",
    "hd",
    "hd95",
    "precision",
    "recall",
    "cc_delta",
    "abs_cc_delta",
)
CC_STRUCTURE_8 = np.ones((3, 3), dtype=np.uint8)
VISUAL_ALPHA = 0.45


@dataclass
class VisualCase:
    case_name: str
    dice: float
    image_path: Path
    gt: np.ndarray
    pred: np.ndarray


class NativeSegmentationDataset(Dataset):
    """Read a fixed split without augmentation, stitching, or spatial resize."""

    def __init__(
        self,
        split_root: Path,
        image_size: int,
        in_channels: int = 3,
        normalize: bool = False,
        input_mean: float = 0.0,
        input_std: float = 1.0,
    ) -> None:
        self.split_root = Path(split_root).expanduser().resolve()
        self.image_dir = self.split_root / "images"
        self.mask_dir = self.split_root / "masks"
        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.normalize = bool(normalize)
        self.input_mean = float(input_mean)
        self.input_std = float(input_std)

        if self.in_channels not in (1, 3):
            raise ValueError(f"in_channels must be 1 or 3, got {self.in_channels}")
        if self.normalize and self.input_std <= 0:
            raise ValueError("input_std must be positive when normalization is enabled")
        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(
                f"Missing images/ or masks/ below split: {self.split_root}"
            )

        images = {
            path.name: path
            for path in sorted(self.image_dir.glob("*.png"))
            if path.is_file()
        }
        masks = {
            path.name: path
            for path in sorted(self.mask_dir.glob("*.png"))
            if path.is_file()
        }
        if not images:
            raise RuntimeError(f"No PNG images found in {self.image_dir}")
        if images.keys() != masks.keys():
            missing_masks = sorted(images.keys() - masks.keys())
            missing_images = sorted(masks.keys() - images.keys())
            raise RuntimeError(
                f"Image/mask mismatch in {self.split_root}: "
                f"missing_masks={missing_masks[:10]}, "
                f"missing_images={missing_images[:10]}"
            )

        self.samples: List[Tuple[Path, Path]] = [
            (images[name], masks[name]) for name in sorted(images)
        ]
        self._validate_native_sizes()

    def _validate_native_sizes(self) -> None:
        target = (self.image_size, self.image_size)
        bad: List[str] = []
        for image_path, mask_path in self.samples:
            with Image.open(image_path) as image:
                image_size = image.size
            with Image.open(mask_path) as mask:
                mask_size = mask.size
            if image_size != target or mask_size != target:
                bad.append(
                    f"{image_path.name}: image={image_size}, mask={mask_size}"
                )
                if len(bad) >= 20:
                    break
        if bad:
            raise RuntimeError(
                "Native-size evaluation forbids resize. Unexpected file sizes:\n  "
                + "\n  ".join(bad)
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_path, mask_path = self.samples[index]
        with Image.open(image_path) as handle:
            image_np = np.asarray(handle.convert("L"), dtype=np.float32) / 255.0
        with Image.open(mask_path) as handle:
            mask_np = np.asarray(handle.convert("L"))

        if self.normalize:
            image_np = (image_np - self.input_mean) / self.input_std
        image_np = np.ascontiguousarray(image_np[None, ...])
        if self.in_channels == 3:
            image_np = np.repeat(image_np, 3, axis=0)
        # Prepared benchmark masks are binary.  >0 also supports 0/255 PNGs.
        mask_np = np.ascontiguousarray((mask_np > 0).astype(np.int64))

        name = image_path.name
        return {
            "image": torch.from_numpy(image_np).float(),
            "mask": torch.from_numpy(mask_np).long(),
            "case_name": name,
            "patient_id": name.split("_", 1)[0],
        }


def resolve_from(base_dir: Path, path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(base_dir).resolve() / path).resolve()


def check_dataset_layout(data_root: Path, splits: Sequence[str]) -> None:
    for split in splits:
        split_root = Path(data_root) / split
        for child in ("images", "masks"):
            path = split_root / child
            if not path.is_dir():
                raise FileNotFoundError(f"Missing dataset directory: {path}")


def discover_model_directories(
    weights_root: Path,
    weight_family: str,
    size: int,
    supported_names: Sequence[str],
    requested: Optional[Sequence[str]],
) -> List[str]:
    """Return actual supported directory names in deterministic order."""

    size_root = Path(weights_root) / weight_family / str(int(size))
    if not size_root.is_dir():
        raise FileNotFoundError(f"Weight size directory not found: {size_root}")
    present = {
        path.name: path
        for path in size_root.iterdir()
        if path.is_dir()
    }
    supported_set = set(supported_names)

    if requested:
        selected: List[str] = []
        for name in requested:
            if name not in supported_set:
                raise KeyError(
                    f"Unsupported model {name!r}; supported={list(supported_names)}"
                )
            if name not in present:
                raise FileNotFoundError(
                    f"Requested model directory not found: {size_root / name}"
                )
            if name not in selected:
                selected.append(name)
        return selected

    return [name for name in supported_names if name in present]


def find_best_checkpoint(
    weights_root: Path,
    weight_family: str,
    size: int,
    model_dir_name: str,
) -> Path:
    model_dir = (
        Path(weights_root) / weight_family / str(int(size)) / model_dir_name
    )
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Weight directory not found: {model_dir}")

    exact = model_dir / "best_model.pth"
    if exact.is_file():
        return exact.resolve()
    candidates = sorted(
        path
        for path in model_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".pth"
        and "best" in path.name.lower()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No .pth filename containing 'best' found in {model_dir}"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Ambiguous best checkpoints in {model_dir}: "
            + ", ".join(path.name for path in candidates)
        )
    return candidates[0].resolve()


def _looks_like_state_dict(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(key, str) for key in value)
        and all(torch.is_tensor(item) for item in value.values())
    )


def extract_state_dict(payload: object, checkpoint_path: Path) -> Mapping[str, Any]:
    if _looks_like_state_dict(payload):
        return payload  # type: ignore[return-value]
    if not isinstance(payload, Mapping):
        raise TypeError(f"Unsupported checkpoint object in {checkpoint_path}")

    for key in (
        "model_state_dict",
        "model_state",
        "state_dict",
        "network_state_dict",
        "net",
    ):
        candidate = payload.get(key)
        if _looks_like_state_dict(candidate):
            return candidate  # type: ignore[return-value]
    raise TypeError(
        f"Could not find a model state_dict in checkpoint: {checkpoint_path}"
    )


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    try:
        payload = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)
    state = extract_state_dict(payload, checkpoint_path)

    attempts: List[Mapping[str, Any]] = [state]
    if state and all(key.startswith("module.") for key in state):
        attempts.append({key[7:]: value for key, value in state.items()})
    if state and all(key.startswith("_orig_mod.") for key in state):
        attempts.append({key[10:]: value for key, value in state.items()})

    errors: List[str] = []
    for candidate in attempts:
        try:
            model.load_state_dict(candidate, strict=True)
            return dict(payload) if isinstance(payload, Mapping) else {}
        except RuntimeError as exc:
            errors.append(str(exc))
    raise RuntimeError(
        f"Strict checkpoint loading failed for {checkpoint_path}:\n"
        + "\n---\n".join(errors)
    )


def unwrap_primary_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, Mapping):
        for key in ("out", "logits", "pred", "prediction"):
            value = output.get(key)
            if torch.is_tensor(value):
                return value
        for value in output.values():
            if torch.is_tensor(value):
                return value
        raise RuntimeError(f"Model dict output has no tensor: {list(output)}")
    if isinstance(output, (list, tuple)):
        for value in output:
            if torch.is_tensor(value):
                return value
        raise RuntimeError("Model sequence output has no tensor")
    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def resolve_amp_dtype(name: str, device: torch.device) -> Optional[torch.dtype]:
    normalized = str(name).lower()
    if normalized == "fp32":
        return None
    if device.type != "cuda":
        raise RuntimeError("BF16/FP16 evaluation requires CUDA")
    if normalized == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Selected GPU does not support BF16")
        return torch.bfloat16
    if normalized == "fp16":
        return torch.float16
    raise ValueError(f"Unknown AMP dtype {name!r}; use bf16, fp16, or fp32")


def autocast_context(device: torch.device, amp_dtype: Optional[torch.dtype]):
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=True,
    )


def foreground_probabilities(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    with autocast_context(device, amp_dtype):
        logits = unwrap_primary_output(model(images))
    if logits.ndim != 4:
        raise RuntimeError(f"Expected BCHW logits, got {tuple(logits.shape)}")
    if tuple(logits.shape[-2:]) != tuple(images.shape[-2:]):
        raise RuntimeError(
            "Native-size evaluation forbids output resize: "
            f"input_hw={tuple(images.shape[-2:])}, "
            f"logit_hw={tuple(logits.shape[-2:])}"
        )
    if logits.shape[1] == 2:
        return torch.softmax(logits.float(), dim=1)[:, 1]
    if logits.shape[1] == 1:
        return torch.sigmoid(logits.float())[:, 0]
    raise RuntimeError(
        f"Expected one or two output channels, got {tuple(logits.shape)}"
    )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(workers),
        pin_memory=bool(pin_memory),
        persistent_workers=int(workers) > 0,
    )


def batch_candidates(maximum: int) -> List[int]:
    maximum = max(1, int(maximum))
    values: List[int] = []
    current = 1
    while current <= maximum:
        values.append(current)
        current *= 2
    if values[-1] != maximum:
        values.append(maximum)
    return values


def _is_cuda_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    return (
        (oom_type is not None and isinstance(exc, oom_type))
        or "out of memory" in str(exc).lower()
    )


@torch.inference_mode()
def benchmark_batch_size(
    model: torch.nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    threshold: float,
    maximum: int,
    warmup_steps: int,
    timed_steps: int,
) -> Tuple[int, List[Dict[str, Any]]]:
    """Choose the tested batch size with the highest measured throughput."""

    if device.type != "cuda":
        raise RuntimeError("The agreed efficiency protocol requires CUDA")
    rows: List[Dict[str, Any]] = []
    model.eval()

    for batch_size in batch_candidates(maximum):
        print(
            f"[batch-tune] trying batch_size={batch_size} "
            f"(maximum={int(maximum)})",
            flush=True,
        )
        images: Optional[torch.Tensor] = None
        try:
            torch.cuda.empty_cache()
            images = sample.repeat(batch_size, 1, 1, 1).to(device)
            torch.cuda.reset_peak_memory_stats(device)
            for _ in range(int(warmup_steps)):
                probs = foreground_probabilities(model, images, device, amp_dtype)
                _ = probs >= float(threshold)
            torch.cuda.synchronize(device)

            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            for _ in range(int(timed_steps)):
                probs = foreground_probabilities(model, images, device, amp_dtype)
                _ = probs >= float(threshold)
            ender.record()
            ender.synchronize()
            elapsed_ms = float(starter.elapsed_time(ender))
            images_seen = int(batch_size) * int(timed_steps)
            rows.append(
                {
                    "batch_size": int(batch_size),
                    "status": "ok",
                    "images_per_s": images_seen / max(elapsed_ms / 1000.0, 1e-12),
                    "ms_per_image": elapsed_ms / max(images_seen, 1),
                    "peak_memory_gb": (
                        float(torch.cuda.max_memory_allocated(device)) / (1024 ** 3)
                    ),
                }
            )
            del probs
        except BaseException as exc:
            if not _is_cuda_oom(exc):
                raise
            rows.append(
                {
                    "batch_size": int(batch_size),
                    "status": "oom",
                    "images_per_s": math.nan,
                    "ms_per_image": math.nan,
                    "peak_memory_gb": math.nan,
                }
            )
            break
        finally:
            if images is not None:
                del images
            torch.cuda.empty_cache()

    valid = [row for row in rows if row["status"] == "ok"]
    if not valid:
        raise RuntimeError("Every candidate batch size ran out of GPU memory")
    best = max(valid, key=lambda row: float(row["images_per_s"]))
    return int(best["batch_size"]), rows


@torch.inference_mode()
def warmup_model(
    model: torch.nn.Module,
    sample: torch.Tensor,
    batch_size: int,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    threshold: float,
    warmup_steps: int,
) -> None:
    images = sample.repeat(int(batch_size), 1, 1, 1).to(device)
    for _ in range(int(warmup_steps)):
        probs = foreground_probabilities(model, images, device, amp_dtype)
        _ = probs >= float(threshold)
    torch.cuda.synchronize(device)
    del images, probs


def connected_components(mask: np.ndarray) -> int:
    _, count = ndimage.label(mask.astype(bool), structure=CC_STRUCTURE_8)
    return int(count)


def surface_distances(gt: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    if not gt.any():
        return math.nan, math.nan
    if not pred.any():
        penalty = float(max(gt.shape))
        return penalty, penalty

    gt_surface = gt ^ ndimage.binary_erosion(
        gt,
        structure=CC_STRUCTURE_8,
        border_value=0,
    )
    pred_surface = pred ^ ndimage.binary_erosion(
        pred,
        structure=CC_STRUCTURE_8,
        border_value=0,
    )
    distance_to_gt = ndimage.distance_transform_edt(~gt_surface)
    distance_to_pred = ndimage.distance_transform_edt(~pred_surface)
    distances = np.concatenate(
        (
            distance_to_pred[gt_surface],
            distance_to_gt[pred_surface],
        )
    ).astype(np.float64, copy=False)
    if distances.size == 0:
        return math.nan, math.nan
    return float(np.max(distances)), float(np.percentile(distances, 95))


def compute_case_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    gt_pixels = int(gt.sum())
    pred_pixels = int(pred.sum())
    if gt_pixels == 0:
        raise ValueError("GT-empty cases must be filtered before metrics")

    tp = int(np.logical_and(gt, pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())
    pred_empty = pred_pixels == 0

    dice = 0.0 if pred_empty else (2.0 * tp / max(2 * tp + fp + fn, 1))
    iou = 0.0 if pred_empty else (tp / max(tp + fp + fn, 1))
    precision = 0.0 if pred_empty else (tp / max(tp + fp, 1))
    recall = 0.0 if pred_empty else (tp / max(tp + fn, 1))
    hd, hd95 = surface_distances(gt, pred)
    gt_cc = connected_components(gt)
    pred_cc = connected_components(pred)

    cc_delta = int(gt_cc - pred_cc)
    return {
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": float(dice),
        "iou": float(iou),
        "hd": float(hd),
        "hd95": float(hd95),
        "precision": float(precision),
        "recall": float(recall),
        "gt_cc": int(gt_cc),
        "pred_cc": int(pred_cc),
        "cc_delta": cc_delta,
        "abs_cc_delta": abs(cc_delta),
        "pred_empty": bool(pred_empty),
    }


def finite_values(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> np.ndarray:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def population_mean_std(values: np.ndarray) -> Tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    return float(np.mean(values)), float(np.std(values, ddof=0))


def summarize_split(
    split: str,
    case_rows: Sequence[Mapping[str, Any]],
    total_samples: int,
    gt_empty_count: int,
    efficiency_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    empty_prediction_count = sum(bool(row["pred_empty"]) for row in case_rows)
    result: Dict[str, Any] = {
        "split": split,
        "total_samples": int(total_samples),
        "gt_empty_excluded_count": int(gt_empty_count),
        "evaluated_gt_nonempty_count": int(len(case_rows)),
        "empty_prediction_count": int(empty_prediction_count),
        "empty_prediction_rate": (
            float(empty_prediction_count) / len(case_rows)
            if case_rows
            else math.nan
        ),
    }
    for metric in METRICS:
        values = finite_values(case_rows, metric)
        mean, std = population_mean_std(values)
        result[f"{metric}_mean"] = mean
        result[f"{metric}_std"] = std
        result[f"{metric}_valid_count"] = int(values.size)

    elapsed = np.asarray(
        [float(row["elapsed_s"]) for row in efficiency_rows],
        dtype=np.float64,
    )
    batch_sizes = np.asarray(
        [int(row["batch_size"]) for row in efficiency_rows],
        dtype=np.int64,
    )
    ms_per_image = np.asarray(
        [float(row["ms_per_image"]) for row in efficiency_rows],
        dtype=np.float64,
    )
    images_per_s = np.asarray(
        [float(row["images_per_s"]) for row in efficiency_rows],
        dtype=np.float64,
    )
    ms_mean, ms_std = population_mean_std(ms_per_image)
    ips_mean, ips_std = population_mean_std(images_per_s)
    timed_images = int(batch_sizes.sum()) if batch_sizes.size else 0
    total_s = float(elapsed.sum()) if elapsed.size else 0.0
    result.update(
        {
            "efficiency_batch_count": int(len(efficiency_rows)),
            "efficiency_timed_images": timed_images,
            "efficiency_total_inference_s": total_s,
            "efficiency_ms_per_image_mean": ms_mean,
            "efficiency_ms_per_image_std": ms_std,
            "efficiency_images_per_s_mean": ips_mean,
            "efficiency_images_per_s_std": ips_std,
            "efficiency_images_per_s_aggregate": (
                timed_images / total_s if total_s > 0 else math.nan
            ),
        }
    )
    return result


def update_visual_cases(
    top_cases: List[VisualCase],
    bottom_cases: List[VisualCase],
    candidate: VisualCase,
    limit: int,
) -> None:
    top_cases.append(candidate)
    top_cases.sort(key=lambda item: (-item.dice, item.case_name))
    del top_cases[int(limit):]

    bottom_cases.append(candidate)
    bottom_cases.sort(key=lambda item: (item.dice, item.case_name))
    del bottom_cases[int(limit):]


def overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: Tuple[int, int, int],
) -> Image.Image:
    base = image.convert("RGB")
    colored = Image.new("RGB", base.size, color)
    alpha = Image.fromarray(
        np.uint8(mask.astype(bool) * round(255 * VISUAL_ALPHA)),
        mode="L",
    )
    return Image.composite(colored, base, alpha)


def save_visual_montage(
    path: Path,
    cases: Sequence[VisualCase],
    image_size: int,
    title: str,
) -> None:
    if not cases:
        return
    header_h = 34
    label_h = 24
    panel = int(image_size)
    canvas = Image.new(
        "RGB",
        (3 * panel, header_h + len(cases) * (label_h + panel)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (6, 5),
        f"{title} | Original | GT overlay (green) | Prediction overlay (red)",
        fill="black",
        font=font,
    )

    for row_index, case in enumerate(cases):
        y = header_h + row_index * (label_h + panel)
        draw.text(
            (6, y + 4),
            f"{case.case_name} | Dice={case.dice:.6f}",
            fill="black",
            font=font,
        )
        with Image.open(case.image_path) as handle:
            original = handle.convert("L")
        if original.size != (panel, panel):
            raise RuntimeError(
                f"Visualisation encountered non-native size: "
                f"{case.image_path} -> {original.size}"
            )
        original_rgb = original.convert("RGB")
        gt_overlay = overlay_mask(original_rgb, case.gt, (0, 255, 0))
        pred_overlay = overlay_mask(original_rgb, case.pred, (255, 0, 0))
        panel_y = y + label_h
        canvas.paste(original_rgb, (0, panel_y))
        canvas.paste(gt_overlay, (panel, panel_y))
        canvas.paste(pred_overlay, (2 * panel, panel_y))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, ensure_ascii=False, indent=2)


@torch.inference_mode()
def evaluate_split(
    model: torch.nn.Module,
    dataset: NativeSegmentationDataset,
    split: str,
    batch_size: int,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    threshold: float,
    workers: int,
    pin_memory: bool,
    visual_cases: int,
    output_dir: Path,
) -> Dict[str, Any]:
    loader = make_loader(dataset, batch_size, workers, pin_memory)
    image_path_by_name = {
        image_path.name: image_path
        for image_path, _ in dataset.samples
    }
    case_rows: List[Dict[str, Any]] = []
    efficiency_rows: List[Dict[str, Any]] = []
    top_cases: List[VisualCase] = []
    bottom_cases: List[VisualCase] = []
    gt_empty_count = 0
    model.eval()

    for batch_index, batch in enumerate(
        tqdm(loader, desc=f"{split} evaluation", unit="batch")
    ):
        # H2D is intentionally outside the timed region.
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"]
        case_names = list(batch["case_name"])
        patient_ids = list(batch["patient_id"])

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        probs = foreground_probabilities(model, images, device, amp_dtype)
        predictions = probs >= float(threshold)
        ender.record()
        ender.synchronize()
        elapsed_s = float(starter.elapsed_time(ender)) / 1000.0
        current_bs = int(images.shape[0])
        efficiency_rows.append(
            {
                "batch_index": int(batch_index),
                "batch_size": current_bs,
                "elapsed_s": elapsed_s,
                "ms_per_image": elapsed_s * 1000.0 / current_bs,
                "images_per_s": current_bs / max(elapsed_s, 1e-12),
            }
        )

        # Metrics begin only after the CUDA timer has stopped.
        pred_np = predictions.detach().cpu().numpy().astype(bool)
        gt_np = targets.numpy().astype(bool)
        for item_index, (case_name, patient_id) in enumerate(
            zip(case_names, patient_ids)
        ):
            gt = gt_np[item_index]
            pred = pred_np[item_index]
            if not gt.any():
                gt_empty_count += 1
                continue
            metrics = compute_case_metrics(gt, pred)
            case_rows.append(
                {
                    "case_name": case_name,
                    "patient_id": patient_id,
                    **metrics,
                }
            )
            candidate = VisualCase(
                case_name=case_name,
                dice=float(metrics["dice"]),
                image_path=image_path_by_name[case_name],
                gt=gt.astype(np.uint8, copy=True),
                pred=pred.astype(np.uint8, copy=True),
            )
            update_visual_cases(
                top_cases,
                bottom_cases,
                candidate,
                visual_cases,
            )
        del images, probs, predictions

    summary = summarize_split(
        split=split,
        case_rows=case_rows,
        total_samples=len(dataset),
        gt_empty_count=gt_empty_count,
        efficiency_rows=efficiency_rows,
    )
    summary.update(
        {
            "batch_size": int(batch_size),
            "threshold": float(threshold),
            "efficiency_timing_scope": (
                "forward+softmax_or_sigmoid+threshold; excludes "
                "disk_io+dataloader+h2d+metrics+csv+visualization"
            ),
            "std_definition": "population_std_ddof_0",
        }
    )

    write_csv(output_dir / f"{split}_cases.csv", case_rows)
    write_csv(
        output_dir / f"{split}_efficiency_batches.csv",
        efficiency_rows,
    )
    write_json(output_dir / f"{split}_summary.json", summary)
    save_visual_montage(
        output_dir / f"{split}_top5_dice.png",
        top_cases,
        dataset.image_size,
        f"{split} top-{len(top_cases)} Dice",
    )
    save_visual_montage(
        output_dir / f"{split}_bottom5_dice.png",
        bottom_cases,
        dataset.image_size,
        f"{split} bottom-{len(bottom_cases)} Dice",
    )
    return summary


def evaluate_model(
    *,
    model: torch.nn.Module,
    model_name: str,
    model_config: Mapping[str, Any],
    family_label: str,
    weight_family: str,
    size: int,
    checkpoint_path: Path,
    data_root: Path,
    output_root: Path,
    splits: Sequence[str],
    device: torch.device,
    amp_name: str,
    threshold: float,
    workers: int,
    auto_batch: bool,
    maximum_batch: int,
    fixed_batch: int,
    warmup_steps: int,
    timed_steps: int,
    pin_memory: bool,
    visual_cases: int = 5,
) -> List[Dict[str, Any]]:
    """Load one model and evaluate every selected split."""

    model = model.to(device)
    checkpoint = load_model_checkpoint(model, checkpoint_path, device)
    model.eval()
    amp_dtype = resolve_amp_dtype(amp_name, device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    datasets = {
        split: NativeSegmentationDataset(
            Path(data_root) / split,
            image_size=int(size),
            in_channels=3,
            normalize=False,
        )
        for split in splits
    }
    sample = next(iter(datasets.values()))[0]["image"].unsqueeze(0)

    if auto_batch:
        eval_batch, tune_rows = benchmark_batch_size(
            model=model,
            sample=sample,
            device=device,
            amp_dtype=amp_dtype,
            threshold=threshold,
            maximum=maximum_batch,
            warmup_steps=warmup_steps,
            timed_steps=timed_steps,
        )
    else:
        eval_batch = int(fixed_batch)
        tune_rows = []
        print(
            f"[batch-tune] disabled; using fixed batch_size={eval_batch}",
            flush=True,
        )
    warmup_model(
        model=model,
        sample=sample,
        batch_size=eval_batch,
        device=device,
        amp_dtype=amp_dtype,
        threshold=threshold,
        warmup_steps=warmup_steps,
    )

    model_output = (
        Path(output_root) / str(int(size)) / model_name
    )
    model_output.mkdir(parents=True, exist_ok=True)
    write_csv(model_output / "batch_tuning.csv", tune_rows)
    settings = {
        "family": family_label,
        "weight_family": weight_family,
        "model": model_name,
        "size": int(size),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_score": checkpoint.get(
            "best_score",
            checkpoint.get("best_val_score"),
        ),
        "data_root": str(data_root),
        "splits": list(splits),
        "threshold": float(threshold),
        "amp_dtype": amp_name,
        "parameter_count": parameter_count,
        "evaluation_batch_size": int(eval_batch),
        "auto_batch_tuning": bool(auto_batch),
        "max_tested_batch_size": int(maximum_batch),
        "model_config": dict(model_config),
        "protocol": {
            "resolution": f"native_{int(size)}_no_stitch_no_resize",
            "gt_empty": "excluded_from_case_metrics",
            "gt_nonempty_pred_empty_overlap_precision_recall": 0,
            "gt_nonempty_pred_empty_hd_hd95": "track_side_length_included",
            "connected_components": "8_connectivity",
            "cc_delta": "gt_cc_minus_pred_cc",
            "abs_cc_delta": "per_image_abs_gt_cc_minus_pred_cc",
            "distance": "symmetric_Euclidean_surface_distance_pixels",
            "aggregation": "per_image_macro_mean_population_std",
            "global_metrics": False,
            "prediction_masks_saved": False,
            "visualizations": "top5_and_bottom5_Dice_per_split_GT_empty_excluded",
        },
    }
    write_json(model_output / "evaluation_settings.json", settings)

    summaries: List[Dict[str, Any]] = []
    for split in splits:
        summary = evaluate_split(
            model=model,
            dataset=datasets[split],
            split=split,
            batch_size=eval_batch,
            device=device,
            amp_dtype=amp_dtype,
            threshold=threshold,
            workers=workers,
            pin_memory=pin_memory,
            visual_cases=visual_cases,
            output_dir=model_output,
        )
        summaries.append(
            {
                "family": family_label,
                "weight_family": weight_family,
                "size": int(size),
                "model": model_name,
                "checkpoint": str(checkpoint_path),
                "amp_dtype": amp_name,
                "parameter_count": parameter_count,
                **summary,
            }
        )
    write_csv(model_output / "summary.csv", summaries)

    del datasets, model
    gc.collect()
    torch.cuda.empty_cache()
    return summaries
