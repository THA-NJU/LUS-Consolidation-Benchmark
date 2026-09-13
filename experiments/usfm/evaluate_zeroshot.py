#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zero-shot transfer evaluation for a previously adapted USFM checkpoint.

The expected checkpoint is the old joint B-line/consolidation model produced by
``ablation.py`` (normally ``usfm_weights/best_usfm_decoder.pth``).  No parameter
is updated.  USFM always receives 224x224 RGB input; predictions are resized to
the native benchmark mask size before metrics are computed.

Old output convention from ablation.py:
    0 = background, 1 = B-line, 2 = consolidation

Expected current benchmark layout:
    <root>/test/images/*
    <root>/test/masks/*

Prepared benchmark masks are binary: 0=background, 1=consolidation.  Masks
encoded as 0/255 are also accepted.  Raw three-class masks are deliberately not
accepted by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# These imports register HVITBackbone4Seg with the MMSeg registry.
try:
    import usdsgen  # noqa: F401
    import usdsgen.models  # noqa: F401
    from mmseg.models import build_segmentor
except ImportError as exc:
    raise SystemExit(
        "USFM dependencies are unavailable. Configure the pinned usfm and "
        "usfm_mmseg sources and its Python environment; see docs/ENVIRONMENTS.md."
    ) from exc


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_DATA_ROOTS = {
    512: Path("./data/Size_512"),
    224: Path("./data/Size_224_filtered"),
}
DEFAULT_CHECKPOINT = Path("./pretrained/usfm/best_usfm_decoder.pth")
DEFAULT_OUTPUT_ROOT = Path("./outputs/evaluation/usfm_zeroshot")
EXPECTED_TEST_COUNTS = {512: 1539, 224: 3557}

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root-512", type=Path, default=DEFAULT_DATA_ROOTS[512])
    parser.add_argument("--data-root-224", type=Path, default=DEFAULT_DATA_ROOTS[224])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sizes", type=int, nargs="+", choices=(512, 224), default=[512, 224])
    parser.add_argument("--split", choices=("test", "val", "train"), default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--consolidation-channel", type=int, default=2)
    parser.add_argument(
        "--decode-mode",
        choices=("argmax", "threshold"),
        default="argmax",
        help="Use argmax for the protocol-faithful old three-class decision rule.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Used only with --decode-mode threshold; never tune it on test data.",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--visual-cases", type=int, default=5)
    parser.add_argument("--skip-count-check", action="store_true")
    parser.add_argument(
        "--allow-nonbinary-mask-values",
        action="store_true",
        help="Explicit override only; by default masks must contain 0/1 or 0/255.",
    )
    return parser.parse_args()


class USFMModel(nn.Module):
    """Exact USFM + UPerHead structure used by the supplied ablation script."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        model_cfg = dict(
            type="EncoderDecoder",
            pretrained=None,
            backbone=dict(
                type="HVITBackbone4Seg",
                img_size=224,
                patch_size=16,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                use_abs_pos_emb=False,
                use_rel_pos_bias=True,
                init_values=0.1,
                drop_path_rate=0.1,
                out_indices=[3, 5, 7, 11],
            ),
            decode_head=dict(
                type="UPerHead",
                in_channels=[768, 768, 768, 768],
                in_index=[0, 1, 2, 3],
                pool_scales=[1, 2, 3, 6],
                channels=768,
                dropout_ratio=0.1,
                num_classes=num_classes,
                norm_cfg=dict(type="BN", requires_grad=True),
                align_corners=False,
                loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            ),
            train_cfg=dict(),
            test_cfg=dict(mode="whole"),
        )
        self.mmseg_model = build_segmentor(model_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.mmseg_model.extract_feat(x)
        logits = self.mmseg_model.decode_head.forward(features)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint: Any) -> Tuple[Dict[str, torch.Tensor], str]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must be a mapping, got {type(checkpoint).__name__}")
    root_tensors = {str(k): v for k, v in checkpoint.items() if torch.is_tensor(v)}
    if root_tensors and len(root_tensors) / max(len(checkpoint), 1) > 0.8:
        return root_tensors, "root"
    for key in ("state_dict", "model_state_dict", "model", "net", "network", "weights"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            tensors = {str(k): v for k, v in value.items() if torch.is_tensor(v)}
            if tensors:
                return tensors, key
    raise KeyError(f"No state_dict found. Top-level keys: {list(checkpoint)[:30]}")


def strip_uniform_prefixes(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    result = dict(state)
    prefixes = ("module.", "model.", "net.", "network.")
    changed = True
    while changed and result:
        changed = False
        for prefix in prefixes:
            if all(key.startswith(prefix) for key in result):
                result = {key[len(prefix):]: value for key, value in result.items()}
                changed = True
                break
    return result


def infer_output_classes(state: Mapping[str, torch.Tensor]) -> Tuple[int, str]:
    candidates = []
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        if key.endswith("decode_head.conv_seg.weight") and value.ndim == 4:
            candidates.append((key, int(value.shape[0])))
        elif key.endswith("decode_head.conv_seg.bias") and value.ndim == 1:
            candidates.append((key, int(value.shape[0])))
    if not candidates:
        raise RuntimeError(
            "This checkpoint has no trained UPerHead output layer. It appears to "
            "be an encoder-only USFM_latest.pth, which cannot be evaluated as a "
            "segmentation model. Use best_usfm_decoder.pth instead."
        )
    classes = {count for _, count in candidates}
    if len(classes) != 1:
        raise RuntimeError(f"Conflicting output class counts: {candidates}")
    return classes.pop(), candidates[0][0]


def load_checkpoint_strict(
    model: nn.Module, raw_state: Mapping[str, torch.Tensor], path: Path
) -> Tuple[str, int, str]:
    state = strip_uniform_prefixes(raw_state)
    num_classes, output_key = infer_output_classes(state)
    model_keys = set(model.state_dict())
    candidates: List[Tuple[str, Dict[str, torch.Tensor]]] = [("as_is", dict(state))]
    if state and not any(key.startswith("mmseg_model.") for key in state):
        candidates.append(("add_mmseg_model", {f"mmseg_model.{k}": v for k, v in state.items()}))
    if any(key.startswith("mmseg_model.") for key in state):
        candidates.append(
            (
                "strip_mmseg_model",
                {key[len("mmseg_model."):]: value for key, value in state.items()
                 if key.startswith("mmseg_model.")},
            )
        )
    name, best = max(candidates, key=lambda item: len(model_keys.intersection(item[1])))
    missing = sorted(model_keys - set(best))
    unexpected = sorted(set(best) - model_keys)
    shape_mismatch = sorted(
        key for key in model_keys.intersection(best)
        if tuple(model.state_dict()[key].shape) != tuple(best[key].shape)
    )
    if missing or unexpected or shape_mismatch:
        backbone_keys = sum("backbone." in key for key in best)
        decoder_keys = sum("decode_head." in key for key in best)
        raise RuntimeError(
            f"Checkpoint is not strictly compatible: {path}\n"
            f"candidate={name}, matched={len(model_keys.intersection(best))}/{len(model_keys)}\n"
            f"backbone_keys={backbone_keys}, decode_head_keys={decoder_keys}\n"
            f"missing({len(missing)}): {missing[:15]}\n"
            f"unexpected({len(unexpected)}): {unexpected[:15]}\n"
            f"shape_mismatch({len(shape_mismatch)}): {shape_mismatch[:15]}"
        )
    model.load_state_dict(best, strict=True)
    return name, num_classes, output_key


def locate_mask(mask_dir: Path, image_path: Path) -> Path:
    candidates: List[Path] = []
    for stem in (image_path.stem, image_path.stem + "_mask"):
        candidates.extend(
            p for p in mask_dir.glob(stem + ".*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise FileNotFoundError(
            f"Expected exactly one mask for {image_path.name} in {mask_dir}; found {unique}"
        )
    return unique[0]


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_binary_mask(path: Path, allow_nonbinary: bool) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Cannot read mask: {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    unique = np.unique(mask)
    values = set(int(v) for v in unique.tolist())
    if values.issubset({0, 1}):
        return (mask == 1).astype(np.uint8)
    if values.issubset({0, 255}):
        return (mask == 255).astype(np.uint8)
    if not allow_nonbinary:
        raise ValueError(
            f"Mask {path} contains values {sorted(values)[:20]}, not binary 0/1 or 0/255. "
            "Do not evaluate raw 0/1/2 masks as prepared benchmark masks."
        )
    return (mask > 0).astype(np.uint8)


class BenchmarkDataset(Dataset):
    def __init__(self, root: Path, split: str, allow_nonbinary_masks: bool) -> None:
        self.root = root
        self.split = split
        self.image_dir = root / split / "images"
        self.mask_dir = root / split / "masks"
        self.allow_nonbinary_masks = allow_nonbinary_masks
        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Expected {self.image_dir} and {self.mask_dir}")
        images = sorted(
            p for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise RuntimeError(f"No images found in {self.image_dir}")
        self.pairs = [(image, locate_mask(self.mask_dir, image)) for image in images]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_path, mask_path = self.pairs[index]
        rgb_native = read_rgb(image_path)
        mask = read_binary_mask(mask_path, self.allow_nonbinary_masks)
        if rgb_native.shape[:2] != mask.shape:
            raise ValueError(
                f"Image/mask mismatch for {image_path.name}: "
                f"{rgb_native.shape[:2]} vs {mask.shape}"
            )
        image_224 = cv2.resize(rgb_native, (224, 224), interpolation=cv2.INTER_LINEAR)
        image = image_224.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN[None, None, :]) / IMAGENET_STD[None, None, :]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
            "name": image_path.name,
            "image_path": str(image_path),
        }


def _surface(mask: np.ndarray) -> np.ndarray:
    return np.logical_xor(
        mask, binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    )


def hd_metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    if not gt.any():
        return float("nan"), float("nan")
    if not pred.any():
        penalty = float(max(gt.shape))
        return penalty, penalty
    pred_surface, gt_surface = _surface(pred), _surface(gt)
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
    name: str, pred: np.ndarray, gt: np.ndarray, inference_ms: float, image_path: str
) -> Dict[str, Any]:
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    gt_pixels, pred_pixels = int(gt.sum()), int(pred.sum())
    dice = 2.0 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
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
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "hd": hd,
        "hd95": hd95,
        "gt_cc": gt_cc,
        "pred_cc": pred_cc,
        "cc_delta": gt_cc - pred_cc,
        "abs_cc_delta": abs(gt_cc - pred_cc),
        "inference_time_ms": float(inference_ms),
    }


SUMMARY_METRICS = (
    "dice", "iou", "precision", "recall", "specificity", "hd", "hd95",
    "cc_delta", "abs_cc_delta", "inference_time_ms",
)


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    included = [row for row in rows if int(row["metric_included"]) == 1]
    result: Dict[str, Any] = {
        "total_samples": len(rows),
        "evaluated_gt_nonempty_count": len(included),
        "gt_empty_excluded_count": len(rows) - len(included),
        "empty_prediction_count": sum(int(row["pred_pixels"]) == 0 for row in included),
    }
    result["empty_prediction_rate"] = result["empty_prediction_count"] / max(len(included), 1)
    for metric in SUMMARY_METRICS:
        source = rows if metric == "inference_time_ms" else included
        values = np.asarray([float(row[metric]) for row in source], dtype=np.float64)
        values = values[np.isfinite(values)]
        result[f"{metric}_mean"] = float(values.mean()) if values.size else float("nan")
        result[f"{metric}_std"] = float(values.std(ddof=0)) if values.size else float("nan")
        result[f"{metric}_valid_count"] = int(values.size)
    return result


def decode_logits(
    logits: torch.Tensor, mode: str, consolidation_channel: int, threshold: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(logits.float(), dim=1)
    consolidation_probability = probabilities[:, consolidation_channel]
    if mode == "argmax":
        prediction = logits.argmax(dim=1) == consolidation_channel
    else:
        prediction = consolidation_probability >= threshold
    return consolidation_probability, prediction


@torch.inference_mode()
def evaluate(
    model: nn.Module,
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
        masks = batch["mask"]
        native_size = tuple(masks.shape[-2:])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(images)
            if tuple(logits.shape[-2:]) != native_size:
                logits = F.interpolate(logits, size=native_size, mode="bilinear", align_corners=False)
            probabilities, predictions = decode_logits(logits, mode, channel, threshold)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        per_image_ms = (time.perf_counter() - start) * 1000.0 / images.shape[0]
        if not torch.isfinite(probabilities).all():
            raise FloatingPointError("USFM prediction contains NaN or Inf")
        pred_np = predictions.cpu().numpy().astype(np.uint8)
        mask_np = masks.numpy().astype(np.uint8)
        for index in range(images.shape[0]):
            rows.append(
                case_metrics(
                    batch["name"][index], pred_np[index], mask_np[index],
                    per_image_ms, batch["image_path"][index],
                )
            )
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=True)


def write_cases(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "filename", "metric_included", "gt_pixels", "pred_pixels", "dice", "iou",
        "precision", "recall", "specificity", "hd", "hd95", "gt_cc", "pred_cc",
        "cc_delta", "abs_cc_delta", "inference_time_ms",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_summary(path: Path, record: Mapping[str, Any]) -> None:
    fields = [
        "model", "size", "split", "checkpoint", "input_size", "native_eval_size",
        "decode_mode", "consolidation_channel", "threshold", "Mean-dice", "Std-dice", "Mean-iou",
        "Std-iou", "Mean-Precision", "Std-Precision", "Mean-Recall", "Std-Recall",
        "Mean-Specificity", "Std-Specificity", "empty_prediction_count", "Mean-HD",
        "Std-HD", "Mean-HD95", "Std-HD95", "HD-valid-count", "Mean-Delta-CC",
        "Std-Delta-CC", "Mean-Abs-Delta-CC", "Std-Abs-Delta-CC",
        "Efficiency-ms-image", "Efficiency-std", "N",
    ]
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def overlay(rgb: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    result = rgb.astype(np.float32).copy()
    selected = mask.astype(bool)
    result[selected] = 0.45 * result[selected] + 0.55 * np.asarray(color, dtype=np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


@torch.inference_mode()
def predict_visual(
    model: nn.Module,
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
    image = item["image"].unsqueeze(0).to(device)
    gt = item["mask"].numpy().astype(np.uint8)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        logits = model(image)
        logits = F.interpolate(logits, size=gt.shape, mode="bilinear", align_corners=False)
        _, pred = decode_logits(logits, mode, channel, threshold)
    rgb = read_rgb(Path(item["image_path"]))
    return rgb, gt, pred[0].cpu().numpy().astype(np.uint8)


def save_ranked_visuals(
    path: Path,
    title: str,
    ranked: Sequence[Mapping[str, Any]],
    model: nn.Module,
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
    fig, axes = plt.subplots(len(ranked), 3, figsize=(10, 3.2 * len(ranked)), squeeze=False)
    for row_index, row in enumerate(ranked):
        name = str(row["filename"])
        rgb, gt, pred = predict_visual(
            model, dataset, index_by_name[name], device, amp_enabled, amp_dtype,
            mode, channel, threshold,
        )
        panels = [rgb, overlay(rgb, gt, (0, 255, 0)), overlay(rgb, pred, (255, 0, 0))]
        labels = [name, "GT: consolidation (green)", f"Prediction (red), Dice={row['dice']:.4f}"]
        for column, (panel, label) in enumerate(zip(panels, labels)):
            axes[row_index, column].imshow(panel)
            axes[row_index, column].set_title(label, fontsize=9)
            axes[row_index, column].axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_record(
    size: int, split: str, checkpoint: Path, mode: str, channel: int,
    threshold: float, summary: Mapping[str, Any]
) -> Dict[str, Any]:
    return {
        "model": "USFM_UPerHead_legacy_transfer",
        "size": size,
        "split": split,
        "checkpoint": str(checkpoint),
        "input_size": 224,
        "native_eval_size": size,
        "decode_mode": mode,
        "consolidation_channel": channel,
        "threshold": threshold if mode == "threshold" else None,
        "Mean-dice": summary["dice_mean"],
        "Std-dice": summary["dice_std"],
        "Mean-iou": summary["iou_mean"],
        "Std-iou": summary["iou_std"],
        "Mean-Precision": summary["precision_mean"],
        "Std-Precision": summary["precision_std"],
        "Mean-Recall": summary["recall_mean"],
        "Std-Recall": summary["recall_std"],
        "Mean-Specificity": summary["specificity_mean"],
        "Std-Specificity": summary["specificity_std"],
        "empty_prediction_count": summary["empty_prediction_count"],
        "Mean-HD": summary["hd_mean"],
        "Std-HD": summary["hd_std"],
        "Mean-HD95": summary["hd95_mean"],
        "Std-HD95": summary["hd95_std"],
        "HD-valid-count": summary["hd_valid_count"],
        "Mean-Delta-CC": summary["cc_delta_mean"],
        "Std-Delta-CC": summary["cc_delta_std"],
        "Mean-Abs-Delta-CC": summary["abs_cc_delta_mean"],
        "Std-Abs-Delta-CC": summary["abs_cc_delta_std"],
        "Efficiency-ms-image": summary["inference_time_ms_mean"],
        "Efficiency-std": summary["inference_time_ms_std"],
        "N": summary["evaluated_gt_nonempty_count"],
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("--batch-size must be >=1 and --workers must be >=0")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.csv"
    if summary_path.exists():
        summary_path.unlink()

    checkpoint = torch_load_cpu(checkpoint_path)
    raw_state, state_container = extract_state_dict(checkpoint)
    normalized_state = strip_uniform_prefixes(raw_state)
    num_classes, inferred_output_key = infer_output_classes(normalized_state)
    if not 0 <= args.consolidation_channel < num_classes:
        raise ValueError(
            f"--consolidation-channel={args.consolidation_channel} is invalid for "
            f"checkpoint output classes={num_classes}"
        )
    if num_classes != 3 or args.consolidation_channel != 2:
        print(
            "WARNING: supplied ablation.py defines [background, B-line, consolidation] "
            "with 3 classes and consolidation channel 2; current override differs."
        )

    model = USFMModel(num_classes=num_classes)
    checkpoint_transform, _, loaded_output_key = load_checkpoint_strict(
        model, raw_state, checkpoint_path
    )
    for parameter in model.parameters():
        parameter.requires_grad = False

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    model.to(device).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    data_roots = {
        512: args.data_root_512.expanduser().resolve(),
        224: args.data_root_224.expanduser().resolve(),
    }

    print("=" * 88)
    print(f"torch={torch.__version__}, python={platform.python_version()}, device={device}")
    print(f"checkpoint={checkpoint_path}")
    print(f"checkpoint_state_container={state_container}, key_transform={checkpoint_transform}")
    print(f"num_classes={num_classes}, output_key={loaded_output_key or inferred_output_key}")
    print(f"decode_mode={args.decode_mode}, consolidation_channel={args.consolidation_channel}")
    print(f"parameters={parameter_count:,}; trainable=0 (strict zero-shot evaluation)")
    print("normalization=Albumentations/ImageNet; USFM input=224x224")
    print("=" * 88)

    for size in args.sizes:
        dataset = BenchmarkDataset(
            data_roots[size], args.split, args.allow_nonbinary_mask_values
        )
        expected = EXPECTED_TEST_COUNTS.get(size) if args.split == "test" else None
        print(f"[USFM][Size_{size}][{args.split}] paired={len(dataset)}")
        if expected is not None and len(dataset) != expected and not args.skip_count_check:
            raise RuntimeError(
                f"Expected {expected} test cases for Size_{size}, got {len(dataset)}. "
                "Use the correct benchmark root or pass --skip-count-check intentionally."
            )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )

        # Preflight before processing the complete split.
        sample = next(iter(loader))
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            sample_logits = model(sample["image"].to(device))
        if sample_logits.shape[1] != num_classes or not torch.isfinite(sample_logits).all():
            raise RuntimeError(
                f"Preflight failed: output={tuple(sample_logits.shape)}, "
                f"finite={bool(torch.isfinite(sample_logits).all())}"
            )
        del sample, sample_logits

        rows = evaluate(
            model, loader, device, amp_enabled, amp_dtype, args.decode_mode,
            args.consolidation_channel, args.threshold,
            f"USFM zero-shot Size_{size} {args.split}",
        )
        summary = summarize(rows)
        run_dir = output_root / str(size)
        run_dir.mkdir(parents=True, exist_ok=True)
        write_cases(run_dir / f"{args.split}_cases.csv", rows)
        write_json(run_dir / f"{args.split}_summary.json", summary)

        settings = {
            "model": "USFM HVITBackbone4Seg + UPerHead",
            "evaluation": "strict zero-shot transfer; no trainable parameters",
            "size": size,
            "split": args.split,
            "data_root": str(data_roots[size]),
            "checkpoint": str(checkpoint_path),
            "checkpoint_state_container": state_container,
            "checkpoint_key_transform": checkpoint_transform,
            "checkpoint_output_key": loaded_output_key,
            "num_classes": num_classes,
            "class_mapping": {"0": "background", "1": "B-line", "2": "consolidation"},
            "consolidation_channel": args.consolidation_channel,
            "decode_mode": args.decode_mode,
            "threshold": args.threshold if args.decode_mode == "threshold" else None,
            "usfm_input_size": [224, 224],
            "native_metric_size": [size, size],
            "resize_protocol": "image -> 224 bilinear; logits -> native size bilinear; then decode",
            "normalization": "ImageNet, identical to Albumentations A.Normalize() defaults",
            "mask_protocol": "prepared binary consolidation mask: 0/1 (0/255 also accepted)",
            "aggregation": "per-image macro mean/std over GT-nonempty cases",
            "hd_empty_prediction": "track side length; included in HD/HD95 mean and population std",
            "connected_components": "8-connectivity; delta = GT CC - prediction CC",
            "efficiency": "forward + native-logit resize + decode; excludes loading and metrics",
            "batch_size": args.batch_size,
            "workers": args.workers,
            "amp": amp_enabled,
            "amp_dtype": args.amp_dtype,
            "parameters_total": parameter_count,
            "parameters_trainable": 0,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
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
            f"USFM zero-shot Size_{size}: Bottom-{visual_n}",
            ranked[:visual_n], model, dataset, index_by_name, device, amp_enabled,
            amp_dtype, args.decode_mode, args.consolidation_channel, args.threshold,
        )
        save_ranked_visuals(
            run_dir / f"{args.split}_top{visual_n}_dice.png",
            f"USFM zero-shot Size_{size}: Top-{visual_n}",
            list(reversed(ranked[-visual_n:])), model, dataset, index_by_name, device,
            amp_enabled, amp_dtype, args.decode_mode, args.consolidation_channel,
            args.threshold,
        )

        record = make_record(
            size, args.split, checkpoint_path, args.decode_mode,
            args.consolidation_channel, args.threshold, summary,
        )
        append_summary(summary_path, record)
        print(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=True))

    print(f"\n[DONE] summary: {summary_path}")


if __name__ == "__main__":
    main()
