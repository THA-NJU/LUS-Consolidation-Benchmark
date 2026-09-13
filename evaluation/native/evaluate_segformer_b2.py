#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified native-resolution evaluator for SegFormer-B2 (224 filtered / 512)."""
from __future__ import annotations

import argparse
import csv
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from segmentation_eval_common import (
    compute_case, list_pairs, read_binary_mask, read_gray, save_ranked_figure,
    summarize, write_cases, write_json, write_summary_csv,
)

FIXED_THRESHOLD = 0.5


class EvalDataset(Dataset):
    def __init__(self, pairs: Sequence[Tuple[Path, Path]], mean: Sequence[float], std: Sequence[float], labels: Sequence[int], size: int) -> None:
        self.pairs, self.labels, self.size = list(pairs), tuple(labels), int(size)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1).clamp_min(1e-6)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]
        image = read_gray(image_path)
        gt = read_binary_mask(mask_path, self.labels)
        if image.shape != (self.size, self.size) or gt.shape != (self.size, self.size):
            raise ValueError(f"Expected {self.size}x{self.size}: {image_path}, {mask_path}")
        x = torch.from_numpy(image).float()[None].repeat(3, 1, 1) / 255.0
        x = (x - self.mean) / self.std
        return x, torch.from_numpy(gt), image_path.name


class SegFormerBinaryWrapper(nn.Module):
    def __init__(self, base_model: str, local_only: bool, *, trained_state=None) -> None:
        super().__init__()
        if trained_state is not None:
            # A complete fine-tuned state replaces every parameter and buffer.
            # Read architecture JSON only: no initialization .bin is needed.
            config_path = Path(base_model) / "config.json"
            raw_config = json.loads(config_path.read_text(encoding="utf-8"))
            if raw_config.get("model_type") != "segformer":
                raise ValueError(f"Not a SegFormer configuration: {config_path}")
            # Replace ADE20K's 150-class maps BEFORE constructing the config.
            raw_config.update(
                num_labels=2,
                id2label={0: "background", 1: "consolidation"},
                label2id={"background": 0, "consolidation": 1},
            )
            architecture = SegformerConfig.from_dict(raw_config)
            self.net = SegformerForSemanticSegmentation(architecture)
            # Any missing, unexpected or incompatible tensor stops inference.
            self.load_state_dict(normalize_state_dict(trained_state), strict=True)
            return
        self.net = SegformerForSemanticSegmentation.from_pretrained(
            base_model, num_labels=2,
            id2label={0: "background", 1: "consolidation"},
            label2id={"background": 0, "consolidation": 1},
            ignore_mismatched_sizes=True, local_files_only=local_only or Path(base_model).is_dir(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(pixel_values=x).logits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--size", type=int, choices=(224, 512), required=True)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--base-model", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--labels", type=int, nargs="+", default=[1])
    p.add_argument("--amp-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
    return p.parse_args()


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "fp32":
        return nullcontext()
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    return torch.amp.autocast("cuda", dtype=dtype)


def normalize_state_dict(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = dict(state)
    if out and all(k.startswith("module.") for k in out):
        out = {k[7:]: v for k, v in out.items()}
    if out and not any(k.startswith("net.") for k in out) and any(k.startswith("segformer.") or k.startswith("decode_head.") for k in out):
        out = {"net." + k: v for k, v in out.items()}
    return out


def update_visual_buffers(top: List[Tuple[float, Mapping[str, object], Tuple[np.ndarray, np.ndarray, np.ndarray]]], bottom: List[Tuple[float, Mapping[str, object], Tuple[np.ndarray, np.ndarray, np.ndarray]]], row: Mapping[str, object], payload: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    if int(row["metric_included"]) != 1:
        return
    score = float(row["dice"])
    top.append((score, row, payload)); top.sort(key=lambda x: x[0], reverse=True); del top[5:]
    bottom.append((score, row, payload)); bottom.sort(key=lambda x: x[0]); del bottom[5:]


def save_visuals(output_dir: Path, split: str, top, bottom) -> None:
    for suffix, entries, title in (("top5", top, "Top-5 Dice"), ("bottom5", bottom, "Bottom-5 Dice")):
        rows = [x[1] for x in entries]
        cache = {str(x[1]["filename"]): x[2] for x in entries}
        save_ranked_figure(output_dir / f"{split}_{suffix}_dice.png", rows, cache, f"{split}: {title}")


@torch.inference_mode()
def evaluate_split(model: nn.Module, data_root: Path, split: str, size: int, mean: Sequence[float], std: Sequence[float], labels: Sequence[int], threshold: float, batch_size: int, workers: int, device: torch.device, amp_dtype: str, output_dir: Path) -> Dict[str, object]:
    pairs = list_pairs(data_root, split)
    dataset = EvalDataset(pairs, mean, std, labels, size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    model.eval()
    # Exclude one-time CUDA/cuDNN initialization from efficiency statistics.
    warmup = dataset[0][0][None].to(device)
    for _ in range(2):
        with autocast_context(device, amp_dtype):
            _ = model(warmup)
    if device.type == "cuda": torch.cuda.synchronize(device)
    del warmup
    rows: List[Dict[str, object]] = []
    top, bottom = [], []
    batch_rows: List[Dict[str, object]] = []
    for batch_index, (images, masks, names) in enumerate(tqdm(loader, desc=f"SegFormer {split}"), 1):
        images = images.to(device, non_blocking=True)
        if device.type == "cuda": torch.cuda.synchronize(device)
        start = time.perf_counter()
        with autocast_context(device, amp_dtype):
            logits = model(images)
            logits = F.interpolate(logits.float(), size=(size, size), mode="bilinear", align_corners=False)
            probs = torch.softmax(logits, dim=1)[:, 1]
        if device.type == "cuda": torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        per_image_ms = elapsed_ms / len(names)
        batch_rows.append({"batch_index": batch_index, "batch_size": len(names), "inference_time_ms": elapsed_ms, "ms_per_image": per_image_ms})
        preds = (probs.cpu().numpy() >= threshold).astype(np.uint8)
        gt_batch = masks.numpy().astype(np.uint8)
        for name, pred, gt in zip(names, preds, gt_batch):
            row = compute_case(str(name), pred, gt, per_image_ms)
            rows.append(row)
            if int(row["metric_included"]) == 1:
                image = read_gray(data_root / split / "images" / str(name))
                update_visual_buffers(top, bottom, row, (image, gt, pred))
    write_cases(output_dir / f"{split}_cases.csv", rows)
    with (output_dir / f"{split}_efficiency_batches.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["batch_index", "batch_size", "inference_time_ms", "ms_per_image"]); writer.writeheader(); writer.writerows(batch_rows)
    summary = summarize(rows, split)
    write_json(output_dir / f"{split}_summary.json", summary)
    save_visuals(output_dir, split, top, bottom)
    return summary


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    base_model = args.base_model or config.get("model_name_or_path")
    if not base_model:
        raise ValueError("Base SegFormer directory is unknown; pass --base-model")
    mean = config.get("image_mean", [0.10063751267950466] * 3)
    std = config.get("image_std", [0.14586819260714984] * 3)
    threshold = FIXED_THRESHOLD
    data_root = args.data_root or Path(f"./datasets/Size_{args.size}{'_filtered' if args.size == 224 else ''}")
    output_dir = args.output_dir or Path(f"../Evaluation/SegFormer/{args.size}/segformer_b2")
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = args.batch_size or (16 if args.size == 224 else 4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state = checkpoint.get("model_state", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint does not contain a state dict")
    model = SegFormerBinaryWrapper(
        str(base_model), args.local_files_only, trained_state=state,
    ).to(device)
    summaries = {}
    for split in args.splits:
        summaries[split] = evaluate_split(model, data_root, split, args.size, mean, std, args.labels, threshold, batch_size, args.num_workers, device, args.amp_dtype, output_dir)
    if set(args.splits) == {"train", "val", "test"}:
        write_summary_csv(output_dir / "summary.csv", "segformer_b2", args.size, summaries)
    write_json(output_dir / "evaluation_settings.json", {
        "model": "segformer_b2", "size": args.size, "checkpoint": str(checkpoint_path),
        "base_model": str(base_model), "data_root": str(data_root), "threshold": threshold,
        "image_mean": mean, "image_std": std, "labels": args.labels,
        "cc_delta": "gt_cc - pred_cc", "abs_cc_delta": "abs(gt_cc - pred_cc)",
        "connected_components": "8-neighborhood", "hd_units": "pixels",
        "empty_prediction_hd_penalty": "track side length (512 or 224 pixels)", "gt_empty_rule": "excluded",
        "efficiency_scope": "model forward + logit resize + softmax; two warmups and metrics excluded",
    })
    print(f"Finished: {output_dir}")


if __name__ == "__main__":
    main()
