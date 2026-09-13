#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified native-resolution evaluator for YOLO26s semantic segmentation."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from segmentation_eval_common import (
    compute_case, list_pairs, read_binary_mask, read_gray, save_ranked_figure,
    summarize, write_cases, write_json, write_summary_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--size", type=int, choices=(224, 512), required=True)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--labels", type=int, nargs="+", default=[1])
    p.add_argument("--device", default="0")
    p.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
    return p.parse_args()


def semantic_class_map(result, size: int) -> np.ndarray:
    semantic_mask = getattr(result, "semantic_mask", None)
    if semantic_mask is None:
        raise RuntimeError(
            "result.semantic_mask is unavailable. Use the same Ultralytics version "
            "that trained yolo26s-sem.pt; do not substitute an instance -seg model."
        )
    data = semantic_mask.data
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    class_map = np.squeeze(np.asarray(data))
    if class_map.ndim != 2:
        raise ValueError(f"Unexpected semantic class-map shape: {class_map.shape}")
    if class_map.shape != (size, size):
        class_map = cv2.resize(class_map.astype(np.int32), (size, size), interpolation=cv2.INTER_NEAREST)
    return class_map.astype(np.int32, copy=False)


def update_visual_buffers(top, bottom, row: Mapping[str, object], payload: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
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


def evaluate_split(model: YOLO, data_root: Path, split: str, size: int, labels, batch_size: int, device: str, output_dir: Path) -> Dict[str, object]:
    pairs = list_pairs(data_root, split)
    rows: List[Dict[str, object]] = []
    batch_rows: List[Dict[str, object]] = []
    top, bottom = [], []
    # Initialize the predictor and CUDA kernels outside reported measurements.
    for _ in range(2):
        model.predict(source=str(pairs[0][0]), imgsz=size, batch=1, device=device, verbose=False, stream=False)
    for batch_index, start_index in enumerate(tqdm(range(0, len(pairs), batch_size), desc=f"YOLO26 {split}"), 1):
        chunk = pairs[start_index:start_index + batch_size]
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start = time.perf_counter()
        results = model.predict(
            source=[str(image) for image, _ in chunk], imgsz=size,
            batch=min(batch_size, len(chunk)), device=device,
            verbose=False, stream=False,
        )
        if torch.cuda.is_available(): torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - start) * 1000.0
        if len(results) != len(chunk):
            raise RuntimeError(f"Prediction count mismatch: {len(results)} != {len(chunk)}")
        reported = [float(getattr(r, "speed", {}).get("inference", float("nan"))) for r in results]
        finite = [x for x in reported if np.isfinite(x)]
        fallback = wall_ms / len(chunk)
        per_case_times = [x if np.isfinite(x) else fallback for x in reported]
        batch_rows.append({
            "batch_index": batch_index, "batch_size": len(chunk),
            "predict_wall_time_ms": wall_ms,
            "reported_inference_mean_ms_per_image": float(np.mean(finite)) if finite else fallback,
        })
        for (image_path, mask_path), result, inference_ms in zip(chunk, results, per_case_times):
            pred = (semantic_class_map(result, size) == 1).astype(np.uint8)
            gt = read_binary_mask(mask_path, labels)
            row = compute_case(image_path.name, pred, gt, inference_ms)
            rows.append(row)
            if int(row["metric_included"]) == 1:
                update_visual_buffers(top, bottom, row, (read_gray(image_path), gt, pred))
    write_cases(output_dir / f"{split}_cases.csv", rows)
    with (output_dir / f"{split}_efficiency_batches.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["batch_index", "batch_size", "predict_wall_time_ms", "reported_inference_mean_ms_per_image"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(batch_rows)
    summary = summarize(rows, split)
    write_json(output_dir / f"{split}_summary.json", summary)
    save_visuals(output_dir, split, top, bottom)
    return summary


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    data_root = args.data_root or Path(f"./datasets/Size_{args.size}{'_filtered' if args.size == 224 else ''}")
    output_dir = args.output_dir or Path(f"../Evaluation/YOLO/{args.size}/yolo26s_sem")
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = args.batch_size or (12 if args.size == 224 else 4)
    model = YOLO(str(checkpoint))
    summaries = {}
    for split in args.splits:
        summaries[split] = evaluate_split(model, data_root, split, args.size, args.labels, batch_size, args.device, output_dir)
    if set(args.splits) == {"train", "val", "test"}:
        write_summary_csv(output_dir / "summary.csv", "yolo26s_sem", args.size, summaries)
    write_json(output_dir / "evaluation_settings.json", {
        "model": "yolo26s_sem", "size": args.size, "checkpoint": str(checkpoint),
        "data_root": str(data_root), "prediction_rule": "semantic class map == 1",
        "labels": args.labels, "cc_delta": "gt_cc - pred_cc",
        "abs_cc_delta": "abs(gt_cc - pred_cc)", "connected_components": "8-neighborhood",
        "hd_units": "pixels", "empty_prediction_hd_penalty": "track side length (512 or 224 pixels)",
        "gt_empty_rule": "excluded",
        "efficiency_scope": "Ultralytics result.speed['inference']; two warmups and metrics excluded",
    })
    print(f"Finished: {output_dir}")


if __name__ == "__main__":
    main()
