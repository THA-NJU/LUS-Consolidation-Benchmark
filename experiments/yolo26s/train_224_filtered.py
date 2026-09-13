#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FINAL_PROTOCOL_V2 — corrected Size-224 benchmark training script.

Protocol identity: epochs=600, patience=15, warmup=10, seed=42,
batch=16, learning rate=1e-4, fixed semantic threshold=0.5, and every epoch
ranked by native-224 validation Mean Dice for checkpointing and early stopping.

This script has no positive oversampling, patch aliases, or 512 reconstruction.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# ----------------------------- CONFIG ---------------------------------
GPU_ID = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
DATA_ROOT = Path("./datasets/Size_224_filtered")
CACHE_ROOT = Path("./_yolo26_semantic_cache/Size_224_filtered")
OUTPUT_ROOT = Path("./sota_yolo26_224_filtered_runs")
MODEL_WEIGHTS = Path("./yolo26s-sem.pt")
IMAGE_SIZE = 224
MASK_LABEL_VALUES = (1,)
EPOCHS = 600
BATCH_SIZE = 16
PRED_BATCH_SIZE = 16
NUM_WORKERS = 4
MAX_SAMPLES_PER_SPLIT = None
PATIENCE = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 10
SEED = 42
FOREGROUND_CLASS = 1
REBUILD_CACHE = False
HORIZONTAL_FLIP_PROB = 0.50
BRIGHTNESS_GAIN = 0.08
TRANSLATE = 0.03
SCALE = 0.08
# ----------------------------------------------------------------------

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.models.yolo.semantic import (
    SemanticSegmentationTrainer,
    SemanticSegmentationValidator,
)

from segmentation_eval_common import IMAGE_EXTENSIONS, find_mask, read_binary_mask, read_gray, write_json


def log(text: str) -> None:
    print(text, flush=True)


def scan_filtered_split(split: str) -> List[Tuple[Path, Path]]:
    image_dir, mask_dir = DATA_ROOT / split / "images", DATA_ROOT / split / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Incomplete split: {image_dir}, {mask_dir}")
    images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if MAX_SAMPLES_PER_SPLIT is not None:
        images = images[: int(MAX_SAMPLES_PER_SPLIT)]
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")
    pairs: List[Tuple[Path, Path]] = []
    for image_path in tqdm(images, desc=f"Validate filtered {split}"):
        mask_path = find_mask(mask_dir, image_path)
        image = read_gray(image_path)
        mask = read_binary_mask(mask_path, MASK_LABEL_VALUES)
        if image.shape != (IMAGE_SIZE, IMAGE_SIZE) or mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"Expected 224x224 pair: {image_path}, {mask_path}")
        if not mask.any():
            raise ValueError(f"Size_224_filtered contains empty mask: {mask_path}")
        if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
            raise ValueError(f"Size_224_filtered foreground touches border: {mask_path}")
        pairs.append((image_path, mask_path))
    log(f"[{split}] validated {len(pairs)} filtered positive images")
    return pairs


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)


def cache_signature() -> Dict[str, object]:
    return {
        "version": 3,
        "source": str(DATA_ROOT.resolve()),
        "image_size": IMAGE_SIZE,
        "mask_label_values": list(MASK_LABEL_VALUES),
        "filtered_positive_nonborder_required": True,
        "balance_train_patches": False,
    }


def prepare_cache(pairs_by_split: Dict[str, Sequence[Tuple[Path, Path]]]) -> Path:
    manifest_path = CACHE_ROOT / "manifest.json"
    signature = cache_signature()
    if REBUILD_CACHE and CACHE_ROOT.exists():
        shutil.rmtree(CACHE_ROOT)
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        yaml_path = CACHE_ROOT / "dataset.yaml"
        if manifest.get("signature") == signature and yaml_path.is_file():
            log(f"Reusing cache: {CACHE_ROOT}")
            return yaml_path
        shutil.rmtree(CACHE_ROOT)

    for split, pairs in pairs_by_split.items():
        image_out, mask_out = CACHE_ROOT / "images" / split, CACHE_ROOT / "masks" / split
        for image_path, mask_path in tqdm(pairs, desc=f"Prepare YOLO cache {split}"):
            link_or_copy(image_path, image_out / image_path.name)
            destination = mask_out / f"{image_path.stem}.png"
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                binary = read_binary_mask(mask_path, MASK_LABEL_VALUES).astype(np.uint8)
                if not cv2.imwrite(str(destination), binary):
                    raise RuntimeError(f"Cannot write binary mask: {destination}")

    dataset_yaml = {
        "path": str(CACHE_ROOT.resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "masks_dir": "masks", "names": {0: "background", 1: "consolidation"},
    }
    yaml_path = CACHE_ROOT / "dataset.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False, allow_unicode=True)
    write_json(manifest_path, {
        "signature": signature,
        "split_counts": {k: len(v) for k, v in pairs_by_split.items()},
        "dataset_yaml": dataset_yaml,
    })
    return yaml_path


def semantic_class_map(result) -> np.ndarray:
    semantic_mask = getattr(result, "semantic_mask", None)
    if semantic_mask is None:
        raise RuntimeError("result.semantic_mask is unavailable; install the same Ultralytics YOLO26 semantic version used for training")
    data = semantic_mask.data
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    class_map = np.squeeze(np.asarray(data))
    if class_map.ndim != 2:
        raise ValueError(f"Unexpected semantic class map: {class_map.shape}")
    if class_map.shape != (IMAGE_SIZE, IMAGE_SIZE):
        class_map = cv2.resize(class_map.astype(np.int32), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    return class_map.astype(np.int32, copy=False)


class Native224MeanDiceValidator(SemanticSegmentationValidator):
    """Add per-image foreground Dice to the normal semantic validator."""

    def init_metrics(self, model) -> None:
        super().init_metrics(model)
        if len(self.names) != 2 or FOREGROUND_CLASS not in self.names:
            raise RuntimeError(
                "The fixed 0.5-equivalent rule requires exactly two classes "
                f"with foreground class {FOREGROUND_CLASS}; got names={self.names}"
            )
        self.native_224_dice: List[float] = []

    def update_metrics(self, preds, batch) -> None:
        super().update_metrics(preds, batch)
        targets = batch["semantic_mask"]
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets[:, 0]
        if preds.ndim == 4 and preds.shape[1] == 1:
            preds = preds[:, 0]
        if preds.shape != targets.shape:
            raise RuntimeError(
                f"Semantic prediction/target shape mismatch: {tuple(preds.shape)} != {tuple(targets.shape)}"
            )
        pred_fg = preds == FOREGROUND_CLASS
        target_fg = targets == FOREGROUND_CLASS
        pred_pixels = pred_fg.flatten(1).sum(1).to(torch.float64)
        target_pixels = target_fg.flatten(1).sum(1).to(torch.float64)
        if torch.any(target_pixels == 0):
            raise RuntimeError("Size_224_filtered validation unexpectedly contains an empty foreground mask")
        intersections = (pred_fg & target_fg).flatten(1).sum(1).to(torch.float64)
        dice = (2.0 * intersections) / (pred_pixels + target_pixels).clamp_min(1.0)
        self.native_224_dice.extend(float(x) for x in dice.detach().cpu().tolist())

    def get_stats(self) -> Dict[str, object]:
        stats = super().get_stats()
        values = np.asarray(self.native_224_dice, dtype=np.float64)
        if values.size == 0:
            raise RuntimeError("Native 224 validation produced no per-image Dice scores")
        mean_dice = float(values.mean())
        stats["benchmark/native_224_mean_dice"] = mean_dice
        stats["benchmark/native_224_std_dice"] = float(values.std(ddof=0))
        stats["benchmark/native_224_n"] = int(values.size)
        # BaseTrainer uses and removes this key before early stopping/checkpointing.
        stats["fitness"] = mean_dice
        return stats


class Native224MeanDiceTrainer(SemanticSegmentationTrainer):
    """Use native-224 validation Mean Dice as Ultralytics fitness."""

    def get_validator(self):
        return Native224MeanDiceValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )


def native_mean_dice(checkpoint: Path, pairs: Sequence[Tuple[Path, Path]], desc: str) -> Dict[str, float]:
    model = YOLO(str(checkpoint))
    scores: List[float] = []
    for start in tqdm(range(0, len(pairs), PRED_BATCH_SIZE), desc=desc):
        chunk = pairs[start:start + PRED_BATCH_SIZE]
        results = model.predict(
            source=[str(image) for image, _ in chunk], imgsz=IMAGE_SIZE,
            batch=min(PRED_BATCH_SIZE, len(chunk)), device=0 if torch.cuda.is_available() else "cpu",
            verbose=False, stream=False,
        )
        if len(results) != len(chunk):
            raise RuntimeError(f"Prediction count mismatch: {len(results)} != {len(chunk)}")
        for (_, mask_path), result in zip(chunk, results):
            pred = semantic_class_map(result) == FOREGROUND_CLASS
            gt = read_binary_mask(mask_path, MASK_LABEL_VALUES).astype(bool)
            inter = int(np.logical_and(pred, gt).sum())
            scores.append(2.0 * inter / max(int(pred.sum()) + int(gt.sum()), 1))
    metrics = {"mean_dice": float(np.mean(scores)), "std_dice": float(np.std(scores)), "n": len(scores)}
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def resolve_weights() -> Path:
    path = MODEL_WEIGHTS.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Local yolo26s-sem.pt not found: {path}")
    return path


SMOKE_CHECK = False


def main() -> None:
    weights = resolve_weights()
    YOLO(str(weights))
    pairs = {split: scan_filtered_split(split) for split in ("train", "val", "test")}
    yaml_path = prepare_cache(pairs)
    run_dir = OUTPUT_ROOT / f"yolo26s_sem_224_filtered_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": "yolo26s_sem", "size": IMAGE_SIZE, "data_root": str(DATA_ROOT),
        "selection_metric": "native_224_val_mean_dice", "balance_train_patches": False,
        "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "effective_batch_size": BATCH_SIZE,
        "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "warmup_epochs": WARMUP_EPOCHS,
        "patience": PATIENCE, "seed": SEED,
        "foreground_class": FOREGROUND_CLASS,
        "prediction_rule": "two-class argmax (equivalent to foreground softmax probability >= 0.5)",
        "split_counts": {k: len(v) for k, v in pairs.items()},
    }
    write_json(run_dir / "run_config.json", config)

    model = YOLO(str(weights))
    model.train(
        task="semantic", data=str(yaml_path), epochs=EPOCHS, imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE, device=0 if torch.cuda.is_available() else "cpu",
        workers=NUM_WORKERS, project=str(run_dir), name="ultralytics_train", exist_ok=True,
        pretrained=True, trainer=Native224MeanDiceTrainer,
        optimizer="AdamW", lr0=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS, patience=PATIENCE, val=True, save=True,
        save_period=-1, cache=False, amp=not SMOKE_CHECK, seed=SEED, deterministic=True,
        nbs=BATCH_SIZE if SMOKE_CHECK else 64,
        rect=False, multi_scale=0.0, cos_lr=True,
        hsv_h=0.0, hsv_s=0.0, hsv_v=BRIGHTNESS_GAIN,
        degrees=0.0, translate=TRANSLATE, scale=SCALE, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=HORIZONTAL_FLIP_PROB,
        mosaic=0.0, mixup=0.0, cutmix=0.0, plots=True, verbose=True,
    )
    trainer = getattr(model, "trainer", None)
    train_dir = Path(trainer.save_dir) if trainer is not None and getattr(trainer, "save_dir", None) else run_dir / "ultralytics_train"
    best_checkpoint = train_dir / "weights" / "best.pt"
    if not best_checkpoint.is_file():
        raise FileNotFoundError(f"Mean-Dice-selected checkpoint not found: {best_checkpoint}")
    del model
    gc.collect()
    selected = run_dir / "best_by_val_mean_dice_224_filtered.pt"
    shutil.copy2(best_checkpoint, selected)
    best_val = native_mean_dice(selected, pairs["val"], "Validate selected checkpoint")
    test_metrics = native_mean_dice(selected, pairs["test"], "Test selected checkpoint")
    write_json(run_dir / "final_metrics.json", {
        "checkpoint": str(selected), "source_checkpoint": str(best_checkpoint),
        "selection_metric": "native_224_val_mean_dice_each_epoch",
        "prediction_rule": "two-class argmax (equivalent to foreground softmax probability >= 0.5)",
        "val": best_val, "test": test_metrics,
    })
    log(f"Finished: {selected}")


if __name__ == "__main__":
    main()
