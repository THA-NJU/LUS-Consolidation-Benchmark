#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final native Size-512 SegFormer-B2 benchmark training script.

Protocol identity: epochs=600, patience=15, warmup=10, seed=42,
effective batch=4, learning rate=1e-4, fixed inference threshold=0.5,
and checkpoint selection by native-512 validation mean per-image Dice.
"""
# This file does not balance images and never searches the test threshold.
from __future__ import annotations

import csv
import gc
import json
import math
import os
import random
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# ----------------------------- CONFIG ---------------------------------
GPU_ID = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
DATA_ROOT = Path("./datasets/Size_512")
OUTPUT_ROOT = Path("./sota_segformer_512_runs")
MODEL_NAME_OR_PATH = "./pretrained/segformer-b2-ade-512"
HF_LOCAL_FILES_ONLY = True
MASK_LABEL_VALUES = (1,)
IMAGE_SIZE = 512
EPOCHS = 600
BATCH_SIZE = 4
VAL_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 1
NUM_WORKERS = 4
MAX_SAMPLES_PER_SPLIT = None
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 10
PATIENCE = 15
MIN_DELTA = 1e-4
SEED = 42
PREDICTION_THRESHOLD = 0.50
HORIZONTAL_FLIP_PROB = 0.50
AMP_ENABLED = True
LOSS_W_BCE, LOSS_W_DICE, LOSS_W_TVERSKY = 0.40, 0.40, 0.20
TVERSKY_ALPHA_FP, TVERSKY_BETA_FN = 0.40, 0.60
POS_WEIGHT_CLIP = (1.0, 50.0)
# ----------------------------------------------------------------------

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation
from transformers.optimization import get_cosine_schedule_with_warmup


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
USE_AMP = bool(AMP_ENABLED and DEVICE.type == "cuda")


def log(text: str) -> None:
    print(text, flush=True)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=True)


def append_csv(path: Path, row: Dict[str, object]) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def read_gray(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Cannot read image: {path}")
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {arr.dtype}: {path}")
    return arr


def read_mask(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Cannot read mask: {path}")
    if arr.ndim == 3:
        arr = arr[..., 0]
    return np.isin(arr, np.asarray(MASK_LABEL_VALUES)).astype(np.uint8)


def find_mask(mask_dir: Path, image: Path) -> Path:
    direct = mask_dir / image.name
    if direct.is_file():
        return direct
    candidates = [p for p in mask_dir.glob(image.stem + ".*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Mask for {image.name}: found {candidates}")
    return candidates[0]


def scan_split(split: str) -> List[Tuple[Path, Path]]:
    image_dir, mask_dir = DATA_ROOT / split / "images", DATA_ROOT / split / "masks"
    images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if MAX_SAMPLES_PER_SPLIT is not None:
        images = images[: int(MAX_SAMPLES_PER_SPLIT)]
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")
    pairs: List[Tuple[Path, Path]] = []
    for image in tqdm(images, desc=f"Validate Size_512 {split}"):
        mask_path = find_mask(mask_dir, image)
        image_arr, mask = read_gray(image), read_mask(mask_path)
        if image_arr.shape != (IMAGE_SIZE, IMAGE_SIZE) or mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"Expected 512x512 pair: {image}, {mask_path}")
        if not mask.any():
            raise ValueError(f"Size_512 benchmark expects a non-empty consolidation mask: {mask_path}")
        pairs.append((image, mask_path))
    log(f"[{split}] validated {len(pairs)} positive Size_512 images")
    return pairs


def estimate_mean_std(pairs: Sequence[Tuple[Path, Path]]) -> Tuple[float, float]:
    total = total2 = 0.0
    count = 0
    for image_path, _ in tqdm(pairs, desc="Estimate train gray mean/std"):
        x = read_gray(image_path).astype(np.float64) / 255.0
        total += float(x.sum())
        total2 += float((x * x).sum())
        count += x.size
    mean = total / count
    std = math.sqrt(max(total2 / count - mean * mean, 1e-12))
    return float(mean), float(std)


class Native512Dataset(Dataset):
    def __init__(self, pairs: Sequence[Tuple[Path, Path]], mean: float, std: float, augment: bool) -> None:
        self.pairs = list(pairs)
        self.mean, self.std, self.augment = mean, max(std, 1e-6), augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]
        image, mask = read_gray(image_path), read_mask(mask_path)
        image_t = torch.from_numpy(image).float()[None].repeat(3, 1, 1) / 255.0
        mask_t = torch.from_numpy(mask).float()[None]
        if self.augment and random.random() < HORIZONTAL_FLIP_PROB:
            image_t, mask_t = torch.flip(image_t, [2]), torch.flip(mask_t, [2])
        image_t = (image_t - self.mean) / self.std
        return image_t, mask_t, image_path.name


class SegFormerBinaryWrapper(nn.Module):
    def __init__(self, base_model: str) -> None:
        super().__init__()
        if globals().get('SMOKE_RANDOM_INIT', False):
            from transformers import SegformerConfig
            config = SegformerConfig.from_pretrained(base_model, local_files_only=True)
            config.num_labels = 2
            config.id2label = {0: 'background', 1: 'consolidation'}
            config.label2id = {'background': 0, 'consolidation': 1}
            self.net = SegformerForSemanticSegmentation(config)
            return
        self.net = SegformerForSemanticSegmentation.from_pretrained(
            base_model,
            num_labels=2,
            id2label={0: "background", 1: "consolidation"},
            label2id={"background": 0, "consolidation": 1},
            ignore_mismatched_sizes=True,
            local_files_only=HF_LOCAL_FILES_ONLY or Path(base_model).is_dir(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(pixel_values=x).logits


class ConsolidationLoss(nn.Module):
    def __init__(self, pos_weight: float) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        fg_logits = logits[:, 1:2] - logits[:, 0:1]
        probs = torch.sigmoid(fg_logits)
        dims = (1, 2, 3)
        bce = F.binary_cross_entropy_with_logits(fg_logits, targets, pos_weight=self.pos_weight)
        dice = 1.0 - ((2 * (probs * targets).sum(dims) + 1e-6) / (probs.sum(dims) + targets.sum(dims) + 1e-6)).mean()
        tp = (probs * targets).sum(dims)
        fp = (probs * (1 - targets)).sum(dims)
        fn = ((1 - probs) * targets).sum(dims)
        tversky = 1.0 - ((tp + 1e-6) / (tp + TVERSKY_ALPHA_FP * fp + TVERSKY_BETA_FN * fn + 1e-6)).mean()
        return LOSS_W_BCE * bce + LOSS_W_DICE * dice + LOSS_W_TVERSKY * tversky


def amp_context():
    return torch.amp.autocast("cuda", enabled=True) if USE_AMP else nullcontext()


def make_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=USE_AMP)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=USE_AMP)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, threshold: float) -> Dict[str, float]:
    scores: List[float] = []
    model.eval()
    for images, masks, _ in tqdm(loader, desc="Native 512 validation", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        masks_np = masks.numpy().astype(bool)[:, 0]
        with amp_context():
            logits = model(images)
        logits = F.interpolate(logits.float(), size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        pred = probs >= float(threshold)
        for p, g in zip(pred, masks_np):
            inter = int(np.logical_and(p, g).sum())
            scores.append(2.0 * inter / max(int(p.sum()) + int(g.sum()), 1))
    return {
        "mean_dice": float(np.mean(scores)),
        "std_dice": float(np.std(scores)),
        "n": len(scores),
        "threshold": float(threshold),
    }


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def main() -> None:
    set_seed()
    run_dir = OUTPUT_ROOT / f"segformer_b2_512_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pairs = {split: scan_split(split) for split in ("train", "val", "test")}
    gray_mean, gray_std = estimate_mean_std(pairs["train"])
    train_ds = Native512Dataset(pairs["train"], gray_mean, gray_std, augment=True)
    val_ds = Native512Dataset(pairs["val"], gray_mean, gray_std, augment=False)
    test_ds = Native512Dataset(pairs["test"], gray_mean, gray_std, augment=False)
    loader_generator = torch.Generator().manual_seed(SEED)
    loader_kwargs = {
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.type == "cuda",
        "persistent_workers": NUM_WORKERS > 0,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_ds, BATCH_SIZE, shuffle=True, generator=loader_generator, **loader_kwargs
    )
    val_loader = DataLoader(val_ds, VAL_BATCH_SIZE, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, VAL_BATCH_SIZE, shuffle=False, **loader_kwargs)

    fg = sum(int(read_mask(m).sum()) for _, m in pairs["train"])
    pixels = len(pairs["train"]) * IMAGE_SIZE * IMAGE_SIZE
    raw_pos_weight = (pixels - fg) / max(fg, 1)
    pos_weight = float(np.clip(raw_pos_weight, *POS_WEIGHT_CLIP))
    config = {
        "model": "segformer_b2", "size": IMAGE_SIZE, "data_root": str(DATA_ROOT),
        "positive_masks_required": True, "selection_metric": "native_512_val_mean_dice",
        "model_name_or_path": MODEL_NAME_OR_PATH, "image_mean": [gray_mean] * 3,
        "image_std": [gray_std] * 3, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE, "warmup_epochs": WARMUP_EPOCHS,
        "patience": PATIENCE, "min_delta": MIN_DELTA, "seed": SEED,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "split_counts": {k: len(v) for k, v in pairs.items()}, "raw_pos_weight": raw_pos_weight,
        "clipped_pos_weight": pos_weight,
    }
    write_json(run_dir / "run_config.json", config)

    model = SegFormerBinaryWrapper(MODEL_NAME_OR_PATH).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = ConsolidationLoss(pos_weight).to(DEVICE)
    updates_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
    total_updates = max(1, EPOCHS * updates_per_epoch)
    warmup_updates = min(total_updates, WARMUP_EPOCHS * updates_per_epoch)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_updates, total_updates)
    scaler = make_scaler()
    best_score, stale = -1.0, 0
    best_path = run_dir / "best_segformer_b2_512.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = seen = 0.0
        for step, (images, masks, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch:03d}/{EPOCHS}"), 1):
            images, masks = images.to(DEVICE, non_blocking=True), masks.to(DEVICE, non_blocking=True)
            with amp_context():
                logits = model(images)
            raw_loss = loss_fn(logits.float().clamp(-30, 30), masks.float())
            if not torch.isfinite(raw_loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch}, step={step}")
            group_start = ((step - 1) // GRAD_ACCUM_STEPS) * GRAD_ACCUM_STEPS + 1
            group_end = min(group_start + GRAD_ACCUM_STEPS - 1, len(train_loader))
            accumulation_divisor = group_end - group_start + 1
            scaler.scale(raw_loss / accumulation_divisor).backward()
            if step % GRAD_ACCUM_STEPS == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            loss_sum += float(raw_loss.item()) * images.size(0)
            seen += images.size(0)
        val_metrics = evaluate(model, val_loader, PREDICTION_THRESHOLD)
        val_score = val_metrics["mean_dice"]
        append_csv(run_dir / "training_history.csv", {
            "epoch": epoch, "train_loss": loss_sum / max(seen, 1),
            "val_threshold": PREDICTION_THRESHOLD, "val_mean_dice": val_score,
            "val_std_dice": val_metrics["std_dice"],
        })
        log(f"epoch={epoch:03d} val_mean_dice={val_score:.5f} threshold={PREDICTION_THRESHOLD:.2f}")
        if val_score > best_score + MIN_DELTA:
            best_score, stale = val_score, 0
            torch.save({
                "model_state": model.state_dict(), "epoch": epoch,
                "fixed_threshold": PREDICTION_THRESHOLD,
                "val_metrics": val_metrics, "config": config,
            }, best_path)
        else:
            stale += 1
            if stale >= PATIENCE:
                log(f"Early stopping; best validation mean Dice={best_score:.5f}")
                break

    checkpoint = torch.load(best_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    threshold = float(checkpoint["fixed_threshold"])
    if not math.isclose(threshold, PREDICTION_THRESHOLD, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"Checkpoint threshold {threshold} != fixed protocol threshold {PREDICTION_THRESHOLD}")
    test_metrics = evaluate(model, test_loader, threshold)
    write_json(run_dir / "final_metrics.json", {
        "checkpoint": str(best_path), "best_epoch": checkpoint["epoch"],
        "threshold_policy": "fixed", "threshold": threshold,
        "val": checkpoint["val_metrics"], "test": test_metrics,
    })
    log(f"Finished: {best_path}")
    del model
    gc.collect()


if __name__ == "__main__":
    main()
