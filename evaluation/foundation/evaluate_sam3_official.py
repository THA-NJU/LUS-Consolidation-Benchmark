#!/usr/bin/env python3
"""Evaluate an official-trainer SAM3 checkpoint on the LUS benchmark.

This evaluator targets checkpoints produced by ``sam3/train/train.py`` with the
provided LUSDiceTrainer YAML.  It reconstructs the official image model from
``sam3.pt``, strictly overlays the fine-tuned trainer state, and uses the
official ``Sam3Processor`` text-prompt inference path at resolution 1008.
Predictions are measured against native 224/512 masks without reconstructing or
resizing the ground truth.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import unified_native_segmentation_eval as common


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--sam3-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True, help="Official sam3.pt")
    parser.add_argument("--finetuned-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(224, 512), required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
    parser.add_argument("--prompt", default="consolidation", help="Text category used by the SAM3 training dataset")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--state-key", default="auto", help="Trainer checkpoint key, or auto")
    parser.add_argument("--visual-cases", type=int, default=5)
    return parser.parse_args()


def normalize_sample_key(path: Path) -> str:
    stem = path.stem
    if stem.lower().endswith("_mask"):
        stem = stem[:-5]
    return stem.casefold()


def resolve_pairs(split_root: Path) -> List[Tuple[Path, Path]]:
    image_dir = split_root / "images"
    mask_dir = split_root / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing images/masks below {split_root}")
    images = {
        normalize_sample_key(path): path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    masks = {
        normalize_sample_key(path): path
        for path in sorted(mask_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    if images.keys() != masks.keys():
        raise RuntimeError(
            f"Image/mask mismatch in {split_root}: "
            f"missing_masks={sorted(images.keys()-masks.keys())[:10]}, "
            f"missing_images={sorted(masks.keys()-images.keys())[:10]}"
        )
    if not images:
        raise RuntimeError(f"No samples found in {split_root}")
    return [(images[key], masks[key]) for key in sorted(images)]


def validate_native_size(pairs: Sequence[Tuple[Path, Path]], size: int) -> None:
    expected = (size, size)
    for image_path, mask_path in pairs:
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != expected or mask.size != expected:
                raise RuntimeError(
                    f"Native-size evaluation forbids resize: {image_path.name} "
                    f"image={image.size}, mask={mask.size}, expected={expected}"
                )


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        return np.asarray(handle.convert("L")) > 0


def looks_like_state_dict(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(key, str) for key in value)
        and all(torch.is_tensor(item) for item in value.values())
    )


def extract_trainer_state(payload: Any, state_key: str) -> Mapping[str, torch.Tensor]:
    if state_key != "auto":
        if not isinstance(payload, Mapping) or state_key not in payload:
            raise KeyError(f"Checkpoint has no state key {state_key!r}")
        state = payload[state_key]
        if not looks_like_state_dict(state):
            raise TypeError(f"Checkpoint[{state_key!r}] is not a state_dict")
        return state
    if looks_like_state_dict(payload):
        return payload
    if isinstance(payload, Mapping):
        for key in ("model", "model_state_dict", "state_dict", "model_ema"):
            if looks_like_state_dict(payload.get(key)):
                return payload[key]
    raise TypeError(
        "Cannot locate a full SAM3 model state. Expected model/model_state_dict/"
        "state_dict/model_ema in the official trainer checkpoint."
    )


def strip_prefix(state: Mapping[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {
        (key[len(prefix):] if key.startswith(prefix) else key): value
        for key, value in state.items()
    }


def strict_load_finetuned_state(
    model: torch.nn.Module, checkpoint: Path, state_key: str
) -> Dict[str, Any]:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    state = extract_trainer_state(payload, state_key)
    candidates: List[Tuple[str, Mapping[str, torch.Tensor]]] = [("raw", state)]
    for prefix in ("module.", "_orig_mod.", "detector."):
        if any(key.startswith(prefix) for key in state):
            candidates.append((f"strip_{prefix}", strip_prefix(state, prefix)))
    errors: List[str] = []
    for label, candidate in candidates:
        try:
            model.load_state_dict(candidate, strict=True)
            return {
                "checkpoint_payload_keys": list(payload.keys()) if isinstance(payload, Mapping) else [],
                "state_load_variant": label,
                "epoch": payload.get("epoch") if isinstance(payload, Mapping) else None,
            }
        except RuntimeError as exc:
            errors.append(f"[{label}] {exc}")
    raise RuntimeError(
        "Strict SAM3 fine-tuned checkpoint loading failed. No partial load was used.\n"
        + "\n---\n".join(errors)
    )


def amp_context(name: str):
    if name == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if name == "bf16" else torch.float16
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 requested but unsupported by this GPU")
    return torch.autocast(device_type="cuda", dtype=dtype)


def build_model_and_processor(args: argparse.Namespace):
    sam3_root = args.sam3_root.expanduser().resolve()
    if not (sam3_root / "sam3/model_builder.py").is_file():
        raise FileNotFoundError(f"Not an official SAM3 source root: {sam3_root}")
    if str(sam3_root) not in sys.path:
        sys.path.insert(0, str(sam3_root))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    bpe = sam3_root / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    if not bpe.is_file():
        raise FileNotFoundError(bpe)
    model = build_sam3_image_model(
        bpe_path=str(bpe),
        device="cpu",
        eval_mode=True,
        checkpoint_path=str(args.base_checkpoint.expanduser().resolve()),
        load_from_HF=False,
        enable_segmentation=True,
    )
    load_info = strict_load_finetuned_state(
        model,
        args.finetuned_checkpoint.expanduser().resolve(),
        args.state_key,
    )
    model = model.to(torch.device(f"cuda:{args.gpu}")).eval()
    processor = Sam3Processor(
        model,
        resolution=1008,
        device=f"cuda:{args.gpu}",
        confidence_threshold=float(args.confidence_threshold),
    )
    return model, processor, load_info


def union_prediction(output: Mapping[str, Any], size: int, mask_threshold: float) -> np.ndarray:
    masks = output.get("masks_logits")
    if masks is None:
        masks = output.get("masks")
    if masks is None or not torch.is_tensor(masks):
        raise RuntimeError("Sam3Processor output lacks masks/masks_logits tensor")
    if masks.numel() == 0 or masks.shape[0] == 0:
        return np.zeros((size, size), dtype=bool)
    if masks.dtype == torch.bool:
        binary = masks
    else:
        binary = masks >= float(mask_threshold)
    while binary.ndim > 3 and binary.shape[1] == 1:
        binary = binary[:, 0]
    if binary.ndim != 3:
        raise RuntimeError(f"Unexpected SAM3 masks shape: {tuple(binary.shape)}")
    union = binary.any(dim=0)
    if tuple(union.shape) != (size, size):
        raise RuntimeError(
            f"Official processor did not return native masks: {tuple(union.shape)}"
        )
    return union.detach().cpu().numpy().astype(bool)


@torch.inference_mode()
def evaluate_split(
    args: argparse.Namespace,
    processor,
    split: str,
    pairs: Sequence[Tuple[Path, Path]],
    output_dir: Path,
) -> Dict[str, Any]:
    case_rows: List[Dict[str, Any]] = []
    efficiency_rows: List[Dict[str, Any]] = []
    top_cases: List[common.VisualCase] = []
    bottom_cases: List[common.VisualCase] = []
    gt_empty_count = 0
    for index, (image_path, mask_path) in enumerate(
        tqdm(pairs, desc=f"SAM3 {split}", unit="image")
    ):
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        gt = load_mask(mask_path)
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        with amp_context(args.amp_dtype):
            state = processor.set_image(image)
            output = processor.set_text_prompt(state=state, prompt=args.prompt)
        pred = union_prediction(output, args.size, args.mask_threshold)
        ender.record()
        ender.synchronize()
        elapsed_s = float(starter.elapsed_time(ender)) / 1000.0
        efficiency_rows.append(
            {
                "batch_index": index,
                "batch_size": 1,
                "elapsed_s": elapsed_s,
                "ms_per_image": elapsed_s * 1000.0,
                "images_per_s": 1.0 / max(elapsed_s, 1e-12),
            }
        )
        if not gt.any():
            gt_empty_count += 1
            continue
        metrics = common.compute_case_metrics(gt, pred)
        case_rows.append(
            {
                "case_name": image_path.name,
                "patient_id": image_path.stem.split("_", 1)[0],
                **metrics,
            }
        )
        candidate = common.VisualCase(
            case_name=image_path.name,
            dice=float(metrics["dice"]),
            image_path=image_path,
            gt=gt.astype(np.uint8, copy=True),
            pred=pred.astype(np.uint8, copy=True),
        )
        common.update_visual_cases(
            top_cases, bottom_cases, candidate, args.visual_cases
        )
        del state, output

    summary = common.summarize_split(
        split=split,
        case_rows=case_rows,
        total_samples=len(pairs),
        gt_empty_count=gt_empty_count,
        efficiency_rows=efficiency_rows,
    )
    summary.update(
        {
            "batch_size": 1,
            "threshold": float(args.mask_threshold),
            "confidence_threshold": float(args.confidence_threshold),
            "text_prompt": args.prompt,
            "efficiency_timing_scope": (
                "official_Sam3Processor_GPU_set_image+set_text_prompt+mask_union; "
                "excludes_disk_io+PIL+metrics+csv+visualization"
            ),
            "std_definition": "population_std_ddof_0",
        }
    )
    common.write_csv(output_dir / f"{split}_cases.csv", case_rows)
    common.write_csv(output_dir / f"{split}_efficiency_batches.csv", efficiency_rows)
    common.write_json(output_dir / f"{split}_summary.json", summary)
    common.save_visual_montage(
        output_dir / f"{split}_top5_dice.png",
        top_cases,
        args.size,
        f"{split} top-{len(top_cases)} Dice",
    )
    common.save_visual_montage(
        output_dir / f"{split}_bottom5_dice.png",
        bottom_cases,
        args.size,
        f"{split} bottom-{len(bottom_cases)} Dice",
    )
    return summary


def main() -> None:
    args = parse_args()
    for name in ("confidence_threshold", "mask_threshold"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(f"Invalid --gpu {args.gpu}")
    torch.cuda.set_device(args.gpu)
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_root.expanduser().resolve() / str(args.size) / "sam3"
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_by_split = {
        split: resolve_pairs(data_root / split) for split in args.splits
    }
    for pairs in pairs_by_split.values():
        validate_native_size(pairs, args.size)
    model, processor, load_info = build_model_and_processor(args)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))

    # Untimed warm-up on the first selected sample.
    first_image = pairs_by_split[args.splits[0]][0][0]
    with Image.open(first_image) as handle:
        warm_image = handle.convert("RGB")
    with torch.inference_mode(), amp_context(args.amp_dtype):
        warm_state = processor.set_image(warm_image)
        _ = processor.set_text_prompt(state=warm_state, prompt=args.prompt)
    torch.cuda.synchronize()
    del warm_state

    summaries: List[Dict[str, Any]] = []
    for split in args.splits:
        summary = evaluate_split(
            args, processor, split, pairs_by_split[split], output_dir
        )
        summaries.append(
            {
                "family": "SAM",
                "weight_family": "SAM",
                "size": args.size,
                "model": "sam3",
                "checkpoint": str(args.finetuned_checkpoint.expanduser().resolve()),
                "amp_dtype": args.amp_dtype,
                "parameter_count": parameter_count,
                **summary,
            }
        )
    common.write_csv(output_dir / "summary.csv", summaries)
    common.write_json(
        output_dir / "evaluation_settings.json",
        {
            "family": "SAM",
            "model": "sam3",
            "size": args.size,
            "official_model_input_resolution": 1008,
            "native_benchmark_resolution": args.size,
            "base_checkpoint": str(args.base_checkpoint.expanduser().resolve()),
            "finetuned_checkpoint": str(args.finetuned_checkpoint.expanduser().resolve()),
            "checkpoint_load": load_info,
            "text_prompt": args.prompt,
            "confidence_threshold": args.confidence_threshold,
            "mask_threshold": args.mask_threshold,
            "parameter_count": parameter_count,
            "connected_components": "8_connectivity",
            "cc_delta": "gt_cc_minus_pred_cc",
            "abs_cc_delta": "per_image_abs_gt_cc_minus_pred_cc",
            "aggregation": "per_image_macro_mean_population_std",
            "prediction_instances": "union_after_confidence_filter",
        },
    )
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    print(f"SAM3 train/val/test metrics written to {output_dir}")


if __name__ == "__main__":
    main()
