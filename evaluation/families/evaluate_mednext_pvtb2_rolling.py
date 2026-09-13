#!/usr/bin/env python3
"""Evaluate the three Size_512 APRIL additional-model routes.

Only these models are supported:
  * ../Pth/CNN/512/mednext
  * ../Pth/Transformer/512/pvtb2_emcad
  * ../Pth/other/512/rolling_unet

The model definitions and inference preprocessing exactly follow
experiments/april/train_additional_512.py. Run this script in the same
environment used for training, and keep
unified_native_segmentation_eval.py beside this file.
"""

from __future__ import annotations

import argparse
import gc
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import unified_native_segmentation_eval as common


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = 512
MODEL_NAMES = (
    "mednext",
    "pvtb2_emcad",
    "rolling_unet",
)
DEFAULT_WEIGHTS_ROOT = Path("../Pth")
DEFAULT_OUTPUT_ROOT = Path("../Evaluation/mednext_pvtb2_rolling")
DEFAULT_DATA_ROOT = Path(
    "./datasets/Size_512"
)
DEFAULT_MAX_BATCH = 64

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "mednext": {
        "weight_family": "CNN",
        "architecture": "mednext",
        "arch_params": {
            "model_id": "S",
            "kernel_size": 3,
        },
        "training_batch_size": 4,
        "amp_dtype": "fp16",
    },
    "pvtb2_emcad": {
        "weight_family": "Transformer",
        "architecture": "pvtb2_emcad",
        "arch_params": {
            "deep_supervision": False,
        },
        "training_batch_size": 2,
        "amp_dtype": "fp16",
    },
    "rolling_unet": {
        "weight_family": "other",
        "architecture": "rolling_unet",
        "arch_params": {
            "deep_supervision": False,
        },
        "training_batch_size": 4,
        "amp_dtype": "fp16",
    },
}


def dependency_preflight(april_root: Path) -> None:
    if not (april_root / "medseg" / "model_builder.py").is_file():
        raise FileNotFoundError(
            f"Not an APRIL-MedSeg repository root: {april_root}"
        )
    if str(april_root) not in sys.path:
        sys.path.insert(0, str(april_root))


def build_model(
    model_name: str,
    april_root: Path,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    if model_name not in MODEL_SPECS:
        raise KeyError(
            f"Unsupported model {model_name!r}; supported={MODEL_NAMES}"
        )
    dependency_preflight(april_root)

    from medseg.model_builder import build_model as april_build_model

    # The APRIL snapshot used for training contains the implementation but
    # does not register PVTB2-EMCAD in _SPECIAL_ARCHS.
    if model_name == "pvtb2_emcad":
        from medseg.models import networks
        from medseg.models.networks.transformer.pvtb2_emcad_model import (
            PVTB2EMCAD,
        )

        networks._SPECIAL_ARCHS.setdefault("pvtb2_emcad", PVTB2EMCAD)

    spec = MODEL_SPECS[model_name]
    model_config = {
        "model": {
            "architecture": spec["architecture"],
            "num_classes": 2,
            "img_size": IMAGE_SIZE,
            "encoder": {
                "in_channels": 3,
                "pretrained": False,
            },
            "arch_params": dict(spec["arch_params"]),
        }
    }
    return april_build_model(model_config), model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--april-root",
        type=Path,
        default=PROJECT_ROOT,
        help="APRIL-MedSeg root containing medseg/model_builder.py.",
    )
    parser.add_argument(
        "--weights-root",
        type=Path,
        default=DEFAULT_WEIGHTS_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Size_512 root containing train/val/test.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=None,
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=["train", "val", "test"],
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--amp-dtype",
        choices=("auto", "fp16", "bf16", "fp32"),
        default="auto",
        help="auto uses FP16, matching the supplied training script.",
    )
    parser.add_argument(
        "--no-auto-batch",
        action="store_true",
        help="Disable throughput-based batch-size selection.",
    )
    parser.add_argument(
        "--fixed-batch",
        type=int,
        default=1,
        help="Batch size used only with --no-auto-batch.",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=DEFAULT_MAX_BATCH,
        help="Largest batch size tested by automatic tuning.",
    )
    parser.add_argument("--batch-warmup-steps", type=int, default=3)
    parser.add_argument("--batch-timed-steps", type=int, default=10)
    parser.add_argument("--no-pin-memory", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    if args.gpu < 0:
        raise ValueError("--gpu must be non-negative")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if args.fixed_batch < 1:
        raise ValueError("--fixed-batch must be positive")
    if args.max_batch < 1:
        raise ValueError("--max-batch must be positive")
    if args.batch_warmup_steps < 1 or args.batch_timed_steps < 1:
        raise ValueError("Batch tuning step counts must be positive")


def discover_requested_models(
    weights_root: Path,
    requested: List[str] | None,
) -> List[str]:
    names = list(requested) if requested else list(MODEL_NAMES)
    selected: List[str] = []
    for name in names:
        spec = MODEL_SPECS[name]
        model_dir = (
            weights_root
            / str(spec["weight_family"])
            / str(IMAGE_SIZE)
            / name
        )
        if not model_dir.is_dir():
            if requested:
                raise FileNotFoundError(
                    f"Requested weight directory not found: {model_dir}"
                )
            print(f"[SKIP] Missing weight directory: {model_dir}")
            continue
        selected.append(name)
    return selected


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required by the efficiency protocol")
    if args.gpu >= torch.cuda.device_count():
        raise ValueError(
            f"Invalid --gpu {args.gpu}; visible GPUs="
            f"{torch.cuda.device_count()}"
        )

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    april_root = common.resolve_from(PROJECT_ROOT, args.april_root)
    weights_root = common.resolve_from(PROJECT_ROOT, args.weights_root)
    output_root = common.resolve_from(PROJECT_ROOT, args.output_root)
    data_root = common.resolve_from(PROJECT_ROOT, args.data_root)

    dependency_preflight(april_root)
    common.check_dataset_layout(data_root, args.splits)
    model_names = discover_requested_models(weights_root, args.models)
    if not model_names:
        raise FileNotFoundError(
            "None of the three supported model directories was found"
        )

    print("=" * 88)
    print("MedNeXt / PVTB2-EMCAD / Rolling U-Net Size_512 evaluation")
    print(f"GPU:          {device} ({torch.cuda.get_device_name(args.gpu)})")
    print(f"APRIL root:   {april_root}")
    print(f"Weights:      {weights_root}")
    print(f"Data:         {data_root}")
    print(f"Output:       {output_root}")
    print(f"Models:       {model_names}")
    print(f"Splits:       {args.splits}")
    print(f"AMP override: {args.amp_dtype}")
    print(f"Threshold:    {args.threshold:.2f}")
    print("=" * 88)

    all_summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for model_name in model_names:
        spec = MODEL_SPECS[model_name]
        weight_family = str(spec["weight_family"])
        print("\n" + "#" * 88)
        print(
            f"EVALUATE | family={weight_family} | "
            f"size={IMAGE_SIZE} | model={model_name}"
        )
        print("#" * 88)
        try:
            checkpoint = common.find_best_checkpoint(
                weights_root=weights_root,
                weight_family=weight_family,
                size=IMAGE_SIZE,
                model_dir_name=model_name,
            )
            model, model_config = build_model(model_name, april_root)
            amp_name = (
                str(spec["amp_dtype"])
                if args.amp_dtype == "auto"
                else str(args.amp_dtype)
            )
            fixed_batch = (
                int(args.fixed_batch)
                if args.no_auto_batch
                else int(spec["training_batch_size"])
            )
            summaries = common.evaluate_model(
                model=model,
                model_name=model_name,
                model_config=model_config,
                family_label=weight_family,
                weight_family=weight_family,
                size=IMAGE_SIZE,
                checkpoint_path=checkpoint,
                data_root=data_root,
                output_root=output_root,
                splits=args.splits,
                device=device,
                amp_name=amp_name,
                threshold=float(args.threshold),
                workers=int(args.workers),
                auto_batch=not args.no_auto_batch,
                maximum_batch=int(args.max_batch),
                fixed_batch=fixed_batch,
                warmup_steps=int(args.batch_warmup_steps),
                timed_steps=int(args.batch_timed_steps),
                pin_memory=not args.no_pin_memory,
            )
            all_summaries.extend(summaries)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            failure = {
                "size": IMAGE_SIZE,
                "family": weight_family,
                "model": model_name,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            print(
                f"[FAILED] {weight_family}/{IMAGE_SIZE}/{model_name}: "
                f"{type(exc).__name__}: {exc}"
            )
            gc.collect()
            torch.cuda.empty_cache()

    output_root.mkdir(parents=True, exist_ok=True)
    common.write_csv(output_root / "evaluation_summary.csv", all_summaries)
    common.write_csv(output_root / "evaluation_failures.csv", failures)
    common.write_json(
        output_root / "evaluation_report.json",
        {
            "models": list(MODEL_NAMES),
            "size": IMAGE_SIZE,
            "completed_model_split_rows": len(all_summaries),
            "failed_model_tasks": len(failures),
            "summaries": all_summaries,
            "failures": failures,
        },
    )

    print("\n" + "=" * 88)
    print(
        f"Completed summary rows={len(all_summaries)}, "
        f"failed tasks={len(failures)}"
    )
    print(f"Summary: {output_root / 'evaluation_summary.csv'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
