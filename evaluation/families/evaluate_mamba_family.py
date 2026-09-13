#!/usr/bin/env python3
"""Evaluate the four Mamba models trained for the LUS benchmark.

Supported Pth directories:
  ../Pth/Mamba/{224,512}/mamba_unet
  ../Pth/Mamba/{224,512}/nnmamba_2d
  ../Pth/Mamba/{224,512}/swin_umamba
  ../Pth/Mamba/{224,512}/vm_unet_v2

Architecture parameters exactly follow:
  * experiments/mamba/train_512.py
  * experiments/monai/train_224_filtered.py

Run this script from the APRIL-MedSeg repository in the same Mamba environment
used for training.  Keep unified_native_segmentation_eval.py beside it.
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
MODEL_NAMES = (
    "mamba_unet",
    "nnmamba_2d",
    "swin_umamba",
    "vm_unet_v2",
)
WEIGHT_FAMILY = "Mamba"
DEFAULT_WEIGHTS_ROOT = Path("../Pth")
DEFAULT_OUTPUT_ROOT = Path("../Evaluation/Mamba")
DEFAULT_DATA_ROOTS = {
    224: Path("./datasets/Size_224_filtered"),
    512: Path("./datasets/Size_512"),
}
DEFAULT_MAX_BATCH = {224: 256, 512: 64}
DEFAULT_AMP = "bf16"

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "mamba_unet": {
        "architecture": "mamba_unet",
        "arch_params": {
            "embed_dim": 96,
            "depths": [2, 2, 2, 2],
            "d_state": 16,
            "drop_path_rate": 0.1,
            "deep_supervision": False,
        },
        "training_batch_size": {224: 1, 512: 1},
        "amp_dtype": "bf16",
    },
    "nnmamba_2d": {
        "architecture": "nnmamba_2d",
        "arch_params": {
            "channels": 32,
            "blocks": 3,
            "deep_supervision": False,
        },
        "training_batch_size": {224: 16, 512: 1},
        "amp_dtype": "bf16",
    },
    "swin_umamba": {
        "architecture": "swin_umamba",
        "arch_params": {
            "feat_size": [48, 96, 192, 384, 768],
            "depths": [2, 2, 2, 2],
            "d_state": 16,
            "drop_path_rate": 0.2,
            "deep_supervision": False,
        },
        "training_batch_size": {224: 1, 512: 1},
        "amp_dtype": "bf16",
    },
    "vm_unet_v2": {
        "architecture": "vm_unet_v2",
        "arch_params": {
            "embed_dim": 64,
            "depths": [2, 2, 6, 2],
            "mid_channel": 32,
            "drop_path_rate": 0.2,
            "deep_supervision": True,
        },
        "training_batch_size": {224: 16, 512: 1},
        "amp_dtype": "bf16",
    },
}


def dependency_preflight(april_root: Path) -> None:
    if not (april_root / "medseg" / "model_builder.py").is_file():
        raise FileNotFoundError(f"Not an APRIL-MedSeg root: {april_root}")
    if str(april_root) not in sys.path:
        sys.path.insert(0, str(april_root))
    try:
        import selective_scan_cuda  # noqa: F401
        from mamba_ssm.ops.selective_scan_interface import (  # noqa: F401
            selective_scan_fn,
        )
    except Exception as exc:
        raise RuntimeError(
            "The CUDA-enabled mamba_ssm/selective_scan environment used for "
            "training is required for Mamba evaluation"
        ) from exc


def build_mamba_model(
    model_name: str,
    image_size: int,
    april_root: Path,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    if model_name not in MODEL_SPECS:
        raise KeyError(f"Unsupported Mamba model: {model_name}")
    dependency_preflight(april_root)
    from medseg.model_builder import build_model

    spec = MODEL_SPECS[model_name]
    model_config = {
        "model": {
            "architecture": spec["architecture"],
            "num_classes": 2,
            "img_size": int(image_size),
            "encoder": {
                "in_channels": 3,
                "pretrained": False,
            },
            "arch_params": dict(spec["arch_params"]),
        }
    }
    return build_model(model_config), model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--april-root",
        type=Path,
        default=PROJECT_ROOT,
        help="APRIL-MedSeg repository root containing medseg/model_builder.py.",
    )
    parser.add_argument("--weights-root", type=Path, default=DEFAULT_WEIGHTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root-224", type=Path, default=DEFAULT_DATA_ROOTS[224])
    parser.add_argument("--data-root-512", type=Path, default=DEFAULT_DATA_ROOTS[512])
    parser.add_argument("--sizes", type=int, nargs="+", choices=(224, 512), default=[224, 512])
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=None)
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
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
        help="auto uses each supplied training specification (BF16 here).",
    )
    parser.add_argument("--no-auto-batch", action="store_true")
    parser.add_argument(
        "--fixed-batch",
        type=int,
        default=1,
        help="Used only with --no-auto-batch.",
    )
    parser.add_argument("--max-batch-224", type=int, default=DEFAULT_MAX_BATCH[224])
    parser.add_argument("--max-batch-512", type=int, default=DEFAULT_MAX_BATCH[512])
    parser.add_argument("--batch-warmup-steps", type=int, default=3)
    parser.add_argument("--batch-timed-steps", type=int, default=10)
    parser.add_argument("--no-pin-memory", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    if args.fixed_batch < 1:
        raise ValueError("--fixed-batch must be positive")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if args.batch_warmup_steps < 1 or args.batch_timed_steps < 1:
        raise ValueError("Batch tuning step counts must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required by the efficiency protocol")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(
            f"Invalid --gpu {args.gpu}; visible GPUs={torch.cuda.device_count()}"
        )

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    april_root = common.resolve_from(PROJECT_ROOT, args.april_root)
    weights_root = common.resolve_from(PROJECT_ROOT, args.weights_root)
    output_root = common.resolve_from(PROJECT_ROOT, args.output_root)
    data_roots = {
        224: common.resolve_from(PROJECT_ROOT, args.data_root_224),
        512: common.resolve_from(PROJECT_ROOT, args.data_root_512),
    }
    max_batches = {
        224: int(args.max_batch_224),
        512: int(args.max_batch_512),
    }
    dependency_preflight(april_root)

    print("=" * 88)
    print("Mamba native-resolution evaluation")
    print(f"GPU:          {device} ({torch.cuda.get_device_name(args.gpu)})")
    print(f"APRIL root:   {april_root}")
    print(f"Weights:      {weights_root}")
    print(f"Output:       {output_root}")
    print(f"Sizes:        {args.sizes}")
    print(f"Splits:       {args.splits}")
    print(f"AMP override: {args.amp_dtype}")
    print(f"Threshold:    {args.threshold:.2f}")
    print("=" * 88)

    all_summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for size in args.sizes:
        data_root = data_roots[int(size)]
        common.check_dataset_layout(data_root, args.splits)
        model_names = common.discover_model_directories(
            weights_root=weights_root,
            weight_family=WEIGHT_FAMILY,
            size=int(size),
            supported_names=MODEL_NAMES,
            requested=args.models,
        )
        if not model_names:
            failures.append(
                {
                    "size": int(size),
                    "model": None,
                    "error_type": "NoSupportedModelDirectory",
                    "error": (
                        f"No supported Mamba model directory under "
                        f"{weights_root / WEIGHT_FAMILY / str(size)}"
                    ),
                }
            )
            continue

        for model_name in model_names:
            print("\n" + "#" * 88)
            print(f"EVALUATE Mamba | size={size} | model={model_name}")
            print("#" * 88)
            try:
                checkpoint = common.find_best_checkpoint(
                    weights_root,
                    WEIGHT_FAMILY,
                    int(size),
                    model_name,
                )
                model, model_config = build_mamba_model(
                    model_name,
                    int(size),
                    april_root,
                )
                spec = MODEL_SPECS[model_name]
                amp_name = (
                    str(spec["amp_dtype"])
                    if args.amp_dtype == "auto"
                    else args.amp_dtype
                )
                fixed_batch = (
                    int(args.fixed_batch)
                    if args.no_auto_batch
                    else int(spec["training_batch_size"][int(size)])
                )
                summaries = common.evaluate_model(
                    model=model,
                    model_name=model_name,
                    model_config=model_config,
                    family_label="Mamba",
                    weight_family=WEIGHT_FAMILY,
                    size=int(size),
                    checkpoint_path=checkpoint,
                    data_root=data_root,
                    output_root=output_root,
                    splits=args.splits,
                    device=device,
                    amp_name=amp_name,
                    threshold=float(args.threshold),
                    workers=int(args.workers),
                    auto_batch=not args.no_auto_batch,
                    maximum_batch=max_batches[int(size)],
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
                    "size": int(size),
                    "model": model_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                print(f"[FAILED] {size}/{model_name}: {type(exc).__name__}: {exc}")
                gc.collect()
                torch.cuda.empty_cache()

    output_root.mkdir(parents=True, exist_ok=True)
    common.write_csv(output_root / "evaluation_summary.csv", all_summaries)
    common.write_csv(output_root / "evaluation_failures.csv", failures)
    common.write_json(
        output_root / "evaluation_report.json",
        {
            "family": "Mamba",
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
