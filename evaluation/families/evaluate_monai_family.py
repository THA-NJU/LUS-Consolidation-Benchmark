#!/usr/bin/env python3
"""Evaluate the four MONAI models trained for the LUS benchmark.

Supported Pth directories:
  ../Pth/CNN/{224,512}/monai_unet
  ../Pth/CNN/{224,512}/monai_unetplusplus
  ../Pth/CNN/{224,512}/monai_attention_unet
  ../Pth/CNN/{224,512}/monai_vnet

The architecture parameters exactly follow:
  * experiments/monai/train_512.py
  * experiments/monai/train_224_filtered.py

Run this script in the MONAI training environment.  Keep
unified_native_segmentation_eval.py beside it.
"""

from __future__ import annotations

import argparse
import gc
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import unified_native_segmentation_eval as common


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_NAMES = (
    "monai_unet",
    "monai_unetplusplus",
    "monai_attention_unet",
    "monai_vnet",
)
WEIGHT_FAMILY = "CNN"
DEFAULT_WEIGHTS_ROOT = Path("../Pth")
DEFAULT_OUTPUT_ROOT = Path("../Evaluation/MONAI")
DEFAULT_DATA_ROOTS = {
    224: Path("./datasets/Size_224_filtered"),
    512: Path("./datasets/Size_512"),
}
DEFAULT_MAX_BATCH = {224: 256, 512: 64}
DEFAULT_AMP = "bf16"


class ReplicatedGrayInput(nn.Module):
    """Restore the native one-channel MONAI input used during training."""

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, image: torch.Tensor):
        if image.ndim != 4:
            raise RuntimeError(f"Expected BCHW input, got {tuple(image.shape)}")
        if image.shape[1] == 3:
            image = image.mean(dim=1, keepdim=True)
        elif image.shape[1] != 1:
            raise RuntimeError(f"Expected C=1 or C=3, got C={image.shape[1]}")
        return self.network(image)


def build_monai_model(
    model_name: str,
    image_size: int,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Build the exact architecture defined by the supplied training scripts."""

    if model_name == "monai_unet":
        from monai.networks.nets import UNet

        params: Dict[str, Any] = {
            "channels": [32, 64, 128, 256, 512],
            "strides": [2, 2, 2, 2],
            "num_res_units": 2,
        }
        network = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=tuple(params["channels"]),
            strides=tuple(params["strides"]),
            num_res_units=int(params["num_res_units"]),
        )
    elif model_name == "monai_unetplusplus":
        try:
            from monai.networks.nets import BasicUNetPlusPlus
        except ImportError as exc:
            raise RuntimeError(
                "BasicUNetPlusPlus is unavailable in this MONAI environment"
            ) from exc
        params = {
            "features": [32, 32, 64, 128, 256, 32],
            "deep_supervision": False,
        }
        network = BasicUNetPlusPlus(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            features=tuple(params["features"]),
            deep_supervision=False,
        )
    elif model_name == "monai_attention_unet":
        from monai.networks.nets import AttentionUnet

        params = {
            "channels": [32, 64, 128, 256, 512],
            "strides": [2, 2, 2, 2],
        }
        network = AttentionUnet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=tuple(params["channels"]),
            strides=tuple(params["strides"]),
        )
    elif model_name == "monai_vnet":
        from monai.networks.nets import VNet

        params = {"dropout_probability": 0.2}
        dropout = float(params["dropout_probability"])
        try:
            network = VNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=2,
                dropout_prob_down=dropout,
                dropout_prob_up=(dropout, dropout),
                dropout_dim=2,
            )
        except TypeError:
            # Compatibility with the older MONAI constructor used by some runs.
            network = VNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=2,
                dropout_prob=dropout,
                dropout_dim=2,
            )
    else:
        raise KeyError(f"Unsupported MONAI model: {model_name}")

    config = {
        "architecture": model_name,
        "img_size": int(image_size),
        "input_contract": "3 replicated grayscale channels -> mean -> 1 channel",
        "in_channels": 1,
        "num_classes": 2,
        "arch_params": params,
    }
    return ReplicatedGrayInput(network), config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
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
    amp_name = DEFAULT_AMP if args.amp_dtype == "auto" else args.amp_dtype

    print("=" * 88)
    print("MONAI native-resolution evaluation")
    print(f"GPU:          {device} ({torch.cuda.get_device_name(args.gpu)})")
    print(f"Weights:      {weights_root}")
    print(f"Output:       {output_root}")
    print(f"Sizes:        {args.sizes}")
    print(f"Splits:       {args.splits}")
    print(f"AMP:          {amp_name}")
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
                        f"No supported MONAI model directory under "
                        f"{weights_root / WEIGHT_FAMILY / str(size)}"
                    ),
                }
            )
            continue

        for model_name in model_names:
            print("\n" + "#" * 88)
            print(f"EVALUATE MONAI | size={size} | model={model_name}")
            print("#" * 88)
            try:
                checkpoint = common.find_best_checkpoint(
                    weights_root,
                    WEIGHT_FAMILY,
                    int(size),
                    model_name,
                )
                model, model_config = build_monai_model(model_name, int(size))
                summaries = common.evaluate_model(
                    model=model,
                    model_name=model_name,
                    model_config=model_config,
                    family_label="MONAI",
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
                    fixed_batch=int(args.fixed_batch),
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
            "family": "MONAI",
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
