#!/usr/bin/env python3
"""Retrain MONAI UNet, Attention U-Net, and VNet on the fixed Size_512 split.

This is a thin model-registration layer over the retained final Size_512
benchmark backend. Reusing that backend
keeps the dataset loading, augmentation, loss, optimizer, schedule, checkpoint
selection, metrics, and output layout identical to the current benchmark.

Place both files in the APRIL repository root, then run this file.  The MONAI
models receive the same three-channel replicated grayscale tensor as the APRIL
models; a parameter-free mean operation restores the native one-channel MONAI
input.  Therefore no image information or learned adapter is added.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn


MODEL_NAMES = (
    "monai_unet",
    "monai_attention_unet",
    "monai_vnet",
    "monai_unetplusplus"
)

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "monai_unet": {
        "architecture": "benchmark_monai_unet",
        "arch_params": {
            "channels": [32, 64, 128, 256, 512],
            "strides": [2, 2, 2, 2],
            "num_res_units": 2,
            "native_input_channels": 1,
        },
        "batch_size": 1,
        "grad_accum": 4,
    },
    "monai_attention_unet": {
        "architecture": "benchmark_monai_attention_unet",
        "arch_params": {
            "channels": [32, 64, 128, 256, 512],
            "strides": [2, 2, 2, 2],
            "native_input_channels": 1,
        },
        "batch_size": 1,
        "grad_accum": 4,
    },
    "monai_vnet": {
        "architecture": "benchmark_monai_vnet",
        "arch_params": {
            "dropout_probability": 0.2,
            "native_input_channels": 1,
        },
        "batch_size": 1,
        "grad_accum": 4,
    },
    "monai_unetplusplus": {
    "architecture": "benchmark_monai_unetplusplus",
    "arch_params": {
        "features": [32, 32, 64, 128, 256, 32],
        "native_input_channels": 1,
        "deep_supervision": False,
    },
    "batch_size": 1,
    "grad_accum": 4,
},
}


def load_protocol_runner(script_dir: Path):
    runner_path = script_dir.parent / "rwkv" / "train_512.py"
    if not runner_path.is_file():
        raise FileNotFoundError(
            "Missing finalized protocol runner in experiments/rwkv:\n"
            f"  {runner_path}\n"
            "Keep the unified repository directory structure intact."
        )
    spec = importlib.util.spec_from_file_location(
        "_size512_final_protocol_runner",
        runner_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import protocol runner: {runner_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROTOCOL = load_protocol_runner(Path(__file__).resolve().parent)


class _ReplicatedGrayInput(nn.Module):
    """Remove channel replication before a native one-channel MONAI network."""

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise RuntimeError(f"Expected BCHW input, got {tuple(image.shape)}")
        if image.shape[1] == 3:
            image = image.mean(dim=1, keepdim=True)
        elif image.shape[1] != 1:
            raise RuntimeError(
                "MONAI benchmark expects one grayscale channel or three "
                f"replicated grayscale channels, got C={image.shape[1]}"
            )
        return self.network(image)


class BenchmarkMonaiUNet(_ReplicatedGrayInput):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 512,
        pretrained: bool = False,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        strides: Sequence[int] = (2, 2, 2, 2),
        num_res_units: int = 2,
        native_input_channels: int = 1,
        **_: Any,
    ) -> None:
        del in_channels, img_size, pretrained
        if native_input_channels != 1:
            raise ValueError("This benchmark intentionally uses grayscale MONAI")
        from monai.networks.nets import UNet

        network = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=num_classes,
            channels=tuple(channels),
            strides=tuple(strides),
            num_res_units=int(num_res_units),
        )
        super().__init__(network)


class BenchmarkMonaiAttentionUNet(_ReplicatedGrayInput):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 512,
        pretrained: bool = False,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        strides: Sequence[int] = (2, 2, 2, 2),
        native_input_channels: int = 1,
        **_: Any,
    ) -> None:
        del in_channels, img_size, pretrained
        if native_input_channels != 1:
            raise ValueError("This benchmark intentionally uses grayscale MONAI")
        from monai.networks.nets import AttentionUnet

        network = AttentionUnet(
            spatial_dims=2,
            in_channels=1,
            out_channels=num_classes,
            channels=tuple(channels),
            strides=tuple(strides),
        )
        super().__init__(network)


class BenchmarkMonaiVNet(_ReplicatedGrayInput):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 512,
        pretrained: bool = False,
        dropout_probability: float = 0.2,
        native_input_channels: int = 1,
        **_: Any,
    ) -> None:
        del in_channels, img_size, pretrained
        if native_input_channels != 1:
            raise ValueError("MONAI VNet must receive one grayscale channel")
        from monai.networks.nets import VNet

        try:
            network = VNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=num_classes,
                dropout_prob_down=float(dropout_probability),
                dropout_prob_up=(
                    float(dropout_probability),
                    float(dropout_probability),
                ),
                dropout_dim=2,
            )
        except TypeError:
            network = VNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=num_classes,
                dropout_prob=float(dropout_probability),
                dropout_dim=2,
            )
        super().__init__(network)

class BenchmarkMonaiUNetPlusPlus(_ReplicatedGrayInput):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 512,
        pretrained: bool = False,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        native_input_channels: int = 1,
        deep_supervision: bool = False,
        **_: Any,
    ) -> None:
        del in_channels, img_size, pretrained

        if native_input_channels != 1:
            raise ValueError(
                "MONAI UNet++ must receive one grayscale channel"
            )

        if deep_supervision:
            raise ValueError(
                "Deep supervision is intentionally disabled."
            )

        from monai.networks.nets import BasicUNetPlusPlus

        network = BasicUNetPlusPlus(
            spatial_dims=2,
            in_channels=1,
            out_channels=num_classes,
            features=tuple(features),
            deep_supervision=False,
        )

        super().__init__(network)

def register_monai_architectures(_: str | None = None) -> None:
    """Register exact benchmark wrappers without editing APRIL itself."""

    from medseg.models import networks

    registrations = {
        "benchmark_monai_unet": BenchmarkMonaiUNet,
        "benchmark_monai_unetplusplus": BenchmarkMonaiUNetPlusPlus,
        "benchmark_monai_attention_unet": BenchmarkMonaiAttentionUNet,
        "benchmark_monai_vnet": BenchmarkMonaiVNet,
    }
    for name, model_class in registrations.items():
        existing = networks._SPECIAL_ARCHS.get(name)
        if existing is not None and existing is not model_class:
            raise RuntimeError(
                f"APRIL architecture name collision for {name!r}: {existing}"
            )
        networks._SPECIAL_ARCHS[name] = model_class


@dataclass
class ExtendedMetricAccumulator:
    """Per-image Dice/IoU plus pooled pixel-level reporting metrics."""

    threshold: float = PROTOCOL.FIXED_THRESHOLD
    count: int = 0
    dice_sum: float = 0.0
    iou_sum: float = 0.0
    precision_sum: float = 0.0
    recall_sum: float = 0.0
    specificity_sum: float = 0.0
    global_tp: int = 0
    global_fp: int = 0
    global_fn: int = 0
    global_tn: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def _ratio(
        numerator: torch.Tensor,
        denominator: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(
            denominator > 0,
            numerator.double() / denominator.clamp_min(1).double(),
            torch.zeros_like(denominator, dtype=torch.float64),
        )

    def update(
        self,
        probabilities: torch.Tensor,
        target: torch.Tensor,
        names: Iterable[str],
        store_rows: bool,
    ) -> None:
        prediction = probabilities >= self.threshold
        truth = target.bool()
        batch_size = target.shape[0]
        prediction = prediction.reshape(batch_size, -1)
        truth = truth.reshape(batch_size, -1)

        tp = (prediction & truth).sum(dim=1)
        fp = (prediction & ~truth).sum(dim=1)
        fn = (~prediction & truth).sum(dim=1)
        tn = (~prediction & ~truth).sum(dim=1)

        dice_denominator = 2 * tp + fp + fn
        union = tp + fp + fn
        dice = torch.where(
            dice_denominator > 0,
            2.0 * tp.double()
            / dice_denominator.clamp_min(1).double(),
            torch.ones_like(dice_denominator, dtype=torch.float64),
        )
        iou = torch.where(
            union > 0,
            tp.double() / union.clamp_min(1).double(),
            torch.ones_like(union, dtype=torch.float64),
        )
        precision = self._ratio(tp, tp + fp)
        recall = self._ratio(tp, tp + fn)
        specificity = self._ratio(tn, tn + fp)

        self.count += batch_size
        self.dice_sum += float(dice.sum().item())
        self.iou_sum += float(iou.sum().item())
        self.precision_sum += float(precision.sum().item())
        self.recall_sum += float(recall.sum().item())
        self.specificity_sum += float(specificity.sum().item())
        self.global_tp += int(tp.sum().item())
        self.global_fp += int(fp.sum().item())
        self.global_fn += int(fn.sum().item())
        self.global_tn += int(tn.sum().item())

        if store_rows:
            name_list = list(names)
            for index in range(batch_size):
                self.rows.append(
                    {
                        "case_name": name_list[index],
                        "dice": float(dice[index].item()),
                        "iou": float(iou[index].item()),
                        "precision": float(precision[index].item()),
                        "recall": float(recall[index].item()),
                        "specificity": float(specificity[index].item()),
                        "tp": int(tp[index].item()),
                        "fp": int(fp[index].item()),
                        "fn": int(fn[index].item()),
                        "tn": int(tn[index].item()),
                        "gt_pixels": int(truth[index].sum().item()),
                        "pred_pixels": int(prediction[index].sum().item()),
                        "threshold": self.threshold,
                    }
                )

    def summary(self) -> dict[str, Any]:
        tp = self.global_tp
        fp = self.global_fp
        fn = self.global_fn
        tn = self.global_tn
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
        return {
            "threshold": self.threshold,
            "image_count": self.count,
            "whole_image_dice_mean": self.dice_sum / max(self.count, 1),
            "whole_image_iou_mean": self.iou_sum / max(self.count, 1),
            "precision_macro": self.precision_sum / max(self.count, 1),
            "recall_macro": self.recall_sum / max(self.count, 1),
            "specificity_macro": self.specificity_sum / max(self.count, 1),
            "global_dice": (2 * tp) / max(2 * tp + fp + fn, 1),
            "global_iou": tp / max(tp + fp + fn, 1),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "accuracy": accuracy,
            "balanced_accuracy": 0.5 * (recall + specificity),
            "global_tp": tp,
            "global_fp": fp,
            "global_fn": fn,
            "global_tn": tn,
        }


def summary_rows(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        test = result["test"]
        rows.append(
            {
                "model": result["model"],
                "best_epoch": result["best_epoch"],
                "val_whole_image_dice_mean": result[
                    "best_val_whole_image_dice_mean"
                ],
                "test_whole_image_dice_mean": test[
                    "whole_image_dice_mean"
                ],
                "test_whole_image_iou_mean": test["whole_image_iou_mean"],
                "test_global_dice": test["global_dice"],
                "test_global_iou": test["global_iou"],
                "test_precision_micro": test["precision"],
                "test_recall_micro": test["recall"],
                "test_specificity_micro": test["specificity"],
                "test_accuracy_micro": test["accuracy"],
                "test_balanced_accuracy_micro": test[
                    "balanced_accuracy"
                ],
                "test_precision_macro": test["precision_macro"],
                "test_recall_macro": test["recall_macro"],
                "test_specificity_macro": test["specificity_macro"],
                "parameters_m": result["parameters_total"] / 1e6,
                "peak_memory_gib": result["peak_memory_gib"],
                "hours": result["total_seconds"] / 3600.0,
                "output_dir": result["output_dir"],
            }
        )
    rows.sort(
        key=lambda row: float(row["test_whole_image_dice_mean"]),
        reverse=True,
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Retrain MONAI UNet, Attention U-Net, and VNet with the finalized "
            "single-GPU Size_512 benchmark protocol."
        ),
    )
    parser.add_argument(
        "--april-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Required when --resume is used.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
    )
    parser.add_argument(
        "--run-mode",
        choices=tuple(PROTOCOL.PRESETS),
        default="formal",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument(
    "--max-fp32-retries-per-epoch",
    type=int,
    default=3,
    help=(
        "Maximum accumulation groups per epoch that may be recomputed "
        "in FP32 after genuinely non-finite gradients."
    ),
    )
    parser.add_argument(
    "--max-stable-norm-recoveries-per-epoch",
    type=int,
    default=3,
    help=(
        "Maximum finite-gradient groups per epoch that may use "
        "the overflow-safe stable norm recovery path."
    ),
)
    parser.add_argument(
        "--amp-dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-foreground-pixels",
        type=int,
        default=100,
        help=(
            "Minimum GT foreground pixels retained in every split. This is "
            "the finalized Size_512 dataset rule."
        ),
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def audit_fixed_split(data_root: Path) -> dict[str, Any]:
    """Fail before training if sample or patient IDs overlap across splits."""

    split_stems: dict[str, set[str]] = {}
    patient_sets: dict[str, set[str]] = {}
    split_counts: dict[str, int] = {}
    manifest_rows: list[str] = []

    for split in ("train", "val", "test"):
        image_dir = data_root / split / "images"
        mask_dir = data_root / split / "masks"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(
                f"Missing images/ or masks/ in {data_root / split}"
            )
        images = sorted(image_dir.glob("*.png"))
        masks: dict[str, Path] = {}
        for path in mask_dir.glob("*.png"):
            key = PROTOCOL.normalize_mask_key(path)
            if key in masks:
                raise RuntimeError(
                    f"Duplicate normalized mask key {key!r} in {mask_dir}"
                )
            masks[key] = path
        stems = {path.stem for path in images}
        missing_masks = sorted(stems - masks.keys())
        extra_masks = sorted(masks.keys() - stems)
        if missing_masks or extra_masks:
            raise RuntimeError(
                f"Unpaired files in {split}: missing_masks={missing_masks[:5]}, "
                f"extra_masks={extra_masks[:5]}"
            )
        split_stems[split] = stems
        patient_sets[split] = {
            stem.split("_", 1)[0]
            for stem in stems
        }
        split_counts[split] = len(stems)
        for image_path in images:
            mask_path = masks[image_path.stem]
            manifest_rows.append(
                "|".join(
                    (
                        split,
                        image_path.name,
                        str(image_path.stat().st_size),
                        mask_path.name,
                        str(mask_path.stat().st_size),
                    )
                )
            )

    overlaps: dict[str, dict[str, list[str]]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        sample_overlap = sorted(split_stems[left] & split_stems[right])
        patient_overlap = sorted(patient_sets[left] & patient_sets[right])
        if sample_overlap or patient_overlap:
            overlaps[f"{left}-{right}"] = {
                "sample_ids": sample_overlap[:50],
                "patient_ids": patient_overlap[:50],
            }
    if overlaps:
        raise RuntimeError(
            "Fixed split audit failed: cross-split overlap detected:\n"
            + json.dumps(overlaps, ensure_ascii=False, indent=2)
        )

    digest = hashlib.sha256(
        "\n".join(manifest_rows).encode("utf-8")
    ).hexdigest()
    return {
        "status": "passed",
        "data_root": str(data_root),
        "split_counts_before_foreground_filter": split_counts,
        "patient_counts": {
            split: len(values)
            for split, values in patient_sets.items()
        },
        "cross_split_sample_overlap": 0,
        "cross_split_patient_overlap": 0,
        "manifest_hash_method": (
            "SHA-256 over ordered split, filename, file-size tuples"
        ),
        "manifest_sha256": digest,
    }


def apply_overrides(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    preset = dict(PROTOCOL.PRESETS[args.run_mode])
    for key in ("epochs", "warmup_epochs", "patience"):
        value = getattr(args, key)
        if value is not None:
            preset[key] = value
    for key in (
        "max_train_samples",
        "max_val_samples",
        "max_test_samples",
    ):
        value = getattr(args, key)
        if value is not None:
            preset[key] = value
    return preset, MODEL_SPECS


def main() -> None:
    args = parse_args()
    april_root = args.april_root.expanduser().resolve()
    if not (april_root / "medseg" / "model_builder.py").is_file():
        raise FileNotFoundError(f"Not an APRIL-MedSeg root: {april_root}")
    sys.path.insert(0, str(april_root))

    try:
        import monai
    except ImportError as exc:
        raise RuntimeError(
            "MONAI is required. Install it in april_ssm with:\n"
            "  python -m pip install monai"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(
            f"Invalid --gpu {args.gpu}; visible GPUs={torch.cuda.device_count()}"
        )
    if args.min_foreground_pixels < 0:
        raise ValueError("--min-foreground-pixels must be non-negative")

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    PROTOCOL.seed_everything(args.seed, args.deterministic)
    amp_dtype = PROTOCOL.amp_dtype_from_name(args.amp_dtype, device)

    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root is not None
        else (
            april_root
            / "./datasets/Size_512"
        ).resolve()
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (
            april_root
            / "output"
            / f"consolidation_size512_monai_benchmark_{timestamp}"
        )
    )
    if args.resume and args.output_dir is None:
        raise ValueError("--resume requires an explicit --output-dir")
    output_root.mkdir(parents=True, exist_ok=True)

    preset, _ = apply_overrides(args)
    split_audit = audit_fixed_split(data_root)
    PROTOCOL.save_json(output_root / "dataset_split_audit.json", split_audit)

    PROTOCOL.MODEL_SPECS = MODEL_SPECS
    PROTOCOL.require_wkv_cuda = register_monai_architectures
    PROTOCOL.MetricAccumulator = ExtendedMetricAccumulator
    PROTOCOL.summary_rows = summary_rows
    register_monai_architectures()

    april_commit = PROTOCOL.command_output(
        ["git", "-C", str(april_root), "rev-parse", "--short", "HEAD"]
    )
    capability = torch.cuda.get_device_capability(device)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"compute capability: {capability[0]}.{capability[1]}")
    print(f"PyTorch: {torch.__version__}, MONAI: {monai.__version__}")
    print(f"APRIL root: {april_root}")
    print(f"APRIL commit: {april_commit}")
    print(f"data root: {data_root}")
    print(f"split manifest SHA-256: {split_audit['manifest_sha256']}")
    print(f"output root: {output_root}")
    print(f"models: {args.models}")
    print(f"run mode: {args.run_mode}, preset={preset}")
    print(
        "protocol: full no-replacement epoch, fixed threshold=0.5, "
        "best checkpoint selected only by mean per-image validation Dice"
    )

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for model_name in args.models:
        try:
            results.append(
                PROTOCOL.train_model(
                    model_name,
                    args=args,
                    preset=preset,
                    output_root=output_root,
                    data_root=data_root,
                    device=device,
                    amp_dtype=amp_dtype,
                    april_commit=april_commit,
                )
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            failure = {
                "model": model_name,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            model_dir = output_root / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            PROTOCOL.save_json(model_dir / "failure.json", failure)
            traceback.print_exc()
            if args.fail_fast:
                raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    if results:
        PROTOCOL.write_csv(
            output_root / "benchmark_summary.csv",
            summary_rows(results),
        )
    PROTOCOL.save_json(
        output_root / "run_report.json",
        {
            "results": results,
            "failures": failures,
            "split_audit": split_audit,
            "monai_version": monai.__version__,
            "torch_version": torch.__version__,
        },
    )
    print(
        f"Completed={len(results)}, failed={len(failures)}, "
        f"output={output_root}"
    )
    for failure in failures:
        print(f"FAILED {failure['model']}: {failure['error']}")


if __name__ == "__main__":
    main()
