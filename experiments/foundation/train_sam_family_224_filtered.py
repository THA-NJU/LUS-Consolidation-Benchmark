#!/usr/bin/env python3
"""
Run the existing Size_512 SAM2 / MedSAM / SAMUS benchmark implementation on
Size_224_filtered.

The existing training scripts already resize predictions back to the native
mask shape before evaluation. Therefore, when the input dataset contains
224x224 masks, validation selection and test evaluation are direct 224-patch
metrics. No 512 reconstruction is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "sam2": {
        "base_script": "train_sam2_512.py",
        "batch_size": 16,
        "grad_accum": 4,
        "train_scope": "decoder",
        "smoke_train": 4,
        "smoke_val": 2,
        "smoke_test": 2,
        "native_input": 1024,
    },
    "medsam": {
        "base_script": "train_medsam_512.py",
        "batch_size": 4,
        "grad_accum": 2,
        "train_scope": "decoder",
        "smoke_train": 4,
        "smoke_val": 2,
        "smoke_test": 2,
        "native_input": 1024,
    },
    "samus": {
        "base_script": "train_samus_512.py",
        "batch_size": 16,
        "grad_accum": 1,
        "train_scope": "native",
        "smoke_train": 8,
        "smoke_val": 2,
        "smoke_test": 2,
        "native_input": 256,
    },
}

IMAGE_SIZE = 224
THRESHOLD = 0.5
MIN_MASK_PIXELS = 1


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Final Size_224 SAM-family benchmark adapter.",
    )
    parser.add_argument("--model", required=True, choices=tuple(MODEL_DEFAULTS))
    parser.add_argument("--run-mode", choices=("smoke", "formal"), default="formal")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=(
            script_dir
            / "./datasets/Size_224_filtered"
        ),
    )
    parser.add_argument("--base-script", type=Path, default=None)
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=script_dir.parent.parent / "evaluation" / "foundation" / "evaluate_sam_checkpoints.py",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python-bin", default=sys.executable)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-scope", default=None)
    parser.add_argument(
        "--amp-dtype",
        choices=("auto", "bf16", "fp16", "none"),
        default="auto",
    )

    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model-cfg", default=None)
    parser.add_argument("--medsam-source-dir", type=Path, default=None)
    parser.add_argument("--samus-source-dir", type=Path, default=None)

    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--skip-rich-eval", action="store_true")
    return parser.parse_args()


def resolve_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def patient_id(filename: str) -> str:
    parts = Path(filename).stem.split("_")
    if len(parts) != 3 or not parts[0].lower().startswith("p"):
        raise ValueError(f"Expected pxxx_xxx_xxxx naming, got {filename!r}")
    return parts[0].lower()


def audit_size224_dataset(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    split_patients: dict[str, set[str]] = {}
    split_names: dict[str, set[str]] = {}
    split_counts: dict[str, int] = {}

    for split in ("train", "val", "test"):
        image_dir = root / split / "images"
        mask_dir = root / split / "masks"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(f"Missing {image_dir} or {mask_dir}")

        image_names = {p.name for p in image_dir.glob("*.png") if p.is_file()}
        mask_names = {p.name for p in mask_dir.glob("*.png") if p.is_file()}
        if image_names != mask_names:
            raise RuntimeError(
                f"{split}: image/mask mismatch; images={len(image_names)}, "
                f"masks={len(mask_names)}, "
                f"missing_masks={sorted(image_names-mask_names)[:10]}, "
                f"missing_images={sorted(mask_names-image_names)[:10]}"
            )
        if not image_names:
            raise RuntimeError(f"{split}: no PNG samples")

        empty: list[str] = []
        edge_touching: list[str] = []
        bad_size: list[str] = []
        for name in sorted(image_names):
            with Image.open(image_dir / name) as image:
                if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                    bad_size.append(f"{name}:image={image.size}")
            with Image.open(mask_dir / name) as mask_image:
                if mask_image.size != (IMAGE_SIZE, IMAGE_SIZE):
                    bad_size.append(f"{name}:mask={mask_image.size}")
                mask = np.asarray(mask_image.convert("L")) > 0

            if not mask.any():
                empty.append(name)
            elif (
                mask[0, :].any()
                or mask[-1, :].any()
                or mask[:, 0].any()
                or mask[:, -1].any()
            ):
                edge_touching.append(name)

        if bad_size:
            raise RuntimeError(f"{split}: non-224 samples: {bad_size[:10]}")
        if empty:
            raise RuntimeError(f"{split}: empty masks remain: {empty[:10]}")
        if edge_touching:
            raise RuntimeError(
                f"{split}: foreground touches patch edge: {edge_touching[:10]}"
            )

        split_names[split] = image_names
        split_patients[split] = {patient_id(name) for name in image_names}
        split_counts[split] = len(image_names)

    splits = ("train", "val", "test")
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            sample_overlap = split_names[left] & split_names[right]
            patient_overlap = split_patients[left] & split_patients[right]
            if sample_overlap:
                raise RuntimeError(
                    f"Sample leakage {left}/{right}: {sorted(sample_overlap)[:10]}"
                )
            if patient_overlap:
                raise RuntimeError(
                    f"Patient leakage {left}/{right}: {sorted(patient_overlap)[:10]}"
                )

    return {
        "data_root": str(root),
        "image_size": IMAGE_SIZE,
        "sample_counts": split_counts,
        "patient_counts": {
            split: len(split_patients[split]) for split in splits
        },
        "patient_overlap": False,
        "empty_masks": 0,
        "edge_touching_masks": 0,
    }


def add_path(command: list[str], flag: str, value: Path | None) -> None:
    if value is not None:
        command.extend([flag, str(value.expanduser().resolve())])


def model_specific_args(args: argparse.Namespace) -> list[str]:
    result: list[str] = []
    add_path(result, "--checkpoint", args.checkpoint)
    if args.model == "sam2" and args.model_cfg is not None:
        result.extend(["--model-cfg", args.model_cfg])
    if args.model == "medsam":
        add_path(result, "--medsam-source-dir", args.medsam_source_dir)
    if args.model == "samus":
        add_path(result, "--samus-source-dir", args.samus_source_dir)
    return result


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("\n" + "=" * 96)
    print("COMMAND:")
    print(" ".join(command))
    print("=" * 96)
    if os.environ.get('LUSBENCH_TRAINING_PROBE') == '1':
        # Keep native child training in the instrumented process for smoke checks.
        import runpy
        previous = sys.argv
        try:
            sys.argv = command[1:]
            runpy.run_path(command[1], run_name='__main__')
        finally:
            sys.argv = previous
    else:
        subprocess.run(command, check=True, env=env)


def write_compact_summary(rich_dir: Path, output_dir: Path) -> None:
    source = rich_dir / "metrics_summary.json"
    if not source.is_file():
        return
    payload = json.loads(source.read_text(encoding="utf-8"))
    test = payload["splits"]["test"]
    val = payload["splits"].get("val", {})
    compact = {
        "evaluation_unit": "individual_224_patch",
        "image_size": IMAGE_SIZE,
        "checkpoint_selection_metric": "val_mean_dice_224",
        "threshold": THRESHOLD,
        "validation": {
            "mean_dice": val.get("whole_image_dice_mean"),
            "mean_iou": val.get("whole_image_iou_mean"),
            "global_dice": val.get("global_dice"),
            "global_iou": val.get("global_iou"),
        },
        "test": {
            "mean_dice": test["whole_image_dice_mean"],
            "mean_iou": test["whole_image_iou_mean"],
            "global_dice": test["global_dice"],
            "global_iou": test["global_iou"],
            "precision_micro": test["precision_micro"],
            "recall_micro": test["recall_micro"],
            "specificity_micro": test["specificity_micro"],
            "accuracy_micro": test["accuracy_micro"],
            "precision_macro": test["precision_macro"],
            "recall_macro": test["recall_macro"],
            "specificity_macro": test["specificity_macro"],
            "dice_p25": test["whole_image_dice_p25"],
            "dice_median": test["whole_image_dice_median"],
            "dice_p75": test["whole_image_dice_p75"],
            "tp": test["tp"],
            "fp": test["fp"],
            "fn": test["fn"],
            "tn": test["tn"],
        },
        "source_metrics": str(source),
    }
    (output_dir / "size224_metrics_summary.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    row = compact["test"]
    with (output_dir / "size224_test_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    defaults = MODEL_DEFAULTS[args.model]
    script_dir = Path(__file__).resolve().parent

    if args.gpu < 0:
        raise ValueError("--gpu must be non-negative")

    data_root = args.data_root.expanduser().resolve()
    audit = audit_size224_dataset(data_root)

    base_script = (
        args.base_script
        if args.base_script is not None
        else script_dir / defaults["base_script"]
    )
    base_script = resolve_file(base_script, f"{args.model} Size_512 base script")

    evaluator = args.evaluator.expanduser().resolve()
    if not args.skip_rich_eval:
        evaluator = resolve_file(evaluator, "SAM rich metric evaluator")

    if args.output_dir is None:
        if args.resume is not None:
            output_dir = args.resume.expanduser().resolve().parent
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = (
                Path.cwd()
                / f"benchmark_{args.model}_consolidation_size224_{timestamp}"
            )
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only and args.resume is None:
        raise ValueError("--eval-only requires --resume")

    batch_size = args.batch_size or int(defaults["batch_size"])
    grad_accum = args.grad_accum or int(defaults["grad_accum"])
    train_scope = args.train_scope or str(defaults["train_scope"])

    if args.run_mode == "smoke":
        epochs = args.epochs if args.epochs is not None else 1
        warmup = args.warmup_epochs if args.warmup_epochs is not None else 0
        patience = args.patience if args.patience is not None else 1
        max_train = int(defaults["smoke_train"])
        max_val = int(defaults["smoke_val"])
        max_test = int(defaults["smoke_test"])
    else:
        epochs = args.epochs if args.epochs is not None else 600
        warmup = args.warmup_epochs if args.warmup_epochs is not None else 10
        patience = args.patience if args.patience is not None else 15
        max_train = max_val = max_test = 0

    protocol = {
        "model": args.model,
        "base_script": str(base_script),
        "data_root": str(data_root),
        "run_mode": args.run_mode,
        "dataset_audit": audit,
        "evaluation_unit": "individual_224_patch",
        "source_patch_size": IMAGE_SIZE,
        "model_native_input_size": int(defaults["native_input"]),
        "min_mask_pixels": MIN_MASK_PIXELS,
        "secondary_area_filter": False,
        "reconstruction_to_512": False,
        "threshold": THRESHOLD,
        "checkpoint_selection_metric": "val_mean_dice_224",
        "training_traversal": "full split once per epoch, no replacement",
        "positive_negative_balancing": False,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "epochs": epochs,
        "warmup_epochs": warmup,
        "patience": patience,
        "train_scope": train_scope,
    }
    (output_dir / "size224_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    specific = model_specific_args(args)
    training_env = os.environ.copy()
    training_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if not args.eval_only:
        train_command = [
            args.python_bin,
            str(base_script),
            "--data-root", str(data_root),
            "--output-dir", str(output_dir),
            "--min-mask-pixels", str(MIN_MASK_PIXELS),
            "--threshold", str(THRESHOLD),
            "--batch-size", str(batch_size),
            "--grad-accum", str(grad_accum),
            "--num-workers", str(args.num_workers),
            "--epochs", str(epochs),
            "--warmup-epochs", str(warmup),
            "--patience", str(patience),
            "--lr", str(args.lr),
            "--weight-decay", str(args.weight_decay),
            "--grad-clip", str(args.grad_clip),
            "--train-scope", train_scope,
            "--amp-dtype", args.amp_dtype,
            "--max-train-samples", str(max_train),
            "--max-val-samples", str(max_val),
            "--max-test-samples", str(max_test),
            *specific,
        ]
        if args.no_augment:
            train_command.append("--no-augment")
        if args.deterministic:
            train_command.append("--deterministic")
        if args.resume is not None:
            train_command.extend(["--resume", str(args.resume.expanduser().resolve())])
        run_command(train_command, training_env)
        best_checkpoint = output_dir / "best_model.pth"
    else:
        best_checkpoint = args.resume.expanduser().resolve()

    if not best_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {best_checkpoint}")

    if args.skip_rich_eval:
        print(f"Training completed. Rich evaluation skipped: {output_dir}")
        return

    rich_dir = output_dir / "size224_full_metrics"
    eval_forwarded = [
        "--data-root", str(data_root),
        "--resume", str(best_checkpoint),
        "--min-mask-pixels", str(MIN_MASK_PIXELS),
        "--threshold", str(THRESHOLD),
        "--batch-size", str(batch_size),
        "--num-workers", str(args.num_workers),
        "--train-scope", train_scope,
        "--amp-dtype", args.amp_dtype,
        "--max-train-samples", "0",
        "--max-val-samples", "0",
        "--max-test-samples", "0",
        *specific,
    ]
    eval_command = [
        args.python_bin,
        str(evaluator),
        "--model", args.model,
        "--benchmark-script", str(base_script),
        "--output-dir", str(rich_dir),
        "--gpu", str(args.gpu),
        "--",
        *eval_forwarded,
    ]
    run_command(eval_command, os.environ.copy())
    write_compact_summary(rich_dir, output_dir)

    print("\n" + "=" * 96)
    print(f"SIZE_224 {args.model.upper()} COMPLETE")
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Training output: {output_dir}")
    print(f"Rich metrics: {rich_dir}")
    print(f"Compact summary: {output_dir / 'size224_test_summary.csv'}")
    print("=" * 96)


if __name__ == "__main__":
    main()
