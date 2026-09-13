#!/usr/bin/env python3
"""Evaluate Size_512 SAM2, MedSAM, or SAMUS benchmark checkpoints.

This evaluator deliberately imports the corresponding training script and
reuses its dataset, preprocessing, model builder, prompt protocol, checkpoint
loader, AMP policy, and fixed threshold.  It only replaces the evaluation
accumulator so that the prediction path is unchanged while additional metrics
are reported.

The selected training script is run in its existing ``--eval-only`` mode.
Arguments after ``--`` are forwarded to that script.  Do not include
``--eval-only`` yourself; this wrapper adds it.
python tools/evaluate.py --model samus --size 512 -- \
  --model samus \
  --benchmark-script experiments/foundation/train_samus_512.py \
  --output-dir ./output/eval_samus \
  --gpu 0 \
  -- \
  --data-root ./datasets/Size_512 \
  --checkpoint pretrained/samus/sam_vit_b_01ec64.pth \
  --samus-source-dir ~/third_party/SAMUS \
  --resume ./benchmark_samus_consolidation_size512_20260722_230836/best_model.pth \
  --min-mask-pixels 100 \
  --threshold 0.5 \
  --batch-size 1 \
  --num-workers 4
Example:

    python tools/evaluate.py --model sam2 --size 512 -- \
      --model sam2 \
      --benchmark-script experiments/foundation/train_sam2_512.py \
      --output-dir output/eval_sam2 \
      -- \
      --data-root ./datasets/Size_512 \
      --checkpoint pretrained/sam2/sam2.1_hiera_base_plus.pt \
      --model-cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
      --resume output/sam2_run/best_model.pth \
      --min-mask-pixels 100 \
      --threshold 0.5 \
      --batch-size 1 \
      --num-workers 4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


MODEL_SCRIPT_DEFAULTS = {
    "sam2": "train_sam2_512.py",
    "medsam": "train_medsam_512.py",
    "samus": "train_samus_512.py",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Add IoU, Precision, Recall, Specificity, Accuracy, and confusion "
            "counts to an existing Size_512 SAM-family checkpoint evaluation."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=tuple(MODEL_SCRIPT_DEFAULTS),
    )
    parser.add_argument(
        "--benchmark-script",
        type=Path,
        default=None,
        help=(
            "Original training script used to create the checkpoint. When "
            "omitted, the model-specific filename is resolved beside this "
            "evaluator and then in the current directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Evaluation output directory; forwarded to the training script.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help=(
            "Physical GPU index exposed as cuda:0 before importing the model "
            "script. Use 0 on a single-GPU machine."
        ),
    )
    parser.add_argument(
        "forwarded",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are passed to the original benchmark script.",
    )
    args = parser.parse_args()
    forwarded = list(args.forwarded)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    return args, forwarded


def resolve_benchmark_script(
    model: str,
    explicit_path: Path | None,
) -> Path:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Benchmark script not found: {path}")
        return path

    filename = MODEL_SCRIPT_DEFAULTS[model]
    candidates = (
        Path(__file__).resolve().parents[2] / "experiments" / "foundation" / filename,
        Path.cwd() / filename,
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"Cannot find {filename}. Supply --benchmark-script explicitly."
    )


def import_benchmark_module(path: Path):
    module_name = f"_size512_{path.stem}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import benchmark script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def summarize_cases(
    model_name: str,
    split: str,
    threshold: float,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise RuntimeError(f"No rows collected for split {split!r}")

    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    tn = sum(int(row["tn"]) for row in rows)
    dice_values = [float(row["dice"]) for row in rows]
    iou_values = [float(row["iou"]) for row in rows]
    precision_values = [float(row["precision"]) for row in rows]
    recall_values = [float(row["recall"]) for row in rows]
    specificity_values = [float(row["specificity"]) for row in rows]
    empty_rows = [row for row in rows if int(row["gt_pixels"]) == 0]
    empty_false_positives = sum(
        int(row["pred_pixels"]) > 0 for row in empty_rows
    )
    image_manifest = "\n".join(
        sorted(str(row["filename"]) for row in rows)
    )

    precision_micro = safe_ratio(tp, tp + fp)
    recall_micro = safe_ratio(tp, tp + fn)
    specificity_micro = safe_ratio(tn, tn + fp)
    accuracy_micro = safe_ratio(tp + tn, tp + fp + fn + tn)
    return {
        "model": model_name,
        "split": split,
        "threshold": threshold,
        "image_count": len(rows),
        "positive_image_count": len(rows) - len(empty_rows),
        "empty_image_count": len(empty_rows),
        "whole_image_dice_mean": mean(dice_values),
        "whole_image_dice_median": percentile(dice_values, 0.5),
        "whole_image_dice_p25": percentile(dice_values, 0.25),
        "whole_image_dice_p75": percentile(dice_values, 0.75),
        "whole_image_iou_mean": mean(iou_values),
        "global_dice": safe_ratio(2 * tp, 2 * tp + fp + fn),
        "global_iou": safe_ratio(tp, tp + fp + fn),
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "specificity_micro": specificity_micro,
        "accuracy_micro": accuracy_micro,
        "balanced_accuracy_micro": 0.5
        * (recall_micro + specificity_micro),
        "precision_macro": mean(precision_values),
        "recall_macro": mean(recall_values),
        "specificity_macro": mean(specificity_values),
        "empty_fp_rate": safe_ratio(
            empty_false_positives,
            len(empty_rows),
        ),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "evaluated_filename_sha256": hashlib.sha256(
            image_manifest.encode("utf-8")
        ).hexdigest(),
    }


def dice_bin_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = [0] * 10
    for row in rows:
        dice = min(max(float(row["dice"]), 0.0), 1.0)
        counts[min(int(dice * 10), 9)] += 1
    total = max(len(rows), 1)
    return [
        {
            "dice_bin": f"{index * 10:02d}-{(index + 1) * 10:02d}%",
            "count": count,
            "fraction": count / total,
        }
        for index, count in enumerate(counts)
    ]


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_rich_evaluate(module, model_name: str, reports: dict[str, Any]):
    torch = module.torch
    functional = module.F
    numpy = module.np
    tqdm = module.tqdm

    def rich_evaluate(
        model,
        loader,
        device,
        amp,
        threshold: float,
        split: str,
        collect_cases: bool,
    ) -> tuple[float, list[dict[str, Any]]]:
        del collect_cases
        model.eval()
        original_case_rows: list[dict[str, Any]] = []
        detailed_rows: list[dict[str, Any]] = []
        dice_values: list[float] = []

        with torch.inference_mode():
            for batch in tqdm(
                loader,
                desc=f"Evaluating {split} + metrics",
                leave=False,
            ):
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                with module.autocast_context(amp):
                    logits, _ = model(images)

                # Reuse the original function so the reported Mean Dice remains
                # bit-for-bit aligned with the training script's evaluation.
                batch_dice = module.per_image_dice(
                    logits,
                    masks,
                    threshold,
                )
                resized_logits = functional.interpolate(
                    logits.float(),
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                prediction = resized_logits.sigmoid() >= threshold
                truth = masks >= 0.5

                batch_size = truth.shape[0]
                prediction_flat = prediction.reshape(batch_size, -1)
                truth_flat = truth.reshape(batch_size, -1)
                tp = (prediction_flat & truth_flat).sum(dim=1)
                fp = (prediction_flat & ~truth_flat).sum(dim=1)
                fn = (~prediction_flat & truth_flat).sum(dim=1)
                tn = (~prediction_flat & ~truth_flat).sum(dim=1)

                for index, name in enumerate(batch["name"]):
                    case_tp = int(tp[index].item())
                    case_fp = int(fp[index].item())
                    case_fn = int(fn[index].item())
                    case_tn = int(tn[index].item())
                    dice = float(batch_dice[index].item())
                    union = case_tp + case_fp + case_fn
                    iou = (
                        case_tp / union
                        if union > 0
                        else 1.0
                    )
                    precision = safe_ratio(
                        case_tp,
                        case_tp + case_fp,
                    )
                    recall = safe_ratio(
                        case_tp,
                        case_tp + case_fn,
                    )
                    specificity = safe_ratio(
                        case_tn,
                        case_tn + case_fp,
                    )
                    gt_pixels = int(truth[index].sum().item())
                    pred_pixels = int(prediction[index].sum().item())
                    dice_values.append(dice)
                    original_case_rows.append(
                        {
                            "split": split,
                            "filename": name,
                            "dice": dice,
                            "gt_pixels": gt_pixels,
                            "pred_pixels": pred_pixels,
                            "threshold": threshold,
                        }
                    )
                    detailed_rows.append(
                        {
                            "split": split,
                            "filename": name,
                            "dice": dice,
                            "iou": iou,
                            "precision": precision,
                            "recall": recall,
                            "specificity": specificity,
                            "tp": case_tp,
                            "fp": case_fp,
                            "fn": case_fn,
                            "tn": case_tn,
                            "gt_pixels": gt_pixels,
                            "pred_pixels": pred_pixels,
                            "threshold": threshold,
                        }
                    )

        if not dice_values:
            raise RuntimeError(
                f"No samples were evaluated for split {split!r}"
            )
        reports[split] = {
            "summary": summarize_cases(
                model_name,
                split,
                threshold,
                detailed_rows,
            ),
            "rows": detailed_rows,
        }
        return float(numpy.mean(dice_values)), original_case_rows

    return rich_evaluate


def ensure_forwarded_args(
    forwarded: list[str],
    output_dir: Path,
) -> list[str]:
    forbidden = {"--eval-only", "--output-dir"}
    collisions = [arg for arg in forwarded if arg in forbidden]
    if collisions:
        raise ValueError(
            "Do not place --eval-only or --output-dir after '--'; the "
            "evaluator controls them."
        )
    if "--resume" not in forwarded:
        raise ValueError(
            "Arguments after '--' must include --resume BEST_MODEL.pth"
        )
    return [
        *forwarded,
        "--output-dir",
        str(output_dir),
        "--eval-only",
    ]


def main() -> None:
    args, forwarded = parse_args()
    if args.gpu < 0:
        raise ValueError("--gpu must be non-negative")

    # The imported benchmark scripts always use cuda:0. Restrict visibility
    # before importing torch so cuda:0 maps to the requested physical GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    script_path = resolve_benchmark_script(
        args.model,
        args.benchmark_script,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    forwarded = ensure_forwarded_args(forwarded, output_dir)

    module = import_benchmark_module(script_path)
    reports: dict[str, Any] = {}
    module.evaluate = make_rich_evaluate(module, args.model, reports)

    old_argv = sys.argv
    sys.argv = [str(script_path), *forwarded]
    try:
        module.main()
    finally:
        sys.argv = old_argv

    if "test" not in reports:
        raise RuntimeError(
            "The benchmark script returned without a test evaluation."
        )

    summaries = {
        split: payload["summary"]
        for split, payload in reports.items()
    }
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "benchmark_script": str(script_path),
        "gpu_requested": args.gpu,
        "metric_conventions": {
            "main_dice": (
                "whole_image_dice_mean: Dice per image, then arithmetic mean"
            ),
            "main_iou": (
                "whole_image_iou_mean: IoU per image, then arithmetic mean"
            ),
            "micro": "pool TP/FP/FN/TN across all test pixels first",
            "macro": "compute each metric per image, then arithmetic mean",
            "undefined_precision_recall": 0.0,
            "empty_empty_dice_iou": 1.0,
        },
        "splits": summaries,
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_fields = list(next(iter(summaries.values())).keys())
    write_csv(
        output_dir / "metrics_summary.csv",
        summaries.values(),
        summary_fields,
    )
    case_fields = (
        "split",
        "filename",
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "tp",
        "fp",
        "fn",
        "tn",
        "gt_pixels",
        "pred_pixels",
        "threshold",
    )
    for split, payload in reports.items():
        rows = payload["rows"]
        write_csv(
            output_dir / f"{split}_cases_metrics.csv",
            rows,
            case_fields,
        )
        write_csv(
            output_dir / f"{split}_dice_bins.csv",
            dice_bin_rows(rows),
            ("dice_bin", "count", "fraction"),
        )

    test = summaries["test"]
    print("\n" + "=" * 76)
    print(f"MODEL: {args.model}")
    print(f"Test images: {test['image_count']}")
    print(
        "Mean Dice / Mean IoU: "
        f"{test['whole_image_dice_mean']:.6f} / "
        f"{test['whole_image_iou_mean']:.6f}"
    )
    print(
        "Global Dice / Global IoU: "
        f"{test['global_dice']:.6f} / {test['global_iou']:.6f}"
    )
    print(
        "Precision / Recall / Specificity (micro): "
        f"{test['precision_micro']:.6f} / "
        f"{test['recall_micro']:.6f} / "
        f"{test['specificity_micro']:.6f}"
    )
    print(f"Metrics written to: {output_dir}")


if __name__ == "__main__":
    main()
