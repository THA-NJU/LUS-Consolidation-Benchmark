#!/usr/bin/env python3
"""Evaluate U-RWKV and RWKV-UNet at native 224/512 resolution.

Each model/size combination is evaluated in an isolated Python process because
the two APRIL implementations register the same ``wkv`` Torch namespace.
Keep ``unified_native_segmentation_eval.py`` beside this script and run from the
APRIL-MedSeg environment used for RWKV training.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import signal
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import unified_native_segmentation_eval as common


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_NAMES = ("u_rwkv", "rwkv_unet")
WEIGHT_FAMILY = "RWKV"
DEFAULT_WEIGHTS_ROOT = Path("../Pth")
DEFAULT_OUTPUT_ROOT = Path("../Evaluation/RWKV")
DEFAULT_DATA_ROOTS = {
    224: Path("./datasets/Size_224_filtered"),
    512: Path("./datasets/Size_512"),
}
DEFAULT_MAX_BATCH = {224: 128, 512: 16}

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "rwkv_unet": {
        "architecture": "rwkv_unet",
        "arch_params": {"variant": "b"},
        "training_batch_size": {224: 32, 512: 1},
        "kernel_tmax": {224: 1024, 512: 8192},
    },
    "u_rwkv": {
        "architecture": "u_rwkv",
        "arch_params": {
            "embed_dims": [64, 128, 256, 512],
            "depths": [2, 2, 2, 2],
            "shift_pixel": 1,
            "se_ratio": 0.25,
            "deep_supervision": False,
        },
        "training_batch_size": {224: 32, 512: 1},
        "kernel_tmax": {224: 16384, 512: 65536},
    },
}


def check_rwkv_environment(model_name: str, april_root: Path) -> None:
    required = (
        april_root / "medseg/kernels/wkv/__init__.py",
        april_root / "medseg/kernels/wkv/wkv_op.cpp",
        april_root / "medseg/kernels/wkv/wkv_cuda.cu",
        april_root / "medseg/models/networks/rwkv/rwkv_unet.py",
        april_root / "medseg/models/networks/rwkv/u_rwkv.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing APRIL RWKV files:\n  " + "\n  ".join(missing))
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None or shutil.which("nvcc") is None:
        raise RuntimeError("RWKV evaluation requires the CUDA toolkit and NVCC")
    if shutil.which("ninja") is None and importlib.util.find_spec("ninja") is None:
        raise RuntimeError("RWKV evaluation requires Ninja")
    capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{capability[0]}.{capability[1]}")
    os.environ.setdefault("MAX_JOBS", "4")
    if str(april_root) not in sys.path:
        sys.path.insert(0, str(april_root))
    print(f"[RWKV] isolated worker: {model_name}; WKV is registered only here")


def configure_isolated_extension_cache(
    model_name: str,
    image_size: int,
    output_root: Path,
) -> Path:
    """Give every WKV/T_MAX variant its own torch-extension build cache.

    APRIL's RWKV implementations compile the extension under a common module
    name while using model/size-specific T_MAX compile flags.  Reusing a build
    produced for another variant can therefore load successfully and still
    access memory illegally during the first CUDA synchronization.
    """
    t_max = int(MODEL_SPECS[model_name]["kernel_tmax"][int(image_size)])
    cache_dir = (
        Path(output_root)
        / "_torch_extensions"
        / f"{model_name}_size{int(image_size)}_v3"
    ).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(cache_dir)
    print(
        "[RWKV] isolated extension cache: "
        f"model={model_name}, size={int(image_size)}, "
        f"training_requested_T_MAX={t_max}, dir={cache_dir}. "
        "APRIL model code owns the actual WKV compilation."
    )
    return cache_dir


def apply_reference_aligned_u_rwkv_patch() -> str:
    """Apply exactly the U-RWKV SpatialMix correction used for training."""
    from medseg.models.networks.rwkv import u_rwkv as u_rwkv_module

    spatial_mix_class = u_rwkv_module.SpatialMix
    patch_id = "public_u_rwkv_spatialmix_decay_first_div_T_post_wkv_ln_v1"
    if getattr(spatial_mix_class, "_size512_reference_patch", None) == patch_id:
        return patch_id

    def reference_aligned_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        shifted = (
            u_rwkv_module.q_shift(x, self.shift_pixel)
            if self.shift_pixel > 0
            else x
        )
        xk = x * self.spatial_mix_k + shifted * (1 - self.spatial_mix_k)
        xv = x * self.spatial_mix_v + shifted * (1 - self.spatial_mix_v)
        xr = x * self.spatial_mix_r + shifted * (1 - self.spatial_mix_r)
        key = self.key(xk)
        value = self.value(xv)
        gate = torch.sigmoid(self.receptance(xr))
        rwkv = u_rwkv_module.wkv_pytorch(
            batch,
            tokens,
            channels,
            self.spatial_decay.float() / tokens,
            self.spatial_first.float() / tokens,
            key.float(),
            value.float(),
        ).to(x.dtype)
        if self.key_norm is not None:
            rwkv = self.key_norm(rwkv)
        return self.output(gate * rwkv)

    spatial_mix_class.forward = reference_aligned_forward
    spatial_mix_class._size512_reference_patch = patch_id
    return patch_id


def build_rwkv_model(
    model_name: str, image_size: int, april_root: Path
) -> Tuple[nn.Module, Dict[str, Any]]:
    check_rwkv_environment(model_name, april_root)
    patch_id = None
    if model_name == "u_rwkv":
        patch_id = apply_reference_aligned_u_rwkv_patch()
    from medseg.model_builder import build_model

    spec = MODEL_SPECS[model_name]
    model_config = {
        "model": {
            "architecture": spec["architecture"],
            "num_classes": 2,
            "img_size": int(image_size),
            "encoder": {"in_channels": 3, "pretrained": False},
            "arch_params": dict(spec["arch_params"]),
        },
        "rwkv_training_alignment_patch": patch_id,
        "kernel_tmax_from_training": int(spec["kernel_tmax"][image_size]),
    }
    return build_model({"model": model_config["model"]}), model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--april-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--weights-root", type=Path, default=DEFAULT_WEIGHTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root-224", type=Path, default=DEFAULT_DATA_ROOTS[224])
    parser.add_argument("--data-root-512", type=Path, default=DEFAULT_DATA_ROOTS[512])
    parser.add_argument("--sizes", type=int, nargs="+", choices=(224, 512), default=[224, 512])
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--no-auto-batch", action="store_true")
    parser.add_argument(
        "--fixed-batch",
        type=int,
        default=None,
        help=(
            "Explicit batch size used with --no-auto-batch. If omitted, an "
            "RWKV worker uses the batch size validated by its training script."
        ),
    )
    parser.add_argument("--max-batch-224", type=int, default=DEFAULT_MAX_BATCH[224])
    parser.add_argument("--max-batch-512", type=int, default=DEFAULT_MAX_BATCH[512])
    parser.add_argument("--batch-warmup-steps", type=int, default=3)
    parser.add_argument("--batch-timed-steps", type=int, default=10)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--worker-model", choices=MODEL_NAMES, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-size", type=int, choices=(224, 512), default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def worker_command(args: argparse.Namespace, model_name: str, size: int) -> List[str]:
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker-model", model_name, "--worker-size", str(size),
        "--april-root", str(args.april_root), "--weights-root", str(args.weights_root),
        "--output-root", str(args.output_root), "--data-root-224", str(args.data_root_224),
        "--data-root-512", str(args.data_root_512), "--gpu", str(args.gpu),
        "--workers", str(args.workers), "--threshold", str(args.threshold),
        "--amp-dtype", args.amp_dtype,
        "--max-batch-224", str(args.max_batch_224), "--max-batch-512", str(args.max_batch_512),
        "--batch-warmup-steps", str(args.batch_warmup_steps),
        "--batch-timed-steps", str(args.batch_timed_steps), "--splits", *args.splits,
    ]
    if args.fixed_batch is not None:
        command.extend(["--fixed-batch", str(args.fixed_batch)])
    if args.no_auto_batch:
        command.append("--no-auto-batch")
    if args.no_pin_memory:
        command.append("--no-pin-memory")
    return command


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def describe_returncode(returncode: int) -> str:
    if returncode >= 0:
        return f"exit_code={returncode}"
    signum = -returncode
    try:
        signal_name = signal.Signals(signum).name
    except ValueError:
        signal_name = f"signal_{signum}"
    return f"terminated_by={signal_name} ({signum})"


def run_logged_worker(command: List[str], log_path: Path) -> int:
    """Tee one isolated worker to the terminal and a persistent log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[RWKV] worker command: " + " ".join(command), flush=True)
    print(f"[RWKV] worker log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return int(process.wait())


def run_parent(args: argparse.Namespace) -> None:
    weights_root = common.resolve_from(PROJECT_ROOT, args.weights_root)
    output_root = common.resolve_from(PROJECT_ROOT, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    failures: List[Dict[str, Any]] = []
    for size in args.sizes:
        for model_name in args.models:
            model_dir = weights_root / WEIGHT_FAMILY / str(size) / model_name
            if not model_dir.is_dir():
                failures.append({"size": size, "model": model_name, "error": f"missing {model_dir}"})
                continue
            print(
                f"\n[ISOLATED] RWKV size={size} model={model_name}",
                flush=True,
            )
            command = worker_command(args, model_name, size)
            log_path = output_root / "worker_logs" / f"{model_name}_size{size}.log"
            returncode = run_logged_worker(command, log_path)
            if returncode != 0:
                failure = {
                    "size": size,
                    "model": model_name,
                    "error": describe_returncode(returncode),
                    "worker_log": str(log_path),
                }
                failures.append(failure)
                print(
                    "[FAILED] "
                    f"model={model_name}, size={size}, {failure['error']}; "
                    f"see {log_path}",
                    file=sys.stderr,
                    flush=True,
                )
    # Rebuild the aggregate from every result already present, not only from
    # this invocation.  This lets a failed U-RWKV worker be resumed without
    # dropping previously completed RWKV-UNet rows from the family report.
    rows: List[Dict[str, Any]] = []
    for size in (224, 512):
        for model_name in MODEL_NAMES:
            summary_path = output_root / str(size) / model_name / "summary.csv"
            if summary_path.is_file():
                rows.extend(read_csv_rows(summary_path))
    common.write_csv(output_root / "evaluation_summary.csv", rows)
    common.write_csv(output_root / "evaluation_failures.csv", failures)
    common.write_json(output_root / "evaluation_report.json", {"summaries": rows, "failures": failures})
    if failures:
        details = "\n".join(
            f"  - {item['model']} size={item['size']}: {item['error']}"
            + (f"; log={item['worker_log']}" if item.get("worker_log") else "")
            for item in failures
        )
        raise SystemExit(
            f"{len(failures)} RWKV evaluation worker(s) failed or were missing:\n"
            + details
        )


def run_worker(args: argparse.Namespace) -> None:
    assert args.worker_model is not None and args.worker_size is not None
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    april_root = common.resolve_from(PROJECT_ROOT, args.april_root)
    weights_root = common.resolve_from(PROJECT_ROOT, args.weights_root)
    output_root = common.resolve_from(PROJECT_ROOT, args.output_root)
    extension_cache = configure_isolated_extension_cache(
        args.worker_model,
        args.worker_size,
        output_root,
    )
    data_root_arg = args.data_root_224 if args.worker_size == 224 else args.data_root_512
    data_root = common.resolve_from(PROJECT_ROOT, data_root_arg)
    common.check_dataset_layout(data_root, args.splits)
    checkpoint = common.find_best_checkpoint(weights_root, WEIGHT_FAMILY, args.worker_size, args.worker_model)
    model, model_config = build_rwkv_model(args.worker_model, args.worker_size, april_root)
    model_config["torch_extensions_dir"] = str(extension_cache)
    maximum = args.max_batch_224 if args.worker_size == 224 else args.max_batch_512

    # A CUDA illegal-memory-access cannot be recovered from inside the same
    # process. U-RWKV is therefore not batch-tuned by repeatedly increasing B.
    # Use the exact per-size batch already validated by training unless the
    # caller explicitly supplies --fixed-batch.
    auto_batch = not args.no_auto_batch
    training_batch = int(
        MODEL_SPECS[args.worker_model]["training_batch_size"][args.worker_size]
    )
    fixed_batch = (
        int(args.fixed_batch)
        if args.fixed_batch is not None
        else training_batch
    )
    if args.worker_model == "u_rwkv" and auto_batch:
        auto_batch = False
        print(
            "[RWKV] U-RWKV safety mode: auto batch tuning disabled; "
            f"using training-validated batch_size={fixed_batch} for "
            f"Size_{args.worker_size}. Pass --no-auto-batch --fixed-batch N "
            "to override it explicitly."
        )
    common.evaluate_model(
        model=model, model_name=args.worker_model, model_config=model_config,
        family_label="RWKV", weight_family=WEIGHT_FAMILY, size=args.worker_size,
        checkpoint_path=checkpoint, data_root=data_root, output_root=output_root,
        splits=args.splits, device=device, amp_name=args.amp_dtype,
        threshold=args.threshold, workers=args.workers,
        auto_batch=auto_batch, maximum_batch=maximum,
        fixed_batch=fixed_batch, warmup_steps=args.batch_warmup_steps,
        timed_steps=args.batch_timed_steps, pin_memory=not args.no_pin_memory,
    )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    if args.fixed_batch is not None and args.fixed_batch < 1:
        raise ValueError("--fixed-batch must be at least 1")
    if args.worker_model is None:
        run_parent(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
