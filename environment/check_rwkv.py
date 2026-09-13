#!/usr/bin/env python3
"""Preflight checks for APRIL-MedSeg RWKV models on one CUDA GPU."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--april-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="APRIL-MedSeg root; defaults to this script's directory.",
    )
    parser.add_argument(
        "--model",
        choices=("rwkv_unet", "u_rwkv"),
        default="rwkv_unet",
    )
    parser.add_argument(
        "--kernel",
        action="store_true",
        help="Compile and validate the WKV CUDA kernel.",
    )
    parser.add_argument(
        "--forward",
        action="store_true",
        help="Run a BF16 batch-1 512x512 forward pass.",
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Run a BF16 batch-1 512x512 forward/backward pass.",
    )
    return parser.parse_args()


def command_version(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return result.stdout.strip()
    except Exception as exc:
        return f"unavailable ({type(exc).__name__}: {exc})"


def model_config(model_name: str) -> dict:
    if model_name == "rwkv_unet":
        arch_params = {"variant": "b"}
    else:
        arch_params = {
            "embed_dims": [64, 128, 256, 512],
            "depths": [2, 2, 2, 2],
            "shift_pixel": 1,
            "se_ratio": 0.25,
            "deep_supervision": False,
        }
    return {
        "model": {
            "architecture": model_name,
            "num_classes": 2,
            "img_size": 512,
            "encoder": {"in_channels": 3, "pretrained": False},
            "arch_params": arch_params,
        }
    }


def require_finite_nonzero(name: str, tensor) -> None:
    import torch

    if tensor is None:
        raise RuntimeError(f"{name}: gradient is None")
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name}: gradient contains NaN/Inf")
    if float(tensor.abs().sum().item()) == 0.0:
        raise RuntimeError(f"{name}: gradient is identically zero")


def main() -> None:
    args = parse_args()
    april_root = args.april_root.expanduser().resolve()
    if not (april_root / "medseg" / "model_builder.py").is_file():
        raise FileNotFoundError(
            f"Not an APRIL-MedSeg root: {april_root}\n"
            "Expected medseg/model_builder.py."
        )
    sys.path.insert(0, str(april_root))

    import numpy as np
    import torch
    import torchvision
    from torch.utils.cpp_extension import CUDA_HOME

    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"numpy: {np.__version__}")
    print(f"APRIL root: {april_root}")
    print(f"APRIL commit: {command_version(['git', '-C', str(april_root), 'rev-parse', '--short', 'HEAD'])}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA_HOME: {CUDA_HOME}")
    print(f"nvcc: {shutil.which('nvcc')}")
    print(f"ninja: {shutil.which('ninja')}")
    print(f"ninja version: {command_version(['ninja', '--version'])}")

    required_files = [
        april_root / "medseg" / "kernels" / "wkv" / "__init__.py",
        april_root / "medseg" / "kernels" / "wkv" / "wkv_op.cpp",
        april_root / "medseg" / "kernels" / "wkv" / "wkv_cuda.cu",
        april_root / "medseg" / "models" / "networks" / "rwkv" / "rwkv_unet.py",
        april_root / "medseg" / "models" / "networks" / "rwkv" / "u_rwkv.py",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The APRIL checkout lacks the unified RWKV implementation:\n  "
            + "\n  ".join(missing)
        )

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the Size_512 RWKV benchmark")
    if CUDA_HOME is None or shutil.which("nvcc") is None:
        raise RuntimeError(
            "WKV needs a local CUDA toolkit/NVCC; a PyTorch CUDA runtime alone "
            "is insufficient."
        )
    if shutil.which("ninja") is None and importlib.util.find_spec("ninja") is None:
        raise RuntimeError("Install Ninja first: python -m pip install ninja")

    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{capability[0]}.{capability[1]}")
    os.environ.setdefault("MAX_JOBS", "4")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"compute capability: {capability[0]}.{capability[1]}")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    print(f"TORCH_CUDA_ARCH_LIST: {os.environ['TORCH_CUDA_ARCH_LIST']}")

    from medseg.kernels.wkv import (
        get_load_error,
        is_cuda_available,
        load_wkv_cuda,
        run_wkv,
    )

    # RWKV-UNet first applies spatial WKV at 64x64 for a 512 input. U-RWKV
    # applies WKV immediately after the 2x stem, hence 256x256 tokens.
    t_max = 8192 if args.model == "rwkv_unet" else 65536
    if args.kernel or args.forward or args.backward:
        print(f"Compiling/loading WKV CUDA kernel with Tmax={t_max} ...")
        op = load_wkv_cuda(t_max=t_max, force=True, verbose=True)
        if op is None or not is_cuda_available():
            raise RuntimeError(f"WKV CUDA compilation failed: {get_load_error()!r}")
        print("WKV CUDA kernel: OK")

        # Verify the custom analytic backward path, not just compilation.
        B, T, C = 1, 32, 32
        w = torch.randn(C, device=device, dtype=torch.float32, requires_grad=True)
        u = torch.randn(C, device=device, dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, T, C, device=device, dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, T, C, device=device, dtype=torch.float32, requires_grad=True)
        y = run_wkv(B, T, C, w, u, k, v, t_max=t_max)
        y.square().mean().backward()
        for name, value in (("w", w), ("u", u), ("k", k), ("v", v)):
            require_finite_nonzero(f"WKV {name}", value.grad)
        print("WKV analytic backward: OK (finite, non-zero gradients)")

    if args.forward or args.backward:
        from medseg.model_builder import build_model

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = build_model(model_config(args.model)).to(device)
        model.train(args.backward)
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        print(
            f"model build: OK ({total / 1e6:.2f} M total, "
            f"{trainable / 1e6:.2f} M trainable)"
        )

        image = torch.randn(1, 3, 512, 512, device=device)
        target = torch.randint(0, 2, (1, 512, 512), device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(image)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            loss = torch.nn.functional.cross_entropy(logits.float(), target)
        if logits.shape != (1, 2, 512, 512):
            raise RuntimeError(f"Unexpected output shape: {tuple(logits.shape)}")
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite forward loss: {loss.item()}")
        print(f"forward: OK {tuple(logits.shape)}, loss={loss.item():.6f}")

        if args.backward:
            loss.backward()
            rwkv_grads = []
            for name, parameter in model.named_parameters():
                if (
                    parameter.requires_grad
                    and parameter.grad is not None
                    and any(
                        token in name.lower()
                        for token in ("spatial_decay", "spatial_first", "mix_k", "spatial_mix")
                    )
                ):
                    if not torch.isfinite(parameter.grad).all():
                        raise RuntimeError(f"Non-finite RWKV gradient: {name}")
                    rwkv_grads.append((name, float(parameter.grad.abs().sum().item())))
            nonzero = [(name, value) for name, value in rwkv_grads if value > 0.0]
            if not nonzero:
                raise RuntimeError(
                    "Model backward completed, but no non-zero RWKV parameter "
                    "gradient was found."
                )
            print(
                f"backward: OK ({len(nonzero)} RWKV tensors with non-zero gradients)"
            )

        peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(f"peak allocated CUDA memory: {peak_gib:.2f} GiB")

    if not (args.kernel or args.forward or args.backward):
        print("static environment check: OK")
        print(
            "Next: rerun with --model rwkv_unet --kernel --forward --backward"
        )


if __name__ == "__main__":
    main()