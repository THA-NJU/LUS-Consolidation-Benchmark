#!/usr/bin/env python3
"""Check the shared SAM2/MedSAM environment and optionally build SAM2.1 B+."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("pretrained/sam2/sam2.1_hiera_base_plus.pt"),
    )
    parser.add_argument(
        "--model-cfg",
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
    )
    parser.add_argument(
        "--build-model",
        action="store_true",
        help="Load the checkpoint on GPU and report parameter counts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    import pandas as pd
    import PIL
    import torch
    import torchvision
    import sam2

    print("python:", platform.python_version())
    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("numpy:", np.__version__)
    print("pandas:", pd.__version__)
    print("pillow:", PIL.__version__)
    print("sam2 package:", sam2.__file__)
    print("cuda available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: PyTorch cannot access the NVIDIA GPU.")

    print("cuda runtime used by torch:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())

    if not args.checkpoint.is_file():
        raise SystemExit(f"ERROR: checkpoint not found: {args.checkpoint}")
    print("checkpoint:", args.checkpoint.resolve())

    if args.build_model:
        from sam2.build_sam import build_sam2

        model = build_sam2(
            args.model_cfg,
            str(args.checkpoint),
            device="cuda",
            mode="eval",
            apply_postprocessing=False,
        )
        total = sum(p.numel() for p in model.parameters())
        print(f"model build: OK ({total / 1e6:.2f} M parameters)")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()