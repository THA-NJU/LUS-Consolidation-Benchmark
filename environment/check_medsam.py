#!/usr/bin/env python3
"""Verify the MedSAM source tree, checkpoint, and existing sam_bench env."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import sys
from pathlib import Path


EXPECTED_MD5 = "3bb6db55bd0c9ca30b61248bca72f8d6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--medsam-source-dir", type=Path, default=Path("~/third_party/MedSAM")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("pretrained/medsam/medsam_vit_b.pth"),
    )
    parser.add_argument(
        "--build-model",
        action="store_true",
        help="Construct MedSAM ViT-B and load its checkpoint on CPU.",
    )
    parser.add_argument(
        "--forward",
        action="store_true",
        help="Run one GPU forward pass; implies --build-model.",
    )
    return parser.parse_args()


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_registry(source_dir: Path):
    package_dir = source_dir / "segment_anything"
    if not (package_dir / "build_sam.py").is_file():
        raise FileNotFoundError(
            f"MedSAM segment_anything source not found below: {source_dir}"
        )
    sys.path.insert(0, str(source_dir))
    from segment_anything import sam_model_registry

    imported_from = Path(inspect.getfile(sys.modules["segment_anything"])).resolve()
    if source_dir not in imported_from.parents:
        raise RuntimeError(
            f"Imported segment_anything from {imported_from}, not {source_dir}."
        )
    return sam_model_registry, imported_from


def main() -> None:
    args = parse_args()
    source_dir = args.medsam_source_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()

    import cv2
    import numpy as np
    import torch
    import torchvision

    print("python:", sys.version.split()[0])
    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("numpy:", np.__version__)
    print("opencv:", cv2.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("bf16 supported:", torch.cuda.is_bf16_supported())

    if not checkpoint.is_file():
        raise FileNotFoundError(f"MedSAM checkpoint not found: {checkpoint}")
    actual_md5 = md5sum(checkpoint)
    print("checkpoint:", checkpoint)
    print("checkpoint md5:", actual_md5)
    if actual_md5 != EXPECTED_MD5:
        raise RuntimeError(
            f"Checkpoint MD5 mismatch: expected {EXPECTED_MD5}, got {actual_md5}"
        )

    registry, imported_from = import_registry(source_dir)
    print("segment_anything:", imported_from)
    print("model registry: OK")

    if args.build_model or args.forward:
        model = registry["vit_b"](checkpoint=str(checkpoint))
        total = sum(parameter.numel() for parameter in model.parameters())
        print(f"model build: OK ({total / 1e6:.2f} M parameters)")
    else:
        model = None

    if args.forward:
        if not torch.cuda.is_available():
            raise RuntimeError("--forward requires a CUDA-capable GPU.")
        assert model is not None
        model = model.cuda().eval()
        image = torch.zeros(1, 3, 1024, 1024, device="cuda")
        box = torch.tensor([[[0.0, 0.0, 1024.0, 1024.0]]], device="cuda")
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        ):
            embedding = model.image_encoder(image)
            sparse, dense = model.prompt_encoder(points=None, boxes=box, masks=None)
            logits, quality = model.mask_decoder(
                image_embeddings=embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
        print("forward: OK", tuple(logits.shape), tuple(quality.shape))


if __name__ == "__main__":
    main()