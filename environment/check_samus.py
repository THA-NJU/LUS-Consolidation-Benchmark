#!/usr/bin/env python3
"""Verify the official SAMUS source, SAM ViT-B checkpoint, and GPU path."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace


EXPECTED_SHA256_PREFIX = "01ec64"
NATIVE_SAMUS_NAME_TOKENS = (
    "cnn_embed",
    "post_pos_embed",
    "Adapter",
    "blocks.2.attn.rel_pos",
    "blocks.5.attn.rel_pos",
    "blocks.8.attn.rel_pos",
    "blocks.11.attn.rel_pos",
    "upneck",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--samus-source-dir", type=Path, default=Path("~/third_party/SAMUS")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("pretrained/samus/sam_vit_b_01ec64.pth"),
    )
    parser.add_argument(
        "--build-model",
        action="store_true",
        help="Construct SAMUS ViT-B and load the official SAM checkpoint on CPU.",
    )
    parser.add_argument(
        "--forward",
        action="store_true",
        help="Run one fixed-full-box GPU forward pass; implies --build-model.",
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Run a native-scope backward check; implies --forward.",
    )
    return parser.parse_args()


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_registry(source_dir: Path):
    models_dir = source_dir / "models"
    build_file = models_dir / "segment_anything_samus" / "build_sam_us.py"
    if not build_file.is_file():
        raise FileNotFoundError(f"SAMUS model builder not found: {build_file}")
    sys.path.insert(0, str(models_dir))
    from segment_anything_samus import samus_model_registry

    imported_from = Path(
        inspect.getfile(sys.modules["segment_anything_samus"])
    ).resolve()
    if models_dir not in imported_from.parents:
        raise RuntimeError(
            f"Imported segment_anything_samus from {imported_from}, "
            f"not {models_dir}."
        )
    return samus_model_registry, imported_from


def fixed_full_box_forward(model, images):
    import torch

    embeddings = model.image_encoder(images)
    boxes = images.new_tensor([0.0, 0.0, 256.0, 256.0])
    boxes = boxes.view(1, 4).expand(images.shape[0], -1)
    with torch.no_grad():
        sparse, dense = model.prompt_encoder(points=None, boxes=boxes, masks=None)
    logits, quality = model.mask_decoder(
        image_embeddings=embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    return logits, quality


def configure_native_scope(model) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.image_encoder.named_parameters():
        if any(token in name for token in NATIVE_SAMUS_NAME_TOKENS):
            parameter.requires_grad = True
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def main() -> None:
    args = parse_args()
    source_dir = args.samus_source_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()

    import einops
    import cv2
    import numpy as np
    import torch
    import torchvision

    print("python:", sys.version.split()[0])
    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("numpy:", np.__version__)
    print("einops:", einops.__version__)
    print("opencv:", cv2.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("bf16 supported:", torch.cuda.is_bf16_supported())

    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM ViT-B checkpoint not found: {checkpoint}")
    actual_sha256 = sha256sum(checkpoint)
    print("checkpoint:", checkpoint)
    print("checkpoint sha256:", actual_sha256)
    if not actual_sha256.startswith(EXPECTED_SHA256_PREFIX):
        raise RuntimeError(
            "Checkpoint SHA-256 mismatch: expected prefix "
            f"{EXPECTED_SHA256_PREFIX}, got {actual_sha256}"
        )

    registry, imported_from = import_registry(source_dir)
    print("segment_anything_samus:", imported_from)
    print("model registry: OK")

    if args.build_model or args.forward or args.backward:
        builder_args = SimpleNamespace(encoder_input_size=256)
        model = registry["vit_b"](args=builder_args, checkpoint=str(checkpoint))
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable_names = configure_native_scope(model)
        trainable = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        if not trainable_names:
            raise RuntimeError("No native SAMUS parameters were selected.")
        print(f"model build: OK ({total / 1e6:.2f} M parameters)")
        print(
            f"native scope: OK ({trainable / 1e6:.2f} M parameters, "
            f"{len(trainable_names)} tensors)"
        )
    else:
        model = None

    if args.forward or args.backward:
        if not torch.cuda.is_available():
            raise RuntimeError("--forward/--backward requires a CUDA-capable GPU.")
        assert model is not None
        model = model.cuda()
        model.eval()
        if args.backward:
            model.image_encoder.train()
        image = torch.zeros(1, 1, 256, 256, device="cuda")
        amp_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

        context = torch.enable_grad() if args.backward else torch.inference_mode()
        with context, torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits, quality = fixed_full_box_forward(model, image)
            loss = logits.float().sigmoid().mean()
        print("forward: OK", tuple(logits.shape), tuple(quality.shape))

        if args.backward:
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if not gradients:
                raise RuntimeError("Backward produced no native-scope gradients.")
            if not all(torch.isfinite(gradient).all() for gradient in gradients):
                raise RuntimeError("Backward produced non-finite gradients.")
            print(
                f"backward: OK ({len(gradients)}/{len(trainable_names)} "
                "trainable tensors received gradients)"
            )


if __name__ == "__main__":
    main()