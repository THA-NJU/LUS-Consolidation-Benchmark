"""Single source of truth for LUSBench model, training, and evaluation routes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ModelRoute:
    model: str
    display_name: str
    family: str
    size: int
    status: str
    route: Optional[str]
    script: Optional[str]
    evaluation_script: Optional[str]
    note: str = ""


def _route(
    model: str,
    display: str,
    family: str,
    size: int,
    route: Optional[str],
    script: Optional[str],
    evaluation_script: Optional[str],
    *,
    status: str = "route_available",
    note: str = "",
) -> ModelRoute:
    return ModelRoute(
        model, display, family, size, status, route, script,
        evaluation_script, note,
    )


def _both(model: str, display: str, family: str, route: str, script: str, evaluator: str):
    return tuple(
        _route(model, display, family, size, route, script, evaluator)
        for size in (512, 224)
    )


ROUTES = [
    *[
        _route(model, display, "CNN", size,
               "monai512" if size == 512 else "monai224",
               "experiments/monai/train_512.py" if size == 512 else "experiments/monai/train_224_filtered.py",
               "evaluation/families/evaluate_monai_family.py")
        for model, display in (
            ("monai_unet", "U-Net"),
            ("monai_vnet", "V-Net"),
            ("monai_attention_unet", "Attention U-Net"),
            ("monai_unetplusplus", "U-Net++"),
        )
        for size in (512, 224)
    ],
    *[
        _route(model, display, family, size,
               "april_additional512" if size == 512 else "april_additional224",
               "experiments/april/train_additional_512.py" if size == 512 else "experiments/april/train_additional_224.py",
               "evaluation/families/evaluate_mednext_pvtb2_rolling.py" if size == 512 else "evaluation/families/evaluate_mednext_pvtb2_rolling_224.py")
        for model, display, family in (
            ("mednext", "MedNeXt", "CNN"),
            ("pvtb2_emcad", "PVTv2-B2-EMCAD", "Transformer"),
            ("rolling_unet", "Rolling-UNet", "Other"),
        )
        for size in (512, 224)
    ],
    *[
        _both(model, display, family, "april_general", "experiments/april/train_general.py", "evaluation/families/evaluate_autodl_easy_models.py")
        for model, display, family in (
            ("nnunet_2d", "nnU-Net", "CNN"),
            ("aau_net", "AAU-Net", "CNN"),
            ("swinunet", "Swin-UNet", "Transformer"),
            ("nnformer_2d", "nnFormer", "Transformer"),
            ("sepnet", "SEPNet", "Transformer"),
            ("nulite", "NuLite", "Transformer"),
            ("ukan", "U-KAN", "Other"),
            ("xlstm_unet_bot", "xLSTM-UNet", "Other"),
        )
    ],
    *[
        _route(model, display, "Mamba/SSM", size,
               "mamba512" if size == 512 else "monai224",
               "experiments/mamba/train_512.py" if size == 512 else "experiments/monai/train_224_filtered.py",
               "evaluation/families/evaluate_mamba_family.py")
        for model, display in (
            ("mamba_unet", "Mamba-UNet"),
            ("vm_unet_v2", "VM-UNet V2"),
            ("nnmamba_2d", "nn-Mamba"),
            ("swin_umamba", "Swin-UMamba"),
        )
        for size in (512, 224)
    ],
    *[
        _route(model, display, "RWKV", size,
               "rwkv512" if size == 512 else "rwkv224",
               "experiments/rwkv/train_512.py" if size == 512 else "experiments/rwkv/train_224_filtered.py",
               "evaluation/families/evaluate_rwkv_family.py",
               note="U-RWKV Size_224 evaluation previously failed with a CUDA illegal-memory-access error." if model == "u_rwkv" and size == 224 else "")
        for model, display in (("u_rwkv", "U-RWKV"), ("rwkv_unet", "RWKV-UNet"))
        for size in (512, 224)
    ],
    *[
        _route("segformer_b2", "SegFormer-B2", "Transformer", size,
               "segformer_native",
               "experiments/segformer/train_512.py" if size == 512 else "experiments/segformer/train_224_filtered.py",
               "evaluation/native/evaluate_segformer_b2.py")
        for size in (512, 224)
    ],
    *[
        _route(model, display, "Foundation", size,
               "sam512" if size == 512 else "sam224",
               f"experiments/foundation/train_{model}_512.py" if size == 512 else "experiments/foundation/train_sam_family_224_filtered.py",
               "evaluation/foundation/evaluate_sam_checkpoints.py")
        for model, display in (("sam2", "SAM2"), ("medsam", "MedSAM"), ("samus", "SAMUS"))
        for size in (512, 224)
    ],
    *[
        _route("sam3", "SAM3", "Foundation", size, "sam3",
               "experiments/foundation/sam3/train_interactive.py",
               "evaluation/foundation/evaluate_sam3_official.py",
               note="Official SAM3 source commit 46957e47805e; weights are not distributed.")
        for size in (512, 224)
    ],
    *[
        _route(model, display, "YOLO", size, "yolo_family",
               "experiments/yolo/train_family.py", None,
               note="Training writes final unified metrics; no separate checkpoint-only evaluator is released for this route.")
        for model, display in (
            ("yolo11s-seg", "YOLO11s-seg"),
            ("yolo11m-seg", "YOLO11m-seg"),
            ("yolo11l-seg", "YOLO11l-seg"),
            ("yolo26m-sem", "YOLO26m-sem"),
            ("yolo26l-sem", "YOLO26l-sem"),
        )
        for size in (512, 224)
    ],
    *[
        _route("yolo26s-sem", "YOLO26s-sem", "YOLO", size, "yolo26s_native",
               "experiments/yolo26s/train_512.py" if size == 512 else "experiments/yolo26s/train_224_filtered.py",
               "evaluation/native/evaluate_yolo26s_sem.py")
        for size in (512, 224)
    ],
    *[
        _route("s2denet", "S2DENet", "Other", size, "s2denet",
               "experiments/s2denet/train.py", None,
               note="The training backend performs validation and test evaluation.")
        for size in (512, 224)
    ],
    _route("usfm_transfer", "USFM transfer", "Foundation", 512, None, None,
           "experiments/usfm/evaluate_zeroshot.py", status="evaluation_only",
           note="Three-class adapted checkpoint evaluated using consolidation channel 2."),
    _route("usfm_transfer", "USFM transfer", "Foundation", 224, "usfm224",
           "experiments/usfm/train_transfer_224.py", "experiments/usfm/evaluate_zeroshot.py",
           note="Transfer experiment; not trained from scratch."),
    *[
        _route(model, display, "Legacy transfer", size, None, None,
               "evaluation/legacy_resnet.py", status="evaluation_only",
               note="Checkpoint evaluation is provided; the original ResNet transfer-training entry point is not included in this repository.")
        for model, display in (
            ("fpn_resnet34", "FPN-ResNet34"),
            ("deeplabv3plus_resnet34", "DeepLabV3+-ResNet34"),
        )
        for size in (512, 224)
    ],
]

# Flatten tuples inserted by _both.
ROUTES = [item for group in ROUTES for item in (group if isinstance(group, tuple) else (group,))]


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_route(model: str, size: int) -> ModelRoute:
    for route in ROUTES:
        if route.model == model and route.size == size:
            return route
    known = ", ".join(sorted({route.model for route in ROUTES}))
    raise KeyError(f"Unknown model/size combination {model}/{size}. Known models: {known}")


def model_count() -> int:
    return len({route.model for route in ROUTES})
