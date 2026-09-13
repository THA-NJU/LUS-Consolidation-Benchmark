"""Lazy registry for the APRIL architectures exercised by LUSBench.

The upstream APRIL package imports every architecture when this module is
loaded.  That behavior pulls in many unrelated optional dependencies.  This
snapshot intentionally resolves only the seventeen architectures used by the
benchmark while keeping their original implementation files unchanged.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any


_SPECIAL_ARCHS: dict[str, object] = {
    "nnunet_2d": ("medseg.models.networks.cnn.nnunet_2d", "NNUNet2D"),
    "aau_net": ("medseg.models.networks.cnn.aau_net", "AAUNet"),
    "mednext": ("medseg.models.networks.cnn.mednext_model", "MedNeXt"),
    "swinunet": ("medseg.models.networks.transformer.swinunet_model", "SwinUNet"),
    "nnformer_2d": ("medseg.models.networks.transformer.nnformer_2d", "NNFormer2D"),
    "pvtb2_emcad": ("medseg.models.networks.transformer.pvtb2_emcad_model", "PVTB2EMCAD"),
    "sepnet": ("medseg.models.networks.transformer.sepnet", "SEPNet"),
    "nulite": ("medseg.models.networks.transformer.nulite", "NuLite"),
    "mamba_unet": ("medseg.models.networks.mamba.mamba_unet", "MambaUNet"),
    "vm_unet_v2": ("medseg.models.networks.mamba.vm_unet_v2", "VMUNetV2"),
    "nnmamba_2d": ("medseg.models.networks.mamba.nnmamba_2d", "NnMamba2D"),
    "swin_umamba": ("medseg.models.networks.mamba.swin_umamba", "SwinUMamba"),
    "rolling_unet": ("medseg.models.networks.kan_mlp.rolling_unet", "RollingUNet"),
    "ukan": ("medseg.models.networks.kan_mlp.ukan", "UKAN"),
    "xlstm_unet_bot": ("medseg.models.networks.linear_attn.xlstm_unet", "XLSTMUNetBot"),
    "u_rwkv": ("medseg.models.networks.rwkv.u_rwkv", "URWKV"),
    "rwkv_unet": ("medseg.models.networks.rwkv.rwkv_unet", "RWKVUNet"),
}


def _resolve(entry: object) -> type:
    if isinstance(entry, tuple):
        module_name, class_name = entry
        return getattr(importlib.import_module(module_name), class_name)
    if inspect.isclass(entry):
        return entry
    raise TypeError(f"Invalid APRIL architecture registry entry: {entry!r}")


def build_special_arch(arch_name: str, cfg: dict[str, Any]):
    """Construct one of the frozen APRIL architectures."""
    if arch_name not in _SPECIAL_ARCHS:
        available = ", ".join(sorted(_SPECIAL_ARCHS))
        raise KeyError(f"{arch_name!r} is not in the LUSBench APRIL subset: {available}")

    cls = _resolve(_SPECIAL_ARCHS[arch_name])
    encoder_cfg = cfg.get("encoder", {})
    kwargs: dict[str, Any] = {
        "in_channels": encoder_cfg.get("in_channels", 3),
        "num_classes": cfg.get("num_classes", 2),
        "img_size": cfg.get("img_size", 224),
        "pretrained": encoder_cfg.get("pretrained", False),
    }
    kwargs.update(cfg.get("arch_params", {}))

    signature = inspect.signature(cls)
    accepts_extra = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if not accepts_extra:
        kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return cls(**kwargs)


__all__ = ["_SPECIAL_ARCHS", "build_special_arch"]
