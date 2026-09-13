"""Compatibility builder for the minimal APRIL source subset."""

from __future__ import annotations

from typing import Any


def build_model(cfg: dict[str, Any]):
    model_cfg = cfg.get("model", cfg)
    architecture = model_cfg.get("architecture")
    if not architecture:
        raise ValueError(
            "The LUSBench APRIL subset supports complete architectures only; "
            "cfg['model']['architecture'] is required."
        )
    from medseg.models.networks import build_special_arch

    return build_special_arch(str(architecture), model_cfg)


__all__ = ["build_model"]
