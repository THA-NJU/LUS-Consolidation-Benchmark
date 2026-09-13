"""Compatibility aliases required by APRIL's Mamba implementations."""

from __future__ import annotations

import sys

from .mamba import vmunet_encoder

sys.modules.setdefault("medseg.models.encoders.vmunet_encoder", vmunet_encoder)

__all__ = ["vmunet_encoder"]
