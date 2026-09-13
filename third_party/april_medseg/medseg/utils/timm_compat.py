"""timm >= 1.0 兼容层重新导出 / Re-exports for timm >= 1.0 compatibility."""

from timm.layers import DropPath, to_2tuple, trunc_normal_, trunc_normal_tf_
from timm.models import named_apply

__all__ = [
    "DropPath",
    "to_2tuple",
    "trunc_normal_",
    "trunc_normal_tf_",
    "named_apply",
]
