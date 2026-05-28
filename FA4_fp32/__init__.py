"""Flash Attention CUTE (CUDA Template Engine) implementation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

import cutlass.cute as cute

# Varlen API was removed in the B200-style strip (MHA happy path only).
from .interface import flash_attn_func

from FA4_fp32.infra.cute_dsl_utils import cute_compile_patched

# Patch cute.compile to optionally dump SASS
cute.compile = cute_compile_patched


__all__ = [
    "flash_attn_func",
]
