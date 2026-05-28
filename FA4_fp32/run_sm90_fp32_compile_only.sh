#!/usr/bin/env bash
# Compile-only smoke test for SM90 fp32 (TF32 WGMMA) forward.
# Uses FLASH_ATTENTION_FAKE_TENSOR=1 so we only exercise the cute DSL
# lowering path, not actual GPU execution (cu13 driver not available).
set -euo pipefail

REPO=/home/psk6950/FlashAttentionBias
PY=$REPO/.pixi/envs/default/bin/python

cd /tmp  # avoid top-level flash_attn/__init__ shadowing

export FLASH_ATTENTION_FAKE_TENSOR=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=0  # no caching for clean repro
export FLASH_ATTENTION_ARCH=90
export CUTE_DSL_ARCH=sm_90a       # cute DSL also needs this when CUDA isn't initialized

echo "=== node: $(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

echo "=== python: $($PY -V)"
$PY -c "
import torch
print('torch', torch.__version__)
import cutlass, cutlass.cute as cute
print('cutlass DSL ok')
import sys; sys.path.insert(0, '$REPO')
import FA4_fp32
print('FA4_fp32 ok')
from FA4_fp32.arch import hopper_helpers
print('hopper_helpers ok; has MmaTF32WgmmaOp:', hasattr(hopper_helpers, 'MmaTF32WgmmaOp'))
"

echo "=== compile-only smoke test (FakeTensorMode)"
$PY -c "
import sys; sys.path.insert(0, '$REPO')
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from FA4_fp32 import flash_attn_func

print('arch override:', __import__('os').environ.get('FLASH_ATTENTION_ARCH'))
with FakeTensorMode(allow_non_fake_inputs=True):
    B, H, L, D = 2, 8, 1024, 64
    q = torch.randn(B, L, H, D, dtype=torch.float32, device='cuda')
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    print(f'  inputs: B={B} H={H} L={L} D={D} dtype={q.dtype}')
    try:
        out = flash_attn_func(q, k, v)
        print('  COMPILE OK -> output shape', tuple(out[0].shape) if isinstance(out, tuple) else tuple(out.shape))
    except Exception as e:
        print(f'  COMPILE FAILED: {type(e).__name__}: {e}')
        import traceback; traceback.print_exc()
"
