#!/usr/bin/env bash
# Install FA4_fp32 dependencies pinned for CUDA 12.9 (cluster driver
# is 575.51 = CUDA 12.9 max), then check imports.
set -euo pipefail

REPO=/home/psk6950/FlashAttentionBias
VENV=$REPO/aug_attn_bench/.venv      # already has torch 2.8.0 + cu126
PY=$VENV/bin/python
PIP=$VENV/bin/pip

cd /tmp

echo "=== node: $(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

# Pin cuda-python / cuda-bindings to 12.9.* so they don't pull cu13.
# Without [cu13] extra, only nvidia-cutlass-dsl-libs-base (CUDA-12-compatible)
# gets installed alongside nvidia-cutlass-dsl.
echo "=== installing cu12-compatible FA4 deps"
$PIP install \
    "cuda-python==12.9.*" \
    "cuda-bindings==12.9.*" \
    "nvidia-cutlass-dsl>=4.4.2" \
    "apache-tvm-ffi>=0.1.5,<0.2" \
    "torch-c-dlpack-ext" \
    "quack-kernels>=0.2.10" \
    "typing_extensions"

# Mark conflict diagnostic.
$PIP check 2>&1 || true

echo "=== environment"
$PY -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
import cuda.bindings as cb
print('cuda.bindings', cb.__version__ if hasattr(cb, '__version__') else 'unknown')
import cutlass, cutlass.cute as cute
print('cutlass DSL imported')
"

echo "=== FA4_fp32 import"
PYTHONPATH=$REPO $PY -c "
import FA4_fp32
print('FA4_fp32 ok')
from FA4_fp32.arch import wgmma_tf32, sm90_utils_tf32, hopper_helpers
print('  arch.* ok')
"

echo "=== fp32 dispatch path (FakeTensorMode)"
PYTHONPATH=$REPO CUTE_DSL_ARCH=sm_90a FLASH_ATTENTION_ARCH=90 $PY -c "
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from FA4_fp32 import flash_attn_func
with FakeTensorMode(allow_non_fake_inputs=True):
    q = torch.randn(2, 512, 4, 32, dtype=torch.float32, device='cuda')
    k = torch.randn_like(q); v = torch.randn_like(q)
    try:
        out = flash_attn_func(q, k, v)
        print('UNEXPECTED: fp32 path succeeded')
    except NotImplementedError as e:
        print('OK: NotImplementedError raised cleanly (expected, integration stub):')
        print(' ', list(str(e).split('. '))[0])
"

echo "=== bf16 actual execution"
PYTHONPATH=$REPO CUTE_DSL_ARCH=sm_90a FLASH_ATTENTION_ARCH=90 $PY -c "
import torch
from FA4_fp32 import flash_attn_func
torch.manual_seed(0)
q = torch.randn(2, 512, 4, 64, dtype=torch.bfloat16, device='cuda')
k = torch.randn_like(q); v = torch.randn_like(q)
try:
    out, _ = flash_attn_func(q, k, v, return_lse=True)
    torch.cuda.synchronize()
    print('bf16 fwd OK, out shape', tuple(out.shape), 'dtype', out.dtype)
except Exception as e:
    import traceback; traceback.print_exc()
    print(f'bf16 fwd FAILED: {type(e).__name__}: {e}')
"
