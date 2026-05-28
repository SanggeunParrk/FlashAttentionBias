#!/usr/bin/env bash
# Smoke-test the SM90 fp32 (TF32 WGMMA) forward path on node01.
set -euo pipefail

REPO=/home/psk6950/FlashAttentionBias
VENV=$REPO/aug_attn_bench/.venv
PY=$VENV/bin/python
PIP=$VENV/bin/pip

cd /tmp
echo "=== node: $(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# Install cute DSL + quack into the cu126 venv if missing.
$PY -c "import cutlass.cute" 2>/dev/null || {
    echo "=== installing nvidia-cutlass-dsl (cu12) + apache-tvm-ffi + torch_c_dlpack_ext + quack-kernels"
    $PIP install --no-deps nvidia-cutlass-dsl
    $PIP install nvidia-cutlass-dsl-libs-cu12
    $PIP install apache-tvm-ffi torch_c_dlpack_ext quack-kernels typing_extensions
}

# Make FA4_fp32 importable (just add repo root to PYTHONPATH; FA4_fp32/ is a
# package directory at the repo root).
export PYTHONPATH=$REPO:${PYTHONPATH:-}

echo "=== import check"
$PY -c "
import torch
print('torch', torch.__version__, 'cuda avail', torch.cuda.is_available())
import cutlass, cutlass.cute as cute
print('cutlass DSL ok')
import FA4_fp32
print('FA4_fp32 ok')
from FA4_fp32.arch import hopper_helpers
print('hopper_helpers ok; has MmaTF32WgmmaOp:', hasattr(hopper_helpers, 'MmaTF32WgmmaOp'))
"

echo "=== run smoke test"
$PY $REPO/FA4_fp32/test_sm90_fp32.py
