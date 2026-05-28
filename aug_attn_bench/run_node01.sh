#!/usr/bin/env bash
# Bench launcher to be run inside an srun allocation on node01.
#
# Avoids the FA4 pixi env (which pulls cu130 torch incompatible with cluster
# driver). Creates a small venv with cu124 torch + triton, and falls back to
# SDPA for the fast-attention path in Option 2.
set -euo pipefail

REPO=/home/psk6950/FlashAttentionBias
VENV=$REPO/aug_attn_bench/.venv
TXT=$REPO/aug_attn_bench/bench_aug_attn.txt

# Run from /tmp so the top-level flash_attn/__init__.py (FA2 stub) doesn't
# shadow anything; not strictly needed here, but cheap.
cd /tmp

echo "=== node: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvidia-smi | grep "CUDA Version" || true

if [ ! -x "$VENV/bin/python" ]; then
    echo "=== creating venv $VENV"
    # Prefer the pixi env's python (3.12) as the venv base.
    PY_BASE=$REPO/.pixi/envs/default/bin/python
    if [ ! -x "$PY_BASE" ]; then
        PY_BASE=$(command -v python3.12 || command -v python3 || command -v python)
    fi
    "$PY_BASE" -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    # team_gm kernels are authored against torch 2.8+ / triton 3.4+ (autotuner
    # resolves `key=[...]` constexpr names from kwargs there). cu124 wheels
    # work with the cluster's R550+ driver.
    "$VENV/bin/pip" install --index-url https://download.pytorch.org/whl/cu126 \
        "torch==2.8.0"
    "$VENV/bin/pip" install einops jaxtyping beartype matplotlib numpy
fi

PY=$VENV/bin/python
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

echo "=== running benchmark ..."
$PY $REPO/aug_attn_bench/bench.py --out "$TXT" "$@"
echo "=== done. txt: $TXT"
