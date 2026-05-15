"""Q=0 diagnostic: with Q=0, S=Q@K^T=0, LSE = log(N), P = 1/N uniform.

Then dV[j, d] = (1/N) * sum_i dO[i, d] -- a known constant, independent of K, V.

If fp32 dV matches that constant, the dV MMA / TMEM A-operand / epilogue are
correct, and the bug is isolated to the P-computation path (LSE/D pairing).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from FA4_fp32 import flash_attn_func


def run(B, L, H, D, dtype):
    torch.manual_seed(0)
    q = torch.zeros(B, L, H, D, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(B, L, H, D, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(B, L, H, D, device="cuda", dtype=dtype, requires_grad=True)
    dout = torch.randn(B, L, H, D, device="cuda", dtype=dtype)

    out, _ = flash_attn_func(q, k, v)
    out.backward(dout)

    # Expected: P uniform 1/L  =>  dV[b, j, h, d] = mean over i of dO[b, i, h, d]
    dv_expected = dout.float().mean(dim=1, keepdim=True).expand_as(v).contiguous()
    dv_actual = v.grad.float()

    err = (dv_actual - dv_expected).abs()
    tag = f"B={B} L={L} H={H} D={D} dtype={str(dtype).rsplit('.', 1)[-1]}"
    print(f"[{tag}] dV vs expected (uniform-P):")
    print(f"  max abs err: {err.max().item():.3e}")
    print(f"  mean abs err: {err.mean().item():.3e}")
    print(f"  expected[0,0,0,:8]: {dv_expected[0, 0, 0, :8].tolist()}")
    print(f"  actual  [0,0,0,:8]: {dv_actual[0, 0, 0, :8].tolist()}")
    # Sample a couple of other rows j (along seqlen). They should all be the same.
    if L >= 30:
        print(f"  actual  [0,30,0,:8]: {dv_actual[0, 30, 0, :8].tolist()}")


if __name__ == "__main__":
    print("── Q=0 diagnostic: dV should equal (1/L) * mean(dO) per col ─────")
    for cfg in [
        (1, 64, 1, 32, torch.float32),
        (1, 128, 1, 32, torch.float32),
        (1, 256, 4, 64, torch.float32),
    ]:
        run(*cfg)
        print()
