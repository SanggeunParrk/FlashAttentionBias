"""End-to-end bwd test: FA4 fp32 backward vs PyTorch autograd reference.

Calls the FA4 public entry (`flash_attn_func` with `requires_grad`) which
runs preprocess → main compute → dQ reduce → postprocess. Compares dQ/dK/dV
against [FA4_fp32.bwd.bwd_ref.bwd_ref].

The point of this file is to (a) let us see *which kernel phase* breaks
first when turning on fp32 bwd, and (b) validate end-to-end correctness
once all phases work.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from FA4_fp32 import flash_attn_func
from FA4_fp32.bwd.bwd_ref import bwd_ref


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def _max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = b.abs().max().item()
    return 0.0 if denom == 0.0 else _max_abs(a, b) / denom


def run_case(B, L, H, D, causal, atol=5e-3, rtol=5e-3, verbose=True):
    torch.manual_seed(0)
    q = torch.randn(B, L, H, D, device="cuda", dtype=torch.float32, requires_grad=True)
    k = torch.randn_like(q).detach().requires_grad_(True)
    v = torch.randn_like(q).detach().requires_grad_(True)
    dout = torch.randn_like(q.detach())

    # FA4 path
    out_fa, _ = flash_attn_func(q, k, v, causal=causal)
    out_fa.backward(dout)
    dq_fa, dk_fa, dv_fa = q.grad.clone(), k.grad.clone(), v.grad.clone()

    # Reference path (autograd on fp32 math-attention).
    dq_ref, dk_ref, dv_ref = bwd_ref(q.detach(), k.detach(), v.detach(), dout,
                                     causal=causal)

    d = {"dQ": (dq_fa, dq_ref), "dK": (dk_fa, dk_ref), "dV": (dv_fa, dv_ref)}
    rows = []
    passed = True
    for name, (a, b) in d.items():
        aerr = _max_abs(a, b)
        rerr = _max_rel(a, b)
        ok = aerr < atol and rerr < rtol
        passed &= ok
        rows.append((name, aerr, rerr, ok))

    if verbose:
        tag = f"B={B} L={L} H={H} D={D} causal={causal}"
        for name, aerr, rerr, ok in rows:
            print(f"  [{tag}] {name}: abs={aerr:.3e} rel={rerr:.3e} "
                  f"{'PASS' if ok else 'FAIL'}")
    return passed


CASES = [
    # B, L, H, D, causal
    (1, 256, 4, 128, False),
    (1, 256, 4, 128, True),
    (2, 512, 4, 128, False),
    (1, 512, 4,  64, False),
    (1, 512, 4,  32, False),
    (1, 512, 2,  96, False),
]


if __name__ == "__main__":
    print("── FA4 fp32 bwd vs PyTorch autograd ref ─────────────────")
    ok = True
    for c in CASES:
        try:
            ok &= run_case(*c)
        except Exception as e:
            print(f"  [B={c[0]} L={c[1]} H={c[2]} D={c[3]} causal={c[4]}]  "
                  f"EXCEPTION: {type(e).__name__}: {str(e).splitlines()[-1][:200]}")
            ok = False
    print(f"\nresult: {'OK' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
