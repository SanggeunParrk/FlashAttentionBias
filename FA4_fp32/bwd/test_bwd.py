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


def run_case(B, L, H, D, dtype=torch.float32, atol=5e-3, rtol=5e-3, verbose=True):
    torch.manual_seed(0)
    # Run FA4 in `dtype` (which may be bf16 for baseline checks); compute the
    # reference in fp32 from the same numerical inputs.
    q_dt = torch.randn(B, L, H, D, device="cuda", dtype=dtype)
    k_dt = torch.randn_like(q_dt)
    v_dt = torch.randn_like(q_dt)
    dout_dt = torch.randn_like(q_dt)

    q = q_dt.clone().requires_grad_(True)
    k = k_dt.clone().requires_grad_(True)
    v = v_dt.clone().requires_grad_(True)
    dout = dout_dt.clone()

    out_fa, _ = flash_attn_func(q, k, v)
    out_fa.backward(dout)
    dq_fa, dk_fa, dv_fa = q.grad.clone(), k.grad.clone(), v.grad.clone()

    # Reference path is always in fp32 (cast inputs up). bwd_ref runs autograd
    # on attention_fp32. Compare in fp32 space.
    dq_ref, dk_ref, dv_ref = bwd_ref(q_dt.float(), k_dt.float(), v_dt.float(),
                                     dout_dt.float())

    d = {"dQ": (dq_fa.float(), dq_ref),
         "dK": (dk_fa.float(), dk_ref),
         "dV": (dv_fa.float(), dv_ref)}
    rows = []
    passed = True
    for name, (a, b) in d.items():
        aerr = _max_abs(a, b)
        rerr = _max_rel(a, b)
        ok = aerr < atol and rerr < rtol
        passed &= ok
        rows.append((name, aerr, rerr, ok))

    if verbose:
        tag = f"B={B} L={L} H={H} D={D} dtype={str(dtype).rsplit('.', 1)[-1]}"
        for name, aerr, rerr, ok in rows:
            print(f"  [{tag}] {name}: abs={aerr:.3e} rel={rerr:.3e} "
                  f"{'PASS' if ok else 'FAIL'}")
    return passed


# Tighter atol for bf16 vs fp32 — bf16 has ~1e-2 noise on dQ/dK/dV.
CASES = [
    # (B, L, H, D, dtype, atol, rtol)
    (1, 128, 1,  32, torch.bfloat16, 1e-2, 1e-2),  # smoke
    (1, 256, 4,  64, torch.bfloat16, 1e-2, 1e-2),
    (2, 512, 4, 128, torch.bfloat16, 1e-2, 1e-2),
    (1, 128, 1,  32, torch.float32,  5e-3, 5e-3),  # fp32 smoke
    (1, 256, 4,  64, torch.float32,  5e-3, 5e-3),
    (1, 512, 4,  64, torch.float32,  5e-3, 5e-3),
]


if __name__ == "__main__":
    print("── FA4 bwd vs PyTorch autograd fp32 ref ─────────────────")
    ok = True
    for B, L, H, D, dtype, atol, rtol in CASES:
        try:
            ok &= run_case(B, L, H, D, dtype=dtype, atol=atol, rtol=rtol)
        except Exception as e:
            tag = f"B={B} L={L} H={H} D={D} dtype={str(dtype).rsplit('.', 1)[-1]}"
            print(f"  [{tag}]  EXCEPTION: "
                  f"{type(e).__name__}: {str(e).splitlines()[-1][:200]}")
            ok = False
    print(f"\nresult: {'OK' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
