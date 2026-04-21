"""Correctness tests for the backward preprocess.

Two layers of check:

1. Self-consistency — the naive reference computes the same thing as an
   einsum rewrite. If this disagrees we have a bug in the reference itself.

2. FA4 vs naive — compile and run the actual FA4 preprocess kernel via the
   private interface entry and compare outputs with the naive reference.
   Currently only exercised for bf16 (fp32 FA4 preprocess not yet enabled
   end-to-end — that's the next step).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from FA4_fp32.bwd.preprocess_ref import bwd_preprocess_ref, LOG2_E


# ── 1) naive vs einsum self-check ─────────────────────────────────────────

def naive_einsum(O, dO, LSE, m_block_size, head_dim_rounded=None, dLSE=None):
    B, L, H, Dv = O.shape
    L_rnd = (L + m_block_size - 1) // m_block_size * m_block_size
    if head_dim_rounded is None:
        head_dim_rounded = (Dv + 31) // 32 * 32
    dpsum = torch.einsum("blhd,blhd->bhl", O.float(), dO.float())
    if dLSE is not None:
        dpsum = dpsum - dLSE
    padded_dpsum = torch.zeros(B, H, L_rnd, device=O.device)
    padded_dpsum[:, :, :L] = dpsum
    padded_lse = torch.zeros(B, H, L_rnd, device=O.device)
    padded_lse[:, :, :L] = LSE * LOG2_E
    dq = torch.zeros(B, H, L_rnd * head_dim_rounded, device=O.device)
    return padded_dpsum, padded_lse, dq


def test_self_consistency(verbose=True):
    torch.manual_seed(0)
    cases = [
        # B, L, H, D, m_block, dtype, dLSE
        (2, 128, 4, 64,  128, torch.float32, False),
        (1, 200, 4, 96,  128, torch.bfloat16, True),
        (2, 512, 8, 128, 128, torch.float16, False),
        (3, 300, 2, 64,  64,  torch.float32, True),
    ]
    ok = True
    for B, L, H, D, m, dt, use_dlse in cases:
        O  = torch.randn(B, L, H, D, device="cuda", dtype=dt)
        dO = torch.randn(B, L, H, D, device="cuda", dtype=dt)
        LSE = torch.randn(B, H, L, device="cuda", dtype=torch.float32)
        dLSE = (torch.randn(B, H, L, device="cuda", dtype=torch.float32)
                if use_dlse else None)

        a = bwd_preprocess_ref(O, dO, LSE, m, dLSE=dLSE)
        b = naive_einsum(O, dO, LSE, m, dLSE=dLSE)
        # Reduction-order differences are fine; bound at float32 rounding.
        diffs = [(a[i] - b[i]).abs().max().item() for i in range(3)]
        passed = diffs[0] < 1e-4 and diffs[1] == 0.0 and diffs[2] == 0.0
        ok &= passed
        if verbose:
            print(f"[self] B={B} L={L} H={H} D={D} m={m} {dt}"
                  f" dLSE={use_dlse}  diffs={diffs}  "
                  f"{'PASS' if passed else 'FAIL'}")
    return ok


# ── 2) FA4 vs naive — kernel correctness (bf16 only for now) ──────────────

def test_fa4_vs_ref_bf16(verbose=True):
    """Drive the FA4 preprocess kernel directly and compare to the reference.

    Uses the internal `_bwd_preprocess` helper from FA4_fp32.interface.
    """
    from FA4_fp32.interface import _bwd_preprocess
    from FA4_fp32.infra.cute_dsl_utils import torch2cute_dtype_map

    torch.manual_seed(0)
    cases = [
        # B, L, H, D, m_block, dtype
        (2, 512, 4, 64,  128, torch.bfloat16),
        (1, 256, 8, 128, 128, torch.bfloat16),
        (2, 200, 2, 64,  128, torch.bfloat16),  # non-divisible L
        # fp32
        (2, 512, 4, 64,  128, torch.float32),
        (1, 256, 8, 128, 128, torch.float32),
        (2, 200, 2, 64,  128, torch.float32),
        (1, 128, 4, 32,  128, torch.float32),
        (1, 512, 2, 96,  128, torch.float32),
    ]
    ok = True
    for B, L, H, D, m, dt in cases:
        O  = torch.randn(B, L, H, D, device="cuda", dtype=dt)
        dO = torch.randn(B, L, H, D, device="cuda", dtype=dt)
        LSE = torch.randn(B, H, L, device="cuda", dtype=torch.float32)

        # Reference outputs
        dp_ref, lse2_ref, dq_ref = bwd_preprocess_ref(O, dO, LSE, m)

        # FA4 outputs — allocate with the same shapes the interface expects.
        L_rnd = dp_ref.shape[-1]
        D_rnd = dq_ref.shape[-1] // L_rnd
        dpsum    = torch.empty_like(dp_ref)
        lse_log2 = torch.empty_like(lse2_ref)
        dq_accum = torch.empty_like(dq_ref)

        cute_dt = torch2cute_dtype_map[dt]
        _bwd_preprocess(
            O, dO, dpsum, LSE, lse_log2, dq_accum,
            None, None, None,       # cu_seqlens_q, seqused_q, dlse
            cute_dt, D, D, m,       # dtype, head_dim, head_dim_v, m_block_size
        )

        # Compare only the valid (un-padded) rows for dpsum/lse_log2.
        # Kernel leaves the padding region unspecified.
        d_dp  = (dpsum[:, :, :L]    - dp_ref [:, :, :L]).abs().max().item()
        d_lse = (lse_log2[:, :, :L] - lse2_ref[:, :, :L]).abs().max().item()
        d_dq  = (dq_accum - dq_ref).abs().max().item()
        passed = d_dp < 1e-3 and d_lse < 1e-6 and d_dq == 0.0
        ok &= passed
        if verbose:
            tag = str(dt).replace("torch.", "")
            print(f"[fa4 {tag}] B={B} L={L} H={H} D={D} m={m}  "
                  f"dpsum={d_dp:.3e}  lse2={d_lse:.3e}  dq={d_dq:.3e}  "
                  f"{'PASS' if passed else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("── self-consistency (ref vs einsum) ─────────────────────")
    ok1 = test_self_consistency()
    print("\n── FA4 kernel vs reference (bf16 & fp32) ────────────────")
    ok2 = test_fa4_vs_ref_bf16()
    print(f"\nresult: self={'OK' if ok1 else 'FAIL'}  "
          f"fa4_kernel={'OK' if ok2 else 'FAIL'}")
    sys.exit(0 if (ok1 and ok2) else 1)
