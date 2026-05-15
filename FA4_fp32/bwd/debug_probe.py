"""Bug-localization probes for fp32 bwd.

Each probe sets up a controlled scenario (Q/K/V/dout with structured content)
and prints the actual FA4 output vs. the closed-form expected result, so we
can see WHERE in the (m, n, d) lattice the values go wrong.

Strategy: pick inputs where dV/dK/dQ have known structure (e.g., all-equal
rows, single nonzero column, etc.). If the FA4 output has a different
structure than expected, the *pattern* of the deviation tells us which
pipeline stage permutes data.

Run with: python FA4_fp32/bwd/debug_probe.py [probe_name]
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from FA4_fp32 import flash_attn_func


def _fa_bwd(q, k, v, dout):
    q = q.clone().requires_grad_(True)
    k = k.clone().requires_grad_(True)
    v = v.clone().requires_grad_(True)
    out, _ = flash_attn_func(q, k, v)
    out.backward(dout)
    return q.grad.detach(), k.grad.detach(), v.grad.detach()


def _ref_bwd(q, k, v, dout, scale=None):
    # Plain fp32 reference: compute attention then autograd.
    qf = q.float().detach().requires_grad_(True)
    kf = k.float().detach().requires_grad_(True)
    vf = v.float().detach().requires_grad_(True)
    if scale is None:
        scale = qf.shape[-1] ** -0.5
    s = torch.einsum("bqhd,bkhd->bhqk", qf, kf) * scale
    p = torch.softmax(s, dim=-1)
    out_ref = torch.einsum("bhqk,bkhd->bqhd", p, vf)
    out_ref.backward(dout.float())
    return qf.grad.detach(), kf.grad.detach(), vf.grad.detach()


def _print_tensor(name, t, idx=(0, slice(None), 0, slice(None))):
    sub = t[idx]
    print(f"  {name} shape={tuple(t.shape)}  view{idx}:")
    print(f"    {sub.cpu().numpy()}")


def _print_row_compare(name, actual, expected, rows=(0, 1, 2, 31, 32, 33, 63)):
    print(f"  {name}: row-by-row first 4 cols (FA4 / ref / diff)")
    for r in rows:
        if r >= actual.shape[1]:
            continue
        a = actual[0, r, 0, :4].float().cpu().tolist()
        e = expected[0, r, 0, :4].float().cpu().tolist()
        d = [ai - ei for ai, ei in zip(a, e)]
        print(f"    row {r:3d}:  FA={[f'{x:+.3f}' for x in a]}")
        print(f"             ref={[f'{x:+.3f}' for x in e]}")
        print(f"             d ={[f'{x:+.3f}' for x in d]}")


def probe_zero_q(B=1, L=64, H=1, D=32):
    """Q=0 → S=0 → P=1/L uniform. dV[j, d] = (1/L)*sum_i dO[i, d] (const per j).
    dK = dS@Q = 0. dQ = dP@K, where dP = dO@V^T."""
    print(f"\n--- probe_zero_q  B={B} L={L} H={H} D={D} fp32 ---")
    torch.manual_seed(0)
    q = torch.zeros(B, L, H, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, L, H, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, L, H, D, device="cuda", dtype=torch.float32)
    dout = torch.randn(B, L, H, D, device="cuda", dtype=torch.float32)

    dq_fa, dk_fa, dv_fa = _fa_bwd(q, k, v, dout)
    dq_ref, dk_ref, dv_ref = _ref_bwd(q, k, v, dout)

    print(f"  dV ref row-0 first 4: {dv_ref[0,0,0,:4].cpu().tolist()}")
    print(f"  dV ref row-31 first 4: {dv_ref[0,31,0,:4].cpu().tolist()}")
    print(f"  dV ref row-63 first 4: {dv_ref[0,63,0,:4].cpu().tolist()}")
    _print_row_compare("dV", dv_fa, dv_ref, rows=(0, 1, 31, 32, 63))
    _print_row_compare("dK", dk_fa, dk_ref, rows=(0, 1, 31, 32, 63))
    _print_row_compare("dQ", dq_fa, dq_ref, rows=(0, 1, 31, 32, 63))


def probe_small_q(B=1, L=64, H=1, D=32, q_scale=1e-3):
    """|Q| tiny → |S|≪LSE → P nearly uniform. Same dV story as Q=0 but
    avoids any LSE-saturation pathology."""
    print(f"\n--- probe_small_q  B={B} L={L} H={H} D={D} q_scale={q_scale} fp32 ---")
    torch.manual_seed(0)
    q = torch.randn(B, L, H, D, device="cuda", dtype=torch.float32) * q_scale
    k = torch.randn(B, L, H, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, L, H, D, device="cuda", dtype=torch.float32)
    dout = torch.randn(B, L, H, D, device="cuda", dtype=torch.float32)

    dq_fa, dk_fa, dv_fa = _fa_bwd(q, k, v, dout)
    dq_ref, dk_ref, dv_ref = _ref_bwd(q, k, v, dout)
    _print_row_compare("dV", dv_fa, dv_ref, rows=(0, 1, 31, 32, 63))
    _print_row_compare("dK", dk_fa, dk_ref, rows=(0, 1, 31, 32, 63))
    _print_row_compare("dQ", dq_fa, dq_ref, rows=(0, 1, 31, 32, 63))


def probe_single_v(B=1, L=64, H=1, D=32):
    """V is zero except V[0, 0, 0, 0] = 1. Then output[i, 0] = P[i, 0],
    dV[0, 0] = sum_i (P[i, 0] * dout[i, 0]). Other dV elements... harder.
    dK depends on dS = P*(dP-D) with dP = dO@V^T. With V single-nonzero,
    dP[i, j] = dO[i, 0] * 1{j==0}. Sparse structure → easy to read."""
    print(f"\n--- probe_single_v  B={B} L={L} H={H} D={D} fp32 ---")
    torch.manual_seed(0)
    q = torch.randn(B, L, H, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, L, H, D, device="cuda", dtype=torch.float32)
    v = torch.zeros(B, L, H, D, device="cuda", dtype=torch.float32)
    v[0, 0, 0, 0] = 1.0
    dout = torch.randn(B, L, H, D, device="cuda", dtype=torch.float32)

    dq_fa, dk_fa, dv_fa = _fa_bwd(q, k, v, dout)
    dq_ref, dk_ref, dv_ref = _ref_bwd(q, k, v, dout)
    _print_row_compare("dV", dv_fa, dv_ref, rows=(0, 1, 31, 32, 63))
    _print_row_compare("dK", dk_fa, dk_ref, rows=(0, 1, 31, 32, 63))


def probe_eye_dout(B=1, L=64, H=1, D=32):
    """dout = identity-like: dout[i, d] = 1{i==d} (when L>=D). Lets us see
    which (i, d) cell of dV gets which value."""
    print(f"\n--- probe_eye_dout  B={B} L={L} H={H} D={D} fp32 ---")
    torch.manual_seed(0)
    q = torch.randn(B, L, H, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, L, H, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, L, H, D, device="cuda", dtype=torch.float32)
    dout = torch.zeros(B, L, H, D, device="cuda", dtype=torch.float32)
    for i in range(min(L, D)):
        dout[0, i, 0, i] = 1.0

    dq_fa, dk_fa, dv_fa = _fa_bwd(q, k, v, dout)
    dq_ref, dk_ref, dv_ref = _ref_bwd(q, k, v, dout)
    _print_row_compare("dV", dv_fa, dv_ref, rows=(0, 1, 15, 16, 31, 32, 63))


PROBES = {
    "zero_q":     probe_zero_q,
    "small_q":    probe_small_q,
    "single_v":   probe_single_v,
    "eye_dout":   probe_eye_dout,
}


if __name__ == "__main__":
    names = sys.argv[1:] or list(PROBES.keys())
    for n in names:
        if n not in PROBES:
            print(f"unknown probe: {n}")
            continue
        PROBES[n]()
