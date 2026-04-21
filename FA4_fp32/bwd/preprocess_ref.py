"""Naive PyTorch reference for FA4 backward preprocess.

The FA4 backward preprocess kernel computes three things per batch row:

    dpsum[b, h, i]  = sum_d O[b, i, h, d] * dO[b, i, h, d]  (in fp32)
                      [- dLSE[b, h, i] if dLSE is given]
    lse_log2[b, h, i] = LSE[b, h, i] * log2(e)
    dq_accum              = zeros of shape (B, H, L_rounded * D_rounded)

Outputs are padded up to m_block_size (rows) and 32 (head_dim). Padded rows
are zero-filled in dpsum / lse_log2.

This naive reference is only for correctness testing. The real kernel is
[FA4_fp32.kernels.flash_bwd_preprocess].
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

LOG2_E = 1.0 / math.log(2.0)  # == log2(e)


def _round_up(x: int, mult: int) -> int:
    return (x + mult - 1) // mult * mult


def bwd_preprocess_ref(
    O: torch.Tensor,                # (B, L, H, D_v), any float dtype
    dO: torch.Tensor,               # (B, L, H, D_v), same dtype as O
    LSE: torch.Tensor,              # (B, H, L), fp32
    m_block_size: int,              # rounding for rows
    head_dim_rounded: Optional[int] = None,  # rounding for dq_accum D dim, default ceil(D/32)*32
    dLSE: Optional[torch.Tensor] = None,     # (B, H, L), fp32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (dpsum, lse_log2, dq_accum).

    Shapes:
        dpsum:    (B, H, L_rounded), fp32
        lse_log2: (B, H, L_rounded), fp32
        dq_accum: (B, H, L_rounded * D_rounded), fp32 (all zeros)
    """
    assert O.dtype == dO.dtype
    assert O.shape == dO.shape
    B, L, H, Dv = O.shape
    assert LSE.shape == (B, H, L)
    assert LSE.dtype == torch.float32
    if dLSE is not None:
        assert dLSE.shape == (B, H, L)
        assert dLSE.dtype == torch.float32

    L_rnd = _round_up(L, m_block_size)
    if head_dim_rounded is None:
        head_dim_rounded = _round_up(Dv, 32)

    # rowsum(O * dO) in fp32. Cast inputs for precision before reducing.
    O_f = O.to(torch.float32)
    dO_f = dO.to(torch.float32)
    # (B, L, H) -> (B, H, L)
    dpsum_unpad = (O_f * dO_f).sum(dim=-1).permute(0, 2, 1).contiguous()
    if dLSE is not None:
        dpsum_unpad = dpsum_unpad - dLSE

    dpsum = dpsum_unpad.new_zeros(B, H, L_rnd)
    dpsum[:, :, :L] = dpsum_unpad

    lse_log2 = LSE.new_zeros(B, H, L_rnd)
    lse_log2[:, :, :L] = LSE * LOG2_E

    dq_accum = torch.zeros(B, H, L_rnd * head_dim_rounded,
                           dtype=torch.float32, device=O.device)

    return dpsum, lse_log2, dq_accum
