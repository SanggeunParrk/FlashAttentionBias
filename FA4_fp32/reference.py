"""fp32 attention reference with TF32 tensor cores enabled internally.

Input layout: (A, B, L, H, D) where A is an augmentation dimension.
Bias not supported yet — will be added in a later step.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

# Enable TF32 for all matmul paths used here.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _softmax_scale(d: int, scale: Optional[float]) -> float:
    return 1.0 / math.sqrt(d) if scale is None else float(scale)


def attention_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
):
    """fp32 exact attention. Internal matmuls use TF32 (tensor cores on B200).

    Args:
        q, k, v: (A, B, L, H, D) fp32, contiguous last dim.
        causal:  if True, apply standard causal mask (query attends to <= key).
        softmax_scale: defaults to 1/sqrt(D).
        return_lse: if True also return LSE = log(sum(exp(scores))) of shape
                    (A, B, H, L).

    Returns:
        out: (A, B, L, H, D) fp32.
        [lse]: optional, shape (A, B, H, L), fp32.
    """
    assert q.dtype == k.dtype == v.dtype == torch.float32, "fp32 only"
    assert q.shape == k.shape == v.shape, "Q/K/V must share shape"
    assert q.dim() == 5, "expected (A, B, L, H, D)"
    A, B, L, H, D = q.shape
    scale = _softmax_scale(D, softmax_scale)

    # Flatten (A, B) -> single batch for a single bmm call.
    # Then move H before L to get (A*B, H, L, D) — the canonical attention layout.
    q_ = q.reshape(A * B, L, H, D).transpose(1, 2).contiguous()
    k_ = k.reshape(A * B, L, H, D).transpose(1, 2).contiguous()
    v_ = v.reshape(A * B, L, H, D).transpose(1, 2).contiguous()

    # scores: (A*B, H, Lq, Lk)
    scores = torch.matmul(q_, k_.transpose(-2, -1)) * scale

    if causal:
        mask = torch.ones(L, L, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))

    if return_lse:
        lse = torch.logsumexp(scores, dim=-1)  # (A*B, H, Lq)

    probs = torch.softmax(scores, dim=-1)
    out_ = torch.matmul(probs, v_)            # (A*B, H, Lq, D)

    out = out_.transpose(1, 2).reshape(A, B, L, H, D).contiguous()
    if return_lse:
        lse = lse.reshape(A, B, H, L).contiguous()
        return out, lse
    return out
