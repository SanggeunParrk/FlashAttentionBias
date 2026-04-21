"""Naive PyTorch reference for the FA4 backward pass (dQ, dK, dV).

Uses the forward-attention reference from FA4_fp32.reference and PyTorch
autograd — no hand-written dS / dP math. This is the ground truth the full
FA4 backward (preprocess + main compute + dQ reduce + postprocess) must
match at the dQ/dK/dV level.

Internal matmuls use TF32 tensor cores (torch.backends.cuda.matmul.allow_tf32
is enabled when FA4_fp32.reference is imported).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from FA4_fp32.reference import attention_fp32


def bwd_ref(
    q: torch.Tensor,              # (B, L, H, D), fp32
    k: torch.Tensor,              # (B, L, H, D), fp32
    v: torch.Tensor,              # (B, L, H, D), fp32
    dout: torch.Tensor,           # (B, L, H, D), fp32
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute dQ, dK, dV via autograd on the fp32 forward reference.

    Accepts 4D (B, L, H, D) inputs. Returns grads of the same shape.
    """
    assert q.dtype == k.dtype == v.dtype == dout.dtype == torch.float32
    assert q.shape == k.shape == v.shape == dout.shape
    assert q.dim() == 4, "expected (B, L, H, D)"

    q_ = q.detach().clone().requires_grad_(True)
    k_ = k.detach().clone().requires_grad_(True)
    v_ = v.detach().clone().requires_grad_(True)
    # reference expects (A, B, L, H, D); use A=1.
    out = attention_fp32(q_.unsqueeze(0), k_.unsqueeze(0), v_.unsqueeze(0),
                         causal=causal, softmax_scale=softmax_scale).squeeze(0)
    out.backward(dout)
    return q_.grad, k_.grad, v_.grad
