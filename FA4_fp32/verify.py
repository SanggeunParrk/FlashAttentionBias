"""Verification harness for fp32 attention kernels.

Usage:
    from FA4_fp32.verify import compare
    compare(my_kernel_fn)

`compare` runs forward + backward of a candidate implementation against the
reference in `reference.py` on a grid of shapes and prints max abs diffs.
Any provided candidate must match the reference signature:

    def candidate(q, k, v, causal=False, softmax_scale=None) -> out

Gradients are obtained via torch.autograd (so the candidate must be autograd-
compatible, e.g. a torch.autograd.Function or a pytorch-composed call).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from FA4_fp32.reference import attention_fp32


@dataclass
class Shape:
    A: int
    B: int
    L: int
    H: int
    D: int
    causal: bool = False

    def __str__(self):
        c = "causal" if self.causal else "full"
        return f"A={self.A} B={self.B} L={self.L} H={self.H} D={self.D} {c}"


DEFAULT_SHAPES = [
    Shape(A=1, B=2, L=128,  H=4, D=32,  causal=False),
    Shape(A=2, B=2, L=128,  H=4, D=32,  causal=False),
    Shape(A=2, B=2, L=256,  H=8, D=64,  causal=False),
    Shape(A=2, B=2, L=256,  H=8, D=64,  causal=True),
    Shape(A=1, B=1, L=512,  H=4, D=128, causal=False),
    Shape(A=2, B=1, L=512,  H=4, D=128, causal=True),
]


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def _rand(shape, device, dtype=torch.float32, requires_grad=True, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    t = torch.randn(*shape, generator=g, device=device, dtype=dtype)
    if requires_grad:
        t = t.detach().requires_grad_(True)
    return t


def compare(
    candidate: Callable,
    shapes=DEFAULT_SHAPES,
    device: str = "cuda",
    atol_fwd: float = 5e-4,
    atol_bwd: float = 2e-3,
    softmax_scale: Optional[float] = None,
    verbose: bool = True,
) -> bool:
    """Compare `candidate` against `attention_fp32` reference.

    Returns True iff every shape passes both fwd and bwd tolerance checks.
    Tolerances are generous enough to accommodate TF32 matmul differences.
    """
    all_ok = True
    header = f"{'shape':<40} {'fwd':>12} {'dQ':>12} {'dK':>12} {'dV':>12}  ok"
    if verbose:
        print(header)
        print("-" * len(header))

    for sh in shapes:
        shape = (sh.A, sh.B, sh.L, sh.H, sh.D)
        q_ref = _rand(shape, device, seed=1)
        k_ref = _rand(shape, device, seed=2)
        v_ref = _rand(shape, device, seed=3)
        g     = _rand(shape, device, requires_grad=False, seed=4)

        # candidate uses clones so gradients land on its own leaves.
        q_c = q_ref.detach().clone().requires_grad_(True)
        k_c = k_ref.detach().clone().requires_grad_(True)
        v_c = v_ref.detach().clone().requires_grad_(True)

        out_ref = attention_fp32(
            q_ref, k_ref, v_ref,
            causal=sh.causal, softmax_scale=softmax_scale,
        )
        out_c = candidate(
            q_c, k_c, v_c,
            causal=sh.causal, softmax_scale=softmax_scale,
        )

        fwd_err = _max_abs(out_ref, out_c)

        out_ref.backward(g)
        out_c.backward(g)

        dq_err = _max_abs(q_ref.grad, q_c.grad)
        dk_err = _max_abs(k_ref.grad, k_c.grad)
        dv_err = _max_abs(v_ref.grad, v_c.grad)

        ok = (fwd_err <= atol_fwd
              and dq_err <= atol_bwd
              and dk_err <= atol_bwd
              and dv_err <= atol_bwd)
        all_ok &= ok

        if verbose:
            flag = "PASS" if ok else "FAIL"
            print(f"{str(sh):<40} {fwd_err:>12.3e} {dq_err:>12.3e} "
                  f"{dk_err:>12.3e} {dv_err:>12.3e}  {flag}")

    if verbose:
        print("-" * len(header))
        print("ALL PASS" if all_ok else "SOME FAILURES")
    return all_ok


if __name__ == "__main__":
    # Sanity check: compare the reference against itself — every error should
    # be exactly 0. Useful to confirm the harness plumbing is correct.
    print("Self-check (reference vs reference):")
    compare(attention_fp32, atol_fwd=0.0, atol_bwd=0.0)
