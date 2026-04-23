"""Minimal seqlen info for fixed-length MHA (Q/K/V share seqlen).

With no varlen support, SeqlenInfoQK degenerates into a two-scalar dataclass.
The former `offset_batch_Q/K` methods now just do dense indexing: in fixed-
length mode the batch dim is at a known position and we slice it directly.

All `has_cu_seqlens_*` / `has_seqused_*` fields remain as compile-time `False`
constants — kernels still `const_expr` against them to prune dead code.
"""
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr  # noqa: F401


@dataclass(frozen=True)
class SeqlenInfoQK:
    seqlen_q: Int32
    seqlen_k: Int32

    # Compile-time constants: varlen is unsupported in this build.
    has_cu_seqlens_q: cutlass.Constexpr[bool] = False
    has_cu_seqlens_k: cutlass.Constexpr[bool] = False
    has_seqused_q: cutlass.Constexpr[bool] = False
    has_seqused_k: cutlass.Constexpr[bool] = False

    @staticmethod
    def create(
        batch_idx: Int32,
        seqlen_q_static: Int32,
        seqlen_k_static: Int32,
        **_ignored,
    ):
        """batch_idx is accepted for call-site compatibility but unused (no varlen)."""
        return SeqlenInfoQK(seqlen_q_static, seqlen_k_static)

    def offset_batch_Q(
        self,
        mQ: cute.Tensor,
        batch_idx: Int32,
        dim: int,
        **_ignored,
    ) -> cute.Tensor:
        """Dense fixed-length indexing: slice the batch dim at position `dim`."""
        idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mQ) - 1 - dim)
        return mQ[idx]

    def offset_batch_K(
        self,
        mK: cute.Tensor,
        batch_idx: Int32,
        dim: int,
        **_ignored,
    ) -> cute.Tensor:
        """Dense fixed-length indexing: slice the batch dim at position `dim`."""
        idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mK) - 1 - dim)
        return mK[idx]
