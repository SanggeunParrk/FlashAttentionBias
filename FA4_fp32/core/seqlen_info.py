"""Minimal seqlen info for fixed-length MHA (Q/K/V share seqlen)."""
from dataclasses import dataclass

import cutlass.cute as cute
from cutlass import Int32


@dataclass(frozen=True)
class SeqlenInfoQK:
    seqlen_q: Int32
    seqlen_k: Int32

    @staticmethod
    def create(batch_idx: Int32, seqlen_q_static: Int32, seqlen_k_static: Int32):
        """batch_idx is accepted for call-site compatibility but unused (no varlen)."""
        return SeqlenInfoQK(seqlen_q_static, seqlen_k_static)

    def offset_batch_Q(self, mQ: cute.Tensor, batch_idx: Int32, dim: int) -> cute.Tensor:
        idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mQ) - 1 - dim)
        return mQ[idx]

    def offset_batch_K(self, mK: cute.Tensor, batch_idx: Int32, dim: int) -> cute.Tensor:
        idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mK) - 1 - dim)
        return mK[idx]
