"""Minimal seqlen info for fixed-length MHA (Q/K/V share seqlen)."""
from dataclasses import dataclass

import cutlass.cute as cute
from cutlass import Int32


@dataclass(frozen=True)
class SeqlenInfoQK:
    seqlen_q: Int32
    seqlen_k: Int32

    # Varlen-related flags kept as constant False shims so the
    # const_expr-dead branches in callers like
    # `if const_expr(not seqlen.has_cu_seqlens_q): ...` resolve their
    # attribute access at Python level. The non-varlen branch always runs.
    has_cu_seqlens_q: bool = False
    has_cu_seqlens_k: bool = False
    has_seqused_q: bool = False
    has_seqused_k: bool = False

    @staticmethod
    def create(batch_idx: Int32, seqlen_q_static: Int32, seqlen_k_static: Int32,
               **_ignored):
        """batch_idx is accepted for call-site compatibility but unused (no varlen).

        `**_ignored` swallows ``mCuSeqlensQ``/``mCuSeqlensK``/``mSeqUsedQ``/
        ``mSeqUsedK`` kwargs from the SM90 fwd ``partial(SeqlenInfoQK.create, ...)``
        — the varlen path was stripped.
        """
        return SeqlenInfoQK(seqlen_q_static, seqlen_k_static)

    def offset_batch_Q(self, mQ: cute.Tensor, batch_idx: Int32, dim: int,
                       **_ignored) -> cute.Tensor:
        idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mQ) - 1 - dim)
        return mQ[idx]

    def offset_batch_K(self, mK: cute.Tensor, batch_idx: Int32, dim: int,
                       **_ignored) -> cute.Tensor:
        idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mK) - 1 - dim)
        return mK[idx]


# Q-only variant used by flash_bwd_preprocess (it doesn't see K).
@dataclass(frozen=True)
class SeqlenInfo:
    seqlen_q: Int32

    has_cu_seqlens_q: bool = False
    has_seqused_q: bool = False

    @staticmethod
    def create(batch_idx: Int32, seqlen_q_static: Int32, **_ignored):
        return SeqlenInfo(seqlen_q_static)

    def offset_batch_Q(self, mQ: cute.Tensor, batch_idx: Int32, dim: int,
                       **_ignored) -> cute.Tensor:
        idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mQ) - 1 - dim)
        return mQ[idx]
