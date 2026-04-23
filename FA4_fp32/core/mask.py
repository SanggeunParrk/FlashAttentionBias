# Copyright (c) 2025, Tri Dao.
"""Right-edge seqlen masking for MHA attention accumulators.

This build supports only non-causal, non-local, no-mask-mod attention with Q/K/V
sharing shape, so the mask reduces to: "set acc_S[i] = -inf for columns of K
beyond seqlen_k".  The helpers below generate a 32-bit R2P bitmask (1=keep,
0=mask) and apply it to the accumulator via PTX R2P.
"""
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, const_expr

from FA4_fp32.core import utils as utils
from FA4_fp32.core.seqlen_info import SeqlenInfoQK

MASK_R2P_CHUNK_SIZE: int = 32


@cute.jit
def r2p_bitmask_below(limit: Int32, s: int) -> Uint32:
    """32-bit R2P bitmask keeping positions < limit (exclusive upper bound).

    Positions 0..limit-1 in chunk `s` get bit=1 (keep), the rest bit=0 (mask).
    Uses inline PTX to avoid shift-by-type-width UB.
    """
    m = max((s + 1) * MASK_R2P_CHUNK_SIZE - limit, 0)
    return utils.shr_u32(Uint32(0xFFFFFFFF), Uint32(m))


@cute.jit
def mask_r2p_lambda(
    X: cute.Tensor,
    mask_gen_fn,
    rank1: bool = False,
) -> None:
    """Apply R2P masking with a custom bitmask generator.

    mask_gen_fn(chunk_idx: constexpr int) -> Uint32:
        Returns a 32-bit bitmask for the chunk. Bit i set means column
        chunk_idx * chunk_size + i is KEPT; bit i clear means masked to -inf.
    """
    ncol = const_expr(cute.size(X.shape[cute.rank(X) - 1]) if not rank1 else cute.size(X.shape))
    CHUNK_SIZE = MASK_R2P_CHUNK_SIZE
    for s in cutlass.range_constexpr(cute.ceil_div(ncol, CHUNK_SIZE)):
        mask = mask_gen_fn(s)
        # range_constexpr is required so the compiler can emit an R2P instruction
        for i in cutlass.range_constexpr(min(CHUNK_SIZE, ncol - s * CHUNK_SIZE)):
            in_bound = cutlass.Boolean(mask & (Uint32(1) << i))
            c = s * CHUNK_SIZE + i
            if const_expr(rank1):
                X[c] = X[c] if in_bound else -Float32.inf
            else:
                for r in cutlass.range_constexpr(cute.size(X.shape[0])):
                    X[r, c] = X[r, c] if in_bound else -Float32.inf


@dataclass(frozen=True)
class AttentionMask:
    """Applies right-K-edge seqlen masking.

    `window_size_left`, `window_size_right`, `qhead_per_kvhead_packgqa`, and
    `swap_AB` are kept as fields so existing `partial(AttentionMask, ...)` call
    sites don't break, but they are all constants in this build.
    """
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    seqlen_info: SeqlenInfoQK
    window_size_left: cutlass.Constexpr = None
    window_size_right: cutlass.Constexpr = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    swap_AB: cutlass.Constexpr[bool] = False

    @property
    def seqlen_q(self) -> Int32:
        return self.seqlen_info.seqlen_q

    @property
    def seqlen_k(self) -> Int32:
        return self.seqlen_info.seqlen_k

    @cute.jit
    def apply_mask_sm100(
        self,
        acc_S: cute.Tensor,
        m_block: Int32,
        n_block: Int32,
        thr_mma: cute.TiledMma,
        thr_tmem_load: cute.TiledCopy,
        mask_seqlen: cutlass.Constexpr[bool],
        # Deprecated constexpr kwargs kept for caller-signature compat.
        mask_causal: cutlass.Constexpr[bool] = False,
        mask_local: cutlass.Constexpr[bool] = False,
        mask_mod=None,
        batch_idx: Int32 = None,
        head_idx: Int32 = None,
        aux_tensors=None,
        fastdiv_mods=(None, None),
        head_divmod=None,
        check_q_boundary: bool = False,
    ) -> None:
        """Forward pass: mask S = Q @ K.T so columns >= seqlen_k are -inf."""
        if const_expr(not mask_seqlen):
            return
        # TMEM-load coordinates give us the thread-local column offset.
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        tScS = thr_mma.partition_C(cS)[(None, None), 0, 0]
        tScS_t2r = thr_tmem_load.partition_D(tScS)  # noqa: F841 (forces compile-time shape)
        if n_block < 0:
            n_block = 0
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n
        mask_r2p_lambda(
            acc_S,
            lambda s: r2p_bitmask_below(seqlenk_col_limit, s),
            rank1=True,
        )

    @cute.jit
    def apply_mask_sm100_transposed(
        self,
        acc_S: cute.Tensor,
        tScS_t2r: cute.Tensor,
        t0ScS_t2r: cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        mask_seqlen: cutlass.Constexpr,
        # Deprecated constexpr kwargs kept for caller-signature compat.
        mask_causal: cutlass.Constexpr = False,
        mask_local: cutlass.Constexpr = False,
        mask_mod=None,
        batch_idx: Int32 = None,
        head_idx: Int32 = None,
        aux_tensors=None,
        fastdiv_mods=(None, None),
        is_full_block: bool = False,
        check_m_boundary: bool = True,
    ) -> None:
        """Backward pass: S = K @ Q.T where cols correspond to Q and rows to K.

        With Q/K/V sharing seqlen and no causal/local/mask_mod, the only mask
        we need is: if the entire tile is past seqlen_k (seqlenk_col_limit<=0)
        the whole accumulator is OOB — nuke it.
        """
        if const_expr(not mask_seqlen):
            return
        COL = 1 if const_expr(not self.swap_AB) else 0
        thr_col_offset = tScS_t2r[0][COL]
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n - thr_col_offset
        if seqlenk_col_limit <= 0:
            for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                acc_S[i] = -cutlass.Float32.inf
