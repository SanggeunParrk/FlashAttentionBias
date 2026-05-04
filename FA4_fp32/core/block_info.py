# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# B200 build: non-causal, non-local, no split-KV, no Pack-GQA. BlockInfo
# degenerates to "(0, ceil(seqlen / tile))" so we keep just the tile sizes.
from typing import Tuple
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32

from FA4_fp32.core.seqlen_info import SeqlenInfoQK


@dataclass(frozen=True)
class BlockInfo:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]

    @cute.jit
    def get_n_block_min_max(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,  # noqa: ARG002 — kept for API symmetry; unused without causal/local
    ) -> Tuple[Int32, Int32]:
        n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.tile_n)
        return Int32(0), n_block_max

    @cute.jit
    def get_m_block_min_max(
        self,
        seqlen_info: SeqlenInfoQK,
        n_block: Int32,  # noqa: ARG002 — kept for API symmetry; unused without causal/local
    ) -> Tuple[Int32, Int32]:
        m_block_max = cute.ceil_div(seqlen_info.seqlen_q, self.tile_m)
        return Int32(0), m_block_max
