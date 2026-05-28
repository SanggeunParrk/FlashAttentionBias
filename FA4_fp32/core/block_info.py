# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Adapted from B200's stripped block_info. H100 build is non-causal, non-local,
# no split-KV, no Pack-GQA — BlockInfo degenerates to "(0, ceil(seqlen / tile))".
from typing import Optional, Tuple
from dataclasses import dataclass, field

import cutlass
import cutlass.cute as cute
from cutlass import Int32

from FA4_fp32.core.seqlen_info import SeqlenInfoQK


@dataclass(frozen=True)
class BlockInfo:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    # Wider ctor for call-site compatibility (SM90 fwd/bwd pass extra args);
    # all of these are ignored at runtime since the H100 build forces
    # non-causal / non-local / non-split-KV / non-pack-gqa.
    is_causal: bool = False
    is_local: bool = False
    is_split_kv: bool = False
    window_size_left: Optional[int] = None
    window_size_right: Optional[int] = None
    qhead_per_kvhead_packgqa: int = 1

    @cute.jit
    def get_n_block_min_max(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,  # noqa: ARG002 — kept for API symmetry
    ) -> Tuple[Int32, Int32]:
        n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.tile_n)
        return Int32(0), n_block_max

    @cute.jit
    def get_m_block_min_max(
        self,
        seqlen_info: SeqlenInfoQK,
        n_block: Int32,  # noqa: ARG002 — kept for API symmetry
    ) -> Tuple[Int32, Int32]:
        m_block_max = cute.ceil_div(seqlen_info.seqlen_q, self.tile_m)
        return Int32(0), m_block_max

    # Compatibility shims for SM90 fwd's "no causal / no local" path. With
    # is_local=False the "min before local mask" is just the global min.
    @cute.jit
    def get_n_block_min_before_local_mask(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,  # noqa: ARG002
        n_block_min: Int32,
    ) -> Int32:
        return n_block_min
