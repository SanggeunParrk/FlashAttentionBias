# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# A reimplementation of https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_bwd_postprocess_kernel.h
# from Cutlass C++ to Cute-DSL.
import math
from typing import Callable, Optional, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass import Float32, const_expr
from cutlass.utils import LayoutEnum

from quack import copy_utils
from quack import layout_utils
from quack import sm90_utils

from FA4_fp32.core import utils
from FA4_fp32.infra.cute_dsl_utils import assume_tensor_aligned
import cutlass.cute.nvgpu.tcgen05 as tcgen05
from quack.cute_dsl_utils import ParamsBase
from FA4_fp32.core.tile_scheduler import (
    SingleTileScheduler,
    TileSchedulerArguments,
)


class FlashAttentionBackwardPostprocess:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        tile_m: int = 128,
        num_threads: int = 256,
        dQ_swapAB: bool = False,
    ):
        """B200 / SM100 / 1-CTA postprocess: convert fp32 dQ accum -> dQ."""
        self.dtype = dtype
        self.tile_m = tile_m
        # padding head_dim to a multiple of 32 as k_block_size
        hdim_multiple_of = 32
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        self.num_threads = num_threads
        self.dQ_swapAB = dQ_swapAB

    @staticmethod
    def can_implement(dtype, head_dim, tile_m, num_threads) -> bool:
        """Check if the kernel can be implemented with the given parameters.

        :param dtype: data type
        :type dtype: cutlass.Numeric
        :param head_dim: head dimension
        :type head_dim: int
        :param tile_m: m block size
        :type tile_m: int

        :return: True if the kernel can be implemented, False otherwise
        :rtype: bool
        """
        if dtype not in [cutlass.Float16, cutlass.BFloat16, cutlass.Float32]:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        return True

    def _get_tiled_mma(self):
        # SM100 only.
        cta_group = tcgen05.CtaGroup.ONE
        return sm100_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            tcgen05.OperandMajorMode.MN,  # dS_major_mode
            tcgen05.OperandMajorMode.MN,  # Kt_major_mode
            Float32,
            cta_group,
            (self.tile_m, self.tile_hdim),
        )

    def _setup_attributes(self):
        # ///////////////////////////////////////////////////////////////////////////////
        # GMEM Tiled copy:
        # ///////////////////////////////////////////////////////////////////////////////
        # Thread layouts for copies
        universal_copy_bits = 128
        async_copy_elems_accum = universal_copy_bits // Float32.width
        atom_async_copy_accum = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            Float32,
            num_bits_per_copy=universal_copy_bits,
        )
        # We don't do bound checking for the gmem -> smem load so we just assert here.
        assert (self.tile_m * self.tile_hdim // async_copy_elems_accum) % self.num_threads == 0
        self.g2s_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
            atom_async_copy_accum,
            cute.make_layout(self.num_threads),
            cute.make_layout(async_copy_elems_accum),
        )
        # SM100 only.
        num_s2r_copy_elems = 4
        self.dQ_reduce_ncol = 32
        dQaccum_reduce_stage = self.tile_hdim // self.dQ_reduce_ncol
        assert self.num_threads == 128  # TODO: currently hard-coded
        self.s2r_tiled_copy_dQaccum = copy_utils.tiled_copy_1d(
            Float32, self.num_threads, num_s2r_copy_elems
        )
        self.sdQaccum_layout = cute.make_layout(
            (self.tile_m * self.tile_hdim // dQaccum_reduce_stage, dQaccum_reduce_stage)
        )

        num_copy_elems = 128 // self.dtype.width
        threads_per_row = math.gcd(128, self.tile_hdim) // num_copy_elems
        self.gmem_tiled_copy_dQ = copy_utils.tiled_copy_2d(
            self.dtype, threads_per_row, self.num_threads, num_copy_elems
        )
        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory layout: dQ
        # ///////////////////////////////////////////////////////////////////////////////
        # We can't just use kHeadDim here. E.g. if MMA shape is 64 x 96 but split across 2 WGs,
        # then setting kBlockKSmem to 32 will cause "Static shape_div failure".
        # We want to treat it as 64 x 48, so kBlockKSmem should be 16.
        # SM100 only. TODO: this is hard-coded for hdim 128
        self.sdQ_layout = sm100_utils_basic.make_smem_layout_epi(
            self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_hdim), 1
        )

    @cute.jit
    def __call__(
        self,
        mdQaccum: cute.Tensor,
        mdQ: cute.Tensor,
        scale: cutlass.Float32,
        # Stream comes right after scale: stripped interface passes (mdQaccum, mdQ, scale, stream).
        stream: cuda.CUstream = None,
    ):
        # Get the data type and check if it is fp16 or bf16
        if const_expr(mdQ.element_type not in [cutlass.Float16, cutlass.BFloat16, cutlass.Float32]):
            raise TypeError("Only Float16/BFloat16/Float32 is supported")
        if const_expr(mdQaccum is not None):
            if const_expr(mdQaccum.element_type not in [cutlass.Float32]):
                raise TypeError("dQaccum tensor must be Float32")

        mdQaccum, mdQ = [assume_tensor_aligned(t) for t in (mdQaccum, mdQ)]

        self.tiled_mma = self._get_tiled_mma()
        self._setup_attributes()

        smem_size = max(
            cute.size_in_bytes(cutlass.Float32, self.sdQaccum_layout),
            cute.size_in_bytes(self.dtype, self.sdQ_layout),
        )

        TileScheduler = SingleTileScheduler
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(mdQ.shape[1], self.tile_m),
            num_head=mdQ.shape[2],
            num_batch=mdQ.shape[0],
            seqlen_k=0,
            headdim=mdQ.shape[2],
            headdim_v=0,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        # grid_dim: (m_block, num_head, batch_size)
        self.kernel(
            mdQaccum,
            mdQ,
            scale,
            self.tiled_mma,
            self.dQ_swapAB,
            self.sdQaccum_layout,
            self.sdQ_layout,
            self.g2s_tiled_copy_dQaccum,
            self.s2r_tiled_copy_dQaccum,
            self.gmem_tiled_copy_dQ,
            tile_sched_params,
            TileScheduler,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=smem_size,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mdQaccum: cute.Tensor,
        mdQ: cute.Tensor,
        scale: cutlass.Float32,
        tiled_mma: cute.TiledMma,
        dQ_swapAB: cutlass.Constexpr,
        sdQaccum_layout: cute.Layout,
        sdQ_layout: cute.ComposedLayout,
        g2s_tiled_copy_dQaccum: cute.TiledCopy,
        s2r_tiled_copy_dQaccum: cute.TiledCopy,
        gmem_tiled_copy_dQ: cute.TiledCopy,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
    ):
        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        sdQaccum = smem.allocate_tensor(cutlass.Float32, sdQaccum_layout, byte_alignment=1024)
        sdQaccum_flat = cute.make_tensor(sdQaccum.iterator, cute.make_layout(cute.size(sdQaccum)))
        # SM100 only: extra stage dimension in sdQ_layout.
        sdQ = cute.make_tensor(
            cute.recast_ptr(sdQaccum.iterator, sdQ_layout.inner, dtype=self.dtype),
            sdQ_layout.outer,
        )[None, None, 0]
        sdQt = layout_utils.transpose_view(sdQ)

        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()

        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()

        m_block, head_idx, batch_idx, _ = work_tile.tile_idx

        if work_tile.is_valid_tile:
            # ///////////////////////////////////////////////////////////////////////////////
            # Get the appropriate tiles for this thread block.
            # ///////////////////////////////////////////////////////////////////////////////

            seqlen_q = mdQ.shape[1]
            mdQ_cur = mdQ[batch_idx, None, head_idx, None]
            mdQaccum_cur = mdQaccum[batch_idx, head_idx, None]
            head_dim = mdQ.shape[3]

            gdQaccum = cute.local_tile(mdQaccum_cur, (self.tile_m * self.tile_hdim,), (m_block,))
            gdQ = cute.local_tile(mdQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))

            seqlen_q_rounded = cute.round_up(seqlen_q, self.tile_m)

            # Step 1: load dQaccum from gmem to smem
            g2s_thr_copy_dQaccum = g2s_tiled_copy_dQaccum.get_slice(tidx)
            tdQgdQaccum = g2s_thr_copy_dQaccum.partition_S(gdQaccum)
            tdQsdQaccumg2s = g2s_thr_copy_dQaccum.partition_D(sdQaccum_flat)
            cute.copy(g2s_tiled_copy_dQaccum, tdQgdQaccum, tdQsdQaccumg2s)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            # Step 2: load dQ from smem to rmem
            s2r_thr_copy_dQaccum = s2r_tiled_copy_dQaccum.get_slice(tidx)
            tdQsdQaccum = s2r_thr_copy_dQaccum.partition_S(sdQaccum)
            acc = None
            if const_expr(self.dtype is Float32):
                # fp32 path: TF32 MMA acc has (inner,outer) split that flat
                # Ld32x32bOp(Repetition(...)) atoms reject. acc is filled from
                # SMEM via autovec_copy, so allocate it directly at SMEM size.
                acc = cute.make_fragment_like(tdQsdQaccum, Float32)
            else:
                thr_mma = tiled_mma.get_slice(0)
                dQacc_shape = tiled_mma.partition_shape_C((self.tile_m, self.tile_hdim))
                tdQtdQ = tiled_mma.make_fragment_C(dQacc_shape)
                tdQcdQ = thr_mma.partition_C(
                    cute.make_identity_tensor((self.tile_m, self.tile_hdim))
                )
                tmem_load_atom = cute.make_copy_atom(
                    tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dQ_reduce_ncol)),
                    Float32,
                )
                tiled_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ)
                thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
                tdQrdQ_t2r_shape = thr_copy_t2r.partition_D(tdQcdQ).shape
                acc = cute.make_fragment(tdQrdQ_t2r_shape, Float32)
            tdQrdQaccum = cute.make_tensor(acc.iterator, cute.make_layout(tdQsdQaccum.shape))
            cute.autovec_copy(tdQsdQaccum, tdQrdQaccum)
            # Convert tdQrdQaccum from fp32 to fp16/bf16
            rdQ = cute.make_fragment_like(acc, self.dtype)
            rdQ.store((acc.load() * scale).to(self.dtype))

            # Step 3: Copy dQ from register to smem
            cute.arch.barrier()  # make sure all threads have finished loading dQaccum
            # SM100 only: plain universal 128-bit store atom.
            thr_layout_r2s_dQ = cute.make_layout((self.num_threads, 1))
            val_layout_r2s_dQ = cute.make_layout((1, 128 // self.dtype.width))
            copy_atom_r2s_dQ = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dtype,
                num_bits_per_copy=128,
            )
            tiled_copy_r2s_dQ = cute.make_tiled_copy_tv(
                copy_atom_r2s_dQ, thr_layout_r2s_dQ, val_layout_r2s_dQ
            )
            thr_copy_r2s_dQ = tiled_copy_r2s_dQ.get_slice(tidx)
            cdQ = cute.make_identity_tensor((self.tile_m, self.tile_hdim))
            taccdQcdQ_shape = thr_copy_r2s_dQ.partition_S(cdQ).shape
            taccdQrdQ = cute.make_tensor(rdQ.iterator, taccdQcdQ_shape)
            taccdQsdQ = thr_copy_r2s_dQ.partition_D(
                sdQ if const_expr(not self.dQ_swapAB) else sdQt
            )
            cute.copy(thr_copy_r2s_dQ, taccdQrdQ, taccdQsdQ)

            # Step 4: Copy dQ from smem to register to prepare for coalesced write to gmem
            cute.arch.barrier()  # make sure all smem stores are done
            gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_slice(tidx)
            tdQgdQ = gmem_thr_copy_dQ.partition_S(gdQ)
            tdQsdQ = gmem_thr_copy_dQ.partition_D(sdQ)
            tdQrdQ = cute.make_fragment_like(tdQsdQ, self.dtype)
            # TODO: check OOB when reading from smem if kBlockM isn't evenly tiled
            cute.autovec_copy(tdQsdQ, tdQrdQ)

            # Step 5: Copy dQ from register to gmem
            tdQcdQ = gmem_thr_copy_dQ.partition_S(cdQ)
            tdQpdQ = utils.predicate_k(tdQcdQ, limit=head_dim)
            for rest_m in cutlass.range(cute.size(tdQrdQ.shape[1]), unroll_full=True):
                if tdQcdQ[0, rest_m, 0][0] < seqlen_q - m_block * self.tile_m:
                    cute.copy(
                        gmem_tiled_copy_dQ,
                        tdQrdQ[None, rest_m, None],
                        tdQgdQ[None, rest_m, None],
                        pred=tdQpdQ[None, rest_m, None],
                    )
