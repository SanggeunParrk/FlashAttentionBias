# B200 SM100 MHA forward kernel with score bias staged through SMEM.
# Q/K/V/O shape: (A, B, L, H, D)
# Bias shape:    (1, B, Lq, H, Lk), broadcast over A.

import math
from typing import Tuple, Callable, Optional, Literal
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute import FastDivmodDivisor
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import ClcDynamicPersistentTileScheduler

from quack import copy_utils, layout_utils
from quack.cute_dsl_utils import ParamsBase

from FA4_fp32.infra.cute_dsl_utils import assume_tensor_aligned
from FA4_fp32.core import utils
from FA4_fp32.core import pipeline as pipeline_custom
from FA4_fp32.core.mask import AttentionMask
from FA4_fp32.core.softmax import SoftmaxSm100
from FA4_fp32.core.seqlen_info import SeqlenInfoQK
from FA4_fp32.core.block_info import BlockInfo

from FA4_fp32.arch import mma_sm100_desc as sm100_desc
from FA4_fp32.arch import blackwell_helpers as sm100_utils
from FA4_fp32.core.named_barrier import NamedBarrierFwdSm100
from FA4_fp32.core.tile_scheduler import (
    ClcState,
    TileSchedulerArguments,
    TileSchedulerProtocol,
)
from FA4_fp32.kernels.flash_fwd_sm100 import FlashAttentionForwardSm100


class FlashAttentionBiasForwardSm100Smem(FlashAttentionForwardSm100):
    """SM100 forward kernel variant for A-batched attention with SMEM-staged score bias."""

    def __init__(
        self,
        head_dim: int,
        m_block_size: int = 128,
        n_block_size: int = 64,
        q_stage: cutlass.Constexpr[int] = 2,
        is_persistent: bool = True,
        use_clc_scheduler: bool = False,
    ):
        super().__init__(
            head_dim,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            q_stage=q_stage,
            is_persistent=is_persistent,
            use_clc_scheduler=use_clc_scheduler,
        )

    def _setup_attributes(self):
        """Configure KV pipeline stages from shared-memory budget including SMEM bias."""
        elem_bytes = self.qkvo_dtype.width // 8
        bias_elem_bytes = self.bias_dtype.width // 8
        smem_size_q = self.q_stage * self.m_block_size * self.head_dim * elem_bytes
        smem_size_o = self.q_stage * self.m_block_size * self.head_dim * elem_bytes
        smem_size_q_o = smem_size_q + smem_size_o
        smem_size_bias = (
            self.q_stage
            * self.m_block_size
            * self.n_block_size
            * bias_elem_bytes
        )
        smem_size_k_per_stage = self.n_block_size * self.head_dim * elem_bytes
        smem_size_v_per_stage = self.n_block_size * self.head_dim * elem_bytes
        smem_size_kv_per_stage = max(smem_size_k_per_stage, smem_size_v_per_stage)
        kv_stage = (224 * 1024 - smem_size_q_o - smem_size_bias) // smem_size_kv_per_stage
        self.kv_stage = kv_stage
        self.s_stage = 2
        assert self.kv_stage >= 1, "SMEM bias staging leaves no room for K/V stages"
        assert self.s_stage >= self.q_stage
        tile_p_like_fp32 = self.n_block_size // Float32.width * self.qkvo_dtype.width
        self.tmem_s_to_p_offset = self.n_block_size - tile_p_like_fp32
        self.tmem_p_offset = [
            self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(2)
        ]

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (a, b, s_q, h, d)
        mK: cute.Tensor,  # (a, b_k, s_k, h_k, d)
        mV: cute.Tensor,  # (a, b_k, s_k, h_k, d)
        mBias: cute.Tensor,  # (1, b, s_q, h, s_k), broadcast over a
        mO: cute.Tensor,  # (a, b, s_q, h, d)
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        stream: cuda.CUstream = None,
    ):
        """Launch the Blackwell SM100 MHA forward kernel with SMEM-staged score bias."""
        assert (
            mQ.element_type == mK.element_type == mV.element_type == mO.element_type
        ), (
            f"Q/K/V/O must share dtype, got {mQ.element_type}, {mK.element_type}, "
            f"{mV.element_type}, {mO.element_type}"
        )
        self.qkvo_dtype = mQ.element_type
        self.bias_dtype = mBias.element_type
        mQ, mK, mV, mBias, mO = [
            assume_tensor_aligned(t) for t in (mQ, mK, mV, mBias, mO)
        ]

        QKO_layout_transpose = [2, 4, 3, 1, 0]  # (a, b, s, h, d) -> (s, d, h, b, a)
        V_layout_transpose = [4, 2, 3, 1, 0]  # (a, b, s, h, d) -> (d, s, h, b, a)
        Bias_layout_transpose = [2, 4, 3, 1, 0]  # (1, b, s_q, h, s_k) -> (s_q, s_k, h, b, 1)
        LSE_layout_transpose = [3, 2, 1, 0]  # (a, b, h, s_q) -> (s_q, h, b, a)
        mQ, mK, mO = [
            cute.make_tensor(
                t.iterator, cute.select(t.layout, mode=QKO_layout_transpose)
            )
            for t in (mQ, mK, mO)
        ]
        mV = cute.make_tensor(
            mV.iterator, cute.select(mV.layout, mode=V_layout_transpose)
        )
        mBias = cute.make_tensor(
            mBias.iterator, cute.select(mBias.layout, mode=Bias_layout_transpose)
        )
        mLSE = (
            cute.make_tensor(
                mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose)
            )
            if const_expr(mLSE is not None)
            else None
        )

        self._setup_attributes()
        self.ex2_emu_freq = 16
        self.ex2_emu_start_frg = 1

        cta_group = tcgen05.CtaGroup.ONE
        q_major_mode = tcgen05.OperandMajorMode.K
        k_major_mode = tcgen05.OperandMajorMode.K
        v_major_mode = tcgen05.OperandMajorMode.MN
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
            self.qkvo_dtype,
            q_major_mode,
            k_major_mode,
            self.acc_dtype,
            cta_group,
            self.mma_tiler_qk[:2],
        )
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.qkvo_dtype,
            p_major_mode,
            v_major_mode,
            self.acc_dtype,
            cta_group,
            self.mma_tiler_pv[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )

        self.epi_tile = (self.m_block_size, self.head_dim)

        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.qkvo_dtype, self.q_stage
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.qkvo_dtype, self.kv_stage
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler_pv, self.qkvo_dtype, self.s_stage
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv, self.mma_tiler_pv, self.qkvo_dtype, self.kv_stage
        )
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.qkvo_dtype, self.o_layout, self.epi_tile, self.q_stage
        )
        sBias_layout = cute.make_layout(
            (self.m_block_size, self.n_block_size, self.q_stage)
        )

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(
                mX.element_type, cute.select(layout, mode=[0, 1, 2])
            )
            for name, mX, layout in [
                ("Q", mQ, sQ_layout),
                ("K", mK, sK_layout),
                ("V", mV, sV_layout),
            ]
        }
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            cta_layout_vmnk.shape,
        )
        tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mK,
            cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            cta_layout_vmnk.shape,
        )
        tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mV,
            cute.select(sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_pv,
            tiled_mma_pv,
            cta_layout_vmnk.shape,
        )
        self.num_epilogue_threads = cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        tma_atom_O, mO = cpasync.make_tiled_tma_atom(
            tma_store_op, mO, cute.select(sO_layout, mode=[0, 1]), self.epi_tile
        )

        TileScheduler = self.TileScheduler
        _num_block_divisor = self.cta_tiler[0]
        num_batch_b = cute.size(mQ.shape[3])
        num_batch_a = cute.size(mQ.shape[4])
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(cute.size(mQ.shape[0]), _num_block_divisor),
            num_head=cute.size(mQ.shape[2]),
            num_batch=num_batch_a * num_batch_b,
            seqlen_k=cute.size(mK.shape[0]),
            headdim=mQ.shape[1],
            headdim_v=mV.shape[0],
            element_size=self.qkvo_dtype.width // 8,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(
            tile_sched_args, scheduling_mode=self.scheduling_mode
        )
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        ab_divmod = FastDivmodDivisor(num_batch_a)

        sO_size = cute.cosize(sO_layout)
        sQ_size = cute.cosize(sQ_layout)
        sBias_size = cute.cosize(sBias_layout)

        clc_response_size = self.sched_stages * 4 if self.use_clc_scheduler else 0
        clc_mbar_size = self.sched_stages * 2 if self.use_clc_scheduler else 0

        @cute.struct
        class SharedStorage:
            mbar_load_Q: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_load_KV: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mbar_S_full_P_full_O_rescaled: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_P_full_lastsplit: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_O_full: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_softmax_stats: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_O_epi: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_s0_s1_sequence: cute.struct.MemRange[Int64, 2 * 2]
            tmem_dealloc_mbar_ptr: Int64
            tmem_holding_buf: Int32
            sScale: cute.struct.MemRange[Float32, self.q_stage * self.m_block_size * 2]
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, clc_mbar_size]
            clc_response: cute.struct.MemRange[Int32, clc_response_size]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.qkvo_dtype, sO_size], self.buffer_align_bytes
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.qkvo_dtype, sQ_size], self.buffer_align_bytes
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.qkvo_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]
            sBias: cute.struct.Align[
                cute.struct.MemRange[self.bias_dtype, sBias_size],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = Float32(LOG2_E)

        self.kernel(
            mQ,
            mK,
            mV,
            mBias,
            mO,
            mLSE,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sBias_layout,
            sO_layout,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            ab_divmod,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=None,
            stream=stream,
            min_blocks_per_mp=1,
        )

    # Device kernel and changed helper methods follow. Most MMA/correction math is inherited.

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mBias: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sBias_layout: cute.Layout,
        sO_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        ab_divmod: FastDivmodDivisor,
    ):
        """Warp-specialized SM100 MHA forward device kernel with SMEM bias."""
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        tidx = cute.arch.thread_idx()[0]

        if warp_idx == self.load_warp_ids[0]:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                cpasync.prefetch_descriptor(tma_atom)

        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )
        mma_tile_coord_v = 0

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE
            * (
                len((self.mma_warp_id,))
                + len(self.softmax0_warp_ids)
                + len(self.softmax1_warp_ids)
                + len(self.correction_warp_ids)
            ),
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        ThreadCooperativeGroup = partial(
            pipeline.CooperativeGroup, pipeline.Agent.Thread
        )
        mma_warp = ThreadCooperativeGroup(len([self.mma_warp_id]))
        tma_warp = ThreadCooperativeGroup(1)
        softmax_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids)
        )
        all_softmax_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids + self.softmax1_warp_ids)
        )
        correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        epilogue_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        )
        softmax_warps_cluster = ThreadCooperativeGroup(len(self.softmax0_warp_ids))
        correction_threads_cluster = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        softmax_correction_threads_cluster = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids + self.correction_warp_ids)
        )

        pipeline_q = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_Q.data_ptr(),
            num_stages=self.q_stage,
            producer_group=tma_warp,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_kv = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_KV.data_ptr(),
            num_stages=self.kv_stage,
            producer_group=tma_warp,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_bytes["K"],
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_s_p_o = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_S_full_P_full_O_rescaled.data_ptr(),
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=softmax_correction_threads_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_p_lastsplit = pipeline_custom.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_P_full_lastsplit.data_ptr(),
            num_stages=self.q_stage,
            producer_group=softmax_warps_cluster,
            consumer_group=mma_warp,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_o_acc = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_O_full.data_ptr(),
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=correction_threads_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        sm_stats_barrier = pipeline_custom.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.SoftmaxStatsW0),
            num_threads=cute.arch.WARP_SIZE * 2,
        )
        bias_smem_barrier = pipeline_custom.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.SoftmaxStatsW7) + 1,
            num_threads=cute.arch.WARP_SIZE * len(self.softmax0_warp_ids),
        )
        pipeline_sm_stats = pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_softmax_stats.data_ptr(),
            num_stages=self.q_stage,
            producer_group=softmax_threads,
            consumer_group=correction_threads,
            defer_sync=True,
        )
        pipeline_o_epi = pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_O_epi.data_ptr(),
            num_stages=self.q_stage,
            producer_group=correction_threads,
            consumer_group=epilogue_threads,
            defer_sync=True,
        )

        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = cute.make_tensor(
            cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer
        )
        sBias = storage.sBias.get_tensor(sBias_layout)
        sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        sScale = storage.sScale.get_tensor(
            cute.make_layout(self.q_stage * self.m_block_size * 2)
        )

        thr_mma_qk = tiled_mma_qk.get_slice(mma_tile_coord_v)
        thr_mma_pv = tiled_mma_pv.get_slice(mma_tile_coord_v)

        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        tStS = thr_mma_qk.make_fragment_C(cute.append(qk_acc_shape, self.s_stage))
        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO = thr_mma_pv.make_fragment_C(cute.append(pv_acc_shape, self.q_stage))
        tOtO = cute.make_tensor(tOtO.iterator + self.tmem_o_offset[0], tOtO.layout)
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]
        tP_width_ratio = Float32.width // self.qkvo_dtype.width
        tP_stage_stride = (
            self.tmem_p_offset[1] - self.tmem_p_offset[0]
        ) * tP_width_ratio
        tOrP = cute.make_tensor(
            tOrP.iterator + self.tmem_p_offset[0] * tP_width_ratio,
            cute.append(
                tOrP.layout,
                cute.make_layout((self.s_stage,), stride=(tP_stage_stride,)),
            ),
        )

        block_info = BlockInfo(self.cta_tiler[0], self.cta_tiler[1])
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
        )
        AttentionMaskCls = partial(AttentionMask, self.m_block_size, self.n_block_size)
        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        if const_expr(self.use_clc_scheduler):
            clc_response_ptr = storage.clc_response.data_ptr()
            clc_mbar_ptr = storage.clc_mbar_ptr.data_ptr()
            tile_scheduler = self.tile_scheduler_cls.create(
                tile_sched_params,
                clc=ClcState(
                    scheduler_warp_id=self.clc_scheduler_warp_id,
                    response_ptr=clc_response_ptr,
                    mbar_ptr=clc_mbar_ptr,
                    stages=self.sched_stages,
                    hw_scheduler=ClcDynamicPersistentTileScheduler.create(
                        self.cluster_shape_mn,
                        cute.arch.grid_dim(),
                        cute.arch.block_idx(),
                    ),
                ),
            )
        else:
            tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params)
        assert isinstance(tile_scheduler, TileSchedulerProtocol), (
            f"tile_scheduler is not a TileSchedulerProtocol: {type(tile_scheduler)}"
        )

        if const_expr(self.use_clc_scheduler):
            if warp_idx == self.clc_scheduler_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                self.clc_scheduler_warp(tile_scheduler)
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if (
                    warp_idx == self.empty_warp_ids[i]
                    and warp_idx != self.clc_scheduler_warp_id
                ):
                    cute.arch.setmaxregister_decrease(self.num_regs_other)
                    self.empty_warp(tile_scheduler)
        else:
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if warp_idx == self.empty_warp_ids[i]:
                    cute.arch.setmaxregister_decrease(self.num_regs_other)

        if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.load(
                thr_mma_qk,
                thr_mma_pv,
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_q,
                pipeline_kv,
                block_info,
                SeqlenInfoCls,
                ab_divmod,
                tile_scheduler=tile_scheduler,
            )

        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                sQ,
                sK,
                sV,
                tStS,
                tOtO,
                tOrP,
                pipeline_q,
                pipeline_kv,
                pipeline_s_p_o,
                pipeline_p_lastsplit,
                pipeline_o_acc,
                block_info,
                SeqlenInfoCls,
                tile_scheduler=tile_scheduler,
            )
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        if (
            warp_idx >= self.epilogue_warp_ids[0]
            and warp_idx <= self.epilogue_warp_ids[-1]
        ):
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.epilogue_s2g(
                mO,
                sO,
                tma_atom_O,
                pipeline_o_epi,
                SeqlenInfoCls,
                ab_divmod,
                tile_scheduler=tile_scheduler,
            )

        if (
            const_expr(self.q_stage == 2) and warp_idx <= self.softmax1_warp_ids[-1]
        ) or (const_expr(self.q_stage == 1) and warp_idx <= self.softmax0_warp_ids[-1]):
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            softmax_loop = partial(
                self.softmax_loop,
                softmax_scale_log2=softmax_scale_log2,
                softmax_scale=softmax_scale,
                thr_mma_qk=thr_mma_qk,
                mBias=mBias,
                sBias=sBias,
                sScale=sScale,
                mLSE=mLSE,
                bias_smem_barrier=bias_smem_barrier,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                block_info=block_info,
                SeqlenInfoCls=SeqlenInfoCls,
                AttentionMaskCls=AttentionMaskCls,
                ab_divmod=ab_divmod,
                tile_scheduler=tile_scheduler,
            )
            stage = Int32(
                0
                if const_expr(self.q_stage == 1) or warp_idx < self.softmax1_warp_ids[0]
                else 1
            )
            softmax_loop(stage=stage, tStS=tStS)
            tmem_alloc_barrier.arrive()

        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_correction)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.correction_loop(
                thr_mma_pv,
                tOtO,
                sScale,
                mLSE,
                sO,
                pipeline_s_p_o,
                pipeline_o_acc,
                pipeline_sm_stats,
                sm_stats_barrier,
                pipeline_o_epi,
                softmax_scale_log2,
                block_info,
                SeqlenInfoCls,
                ab_divmod,
                tile_scheduler=tile_scheduler,
            )
            tmem_alloc_barrier.arrive()

        return

    @cute.jit
    def load(
        self,
        thr_mma_qk: cute.core.ThrMma,
        thr_mma_pv: cute.core.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        ab_divmod: FastDivmodDivisor,
        tile_scheduler: TileSchedulerProtocol,
    ):
        q_producer_phase = Int32(1)
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, ab_idx, _ = work_tile.tile_idx
            batch_idx, a_idx = divmod(ab_idx, ab_divmod)
            seqlen = SeqlenInfoCls(batch_idx)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[
                None, None, head_idx, a_idx
            ]
            mK_cur, mV_cur = [
                t[None, None, head_idx, batch_idx, a_idx] for t in (mK, mV)
            ]
            gK = cute.local_tile(
                mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0)
            )
            gV = cute.local_tile(
                mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None)
            )
            tSgK = thr_mma_qk.partition_B(gK)
            tOgV = thr_mma_pv.partition_B(gV)
            tiler_gQ = ((self.mma_tiler_qk[0] * self.q_stage), self.head_dim)
            gQ = cute.local_tile(mQ_cur, tiler_gQ, (m_block, 0))
            gQ = layout_utils.select(
                cute.flat_divide(gQ, (self.mma_tiler_qk[0],)), mode=[0, 2, 1]
            )
            tSgQ = thr_mma_qk.partition_A(gQ)
            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ
            )
            load_Q = partial(
                self.load_Q, load_Q_fn, pipeline_q=pipeline_q, phase=q_producer_phase
            )

            tKsK, tKgK = cpasync.tma_partition(
                tma_atom_K,
                0,
                cute.make_layout(1),
                cute.group_modes(sK, 0, 3),
                cute.group_modes(tSgK, 0, 3),
            )
            tVsV, tVgV = cpasync.tma_partition(
                tma_atom_V,
                0,
                cute.make_layout(1),
                cute.group_modes(sV, 0, 3),
                cute.group_modes(tOgV, 0, 3),
            )
            load_K = partial(
                self.load_KV,
                tma_atom_K,
                tKgK,
                tKsK,
                pipeline_kv=pipeline_kv,
                K_or_V="K",
            )
            load_V = partial(
                self.load_KV,
                tma_atom_V,
                tVgV,
                tVsV,
                pipeline_kv=pipeline_kv,
                K_or_V="V",
            )

            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
            load_K(block=n_block_max - 1, producer_state=kv_producer_state)
            load_Q(block=0, stage=0)
            kv_producer_state.advance()
            if const_expr(self.q_stage == 2):
                load_Q(block=1, stage=1)
            q_producer_phase ^= 1
            load_V(block=n_block_max - 1, producer_state=kv_producer_state)
            kv_producer_state.advance()
            for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                n_block = n_block_max - 2 - i
                load_K(block=n_block, producer_state=kv_producer_state)
                kv_producer_state.advance()
                load_V(block=n_block, producer_state=kv_producer_state)
                kv_producer_state.advance()

            work_tile = tile_scheduler.advance_to_next_work()

        pipeline_kv.producer_tail(kv_producer_state)
        pipeline_q.producer_acquire_w_index_phase(self.q_stage - 1, q_producer_phase)

    @cute.jit
    def softmax_loop(
        self,
        tStS: cute.Tensor,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        thr_mma_qk: cute.core.ThrMma,
        mBias: cute.Tensor,
        sBias: cute.Tensor,
        sScale: cute.Tensor,
        mLSE: cute.Tensor,
        bias_smem_barrier: pipeline.NamedBarrier,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        ab_divmod: FastDivmodDivisor,
        tile_scheduler: TileSchedulerProtocol,
        stage: int | Int32,
    ):
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids)
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        tSAcc = tStS[(None, None), 0, 0, stage]
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]

        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.qkvo_dtype.width
        tStP_layout = cute.composition(
            tSAcc.layout, cute.make_layout((self.m_block_size, tilePlikeFP32))
        )
        tStP = cute.make_tensor(tSAcc.iterator + self.tmem_s_to_p_offset, tStP_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tSAcc).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tSAcc)

        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32
        )
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)

        mma_si_consumer_phase = Int32(0)
        sm_stats_producer_phase = Int32(1)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, ab_idx, _ = work_tile.tile_idx
            batch_idx, _ = divmod(ab_idx, ab_divmod)
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
            mBias_cur = mBias[None, None, head_idx, batch_idx, 0]

            mask = AttentionMaskCls(seqlen)
            mask_fn = partial(
                mask.apply_mask_sm100,
                m_block=self.q_stage * m_block + stage,
                thr_mma=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
            )

            softmax = SoftmaxSm100.create(
                softmax_scale_log2,
                rescale_threshold=8.0
                if const_expr(self.qkvo_dtype.width == 16)
                else 0.0,
                softmax_scale=softmax_scale,
            )
            softmax.reset()

            softmax_step = partial(
                self.softmax_step,
                softmax=softmax,
                thr_mma_qk=thr_mma_qk,
                mBias_cur=mBias_cur,
                bias_smem_barrier=bias_smem_barrier,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                tStS_t2r=tStS_t2r,
                tStP_r2t=tStP_r2t,
                sBias=sBias,
                sScale=sScale,
                seqlen=seqlen,
                m_block=m_block,
                stage=stage,
            )

            pipeline_sm_stats.producer_acquire_w_index_phase(
                stage, sm_stats_producer_phase
            )
            sm_stats_producer_phase ^= 1

            mma_si_consumer_phase, sm_stats_producer_phase = softmax_step(
                mma_si_consumer_phase,
                sm_stats_producer_phase,
                n_block_max - 1,
                is_first=True,
                mask_fn=partial(mask_fn, mask_seqlen=True),
            )
            n_block_max -= 1
            for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                n_block = n_block_max - n_tile - 1
                mma_si_consumer_phase, sm_stats_producer_phase = softmax_step(
                    mma_si_consumer_phase,
                    sm_stats_producer_phase,
                    n_block,
                )

            sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
            if const_expr(mLSE is not None):
                sScale[
                    tidx + stage * self.m_block_size + self.q_stage * self.m_block_size
                ] = softmax.row_max[0]
            sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)

            work_tile = tile_scheduler.advance_to_next_work()

        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        sm_stats_producer_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        thr_mma_qk: cute.core.ThrMma,
        mBias_cur: cute.Tensor,
        bias_smem_barrier: pipeline.NamedBarrier,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sBias: cute.Tensor,
        sScale: cute.Tensor,
        seqlen,
        m_block: Int32,
        stage: int | Int32,
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
    ) -> Tuple[cute.Int32, cute.Int32]:
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids)
        )
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.qkvo_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        tScP_shape = (self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape, tilePlikeFP32)

        self.load_bias_smem(
            mBias_cur, sBias, n_block, m_block, stage, tidx, seqlen, bias_smem_barrier
        )
        pipeline_s_p_o.consumer_wait_w_index_phase(stage, mma_si_consumer_phase)
        tSrS_t2r = cute.make_fragment(
            thr_tmem_load.partition_D(tScS).shape, self.acc_dtype
        )
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
        self.apply_bias_smem(tSrS_t2r, tScS_t2r, sBias, stage, softmax.softmax_scale)

        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block)
        row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)

        if const_expr(not is_first):
            thread_idx = thr_tmem_load.thr_idx
            sScale[thread_idx + stage * self.m_block_size] = acc_scale
        sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)

        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        tSrP_r2t_f32 = cute.make_fragment(
            thr_tmem_store.partition_S(cute.make_identity_tensor(tScP_shape)).shape,
            Float32,
        )
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.qkvo_dtype),
            tSrS_t2r.layout,
        )
        softmax.apply_exp2_convert(
            tSrS_t2r,
            tSrP_r2t,
            ex2_emu_freq=self.ex2_emu_freq if const_expr(mask_fn is None) else 0,
            ex2_emu_start_frg=self.ex2_emu_start_frg,
        )
        split_P_arrive_idx = (
            cute.size(tStP_r2t.shape[2]) * self.split_P_arrive // self.n_block_size
        )
        for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2])):
            cute.copy(
                thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i]
            )
            if const_expr(i + 1 == split_P_arrive_idx):
                cute.arch.fence_view_async_tmem_store()
                pipeline_s_p_o.consumer_release_w_index(stage)
        cute.arch.fence_view_async_tmem_store()
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_p_lastsplit.producer_commit_w_index(stage)
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
        softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        return mma_si_consumer_phase ^ 1, sm_stats_producer_phase ^ 1

    @cute.jit
    def load_bias_smem(
        self,
        mBias_cur: cute.Tensor,
        sBias: cute.Tensor,
        n_block: Int32,
        m_block: Int32,
        stage: int | Int32,
        tidx: Int32,
        seqlen,
        bias_smem_barrier: pipeline.NamedBarrier,
    ):
        bias_smem_barrier.arrive_and_wait_w_index(stage)
        q_tile = m_block * self.q_stage + stage
        total = self.m_block_size * self.n_block_size
        for offset in cutlass.range(
            tidx,
            total,
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids),
            unroll=1,
        ):
            q = offset // self.n_block_size
            k = offset - q * self.n_block_size
            q_abs = q_tile * self.m_block_size + q
            k_abs = n_block * self.n_block_size + k
            value = Float32(0.0)
            if q_abs < seqlen.seqlen_q and k_abs < seqlen.seqlen_k:
                value = Float32(mBias_cur[q_abs, k_abs])
            sBias[q, k, stage] = value
        cute.arch.fence_view_async_shared()
        bias_smem_barrier.arrive_and_wait_w_index(stage)

    @cute.jit
    def apply_bias_smem(
        self,
        tSrS_t2r: cute.Tensor,
        tScS_t2r: cute.Tensor,
        sBias: cute.Tensor,
        stage: int | Int32,
        softmax_scale: Float32,
    ):
        for i in cutlass.range(0, cute.size(tSrS_t2r.shape), 2, unroll_full=True):
            q0 = tScS_t2r[i][0]
            k0 = tScS_t2r[i][1]
            q1 = tScS_t2r[i + 1][0]
            k1 = tScS_t2r[i + 1][1]
            b0 = Float32(sBias[q0, k0, stage])
            b1 = Float32(sBias[q1, k1, stage])
            tSrS_t2r[i], tSrS_t2r[i + 1] = cute.arch.fma_packed_f32x2(
                (tSrS_t2r[i], tSrS_t2r[i + 1]),
                (softmax_scale, softmax_scale),
                (b0, b1),
            )

    @cute.jit
    def correction_loop(
        self,
        thr_mma_pv: cute.core.ThrMma,
        tOtO: cute.Tensor,
        sScale: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_o_epi: pipeline.PipelineAsync,
        softmax_scale_log2: Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        ab_divmod: FastDivmodDivisor,
        tile_scheduler=None,
    ):
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        for stage in cutlass.range(self.q_stage):
            pipeline_s_p_o.consumer_release_w_index(stage)

        sm_stats_consumer_phase = Int32(0)
        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, ab_idx, _ = work_tile.tile_idx
            batch_idx, a_idx = divmod(ab_idx, ab_divmod)
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)

            stats = [
                (0.0, -Float32.inf if const_expr(mLSE is not None) else None, True)
            ] * self.q_stage
            total_block_count = n_block_max - n_block_min

            sm_stats_barrier.arrive_and_wait_w_index(index=0 * 4 + warp_idx)
            pipeline_sm_stats.consumer_release_w_index(0)
            if const_expr(self.q_stage == 2):
                sm_stats_barrier.arrive_and_wait_w_index(index=1 * 4 + warp_idx)
            sm_stats_consumer_phase ^= 1

            for i in cutlass.range(total_block_count - 1, unroll=1):
                for stage in cutlass.range_constexpr(self.q_stage):
                    sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
                    scale = sScale[tidx + stage * self.m_block_size]
                    should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
                    if should_rescale:
                        self.correction_rescale(
                            thr_mma_pv, tOtO[None, None, None, stage], tidx, scale
                        )
                    pipeline_s_p_o.consumer_release_w_index(stage)
                    pipeline_sm_stats.consumer_release_w_index(self.q_stage - 1 - stage)
                sm_stats_consumer_phase ^= 1
            if const_expr(self.q_stage == 2):
                pipeline_sm_stats.consumer_release_w_index(1)

            for stage in cutlass.range_constexpr(self.q_stage):
                sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
                row_sum = sScale[tidx + stage * self.m_block_size]
                if const_expr(mLSE is not None):
                    row_max = sScale[
                        tidx
                        + stage * self.m_block_size
                        + self.q_stage * self.m_block_size
                    ]
                else:
                    row_max = None
                pipeline_sm_stats.consumer_release_w_index(stage)
                acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                stats[stage] = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                scale = cute.arch.rcp_approx(
                    row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0
                )
                pipeline_o_acc.consumer_wait_w_index_phase(stage, o_corr_consumer_phase)
                pipeline_o_epi.producer_acquire_w_index_phase(
                    stage, corr_epi_producer_phase
                )
                self.correction_epilogue(
                    thr_mma_pv,
                    tOtO[None, None, None, stage],
                    tidx,
                    scale,
                    sO[None, None, stage],
                )
                pipeline_s_p_o.consumer_release_w_index(stage)
                pipeline_o_epi.producer_commit_w_index(stage)

            o_corr_consumer_phase ^= 1
            sm_stats_consumer_phase ^= 1
            corr_epi_producer_phase ^= 1

            if const_expr(mLSE is not None):
                mLSE_cur = mLSE[None, head_idx, batch_idx, a_idx]
                for stage in cutlass.range_constexpr(self.q_stage):
                    m_tile_idx = m_block * self.q_stage + stage
                    row_sum, row_max, acc_O_mn_row_is_zero_or_nan = stats[stage]
                    LN2 = math.log(2.0)
                    lse = (
                        (
                            row_max * softmax_scale_log2
                            + cute.math.log2(row_sum, fastmath=True)
                        )
                        * LN2
                        if not acc_O_mn_row_is_zero_or_nan
                        else -Float32.inf
                    )
                    seqlen_q = seqlen.seqlen_q
                    gLSE = cute.local_tile(
                        mLSE_cur, (self.m_block_size,), (m_tile_idx,)
                    )
                    if tidx < seqlen_q - m_tile_idx * self.m_block_size:
                        gLSE[tidx] = lse

            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        tma_atom_O: cute.CopyAtom,
        pipeline_o_epi: pipeline.PipelineAsync,
        SeqlenInfoCls: Callable,
        ab_divmod: FastDivmodDivisor,
        tile_scheduler: TileSchedulerProtocol,
    ):
        epi_consumer_phase = Int32(0)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, ab_idx, _ = work_tile.tile_idx
            batch_idx, a_idx = divmod(ab_idx, ab_divmod)
            seqlen = SeqlenInfoCls(batch_idx)

            mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[
                None, None, head_idx, a_idx
            ]
            tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.head_dim)
            gO = cute.local_tile(mO_cur, tiler_gO, (m_block, 0))
            gO = layout_utils.select(
                cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1]
            )
            gO = cute.flat_divide(gO, (self.mma_tiler_pv[0],))[None, 0, None, None]

            store_O, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_O, 0, cute.make_layout(1), sO, gO
            )
            for stage in cutlass.range(self.q_stage, unroll_full=True):
                pipeline_o_epi.consumer_wait_w_index_phase(stage, epi_consumer_phase)
                store_O(src_idx=stage, dst_idx=stage)
                cute.arch.cp_async_bulk_commit_group()
            for stage in cutlass.range_constexpr(self.q_stage):
                cute.arch.cp_async_bulk_wait_group(self.q_stage - 1 - stage, read=True)
                pipeline_o_epi.consumer_release_w_index(stage)
            epi_consumer_phase ^= 1
            work_tile = tile_scheduler.advance_to_next_work()


# Short alias matching the existing forward-kernel naming style.
FlashAttentionBiasForwardSm100 = FlashAttentionBiasForwardSm100Smem
