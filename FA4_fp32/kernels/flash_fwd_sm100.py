# B200 SM100 MHA happy-path forward kernel.
# Supported:
# - dtype fp16 / bf16 / fp32 (TF32 MMA)
# - Q/K/V share (batch, seqlen, num_heads, head_dim) — MHA only, non-causal
# - head_dim in [8..128], multiple of (16 / element_size)
# - optional CLC scheduler (FA_CLC=1 env)
# Based on the cutlass example and cute-dsl example:
# https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha

import math
from typing import Tuple, Callable, Optional, Literal
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import ClcDynamicPersistentTileScheduler
from cutlass.base_dsl.arch import Arch
from cutlass.cutlass_dsl import BaseDSL

from quack import copy_utils, layout_utils

from FA4_fp32.infra.cute_dsl_utils import assume_tensor_aligned
from FA4_fp32.core import utils
from FA4_fp32.core import pipeline as pipeline_custom
import cutlass.pipeline as cutlass_pipeline
from FA4_fp32.core.mask import AttentionMask
from FA4_fp32.core.softmax import SoftmaxSm100
from FA4_fp32.core.seqlen_info import SeqlenInfoQK
from FA4_fp32.core.block_info import BlockInfo

from FA4_fp32.arch import mma_sm100_desc as sm100_desc
from FA4_fp32.arch import blackwell_helpers as sm100_utils
from FA4_fp32.core.named_barrier import NamedBarrierFwdSm100
from quack.cute_dsl_utils import ParamsBase
from FA4_fp32.core.tile_scheduler import (
    ClcState,
    SchedulingMode,
    TileSchedulerArguments,
    TileSchedulerProtocol,
    SingleTileScheduler,
    StaticPersistentTileScheduler,
    SingleTileLPTScheduler,
)
from FA4_fp32.infra.fa_logging import fa_log, fa_printf
from FA4_fp32.core.utils import smid


class FlashAttentionForwardSm100:

    def __init__(
        self,
        head_dim: int,
        m_block_size: int = 128,
        n_block_size: int = 128,
        q_stage: cutlass.Constexpr[int] = 2,
        is_persistent: bool = True,
        use_clc_scheduler: bool = False,
    ):
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        # MHA-only build: head_dim_v == head_dim, always.
        self.head_dim_v_padded = self.head_dim_padded
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.q_stage = q_stage
        assert self.q_stage in [1, 2]
        # If split_P_arrive, the softmax warps write some columns of P first, signal to the MMA warp
        # to being the P @ V MMA, then write the rest of P and signal again. This allows some overlap
        # between compute the last couple columns of P and the P @ V MMA.
        self.split_P_arrive = n_block_size // 4 * 3
        self.split_P_arrive = int(self.split_P_arrive / 32) * 32  # multiple of 32
        assert self.split_P_arrive % 32 == 0
        assert self.split_P_arrive < self.n_block_size
        self.arch = BaseDSL._get_dsl().get_arch_enum()
        assert self.arch >= Arch.sm_100 and self.arch <= Arch.sm_110f, "Only SM 10.x and 11.x are supported"

        # Stripped build: 1-CTA only, cluster shape is (1, 1).
        self.cta_tiler = (self.q_stage * m_block_size, n_block_size, self.head_dim_padded)
        self.mma_tiler_qk = (m_block_size, n_block_size, self.head_dim_padded)
        self.mma_tiler_pv = (m_block_size, self.head_dim_v_padded, n_block_size)
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.cluster_shape_mn = (1, 1)
        self.is_persistent = is_persistent
        is_sm103 = self.arch >= Arch.sm_103 and self.arch <= Arch.sm_103f
        # head_dim is always in [8..128] in this stripped build, so ex2 emulation is enabled on non-sm103.
        self.enable_ex2_emu = not is_sm103

        self.use_clc_scheduler = use_clc_scheduler
        self.sched_stages = 1

        self.scheduling_mode = SchedulingMode.CLC if self.use_clc_scheduler else SchedulingMode.STATIC

        if self.use_clc_scheduler:
            self.TileScheduler = SingleTileLPTScheduler
        elif self.is_persistent:
            self.TileScheduler = StaticPersistentTileScheduler
        else:
            self.TileScheduler = SingleTileScheduler

        fa_log(1, f"TileScheduler={self.TileScheduler.__name__}, scheduling_mode={self.scheduling_mode.name}")

        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.epilogue_warp_ids = (13,)
        self.load_warp_ids = (14,)
        self.empty_warp_ids = (15,)
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                *self.load_warp_ids,
                *self.epilogue_warp_ids,
                *self.empty_warp_ids,
            )
        )

        if self.q_stage == 1:
            self.empty_warp_ids = self.empty_warp_ids + self.softmax1_warp_ids
            self.softmax1_warp_ids = ()

        self.clc_scheduler_warp_id = self.empty_warp_ids[0] if self.use_clc_scheduler else None

        self.tmem_s_offset = [0, self.n_block_size]  # e.g., 0, 128
        self.tmem_o_offset = [
            self.tmem_s_offset[-1] + self.n_block_size + i * self.head_dim_v_padded
            for i in range(self.q_stage)
        ]  # e.g., 256, 384
        self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v_padded
        assert self.tmem_total <= self.tmem_alloc_cols
        self.tmem_s_to_p_offset = self.n_block_size // 2
        self.tmem_p_offset = [
            self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(2)
        ]  # 0, 128

        # Stripped build: no tuning overrides — use the defaults.
        if self.head_dim_padded < 96:
            self.num_regs_softmax = 200
            self.num_regs_correction = 64
            self.num_regs_other = 48
        else:
            self.num_regs_softmax = 192
            self.num_regs_correction = 80
            self.num_regs_other = 512 - self.num_regs_softmax * 2 - self.num_regs_correction

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        """Configure KV pipeline stages from shared-memory budget."""
        smem_size_q = self.q_stage * self.m_block_size * self.head_dim_padded * self.q_dtype.width // 8
        smem_size_o = self.q_stage * self.m_block_size * self.head_dim_v_padded * self.o_dtype.width // 8
        smem_size_q_o = smem_size_q + smem_size_o
        smem_size_k_per_stage = self.n_block_size * self.head_dim_padded * self.k_dtype.width // 8
        smem_size_v_per_stage = self.n_block_size * self.head_dim_v_padded * self.v_dtype.width // 8
        smem_size_kv_per_stage = max(smem_size_k_per_stage, smem_size_v_per_stage)
        kv_stage = (224 * 1024 - smem_size_q_o) // smem_size_kv_per_stage
        self.kv_stage = kv_stage
        self.s_stage = 2
        assert self.s_stage >= self.q_stage

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d)
        mK: cute.Tensor,  # (b_k, s_k, h_k, d)
        mV: cute.Tensor,  # (b_k, s_k, h_k, dv)
        mO: cute.Tensor,  # (b, s_q, h, dv)
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        stream: cuda.CUstream = None,
    ):
        """Launch the Blackwell SM100 MHA forward kernel."""
        # setup static attributes before smem/grid/tma computation
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        Q_layout_transpose = [1, 3, 2, 0]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose))
        KV_layout_transpose = [1, 3, 2, 0]
        mK, mV = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=KV_layout_transpose))
            for t in (mK, mV)
        ]
        O_layout_transpose = [1, 3, 2, 0]
        LSE_layout_transpose = [2, 1, 0]
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=O_layout_transpose))
        mLSE = (
            cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
            if const_expr(mLSE is not None)
            else None
        )
        # (s, d, h, b) -> (d, s, h, b)
        V_layout_transpose = [1, 0, 2, 3]
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=V_layout_transpose))

        # check type consistency
        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        self._setup_attributes()
        self.ex2_emu_freq = 0
        self.ex2_emu_start_frg = 1
        if const_expr(self.enable_ex2_emu):
            self.ex2_emu_freq = 16

        cta_group = tcgen05.CtaGroup.ONE
        q_major_mode = tcgen05.OperandMajorMode.K
        k_major_mode = tcgen05.OperandMajorMode.K
        v_major_mode = tcgen05.OperandMajorMode.MN
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)
        # the intermediate tensor p is from tmem & mK-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            q_major_mode,
            k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.mma_tiler_qk[:2],
        )
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.mma_tiler_pv[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )

        # epi_tile is per-CTA (not full 2CTA) since each CTA writes its own O portion
        self.epi_tile = (self.m_block_size, self.head_dim_v_padded)

        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.q_stage
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler_pv, self.q_dtype, self.s_stage
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv, self.mma_tiler_pv, self.v_dtype, self.kv_stage
        )
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype, self.o_layout, self.epi_tile, self.q_stage
        )

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", mQ, sQ_layout),
                ("K", mK, sK_layout),
                ("V", mV, sV_layout),
            ]
        }

        # TMA load for Q
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

        # TMA load for K
        tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mK,
            cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            cta_layout_vmnk.shape,
        )
        # TMA load for V
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
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(cute.size(mQ.shape[0]), _num_block_divisor),
            num_head=cute.size(mQ.shape[2]),
            num_batch=cute.size(mQ.shape[3]),
            seqlen_k=cute.size(mK.shape[0]),
            headdim=mQ.shape[1],
            headdim_v=mV.shape[0],  # mV is already transposed on SM100
            element_size=self.k_dtype.width // 8,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(
            tile_sched_args, scheduling_mode=self.scheduling_mode
        )
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        sO_size = cute.cosize(sO_layout)
        sQ_size = cute.cosize(sQ_layout)

        clc_response_size = self.sched_stages * 4 if self.use_clc_scheduler else 0
        clc_mbar_size = self.sched_stages * 2 if self.use_clc_scheduler else 0

        @cute.struct
        class SharedStorage:
            # m_barriers for pipelines
            mbar_load_Q: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_load_KV: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mbar_S_full_P_full_O_rescaled: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_P_full_lastsplit: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_O_full: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_softmax_stats: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_O_epi: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_s0_s1_sequence: cute.struct.MemRange[Int64, 2 * 2]
            # Tmem dealloc cluster barrier
            tmem_dealloc_mbar_ptr: Int64
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            # store row max and row sum
            sScale: cute.struct.MemRange[Float32, self.q_stage * self.m_block_size * 2]
            # CLC buffers placed here to utilize padding before sO's 1024-byte alignment.
            # This avoids adding bytes at the end when we're at the smem limit.
            # PipelineClcFetchAsync expects 2 * sched_stages mbarriers (full + empty).
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, clc_mbar_size]
            # CLC response storage (16 bytes per stage, stored as 4 Int32s).
            clc_response: cute.struct.MemRange[Int32, clc_response_size]
            # Large TMA buffers with 1024-byte alignment
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size], self.buffer_align_bytes
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size], self.buffer_align_bytes
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(softmax_scale)

        # Launch the kernel synchronously
        self.kernel(
            mQ,
            mK,
            mV,
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
            sO_layout,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=None,
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # (s_q, d, h, b)
        mK: cute.Tensor,  # (s_k, d, h_k, b_k)
        mV: cute.Tensor,  # (d, s_k, h_k, b_k)
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
    ):
        """Warp-specialized SM100 MHA forward device kernel (load / MMA / softmax / correction / epilogue)."""

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                cpasync.prefetch_descriptor(tma_atom)

        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )
        # Setup cta/thread coordinates
        mma_tile_coord_v = 0

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE * len(
                (self.mma_warp_id,
                 *self.softmax0_warp_ids,
                 *self.softmax1_warp_ids,
                 *self.correction_warp_ids)
            ),
        )
        # Tensor memory dealloc barrier init
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        mma_warp = ThreadCooperativeGroup(len([self.mma_warp_id]))
        tma_warp = ThreadCooperativeGroup(1)
        softmax_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE * len(self.softmax0_warp_ids))
        correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        epilogue_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE * len(self.epilogue_warp_ids))
        softmax_warps_cluster = ThreadCooperativeGroup(
            len(self.softmax0_warp_ids)
        )
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
        # This pipeline is not the typical producer-consumer pipeline. The "producer" mma warp
        # uses it to signal that S is ready, and the softmax threads wait for S to be ready.
        # When softmax threads write P to tmem and the correction threads have rescaled O, they
        # signal as "consumer". The mma warp then waits for that signal to do the P @ V gemm.
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
        # MMA warp uses this to signal to the correction warps that O is ready.
        pipeline_o_acc = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_O_full.data_ptr(),
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=correction_threads_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_sm_stats = pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_softmax_stats.data_ptr(),
            num_stages=self.q_stage,
            producer_group=softmax_threads,
            consumer_group=correction_threads,
            defer_sync=True,
        )
        # Should put the NamedBarrier inside the pipeline class so we'll just have pipeline_sm_stats
        sm_stats_barrier = pipeline_custom.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.SoftmaxStatsW0), num_threads=cute.arch.WARP_SIZE * 2
        )
        pipeline_o_epi = pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_O_epi.data_ptr(),
            num_stages=self.q_stage,
            producer_group=correction_threads,
            consumer_group=epilogue_threads,
            defer_sync=True,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        # Strip swizzle info to reuse smem
        sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)
        sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)

        sScale = storage.sScale.get_tensor(cute.make_layout(self.q_stage * self.m_block_size * 2))

        thr_mma_qk = tiled_mma_qk.get_slice(mma_tile_coord_v)
        thr_mma_pv = tiled_mma_pv.get_slice(mma_tile_coord_v)

        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        # This is a fake tensor, by right we need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tStS = thr_mma_qk.make_fragment_C(cute.append(qk_acc_shape, self.s_stage))
        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO = thr_mma_pv.make_fragment_C(cute.append(pv_acc_shape, self.q_stage))
        tOtO = cute.make_tensor(tOtO.iterator + self.tmem_o_offset[0], tOtO.layout)
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]
        # Need to multiply by width ratio bc tP is in v_dtype but tmem offsets are in FP32
        tP_width_ratio = Float32.width // self.v_dtype.width
        # Need to adjust the stage stride manually since the two stages aren't contiguous in tmem
        tP_stage_stride = (self.tmem_p_offset[1] - self.tmem_p_offset[0]) * tP_width_ratio
        tOrP = cute.make_tensor(
            tOrP.iterator + self.tmem_p_offset[0] * tP_width_ratio,
            cute.append(tOrP.layout, cute.make_layout((self.s_stage,), stride=(tP_stage_stride,)))
        )

        block_info = BlockInfo(self.cta_tiler[0], self.cta_tiler[1])
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
        )
        AttentionMaskCls = partial(AttentionMask, self.m_block_size, self.n_block_size)
        # Cluster wait before tensor memory alloc
        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        if const_expr(self.use_clc_scheduler):
            clc_response_ptr = storage.clc_response.data_ptr()
            clc_mbar_ptr = storage.clc_mbar_ptr.data_ptr()

            clc_pipeline_producer_group = cutlass_pipeline.CooperativeGroup(
                cutlass_pipeline.Agent.Thread
            )
            num_clc_consumer_warps_per_cta = self.threads_per_cta // cute.arch.WARP_SIZE
            num_clc_consumer_warps = num_clc_consumer_warps_per_cta
            clc_pipeline_consumer_group = cutlass_pipeline.CooperativeGroup(
                cutlass_pipeline.Agent.Thread, cute.arch.WARP_SIZE * num_clc_consumer_warps
            )

            block_idx = cute.arch.block_idx()
            clc = ClcState.create(
                hw_scheduler=ClcDynamicPersistentTileScheduler.create(
                    self.tile_scheduler_cls.clc_problem_shape(tile_sched_params),
                    block_idx,
                    cute.arch.grid_dim(),
                    clc_response_ptr,
                ),
                pipeline=cutlass_pipeline.PipelineClcFetchAsync.create(
                    barrier_storage=clc_mbar_ptr,
                    num_stages=self.sched_stages,
                    producer_group=clc_pipeline_producer_group,
                    consumer_group=clc_pipeline_consumer_group,
                    tx_count=16,
                    cta_layout_vmnk=cta_layout_vmnk,
                ),
                consumer_state=cutlass_pipeline.make_pipeline_state(
                    cutlass_pipeline.PipelineUserType.Consumer, self.sched_stages
                ),
                producer_state=cutlass_pipeline.make_pipeline_state(
                    cutlass_pipeline.PipelineUserType.Producer, self.sched_stages
                ),
            )
            tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params, clc=clc)
        else:
            tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params)
        assert isinstance(tile_scheduler, TileSchedulerProtocol), f"tile_scheduler is not a TileSchedulerProtocol: {type(tile_scheduler)}"

        # --- EMPTY / CLC SCHEDULER WARP ---
        if const_expr(self.use_clc_scheduler):
            if warp_idx == self.clc_scheduler_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                self.clc_scheduler_warp(tile_scheduler)
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if warp_idx == self.empty_warp_ids[i] and warp_idx != self.clc_scheduler_warp_id:
                    cute.arch.setmaxregister_decrease(self.num_regs_other)
                    self.empty_warp(tile_scheduler)
        else:
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if warp_idx == self.empty_warp_ids[i]:
                    cute.arch.setmaxregister_decrease(self.num_regs_other)

        # --- LOAD ---
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
                tile_scheduler=tile_scheduler,
            )

        # --- MMA ---
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            # Alloc tensor memory buffer
            tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
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
            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        # --- Epilogue ---
        if warp_idx >= self.epilogue_warp_ids[0] and warp_idx <= self.epilogue_warp_ids[-1]:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.epilogue_s2g(
                mO,
                sO,
                tma_atom_O,
                pipeline_o_epi,
                SeqlenInfoCls,
                tile_scheduler=tile_scheduler,
            )

        # --- Softmax ---
        if (
            (const_expr(self.q_stage == 2) and warp_idx <= self.softmax1_warp_ids[-1]) or
            (const_expr(self.q_stage == 1) and warp_idx <= self.softmax0_warp_ids[-1])
        ):
            # increase register after decreasing
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            # sync with mma warp before retrieving tmem ptr
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            softmax_loop = partial(
                self.softmax_loop,
                softmax_scale_log2=softmax_scale_log2,
                softmax_scale=softmax_scale,
                thr_mma_qk=thr_mma_qk,
                sScale=sScale,
                mLSE=mLSE,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                block_info=block_info,
                SeqlenInfoCls=SeqlenInfoCls,
                AttentionMaskCls=AttentionMaskCls,
                tile_scheduler=tile_scheduler,
            )

            stage = Int32(0 if const_expr(self.q_stage == 1) or warp_idx < self.softmax1_warp_ids[0] else 1)
            softmax_loop(stage=stage, tStS=tStS)

            tmem_alloc_barrier.arrive()

        # --- Correction ---
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_correction)
            # sync with mma warp before retrieving tmem ptr
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
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
        tile_scheduler: TileSchedulerProtocol,
    ):
        q_producer_phase = Int32(1)
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]

            mK_cur, mV_cur = [t[None, None, head_idx, batch_idx] for t in (mK, mV)]
            gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0))
            gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None))
            tSgK = thr_mma_qk.partition_B(gK)
            tOgV = thr_mma_pv.partition_B(gV)
            tiler_gQ = ((self.mma_tiler_qk[0] * self.q_stage), self.head_dim_padded)
            gQ = cute.local_tile(mQ_cur, tiler_gQ, (m_block, 0))
            gQ = layout_utils.select(
                cute.flat_divide(gQ, (self.mma_tiler_qk[0],)), mode=[0, 2, 1]
            )
            tSgQ = thr_mma_qk.partition_A(gQ)
            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ
            )
            load_Q = partial(self.load_Q, load_Q_fn, pipeline_q=pipeline_q, phase=q_producer_phase)

            tKsK, tKgK = cpasync.tma_partition(
                tma_atom_K,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sK, 0, 3),
                cute.group_modes(tSgK, 0, 3),
            )
            tVsV, tVgV = cpasync.tma_partition(
                tma_atom_V,
                0,  # no multicast
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
            load_K(block=n_block_max - 1, producer_state=kv_producer_state)  # K0
            load_Q(block=0, stage=0)
            kv_producer_state.advance()
            if const_expr(self.q_stage == 2):
                load_Q(block=1, stage=1)
            q_producer_phase ^= 1
            load_V(block=n_block_max - 1, producer_state=kv_producer_state)  # V0
            kv_producer_state.advance()
            for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                n_block = n_block_max - 2 - i
                load_K(block=n_block, producer_state=kv_producer_state)
                kv_producer_state.advance()
                load_V(block=n_block, producer_state=kv_producer_state)
                kv_producer_state.advance()

            work_tile = tile_scheduler.advance_to_next_work()
            # End of persistent scheduler loop

        pipeline_kv.producer_tail(kv_producer_state)
        # This is equivalent to pipeline_q.producer_tail for the TMA-Q producer warp.
        pipeline_q.producer_acquire_w_index_phase(self.q_stage - 1, q_producer_phase)

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.core.ThrMma,
        tiled_mma_pv: cute.core.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        tOrP: cute.Tensor,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        tile_scheduler=None,
    ):
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)

        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op
        q_smem_base = sm100_desc.smem_desc_base_from_tensor(sQ, sm100_desc.Major.K)
        k_smem_base = sm100_desc.smem_desc_base_from_tensor(sK, sm100_desc.Major.K)
        q_smem_start = [sm100_desc.make_smem_desc_start_addr(sQ[None, None, None, stage].iterator) for stage in range(self.q_stage)]

        sm100_utils.declare_ptx_smem_desc(q_smem_start[self.q_stage - 1], q_smem_base, tSrQ[None, None, None, 0].layout, var_name_prefix="fa_fwd_q_smem_desc")
        sm100_utils.declare_ptx_idesc(qk_mma_op, var_name="fa_fwd_qk_mma_idesc")
        sm100_utils.declare_ptx_idesc(pv_mma_op, var_name="fa_fwd_pv_mma_idesc")

        sQ_stage_stride = (sQ.layout.stride[-1] * sQ.element_type.width // 8) >> 4
        if const_expr(self.q_stage == 1):
            sQ_stage_stride = 0
        gemm_Si = [
            partial(
                sm100_utils.gemm_ptx_precomputed_varname,
                self.tmem_s_offset[stage],
                smem_desc_base_b=k_smem_base,
                tCrB_layout=tSrK[None, None, None, 0].layout,
                smem_var_name_prefix=f"fa_fwd_q_smem_desc",
                idesc_var_name=f"fa_fwd_qk_mma_idesc",
                smem_offset=-sQ_stage_stride if stage == 0 else sQ_stage_stride,
                zero_init=True,
                cta_group=1,
            )
            for stage in range(self.q_stage)
        ]
        gemm_Pi = [
            partial(
                sm100_utils.gemm_ptx_partial,
                pv_mma_op,
                self.tmem_o_offset[stage],
                tOrP[None, None, None, stage],
                sA=None,
                split_arrive=self.split_P_arrive if self.split_P_arrive > 0 else None,
                cta_group=1,
            )
            for stage in range(self.q_stage)
        ]

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        P_full_O_rescaled_phase = Int32(0)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
            block_iter_count = n_block_max - n_block_min

            for stage in cutlass.range_constexpr(self.q_stage):
                # GEMM_QK00 (Q0 * K0 -> S0) or GEMM_QK01 (Q1 * K0 -> S1)
                # 1. wait for Q0 / Q1
                pipeline_q.consumer_wait_w_index_phase(stage, mma_q_consumer_phase)
                # 2. wait for K0
                if const_expr(stage == 0):
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                Ki_index = mma_kv_consumer_state.index
                sK_cur = sK[None, None, None, Ki_index]
                gemm_Si[stage](
                    smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator)
                )
                # 4. release S0 / S1
                pipeline_s_p_o.producer_commit_w_index(stage)
            mma_q_consumer_phase ^= 1
            # 5. release K0
            pipeline_kv.consumer_release(mma_kv_consumer_state)
            mma_kv_consumer_state.advance()

            # O hasn't been accumulated yet, its first MMA calculation doesn't need to accumulate
            block_loop_count = block_iter_count - 1
            O_should_accumulate = False
            for i in cutlass.range(block_loop_count, unroll=1):
                # 1. wait for V0
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                mma_kv_release_state = mma_kv_consumer_state.clone()
                Vi_index = mma_kv_consumer_state.index
                tOrVi = tOrV[None, None, None, Vi_index]
                for stage in cutlass.range_constexpr(self.q_stage):
                    pipeline_s_p_o.producer_acquire_w_index_phase(stage, P_full_O_rescaled_phase)
                    sV_cur = sV[None, None, None, Vi_index]
                    gemm_Pi[stage](
                        tCrB=tOrVi,
                        sB=sV_cur,
                        zero_init=not O_should_accumulate,
                        mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(stage) if self.split_P_arrive > 0 else None,
                        mbar_phase=P_full_O_rescaled_phase,
                    )
                    # 4. release V(i-1)
                    if const_expr(stage == self.q_stage - 1):
                        pipeline_kv.consumer_release(mma_kv_release_state)
                        mma_kv_release_state.advance()

                    # GEMM_QK0i (Q0 * Ki -> S0)
                    # 1. wait for Ki
                    if const_expr(stage == 0):
                        mma_kv_consumer_state.advance()
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    Ki_index = mma_kv_consumer_state.index
                    sK_cur = sK[None, None, None, Ki_index]
                    gemm_Si[stage](
                        smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator)
                    )
                    pipeline_s_p_o.producer_commit_w_index(stage)
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                P_full_O_rescaled_phase ^= 1
                O_should_accumulate = True

            # release Q0 & Q1
            for stage in cutlass.range(self.q_stage):
                pipeline_q.consumer_release_w_index(stage)

            # 1. wait for V0
            pipeline_kv.consumer_wait(mma_kv_consumer_state)
            Vi_index = mma_kv_consumer_state.index
            tOrVi = tOrV[None, None, None, Vi_index]
            for stage in cutlass.range_constexpr(self.q_stage):
                pipeline_s_p_o.producer_acquire_w_index_phase(stage, P_full_O_rescaled_phase)
                sV_cur = sV[None, None, None, Vi_index]
                gemm_Pi[stage](
                    tCrB=tOrVi,
                    sB=sV_cur,
                    zero_init=not O_should_accumulate,
                    mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(stage) if self.split_P_arrive > 0 else None,
                    mbar_phase=P_full_O_rescaled_phase,
                )
                # We do need O_full here since for the last tile, by the time the softmax warp
                # has signaled to the correction warps, the softmax warp has just finished
                # computing the row sum of the current tile. It does not guarantee that the 1st
                # tile of the next work tile has been computed yet.
                pipeline_o_acc.producer_commit_w_index(stage)
            P_full_O_rescaled_phase ^= 1
            pipeline_kv.consumer_release(mma_kv_consumer_state)
            mma_kv_consumer_state.advance()

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

    # for both softmax0 and softmax1 warp group
    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        thr_mma_qk: cute.core.ThrMma,
        tStS: cute.Tensor,  # ((TILE_M, TILE_N), 1, 1, q_stage)
        sScale: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        tile_scheduler=None,
    ):
        """Softmax loop for one Q stage (of q_stage). Handles seqlen masking on the first N-block."""
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids)
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        tSAcc = tStS[(None, None), 0, 0, stage]  # (128, 128)
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]  # (128, 128)

        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tStP_layout = cute.composition(
            tSAcc.layout, cute.make_layout((self.m_block_size, tilePlikeFP32))
        )
        tStP = cute.make_tensor(tSAcc.iterator + self.tmem_s_to_p_offset, tStP_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.qk_acc_dtype
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
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)

            mask = AttentionMaskCls(seqlen)
            mask_fn = partial(
                mask.apply_mask_sm100,
                m_block=self.q_stage * m_block + stage,
                thr_mma=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
            )

            softmax = SoftmaxSm100.create(
                softmax_scale_log2,
                rescale_threshold=8.0 if const_expr(self.q_dtype.width == 16) else 0.0,
                softmax_scale=softmax_scale,
            )
            softmax.reset()

            softmax_step = partial(
                self.softmax_step,
                softmax=softmax,
                thr_mma_qk=thr_mma_qk,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                tStS_t2r=tStS_t2r,
                tStP_r2t=tStP_r2t,
                sScale=sScale,
                stage=stage,
            )

            pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
            sm_stats_producer_phase ^= 1

            mma_si_consumer_phase, sm_stats_producer_phase = softmax_step(
                mma_si_consumer_phase,
                sm_stats_producer_phase,
                n_block_max - 1,
                is_first=True,
                mask_fn=partial(mask_fn, mask_seqlen=True),
            )
            n_block_max -= 1
            # The remaining iterations have no masking
            for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                n_block = n_block_max - n_tile - 1
                mma_si_consumer_phase, sm_stats_producer_phase = softmax_step(
                    mma_si_consumer_phase, sm_stats_producer_phase, n_block,
                )

            sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
            if const_expr(mLSE is not None):
                sScale[
                    tidx + stage * self.m_block_size + self.q_stage * self.m_block_size
                ] = softmax.row_max[0]
            sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
        # End of persistent scheduler loop

        # This is equivalent to pipeline_sm_stats.producer_tail
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        sm_stats_producer_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        thr_mma_qk: cute.core.ThrMma,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        stage: int | Int32,
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
    ) -> Tuple[cute.Int32, cute.Int32]:
        """One online-softmax block step: load S, apply mask, update row_max / row_sum, write P."""
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]  # (128, 128)
        tScP_shape = (self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape, tilePlikeFP32)

        # Wait for Si
        pipeline_s_p_o.consumer_wait_w_index_phase(stage, mma_si_consumer_phase)
        tSrS_t2r = cute.make_fragment(thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype)
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)

        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block)
        row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)

        if const_expr(not is_first):
            thread_idx = thr_tmem_load.thr_idx
            sScale[thread_idx + stage * self.m_block_size] = acc_scale
        # Notify correction wg that row_max is ready
        sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)

        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        tSrP_r2t_f32 = cute.make_fragment(
            thr_tmem_store.partition_S(cute.make_identity_tensor(tScP_shape)).shape, Float32
        )
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype), tSrS_t2r.layout
        )
        softmax.apply_exp2_convert(
            tSrS_t2r,
            tSrP_r2t,
            ex2_emu_freq=self.ex2_emu_freq if const_expr(mask_fn is None) else 0,
            ex2_emu_start_frg=self.ex2_emu_start_frg,
        )
        for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2])):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
            if const_expr(self.split_P_arrive > 0):
                split_P_arrive_idx = cute.size(tStP_r2t.shape[2]) * self.split_P_arrive // self.n_block_size
                if const_expr(i + 1 == split_P_arrive_idx):
                    # Notify mma warp that the 1st half of P is ready
                    cute.arch.fence_view_async_tmem_store()
                    pipeline_s_p_o.consumer_release_w_index(stage)
        # Notify mma warp that the 2nd half of P is ready
        cute.arch.fence_view_async_tmem_store()
        if const_expr(self.split_P_arrive > 0):
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                pipeline_p_lastsplit.producer_commit_w_index(stage)
        else:
            pipeline_s_p_o.consumer_release_w_index(stage)
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
        softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        return mma_si_consumer_phase ^ 1, sm_stats_producer_phase ^ 1

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
        tile_scheduler=None,
    ):
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.correction_warp_ids))
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        # First iter: no correction is required
        # Notify mma warp that O has been rescaled
        for stage in cutlass.range(self.q_stage):
            pipeline_s_p_o.consumer_release_w_index(stage)

        sm_stats_consumer_phase = Int32(0)
        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)

            # LSE rows default to -inf for masked-out rows
            stats = [(0.0, -Float32.inf if const_expr(mLSE is not None) else None, True)] * self.q_stage

            total_block_count = n_block_max - n_block_min

            # Ignore first signal from softmax as no correction is required
            sm_stats_barrier.arrive_and_wait_w_index(index=0 * 4 + warp_idx)
            pipeline_sm_stats.consumer_release_w_index(0)
            if const_expr(self.q_stage == 2):
                sm_stats_barrier.arrive_and_wait_w_index(index=1 * 4 + warp_idx)
            sm_stats_consumer_phase ^= 1

            for i in cutlass.range(total_block_count - 1, unroll=1):
                for stage in cutlass.range_constexpr(self.q_stage):
                    # wait for S0 / S1
                    sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
                    scale = sScale[tidx + stage * self.m_block_size]
                    should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
                    if should_rescale:
                        self.correction_rescale(thr_mma_pv, tOtO[None, None, None, stage], tidx, scale)
                    # Notify mma warp that O has been rescaled
                    pipeline_s_p_o.consumer_release_w_index(stage)
                    pipeline_sm_stats.consumer_release_w_index(self.q_stage - 1 - stage)
                sm_stats_consumer_phase ^= 1
            if const_expr(self.q_stage == 2):
                pipeline_sm_stats.consumer_release_w_index(1)
            # End of seqlen_corr_loop_steps

            for stage in cutlass.range_constexpr(self.q_stage):
                sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
                row_sum = sScale[tidx + stage * self.m_block_size]
                if const_expr(mLSE is not None):
                    row_max = sScale[tidx + stage * self.m_block_size + self.q_stage * self.m_block_size]
                else:
                    row_max = None
                pipeline_sm_stats.consumer_release_w_index(stage)
                acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                stats[stage] = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                scale = cute.arch.rcp_approx(row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0)
                # Wait for the last O to be ready from the MMA warp
                pipeline_o_acc.consumer_wait_w_index_phase(stage, o_corr_consumer_phase)
                pipeline_o_epi.producer_acquire_w_index_phase(stage, corr_epi_producer_phase)
                self.correction_epilogue(
                    thr_mma_pv,
                    tOtO[None, None, None, stage],
                    tidx,
                    scale,
                    sO[None, None, stage],
                )
                # Signal for the next work tile that O buffers in tmem are already read, so
                # mma warp can write to them
                pipeline_s_p_o.consumer_release_w_index(stage)
                pipeline_o_epi.producer_commit_w_index(stage)

            o_corr_consumer_phase ^= 1
            sm_stats_consumer_phase ^= 1
            corr_epi_producer_phase ^= 1

            if const_expr(mLSE is not None):
                mLSE_cur = mLSE[None, head_idx, batch_idx]
                for stage in cutlass.range_constexpr(self.q_stage):
                    m_tile_idx = m_block * self.q_stage + stage
                    row_sum, row_max, acc_O_mn_row_is_zero_or_nan = stats[stage]
                    LN2 = math.log(2.0)
                    lse = (
                        (row_max * softmax_scale_log2 + cute.math.log2(row_sum, fastmath=True)) * LN2
                        if not acc_O_mn_row_is_zero_or_nan
                        else -Float32.inf
                    )
                    seqlen_q = seqlen.seqlen_q
                    gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_tile_idx,))
                    if tidx < seqlen_q - m_tile_idx * self.m_block_size:
                        gLSE[tidx] = lse

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
        # End of persistent scheduler loop

        # This is equivalent to pipeline_o_epi.consumer_tail() for the correction warps
        pipeline_o_epi.producer_acquire_w_index_phase(self.q_stage - 1, corr_epi_producer_phase)

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.core.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        scale: Float32,
    ):
        """Rescale the in-flight O accumulator in tmem by `scale` (online-softmax correction)."""
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        corr_tile_size = 16  # tuneable parameter
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)), self.pv_acc_dtype
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tOtO_i = cute.composition(tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.composition(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i).get_slice(tidx)
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i).get_slice(tidx)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i)
        tOrO_t2r_shape = thr_tmem_load.partition_D(tOcO_i).shape
        tOtO_r2t = thr_tmem_store.partition_D(tOtO_i)

        frg_count = self.head_dim_v_padded // corr_tile_size
        tOrO_frg = cute.make_fragment((tOrO_t2r_shape, frg_count), self.pv_acc_dtype)
        for i in cutlass.range_constexpr(frg_count):
            tOrO_frg = cute.make_fragment(tOrO_t2r_shape, self.pv_acc_dtype)
            tOtO_t2r_i = cute.make_tensor(tOtO_t2r.iterator + i * corr_tile_size, tOtO_t2r.layout)
            cute.copy(thr_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]), (scale, scale)
                )
            tOtO_r2t_i = cute.make_tensor(tOtO_r2t.iterator + i * corr_tile_size, tOtO_r2t.layout)
            cute.copy(thr_tmem_store, tOrO_frg, tOtO_r2t_i)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_epilogue(
        self,
        thr_mma: cute.core.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        scale: Float32,
        sO: cute.Tensor,
    ):
        """Apply final scaling and write attention output to smem for the TMA store epilogue."""

        corr_tile_size = 8 * 32 // self.o_dtype.width
        # Use CTA 0 mapping for smem partitioning since sO is per-CTA sized
        tOsO = thr_mma.get_slice(0).partition_C(sO)
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))

        tOtO_i = cute.logical_divide(tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.logical_divide(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOsO_i = cute.logical_divide(tOsO, cute.make_layout((self.m_block_size, corr_tile_size)))

        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
            self.mma_tiler_pv,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=False,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_i[(None, None), 0])
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tOsO_s2r = copy_utils.partition_D_position_independent(thr_tmem_load, tOsO_i[(None, None), None])
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
        for i in cutlass.range(self.head_dim_v_padded // corr_tile_size, unroll_full=True):
            tOtO_t2r_i = tOtO_t2r[None, 0, 0, i]
            tOsO_r2s_i = tOsO_s2r[None, 0, 0, i]
            tOrO_frg = cute.make_fragment(tOcO_t2r[None, 0, 0, i].shape, self.pv_acc_dtype)
            cute.copy(tiled_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]), (scale, scale)
                )
            copy_utils.cvt_copy(tiled_smem_store, tOrO_frg, tOsO_r2s_i)
        cute.arch.fence_view_async_shared()

    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        tma_atom_O: cute.CopyAtom,
        pipeline_o_epi: pipeline.PipelineAsync,
        SeqlenInfoCls: Callable,
        tile_scheduler=None,
    ):
        epi_consumer_phase = Int32(0)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
            tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.head_dim_v_padded)
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

    @cute.jit
    def clc_scheduler_warp(
        self,
        tile_scheduler: TileSchedulerProtocol,
    ):
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            tile_scheduler.prefetch_next_work()
            work_tile = tile_scheduler.advance_to_next_work()
            if cute.arch.thread_idx()[0] == self.clc_scheduler_warp_id * cute.arch.WARP_SIZE:
                fa_printf(
                    3,
                    "[CLC] query sm={} cta={} (m_blk={},h={},b={},s={}) valid={}\n",
                    smid(),
                    cute.arch.block_idx()[0],
                    work_tile.tile_idx[0],
                    work_tile.tile_idx[1],
                    work_tile.tile_idx[2],
                    work_tile.tile_idx[3],
                    work_tile.is_valid_tile,
                )
        tile_scheduler.producer_tail()

    @cute.jit
    def empty_warp(
        self,
        tile_scheduler: TileSchedulerProtocol,
    ):
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_tile = tile_scheduler.advance_to_next_work()

    def load_Q(
        self,
        load_Q_fn: Callable,
        pipeline_q: pipeline.PipelineAsync,
        block: Int32,
        stage: int,
        phase: Int32,
    ):
        pipeline_q.producer_acquire_w_index_phase(stage, phase)
        load_Q_fn(src_idx=block, dst_idx=stage, tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(stage))

    @cute.jit
    def load_KV(
        self,
        tma_atom: cute.CopyAtom,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        block: Int32,
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
    ):
        assert K_or_V in ("K", "V")
        stage = producer_state.index
        # K and V have identical tx counts in this MHA build; no extra_tx_count needed.
        pipeline_kv.producer_acquire(producer_state)
        tXsX_cur = tXsX[None, stage]
        tXgX_cur = tXgX[None, block]
        cute.copy(tma_atom, tXgX_cur, tXsX_cur, tma_bar_ptr=pipeline_kv.producer_get_barrier(producer_state))
