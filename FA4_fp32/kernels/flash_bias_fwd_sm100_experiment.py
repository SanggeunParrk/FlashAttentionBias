# Experimental A=1 bias-path variants for diagnosing SM100 fp32 bias overhead.

from typing import Tuple, Optional, Callable

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass import pipeline

from FA4_fp32.kernels.flash_bias_fwd_sm100_smem import FlashAttentionBiasForwardSm100Smem


class FlashAttentionBiasForwardSm100ScaleOnly(FlashAttentionBiasForwardSm100Smem):
    """Same 5D bias kernel plumbing, but no bias staging; only scale S in registers."""

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        sm_stats_producer_phase: Int32,
        n_block: Int32,
        softmax,
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
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.qkvo_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]
        tScP_shape = (self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape, tilePlikeFP32)

        pipeline_s_p_o.consumer_wait_w_index_phase(stage, mma_si_consumer_phase)
        tSrS_t2r = cute.make_fragment(
            thr_tmem_load.partition_D(tScS).shape, self.acc_dtype
        )
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
        self.apply_scale_only(tSrS_t2r, softmax.softmax_scale)

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
    def apply_scale_only(self, tSrS_t2r: cute.Tensor, softmax_scale: Float32):
        z = Float32(0.0)
        for i in cutlass.range(0, cute.size(tSrS_t2r.shape), 2, unroll_full=True):
            tSrS_t2r[i], tSrS_t2r[i + 1] = cute.arch.fma_packed_f32x2(
                (tSrS_t2r[i], tSrS_t2r[i + 1]),
                (softmax_scale, softmax_scale),
                (z, z),
            )


class FlashAttentionBiasForwardSm100ZeroStage(FlashAttentionBiasForwardSm100ScaleOnly):
    """No GMEM bias read, but keep zero SMEM staging, barriers, and SMEM->register apply."""

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        sm_stats_producer_phase: Int32,
        n_block: Int32,
        softmax,
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

        self.load_zero_bias_smem(sBias, stage, tidx, bias_smem_barrier)
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
    def load_zero_bias_smem(
        self,
        sBias: cute.Tensor,
        stage: int | Int32,
        tidx: Int32,
        bias_smem_barrier: pipeline.NamedBarrier,
    ):
        bias_smem_barrier.arrive_and_wait_w_index(stage)
        total = self.m_block_size * self.n_block_size
        z = Float32(0.0)
        for offset in cutlass.range(
            tidx,
            total,
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids),
            unroll=1,
        ):
            q = offset // self.n_block_size
            k = offset - q * self.n_block_size
            sBias[q, k, stage] = z
        cute.arch.fence_view_async_shared()
        bias_smem_barrier.arrive_and_wait_w_index(stage)
