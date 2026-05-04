# Copyright (c) 2025, Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao.
import math
from typing import Callable, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute import FastDivmodDivisor
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.utils import LayoutEnum
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass.pipeline import PipelineAsync

import quack.activation
from quack import layout_utils
from FA4_fp32.core import utils
from FA4_fp32.infra.cute_dsl_utils import assume_tensor_aligned
from FA4_fp32.core import copy_utils
from FA4_fp32.core import pipeline
from FA4_fp32.arch.blackwell_helpers import gemm_w_idx, gemm_ptx_w_idx  # noqa
from FA4_fp32.core.mask import AttentionMask
from FA4_fp32.core.seqlen_info import SeqlenInfoQK
from FA4_fp32.core.block_info import BlockInfo
from quack.cute_dsl_utils import ParamsBase
from FA4_fp32.core.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
)

from FA4_fp32.core.named_barrier import NamedBarrierBwdSm100


class FlashAttentionBackwardSm100:
    arch = 100

    def __init__(
        self,
        head_dim: int,
        tile_m: int = 128,
        tile_n: int = 128,
        subtile_factor: cutlass.Constexpr[int] = 1,
        q_dtype: Optional[type] = None,
    ):
        # B200 / non-causal / MHA / Q=K=V (shape, dtype, head_dim).
        # Cache q_dtype so __init__-time layout decisions (TMEM offsets) can branch.
        # __call__ will re-confirm from the actual tensor element_type.
        self._init_q_dtype = q_dtype
        # Q/K/V share shape and dtype, so head_dim_v == head_dim always.
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        self.tile_hdimv = self.tile_hdim

        self.tile_m = tile_m
        self.tile_n = tile_n

        assert self.tile_hdim <= 128

        # CTA tiler
        self.cta_tiler = (tile_n, tile_m, self.tile_hdim)
        # S = K @ Q.T
        self.mma_tiler_kq = (tile_n, tile_m, self.tile_hdim)
        # dP = V @ dO.T
        self.mma_tiler_vdo = (tile_n, tile_m, self.tile_hdimv)
        # dV = P.T @ dO
        self.mma_tiler_pdo = (tile_n, self.tile_hdimv, tile_m)
        # dK = dS.T @ Q
        self.mma_tiler_dsq = (tile_n, self.tile_hdim, tile_m)
        # dQ = dS @ K
        self.mma_tiler_dsk = (tile_m, self.tile_hdim, tile_n)

        self.acc_dtype = Float32

        self.cluster_shape_mn = (1, 1)
        self.subtile_factor = subtile_factor
        self.vec_size: cutlass.Constexpr = 4
        self.qk_acc_dtype = Float32

        self.reduce_warp_ids = (0, 1, 2, 3)
        self.compute_warp_ids = (4, 5, 6, 7, 8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.empty_warp_id = 15

        # 15 live warps + 1 unused (warp 14 left idle for stack alignment / tests).
        # We keep the slot count at 16 warps -> 512 threads for SMEM budget parity.
        self.threads_per_cta = cute.arch.WARP_SIZE * 16
        # NamedBarrier
        self.compute_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.Compute),
            num_threads=len(self.compute_warp_ids) * cute.arch.WARP_SIZE,
        )
        self.reduce_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.dQaccReduce),
            num_threads=len(self.reduce_warp_ids) * cute.arch.WARP_SIZE,
        )
        # TMEM setup (1-CTA, hdim <= 128 layout)
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        self.tmem_S_offset = 0
        self.tmem_P_offset = 0  # overlap with S (fp16/bf16 packing trick)
        self.tmem_dV_offset = self.tmem_S_offset + self.tile_n
        self.tmem_dP_offset = self.tmem_dV_offset + self.tile_hdimv
        self.tmem_dQ_offset = self.tmem_dP_offset
        self.tmem_dK_offset = self.tmem_dP_offset + self.tile_m
        self.tmem_dS_offset = self.tmem_dP_offset  # overlap with dP

        # Non-causal/local, non-deterministic, non-2CTA register budget.
        self.num_regs_reduce = 152
        self.num_regs_compute = 136
        self.num_regs_load = 96 - 8
        self.num_regs_mma = self.num_regs_load
        self.num_regs_empty = 24

        assert (
            self.num_regs_reduce
            + self.num_regs_compute * 2
            + max(self.num_regs_load, self.num_regs_mma)
            <= 512
        )
        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.Q_stage = 2
        self.dO_stage = 1
        self.single_stage = 1
        # LSE_stage = Q_stage and dPsum_stage = dO_stage
        self.sdKVaccum_stage = 2
        # number of tma reduce adds per dQacc mma
        # fp32 I/O path: TMEM dQ accumulator is tiled with 16-col granularity because
        # TF32 MMA produces half the columns-per-instr of fp16/bf16 MMA.
        if self.q_dtype is Float32:
            self.dQ_reduce_ncol = 16
        else:
            self.dQ_reduce_ncol = 32
        self.sdQaccum_stage = 64 // self.dQ_reduce_ncol
        self.dQ_reduce_ncol_t2r = self.dQ_reduce_ncol
        assert self.tile_hdim % self.dQ_reduce_ncol == 0
        self.dQaccum_reduce_stage = self.tile_hdim // self.dQ_reduce_ncol
        self.dQaccum_reduce_stage_t2r = self.tile_hdim // self.dQ_reduce_ncol_t2r
        # number of tma reduce adds for dKacc and dVacc epilogue (must divide hdim_per_wg)
        self.dK_reduce_ncol = math.gcd(32, self.tile_hdim // 2)

    def _get_tiled_mma(self):
        # S.T = K @ Q.T
        tiled_mma_S = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_kq[:2],
        )
        # dP.T = V @ dO.T
        tiled_mma_dP = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_vdo[:2],
        )
        # dV += P.T @ dO --> (K, MN) major
        tiled_mma_dV = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,  # P_major_mode
            tcgen05.OperandMajorMode.MN,  # dO_major_mode
            self.acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_pdo[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        # dK += dS.T @ Q from TMEM
        mma_dK_a_src = tcgen05.OperandSource.TMEM
        tiled_mma_dK = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,  # dS_major_mode
            tcgen05.OperandMajorMode.MN,  # Q_major_mode
            self.acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_dsq[:2],
            a_source=mma_dK_a_src,
        )
        # dQ = dS @ K
        tiled_mma_dQ = sm100_utils_basic.make_trivial_tiled_mma(
            self.k_dtype,
            tcgen05.OperandMajorMode.MN,  # dS_major_mode
            tcgen05.OperandMajorMode.MN,  # Kt_major_mode
            self.acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_dsk[:2],
        )
        return tiled_mma_S, tiled_mma_dP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ

    def _setup_smem_layout(self):
        # S.T = K @ Q.T
        sK_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_S,
            self.mma_tiler_kq,
            self.k_dtype,
            1,
        )
        self.sK_layout = cute.slice_(sK_layout, (None, None, None, 0))
        self.sQ_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_S,
            self.mma_tiler_kq,
            self.q_dtype,
            self.Q_stage,
        )
        # dP.T = V @ dO.T
        sV_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            self.v_dtype,
            1,
        )
        self.sV_layout = cute.slice_(sV_layout, (None, None, None, 0))
        self.sdOt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            self.do_dtype,
            self.dO_stage,
        )
        # dV += P.T @ dO
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.do_dtype,
            1,
        )
        self.tP_layout = cute.slice_(tP_layout, (None, None, None, 0))
        self.sdO_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.do_dtype,
            self.dO_stage,
        )
        # dK += dS.T @ Q
        sdSt_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.ds_dtype,
            1,
        )
        self.sdSt_layout = cute.slice_(sdSt_layout, (None, None, None, 0))
        tdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.ds_dtype,
            1,
        )
        self.tdS_layout = cute.slice_(tdS_layout, (None, None, None, 0))
        self.sQt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.q_dtype,
            self.Q_stage,
        )
        # dQ = dS @ K
        sdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.ds_dtype,
            1,
        )
        self.sdS_layout = cute.slice_(sdS_layout, (None, None, None, 0))
        sKt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.k_dtype,
            1,
        )
        self.sKt_layout = cute.slice_(sKt_layout, (None, None, None, 0))
        self.sdS_xchg_layout = cute.make_layout(shape=(self.tile_n, self.tile_m // 2))

        self.sdQaccum_layout = cute.make_layout(
            (self.tile_m * self.dQ_reduce_ncol, self.sdQaccum_stage)
        )
        self.sLSE_layout = cute.make_layout(
            shape=(self.tile_m, self.Q_stage), stride=(1, cute.round_up(self.tile_m, 64))
        )
        self.sdPsum_layout = cute.make_layout(
            shape=(self.tile_m, self.dO_stage),
            stride=(1, cute.round_up(self.tile_m, 64)),
        )
        self.sdK_epi_tile = (
            self.tile_n,
            math.gcd(128 // (self.dk_dtype.width // 8), self.tile_hdim // 2),  # 64 or 32
        )  # subtiles mma_tiler_dsq[:2] = mma_tiler_pdo[:2]
        self.sdV_epi_tile = (
            self.tile_n,
            math.gcd(128 // (self.dk_dtype.width // 8), self.tile_hdimv // 2),  # 64 or 32
        )  # subtiles mma_tiler_dsq[:2] = mma_tiler_pdo[:2]
        # headdim_64 gets 1 stage
        self.num_epi_stages = max(1, (self.tile_hdim // 2) // self.sdK_epi_tile[1])
        self.num_epi_stages_v = max(1, (self.tile_hdimv // 2) // self.sdV_epi_tile[1])
        self.sdK_flat_epi_tile = self.tile_n * (self.tile_hdim // 2) // self.num_epi_stages
        self.sdV_flat_epi_tile = self.tile_n * (self.tile_hdimv // 2) // self.num_epi_stages_v
        self.sdK_layout = sm100_utils_basic.make_smem_layout_epi(
            self.dk_dtype,
            LayoutEnum.ROW_MAJOR,
            self.sdK_epi_tile,
            2,  # num compute wgs
        )
        self.sdV_layout = sm100_utils_basic.make_smem_layout_epi(
            self.dv_dtype,
            LayoutEnum.ROW_MAJOR,
            self.sdV_epi_tile,
            2,  # num compute wgs
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        # Stream must come right after softmax_scale: the stripped interface
        # only passes the dense tensors + softmax_scale + stream.
        stream: cuda.CUstream = None,
    ):
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.do_dtype = mdO.element_type
        self.lse_dtype = mLSE.element_type
        self.dpsum_dtype = mdPsum.element_type
        self.dqaccum_dtype = mdQaccum.element_type
        self.dk_dtype = mdK.element_type
        self.dv_dtype = mdV.element_type
        self.ds_dtype = self.q_dtype

        mdQaccum, mdK, mdV = [assume_tensor_aligned(t) for t in (mdQaccum, mdK, mdV)]

        # Dense: (b, s, n, h) --> (s, h, n, b)
        mQ, mdO = [layout_utils.select(t, mode=[1, 3, 2, 0]) for t in (mQ, mdO)]
        mK, mV = [layout_utils.select(t, mode=[1, 3, 2, 0]) for t in (mK, mV)]
        mdK, mdV = [layout_utils.select(t, mode=[1, 3, 2, 0]) for t in (mdK, mdV)]

        # Dense: (b, n, s) --> (s, n, b)
        mLSE, mdPsum, mdQaccum = [
            layout_utils.select(t, mode=[2, 1, 0])
            for t in (mLSE, mdPsum, mdQaccum)
        ]

        # (s, h, n, b) --> (h, s, n, b)
        mdO = layout_utils.select(mdO, mode=[1, 0, 2, 3])

        self._setup_attributes()
        (
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dK,
            self.tiled_mma_dV,
            self.tiled_mma_dQ,
        ) = self._get_tiled_mma()
        self._setup_smem_layout()

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (self.tiled_mma_S.thr_id.shape,),
        )

        self.mdK_layout_enum = LayoutEnum.from_tensor(mdK)
        self.mdV_layout_enum = LayoutEnum.from_tensor(mdV)

        tma_copy_op_dKV = cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_dK, mdK_tma_tensor = cpasync.make_tiled_tma_atom(
            tma_copy_op_dKV,
            mdK,
            cute.select(self.sdK_layout, mode=[0, 1]),
            self.sdK_epi_tile,
            1,  # no mcast
        )
        tma_atom_dV, mdV_tma_tensor = cpasync.make_tiled_tma_atom(
            tma_copy_op_dKV,
            mdV,
            cute.select(self.sdV_layout, mode=[0, 1]),
            self.sdV_epi_tile,
            1,  # no mcast
        )

        thr_layout_r2s_dKV = cute.make_ordered_layout((128, 1), order=(1, 0))  # 128 threads
        val_layout_r2s_dKV = cute.make_ordered_layout(
            (1, 128 // self.dk_dtype.width), order=(1, 0)
        )  # 4 or 8 vals for 16 byte store
        copy_atom_r2s_dKV = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dk_dtype,
            num_bits_per_copy=128,
        )
        tiled_copy_r2s_dKV = cute.make_tiled_copy_tv(
            copy_atom_r2s_dKV, thr_layout_r2s_dKV, val_layout_r2s_dKV
        )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        # S.T = K @ Q.T
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mK,
            cute.select(self.sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_S,
            self.cluster_layout_vmnk.shape,
        )
        Q_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma_S.thr_id
        )
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            Q_tma_op,
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_S,
            self.cluster_layout_vmnk.shape,
        )
        # dP.T = V @ dO.T
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mV,
            cute.select(self.sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_vdo,
            self.tiled_mma_dP,
            self.cluster_layout_vmnk.shape,
        )
        # dV = P.T @ dO
        dO_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma_dV.thr_id
        )
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
            dO_tma_op,
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1, 2]),
            self.mma_tiler_pdo,
            self.tiled_mma_dV,
            self.cluster_layout_vmnk.shape,
        )
        self.tma_copy_bytes = {
            name: 1
            * cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
                ("dO", mdO, self.sdO_layout),
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dQ"] = self.tile_m * self.dQ_reduce_ncol * Float32.width // 8
        self.tma_copy_bytes["dKacc"] = self.tile_n * self.dK_reduce_ncol * Float32.width // 8
        self.tma_copy_bytes["dS"] = cute.size_in_bytes(self.ds_dtype, self.sdS_layout)
        self.tma_copy_bytes["sdS_xchg"] = self.tma_copy_bytes["dS"] // 2  # Half of dS for exchange

        TileScheduler = SingleTileScheduler
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(cute.size(mK.shape[0]), self.cta_tiler[0]),
            num_head=cute.size(mQ.shape[2]),
            num_batch=cute.size(mK.shape[3]),
            seqlen_k=cute.size(mQ.shape[0]),  # pass seqlen_q (S = K @ Q.T tiles over Q)
            headdim=mQ.shape[1],
            headdim_v=mV.shape[1],
            element_size=self.k_dtype.width // 8,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        # Compute allocation sizes for shared buffers that are reused
        # sQ is reused for sdK, sdO is reused for sdV
        sQ_alloc_bytes = max(
            cute.size_in_bytes(self.q_dtype, self.sQ_layout),
            cute.size_in_bytes(self.dk_dtype, self.sdK_layout),
        )
        sdO_alloc_bytes = max(
            cute.size_in_bytes(self.dv_dtype, self.sdV_layout),
            cute.size_in_bytes(self.do_dtype, self.sdO_layout),
        )

        sdK_bytes = cute.size_in_bytes(self.dk_dtype, self.sdK_layout)
        sdV_bytes = cute.size_in_bytes(self.dv_dtype, self.sdV_layout)
        assert sdV_bytes <= sdO_alloc_bytes, "sdV doesn't fit in sdO storage allocation"
        assert sdK_bytes <= sQ_alloc_bytes, "sdK doesn't fit in sQ storage allocation"

        @cute.struct
        class SharedStorage:
            Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
            dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
            LSE_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
            dPsum_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
            S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
            dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
            dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
            dKV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.sdKVaccum_stage]
            dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding_buf: Int32
            tmem_dealloc_mbar_ptr: Int64

            sQ: cute.struct.Align[
                cute.struct.MemRange[cute.Uint8, sQ_alloc_bytes],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(self.sK_layout)],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(self.sV_layout)],
                self.buffer_align_bytes,
            ]
            sdO: cute.struct.Align[
                cute.struct.MemRange[cute.Uint8, sdO_alloc_bytes],
                self.buffer_align_bytes,
            ]
            sdS: cute.struct.Align[
                cute.struct.MemRange[self.ds_dtype, cute.cosize(self.sdSt_layout)],
                128,
            ]
            sLSE: cute.struct.Align[
                cute.struct.MemRange[self.lse_dtype, cute.cosize(self.sLSE_layout)],
                128,
            ]
            sdPsum: cute.struct.Align[
                cute.struct.MemRange[self.dpsum_dtype, cute.cosize(self.sdPsum_layout)],
                128,
            ]
            sdQaccum: cute.struct.Align[
                cute.struct.MemRange[self.dqaccum_dtype, cute.cosize(self.sdQaccum_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        LOG2_E = math.log2(math.e)
        # No score_mod: bake scale into log2.
        softmax_scale_log2 = softmax_scale * LOG2_E

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            mLSE,
            mdPsum,
            tma_tensor_dO,
            mdV,
            mdK,
            mdQaccum,
            mdV_tma_tensor,
            mdK_tma_tensor,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dV,
            tma_atom_dK,
            self.sQ_layout,
            self.sQt_layout,
            self.sK_layout,
            self.sKt_layout,
            self.sV_layout,
            self.sLSE_layout,
            self.sdPsum_layout,
            self.sdO_layout,
            self.sdOt_layout,
            self.sdSt_layout,
            self.sdS_layout,
            self.sdQaccum_layout,
            self.sdK_layout,
            self.sdV_layout,
            self.tP_layout,
            self.tdS_layout,
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dV,
            self.tiled_mma_dK,
            self.tiled_mma_dQ,
            tiled_copy_r2s_dKV,
            softmax_scale,
            softmax_scale_log2,
            tile_sched_params,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=None,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdO: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdV_tma_tensor: cute.Tensor,
        mdK_tma_tensor: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        tma_atom_dK: cute.CopyAtom,
        sQ_layout: cute.ComposedLayout,
        sQt_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sKt_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        sdPsum_layout: cute.Layout,
        sdO_layout: cute.ComposedLayout,
        sdOt_layout: cute.ComposedLayout,
        sdSt_layout: cute.ComposedLayout,
        sdS_layout: cute.ComposedLayout,
        sdQaccum_layout: cute.Layout,
        sdK_layout: cute.ComposedLayout | cute.Layout,
        sdV_layout: cute.ComposedLayout | cute.Layout,
        tP_layout: cute.ComposedLayout,
        tdS_layout: cute.ComposedLayout,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        tiled_copy_r2s_dKV: cute.TiledCopy,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        tile_sched_params: ParamsBase,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = 0  # cta_group_size == 1

        # Prefetch tma descriptor
        if warp_idx == self.load_warp_id:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_Q)
                cpasync.prefetch_descriptor(tma_atom_K)
                cpasync.prefetch_descriptor(tma_atom_V)
                cpasync.prefetch_descriptor(tma_atom_dO)
                cpasync.prefetch_descriptor(tma_atom_dV)
                cpasync.prefetch_descriptor(tma_atom_dK)

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_S.thr_id.shape,),
        )

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_alloc_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE
            * len((self.mma_warp_id, *self.compute_warp_ids, *self.reduce_warp_ids)),
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        # UMMA producers and AsyncThread consumers
        pipeline_producer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        pipeline_consumer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids)
        )
        pipeline_S_P = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.S_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_dP = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dP_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_dKV = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=2,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dKV_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_consumer_group_MMA_AsyncThread_dQ = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.reduce_warp_ids) * 1,
        )  # Compute
        pipeline_dQ = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread_dQ,
            barrier_storage=storage.dQ_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # AsyncThread producers and UMMA consumers
        # Only 1 thread per warp will signal
        pipeline_PdS_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.compute_warp_ids) * 1,
        )  # Compute
        pipeline_PdS_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )  # MMA
        pipeline_dS = cutlass.pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=pipeline_PdS_producer_group,
            consumer_group=pipeline_PdS_consumer_group,
            barrier_storage=storage.dS_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # TMA producer and UMMA consumers
        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.load_warp_id])
        )
        # cluster=(1,1) so mcast factor is 1.
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        pipeline_consumer_group_compute = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.compute_warp_ids) * 1,
        )
        pipeline_LSE = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.LSE_mbar_ptr.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group_compute,
            tx_count=self.tma_copy_bytes["LSE"],
            # cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_dPsum = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.dPsum_mbar_ptr.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group_compute,
            tx_count=self.tma_copy_bytes["dPsum"],
            # cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_Q = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.Q_mbar_ptr.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        pipeline_dO = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.dO_mbar_ptr.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["dO"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=False,
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner, dtype=self.q_dtype)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sdSt = storage.sdS.get_tensor(sdSt_layout.outer, swizzle=sdSt_layout.inner)
        sdS = cute.make_tensor(cute.recast_ptr(sdSt.iterator, sdS_layout.inner), sdS_layout.outer)

        sdO = storage.sdO.get_tensor(
            sdO_layout.outer, swizzle=sdO_layout.inner, dtype=self.do_dtype
        )

        # Transposed views of Q/K/dO — required as B operand of dK/dQ/dP GEMMs.
        sQt = storage.sQ.get_tensor(
            sQt_layout.outer, swizzle=sQt_layout.inner, dtype=self.q_dtype
        )
        sKt = storage.sK.get_tensor(
            sKt_layout.outer, swizzle=sKt_layout.inner, dtype=self.k_dtype
        )
        sdOt = storage.sdO.get_tensor(
            sdOt_layout.outer, swizzle=sdOt_layout.inner, dtype=self.do_dtype
        )

        sLSE = storage.sLSE.get_tensor(sLSE_layout)
        sdPsum = storage.sdPsum.get_tensor(sdPsum_layout)
        sdV = storage.sdO.get_tensor(
            sdV_layout.outer, swizzle=sdV_layout.inner, dtype=self.dv_dtype
        )
        sdK = storage.sQ.get_tensor(
            sdK_layout.outer, swizzle=sdK_layout.inner, dtype=self.dk_dtype
        )

        # Buffer sizing is guaranteed by max(...) in SharedStorage declarations
        # for both sQ (reused as sdK) and sdO (reused as sdV)
        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)

        # TMEM
        # This is a fake tensor, by right need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
        # S
        thr_mma_S = tiled_mma_S.get_slice(mma_tile_coord_v)
        Sacc_shape = thr_mma_S.partition_shape_C(self.mma_tiler_kq[:2])  # (M, N)
        tStS = thr_mma_S.make_fragment_C(Sacc_shape)
        # (MMA, MMA_M, MMA_N)
        tStS = cute.make_tensor(tmem_ptr + self.tmem_S_offset, tStS.layout)
        # dP
        thr_mma_dP = tiled_mma_dP.get_slice(mma_tile_coord_v)
        dPacc_shape = thr_mma_dP.partition_shape_C(self.mma_tiler_vdo[:2])
        tdPtdP = thr_mma_dP.make_fragment_C(dPacc_shape)
        tdPtdP = cute.make_tensor(tmem_ptr + self.tmem_dP_offset, tdPtdP.layout)
        # dV
        thr_mma_dV = tiled_mma_dV.get_slice(mma_tile_coord_v)
        dvacc_shape = thr_mma_dV.partition_shape_C(self.mma_tiler_pdo[:2])
        tdVtdV = thr_mma_dV.make_fragment_C(dvacc_shape)
        tdVtdV = cute.make_tensor(tmem_ptr + self.tmem_dV_offset, tdVtdV.layout)
        tP = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_P_offset, dtype=self.do_dtype), tP_layout.outer
        )
        # dK
        thr_mma_dK = tiled_mma_dK.get_slice(mma_tile_coord_v)
        dkacc_shape = thr_mma_dK.partition_shape_C(self.mma_tiler_dsq[:2])
        tdKtdK = thr_mma_dK.make_fragment_C(dkacc_shape)
        tdKtdK = cute.make_tensor(tmem_ptr + self.tmem_dK_offset, tdKtdK.layout)
        tdS = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_dS_offset, dtype=self.ds_dtype), tdS_layout.outer
        )
        # dQ
        thr_mma_dQ = tiled_mma_dQ.get_slice(mma_tile_coord_v)
        dQacc_shape = thr_mma_dQ.partition_shape_C(self.mma_tiler_dsk[:2])
        tdQtdQ = thr_mma_dQ.make_fragment_C(dQacc_shape)
        tdQtdQ = cute.make_tensor(tmem_ptr + self.tmem_dQ_offset, tdQtdQ.layout)

        block_info = BlockInfo(self.tile_m, self.tile_n)
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
        )
        TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)

        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            swap_AB=True,
        )
        #  EMPTY (warps 14 and 15 are idle)
        if warp_idx == self.empty_warp_id or warp_idx == 14:
            cute.arch.setmaxregister_decrease(self.num_regs_empty)

        #  LOAD
        # (13)
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            self.load(
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                thr_mma_dK,
                thr_mma_dQ,
                mQ,
                mK,
                mV,
                mdO,
                mLSE,
                mdPsum,
                sQ,
                sK,
                sV,
                sdO,
                sLSE,
                sdPsum,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_dO,
                pipeline_Q,
                pipeline_dO,
                pipeline_LSE,
                pipeline_dPsum,
                cluster_layout_vmnk,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
            )

        #  MMA
        # (12)
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)

            # Alloc tmem buffer
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)

            self.mma(
                tiled_mma_S,
                tiled_mma_dP,
                tiled_mma_dV,
                tiled_mma_dK,
                tiled_mma_dQ,
                sQ,
                sK,
                sV,
                sdO,
                tP,
                sdSt,
                sdS,
                tdS,
                tStS,
                tdPtdP,
                tdVtdV,
                tdKtdK,
                tdQtdQ,
                pipeline_Q,
                pipeline_dO,
                pipeline_S_P,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                pipeline_dQ,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                sQt=sQt,
                sKt=sKt,
                sdOt=sdOt,
            )
            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        # Compute
        # (4, 5, 6, 7, 8, 9, 10, 11) --> 8 warps
        if warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_compute)  # 8 warps
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            self.compute_loop(
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                thr_mma_dK,
                tStS,
                tdPtdP,
                tdVtdV,
                tdKtdK,
                sLSE,
                sdPsum,
                mdV,
                mdK,
                sdS,
                pipeline_LSE,
                pipeline_dPsum,
                pipeline_S_P,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                softmax_scale,
                softmax_scale_log2,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                sdV,
                sdK,
                mdV_tma_tensor,
                mdK_tma_tensor,
                tma_atom_dV,
                tma_atom_dK,
                tiled_copy_r2s_dKV,
            )
            tmem_alloc_barrier.arrive()

        # Reduce
        # (0, 1, 2, 3) - dQ
        if warp_idx >= self.reduce_warp_ids[0] and warp_idx <= self.reduce_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_reduce)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            self.dQacc_reduce(
                mdQaccum,
                sdQaccum,
                thr_mma_dQ,
                tdQtdQ,
                pipeline_dQ,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
            )
            tmem_alloc_barrier.arrive()

        return

    @cute.jit
    def load(
        self,
        thr_mma_S: cute.core.ThrMma,
        thr_mma_dP: cute.core.ThrMma,
        thr_mma_dV: cute.core.ThrMma,
        thr_mma_dK: cute.core.ThrMma,
        thr_mma_dQ: cute.core.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        pipeline_Q: PipelineAsync,
        pipeline_dO: PipelineAsync,
        pipeline_LSE: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        cluster_layout_vmnk: cute.Layout,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        producer_state_Q_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_dO_dPsum = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
        )

        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(0)
        q_do_mcast_mask = None

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)

            # GMEM tensors (dense, 1-CTA)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx]
            mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx]
            mdO_cur = mdO[None, None, head_idx, batch_idx]
            mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2)[None, head_idx]
            mdPsum_cur = seqlen.offset_batch_Q(mdPsum, batch_idx, dim=2)[
                None, head_idx
            ]

            # (1) S.T = K @ Q.T
            gK = cute.local_tile(
                mK_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (n_block, 0)
            )
            tSgK = thr_mma_S.partition_A(gK)

            gQ = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_kq, mode=[1, 2]), (None, 0))
            tSgQ = thr_mma_S.partition_B(gQ)
            gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
            gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))
            gdO = cute.local_tile(mdO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None))

            a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            load_K, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_K,
                block_in_cluster_coord_vmnk[2],
                a_cta_layout,
                tSgK,
                sK,
                single_stage=True,
            )

            b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
            load_Q, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q,
                cta_coord=block_in_cluster_coord_vmnk[1],
                cta_layout=b_cta_layout,
                src_tensor=tSgQ,
                dst_tensor=sQ,
                mcast_mask=q_do_mcast_mask,
            )
            load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)

            # (2) dP = V @ dO.T
            gV = cute.local_tile(
                mV_cur, cute.select(self.mma_tiler_vdo, mode=[0, 2]), (n_block, 0)
            )
            tdPgV = thr_mma_dP.partition_A(gV)

            load_V, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_V,
                0,
                cute.make_layout(1),
                tdPgV,
                sV,
                single_stage=True,
            )

            # (3) dV += P.T @ dO
            gdO = cute.local_tile(mdO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None))
            tdVgdO = thr_mma_dV.partition_B(gdO)
            load_dO, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dO,
                cta_coord=block_in_cluster_coord_vmnk[1],
                cta_layout=b_cta_layout,
                src_tensor=tdVgdO,
                dst_tensor=sdO,
                mcast_mask=q_do_mcast_mask,
            )
            load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)

            copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
            copy_stats = partial(cute.copy, copy_atom_stats)

            #### Prologue ####
            first_m_block = m_block_min
            # K & Q (for S)
            pipeline_Q.producer_acquire(
                producer_state_Q_LSE, extra_tx_count=self.tma_copy_bytes["K"]
            )
            load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_LSE))
            load_Q(first_m_block, producer_state=producer_state_Q_LSE)
            pipeline_Q.producer_commit(producer_state_Q_LSE)

            # LSE
            pipeline_LSE.producer_acquire(producer_state_Q_LSE)
            with cute.arch.elect_one():
                copy_stats(
                    gLSE[None, first_m_block],
                    sLSE[None, producer_state_Q_LSE.index],
                    mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                )
            producer_state_Q_LSE.advance()

            # V + dO
            pipeline_dO.producer_acquire(
                producer_state_dO_dPsum,
                extra_tx_count=self.tma_copy_bytes["V"],
            )
            load_V(
                tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_dPsum)
            )
            load_dO(first_m_block, producer_state=producer_state_dO_dPsum)
            pipeline_dO.producer_commit(producer_state_dO_dPsum)

            # dPsum
            pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
            with cute.arch.elect_one():
                copy_stats(
                    gdPsum[None, first_m_block],
                    sdPsum[None, producer_state_dO_dPsum.index],
                    mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO_dPsum),
                )
            producer_state_dO_dPsum.advance()

            #### Main Loop ####
            for m_block in cutlass.range(m_block_min + 1, m_block_max, unroll=1):
                # Q (for S)
                pipeline_Q.producer_acquire(producer_state_Q_LSE)
                load_Q(m_block, producer_state=producer_state_Q_LSE)
                pipeline_Q.producer_commit(producer_state_Q_LSE)

                # LSE
                pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                with cute.arch.elect_one():
                    copy_stats(
                        gLSE[None, m_block],
                        sLSE[None, producer_state_Q_LSE.index],
                        mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                    )
                producer_state_Q_LSE.advance()

                # dO
                pipeline_dO.producer_acquire(producer_state_dO_dPsum)
                load_dO(m_block, producer_state=producer_state_dO_dPsum)
                pipeline_dO.producer_commit(producer_state_dO_dPsum)

                # dPsum
                pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                with cute.arch.elect_one():
                    copy_stats(
                        gdPsum[None, m_block],
                        sdPsum[None, producer_state_dO_dPsum.index],
                        mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO_dPsum),
                    )
                producer_state_dO_dPsum.advance()

            #### Tail ####
            pipeline_Q.producer_tail(producer_state_Q_LSE.clone())
            pipeline_LSE.producer_tail(producer_state_Q_LSE)
            pipeline_dO.producer_tail(producer_state_dO_dPsum.clone())
            pipeline_dPsum.producer_tail(producer_state_dO_dPsum)

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma(
        self,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        tP: cute.Tensor,
        sdSt: cute.Tensor,
        sdS: cute.Tensor,
        tdS: cute.Tensor,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdQtdQ: cute.Tensor,
        pipeline_Q: PipelineAsync,
        pipeline_dO: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_dS: PipelineAsync,
        pipeline_dKV: PipelineAsync,
        pipeline_dP: PipelineAsync,
        pipeline_dQ: PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        sQt: cute.Tensor = None,
        sKt: cute.Tensor = None,
        sdOt: cute.Tensor = None,
    ):
        # [2025-10-21] For reasons I don't understand, putting these partitioning in the main
        # kernel (before warp specialization) is a lot slower tha putting them here.
        # Partition smem / tmem tensors
        # S = K @ Q.T
        tSrK = tiled_mma_S.make_fragment_A(sK)
        tSrQ = tiled_mma_S.make_fragment_B(sQ)
        # dP = V @ dOt.T
        tdPrV = tiled_mma_dP.make_fragment_A(sV)
        tdPrdOt = tiled_mma_dP.make_fragment_B(sdOt)
        # dK = dS.T @ Q
        tdKrdS = tiled_mma_dK.make_fragment_A(tdS)  # From TMEM

        tdKrQ = tiled_mma_dK.make_fragment_B(sQt)
        # dQ = dS @ K
        tdQrdS = tiled_mma_dQ.make_fragment_A(sdS)
        tdQrK = tiled_mma_dQ.make_fragment_B(sKt)
        # dV = P @ dO.T
        tdVrdO = tiled_mma_dV.make_fragment_B(sdO)
        tdVrP = tiled_mma_dV.make_fragment_A(tP)

        # mma_qk_fn = partial(gemm_w_idx, tiled_mma_S, tStS, tSrK, tSrQ, zero_init=True)
        mma_qk_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_S,
            tStS,
            tSrK,
            tSrQ,
            sA=sK,
            sB=sQ,
            zero_init=True,
            cta_group=1,
        )
        # mma_dov_fn = partial(gemm_w_idx, tiled_mma_dP, tdPtdP, tdPrV, tdPrdOt, zero_init=True)
        mma_dov_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_dP,
            tdPtdP,
            tdPrV,
            tdPrdOt,
            sA=sV,
            sB=sdOt,
            zero_init=True,
            cta_group=1,
        )
        # mma_pdo_fn = partial(gemm_w_idx, tiled_mma_dV, tdVtdV, tdVrP, tdVrdO)
        mma_pdo_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_dV,
            tdVtdV,
            tdVrP,
            tdVrdO,
            sA=None,
            sB=sdO,
            tA_addr=self.tmem_P_offset,
            cta_group=1,
        )
        num_unroll_groups = 1
        mma_dsk_fn = partial(
            gemm_w_idx,
            tiled_mma_dQ,
            tdQtdQ,
            tdQrdS,
            tdQrK,
            zero_init=True,
            num_unroll_groups=num_unroll_groups,
        )
        # mma_dsk_fn = partial(
        #     gemm_ptx_w_idx, tiled_mma_dQ, tdQtdQ, tdQrdS, tdQrK, sA=sdS, sB=sKt, zero_init=True
        # )
        # Need to explicitly pass in tA_addr for correctness
        mma_dsq_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_dK,
            tdKtdK,
            tdKrdS,
            tdKrQ,
            sA=None,
            sB=sQt,
            tA_addr=self.tmem_dS_offset,
            cta_group=1,
        )

        pipeline_Q_consumer = pipeline_Q.make_consumer()

        consumer_state_dO = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )
        producer_phase_acc = Int32(1)  # For S & P, dP, dQ
        consumer_state_dS = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        producer_phase_dKV = Int32(1)
        cta_group = pipeline_S_P.cta_group

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)  # must be seqlen_k
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)

            accumulate_dK = False
            # -----------------------------------------------------------
            ###### Prologue
            # -----------------------------------------------------------
            # 1. S  = Q0 @ K.T
            # 2. dP = V @ dOt.T
            # 3. dV = P @ dO

            # 1) S = K @ Q
            handle_Q = pipeline_Q_consumer.wait_and_advance()
            pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
            mma_qk_fn(B_idx=handle_Q.index)
            pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)

            # 2) dP = V @ dOt.T
            pipeline_dO.consumer_wait(consumer_state_dO)
            pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)
            pipeline_dQ.sync_object_empty.wait(0, producer_phase_acc)
            mma_dov_fn(B_idx=consumer_state_dO.index)
            pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

            producer_phase_acc ^= 1
            # 3) dV = P.T @ dO
            pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
            mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=True)
            pipeline_dO.consumer_release(consumer_state_dO)
            consumer_state_dO.advance()

            # -----------------------------------------------------------
            ###### MAIN LOOP
            # -----------------------------------------------------------
            # 1. S  = K    @ Q.T
            # 2. dQ = dS   @ K
            # 3. dK = dS.T @ Q
            # 4. dP = V    @ dOt.T
            # 5. dV = P.T  @ dO

            main_loop_iters = m_block_max - m_block_min - 1

            handle_Q_next = handle_Q
            for _ in cutlass.range(main_loop_iters, unroll=1):
                # (1) S.T = K @ Q.T
                handle_Q_next = pipeline_Q_consumer.wait_and_advance()
                mma_qk_fn(B_idx=handle_Q_next.index)
                pipeline_S_P.sync_object_full.arrive(
                    0, pipeline_S_P.producer_mask, cta_group
                )

                # (2) dK += dS.T @ Q
                pipeline_dS.consumer_wait(consumer_state_dS)
                mma_dsq_fn(B_idx=handle_Q.index, zero_init=not accumulate_dK)
                accumulate_dK = True
                handle_Q.release()

                # (3) dQ = dS @ K
                mma_dsk_fn()
                pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                pipeline_dS.consumer_release(consumer_state_dS)
                consumer_state_dS.advance()

                # (4) dP = V @ dO.T
                pipeline_dO.consumer_wait(consumer_state_dO)
                pipeline_dQ.sync_object_empty.wait(0, producer_phase_acc)
                mma_dov_fn(B_idx=consumer_state_dO.index)
                pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

                # (5) dV += P.T @ dO
                producer_phase_acc ^= 1
                pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=False)
                pipeline_dO.consumer_release(consumer_state_dO)
                consumer_state_dO.advance()

                handle_Q = handle_Q_next

            pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)

            # signal to the epilogue that dV is ready
            pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
            pipeline_dKV.sync_object_full.arrive(0, pipeline_dKV.producer_mask, cta_group)
            pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)

            # -----------------------------------------------------------
            # Tail: Remaining dK and dQ
            # -----------------------------------------------------------
            # 1) dK += dS.T @ Q
            pipeline_dS.consumer_wait(consumer_state_dS)
            mma_dsq_fn(B_idx=handle_Q.index, zero_init=not accumulate_dK)
            # signal to the epilogue that dK is ready
            pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
            producer_phase_dKV ^= 1

            # 2) dQ = dS @ K
            mma_dsk_fn()
            pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
            handle_Q.release()
            pipeline_dS.consumer_release(consumer_state_dS)
            consumer_state_dS.advance()

            producer_phase_acc ^= 1
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # Currently it hangs if we have this S_P.producer_tail, will need to understand why
        # pipeline_S_P.producer_tail(producer_state_S_P)
        # pipeline_dP.producer_tail(producer_state_dP)
        # pipeline_dKV.producer_tail(producer_state_dKV)
        # pipeline_dQ.producer_tail(producer_state_dQ)

    @cute.jit
    def split_wg(
        self,
        t: cute.Tensor,
        wg_idx: cutlass.Int32,
        num_wg: cutlass.Constexpr[int],
    ):
        reduced_shape = cute.product_each(t.shape)
        rank = len(reduced_shape)
        if const_expr(reduced_shape[1] > 1):
            assert rank >= 2, "Need rank >= 2 for t in split_wg"
            t = cute.logical_divide(t, (reduced_shape[0], reduced_shape[1] // num_wg))
            coord = (None, (None, wg_idx)) + (None,) * (rank - 2)
        else:
            assert rank >= 3, "Need rank >= 3 for t in split_wg"
            if const_expr(rank == 3):
                t = cute.logical_divide(
                    t, (reduced_shape[0], reduced_shape[1], reduced_shape[2] // num_wg)
                )
                coord = (
                    None,
                    None,
                    (None, wg_idx),
                ) + (None,) * (rank - 3)
            else:
                t = cute.logical_divide(
                    t,
                    (
                        reduced_shape[0],
                        reduced_shape[1],
                        reduced_shape[2],
                        reduced_shape[3] // num_wg,
                    ),
                )
                coord = (
                    None,
                    None,
                    None,
                    (None, wg_idx),
                ) + (None,) * (rank - 4)
        return t[coord]

    @cute.jit
    def compute_loop(
        self,
        thr_mma_S: cute.core.ThrMma,
        thr_mma_dP: cute.core.ThrMma,
        thr_mma_dV: cute.core.ThrMma,
        thr_mma_dK: cute.core.ThrMma,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        sdS: cute.Tensor,
        pipeline_LSE: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_dS: PipelineAsync,
        pipeline_dKV: PipelineAsync,
        pipeline_dP: PipelineAsync,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        sdV: Optional[cute.Tensor] = None,
        sdK: Optional[cute.Tensor] = None,
        mdV_tma_tensor: Optional[cute.Tensor] = None,
        mdK_tma_tensor: Optional[cute.Tensor] = None,
        tma_atom_dV: Optional[cute.CopyAtom] = None,
        tma_atom_dK: Optional[cute.CopyAtom] = None,
        tiled_copy_r2s_dKV: Optional[cute.TiledCopy] = None,
    ):
        sLSE_2D = cute.make_tensor(
            sLSE.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.Q_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        sdPsum_2D = cute.make_tensor(
            sdPsum.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.dO_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        # SdP_swapAB is always True in this build.
        sLSE_2D = layout_utils.transpose_view(sLSE_2D)
        sdPsum_2D = layout_utils.transpose_view(sdPsum_2D)

        # tix: [128...384]  8 warps
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())  # 4-11
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))
        # tidx = cute.arch.thread_idx()[0] - (cute.arch.WARP_SIZE * self.compute_warp_ids[0])
        dp_idx = tidx % 128
        num_wg = len(self.compute_warp_ids) // 4  # 2
        # wg_idx:
        # 0: [256...384]
        # 1: [128...256]

        tileP_f32_like = self.cta_tiler[1] // 32 * self.v_dtype.width
        # tStS has shape ((128, 128), 1, 1), tStP has shape ((128, 64), 1, 1)
        # tP overlap with tS
        tStP = cute.composition(tStS, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        tStP = cute.make_tensor(tStS.iterator, tStP.layout)  # Otherwise the tmem address is wrong
        tScS = thr_mma_S.partition_C(cute.make_identity_tensor(self.mma_tiler_kq[:2]))
        tScP = cute.composition(tScS, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        # tdS overlap with tdP
        tdPtdS = cute.composition(tdPtdP, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        tdPcdP = thr_mma_dP.partition_C(cute.make_identity_tensor(self.mma_tiler_vdo[:2]))
        tdPcdS = cute.composition(tdPcdP, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32
        )

        # tmem -> rmem
        thr_copy_t2r = copy_utils.make_tmem_copy(tmem_load_atom, num_wg).get_slice(tidx)
        tStS_t2r = thr_copy_t2r.partition_S(tStS)  # (((32, 32), 1), 2, 1, 1)
        tdPtdP_t2r = thr_copy_t2r.partition_S(tdPtdP)
        tScS_t2r = thr_copy_t2r.partition_D(tScS)  # ((32, 1), 2, 1, 1)
        t0ScS_t2r = thr_copy_t2r.get_slice(0).partition_D(tScS)  # ((32, 1), 2, 1, 1)
        # ((32, 1), 2, 1, 1, STAGE)
        tSsLSE = thr_copy_t2r.partition_D(thr_mma_S.partition_C(sLSE_2D))
        tSsdPsum = thr_copy_t2r.partition_D(thr_mma_dP.partition_C(sdPsum_2D))
        # rmem -> tmem
        thr_copy_r2t = copy_utils.make_tmem_copy(tmem_store_atom, num_wg).get_slice(tidx)
        tScP_r2t = thr_copy_r2t.partition_S(tScP)
        tStP_r2t = thr_copy_r2t.partition_D(tStP)
        tdPcdS_r2t = thr_copy_r2t.partition_S(tdPcdS)
        tdPtdS_r2t = thr_copy_r2t.partition_D(tdPtdS)
        # rmem -> smem
        # This part is a bit iffy, we might be making a lot of assumptions here
        copy_atom_r2s = sm100_utils_basic.get_smem_store_op(
            LayoutEnum.ROW_MAJOR, self.ds_dtype, Float32, thr_copy_t2r
        )
        thr_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, thr_copy_t2r).get_slice(tidx)

        # We assume the swizzle (i.e. layout.inner) stays the same
        sdS_epi_layout = sm100_utils_basic.make_smem_layout_epi(
            self.ds_dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_m), 1
        )
        sdS_layout = cute.slice_(sdS_epi_layout.outer, (None, None, 0))  # ((8,16), (64,2))
        # Need to group into 1 mode to be compatible w thr_copy_r2s
        sdS_layout = cute.make_layout((sdS_layout.shape,), stride=(sdS_layout.stride,))
        sdS_epi = cute.make_tensor(sdS.iterator, sdS_layout)
        tRS_sdS = thr_copy_r2s.partition_D(sdS_epi)

        consumer_state_S_P_dP = pipeline.make_pipeline_state(  # Our impl has shortcut for stage==1
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        # consumer_phase_S_P_dP = Int32(0)
        producer_state_dS = pipeline.make_pipeline_state(  # Our impl has shortcut for stage==1
            cutlass.pipeline.PipelineUserType.Producer, 1
        )
        consumer_state_dKV = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 2
        )
        consumer_state_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_dPsum = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            mask = AttentionMaskCls(seqlen)
            mask_fn = partial(
                mask.apply_mask_sm100_transposed,
                tScS_t2r=tScS_t2r,
                t0ScS_t2r=t0ScS_t2r,
                n_block=n_block,
                mask_seqlen=True,
            )

            loop_count = m_block_max - m_block_min

            # Mainloop
            for iter_idx in cutlass.range(loop_count, unroll=1):
                m_block = m_block_min + iter_idx
                # Prefetch 1 stage of LSE
                pipeline_LSE.consumer_wait(consumer_state_LSE)
                tSrLSE_s2r = cute.make_fragment(tScS_t2r[None, 0, 0, 0].shape, Float32)

                pipeline_S_P.consumer_wait(consumer_state_S_P_dP)
                # pipeline_S_P.sync_object_full.wait(0, consumer_phase_S_P_dP)
                #### TMEM->RMEM (Load S from TMEM)
                tSrS_t2r = cute.make_fragment(tScS_t2r.shape, Float32)
                cute.copy(thr_copy_t2r, tStS_t2r, tSrS_t2r)

                #### APPLY MASK
                mask_fn(tSrS_t2r, m_block=m_block)
                num_stages = cute.size(tScS_t2r, mode=[1])
                # ---------------------------------------------
                #### P = exp(S - LSE)
                # ---------------------------------------------
                lane_idx = cute.arch.lane_idx()
                tSrP_r2t_f32 = cute.make_fragment(tScP_r2t.shape, Float32)  # 64
                tSrP_r2t = cute.recast_tensor(tSrP_r2t_f32, self.q_dtype)
                for stage in cutlass.range_constexpr(num_stages):
                    tSrS_cur = tSrS_t2r[None, stage, 0, 0]
                    tSsLSE_cur = tSsLSE[None, stage, 0, 0, consumer_state_LSE.index]
                    cute.autovec_copy(tSsLSE_cur, tSrLSE_s2r)
                    tSrLSE = tSrLSE_s2r
                    for v in cutlass.range_constexpr(cute.size(tSrS_t2r, mode=[0]) // 2):
                        lse_pair = (tSrLSE[2 * v], tSrLSE[2 * v + 1])
                        tSrS_cur[2 * v], tSrS_cur[2 * v + 1] = cute.arch.fma_packed_f32x2(
                            ((tSrS_cur[2 * v], tSrS_cur[2 * v + 1])),
                            (softmax_scale_log2, softmax_scale_log2),
                            (-lse_pair[0], -lse_pair[1]),
                        )
                        tSrS_cur[2 * v] = cute.math.exp2(tSrS_cur[2 * v], fastmath=True)
                        tSrS_cur[2 * v + 1] = cute.math.exp2(tSrS_cur[2 * v + 1], fastmath=True)
                    # fp32 bwd P-storage TODO: see top-of-__call__ gate.
                    utils.cvt_f16(tSrS_cur, tSrP_r2t[None, stage, 0, 0])
                    if const_expr(stage == 0):
                        cute.arch.fence_view_async_tmem_load()
                        # Without this barrier, we could have 1 warp writing to P in tmem while
                        # another warp is still reading S from tmem.
                        self.compute_sync_barrier.arrive_and_wait()
                    cute.copy(
                        thr_copy_r2t,
                        tSrP_r2t_f32[None, stage, None, None],
                        tStP_r2t[None, stage, None, None],
                    )

                cute.arch.fence_view_async_tmem_store()
                cute.arch.fence_view_async_shared()
                self.compute_sync_barrier.arrive_and_wait()
                # Signal tmem store P completion with pipeline_S_P
                with cute.arch.elect_one():
                    pipeline_S_P.consumer_release(consumer_state_S_P_dP)
                # Normally we'd need syncwarp here since only 1 thread will signal in
                # consumer_release, but we already have the self.compute_sync_barrier before this
                pipeline_LSE.consumer_release(consumer_state_LSE)
                consumer_state_LSE.advance()
                # ---------------------------------------------
                # dS.T = P.T * (dP.T - D)
                # ---------------------------------------------
                pipeline_dPsum.consumer_wait(consumer_state_dPsum)
                pipeline_dP.consumer_wait(consumer_state_S_P_dP)
                # pipeline_dP.sync_object_full.wait(0, consumer_phase_S_P_dP)
                ### Now delayed to after loop
                # consumer_state_S_P_dP.advance()
                # consumer_phase_S_P_dP ^= 1

                ##### dS.T = P.T * (dP.T - Psum)
                for stage in cutlass.range_constexpr(num_stages):
                    tdPrdP_t2r = cute.make_fragment(tScS_t2r[None, 0, None, None].shape, Float32)
                    cute.copy(thr_copy_t2r, tdPtdP_t2r[None, stage, None, None], tdPrdP_t2r)
                    cute.arch.fence_view_async_tmem_load()
                    self.compute_sync_barrier.arrive_and_wait()
                    tdPrdP_cur = tdPrdP_t2r[None, 0, 0]
                    tSrS_cur = tSrS_t2r[None, stage, 0, 0]
                    tSsdPsum_cur = tSsdPsum[None, stage, 0, 0, consumer_state_dPsum.index]
                    tSrdPsum = cute.make_fragment_like(tSsdPsum_cur, Float32)
                    cute.autovec_copy(tSsdPsum_cur, tSrdPsum)
                    for v in cutlass.range_constexpr(cute.size(tdPrdP_t2r, mode=[0]) // 2):
                        dPsum_pair = (tSrdPsum[2 * v], tSrdPsum[2 * v + 1])
                        tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = (
                            quack.activation.sub_packed_f32x2(
                                (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]), dPsum_pair
                            )
                        )
                        tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = cute.arch.mul_packed_f32x2(
                            (tSrS_cur[2 * v], tSrS_cur[2 * v + 1]),
                            (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                        )

                    tdPrdS_cvt = cute.make_fragment_like(tdPrdP_cur, self.ds_dtype)
                    utils.cvt_f16(tdPrdP_cur, tdPrdS_cvt)
                    if const_expr(stage == 0):
                        pipeline_dS.producer_acquire(producer_state_dS)

                    # RMEM->TMEM: always write to TMEM for MMA
                    tdPrdS_r2t_f32 = cute.recast_tensor(tdPrdS_cvt, Float32)
                    cute.copy(thr_copy_r2t, tdPrdS_r2t_f32, tdPtdS_r2t[None, stage, 0, 0])

                    # RMEM->SMEM
                    cute.autovec_copy(tdPrdS_cvt, tRS_sdS[None, stage])

                cute.arch.fence_view_async_tmem_store()

                consumer_state_S_P_dP.advance()

                cute.arch.fence_view_async_shared()
                self.compute_sync_barrier.arrive_and_wait()
                # Normally we'd need syncwarp here since only 1 thread will signal in
                # consumer_release, but we already have the self.compute_sync_barrier before this
                pipeline_dPsum.consumer_release(consumer_state_dPsum)
                consumer_state_dPsum.advance()
                with cute.arch.elect_one():
                    pipeline_dS.producer_commit(producer_state_dS)
                producer_state_dS.advance()

            # Epilogue
            thr_copy_r2s_dKV = tiled_copy_r2s_dKV.get_slice(dp_idx)
            #### STORE dV
            consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                dp_idx,
                batch_idx,
                head_idx,
                n_block,
                seqlen,
                thr_mma_dV,
                tdVtdV,
                mdV_tma_tensor,
                sdV,
                tma_atom_dV,
                thr_copy_r2s_dKV,
                pipeline_dKV,
                consumer_state_dKV,
                None,  # Don't scale
                int(NamedBarrierBwdSm100.EpilogueWG1),  # barrier_id
                None,  # mdV_semaphore (deterministic disabled)
                "V",
            )
            #### STORE dK
            consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                dp_idx,
                batch_idx,
                head_idx,
                n_block,
                seqlen,
                thr_mma_dK,
                tdKtdK,
                mdK_tma_tensor,
                sdK,
                tma_atom_dK,
                thr_copy_r2s_dKV,
                pipeline_dKV,
                consumer_state_dKV,
                softmax_scale,
                int(NamedBarrierBwdSm100.EpilogueWG1),  # barrier_id
                None,  # mdK_semaphore (deterministic disabled)
                "K",
            )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def dQacc_reduce(
        self,
        mdQaccum: cute.Tensor,
        sdQaccum: cute.Tensor,
        thr_mma_dQ: cute.core.ThrMma,
        tdQtdQ: cute.Tensor,
        pipeline_dQ: PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        num_reduce_threads = cute.arch.WARP_SIZE * len(self.reduce_warp_ids)
        tidx = cute.arch.thread_idx()[0] % num_reduce_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx() % len(self.reduce_warp_ids))
        is_tma_warp = warp_idx == 0
        # TMEM -> RMEM
        # fp16/bf16: flat Ld32x32bOp covers the full reduce-ncol width at once.
        # fp32: TF32 MMA splits the accumulator into (inner 16, outer stacks)
        # along the N (head_dim) axis, so a flat 32-col atom does not match.
        # We pick the atom via get_tmem_load_op for a 16-col sub-tile (same
        # pattern the forward epilogue uses) and loop over the chunks.
        fp32_dQ = self.q_dtype is Float32
        if const_expr(fp32_dQ):
            # Chunked view of the dQ accumulator: inner (tile_m, dQ_reduce_ncol),
            # outer = (1, num_chunks) where num_chunks * dQ_reduce_ncol == tile_hdim.
            tdQtdQ_chunks = cute.logical_divide(
                tdQtdQ,
                cute.make_layout((self.tile_m, self.dQ_reduce_ncol)),
            )
            tmem_load_atom = sm100_utils_basic.get_tmem_load_op(
                self.mma_tiler_dsk,
                LayoutEnum.ROW_MAJOR,
                Float32,  # dst dtype
                Float32,  # acc dtype
                (self.tile_m, self.dQ_reduce_ncol),
                use_2cta_instrs=False,
            )
            tiled_t2r = tcgen05.make_tmem_copy(
                tmem_load_atom, tdQtdQ_chunks[(None, None), 0]
            )
            thr_copy_t2r = tiled_t2r.get_slice(tidx)
            # Partition the full chunked tensor; outer mode indexes the chunks.
            tdQtdQ_t2r = thr_copy_t2r.partition_S(
                tdQtdQ_chunks[(None, None), None]
            )
        else:
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dQ_reduce_ncol_t2r)), Float32
            )
            thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ).get_slice(tidx)
            tdQtdQ_t2r = thr_copy_t2r.partition_S(tdQtdQ)
        tdQcdQ = thr_mma_dQ.partition_C(cute.make_identity_tensor(self.mma_tiler_dsk[:2]))
        if const_expr(fp32_dQ):
            # Use the same chunked partition so partition_D matches partition_S.
            tdQcdQ_chunks = cute.logical_divide(
                tdQcdQ, cute.make_layout((self.tile_m, self.dQ_reduce_ncol))
            )
            tdQrdQ_t2r_shape = thr_copy_t2r.partition_D(
                tdQcdQ_chunks[(None, None), None]
            ).shape
        else:
            tdQrdQ_t2r_shape = thr_copy_t2r.partition_D(tdQcdQ).shape
        expected_reduce_stages_t2r = self.dQaccum_reduce_stage_t2r // 1
        # fp32 path: tdQrdQ_t2r_shape is (per_thread, 1, 1, num_chunks).
        # fp16/bf16 path: (per_thread, num_stages).
        _mode_check = 3 if fp32_dQ else 1
        assert cute.size(tdQrdQ_t2r_shape, mode=[_mode_check]) == expected_reduce_stages_t2r, (
            f"dQaccum t2r reduce stage mismatch: got shape {tdQrdQ_t2r_shape}, "
            f"expected stages {expected_reduce_stages_t2r}"
        )

        thr_copy_dQaccum_r2s = copy_utils.tiled_copy_1d(
            self.dqaccum_dtype, num_reduce_threads, num_copy_elems=128 // self.dqaccum_dtype.width
        ).get_slice(tidx)
        tdQsdQ = thr_copy_dQaccum_r2s.partition_D(sdQaccum)

        read_flag = const_expr(True)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        dQ_consumer_state = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        dQ_tma_store_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.sdQaccum_stage
        )
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            mdQaccum_cur = mdQaccum[None, head_idx, batch_idx]
            gdQaccum_ = cute.local_tile(mdQaccum_cur, (self.tile_m * self.tile_hdim,), (None,))
            # (M * K / STAGE, STAGE, _)
            gdQaccum = cute.flat_divide(
                gdQaccum_, (self.tile_m * self.tile_hdim // self.dQaccum_reduce_stage,)
            )

            loop_count = m_block_max - m_block_min

            # dQacc_reduce mainloop
            for iter_idx in cutlass.range(loop_count, unroll=1):
                m_block = m_block_min + iter_idx
                pipeline_dQ.consumer_wait(dQ_consumer_state)
                # TMEM -> RMEM
                tdQrdQ_t2r = cute.make_fragment(tdQrdQ_t2r_shape, Float32)
                if const_expr(fp32_dQ):
                    # TF32 accumulator is chunked along N; copy each chunk.
                    num_chunks = self.tile_hdim // self.dQ_reduce_ncol
                    for chunk_idx in cutlass.range_constexpr(num_chunks):
                        cute.copy(
                            thr_copy_t2r,
                            tdQtdQ_t2r[None, 0, 0, chunk_idx],
                            tdQrdQ_t2r[None, 0, 0, chunk_idx],
                        )
                else:
                    cute.copy(thr_copy_t2r, tdQtdQ_t2r, tdQrdQ_t2r)
                cute.arch.fence_view_async_tmem_load()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    pipeline_dQ.consumer_release(dQ_consumer_state)
                dQ_consumer_state.advance()

                gdQaccum_cur = gdQaccum[None, None, m_block]

                tdQrdQ_shape = (
                    self.dQ_reduce_ncol,
                    self.tile_hdim // 1 // self.dQ_reduce_ncol,
                )
                tdQrdQ = cute.make_tensor(tdQrdQ_t2r.iterator, tdQrdQ_shape)

                for stage in cutlass.range_constexpr(cute.size(tdQrdQ, mode=[1])):
                    smem_idx = dQ_tma_store_producer_state.index
                    tdQsdQ_r2s = tdQsdQ[None, None, smem_idx]
                    tdQrdQ_r2s = cute.make_tensor(tdQrdQ[None, stage].iterator, tdQsdQ_r2s.shape)
                    cute.copy(thr_copy_dQaccum_r2s, tdQrdQ_r2s, tdQsdQ_r2s)
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_view_async_shared()
                    self.reduce_sync_barrier.arrive_and_wait()
                    # Copy from shared memory to global memory
                    if is_tma_warp:
                        with cute.arch.elect_one():
                            copy_utils.cpasync_reduce_bulk_add_f32(
                                sdQaccum[None, smem_idx].iterator,
                                gdQaccum_cur[None, stage].iterator,
                                self.tma_copy_bytes["dQ"] // 1,
                            )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(self.sdQaccum_stage - 1, read=read_flag)
                    self.reduce_sync_barrier.arrive_and_wait()
                    dQ_tma_store_producer_state.advance()

            if is_tma_warp:
                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
            self.reduce_sync_barrier.arrive_and_wait()

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def epilogue_dK_or_dV_tma(
        self,
        tidx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        seqlen,
        thr_mma: cute.core.ThrMma,
        tdKVtdKV: cute.Tensor,
        mdKV: cute.Tensor,
        sdKV: cute.Tensor,
        tma_atom_dKV: cute.CopyAtom,
        thr_copy_r2s_dKV: cute.TiledCopy,
        pipeline_dKV: PipelineAsync,
        consumer_state_dKV: cutlass.pipeline.PipelineState,
        scale: Optional[Float32],
        barrier_id: Int32,
        mdKV_semaphore: Optional[cute.Tensor],
        K_or_V: cutlass.Constexpr[str],
    ) -> cutlass.pipeline.PipelineState:
        assert K_or_V in ("K", "V")
        tile_hdim = self.tile_hdim if const_expr(K_or_V == "K") else self.tile_hdimv
        dtype = self.dk_dtype if const_expr(K_or_V == "K") else self.dv_dtype
        epi_tile = self.sdK_epi_tile if const_expr(K_or_V == "K") else self.sdV_epi_tile
        flat_epi_tile = (
            self.sdK_flat_epi_tile if const_expr(K_or_V == "K") else self.sdV_flat_epi_tile
        )
        num_compute_threads = cute.arch.WARP_SIZE * len(self.compute_warp_ids)
        wg_idx = (cute.arch.thread_idx()[0] % num_compute_threads) // 128
        num_wg = num_compute_threads // 128
        leader_warp = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) == 0

        cta_group_tile_n = const_expr(self.tile_n * 1)

        sdKV = sdKV[None, None, wg_idx]  # (tile_n, 64) for bf16

        # (8, tile_n / 128, 64 / 8) = (8, 1, 8) or (4, tile_n * 32 / (128 * 4)) = (4, 8)
        tdKVsdKV_r2s = thr_copy_r2s_dKV.partition_D(sdKV)

        head_idx_kv = head_idx
        mdKV_cur = mdKV[None, None, head_idx_kv, batch_idx]  # (seqlen, hdim)
        gdKV_p = cute.local_tile(
            mdKV_cur, (self.tile_n, tile_hdim), (n_block, 0)
        )  # (tile_n, hdim) - per CTA
        gdKV = self.split_wg(gdKV_p, wg_idx, num_wg)  # (tile_n, hdim / 2)
        gdKV_epi = cute.local_tile(
            gdKV, epi_tile, (0, None)
        )  # (tile_n, 64, epi_stage = (hdim / 2) / 64)

        tdKVsdKV, tdKVgdKV = cpasync.tma_partition(
            tma_atom_dKV,
            0,  # no multicast
            cute.make_layout(1),
            cute.group_modes(sdKV, 0, 2),
            cute.group_modes(gdKV_epi, 0, 2),
        )  # (TMA) and (TMA, EPI_STAGE)
        assert len(tdKVsdKV.shape) == 1, "Wrong rank for SMEM fragment tdKVsdKV"
        assert len(tdKVgdKV.shape) == 2, "Wrong rank for GMEM fragment tdKVgdKV"
        num_epi_stages = cute.size(tdKVgdKV.shape[1])
        if const_expr(K_or_V == "K"):
            assert num_epi_stages == self.num_epi_stages, "Epi stage calculation is wrong (K)"
        else:
            assert num_epi_stages == self.num_epi_stages_v, "Epi stage calculation is wrong (V)"

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dK_reduce_ncol)), Float32
        )

        read_flag = const_expr(True)

        pipeline_dKV.consumer_wait(consumer_state_dKV)

        for epi_stage in cutlass.range_constexpr(num_epi_stages):
            # TMEM -> RMEM -- setup
            thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV).get_slice(tidx)
            tdKVtdKV_t2r_p = thr_copy_t2r.partition_S(tdKVtdKV)
            tdKVtdKV_t2r = self.split_wg(tdKVtdKV_t2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVtdKV_t2r = tdKVtdKV_t2r[None, epi_stage]

            cdKV = cute.make_identity_tensor((cta_group_tile_n, tile_hdim))
            tdKVcdKV = thr_mma.partition_C(cdKV)
            tdKVcdKV_t2r_p = thr_copy_t2r.partition_D(tdKVcdKV)
            tdKVcdKV_t2r = self.split_wg(tdKVcdKV_t2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVcdKV_t2r = tdKVcdKV_t2r[None, epi_stage]

            tdKVrdKV_t2r = cute.make_fragment(tdKVcdKV_t2r.shape, Float32)

            assert cute.size(tdKVrdKV_t2r) == cute.size(tdKVtdKV_t2r) // cute.arch.WARP_SIZE, (
                "RMEM<->TMEM fragment size mismatch"
            )

            # TMEM -> RMEM -- copy and fence
            cute.copy(thr_copy_t2r, tdKVtdKV_t2r, tdKVrdKV_t2r)
            cute.arch.fence_view_async_tmem_load()

            # RMEM -- scale and convert
            if const_expr(scale is not None):
                for i in cutlass.range(cute.size(tdKVrdKV_t2r.shape) // 2, unroll_full=True):
                    tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1] = cute.arch.mul_packed_f32x2(
                        (tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1]), (scale, scale)
                    )
            tdKVrdKV = cute.make_fragment(tdKVrdKV_t2r.shape, dtype)  # (32 columns)
            tdKVrdKV.store(tdKVrdKV_t2r.load().to(dtype))

            # RMEM -> SMEM -- copy, fence and barrier
            tdKVrdKV_r2s = cute.make_tensor(tdKVrdKV.iterator, tdKVsdKV_r2s.shape)
            cute.copy(thr_copy_r2s_dKV, tdKVrdKV_r2s, tdKVsdKV_r2s)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)

            # SMEM -> GMEM
            if leader_warp:
                cute.copy(tma_atom_dKV, tdKVsdKV, tdKVgdKV[None, epi_stage])
                if const_expr(epi_stage < num_epi_stages - 1):
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                cute.arch.barrier_arrive(
                    barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE
                )

            # Barrier since all warps need to wait for SMEM to be freed
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE
            )

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()
        return consumer_state_dKV
