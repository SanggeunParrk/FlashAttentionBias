# Copyright (c) 2025, Tri Dao.

from enum import IntEnum, auto
from typing import Optional, Tuple, Protocol, runtime_checkable
from dataclasses import dataclass

try:
    from typing import override
except ImportError:  # Python < 3.12
    from typing_extensions import override

import cutlass
from cutlass.pipeline import PipelineClcFetchAsync, PipelineState
from cutlass._mlir import ir
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute import FastDivmodDivisor
from cutlass.utils import ClcDynamicPersistentTileScheduler, ClcDynamicPersistentTileSchedulerParams

from quack.cute_dsl_utils import ParamsBase

from FA4_fp32.core import utils as utils
from FA4_fp32.core.fast_math import clz


class SchedulingMode(IntEnum):
    NONE = auto()
    STATIC = auto()
    DYNAMIC = auto()
    CLC = auto()


@dataclass
class ClcState(ParamsBase):
    """Owns the runtime state shared by CLC-capable tile schedulers.

    `FlashAttentionForwardSm100` constructs this state because it owns the CLC
    response buffer, mbarrier storage, and launch geometry needed to initialize
    the hardware scheduler and async pipeline. Individual tile schedulers then
    consume this state and map the returned hardware work tiles into their own
    logical `WorkTileInfo` coordinates.

    To add CLC support to a scheduler:
    - implement `clc_problem_shape(params)` so the kernel can create the hardware scheduler
    - accept `clc: ClcState | None` in `create(...)` / `__init__`
    - map `clc.initial_work_tile_info()` and `clc.get_current_work()` into scheduler coordinates
    """

    _hw_scheduler: ClcDynamicPersistentTileScheduler
    _pipeline: PipelineClcFetchAsync
    _consumer_state: PipelineState
    _producer_state: PipelineState

    @staticmethod
    def create(
        *,
        hw_scheduler: ClcDynamicPersistentTileScheduler,
        pipeline: PipelineClcFetchAsync,
        consumer_state: PipelineState,
        producer_state: PipelineState,
    ) -> "ClcState":
        return ClcState(hw_scheduler, pipeline, consumer_state, producer_state)

    def initial_work_tile_info(self):
        return self._hw_scheduler.initial_work_tile_info()

    def get_current_work(self):
        return self._hw_scheduler.get_current_work()

    def prefetch_next_work(self, *, loc=None, ip=None):
        self._pipeline.producer_acquire(self._producer_state, loc=loc, ip=ip)
        mbarrier_addr = self._pipeline.producer_get_barrier(self._producer_state, loc=loc, ip=ip)
        self._hw_scheduler.advance_to_next_work(mbarrier_addr, loc=loc, ip=ip)
        self._producer_state.advance(loc=loc, ip=ip)

    def consumer_wait(self, *, loc=None, ip=None):
        self._pipeline.consumer_wait(self._consumer_state, loc=loc, ip=ip)

    def consumer_release(self, *, loc=None, ip=None):
        self._pipeline.consumer_release(self._consumer_state, loc=loc, ip=ip)
        self._consumer_state.advance(loc=loc, ip=ip)

    def producer_tail(self, *, loc=None, ip=None):
        self._pipeline.producer_tail(self._producer_state, loc=loc, ip=ip)


class WorkTileInfo(cutlass.utils.WorkTileInfo):
    """Altered WorkTileInfo which includes four axes: (block, head, batch, split)"""

    @override
    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "WorkTileInfo":
        assert len(values) == 5
        new_tile_idx = cutlass.new_from_mlir_values(self._tile_idx, values[:-1])
        new_is_valid_tile = cutlass.new_from_mlir_values(self._is_valid_tile, [values[-1]])
        return WorkTileInfo(new_tile_idx, new_is_valid_tile)


@runtime_checkable
class TileSchedulerProtocol(Protocol):
    """Protocol defining the interface all tile schedulers must implement.

    Schedulers are responsible for:
    1. Coordinate mapping: linear tile index -> (m_block, head, batch, split)
    2. Work distribution: how to get the next tile (static grid-stride vs CLC dynamic)
    """

    def get_current_work(self) -> WorkTileInfo:
        """Get the current work tile coordinates."""
        ...

    def initial_work_tile_info(self) -> WorkTileInfo:
        """Get the initial work tile for this CTA."""
        ...

    def advance_to_next_work(self, *, loc=None, ip=None):
        """Consumer-side advance: move to next tile and return it.

        For static schedulers: grid-stride increment + get_current_work.
        For CLC schedulers: consumer wait + get_current_work + consumer release + state advance.
        """
        ...

    def prefetch_next_work(self, *, loc=None, ip=None) -> None:
        """Producer-side prefetch of next work tile (no-op for static schedulers).

        For CLC schedulers: producer acquire + issue CLC query + producer state advance.
        Only called by the scheduler warp.
        """
        ...

    def producer_tail(self, *, loc=None, ip=None) -> None:
        """Producer-side cleanup after the last tile.

        No-op for static schedulers. For CLC schedulers: pipeline producer_tail.
        """
        ...


@dataclass
class TileSchedulerArguments(ParamsBase):
    num_block: Int32
    num_head: Int32
    num_batch: Int32
    num_splits: Int32
    seqlen_k: Int32
    headdim: Int32
    headdim_v: Int32
    total_q: Int32
    tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
    cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)
    mCuSeqlensQ: Optional[cute.Tensor] = None
    mSeqUsedQ: Optional[cute.Tensor] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    element_size: cutlass.Constexpr[int] = 2
    is_persistent: cutlass.Constexpr[bool] = False
    lpt: cutlass.Constexpr[bool] = False
    is_split_kv: cutlass.Constexpr[bool] = False
    head_swizzle: cutlass.Constexpr[bool] = False
    use_cluster_idx: cutlass.Constexpr[bool] = False


class SingleTileScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block: Int32
        num_head: Int32
        num_batch: Int32
        num_splits: Int32
        num_splits_divmod: FastDivmodDivisor
        is_split_kv: cutlass.Constexpr[bool] = False
        cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)
        use_cluster_idx: cutlass.Constexpr[bool] = False

        @staticmethod
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "SingleTileScheduler.Params":
            return SingleTileScheduler.Params(
                args.num_block,
                args.num_head,
                args.num_batch,
                args.num_splits,
                FastDivmodDivisor(args.num_splits),
                args.is_split_kv,
                args.cluster_shape_mn,
                args.use_cluster_idx,
            )

    def __init__(self, params: Params, blk_coord: cute.Coord, *, loc=None, ip=None):
        self.params = params
        self._blk_coord = blk_coord
        self._is_first_block = True
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> Params:
        assert scheduling_mode == SchedulingMode.STATIC, (
            f"SingleTileScheduler only supports STATIC, got {scheduling_mode!r}"
        )
        return SingleTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(
        params: Params, clc: ClcState | None = None, *, loc=None, ip=None
    ) -> "SingleTileScheduler":
        if const_expr(cute.size(params.cluster_shape_mn) == 1 or not params.use_cluster_idx):
            blk_coord = cute.arch.block_idx()
        else:
            blk_coord = cute.arch.cluster_idx()
        return SingleTileScheduler(params, blk_coord, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        # TODO: this hard-codes the fact that we only use cluster = (1, 1) or (2, 1)
        assert params.cluster_shape_mn[1] == 1, "Only cluster_shape_mn[1] == 1 is supported"
        if const_expr(params.use_cluster_idx):
            # Grid must have num_block * cluster_m physical blocks so that there are num_block clusters
            grid_x = params.num_block * params.cluster_shape_mn[0]
        else:
            grid_x = cute.round_up(params.num_block, params.cluster_shape_mn[0])
        return (
            grid_x,
            params.num_head * params.num_splits,
            params.num_batch,
        )

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        block_idx, head_idx, batch_idx = self._blk_coord
        if const_expr(self.params.is_split_kv):
            head_idx, split_idx = divmod(head_idx, self.params.num_splits_divmod)
        else:
            split_idx = Int32(0)
        return WorkTileInfo(
            (block_idx, head_idx, batch_idx, split_idx),
            self._is_first_block,
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._is_first_block = False
        return self.get_current_work()

    def producer_tail(self, *, loc=None, ip=None):
        pass

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._blk_coord]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._blk_coord], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return SingleTileScheduler(*(tuple(obj_list)), loc=self._loc)


class StaticPersistentTileScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block_cluster_divmod: FastDivmodDivisor
        num_head_divmod: FastDivmodDivisor
        total_blocks_cluster: Int32
        cluster_shape_m: cutlass.Constexpr[int] = 1

        @staticmethod
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "StaticPersistentTileScheduler.Params":
            num_block_cluster = cute.ceil_div(args.num_block, cute.size(args.cluster_shape_mn))
            total_blocks_cluster = num_block_cluster * args.num_head * args.num_batch
            return StaticPersistentTileScheduler.Params(
                FastDivmodDivisor(num_block_cluster),
                FastDivmodDivisor(args.num_head),
                total_blocks_cluster,
                cluster_shape_m=args.cluster_shape_mn[0],
            )

    def __init__(self, params: Params, tile_idx: Int32, *, loc=None, ip=None):
        self.params = params
        self._tile_idx = tile_idx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> Params:
        assert scheduling_mode == SchedulingMode.STATIC, (
            f"StaticPersistentTileScheduler only supports STATIC, got {scheduling_mode!r}"
        )
        return StaticPersistentTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(
        params: Params, clc: ClcState | None = None, *, loc=None, ip=None
    ) -> "StaticPersistentTileScheduler":
        if const_expr(cute.size(params.cluster_shape_m) == 1):
            tile_idx = cute.arch.block_idx()[0]
        else:
            tile_idx = cute.arch.cluster_idx()[0]
        return StaticPersistentTileScheduler(params, tile_idx, loc=loc, ip=ip)

    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        hardware_info = cutlass.utils.HardwareInfo()
        sm_count = hardware_info.get_device_multiprocessor_count()
        max_ctas = (sm_count // params.cluster_shape_m) * params.cluster_shape_m
        grid_x = cutlass.min(max_ctas, params.total_blocks_cluster * params.cluster_shape_m)
        return (grid_x, Int32(1), Int32(1))

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        hn_idx, block_idx = divmod(self._tile_idx, self.params.num_block_cluster_divmod)
        batch_idx, head_idx = divmod(hn_idx, self.params.num_head_divmod)
        is_valid = self._tile_idx < self.params.total_blocks_cluster
        return WorkTileInfo(
            (Int32(block_idx), Int32(head_idx), Int32(batch_idx), Int32(0)), is_valid
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.cluster_shape_m == 1):
            self._tile_idx += cute.arch.grid_dim()[0]
        else:
            self._tile_idx += cute.arch.cluster_dim()[0]
        return self.get_current_work()

    def producer_tail(self, *, loc=None, ip=None):
        pass

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._tile_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.params, self._tile_idx],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return StaticPersistentTileScheduler(*(tuple(obj_list)), loc=self._loc)


class SingleTileLPTScheduler:
    @dataclass
    class Params(ParamsBase):
        total_blocks: Int32
        num_splits: Int32
        num_block: Int32
        num_head: Int32
        num_batch: Int32
        l2_minor: Int32
        num_head_divmod: FastDivmodDivisor
        l2_minor_divmod: FastDivmodDivisor
        l2_major_divmod: FastDivmodDivisor
        l2_minor_residual_divmod: FastDivmodDivisor
        num_hb_quotient: Int32
        num_splits_divmod: FastDivmodDivisor
        is_split_kv: cutlass.Constexpr[bool] = False
        cluster_shape_m: cutlass.Constexpr[int] = 1
        scheduling_mode: cutlass.Constexpr[SchedulingMode] = SchedulingMode.STATIC
        lpt: cutlass.Constexpr[bool] = True

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments,
            *,
            scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
            loc=None,
            ip=None,
        ) -> "SingleTileLPTScheduler.Params":
            assert scheduling_mode in (SchedulingMode.STATIC, SchedulingMode.CLC), (
                f"Only STATIC and CLC are supported, got {scheduling_mode!r}"
            )
            size_one_kv_head = args.seqlen_k * (args.headdim + args.headdim_v) * args.element_size
            size_one_head = size_one_kv_head
            size_l2 = 50 * 1024 * 1024  # 40 MB for K & V
            # Swizzle is the size of each "section". Round swizzle to a power of 2
            # Need to be careful about the case where only one head will fit
            # swizzle is how many heads can fit in L2
            # Seems faster if swizzle is a power of 2
            log2_floor = lambda n: 31 - clz(n)
            swizzle = 1 if size_l2 < size_one_head else (1 << log2_floor(size_l2 // size_one_head))
            # If we're in the last section (called residual), we don't want to divide by
            # swizzle. Instead we want to divide by the remainder.
            num_hb_quotient = (args.num_head * args.num_batch) // swizzle
            num_hb_remainder = (args.num_head * args.num_batch) % swizzle
            return SingleTileLPTScheduler.Params(
                total_blocks=args.num_block * args.num_head * args.num_batch,
                num_block=args.num_block,
                num_head=args.num_head,
                num_batch=args.num_batch,
                l2_minor=Int32(swizzle),
                num_head_divmod=FastDivmodDivisor(args.num_head),
                l2_minor_divmod=FastDivmodDivisor(swizzle),
                l2_major_divmod=FastDivmodDivisor(swizzle * args.num_block),
                l2_minor_residual_divmod=FastDivmodDivisor(max(num_hb_remainder, 1)),
                num_hb_quotient=Int32(num_hb_quotient),
                num_splits=args.num_splits,
                num_splits_divmod=FastDivmodDivisor(args.num_splits),
                is_split_kv=args.is_split_kv,
                cluster_shape_m=args.cluster_shape_mn[0],
                scheduling_mode=scheduling_mode,
                lpt=args.lpt,
            )

    def __init__(
        self,
        params: Params,
        tile_idx: Int32,
        split_idx: Int32,
        clc: ClcState | None = None,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self._tile_idx = tile_idx
        self._split_idx = split_idx
        self.clc = clc
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> Params:
        return SingleTileLPTScheduler.Params.create(
            args, scheduling_mode=scheduling_mode, loc=loc, ip=ip
        )

    @staticmethod
    def _clc_grid_shape(params: Params):
        num_batch_splits = (
            params.num_batch * params.num_splits
            if const_expr(params.is_split_kv)
            else params.num_batch
        )
        return (
            cute.round_up(params.num_block, params.cluster_shape_m),
            params.num_head,
            num_batch_splits,
        )

    @staticmethod
    @cute.jit
    def clc_problem_shape(params: Params):
        return ClcDynamicPersistentTileSchedulerParams(
            problem_shape_ntile_mnl=SingleTileLPTScheduler._clc_grid_shape(params),
            cluster_shape_mnk=(params.cluster_shape_m, 1, 1),
        )

    @staticmethod
    @cute.jit
    def create(
        params: Params, clc: ClcState | None = None, *, loc=None, ip=None
    ) -> "SingleTileLPTScheduler":
        if const_expr(params.scheduling_mode == SchedulingMode.CLC):
            return SingleTileLPTScheduler(
                params, cute.arch.block_idx()[0], Int32(0), clc, loc=loc, ip=ip
            )
        tile_idx, split_idx, _ = cute.arch.block_idx()
        return SingleTileLPTScheduler(params, tile_idx, split_idx, loc=loc, ip=ip)

    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        if const_expr(params.scheduling_mode == SchedulingMode.CLC):
            return SingleTileLPTScheduler._clc_grid_shape(params)
        return (params.total_blocks, params.num_splits, Int32(1))

    @cute.jit
    def clc_work_to_coords(self, work) -> WorkTileInfo:
        """Convert CLC response (block, head, batch_split) to WorkTileInfo.

        CLC returns raw grid coordinates — no L2 swizzle (hardware decides order).
        We only apply cluster division, optional LPT block reversal, and split_kv unpacking.
        """
        block_idx = work.tile_idx[0]
        if const_expr(self.params.cluster_shape_m > 1):
            block_idx = block_idx // self.params.cluster_shape_m
        if const_expr(self.params.lpt):
            # Longest-processing-time-first: reverse block order
            block_idx = self.params.num_block - 1 - block_idx
        split_idx = Int32(0)
        if const_expr(self.params.is_split_kv):
            batch_idx, split_idx = divmod(work.tile_idx[2], self.params.num_splits_divmod)
        else:
            batch_idx = work.tile_idx[2]
        return WorkTileInfo(
            (Int32(block_idx), Int32(work.tile_idx[1]), Int32(batch_idx), Int32(split_idx)),
            work.is_valid_tile,
        )

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            work = self.clc.get_current_work()
            self._tile_idx = work.tile_idx[0]
            return self.clc_work_to_coords(work)
        # Static path: L2-swizzled coordinate mapping
        params = self.params
        # Implement LPT scheduling coordinate calculation
        bidhb, l2_mod = divmod(self._tile_idx, params.l2_major_divmod)
        # If we're in the last section (called residual), we don't want to divide by
        # swizzle. Instead we want to divide by the remainder.
        block, bidhb_residual = 0, 0
        if bidhb < params.num_hb_quotient:
            block, bidhb_residual = divmod(l2_mod, params.l2_minor_divmod)
        else:
            block, bidhb_residual = divmod(l2_mod, params.l2_minor_residual_divmod)
        bidhb_actual = bidhb * params.l2_minor + bidhb_residual
        batch_idx, head_idx = divmod(bidhb_actual, params.num_head_divmod)
        # Longest-processing-time-first
        if const_expr(params.lpt):
            block = params.num_block - 1 - block
        is_valid = self._tile_idx < params.total_blocks
        return WorkTileInfo(
            (Int32(block), Int32(head_idx), Int32(batch_idx), Int32(self._split_idx)), is_valid
        )

    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            work = self.clc.initial_work_tile_info()
            self._tile_idx = work.tile_idx[0]
            return self.clc_work_to_coords(work)
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self.clc.prefetch_next_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self.clc.consumer_wait(loc=loc, ip=ip)
            work = self.get_current_work()
            self.clc.consumer_release(loc=loc, ip=ip)
            return work
        # Single tile scheduler - set to invalid tile_idx to indicate no more work
        self._tile_idx = self.params.total_blocks
        return self.get_current_work()

    def producer_tail(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self.clc.producer_tail(loc=loc, ip=ip)

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        objs = [self.params, self._tile_idx, self._split_idx]
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            objs += [self.clc]
        for obj in objs:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        objs = [self.params, self._tile_idx, self._split_idx]
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            objs += [self.clc]
        for obj, n_items in zip(objs, self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*obj_list, loc=self._loc)


