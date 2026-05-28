"""TF32 counterparts of ``quack.sm90_utils.{partition_fragment_ABC, gemm_zero_init,
gemm_w_idx, gemm}`` that call our inline-asm WGMMA TF32 primitives.

Why this exists
---------------
cutlass DSL 4.5.x has no TF32 WGMMA path (see ``FA4_fp32/arch/hopper_helpers.py``
probe). The quack helpers go through ``thr_mma.make_fragment_*`` which derives
register fragment shapes from the MMA atom — that's exactly the call that fails
for our :class:`MmaTF32WgmmaOp` (``Operation creation failed`` at
``mma_make_fragment``).

This module provides a parallel implementation that:
  1. Builds register tensors of the correct *total size* via
     :func:`cute.make_rmem_tensor` directly — bypassing the broken
     ``make_fragment_*`` path.
  2. Uses the per-thread WGMMA TF32 fragment layout from the PTX manual to
     index those tensors.
  3. Wraps :mod:`FA4_fp32.arch.wgmma_tf32` primitives (fence / commit /
     wait / mma_async) to issue the actual WGMMA instructions.

Status
------
This is a **scaffold**. Function signatures match the quack helpers so the
dispatcher in ``flash_fwd_sm90.py`` can route fp32 here, but the actual
fragment-partition + WGMMA-issue body is **not yet validated on H100** —
a CUDA-13 driver is required to test, and our cluster has CUDA 12.9.

The TODOs below mark the exact spots where the layout math needs to be
written + verified against the PTX docs:

  PTX docs §asynchronous-warpgroup-level-matrix-instructions-mma-matrix-fragments
  https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-mma-matrix-fragments

For m64nNk8.f32 acc fragment, per thread t (laneID = t % 32, warp = t / 32):
  * 2 rows × N/4 column-pairs = N/2 fp32 regs.
  * Row r in {0, 1} maps to global row ``warp*16 + 8*(laneID//4) + r``.
  * Col c in {0, ..., N/2 - 1} maps to global col ``8*(c//2) + (laneID%4)*2 + (c%2)``.

For m64nNk8 A-source-SMEM fragment, per thread: 4 .b32 (tf32) elements per
k-tick, distributed similarly. For A-source-RMEM, the same 4-element fragment
must already be in registers.
"""

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import warpgroup

from FA4_fp32.arch import wgmma_tf32


def partition_fragment_ABC(
    thr_mma: cute.ThrMma,
    shape_mnk: cute.Shape,
    sA: Optional[cute.Tensor],
    sB: Optional[cute.Tensor],
    swap_AB: bool = False,
) -> Tuple[cute.Tensor, cute.Tensor, cute.Tensor]:
    """TF32 fragment partition.

    Returns ``(acc, tCrA, tCrB)`` register tensors.

    NOT IMPLEMENTED — see module docstring. Raises NotImplementedError on
    entry so callers fail loudly until the layout math is filled in. The
    structure below shows where each piece goes.
    """
    is_rs = thr_mma.op.a_src == warpgroup.OperandSource.RMEM
    if const_expr(swap_AB):
        raise NotImplementedError("TF32 WGMMA + swap_AB not implemented yet")
    tile_m, tile_n, tile_k = shape_mnk[0], shape_mnk[1], shape_mnk[2]

    # TODO(H100): cute.make_rmem_tensor with a layout that matches the WGMMA
    # fp32 acc fragment shape (per-thread N/2 elements at the indices above).
    # Likely shape: ((2, N/2), tile_m // 64), strides matched to thread tile.
    # acc = cute.make_rmem_tensor(_acc_layout(tile_m, tile_n), Float32)
    raise NotImplementedError(
        "TF32 partition_fragment_ABC scaffold reached. Fill in fragment "
        "layout per the PTX m64nNk8.f32 fragment spec, then remove this raise."
    )


def gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: cute.Tensor,
    sB: cute.Tensor,
    zero_init: cutlass.Constexpr[bool] = False,
    wg_wait: cutlass.Constexpr[int] = 0,
    swap_AB: cutlass.Constexpr[bool] = False,
) -> None:
    """TF32 WGMMA gemm loop. Mirrors :func:`quack.sm90_utils.gemm`.

    NOT IMPLEMENTED. The skeleton is documented; the call sites in
    :mod:`FA4_fp32.arch.wgmma_tf32` already exist for the per-instruction emit.

    Outline of the body once filled in::

        wgmma_tf32.fence_aligned()
        scale_d = not zero_init
        # head_dim is tile_k; one wgmma per 8 K-elements.
        for k in range(tile_k // 8):
            desc_a = wgmma_tf32.pack_smem_descriptor_runtime(
                sA_byte_addr_at_k(k), leading_a, stride_a,
                swizzle_mode=swizzle_a)
            desc_b = wgmma_tf32.pack_smem_descriptor_runtime(
                sB_byte_addr_at_k(k), leading_b, stride_b,
                swizzle_mode=swizzle_b)
            new_acc = wgmma_tf32.wgmma_mma_async_tf32_m64nNk8(
                N=tile_n,
                acc_regs=[acc[i] for i in range(tile_n // 2)],
                desc_a=desc_a,
                desc_b=desc_b,
                scale_d=scale_d,
                k_major_a=True,
                k_major_b=k_major_b,
            )
            # write new_acc back into the acc tensor
            scale_d = True   # subsequent k-ticks accumulate
        wgmma_tf32.commit_group()
        if wg_wait >= 0:
            wgmma_tf32.wait_group(wg_wait)
    """
    raise NotImplementedError(
        "TF32 WGMMA gemm scaffold reached. Fill in the K loop with descriptor "
        "packing + wgmma_tf32.wgmma_mma_async_tf32_m64nNk8 calls."
    )


def gemm_zero_init(
    tiled_mma: cute.TiledMma,
    shape: cute.Shape,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: cute.Tensor,
    sB: cute.Tensor,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    wg_wait: int = -1,
    swap_AB: bool = False,
) -> cute.Tensor:
    """TF32 zero-init gemm; signature mirrors :func:`quack.sm90_utils.gemm_zero_init`.

    NOT IMPLEMENTED — composes :func:`partition_fragment_ABC` (acc allocation)
    + :func:`gemm` with ``zero_init=True``.
    """
    raise NotImplementedError(
        "TF32 gemm_zero_init scaffold reached. Fill once partition_fragment_ABC "
        "+ gemm are implemented."
    )


def gemm_w_idx(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: cute.Tensor,
    sB: cute.Tensor,
    zero_init: Boolean,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    wg_wait: int = -1,
    swap_AB: bool = False,
) -> None:
    """TF32 indexed gemm; signature mirrors :func:`quack.sm90_utils.gemm_w_idx`.

    NOT IMPLEMENTED.
    """
    raise NotImplementedError(
        "TF32 gemm_w_idx scaffold reached. Fill once gemm is implemented."
    )
