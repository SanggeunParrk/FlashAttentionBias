"""SM90 (Hopper) MMA helpers for fp32 (TF32 WGMMA).

cutlass DSL 4.5.x only exposes ``MmaF16BF16Op``/``MmaF8Op``/``MmaI8Op`` for
Hopper warpgroup MMA — there is no upstream Python wrapper for the
``wgmma.mma_async.sync.aligned.m64nNk8.f32.tf32.tf32.f32`` instruction.

This module defines :class:`MmaTF32WgmmaOp` by subclassing the same
:class:`cute.nvgpu.warpgroup.MmaOp` base that ``MmaF16BF16Op`` uses and
constructing the underlying ``MmaAtomSM90Type`` with Float32 / TFloat32
element types.

PROBE RESULT (cutlass DSL 4.5.2, FA4_fp32 H100 branch, 2026-05-28):
    ``MmaAtomSM90Type.get(...)`` succeeds for both Float32 and TFloat32
    mlir_type operands, but ``mma_make_fragment`` (via ``cute.gemm`` ->
    ``make_fragment_B``) fails with ``ValueError: Operation creation failed``.

    Conclusion: the MLIR lowering for ``MmaAtomSM90Type`` is hard-coded to
    fp16/bf16/fp8/int8 fragment shapes. There is no TF32 WGMMA codegen path
    in cutlass DSL 4.5.x.

NEXT STEP (not implemented here):
    Write the TF32 WGMMA path via ``llvm.inline_asm`` directly:
      * pack WGMMA SMEM descriptors (64-bit) for Q/K/V tiles
      * emit ``wgmma.fence.sync.aligned`` / ``fence.proxy.async``
      * emit ``wgmma.mma_async.sync.aligned.m64nNk8.f32.tf32.tf32.f32``
        instructions (one per K iteration)
      * emit ``wgmma.commit_group.sync.aligned`` /
        ``wgmma.wait_group.sync.aligned``
    Replace ``cute.gemm(tiled_mma, ...)`` calls in
    :func:`FA4_fp32.kernels.flash_fwd_sm90.FlashAttentionForwardSm90.mma`
    with the manual sequence when ``self.dtype is Float32``.
"""

from typing import Any, Optional, Tuple, Type, Union, cast
from dataclasses import dataclass

import cutlass.cute as cute
from cutlass import Boolean, Float32, TFloat32
from cutlass.base_dsl.typing import Numeric
from cutlass.cute.core import _pack_shape, rank
from cutlass.cute.typing import Shape
from cutlass.cute.atom import make_atom
from cutlass.cute.nvgpu import OperandMajorMode as _OperandMajorMode
from cutlass.cute.nvgpu.common import OpError
from cutlass.cute.nvgpu.warpgroup.mma import (
    MmaOp,
    MmaTraits,
    OperandMajorMode,
    OperandSource,
)
from cutlass._mlir import ir
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir


@dataclass(frozen=True)
class MmaTF32WgmmaOp(MmaOp):
    """TF32 warpgroup MMA Operation (experimental).

    Targets ``wgmma.mma_async.sync.aligned.m64nNk8.f32.tf32.tf32.f32``.
    Instruction K is 8 (vs 16 for fp16/bf16 WGMMA).
    """

    descriptive_name = "warpgroup TF32 MMA Operation (experimental)"

    def __init__(
        self,
        instruction_shape: Shape,
        a_src: OperandSource,
        a_major_mode: Union[_OperandMajorMode, OperandMajorMode],
        b_major_mode: Union[_OperandMajorMode, OperandMajorMode],
    ) -> None:
        super().__init__(
            TFloat32,
            TFloat32,
            Float32,
            instruction_shape,
            a_src,
            a_major_mode,
            b_major_mode,
        )
        self._verify()

    def _verify(self) -> None:
        # TF32 WGMMA: instruction k = 8.
        instruction_k = 8
        shape_mnk_tuple: Any = cast(Any, self.shape_mnk)
        if rank(shape_mnk_tuple) == 2:
            object.__setattr__(self, "shape_mnk", (*shape_mnk_tuple, instruction_k))
            shape_mnk_tuple = cast(Any, self.shape_mnk)
        if shape_mnk_tuple[2] != instruction_k:
            raise OpError(
                self,
                f"expects the instruction extent in the K-mode to be {instruction_k}, "
                f"but got {shape_mnk_tuple[2]}",
            )

    def _make_trait(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
        **kwargs: Any,
    ) -> MmaTraits:
        shape_mnk = _pack_shape(self.shape_mnk, loc=loc, ip=ip)
        ty = _cute_nvgpu_ir.MmaAtomSM90Type.get(
            shape_mnk.type.attribute,
            self.a_major_mode._to_ir(),
            self.b_major_mode._to_ir(),
            self.a_dtype.mlir_type,
            self.b_dtype.mlir_type,
            self.acc_dtype.mlir_type,
            self.a_src._to_ir(),
        )
        # Reachable; the actual MLIR rejection happens later at
        # mma_make_fragment time when the kernel does cute.gemm(...).
        return MmaTraits(
            make_atom(ty, [Boolean(False).ir_value(loc=loc, ip=ip)], loc=loc, ip=ip)
        )


def make_trivial_tiled_mma(
    a_dtype: Type[Numeric],
    b_dtype: Type[Numeric],
    a_leading_mode: Union[_OperandMajorMode, OperandMajorMode],
    b_leading_mode: Union[_OperandMajorMode, OperandMajorMode],
    acc_dtype: Type[Numeric],
    atom_layout_mnk: Tuple[int, int, int],
    tiler_mn: Tuple[int, int],
    a_source: OperandSource = OperandSource.SMEM,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.TiledMma:
    """Like ``cutlass.utils.hopper_helpers.make_trivial_tiled_mma`` but routes
    Float32 / TFloat32 operands through :class:`MmaTF32WgmmaOp` instead of
    raising. Falls back to the upstream helper for fp16/bf16/fp8/int8.
    """
    if a_dtype in {Float32, TFloat32} and b_dtype == a_dtype:
        assert acc_dtype is Float32, "TF32 WGMMA accumulator must be Float32"
        mma_op = MmaTF32WgmmaOp(
            (*tiler_mn, 8),
            a_source,
            a_leading_mode,
            b_leading_mode,
        )
        return cute.make_tiled_mma(
            cute.make_mma_atom(mma_op, loc=loc, ip=ip), atom_layout_mnk
        )

    # Defer to upstream helper for other dtypes.
    import cutlass.utils.hopper_helpers as _upstream

    return _upstream.make_trivial_tiled_mma(
        a_dtype,
        b_dtype,
        a_leading_mode,
        b_leading_mode,
        acc_dtype,
        atom_layout_mnk,
        tiler_mn,
        a_source,
        loc=loc,
        ip=ip,
    )
