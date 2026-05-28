"""Inline-asm WGMMA TF32 primitives for Hopper SM90.

Workaround for cutlass DSL 4.5.x lacking TF32 WGMMA codegen
(``MmaAtomSM90Type`` only lowers fp16/bf16/fp8/int8 — see
``FA4_fp32/arch/hopper_helpers.py`` for the upstream probe).

This module emits the underlying PTX instructions directly via
``llvm.inline_asm`` so callers can build TF32 WGMMA kernels without going
through ``cute.gemm()``.

PTX reference (`PTX docs §asynchronous-warpgroup-level-matrix-instructions`__):

  wgmma.mma_async.sync.aligned.m64n{N}k8.f32.tf32.tf32.f32
      {d0, d1, ..., d_{N/2 - 1}},   // N/2 32-bit fp32 dst regs per thread
      desc-a | {Ra0, Ra1, Ra2, Ra3},
      desc-b,
      p,                              // scale-d (1=acc, 0=overwrite)
      imm-sa,                         // scale-a (1 or -1)
      imm-sb,                         // scale-b (1 or -1)
      [, imm-ta, imm-tb] ;            // operand transpose (MN-major)

SMEM descriptor format (64 bits, see PTX §asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor):

  bits  field
  ------------
   0-13 matrix start address  (smem byte addr >> 4)
  14-15 reserved
  16-29 leading dim byte offset (>> 4)
  30-31 reserved
  32-45 stride dim byte offset  (>> 4)
  46-48 reserved
  49-51 matrix base offset (for swizzled layouts)
  52-60 reserved
  61-63 swizzling mode (0=none, 1=128B, 2=64B, 3=32B)


Scope of this module
====================
* :func:`fence_aligned`, :func:`commit_group`, :func:`wait_group` — sync primitives.
* :func:`pack_smem_descriptor` — host-side helper that builds the 64-bit descriptor.
* :func:`wgmma_mma_async_tf32_m64nNk8` — emits the WGMMA TF32 instruction for
  the (currently) supported tile widths ``N in {32, 64, 96, 128}``.

Integration into ``FA4_fp32/kernels/flash_fwd_sm90.py`` is NOT done here
because the quack-based fragment partition (``sm90_utils.partition_fragment_ABC``)
relies on cute's MMA atom which has no TF32 fragment shape. Integration
requires writing a parallel fragment partition (register allocation + smem
descriptor derivation from the smem layout) — substantial extra work
tracked separately.
"""

from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


# ---------------------------------------------------------------------------
# Sync primitives
# ---------------------------------------------------------------------------

@dsl_user_op
def fence_aligned(*, loc=None, ip=None) -> None:
    """``wgmma.fence.sync.aligned;`` — must precede the first MMA in a group.

    Tells the compiler/hardware that all prior register/SMEM writes by this
    warpgroup are visible to the upcoming WGMMA. Issue from all 128 threads
    in the warpgroup.
    """
    llvm.inline_asm(
        None,
        [],
        "wgmma.fence.sync.aligned;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def commit_group(*, loc=None, ip=None) -> None:
    """``wgmma.commit_group.sync.aligned;`` — closes the current MMA group.

    Subsequent WGMMAs go into the next group. Pair with :func:`wait_group`.
    """
    llvm.inline_asm(
        None,
        [],
        "wgmma.commit_group.sync.aligned;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def wait_group(n: cutlass.Constexpr[int], *, loc=None, ip=None) -> None:
    """``wgmma.wait_group.sync.aligned N;`` — waits until at most N groups are pending.

    Use ``wait_group(0)`` to drain everything before reading the accumulators.
    """
    llvm.inline_asm(
        None,
        [],
        f"wgmma.wait_group.sync.aligned {int(n)};",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# ---------------------------------------------------------------------------
# SMEM descriptor packing
# ---------------------------------------------------------------------------

# Swizzle mode encodings used in the descriptor (bits 61-63).
SWIZZLE_NONE = 0
SWIZZLE_128B = 1
SWIZZLE_64B = 2
SWIZZLE_32B = 3


def encode_smem_descriptor(
    matrix_start_byte_addr: int,
    leading_byte_offset: int,
    stride_byte_offset: int,
    base_offset: int = 0,
    swizzle_mode: int = SWIZZLE_NONE,
) -> int:
    """Pack a WGMMA SMEM matrix descriptor (host-side, compile-time).

    All byte offsets must be multiples of 16.

    See PTX §asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor.
    """
    assert matrix_start_byte_addr % 16 == 0
    assert leading_byte_offset % 16 == 0
    assert stride_byte_offset % 16 == 0
    assert 0 <= swizzle_mode <= 3
    assert 0 <= base_offset <= 7
    desc = 0
    desc |= ((matrix_start_byte_addr >> 4) & ((1 << 14) - 1)) << 0
    desc |= ((leading_byte_offset >> 4) & ((1 << 14) - 1)) << 16
    desc |= ((stride_byte_offset >> 4) & ((1 << 14) - 1)) << 32
    desc |= (base_offset & 0x7) << 49
    desc |= (swizzle_mode & 0x7) << 61
    return desc


@dsl_user_op
def pack_smem_descriptor_runtime(
    smem_byte_addr: Int64,
    leading_byte_offset_div16: Int32,
    stride_byte_offset_div16: Int32,
    base_offset: cutlass.Constexpr[int] = 0,
    swizzle_mode: cutlass.Constexpr[int] = SWIZZLE_NONE,
    *,
    loc=None,
    ip=None,
) -> Int64:
    """Runtime variant: pack a WGMMA SMEM descriptor when the smem address
    is only known at runtime (typical case — TMA target buffer).

    ``leading_byte_offset_div16`` and ``stride_byte_offset_div16`` should
    already be byte_offset // 16 (small ints).

    Emits a short PTX sequence that shifts the smem address right by 4 then
    ORs in the constant pieces.
    """
    # Compose the constant high half on the host:
    const_bits = (
        ((int(leading_byte_offset_div16) & ((1 << 14) - 1)) << 16)
        | ((int(stride_byte_offset_div16) & ((1 << 14) - 1)) << 32)
        | ((int(base_offset) & 0x7) << 49)
        | ((int(swizzle_mode) & 0x7) << 61)
    )
    # desc = (smem_byte_addr >> 4) | const_bits
    addr_i64 = smem_byte_addr.ir_value()
    desc = llvm.inline_asm(
        T.i64(),
        [addr_i64, Int64(const_bits).ir_value(loc=loc, ip=ip)],
        # encode start addr (>> 4) and OR with precomputed constant
        "{ .reg .b64 a; shr.u64 a, $1, 4; and.b64 a, a, 0x3fff; or.b64 $0, a, $2; }",
        "=l,l,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int64(desc)


# ---------------------------------------------------------------------------
# WGMMA TF32 instruction (m64 x N x k8). N is determined at compile time.
#
# Each variant is its own inline_asm template because the destination
# register list length is N/2 and PTX inline_asm doesn't have variadic
# placeholders.
#
# For each variant we accept the destination accumulator registers as a list
# of fp32 references (read-write inout). Caller is responsible for fragment
# allocation and layout matching.
# ---------------------------------------------------------------------------


def _make_dst_list(n_regs: int) -> str:
    return "{" + ", ".join(f"${i}" for i in range(n_regs)) + "}"


def _wgmma_tf32_asm(N: int, k_major_a: bool, k_major_b: bool) -> str:
    """Build the PTX template for ``wgmma.mma_async ... m64n{N}k8 ...``.

    Operands placeholders (in order):
        $0..$(N/2 - 1)    fp32 accumulator regs (rw inout)
        $N/2              SMEM descriptor A (i64)
        $N/2 + 1          SMEM descriptor B (i64)
        $N/2 + 2          scale-d (i32 boolean)

    scale-a / scale-b / trans-a / trans-b are baked in as immediates (+1, +1,
    and 0/1 transpose flags decided by k_major_{a,b}).
    """
    n_regs = N // 2
    dst = _make_dst_list(n_regs)
    desc_a = f"${n_regs}"
    desc_b = f"${n_regs + 1}"
    scale_d = f"${n_regs + 2}"
    # trans flags: 0 if K-major, 1 if MN-major.
    ta = 0 if k_major_a else 1
    tb = 0 if k_major_b else 1
    return (
        f"wgmma.mma_async.sync.aligned.m64n{N}k8.f32.tf32.tf32.f32 "
        f"{dst}, {desc_a}, {desc_b}, {scale_d}, 1, 1, {ta}, {tb};"
    )


def _wgmma_tf32_constraints(N: int) -> str:
    """LLVM inline_asm constraint string for the m64n{N}k8 variant."""
    n_regs = N // 2
    # n_regs fp32 inout (=+f), then 2 i64 desc inputs (l), then 1 i32 scale-d (r).
    return ",".join(["+f"] * n_regs + ["l", "l", "r"])


_SUPPORTED_N = (32, 64, 96, 128)


@dsl_user_op
def wgmma_mma_async_tf32_m64nNk8(
    N: cutlass.Constexpr[int],
    acc_regs: list,
    desc_a: Int64,
    desc_b: Int64,
    scale_d: cutlass.Constexpr[bool] = True,
    k_major_a: cutlass.Constexpr[bool] = True,
    k_major_b: cutlass.Constexpr[bool] = True,
    *,
    loc=None,
    ip=None,
) -> list:
    """Emit one ``wgmma.mma_async.sync.aligned.m64n{N}k8.f32.tf32.tf32.f32``.

    ``acc_regs`` is a list of N/2 ``cutlass.Float32`` values (inout).
    Returns the updated list.

    The caller must surround issued WGMMAs with :func:`fence_aligned`
    BEFORE the first MMA in a group, then :func:`commit_group` after issuing
    a group, and :func:`wait_group` before reading any ``acc_regs``.
    """
    if int(N) not in _SUPPORTED_N:
        raise NotImplementedError(
            f"WGMMA TF32 m64n{int(N)}k8 not implemented yet; supported: {_SUPPORTED_N}"
        )
    n_regs = int(N) // 2
    if len(acc_regs) != n_regs:
        raise ValueError(
            f"acc_regs has length {len(acc_regs)}, expected {n_regs} for N={int(N)}"
        )

    asm_template = _wgmma_tf32_asm(int(N), bool(k_major_a), bool(k_major_b))
    constraints = _wgmma_tf32_constraints(int(N))

    operand_ir = (
        [r.ir_value() for r in acc_regs]
        + [desc_a.ir_value(), desc_b.ir_value()]
        + [Int32(1 if scale_d else 0).ir_value(loc=loc, ip=ip)]
    )

    # llvm.inline_asm returns a single result for the first =/+ constraint
    # but with n_regs '+f' constraints, MLIR will produce a tuple. The
    # cutlass DSL `llvm.inline_asm` wrapper handles this via the result
    # type. We pass a tuple of n_regs i32 (fp32 reinterpreted) types.
    fp32_ty = T.f32()
    result_ty = [fp32_ty] * n_regs

    new_vals = llvm.inline_asm(
        result_ty,
        operand_ir,
        asm_template,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    # `new_vals` is a sequence of MLIR Values; wrap each back to Float32.
    return [cutlass.Float32(v) for v in (new_vals if isinstance(new_vals, (list, tuple)) else [new_vals])]


# ---------------------------------------------------------------------------
# High-level helper (NOT YET wired into flash_fwd_sm90.py).
#
# To complete integration:
#   * Compute the SMEM descriptor for the Q / K / V tiles using
#     :func:`pack_smem_descriptor_runtime` with the layout's leading and
#     stride offsets pulled from `cute.size_in_bytes` of the smem layout.
#   * Replace the ``sm90_utils.partition_fragment_ABC`` + ``cute.gemm`` pair
#     in ``FlashAttentionForwardSm90.mma`` with:
#       - A fp32 register tensor for the accumulator (shape (tile_m/64, n_regs)).
#       - A K loop over k8 ticks (head_dim / 8 iterations) issuing
#         :func:`wgmma_mma_async_tf32_m64nNk8` per tick with shifted descriptors.
#       - Fence / commit_group / wait_group around the loop.
# ---------------------------------------------------------------------------
