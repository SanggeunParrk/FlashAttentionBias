# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# B200-only, MHA-only, minimal flash attention.
#
# Supported:
# - dtype fp16 / bf16 / fp32 (TF32 MMA)
# - SM100 only (Blackwell B200)
# - MHA only (num_heads_q == num_heads_kv)
# - fixed-length, non-causal, no bias yet
# - forward + backward, optional deterministic bwd, return_lse

import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

import cuda.bindings.driver as cuda  # noqa: F401

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from quack.compile_utils import make_fake_tensor as fake_tensor

from FA4_fp32.infra.cache_utils import get_jit_cache
from FA4_fp32.infra.testing import is_fake_mode


if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    from FA4_fp32.infra import cute_dsl_ptxas  # noqa: F401
    cute_dsl_ptxas.patch()


from FA4_fp32.core import utils
from FA4_fp32.infra import fa_logging  # noqa: F401
from FA4_fp32.infra.cute_dsl_utils import to_cute_tensor, get_broadcast_dims
from FA4_fp32.kernels.flash_fwd_sm100 import FlashAttentionForwardSm100
from FA4_fp32.kernels.flash_bwd_preprocess import FlashAttentionBackwardPreprocess
from FA4_fp32.kernels.flash_bwd_sm100 import FlashAttentionBackwardSm100
from FA4_fp32.kernels.flash_bwd_postprocess import FlashAttentionBackwardPostprocess


_SM100_ARCHES = (100, 101, 103, 110)


def _get_device_arch() -> int:
    arch_override = os.environ.get("FLASH_ATTENTION_ARCH", None)
    if arch_override is not None:
        import re
        m = re.match(r"^(?:sm_?|SM_?)?(\d+)(\d)([af]?)$", arch_override)
        if not m:
            raise ValueError(f"Invalid arch format: {arch_override}")
        major, minor, _ = m.groups()
        return int(major) * 10 + int(minor)
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + int(minor)


def _validate_head_dims(head_dim: int, head_dim_v: int, alignment: int) -> None:
    is_deepseek_shape = head_dim == 192 and head_dim_v == 128
    is_standard_range = 8 <= head_dim <= 128 and 8 <= head_dim_v <= 128
    assert (is_standard_range or is_deepseek_shape) and head_dim % alignment == 0 and head_dim_v % alignment == 0, (
        f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) not supported on SM100. "
        f"Must be 8..128 divisible by {alignment}, or (192, 128)."
    )


@dataclass(frozen=True)
class FwdConfig:
    m_block_size: int
    n_block_size: int
    mma_pv_is_rs: bool
    intra_wg_overlap: bool


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _validate_tensor(t, name, expected_shape, expected_dtype, expected_device):
    assert t.shape == expected_shape, f"{name} shape {t.shape} != expected {expected_shape}"
    assert t.dtype == expected_dtype, f"{name} dtype {t.dtype} != expected {expected_dtype}"
    assert t.device == expected_device, f"{name} device {t.device} != expected {expected_device}"
    if not is_fake_mode():
        assert t.is_cuda, f"{name} must be on CUDA"


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


# ======================================================================
# Forward
# ======================================================================

def _flash_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    batch_size, seqlen_q, num_head, head_dim = q.shape
    seqlen_k = k.shape[1]
    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]

    assert num_head_kv == num_head, "MHA only (num_heads_q must equal num_heads_kv)"
    assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
    assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
    assert q.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        "inputs must be float16, bfloat16, or float32 (TF32 MMA)"
    )
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
    if not is_fake_mode():
        assert all(t.is_cuda for t in (q, k, v)), "inputs must be on CUDA device"

    arch = _get_device_arch()
    assert arch in _SM100_ARCHES, f"Unsupported compute capability {arch}; only SM100/SM110 supported"

    alignment = 16 // q.element_size()
    _validate_head_dims(head_dim, head_dim_v, alignment)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    device = q.device
    out_torch_dtype = q.dtype
    requires_grad = q.requires_grad or k.requires_grad or v.requires_grad

    if out is None:
        out = torch.empty(batch_size, seqlen_q, num_head, head_dim_v,
                          dtype=out_torch_dtype, device=device)
    else:
        _validate_tensor(out, "out", (batch_size, seqlen_q, num_head, head_dim_v),
                         out_torch_dtype, device)

    lse_shape = (batch_size, num_head, seqlen_q)
    if lse is None:
        lse = torch.empty(lse_shape, dtype=torch.float32, device=device) if (requires_grad or return_lse) else None
    else:
        _validate_tensor(lse, "lse", lse_shape, torch.float32, device)

    dtype = torch2cute_dtype_map[q.dtype]

    # Default SM100 tile config for MHA. fp32 input uses 2x smem, so smaller tile.
    if q.dtype == torch.float32:
        tile_m, tile_n = 128, 64
        q_stage = 1
    else:
        tile_m, tile_n = 128, 128
        q_stage = 2 if seqlen_q > tile_m else 1

    # 2-CTA eligible when hdim padded to 128/192 on V-side 128 and seqlen big enough.
    requested_disable_2cta = utils._get_disable_2cta_default()
    use_2cta_instrs = (
        not requested_disable_2cta
        and q.dtype != torch.float32
        and int(math.ceil(head_dim / 16) * 16) in [128, 192]
        and int(math.ceil(head_dim_v / 16) * 16) == 128
        and seqlen_q > 2 * tile_m
    )
    use_clc_scheduler = utils._get_use_clc_scheduler_default()

    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compile_key = (
        dtype, head_dim, head_dim_v, tile_m, tile_n, q_stage,
        use_2cta_instrs, use_clc_scheduler, lse is None,
        get_broadcast_dims(q), get_broadcast_dims(k), get_broadcast_dims(v),
        arch,
    )
    if compile_key not in _flash_attn_fwd.compile_cache:
        q_tensor, k_tensor, v_tensor, o_tensor = [to_cute_tensor(t) for t in (q, k, v, out)]
        lse_tensor = to_cute_tensor(lse, assumed_align=4) if lse is not None else None

        fa_fwd = FlashAttentionForwardSm100(
            head_dim,
            head_dim_v,
            qhead_per_kvhead=1,
            m_block_size=tile_m,
            n_block_size=tile_n,
            q_stage=q_stage,
            is_persistent=True,
            use_2cta_instrs=use_2cta_instrs,
            use_clc_scheduler=use_clc_scheduler,
        )
        _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
            fa_fwd,
            q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor,
            softmax_scale,
            None, None, None, None,          # cu_seqlens_q/k, seqused_q/k
            None,                             # page_table
            None, None,                       # window_size_left/right
            None,                             # learnable_sink
            None,                             # block_sparse_tensors
            None,                             # aux_tensors
            current_stream,
            options="--enable-tvm-ffi",
        )

    if not is_fake_mode():
        _flash_attn_fwd.compile_cache[compile_key](
            q.detach(), k.detach(), v.detach(), out.detach(), lse,
            softmax_scale,
            None, None, None, None,
            None,
            None, None,
            None,
            None,
            None,
        )
    return out, lse


_flash_attn_fwd.compile_cache = get_jit_cache("fwd")


# ======================================================================
# Backward preprocess / postprocess helpers
# ======================================================================

def _make_fake_bwd_tensors(dtype):
    sym = cute.sym_int
    div = 128 // dtype.width
    b, seqlen_q, seqlen_k, h, d, d_v = sym(), sym(), sym(), sym(), sym(), sym()
    seqlen_q_rounded = sym()
    mQ = fake_tensor(dtype, (b, seqlen_q, h, d), divisibility=div)
    mO = fake_tensor(dtype, (b, seqlen_q, h, d_v), divisibility=div)
    mdO = fake_tensor(dtype, (b, seqlen_q, h, d_v), divisibility=div)
    mK = fake_tensor(dtype, (b, seqlen_k, h, d), divisibility=div)
    mV = fake_tensor(dtype, (b, seqlen_k, h, d_v), divisibility=div)
    mdQ = fake_tensor(dtype, (b, seqlen_q, h, d), divisibility=div)
    mdK = fake_tensor(dtype, (b, seqlen_k, h, d), divisibility=div)
    mdV = fake_tensor(dtype, (b, seqlen_k, h, d_v), divisibility=div)
    mLSE = fake_tensor(Float32, (b, h, seqlen_q), divisibility=1)
    mLSElog2 = fake_tensor(Float32, (b, h, seqlen_q_rounded), divisibility=4)
    mPdPsum = fake_tensor(Float32, (b, h, seqlen_q_rounded), divisibility=4)
    mdQaccum = fake_tensor(Float32, (b, h, sym()), divisibility=4)
    return mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, mdQaccum


def _compile_bwd_preprocess(dtype, head_dim, head_dim_v, m_block_size, has_dlse):
    mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, mdQaccum = _make_fake_bwd_tensors(dtype)
    mdLSE = fake_tensor(Float32, mLSE.shape, divisibility=1) if has_dlse else None
    fa_bwd_pre = FlashAttentionBackwardPreprocess(dtype, head_dim, head_dim_v, m_block_size)
    return cute.compile(
        fa_bwd_pre, mO, mdO, mPdPsum, mLSE, mLSElog2, mdQaccum, None, None, mdLSE,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_preprocess(out, dout, dpsum, lse, lse_log2, dq_accum, dlse,
                    dtype, head_dim, head_dim_v, m_block_size):
    compile_key = (dtype, head_dim, head_dim_v, m_block_size, dlse is not None)
    if compile_key not in _bwd_preprocess.compile_cache:
        _bwd_preprocess.compile_cache[compile_key] = _compile_bwd_preprocess(*compile_key)
    if not is_fake_mode():
        _bwd_preprocess.compile_cache[compile_key](
            out, dout, dpsum, lse, lse_log2, dq_accum, None, None, dlse
        )


_bwd_preprocess.compile_cache = get_jit_cache("bwd_pre")


def _compile_bwd_postprocess(dtype, hdim, block_size, num_threads, atom_layout, swap_ab, arch):
    mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, mdQaccum = _make_fake_bwd_tensors(dtype)
    fa_bwd_post = FlashAttentionBackwardPostprocess(
        dtype, hdim, arch, block_size, num_threads, atom_layout, swap_ab,
        use_2cta_instrs=False, cluster_size=1,
    )
    return cute.compile(
        fa_bwd_post, mdQaccum, mdQ, Float32(0.0), None, None,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_postprocess_convert(accum, output, scale, arch, dtype, hdim, block_size, num_threads,
                             atom_layout, swap_ab):
    compile_key = (dtype, hdim, block_size, num_threads, atom_layout, swap_ab, arch)
    if compile_key not in _bwd_postprocess_convert.compile_cache:
        _bwd_postprocess_convert.compile_cache[compile_key] = _compile_bwd_postprocess(*compile_key)
    if not is_fake_mode():
        _bwd_postprocess_convert.compile_cache[compile_key](accum, output, scale, None, None)


_bwd_postprocess_convert.compile_cache = get_jit_cache("bwd_post")


# ======================================================================
# Backward
# ======================================================================

def _flash_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    deterministic: bool = False,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    dlse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    arch = _get_device_arch()
    assert arch in _SM100_ARCHES, f"Unsupported compute capability {arch}; only SM100/SM110 supported"

    num_head, head_dim = q.shape[-2:]
    head_dim_v = v.shape[-1]

    # Default SM100 MHA tile sizes (mirrors previous interface logic).
    m_block_size = 128
    n_block_size = 128
    dQ_swapAB = False
    dKV_swapAB = False
    AtomLayoutMdQ = 1
    AtomLayoutNdKV = 1
    requested_disable_2cta = utils._get_disable_2cta_default()
    cluster_size = 2 if head_dim >= 128 and not requested_disable_2cta else 1
    use_2cta_instrs = cluster_size == 2
    if q.dtype == torch.float32:
        m_block_size = 64
        n_block_size = 128
        cluster_size = 1
        use_2cta_instrs = False
    num_threads = 384

    q, k, v, out, dout, lse = [maybe_contiguous(t) for t in (q, k, v, out, dout, lse)]
    batch_size, seqlen_q = q.shape[:2]
    seqlen_k = k.shape[1]
    num_head_kv = k.shape[-2]

    assert num_head_kv == num_head, "MHA only (num_heads_q must equal num_heads_kv)"

    seqlen_q_rounded = (seqlen_q + m_block_size - 1) // m_block_size * m_block_size
    seqlen_k_rounded = (seqlen_k + n_block_size - 1) // n_block_size * n_block_size
    num_n_blocks = seqlen_k_rounded // n_block_size
    if cluster_size == 2 and num_n_blocks % cluster_size != 0:
        seqlen_k_rounded = seqlen_k_rounded + n_block_size

    assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
    assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
    assert out.shape == (batch_size, seqlen_q, num_head, head_dim_v)
    assert dout.shape == (batch_size, seqlen_q, num_head, head_dim_v)
    assert lse.shape == (batch_size, num_head, seqlen_q)
    assert q.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype
    assert lse.dtype == torch.float32

    if dlse is not None:
        dlse = maybe_contiguous(dlse)
    if not is_fake_mode():
        assert all(t.is_cuda for t in (q, k, v, out, dout, lse))

    alignment = 16 // q.element_size()
    _validate_head_dims(head_dim, head_dim_v, alignment)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    device = q.device
    out_torch_dtype = q.dtype

    if dq is None:
        dq = torch.empty_like(q)
    else:
        _validate_tensor(dq, "dq", q.shape, out_torch_dtype, device)
    if dk is None:
        dk = torch.empty_like(k)
    else:
        _validate_tensor(dk, "dk", k.shape, out_torch_dtype, device)
    if dv is None:
        dv = torch.empty_like(v)
    else:
        _validate_tensor(dv, "dv", v.shape, out_torch_dtype, device)

    head_dim_rounded = (head_dim + 32 - 1) // 32 * 32

    dq_accum = torch.empty(batch_size, num_head, seqlen_q_rounded * head_dim_rounded,
                           dtype=torch.float32, device=device)
    dpsum = torch.empty(batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device)
    lse_log2 = torch.empty(batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device)

    dtype = torch2cute_dtype_map[q.dtype]
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    if deterministic:
        dQ_semaphore = torch.zeros(batch_size, num_head, seqlen_q_rounded // m_block_size, cluster_size,
                                   dtype=torch.int32, device=device)
    else:
        dQ_semaphore = None

    _bwd_preprocess(out, dout, dpsum, lse, lse_log2, dq_accum, dlse,
                    dtype, head_dim, head_dim_v, m_block_size)

    compile_key = (
        arch, dtype, head_dim, head_dim_v,
        m_block_size, n_block_size, num_threads,
        cluster_size, use_2cta_instrs, deterministic,
        get_broadcast_dims(q), get_broadcast_dims(k), get_broadcast_dims(v), get_broadcast_dims(dout),
        (seqlen_q_rounded // m_block_size == 1),
        (seqlen_k_rounded // n_block_size == 1),
    )
    if compile_key not in _flash_attn_bwd.compile_cache:
        q_tensor, k_tensor, v_tensor, do_tensor, dq_tensor, dk_tensor, dv_tensor = [
            to_cute_tensor(t) for t in (q, k, v, dout, dq, dk, dv)
        ]
        dq_accum_tensor, dpsum_tensor, lse_log2_tensor = [
            to_cute_tensor(t) for t in (dq_accum, dpsum, lse_log2)
        ]
        dQ_semaphore_tensor = (
            utils.convert_from_dlpack_leading_static(
                dQ_semaphore.detach(), leading_dim=3, alignment=4, stride_order=dQ_semaphore.dim_order()
            ) if dQ_semaphore is not None else None
        )

        fa_bwd_obj = FlashAttentionBackwardSm100(
            head_dim,
            head_dim_v,
            qhead_per_kvhead=1,
            tile_m=m_block_size,
            tile_n=n_block_size,
            cluster_size=cluster_size,
            use_2cta_instrs=use_2cta_instrs,
            deterministic=deterministic,
            subtile_factor=2,
            q_dtype=dtype,
        )

        _flash_attn_bwd.compile_cache[compile_key] = cute.compile(
            fa_bwd_obj,
            q_tensor, k_tensor, v_tensor, do_tensor,
            lse_log2_tensor, dpsum_tensor, dq_accum_tensor,
            dk_tensor, dv_tensor,
            softmax_scale,
            None, None, None, None,           # cu_seqlens_q/k, seqused_q/k
            None, None,                       # window_size_left/right
            dQ_semaphore_tensor, None, None,  # dQ/dK/dV semaphores
            None,                             # aux_tensors
            None,                             # block_sparse_tensors
            current_stream,
            options="--enable-tvm-ffi",
        )

    if not is_fake_mode():
        _flash_attn_bwd.compile_cache[compile_key](
            q.detach(), k.detach(), v.detach(), dout,
            lse_log2, dpsum, dq_accum, dk, dv,
            softmax_scale,
            None, None, None, None,
            None, None,
            dQ_semaphore, None, None,
            None,
            None,
        )

    # Postprocess: convert dq_accum (float32) → dq (orig dtype) with softmax_scale.
    _bwd_postprocess_convert(
        dq_accum, dq, softmax_scale,
        arch, dtype, head_dim, m_block_size, 128,
        AtomLayoutMdQ, dQ_swapAB,
    )
    return dq, dk, dv


_flash_attn_bwd.compile_cache = get_jit_cache("bwd")


# ======================================================================
# Autograd wrapper + public API
# ======================================================================

class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: Optional[float] = None,
        deterministic: bool = False,
        return_lse: bool = False,
    ):
        out, lse = _flash_attn_fwd(q, k, v, softmax_scale=softmax_scale, return_lse=return_lse)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.deterministic = deterministic
        ctx.return_lse = return_lse
        ctx.set_materialize_grads(False)
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        q, k, v, out, lse = ctx.saved_tensors
        if not ctx.return_lse:
            dlse = None
        if dout is None:
            dout = torch.zeros_like(out)
        dq, dk, dv = _flash_attn_bwd(
            q, k, v, out, dout, lse,
            softmax_scale=ctx.softmax_scale,
            deterministic=ctx.deterministic,
            dlse=dlse,
        )
        return dq, dk, dv, None, None, None


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    deterministic: bool = False,
    return_lse: bool = False,
):
    return FlashAttnFunc.apply(q, k, v, softmax_scale, deterministic, return_lse)


# Backward-compat alias: tests/benchmarks that imported flash_attn_varlen_func will fail loudly.
def flash_attn_varlen_func(*args, **kwargs):
    raise NotImplementedError(
        "flash_attn_varlen_func was removed in the B200 simplified build. "
        "Use flash_attn_func with fixed-length inputs."
    )
