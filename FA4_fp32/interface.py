# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# B200-only, minimal happy-path flash attention.
#
# Supported:
# - dtype fp16 / bf16 / fp32 (TF32 MMA)
# - SM100 only (Blackwell B200)
# - MHA only (Q/K/V share shape: same seqlen, same num_heads, same head_dim)
# - fixed-length, non-causal, head_dim in [8..128] multiple of 16 // element_size
# - forward + backward, return_lse

import os
import math
from typing import Optional, Tuple

import torch

import cuda.bindings.driver as cuda  # noqa: F401

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from quack.compile_utils import make_fake_tensor as fake_tensor

from FA4_fp32.infra.cache_utils import get_jit_cache
from FA4_fp32.infra.testing import is_fake_mode


if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    from FA4_fp32.infra import cute_dsl_ptxas  # noqa: F401
    cute_dsl_ptxas.patch()


from FA4_fp32.infra import fa_logging  # noqa: F401
from FA4_fp32.infra.cute_dsl_utils import to_cute_tensor, get_broadcast_dims
from FA4_fp32.kernels.flash_fwd_sm100 import FlashAttentionForwardSm100
from FA4_fp32.kernels.flash_bwd_preprocess import FlashAttentionBackwardPreprocess
from FA4_fp32.kernels.flash_bwd_sm100 import FlashAttentionBackwardSm100
from FA4_fp32.kernels.flash_bwd_postprocess import FlashAttentionBackwardPostprocess


_SUPPORTED_ARCHES = (100,)  # B100/B200 only


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


def _validate_head_dim(head_dim: int, alignment: int) -> None:
    assert 8 <= head_dim <= 128 and head_dim % alignment == 0, (
        f"head_dim={head_dim} not supported. Must be 8..128 divisible by {alignment}."
    )


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
    assert q.shape == k.shape == v.shape, "MHA only: Q/K/V must have identical shape"
    assert q.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        "inputs must be float16, bfloat16, or float32 (TF32 MMA)"
    )
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
    if not is_fake_mode():
        assert all(t.is_cuda for t in (q, k, v)), "inputs must be on CUDA device"

    arch = _get_device_arch()
    assert arch in _SUPPORTED_ARCHES, f"Unsupported compute capability {arch}; only SM100 (B100/B200) supported"

    alignment = 16 // q.element_size()
    _validate_head_dim(head_dim, alignment)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    device = q.device
    out_torch_dtype = q.dtype
    requires_grad = q.requires_grad or k.requires_grad or v.requires_grad

    if out is None:
        out = torch.empty(batch_size, seqlen_q, num_head, head_dim,
                          dtype=out_torch_dtype, device=device)
    else:
        _validate_tensor(out, "out", (batch_size, seqlen_q, num_head, head_dim),
                         out_torch_dtype, device)

    lse_shape = (batch_size, num_head, seqlen_q)
    if lse is None:
        lse = torch.empty(lse_shape, dtype=torch.float32, device=device) if (requires_grad or return_lse) else None
    else:
        _validate_tensor(lse, "lse", lse_shape, torch.float32, device)

    dtype = torch2cute_dtype_map[q.dtype]

    # Fixed tile config. fp32 path is built only for head_dim <= 64 — that bound
    # keeps sQ+sO <= 128 KiB even with q_stage=2 + tile_n=128, leaving kv_stage>=3.
    if q.dtype == torch.float32:
        assert head_dim <= 64, "fp32 path is built for head_dim <= 64 only"
        tile_m, tile_n = 128, 128
        q_stage = 2 if seqlen_q > tile_m else 1
    else:
        tile_m, tile_n = 128, 128
        q_stage = 2 if seqlen_q > tile_m else 1
    # Optional overrides for benchmarking.
    _q_stage_env = os.environ.get("FA_Q_STAGE")
    if _q_stage_env is not None and seqlen_q > tile_m:
        q_stage = int(_q_stage_env)
    _tile_n_env = os.environ.get("FA_TILE_N")
    if _tile_n_env is not None:
        tile_n = int(_tile_n_env)

    use_clc_scheduler = os.environ.get("FA_CLC", "0") == "1"

    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compile_key = (
        dtype, head_dim, tile_m, tile_n, q_stage,
        use_clc_scheduler, lse is None,
        get_broadcast_dims(q), get_broadcast_dims(k), get_broadcast_dims(v),
        arch,
    )
    if compile_key not in _flash_attn_fwd.compile_cache:
        q_tensor, k_tensor, v_tensor, o_tensor = [to_cute_tensor(t) for t in (q, k, v, out)]
        lse_tensor = to_cute_tensor(lse, assumed_align=4) if lse is not None else None

        fa_fwd = FlashAttentionForwardSm100(
            head_dim,
            m_block_size=tile_m,
            n_block_size=tile_n,
            q_stage=q_stage,
            is_persistent=True,
            use_clc_scheduler=use_clc_scheduler,
        )
        _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
            fa_fwd,
            q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor,
            softmax_scale,
            current_stream,
            options="--enable-tvm-ffi",
        )

    if not is_fake_mode():
        _flash_attn_fwd.compile_cache[compile_key](
            q.detach(), k.detach(), v.detach(), out.detach(), lse,
            softmax_scale,
        )
    return out, lse


_flash_attn_fwd.compile_cache = get_jit_cache("fwd")


# ======================================================================
# Backward preprocess / postprocess helpers
# ======================================================================

def _make_fake_bwd_tensors(dtype):
    sym = cute.sym_int
    div = 128 // dtype.width
    b, seqlen_q, h, d = sym(), sym(), sym(), sym()
    seqlen_q_rounded = sym()
    mQ = fake_tensor(dtype, (b, seqlen_q, h, d), divisibility=div)
    mO = fake_tensor(dtype, (b, seqlen_q, h, d), divisibility=div)
    mdO = fake_tensor(dtype, (b, seqlen_q, h, d), divisibility=div)
    mK = fake_tensor(dtype, (b, seqlen_q, h, d), divisibility=div)
    mV = fake_tensor(dtype, (b, seqlen_q, h, d), divisibility=div)
    mdQ = fake_tensor(dtype, (b, seqlen_q, h, d), divisibility=div)
    mdK = fake_tensor(dtype, (b, seqlen_q, h, d), divisibility=div)
    mdV = fake_tensor(dtype, (b, seqlen_q, h, d), divisibility=div)
    mLSE = fake_tensor(Float32, (b, h, seqlen_q), divisibility=1)
    mLSElog2 = fake_tensor(Float32, (b, h, seqlen_q_rounded), divisibility=4)
    mPdPsum = fake_tensor(Float32, (b, h, seqlen_q_rounded), divisibility=4)
    mdQaccum = fake_tensor(Float32, (b, h, sym()), divisibility=4)
    return mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, mdQaccum


def _compile_bwd_preprocess(dtype, head_dim, m_block_size, has_dlse):
    mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, mdQaccum = _make_fake_bwd_tensors(dtype)
    mdLSE = fake_tensor(Float32, mLSE.shape, divisibility=1) if has_dlse else None
    fa_bwd_pre = FlashAttentionBackwardPreprocess(dtype, head_dim, m_block_size)
    return cute.compile(
        fa_bwd_pre, mO, mdO, mPdPsum, mLSE, mLSElog2, mdQaccum, mdLSE,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_preprocess(out, dout, dpsum, lse, lse_log2, dq_accum, dlse,
                    dtype, head_dim, m_block_size):
    compile_key = (dtype, head_dim, m_block_size, dlse is not None)
    if compile_key not in _bwd_preprocess.compile_cache:
        _bwd_preprocess.compile_cache[compile_key] = _compile_bwd_preprocess(*compile_key)
    if not is_fake_mode():
        _bwd_preprocess.compile_cache[compile_key](
            out, dout, dpsum, lse, lse_log2, dq_accum, dlse
        )


_bwd_preprocess.compile_cache = get_jit_cache("bwd_pre")


def _compile_bwd_postprocess(dtype, hdim, block_size, num_threads):
    mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, mdQaccum = _make_fake_bwd_tensors(dtype)
    fa_bwd_post = FlashAttentionBackwardPostprocess(dtype, hdim, block_size, num_threads)
    return cute.compile(
        fa_bwd_post, mdQaccum, mdQ, Float32(0.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_postprocess_convert(accum, output, scale, dtype, hdim, block_size, num_threads):
    compile_key = (dtype, hdim, block_size, num_threads)
    if compile_key not in _bwd_postprocess_convert.compile_cache:
        _bwd_postprocess_convert.compile_cache[compile_key] = _compile_bwd_postprocess(*compile_key)
    if not is_fake_mode():
        _bwd_postprocess_convert.compile_cache[compile_key](accum, output, scale)


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
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    dlse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    arch = _get_device_arch()
    assert arch in _SUPPORTED_ARCHES, f"Unsupported compute capability {arch}; only SM100 (B100/B200) supported"

    num_head, head_dim = q.shape[-2:]

    # Fixed tile sizes. fp32 uses a smaller m tile due to higher smem cost.
    if q.dtype == torch.float32:
        m_block_size = 64
    else:
        m_block_size = 128
    n_block_size = 128
    num_threads = 384

    q, k, v, out, dout, lse = [maybe_contiguous(t) for t in (q, k, v, out, dout, lse)]
    batch_size, seqlen_q = q.shape[:2]
    assert q.shape == k.shape == v.shape, "MHA only: Q/K/V must have identical shape"

    seqlen_q_rounded = (seqlen_q + m_block_size - 1) // m_block_size * m_block_size
    seqlen_k = k.shape[1]
    seqlen_k_rounded = (seqlen_k + n_block_size - 1) // n_block_size * n_block_size

    assert out.shape == q.shape
    assert dout.shape == q.shape
    assert lse.shape == (batch_size, num_head, seqlen_q)
    assert q.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype
    assert lse.dtype == torch.float32

    if dlse is not None:
        dlse = maybe_contiguous(dlse)
    if not is_fake_mode():
        assert all(t.is_cuda for t in (q, k, v, out, dout, lse))

    alignment = 16 // q.element_size()
    _validate_head_dim(head_dim, alignment)
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

    _bwd_preprocess(out, dout, dpsum, lse, lse_log2, dq_accum, dlse,
                    dtype, head_dim, m_block_size)

    compile_key = (
        arch, dtype, head_dim,
        m_block_size, n_block_size, num_threads,
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

        fa_bwd_obj = FlashAttentionBackwardSm100(
            head_dim,
            tile_m=m_block_size,
            tile_n=n_block_size,
            subtile_factor=2,
            q_dtype=dtype,
        )

        _flash_attn_bwd.compile_cache[compile_key] = cute.compile(
            fa_bwd_obj,
            q_tensor, k_tensor, v_tensor, do_tensor,
            lse_log2_tensor, dpsum_tensor, dq_accum_tensor,
            dk_tensor, dv_tensor,
            softmax_scale,
            current_stream,
            options="--enable-tvm-ffi",
        )

    if not is_fake_mode():
        _flash_attn_bwd.compile_cache[compile_key](
            q.detach(), k.detach(), v.detach(), dout,
            lse_log2, dpsum, dq_accum, dk, dv,
            softmax_scale,
        )

    # Postprocess: convert dq_accum (float32) → dq (orig dtype) with softmax_scale.
    _bwd_postprocess_convert(
        dq_accum, dq, softmax_scale,
        dtype, head_dim, m_block_size, 128,
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
        return_lse: bool = False,
    ):
        out, lse = _flash_attn_fwd(q, k, v, softmax_scale=softmax_scale, return_lse=return_lse)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
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
            dlse=dlse,
        )
        return dq, dk, dv, None, None


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
):
    return FlashAttnFunc.apply(q, k, v, softmax_scale, return_lse)
