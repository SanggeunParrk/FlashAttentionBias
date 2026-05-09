"""Forward validation for the SMEM-staged score-bias SM100 kernel.

Default target shape is the requested A=48, B=1, H=16, D=64.  Pass --seqlen to
choose the current L under test.  The exact reference is materialized only when
the score tensor is reasonably small; otherwise use --sample-rows for a bounded
row-sampled reference.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

import cutlass
import cutlass.cute as cute

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FA4_fp32.infra.cute_dsl_utils import to_cute_tensor, get_broadcast_dims
from FA4_fp32.kernels.flash_bias_fwd_sm100_smem import (
    FlashAttentionBiasForwardSm100Smem,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _ref_full(q, k, v, bias, scale):
    q_ = q.permute(0, 1, 3, 2, 4).contiguous()
    k_ = k.permute(0, 1, 3, 2, 4).contiguous()
    v_ = v.permute(0, 1, 3, 2, 4).contiguous()
    b_ = bias[0].permute(0, 2, 1, 3).contiguous()
    scores = torch.matmul(q_, k_.transpose(-2, -1)) * scale + b_.unsqueeze(0)
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v_)
    lse = torch.logsumexp(scores, dim=-1)
    return out.permute(0, 1, 3, 2, 4).contiguous(), lse


def _ref_rows(q, k, v, bias, scale, rows):
    A, B, _, H, D = q.shape
    out = torch.empty((A, B, len(rows), H, D), dtype=q.dtype, device=q.device)
    lse = torch.empty((A, B, H, len(rows)), dtype=torch.float32, device=q.device)
    for ai in range(A):
        for bi in range(B):
            for hi in range(H):
                q_sel = q[ai, bi, rows, hi, :].float()
                k_sel = k[ai, bi, :, hi, :].float()
                v_sel = v[ai, bi, :, hi, :].float()
                b_sel = bias[0, bi, rows, hi, :].float()
                scores = q_sel @ k_sel.T * scale + b_sel
                probs = torch.softmax(scores, dim=-1)
                out[ai, bi, :, hi, :] = probs @ v_sel
                lse[ai, bi, hi, :] = torch.logsumexp(scores, dim=-1)
    return out, lse


def _compile_and_run(q, k, v, bias, out, lse, scale, tile_n, q_stage):
    dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }[q.dtype]
    head_dim = q.shape[-1]
    use_clc_scheduler = False
    compile_key = (
        dtype,
        bias.dtype,
        head_dim,
        tile_n,
        q_stage,
        get_broadcast_dims(q),
        get_broadcast_dims(k),
        get_broadcast_dims(v),
        get_broadcast_dims(bias),
    )
    if compile_key not in _compile_and_run.cache:
        q_tensor, k_tensor, v_tensor, bias_tensor, o_tensor = [
            to_cute_tensor(t) for t in (q, k, v, bias, out)
        ]
        lse_tensor = to_cute_tensor(lse, assumed_align=4)
        kernel = FlashAttentionBiasForwardSm100Smem(
            head_dim,
            m_block_size=128,
            n_block_size=tile_n,
            q_stage=q_stage,
            is_persistent=True,
            use_clc_scheduler=use_clc_scheduler,
        )
        _compile_and_run.cache[compile_key] = cute.compile(
            kernel,
            q_tensor,
            k_tensor,
            v_tensor,
            bias_tensor,
            o_tensor,
            lse_tensor,
            scale,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _compile_and_run.cache[compile_key](q, k, v, bias, out, lse, scale)


_compile_and_run.cache = {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--A", type=int, default=48)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--L", "--seqlen", type=int, default=1024)
    parser.add_argument("--H", type=int, default=16)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--tile-n", type=int, default=64)
    parser.add_argument("--q-stage", type=int, default=2)
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"
    shape = (args.A, args.B, args.L, args.H, args.D)
    bias_shape = (1, args.B, args.L, args.H, args.L)
    q = torch.randn(shape, device=device, dtype=torch.float32)
    k = torch.randn(shape, device=device, dtype=torch.float32)
    v = torch.randn(shape, device=device, dtype=torch.float32)
    bias = torch.randn(bias_shape, device=device, dtype=torch.float32) * 0.1
    out = torch.empty_like(q)
    lse = torch.empty((args.A, args.B, args.H, args.L), device=device, dtype=torch.float32)
    scale = 1.0 / math.sqrt(args.D)

    _compile_and_run(q, k, v, bias, out, lse, scale, args.tile_n, args.q_stage)
    torch.cuda.synchronize()

    score_elems = args.A * args.B * args.H * args.L * args.L
    if args.sample_rows > 0:
        rows = torch.linspace(0, args.L - 1, args.sample_rows, device=device).long()
        ref_out, ref_lse = _ref_rows(q, k, v, bias, scale, rows)
        got_out = out[:, :, rows, :, :]
        got_lse = lse[:, :, :, rows]
    elif score_elems <= 2_000_000_000:
        ref_out, ref_lse = _ref_full(q, k, v, bias, scale)
        got_out = out
        got_lse = lse
    else:
        rows = torch.linspace(0, args.L - 1, 16, device=device).long()
        ref_out, ref_lse = _ref_rows(q, k, v, bias, scale, rows)
        got_out = out[:, :, rows, :, :]
        got_lse = lse[:, :, :, rows]

    out_err = (got_out - ref_out).abs()
    lse_err = (got_lse - ref_lse).abs()
    print(
        f"shape A={args.A} B={args.B} L={args.L} H={args.H} D={args.D} "
        f"tile_n={args.tile_n} q_stage={args.q_stage}"
    )
    print(f"out max={out_err.max().item():.6e} mean={out_err.mean().item():.6e}")
    print(f"lse max={lse_err.max().item():.6e} mean={lse_err.mean().item():.6e}")


if __name__ == "__main__":
    main()
