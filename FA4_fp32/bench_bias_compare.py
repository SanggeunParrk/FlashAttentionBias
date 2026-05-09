"""Benchmark fp32 FA4, bf16 FA4, and fp32+bias FA4 forward kernels.

The bias kernel uses q/k/v/o shape (A, B, L, H, D) and
bias shape (1, B, L, H, L).  For comparison with the existing B=48
benchmarks, the default maps fp32/bf16 to batch=48 and fp32_bias to
A=48, B=1, i.e. the same total number of attention problems.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from triton.testing import do_bench

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FA4_fp32.interface import _flash_attn_fwd
from FA4_fp32.verify_bias_smem import _compile_and_run as _run_bias_fwd


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def flops_fwd(total_batch: int, heads: int, seqlen: int, head_dim: int) -> float:
    return 4.0 * total_batch * heads * seqlen * seqlen * head_dim


def _bench_one(args, seqlen: int) -> dict[str, float]:
    device = "cuda"
    scale = 1.0 / math.sqrt(args.D)
    total_batch = args.A * args.B

    q32 = torch.randn(total_batch, seqlen, args.H, args.D, device=device, dtype=torch.float32)
    k32 = torch.randn_like(q32)
    v32 = torch.randn_like(q32)
    o32 = torch.empty_like(q32)
    lse32 = torch.empty(total_batch, args.H, seqlen, device=device, dtype=torch.float32)

    q16 = q32.bfloat16()
    k16 = k32.bfloat16()
    v16 = v32.bfloat16()
    o16 = torch.empty_like(q16)
    lse16 = torch.empty_like(lse32)

    q_bias = q32.view(args.A, args.B, seqlen, args.H, args.D)
    k_bias = k32.view(args.A, args.B, seqlen, args.H, args.D)
    v_bias = v32.view(args.A, args.B, seqlen, args.H, args.D)
    o_bias = torch.empty_like(q_bias)
    lse_bias = torch.empty(args.A, args.B, args.H, seqlen, device=device, dtype=torch.float32)
    bias = torch.randn(
        1, args.B, seqlen, args.H, seqlen, device=device, dtype=torch.float32
    ) * args.bias_scale

    def fa4_fp32():
        _flash_attn_fwd(
            q32, k32, v32, softmax_scale=scale, return_lse=True, out=o32, lse=lse32
        )
        return o32

    def fa4_bf16():
        _flash_attn_fwd(
            q16, k16, v16, softmax_scale=scale, return_lse=True, out=o16, lse=lse16
        )
        return o16

    def fa4_fp32_bias():
        _run_bias_fwd(
            q_bias,
            k_bias,
            v_bias,
            bias,
            o_bias,
            lse_bias,
            scale,
            args.tile_n,
            args.q_stage,
        )
        return o_bias

    fa4_fp32()
    fa4_bf16()
    fa4_fp32_bias()
    torch.cuda.synchronize()

    t_fp32 = do_bench(fa4_fp32, warmup=args.warmup, rep=args.rep)
    t_bf16 = do_bench(fa4_bf16, warmup=args.warmup, rep=args.rep)
    t_bias = do_bench(fa4_fp32_bias, warmup=args.warmup, rep=args.rep)
    torch.cuda.synchronize()

    f = flops_fwd(total_batch, args.H, seqlen, args.D)
    return {
        "fp32_ms": t_fp32,
        "bf16_ms": t_bf16,
        "bias_ms": t_bias,
        "fp32_tflops": f / (t_fp32 * 1e9),
        "bf16_tflops": f / (t_bf16 * 1e9),
        "bias_tflops": f / (t_bias * 1e9),
        "bias_over_fp32": t_bias / t_fp32,
        "bias_over_bf16": t_bias / t_bf16,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--A", type=int, default=48)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--H", type=int, default=16)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--L", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--tile-n", type=int, default=64)
    parser.add_argument("--q-stage", type=int, default=2)
    parser.add_argument("--bias-scale", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=40)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("bench_results/bench_fp32_bf16_fp32bias_B48_H16D64.txt"),
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# fp32/bf16 = FA4_fp32 fwd on shape (Btot,L,H,D).",
        "# fp32_bias = SMEM bias fwd on q/k/v/o (A,B,L,H,D), bias (1,B,L,H,L).",
        f"# A={args.A} B={args.B} Btot={args.A * args.B} H={args.H} D={args.D} "
        f"bias_tile_n={args.tile_n} q_stage={args.q_stage}.",
        "# TFLOPS uses 4*Btot*H*L^2*D and does not count bias load/add as extra FLOPs.",
        f"{'A':>3} {'B':>2} {'Btot':>5} {'H':>3} {'L':>6} {'D':>4} "
        f"{'fp32 ms':>10} {'bf16 ms':>10} {'fp32_bias ms':>13} "
        f"{'fp32 TF':>10} {'bf16 TF':>10} {'bias TF':>10} "
        f"{'bias/fp32':>11} {'bias/bf16':>11}",
        "-" * 131,
    ]

    print("\n".join(lines))
    for seqlen in args.L:
        result = _bench_one(args, seqlen)
        line = (
            f"{args.A:>3} {args.B:>2} {args.A * args.B:>5} {args.H:>3} "
            f"{seqlen:>6} {args.D:>4} "
            f"{result['fp32_ms']:>10.3f} {result['bf16_ms']:>10.3f} "
            f"{result['bias_ms']:>13.3f} "
            f"{result['fp32_tflops']:>10.1f} {result['bf16_tflops']:>10.1f} "
            f"{result['bias_tflops']:>10.1f} "
            f"{result['bias_over_fp32']:>10.2f}x {result['bias_over_bf16']:>10.2f}x"
        )
        print(line)
        lines.append(line)
        args.out.write_text("\n".join(lines) + "\n")
        torch.cuda.empty_cache()

    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
