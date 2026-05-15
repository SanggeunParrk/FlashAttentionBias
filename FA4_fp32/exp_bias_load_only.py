"""Fake kernel: bias TMA loads run normally, but softmax never reads them.

Goal: measure the cost of just the bias TMA / pipeline_bias side of the
staging path, separated from the softmax-side work (SMEM bias read +
fma_packed_f32x2). Combined with the prior numbers this splits the
+47 ms / +205% bias overhead into a TMA portion and a softmax-side portion.

Output is incorrect (bias is loaded but unused). Only the timing matters.

We override `apply_bias_smem` to a no-op. Producer/consumer pipeline_bias
mbarriers are still acquired/waited/released, so the TMA path stays in
the critical chain — the compiler can't DCE the load away.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from triton.testing import do_bench

import cutlass
import cutlass.cute as cute

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FA4_fp32.infra.cute_dsl_utils import to_cute_tensor, get_broadcast_dims
from FA4_fp32.kernels.flash_bias_fwd_sm100_smem import (
    FlashAttentionBiasForwardSm100Smem,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class FlashAttentionBiasForwardSm100LoadOnly(FlashAttentionBiasForwardSm100Smem):
    """Bias TMA still runs; softmax doesn't read sBias and doesn't fma."""

    @cute.jit
    def apply_bias_smem(self, tSrS_t2r, tScS_t2r, sBias, stage, softmax_scale):
        return


_compile_cache: dict = {}


def _run(kernel_cls, q, k, v, bias, out, lse, scale, tile_n, q_stage):
    head_dim = q.shape[-1]
    key = (
        kernel_cls.__name__, q.dtype, bias.dtype, head_dim, tile_n, q_stage,
        get_broadcast_dims(q), get_broadcast_dims(k),
        get_broadcast_dims(v), get_broadcast_dims(bias),
    )
    if key not in _compile_cache:
        qt, kt, vt, bt, ot = [to_cute_tensor(t) for t in (q, k, v, bias, out)]
        lt = to_cute_tensor(lse, assumed_align=4)
        kernel = kernel_cls(
            head_dim,
            m_block_size=128,
            n_block_size=tile_n,
            q_stage=q_stage,
            is_persistent=True,
            use_clc_scheduler=False,
        )
        _compile_cache[key] = cute.compile(
            kernel, qt, kt, vt, bt, ot, lt, scale,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _compile_cache[key](q, k, v, bias, out, lse, scale)


def bench_one(kernel_cls, args, L):
    device = "cuda"
    scale = 1.0 / math.sqrt(args.D)
    q = torch.randn(args.A, args.B, L, args.H, args.D, device=device, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    bias = torch.randn(1, args.B, L, args.H, L, device=device,
                       dtype=torch.float32) * args.bias_scale
    out = torch.empty_like(q)
    lse = torch.empty(args.A, args.B, args.H, L, device=device, dtype=torch.float32)

    def call():
        _run(kernel_cls, q, k, v, bias, out, lse, scale, args.tile_n, args.q_stage)
        return out

    call()
    torch.cuda.synchronize()
    t = do_bench(call, warmup=args.warmup, rep=args.rep)
    total_batch = args.A * args.B
    f = 4.0 * total_batch * args.H * L * L * args.D
    return t, f / (t * 1e9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--A", type=int, default=48)
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--H", type=int, default=16)
    p.add_argument("--D", type=int, default=64)
    p.add_argument("--L", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    p.add_argument("--tile-n", type=int, default=64)
    p.add_argument("--q-stage", type=int, default=2)
    p.add_argument("--bias-scale", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--rep", type=int, default=20)
    p.add_argument("--out-txt", type=Path,
                   default=Path("bench_results/exp_bias_load_only.txt"))
    args = p.parse_args()
    args.out_txt.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "# Fake kernel: bias TMA runs (with pipeline_bias mbarriers); softmax skips apply_bias_smem.",
        "# Output is wrong; timing only.",
        f"# A={args.A} B={args.B} H={args.H} D={args.D} tile_n={args.tile_n} q_stage={args.q_stage}",
        "#",
        "# Reference points (from earlier experiments, same shape):",
        "#   no-bias  kv_stage=2 (cap): see exp_kv_stage_isolate.txt 'capped'",
        "#   fp32-bias kv_stage=2     : see exp_bias_hbm_isolate.txt  'fp32_bias'",
        "#",
        f"{'L':>6} {'load_only ms':>13} {'load_only TF':>13}",
        "-" * 50,
    ]
    print("\n".join(header))
    lines = list(header)
    for L in args.L:
        t, tf = bench_one(FlashAttentionBiasForwardSm100LoadOnly, args, L)
        line = f"{L:>6} {t:>13.3f} {tf:>13.1f}"
        print(line)
        lines.append(line)
        args.out_txt.write_text("\n".join(lines) + "\n")
        torch.cuda.empty_cache()
    print(f"saved: {args.out_txt}")


if __name__ == "__main__":
    main()
