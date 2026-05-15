"""Decompose the apply_bias_smem cost into SMEM-read vs fma portions.

Three fake variants share the same kernel except for the body of
`apply_bias_smem`. Bias TMA and pipeline_bias consumer_wait/release are
unchanged in all of them, so the bias TMA → SMEM dependency stays on the
critical path.

  - noop      : skip SMEM read AND skip fma  (already covered by exp_bias_load_only)
  - read_only : do the SMEM read (with coord lookup + Float32 cast), accumulate
                into a register sink and add the sink into tSrS[0] so the read
                isn't DCE'd. Skip fma.
  - fma_only  : skip SMEM read; run fma_packed_f32x2 with bias = (0,0) register
                constants. Same packed-fma instruction count as the real kernel.

Comparing the three (plus baseline no-bias and real bias) splits the
+135% softmax-side overhead into SMEM-side vs fma-side.

Outputs are wrong; timing only.
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
from cutlass import Float32

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FA4_fp32.infra.cute_dsl_utils import to_cute_tensor, get_broadcast_dims
from FA4_fp32.kernels.flash_bias_fwd_sm100_smem import (
    FlashAttentionBiasForwardSm100Smem,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class BiasReadOnly(FlashAttentionBiasForwardSm100Smem):
    """Read sBias from SMEM but do not fma. Sink is added into tSrS[0] to
    keep the SMEM read on a live dependency chain."""

    @cute.jit
    def apply_bias_smem(self, tSrS_t2r, tScS_t2r, sBias, stage, softmax_scale):
        sink = Float32(0.0)
        for i in cutlass.range(0, cute.size(tSrS_t2r.shape), 2, unroll_full=True):
            q0 = tScS_t2r[i][0]
            k0 = tScS_t2r[i][1]
            q1 = tScS_t2r[i + 1][0]
            k1 = tScS_t2r[i + 1][1]
            b0 = Float32(sBias[q0, k0, stage])
            b1 = Float32(sBias[q1, k1, stage])
            sink = sink + b0 + b1
        tSrS_t2r[0] = tSrS_t2r[0] + sink


class BiasFmaOnly(FlashAttentionBiasForwardSm100Smem):
    """No SMEM read; run fma_packed_f32x2 with bias=(0,0) register constants.
    Same packed-fma instruction count as the real kernel."""

    @cute.jit
    def apply_bias_smem(self, tSrS_t2r, tScS_t2r, sBias, stage, softmax_scale):
        b0 = Float32(0.0)
        b1 = Float32(0.0)
        for i in cutlass.range(0, cute.size(tSrS_t2r.shape), 2, unroll_full=True):
            tSrS_t2r[i], tSrS_t2r[i + 1] = cute.arch.fma_packed_f32x2(
                (tSrS_t2r[i], tSrS_t2r[i + 1]),
                (softmax_scale, softmax_scale),
                (b0, b1),
            )


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
                   default=Path("bench_results/exp_bias_apply_decompose.txt"))
    args = p.parse_args()
    args.out_txt.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "# Decompose softmax-side bias overhead into SMEM-read vs fma.",
        "# Bias TMA + pipeline_bias consumer_wait/release are unchanged in all variants.",
        f"# A={args.A} B={args.B} H={args.H} D={args.D} tile_n={args.tile_n} q_stage={args.q_stage}",
        "#",
        "# Reference (from earlier runs, same shape):",
        "#   no-bias  kv_stage=2 (cap)    : see exp_kv_stage_isolate.txt 'capped'",
        "#   TMA-only (apply=noop)        : see exp_bias_load_only.txt",
        "#   real bias fp32 (apply=full)  : see exp_bias_hbm_isolate.txt 'fp32_bias'",
        "#",
        f"{'L':>6} {'read_only ms':>13} {'fma_only ms':>13} "
        f"{'read_only TF':>13} {'fma_only TF':>13}",
        "-" * 80,
    ]
    print("\n".join(header))
    lines = list(header)
    for L in args.L:
        t_r, tf_r = bench_one(BiasReadOnly, args, L)
        t_f, tf_f = bench_one(BiasFmaOnly, args, L)
        line = (
            f"{L:>6} {t_r:>13.3f} {t_f:>13.3f} "
            f"{tf_r:>13.1f} {tf_f:>13.1f}"
        )
        print(line)
        lines.append(line)
        args.out_txt.write_text("\n".join(lines) + "\n")
        torch.cuda.empty_cache()
    print(f"saved: {args.out_txt}")


if __name__ == "__main__":
    main()
