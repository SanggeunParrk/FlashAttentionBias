"""Isolate the bias HBM traffic contribution to the slowdown.

Both variants force kv_stage=2 (same as fp32-bias baseline's natural value), so
SMEM occupancy / pipeline depth are matched. The only thing changing across
variants is the bias element type (fp32 vs bf16), which halves bias HBM traffic
and bias SMEM bytes per stage but keeps the rest of the pipeline identical.

If bf16-bias is noticeably faster, the bottleneck has a meaningful HBM-traffic
component. If perf is essentially flat, then KV-pipeline-depth (kv_stage being
small) — not bias HBM — is the dominant cost.
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


class BiasSmemKvCapped(FlashAttentionBiasForwardSm100Smem):
    KV_STAGE_CAP = 2

    def _setup_attributes(self):
        super()._setup_attributes()
        natural = self.kv_stage
        self.kv_stage = min(self.kv_stage, self.KV_STAGE_CAP)
        self._natural_kv_stage = natural


_compile_cache: dict = {}


def _run(q, k, v, bias, out, lse, scale, tile_n, q_stage):
    dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }[q.dtype]
    head_dim = q.shape[-1]
    key = (
        dtype, bias.dtype, head_dim, tile_n, q_stage,
        get_broadcast_dims(q), get_broadcast_dims(k),
        get_broadcast_dims(v), get_broadcast_dims(bias),
    )
    if key not in _compile_cache:
        qt, kt, vt, bt, ot = [to_cute_tensor(t) for t in (q, k, v, bias, out)]
        lt = to_cute_tensor(lse, assumed_align=4)
        kernel = BiasSmemKvCapped(
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


def _ref_full(q, k, v, bias, scale):
    q_ = q.permute(0, 1, 3, 2, 4).float().contiguous()
    k_ = k.permute(0, 1, 3, 2, 4).float().contiguous()
    v_ = v.permute(0, 1, 3, 2, 4).float().contiguous()
    b_ = bias[0].permute(0, 2, 1, 3).float().contiguous()
    scores = torch.matmul(q_, k_.transpose(-2, -1)) * scale + b_.unsqueeze(0)
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v_)
    return out.permute(0, 1, 3, 2, 4).to(q.dtype)


def verify_one(A, B, L, H, D, tile_n, q_stage, bias_dtype, scale, seed):
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(seed)
    shape = (A, B, L, H, D)
    bs = (1, B, L, H, L)
    q = torch.randn(shape, device=device, dtype=torch.float32, generator=g)
    k = torch.randn(shape, device=device, dtype=torch.float32, generator=g)
    v = torch.randn(shape, device=device, dtype=torch.float32, generator=g)
    bias_fp32 = torch.randn(bs, device=device, dtype=torch.float32, generator=g) * 0.1
    bias = bias_fp32.to(bias_dtype)
    out = torch.empty_like(q)
    lse = torch.empty(A, B, H, L, device=device, dtype=torch.float32)
    _run(q, k, v, bias, out, lse, scale, tile_n, q_stage)
    torch.cuda.synchronize()
    ref = _ref_full(q, k, v, bias.float(), scale)
    return (out - ref).abs().max().item(), (out - ref).abs().mean().item()


def bench_one(args, L, bias_dtype):
    device = "cuda"
    scale = 1.0 / math.sqrt(args.D)
    q = torch.randn(args.A, args.B, L, args.H, args.D, device=device, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    bias_fp32 = torch.randn(1, args.B, L, args.H, L, device=device,
                            dtype=torch.float32) * args.bias_scale
    bias = bias_fp32.to(bias_dtype)
    out = torch.empty_like(q)
    lse = torch.empty(args.A, args.B, args.H, L, device=device, dtype=torch.float32)

    def call():
        _run(q, k, v, bias, out, lse, scale, args.tile_n, args.q_stage)
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
    p.add_argument("--verify-L", type=int, default=1024)
    p.add_argument("--out-txt", type=Path,
                   default=Path("bench_results/exp_bias_hbm_isolate.txt"))
    args = p.parse_args()
    args.out_txt.parent.mkdir(parents=True, exist_ok=True)

    scale = 1.0 / math.sqrt(args.D)
    print("# Verify (kv_stage capped to 2 for both):")
    print(f"{'bias_dtype':>12} {'out_max':>11} {'out_mean':>11}")
    for bd_label, bd in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        mx, mn = verify_one(1, 1, args.verify_L, args.H, args.D,
                            args.tile_n, args.q_stage, bd, scale, seed=123)
        print(f"{bd_label:>12} {mx:>11.3e} {mn:>11.3e}")
    print()

    header = (
        f"# A={args.A} B={args.B} H={args.H} D={args.D} tile_n={args.tile_n} "
        f"q_stage={args.q_stage} kv_stage_cap={BiasSmemKvCapped.KV_STAGE_CAP}",
        f"{'L':>6} {'fp32_bias ms':>13} {'bf16_bias ms':>13} "
        f"{'fp32 TF':>10} {'bf16 TF':>10} {'bf16/fp32':>10}",
        "-" * 70,
    )
    lines = list(header)
    print("\n".join(lines))
    for L in args.L:
        t_fp32, tf_fp32 = bench_one(args, L, torch.float32)
        t_bf16, tf_bf16 = bench_one(args, L, torch.bfloat16)
        ratio = t_bf16 / t_fp32
        line = (
            f"{L:>6} {t_fp32:>13.3f} {t_bf16:>13.3f} "
            f"{tf_fp32:>10.1f} {tf_bf16:>10.1f} {ratio:>9.3f}x"
        )
        print(line)
        lines.append(line)
        args.out_txt.write_text("\n".join(lines) + "\n")
        torch.cuda.empty_cache()
    print(f"saved: {args.out_txt}")


if __name__ == "__main__":
    main()
