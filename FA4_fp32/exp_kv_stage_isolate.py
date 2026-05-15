"""Isolate the kv_stage-depth effect on the no-bias fp32 forward.

Compares the bias-free fp32 FA4 kernel at:
  (a) natural kv_stage (= (224 - smem_q_o) / smem_kv), and
  (b) kv_stage forced to 2 via a subclass cap.

Both runs use tile_n=64 to match the bias kernel's geometry, so the only
thing changing is KV pipeline depth. Combined with the prior fp32-vs-bf16
bias isolation, this triangulates how much of the bias slowdown is from
kv_stage shrinkage alone, separately from bias staging cost and bias HBM.
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
from FA4_fp32.kernels.flash_fwd_sm100 import FlashAttentionForwardSm100


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class NoBiasKvCapped(FlashAttentionForwardSm100):
    KV_STAGE_CAP = 2

    def _setup_attributes(self):
        super()._setup_attributes()
        self._natural_kv_stage = self.kv_stage
        self.kv_stage = min(self.kv_stage, self.KV_STAGE_CAP)


_compile_cache: dict = {}


def _run(kernel_cls, q, k, v, out, lse, scale, tile_n, q_stage):
    dtype = cutlass.Float32
    head_dim = q.shape[-1]
    key = (
        kernel_cls.__name__, dtype, head_dim, tile_n, q_stage,
        get_broadcast_dims(q), get_broadcast_dims(k),
        get_broadcast_dims(v),
    )
    if key not in _compile_cache:
        qt, kt, vt, ot = [to_cute_tensor(t) for t in (q, k, v, out)]
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
            kernel, qt, kt, vt, ot, lt, scale,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _compile_cache[key](q, k, v, out, lse, scale)


def bench_one(kernel_cls, args, L):
    device = "cuda"
    scale = 1.0 / math.sqrt(args.D)
    total_batch = args.A * args.B
    q = torch.randn(total_batch, L, args.H, args.D, device=device, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = torch.empty_like(q)
    lse = torch.empty(total_batch, args.H, L, device=device, dtype=torch.float32)

    def call():
        _run(kernel_cls, q, k, v, out, lse, scale, args.tile_n, args.q_stage)
        return out

    call()
    torch.cuda.synchronize()
    t = do_bench(call, warmup=args.warmup, rep=args.rep)
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
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--rep", type=int, default=20)
    p.add_argument("--out-txt", type=Path,
                   default=Path("bench_results/exp_kv_stage_isolate.txt"))
    args = p.parse_args()
    args.out_txt.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "# Bias-free fp32 FA4 fwd, A=48 B=1 H=16 D=64 tile_n=64 q_stage=2.",
        "# natural kv_stage = (224KB - smem_q_o) / smem_kv = (224-128)/16 = 6.",
        "# capped  kv_stage = 2 (matches what the SMEM-staged bias kernel runs at).",
        f"{'L':>6} {'natural ms':>11} {'capped ms':>11} "
        f"{'natural TF':>11} {'capped TF':>11} {'capped/natural':>15}",
        "-" * 80,
    ]
    print("\n".join(header))
    lines = list(header)
    for L in args.L:
        t_nat, tf_nat = bench_one(FlashAttentionForwardSm100, args, L)
        t_cap, tf_cap = bench_one(NoBiasKvCapped, args, L)
        ratio = t_cap / t_nat
        line = (
            f"{L:>6} {t_nat:>11.3f} {t_cap:>11.3f} "
            f"{tf_nat:>11.1f} {tf_cap:>11.1f} {ratio:>14.3f}x"
        )
        print(line)
        lines.append(line)
        args.out_txt.write_text("\n".join(lines) + "\n")
        torch.cuda.empty_cache()
    print(f"saved: {args.out_txt}")


if __name__ == "__main__":
    main()
