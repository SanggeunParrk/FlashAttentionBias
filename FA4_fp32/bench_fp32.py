"""Benchmark fp32 FA4 vs bf16 FA4 vs fp32 PyTorch reference (non-causal).

Both fp32 sides use TF32 tensor cores internally; bf16 FA4 uses fp16/bf16 MMA.
D=48 is currently unsupported in fp32 FA4 (TF32 swizzle constraint).
"""
import sys
sys.path.insert(0, "/home/snu_hwle/psk/kernels/flash-attention")

import torch
from triton.testing import do_bench

from FA4_fp32 import flash_attn_func, attention_fp32

torch.backends.cuda.matmul.allow_tf32 = True


def flops_fwd(B, H, L, D):
    return 4.0 * B * H * L * L * D


def bench(B, H, L, D):
    q = torch.randn(B, L, H, D, device="cuda", dtype=torch.float32)
    k = torch.randn_like(q); v = torch.randn_like(q)
    q16, k16, v16 = q.bfloat16(), k.bfloat16(), v.bfloat16()

    def fa4_fp32():
        return flash_attn_func(q, k, v)[0]

    def fa4_bf16():
        return flash_attn_func(q16, k16, v16)[0]

    def ref():
        return attention_fp32(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)).squeeze(0)

    fa4_fp32(); fa4_bf16(); ref(); torch.cuda.synchronize()
    t32 = do_bench(fa4_fp32, warmup=10, rep=40)
    t16 = do_bench(fa4_bf16, warmup=10, rep=40)
    t_ref = do_bench(ref, warmup=10, rep=40)
    f = flops_fwd(B, H, L, D)
    return {
        "fa32_ms": t32, "fa16_ms": t16, "ref_ms": t_ref,
        "fa32_tflops": f / (t32 * 1e9),
        "fa16_tflops": f / (t16 * 1e9),
        "ref_tflops":  f / (t_ref * 1e9),
    }


CONFIGS = [(2, 8, L, D) for D in [32, 64, 96, 128] for L in [1024, 2048, 4096, 8192]]

print(f"{'B':>2} {'H':>2} {'L':>6} {'D':>4} "
      f"{'FA fp32':>10} {'FA bf16':>10} {'ref':>10} "
      f"{'fp32 TF':>10} {'bf16 TF':>10} {'ref TF':>10} "
      f"{'bf16/fp32':>10}")
print("-" * 100)
for B, H, L, D in CONFIGS:
    r = bench(B, H, L, D)
    ratio = r["fa32_ms"] / r["fa16_ms"]
    print(f"{B:>2} {H:>2} {L:>6} {D:>4} "
          f"{r['fa32_ms']:>10.3f} {r['fa16_ms']:>10.3f} {r['ref_ms']:>10.3f} "
          f"{r['fa32_tflops']:>10.1f} {r['fa16_tflops']:>10.1f} {r['ref_tflops']:>10.1f} "
          f"{ratio:>9.2f}x")
