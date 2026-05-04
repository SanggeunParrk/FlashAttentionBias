"""Benchmark persistent vs non-persistent fwd for the user's workload.

Compares two settings of FlashAttentionForwardSm100:
  - persistent  (default; StaticPersistentTileScheduler, SM-count CTAs)
  - non-persistent (SingleTileScheduler, one CTA per tile)

The toggle is gated by the temporary FA_NO_PERSISTENT env var that
interface.py reads (revert that change after benchmarking).

Workload sweep targets the user's regime: B=48, seqlen ~1k–4k,
H ∈ {8, 16}, D ∈ {64, 128}, dtype ∈ {bf16, fp32}, non-causal.
"""
import os
import sys
import gc

import torch
from triton.testing import do_bench

sys.path.insert(0, "/home/FlashAttentionBias")


def flops_fwd(B, H, L, D):
    return 4.0 * B * H * L * L * D


def run_one(B, H, L, D, dtype, persistent):
    """Build inputs, call fwd, do_bench. Caller flips FA_NO_PERSISTENT."""
    # Reload FA4 with the current env-var setting (compile_cache key includes
    # is_persistent; the second build picks up the new setting).
    from FA4_fp32 import flash_attn_func

    q = torch.randn(B, L, H, D, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def fa():
        return flash_attn_func(q, k, v)[0]

    fa()  # warm up + JIT compile
    torch.cuda.synchronize()
    t = do_bench(fa, warmup=10, rep=40)

    del q, k, v
    gc.collect()
    torch.cuda.empty_cache()
    return t


def bench_pair(B, H, L, D, dtype):
    os.environ["FA_NO_PERSISTENT"] = "0"
    t_p = run_one(B, H, L, D, dtype, persistent=True)
    os.environ["FA_NO_PERSISTENT"] = "1"
    t_np = run_one(B, H, L, D, dtype, persistent=False)
    return t_p, t_np


def fmt_dtype(d):
    return {torch.bfloat16: "bf16", torch.float32: "fp32"}[d]


def main():
    assert torch.cuda.is_available(), "need CUDA"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"SM count: {torch.cuda.get_device_properties(0).multi_processor_count}")

    B = 48
    HS = [8, 16]
    LS = [1024, 2048, 4096]
    DS = [64, 128]
    DTYPES = [torch.bfloat16, torch.float32]

    rows = []
    header = (
        f"{'dtype':>5} {'B':>3} {'H':>3} {'L':>5} {'D':>4} "
        f"{'tiles':>6} {'waves':>6} "
        f"{'pers ms':>9} {'non-p ms':>9} "
        f"{'pers TF':>9} {'non-p TF':>9} {'speedup':>8}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    for dtype in DTYPES:
        for H in HS:
            for L in LS:
                for D in DS:
                    # tile_m is 128 in both fp32 and bf16 fwd
                    tiles = B * H * ((L + 127) // 128)
                    waves = (tiles + sm_count - 1) // sm_count
                    try:
                        t_p, t_np = bench_pair(B, H, L, D, dtype)
                    except Exception as e:
                        line = (
                            f"{fmt_dtype(dtype):>5} {B:>3} {H:>3} {L:>5} {D:>4} "
                            f"{tiles:>6} {waves:>6}  FAILED: {type(e).__name__}: "
                            f"{str(e).splitlines()[0][:80]}"
                        )
                        print(line)
                        rows.append(line)
                        continue
                    f = flops_fwd(B, H, L, D)
                    tf_p = f / (t_p * 1e9)
                    tf_np = f / (t_np * 1e9)
                    speedup = t_np / t_p  # >1 means persistent is faster
                    line = (
                        f"{fmt_dtype(dtype):>5} {B:>3} {H:>3} {L:>5} {D:>4} "
                        f"{tiles:>6} {waves:>6} "
                        f"{t_p:>9.3f} {t_np:>9.3f} "
                        f"{tf_p:>9.1f} {tf_np:>9.1f} {speedup:>7.3f}x"
                    )
                    print(line, flush=True)
                    rows.append(line)

    out_path = "/home/FlashAttentionBias/bench_results/bench_persistent.txt"
    with open(out_path, "w") as f:
        f.write("# Persistent vs non-persistent fwd, user workload (B=48, seqlen 1k-4k)\n")
        f.write(f"# GPU: {torch.cuda.get_device_name(0)}, "
                f"SMs={sm_count}\n")
        f.write("# pers   = StaticPersistentTileScheduler (default)\n")
        f.write("# non-p  = SingleTileScheduler (one CTA per tile)\n")
        f.write("# speedup = non-persistent_ms / persistent_ms; >1 means persistent wins\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        for r in rows:
            f.write(r + "\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
