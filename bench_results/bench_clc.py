"""Benchmark FA_CLC=0 (StaticPersistentTileScheduler) vs FA_CLC=1
(SingleTileLPTScheduler, CLC mode) for fwd on B200.

Both are persistent. The difference:
  - FA_CLC=0: static grid-stride loop, SM_count CTAs.
  - FA_CLC=1: Blackwell hardware dynamic scheduler. One warp issues
    CLC prefetches; tiles are dispatched dynamically by the HW unit
    instead of via a fixed CTA→tile mapping.

Sweep targets the user's workload (B=48, H ∈ {4,8,16},
D ∈ {32, 64}, seqlen ∈ {1k, 2k, 4k}, dtype ∈ {bf16, fp32}, non-causal).
"""
import os
import sys
import gc

import torch
from triton.testing import do_bench

sys.path.insert(0, "/home/FlashAttentionBias")


def flops_fwd(B, H, L, D):
    return 4.0 * B * H * L * L * D


def bench_one(B, H, L, D, dtype):
    from FA4_fp32 import flash_attn_func

    q = torch.randn(B, L, H, D, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def fa():
        return flash_attn_func(q, k, v)[0]

    fa()
    torch.cuda.synchronize()
    t = do_bench(fa, warmup=10, rep=40)

    del q, k, v
    gc.collect()
    torch.cuda.empty_cache()
    return t


def bench_pair(B, H, L, D, dtype):
    os.environ["FA_CLC"] = "0"
    t_static = bench_one(B, H, L, D, dtype)
    os.environ["FA_CLC"] = "1"
    t_clc = bench_one(B, H, L, D, dtype)
    return t_static, t_clc


def fmt_dtype(d):
    return {torch.bfloat16: "bf16", torch.float32: "fp32"}[d]


def main():
    assert torch.cuda.is_available(), "need CUDA"
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"GPU: {torch.cuda.get_device_name(0)}, SMs={sm_count}")

    B = 48
    HS = [4, 8, 16]
    LS = [1024, 2048, 4096]
    DS = [32, 64]
    DTYPES = [torch.bfloat16, torch.float32]

    header = (
        f"{'dtype':>5} {'B':>3} {'H':>3} {'L':>5} {'D':>4} "
        f"{'tiles':>6} {'waves':>6} "
        f"{'static ms':>10} {'CLC ms':>9} "
        f"{'static TF':>10} {'CLC TF':>9} {'speedup':>8}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    rows = []
    for dtype in DTYPES:
        for H in HS:
            for L in LS:
                for D in DS:
                    # fwd m_block_size = 128 in both fp32 and bf16
                    tiles = B * H * ((L + 127) // 128)
                    waves = (tiles + sm_count - 1) // sm_count
                    try:
                        t_static, t_clc = bench_pair(B, H, L, D, dtype)
                    except Exception as e:
                        line = (
                            f"{fmt_dtype(dtype):>5} {B:>3} {H:>3} {L:>5} {D:>4} "
                            f"{tiles:>6} {waves:>6}  FAILED: {type(e).__name__}: "
                            f"{str(e).splitlines()[0][:80]}"
                        )
                        print(line); rows.append(line); continue
                    f = flops_fwd(B, H, L, D)
                    tf_s = f / (max(t_static, 1e-9) * 1e9)
                    tf_c = f / (max(t_clc, 1e-9) * 1e9)
                    # speedup > 1 means CLC is faster than static
                    speedup = t_static / max(t_clc, 1e-9)
                    line = (
                        f"{fmt_dtype(dtype):>5} {B:>3} {H:>3} {L:>5} {D:>4} "
                        f"{tiles:>6} {waves:>6} "
                        f"{t_static:>10.3f} {t_clc:>9.3f} "
                        f"{tf_s:>10.1f} {tf_c:>9.1f} {speedup:>7.3f}x"
                    )
                    print(line, flush=True)
                    rows.append(line)

    out_path = "/home/FlashAttentionBias/bench_results/bench_clc.txt"
    with open(out_path, "w") as f:
        f.write("# CLC scheduler (FA_CLC=1) vs StaticPersistentTileScheduler (FA_CLC=0), fwd\n")
        f.write(f"# GPU: {torch.cuda.get_device_name(0)}, SMs={sm_count}\n")
        f.write("# static = StaticPersistentTileScheduler (default)\n")
        f.write("# CLC    = SingleTileLPTScheduler in CLC mode (Blackwell HW dynamic)\n")
        f.write("# speedup = static_ms / CLC_ms; >1 means CLC wins\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        for r in rows:
            f.write(r + "\n")
    # Reset env so the toggle doesn't leak.
    os.environ.pop("FA_CLC", None)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
