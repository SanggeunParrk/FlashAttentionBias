# fp32 bias fwd: TMEM-init variant postmortem

## TL;DR

Experimented with seeding the S TMEM tile with `bias / softmax_scale` before
the QK MMA and using `zero_init=False` so the MMA accumulates `Q@K^T` on top
of pre-loaded bias (instead of staging bias through SMEM and folding it with
an `fma_packed_f32x2` inside softmax).

Result: the variant is **~1.9× slower** than the SMEM-staged baseline at every
L tested, and additionally **regresses in accuracy at L ≥ 4096**. Variant
rolled back; baseline (`flash_bias_fwd_sm100_smem.py`) remains the production
path.

## Design under test

`flash_bias_fwd_sm100_tmem_init.py` (now removed):

- New `pipeline_bias_tmem` (producer = softmax warps, consumer = MMA warp,
  `q_stage` stages).
- Softmax warps, per K block: wait on `pipeline_bias` (SMEM bias) → register
  load → r2t TMEM store of `bias / softmax_scale` into the S slot →
  `fence_view_async_tmem_store()` → release SMEM bias / commit `pipeline_bias_tmem`.
- MMA warp, per K block: `consumer_wait(pipeline_bias_tmem)` → QK MMA with
  `zero_init=False` so `S = Q@K^T + bias/scale` in the unscaled domain.
- Softmax then uses `softmax_scale_log2 = softmax_scale * log2(e)` instead of
  the baseline's `log2(e)`, folding the scale into the exp2 factor.

Math sanity check: `exp2(S * scale * log2_e) = exp(scale * Q@K^T + bias)` ✓.

## Accuracy results

`verify_bias_compare.py` (A=1, B=1, H=16, D=64, tile_n=64, q_stage=2),
each row is the same kernel vs the fp32 reference:

| L | variant | out max | out mean | lse max | lse mean |
|---|---|---|---|---|---|
| 1024 | smem      | 1.6e-3 | 6.7e-5 | 1.9e-3 | 7.0e-4 |
| 1024 | tmem_init | 1.6e-3 | 6.7e-5 | 1.9e-3 | 7.0e-4 |
| 2048 | smem      | 1.1e-3 | 4.8e-5 | 1.8e-3 | 7.0e-4 |
| 2048 | tmem_init | 1.1e-3 | 4.8e-5 | 1.8e-3 | 7.0e-4 |
| 4096 | smem      | 1.2e-3 | 3.5e-5 | 1.5e-3 | 7.0e-4 |
| 4096 | tmem_init | **1.7e-1** | **6.9e-3** | **9.1e-1** | **2.1e-1** |

At L=1024/2048 the two variants are bitwise-equivalent up to fp32 accumulation
order. At L=4096 tmem_init's error explodes. Cause not investigated further
(suspect phase / TMEM aliasing in the new pipeline at deeper K-block counts);
moot once perf was measured.

## Performance results

`bench_bias_compare_with_tmem_init.py` (A=48, B=1, H=16, D=64, tile_n=64,
q_stage=2, warmup=5, rep=20, B200):

| L | fp32 ms · TF | bf16 ms · TF | bias_smem ms · TF | bias_tmem_init ms · TF | tmem / smem |
|---|---|---|---|---|---|
| 1024 | 0.31 · 659 | 0.31 · 655 | 1.21 · 170 |  2.30 ·  90 | 1.90× |
| 2048 | 1.15 · 715 | 1.42 · 579 | 5.02 · 164 |  9.22 ·  89 | 1.84× |
| 4096 | 4.67 · 706 | 5.26 · 627 | 18.86 · 175 | 36.82 ·  90 | 1.95× |
| 8192 | 18.92 · 698 | 18.33 · 720 | 74.16 · 178 | 146.20 ·  90 | 1.97× |

(`tmem_init` numbers at L=4096+ are with broken outputs; perf alone is still
representative of the steady-state pipeline.)

Codex's own bench at A=1 reproduced the same ratio independently
(`bench_results/bench_bias_tmem_init_A1_H16D64.txt`): tmem/smem 1.92×–2.04×.

## Why it is ~2× slower

Steady-state warp-group schedule for one K block, sketched:

```
SMEM baseline (bias folded into softmax):
  MMA:        [QK0]              [QK1]              [PV0]   [QK0']  [PV1] ...
  softmax:           [load_S+fma_bias+softmax+P_store]  ‖  MMA-QK
  → softmax work and MMA-QK overlap; bias is one fma_packed_f32x2 line.

TMEM-init:
  MMA:        [wait_init0][QK0]  [wait_init1][QK1]  [PV0]   [QK0']  [PV1] ...
  softmax:    [INIT0    ][INIT1            ][SF0]                  [SF1]
  where INIT = SMEM_load + r2t TMEM_store + fence_view_async_tmem_store
        SF   = wait_S + softmax + P_store   (bias already in S)
  → MMA gates on INIT; softmax warps must produce INITs faster than MMA
    consumes them, otherwise MMA stalls on consumer_wait(pipeline_bias_tmem).
```

What broke the overlap:

1. **INIT throughput < QK throughput.** The fused fma in baseline is ~32 fma
   instructions per softmax thread, all register-resident, no fence. INIT is
   an 8K-element SMEM→register→TMEM r2t store plus a mandatory
   `fence_view_async_tmem_store()` for ordering — substantially heavier per
   K block.
2. **Softmax warp group serializes INIT and SF on the same warps.** The same
   four warps that do INIT also do the wait-S / softmax / P-store. The
   prefetch hide that q_stage=2 *could* provide (INIT_{n+1} during MMA's
   QK_n) requires INIT to finish in the QK_n window; here INIT alone exceeds
   it before SF is even counted.
3. **QK switches to `mma.add` (zero_init=False).** Same tensor-core
   throughput, but issue carries a TMEM read dependency. Tiny effect relative
   to (1) and (2).
4. **Extra pipeline (`pipeline_bias_tmem`) mbarrier traffic.** Negligible.

The ~1.9× ratio is roughly K-block independent across L=1024..8192, which is
the signature of a per-K-block serialization (the hidden-cost-per-step term),
not a one-time prologue cost. That matches the picture above.

## Conclusion

The baseline's "fold bias into softmax via one fma_packed_f32x2" is hard to
beat because it is:

- a few register-resident fmas per thread,
- fully overlapped with the QK MMA (different warp group, different stage),
- no extra TMEM store and no fence on the critical path.

The TMEM-init design moves bias into the MMA's critical path and replaces a
cheap fma with an expensive store+fence, which is exactly the wrong direction
on B200. If revisited, viable angles would be (a) doing the SMEM→TMEM bias
fill from the *load* warps (TMA path) rather than softmax warps, (b) deeper
q_stage to widen the prefetch window, or (c) interleaving the TMEM store
within softmax to fold the fence. None of these are pursued now; SMEM
staging stays.

## Artifacts

- Variant kernel: `FA4_fp32/kernels/flash_bias_fwd_sm100_tmem_init.py` (deleted)
- Accuracy harness: `FA4_fp32/verify_bias_tmem_init.py`,
  `FA4_fp32/verify_bias_compare.py` (deleted)
- 4-way bench: `FA4_fp32/bench_bias_compare_with_tmem_init.py` (deleted)
- Codex's earlier A=1 bench result kept: `bench_results/bench_bias_tmem_init_A1_H16D64.{txt,png}`
