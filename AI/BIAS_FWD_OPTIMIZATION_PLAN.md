# fp32 dense-bias forward: bottleneck analysis & optimization plan

Working notes for the SMEM-staged bias fwd kernel
(`flash_bias_fwd_sm100_smem.py`). Constraints set by user:
**fp32 dtype is fixed and the algorithm cannot change** — final result must
be bit-equivalent to the baseline. So everything below is schedule / memory
layout / hardware feature rework, not numerical rework.

Companion notes:
- [TMEM_INIT_BIAS_POSTMORTEM.md](./TMEM_INIT_BIAS_POSTMORTEM.md) — failed
  TMEM-init variant (≈1.9× slower + accuracy regression at L≥4096).

## Where we are today (B200, A=48, H=16, D=64, fp32)

| L | fp32 no-bias TF | bias_smem TF | bias slowdown |
|---|---|---|---|
| 1024 | 659 | 170 | 3.9× |
| 8192 | 698 | 178 | 3.9× |

bias kernel ≈ ¼ of no-bias throughput at every L. Goal: understand where
that gap comes from, then close it.

## SMEM budget — the smoking gun

`_setup_attributes` in [flash_bias_fwd_sm100_smem.py:174](../FA4_fp32/kernels/flash_bias_fwd_sm100_smem.py#L174):

```
(m=128, n=64, D=64, q_stage=2, fp32)
smem_q   = 2·128·64·4 =  64 KB
smem_o   = 2·128·64·4 =  64 KB
smem_bias= 2·128·64·4 =  64 KB    (q_stage · m · n · 4, fp32 bias)
smem_kv  = 64·64·4    =  16 KB / stage
kv_stage = (224 − 64 − 64 − 64) / 16 = 32 / 16 = 2
```

No-bias baseline would get `kv_stage = (224 − 128) / 16 = 6`. So **bias
staging cuts KV pipeline depth 6 → 2.** With depth 2 the QK MMA is exposed
to K/V TMA latency on essentially every K block — a strong candidate for
the dominant slowdown.

## What we already get for free

`BiasAfastPersistentTileScheduler` in
[flash_bias_fwd_sm100_smem.py:55](../FA4_fp32/kernels/flash_bias_fwd_sm100_smem.py#L55)
orders tiles "A-fastest":

```python
tile_wo_a, a_idx       = divmod(tile_idx, num_a)       # A innermost
hb_idx,    block_idx   = divmod(tile_wo_a, num_block)  # then M
batch_b_idx, head_idx  = divmod(hb_idx,   num_head)
```

bias has shape `(1, B, Lq, H, Lk)` and is broadcast over A. With A-fast,
the 132 SMs at any instant are working on the same `(b, m, h)` tile across
different `a` values, so they all read the same bias rows. First SM faults
the bias into L2 (≈4 MB at L=8192, well within B200's ~60 MB L2); the rest
are L2-hits. Effective bias HBM traffic is ≈ 1/A of the naive expectation.

So scheduler / L2-reuse lever is **already exhausted by the baseline**.
Whatever we measure today already assumes good L2 reuse.

## HBM isolation experiment (decisive)

[`FA4_fp32/exp_bias_hbm_isolate.py`](../FA4_fp32/exp_bias_hbm_isolate.py)
forces `kv_stage = 2` for both variants (subclassing
`FlashAttentionBiasForwardSm100Smem` and capping `kv_stage` after
`_setup_attributes`). Then bias dtype is the only thing changing —
fp32 vs bf16 — which halves bias HBM traffic and bias SMEM bytes per stage
but leaves the pipeline structure identical.

`bench_results/exp_bias_hbm_isolate.txt`:

| L | fp32-bias ms | bf16-bias ms | bf16/fp32 |
|---|---|---|---|
| 1024 | 1.209 | 0.836 | 0.692× |
| 2048 | 4.771 | 3.281 | 0.688× |
| 4096 | 18.866 | 12.773 | 0.677× |
| 8192 | 74.165 | 50.144 | 0.676× |

Reading: **halving bias HBM ⇒ ≈30% wall-clock reduction**. So bias HBM
traffic is a real but **not dominant** cost. The remaining ≈70% comes from
something else — most likely the `kv_stage = 2` depth itself (the pipeline
can't hide K/V TMA latency), plus second-order effects (SMEM bias buffer
size, TMA dispatch count, softmax SMEM read).

Since dtype is off the table, this lever is closed. Anything left has to
free SMEM or restructure scheduling.

## Lever inventory (all preserve numerics)

Ordered by approximate (effect / effort) ratio. Numbers in `effort` are
qualitative (1=single-line, 3=multi-file pipeline rework).

| # | Lever | Mechanism | Expected gain | Effort |
|---|---|---|---|---|
| L1 | `tile_n = 32` measurement | smem_bias → 32 KB, smem_kv → 8 KB ⇒ kv_stage = 8. 2× K-block count, but tests whether kv_stage is the dominator. | unknown; diagnostic | 1 |
| L2 | `bias_stage = 1` (decouple from q_stage) | smem_bias → 32 KB ⇒ kv_stage = 4. Loses bias prefetch depth across q_stages — needs measurement. | maybe 1.2–1.5× | 2 |
| L3 | bias / K SMEM time-alias | bias is consumed by softmax *after* K is consumed by QK MMA. Reuse the same SMEM slot. Pushes effective SMEM usage from "static both" to "either, never both". | maybe 1.3–1.6× | 3 |
| L4 | 2-CTA cluster TMA multicast for bias | bias broadcasts across A; A-fast scheduler already groups same-bias CTAs in time. Cluster of 2 (or 4) SMs lets one CTA HBM-load and the rest receive via SMEM-multicast. HBM bias traffic ÷ cluster_size. Bit-exact. Reference path exists in `flash_fwd_sm100.py` (D=128 2-CTA). | 1.1–1.3× (bias HBM was already L2-reused; this trims the L2 misses) | 3 |
| L5 | Dedicated bias producer warp | Currently K/V/Bias TMA issued by the same load warp. Split bias to its own warp so its dispatch doesn't serialize with KV dispatch. | small (TMA issue is async; gain mostly latency-hide quality) | 2 |
| L6 | `m_block_size = 64` | smem_q/o/bias all halve ⇒ kv_stage = 4. Trades tile efficiency for pipeline depth. | unknown; diagnostic | 1 |

Cross-ruled-out:
- **bias dtype reduction** (bf16/fp8/int8) — user constraint.
- **low-rank baked-in (`[Q;U]·[K;V]ᵀ`)** — changes algorithm shape, user constraint.
- **score_mod path** — would skip SMEM staging but still has to load bias somewhere; not bit-exact for arbitrary fp32 ordering.
- **Different scheduler order** — A-fast is already correct.

## Diagnostic order

Goal: identify the dominator with the cheapest lever before committing to
a heavy rewrite.

1. **L1 (tile_n=32)** first. One-argument change, no code edit. Verify
   bit-exactness, then bench.
   - If notably faster → kv_stage is the dominator → L2 / L3 are the
     right rewrites.
   - If flat or slower → kv_stage isn't the dominator, then the cost is
     either TMA dispatch overhead (more K-blocks didn't help) or SMEM
     bias buffer access (size irrelevant). Skips L2/L3, goes to L4/L5.
2. From L1 result, pick L2 or L4 as the first non-trivial rewrite.

## Decision rationale (why not pour effort into FlashBias-style IO reduction)

The published direction (FlashBias, FlashIPA) attacks **HBM IO complexity**,
which our experiment shows accounts for ~30% of the gap. The remaining
~70% is internal to the kernel's SMEM / pipeline geometry. Optimizing IO
further (when we already do A-fast L2 reuse) brings rapidly diminishing
returns. The leverage is in **putting kv_stage back to 4–6** without
changing fp32 or algorithm, which means SMEM rework, not IO rework.

## Open questions to answer next

- L1 measurement result (kv_stage dominator?)
- Whether L3 (SMEM time-alias) is feasible given current pipeline
  ordering — needs a look at `mma` ↔ `softmax_step` ordering to see
  whether bias is read strictly after K in every stage.
- Whether 2-CTA cluster (L4) can be combined with A-fast scheduler
  without losing the L2-reuse property.
