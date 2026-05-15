# fp32 dense-bias fwd: bottleneck decomposition (measured)

Follow-up to [BIAS_FWD_OPTIMIZATION_PLAN.md](./BIAS_FWD_OPTIMIZATION_PLAN.md).
That note guessed kv_stage shrinkage was the dominator. **Direct
measurement says otherwise**: the bias overhead is ~80% SMEM bias read,
~20% TMA path, with kv_stage shrinkage and fma both small.

Constraint reminder: fp32 dtype and the algorithm cannot change. All fake
kernels below preserve the bias TMA + pipeline_bias dependency chain;
only the body of `apply_bias_smem` is altered.

## Setup

Five timing points, all on B200, A=48, B=1, H=16, D=64, tile_n=64,
q_stage=2:

| label | scripts | what it runs |
|---|---|---|
| `no-bias kv=6` (natural) | [`exp_kv_stage_isolate.py`](../FA4_fp32/exp_kv_stage_isolate.py) | bias-free fp32 fwd, natural kv_stage |
| `no-bias kv=2` (capped)  | [`exp_kv_stage_isolate.py`](../FA4_fp32/exp_kv_stage_isolate.py) | same, kv_stage subclass-capped to 2 |
| `TMA-only` (apply=noop)  | [`exp_bias_load_only.py`](../FA4_fp32/exp_bias_load_only.py) | bias TMA + pipeline_bias wait/release stay; `apply_bias_smem` body replaced by `return` |
| `read_only`              | [`exp_bias_apply_decompose.py`](../FA4_fp32/exp_bias_apply_decompose.py) | SMEM read + Float32 cast still happen; results accumulated into a register sink and added into `tSrS[0]` to keep the SMEM loads live; **no fma** |
| `fma_only`               | [`exp_bias_apply_decompose.py`](../FA4_fp32/exp_bias_apply_decompose.py) | `fma_packed_f32x2` runs with `b=(0,0)` register constants; **no SMEM read** |
| `real bias fp32`         | [`exp_bias_hbm_isolate.py`](../FA4_fp32/exp_bias_hbm_isolate.py) (fp32 case) | unchanged baseline `FlashAttentionBiasForwardSm100Smem` |

## Raw numbers (ms / TFLOPS)

| L | no-bias kv=6 | no-bias kv=2 | TMA-only | fma_only | read_only | real bias |
|---|---|---|---|---|---|---|
| 1024 | 0.363 / 568  | 0.524 / 393 | 0.512 / 401 | 0.514 / 401 | 1.225 / 168 | 1.209 / 170 |
| 2048 | 1.397 / 590  | 1.988 / 415 | 1.938 / 425 | 1.954 / 422 | 4.912 / 168 | 4.771 / 173 |
| 4096 | 5.576 / 592  | 7.127 / 463 | 8.138 / 405 | 8.261 / 399 | 19.117 / 172 | 18.866 / 175 |
| 8192 | 22.930 / 575 | 27.121 / 487 | 37.488 / 352 | 36.844 / 358 | 75.608 / 175 | 74.165 / 178 |

Two equivalences fall out cleanly:

- **`fma_only` ≈ `TMA-only`** at every L. The packed-fma instruction
  stream itself is essentially free — register-resident, plenty of ILP,
  fully hidden behind whatever else is on the critical path.
- **`read_only` ≈ `real bias`** at every L. The bias path's runtime is
  fully accounted for by the SMEM-read path; the fma adds nothing
  measurable.

## Cost decomposition (L=8192, headline numbers)

Walking from the cleanest baseline up to the real kernel:

| step | Δms | Δ% of overall bias overhead |
|---|---|---|
| no-bias kv=6 → no-bias kv=2 (kv_stage shrinkage) | +4.19 | 9% |
| no-bias kv=2 → TMA-only (bias TMA + pipeline_bias sync) | +10.37 | 22% |
| TMA-only → real bias (`apply_bias_smem` body) | +36.68 | 80% (of which fma ≈ 0, SMEM read ≈ all) |
| Total: no-bias kv=6 → real bias | +51.24 ms | 100% (= 27.12 → 74.17) |

Within the +36.68 ms `apply_bias_smem` cost, `fma_only` vs `read_only`
show fma at ~0 ms and SMEM-read at ~+38 ms. So **the bias overhead is
essentially: 80% SMEM bias read, 20% bias TMA path, with kv_stage and
fma rounding errors**.

## L dependence

| L | kv shrink (% of total) | TMA path (%) | apply (%) |
|---|---|---|---|
| 1024 | 23% | -1% (noise) | 78% |
| 2048 | 21% | -1% | 80% |
| 4096 | 13% | +9% | 78% |
| 8192 | 8%  | +20% | 72% |

- The `apply` (SMEM read) share is essentially fixed across L — it's a
  per-element cost.
- TMA path becomes more visible at larger L (HBM bandwidth pressure and
  larger K-block counts).
- kv_stage shrinkage matters at small L (pipeline fill effects) and fades
  at large L.

## What `apply_bias_smem` is, exactly

`flash_bias_fwd_sm100_smem.py:1302–1321`. Per K block, for each (q,k)
element this softmax warp is responsible for, it computes
`S[q,k] = S[q,k] · softmax_scale + bias[q,k]`:

```python
for i in range(0, size(tSrS), 2, unroll_full):
    q0,k0 = tScS_t2r[i]
    q1,k1 = tScS_t2r[i+1]
    b0 = Float32(sBias[q0,k0,stage])     # SMEM read + cast
    b1 = Float32(sBias[q1,k1,stage])     # SMEM read + cast
    tSrS[i], tSrS[i+1] = fma_packed_f32x2(
        (tSrS[i], tSrS[i+1]),
        (scale, scale),
        (b0, b1),
    )
```

128 softmax threads each issue ~32 packed-fma per K block (64 elements /
2 per fma). The two SMEM reads (b0, b1) per fma are the bytes that
dominate; the fma itself measured to ~zero cost.

## Caveat on `read_only`

The `read_only` variant uses a register sink to keep SMEM loads
live (otherwise the compiler would DCE them). The sink is a single
serial accumulator and adds a 32-deep register-add chain into the
critical path. The reason this is acceptable: `fma_only` proves the
register fp-add work is itself ~zero, so what's left in `read_only`
above `TMA-only` is the SMEM-read path proper (load + bank pattern +
read→consumer dependency).

## Updated lever priorities

Rewriting the plan-md priorities now that we have measurements:

| lever | targets | rough upper bound on gain |
|---|---|---|
| **A. Cut SMEM bias bytes per element** | SMEM read path | bf16 bias previously gave ~30% — that was mostly *halved SMEM bytes*, not halved HBM. Off the table per user constraint, but it confirms direction. |
| **B. SMEM bank-conflict / layout audit** | SMEM read path | unknown; needs `cuobjdump` / `ncu` look |
| **C. Bypass SMEM: TMA → register direct stream** | SMEM read path entirely | potentially large; need to check whether SM100 cp.async / tcgen05 can target register without SMEM intermediary, or whether a producer warp can stream bias into a register buffer the softmax warps read |
| **D. Distribute the read across more warps** | SMEM read path | small-ish; softmax warp count is already 4 |
| **E. 2-CTA TMA multicast for bias** | TMA path | ≤ 22% (TMA path total) |
| **F. kv_stage rework (bias_stage decoupling etc.)** | kv shrink | ≤ 9% at L=8192, ≤ 23% at L=1024 |
| **G. fma re-shaping** | fma | ~0 — nothing to win |

The clear top of the list is **A/B/C — anything that cuts SMEM read on
the bias side**. Earlier guesses about kv_stage / SMEM occupancy / IO
reduction (FlashBias-style) all sit lower than expected.

## Open questions for next step

- Is there a SM100 path to deliver bias to softmax warps without going
  through SMEM (TMA → register, or load-warp register stream)?
- What is the actual SMEM bank pattern for `sBias[q,k,stage]` at the
  current layout? Worth checking with `ncu` or a small bank-conflict
  micro-probe before committing to layout changes.
- Does softmax-warp count = 4 starve the SMEM ports?
