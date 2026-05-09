# fp32 backward bug diagnosis — TMEM layout was a red herring

## TL;DR

The fp32 bwd produces wrong dQ/dK/dV. After many failed attempts on the
TMEM-layout hypothesis, a controlled experiment isolated the real cause:
**LSE/D pairing mismatch between `copy_utils.make_tmem_copy` (bf16-tuned) and
TF32 MMA C-fragment thread mapping**. The TMEM packing trick / store layout is
NOT the issue.

## Key experiment

Set `Q = 0` so `S = 0` and `P = exp(0 - LSE) = 1/L` uniform. Then
`dV[j, d] = (1/L) * sum_i dO[i, d]` — a known constant per d, independent of LSE.

Run with B=1, L=64, H=1, D=32 fp32:

```
expected dV (= mean of dO per col):  [-0.1122, 0.0721, -0.1652, 0.1634, ...]
fa     dV  [0, 0, 0, :8]:            [-0.1122, 0.0720, -0.1652, 0.1633, ...]
fa     dV  [0, 30, 0, :8]:           [-0.1122, 0.0720, -0.1652, 0.1633, ...]
max abs err: 1.6e-04   (TF32 noise level)
```

✓ **fp32 dV is fully correct when P is uniform**.

This rules out:
- TMEM A-operand layout for dV MMA (would scramble even uniform P)
- dV MMA computation
- dV epilogue (TMEM → SMEM → gmem path)
- dV gmem layout

## Why standard random Q/K/V fails

With non-uniform S (= QK^T), each P[i, j] depends on the right LSE[i]. If the
LSE-pairing logic gives thread t the LSE for *different* M values than the S
elements thread t holds, then thread t computes
`P_wrong[t_v] = exp(S[lane_t, n_v] - LSE[lane_t', different m])` — uses wrong LSE.

Pattern of failure consistent with this:
- dV rows partially correct (cosine sim ~0.77) — some rows happen to align
- dK / dQ fully wrong (sim ~0.04) — depend on dS = P * (dP - D), accumulating
  the LSE error and a similar dPsum-pairing error

## The thread-mapping mismatch

```
flash_bwd_sm100.compute_loop:

    thr_copy_t2r = copy_utils.make_tmem_copy(load_atom, num_wg).get_slice(tidx)
    # ↑ HARDCODED layout_tv: ((32, 4, num_wg), (num_rep, 32))
    #   This was tuned for bf16 K=16 atom. Doesn't match TF32 K=8 atom.

    tSsLSE = thr_copy_t2r.partition_D(thr_mma_S.partition_C(sLSE_2D))
    # ↑ Composes:
    #     1) partition_C(sLSE) — gives MMA-thread-aware (m, n) view
    #     2) partition_D(...) — re-partitions over t2r threads
    #
    # If thr_copy_t2r ≠ MMA C-fragment thread layout, this composition mixes
    # two different per-thread (m, n) mappings. Thread t loads S elements at
    # t2r-positions but LSE values at MMA-positions. Mismatch.

    tSsdPsum = thr_copy_t2r.partition_D(thr_mma_dP.partition_C(sdPsum_2D))
    # ↑ Same problem on dPsum side, propagates into dS.
```

For bf16 the `copy_utils` tile_copy was specifically designed to match bf16 MMA
C-fragment, so the composition is benign. For TF32 (fp32), they diverge.

## Failed fix attempts (in this session)

These all targeted the wrong root cause:

| Attempt | Result |
|---|---|
| Replace `cvt_f16` with direct fp32 copy (P / dS) | Compiles, runs — wrong values |
| Postprocess `dQ_reduce_ncol = 16` for fp32 | Removes one OOB but doesn't fix accuracy |
| Postprocess `thr_layout_r2s_dQ` cap by `tile_m` (gcd) | Removes another OOB, no accuracy fix |
| Switch P-store to `tcgen05.make_tmem_copy(atom, tStP)` | dV worse (abs 1→144) |
| `St32x32bOp(Repetition(8))` for fp32 | NaN/inf |
| Use `tP_layout.outer` as P destination | MLIR rejects structured layout |
| 2-mode tStP via fwd-style composition | Index pattern breaks downstream |
| dKV epilogue: derive r2s from MMA-aware t2r | No change (epilogue isn't the bug) |

## What actually needs to happen

Two paths, in order of preference:

### A. Switch t2r to `tcgen05.make_tmem_copy(load_atom, tStS)` (tensor-aware)

This auto-derives the tile_copy from `tStS`'s actual layout — which IS the
TF32 MMA C-fragment layout. So `thr_copy_t2r`'s threads then align with MMA's
threads. `partition_D(partition_C(sLSE_2D))` becomes consistent.

Downstream impedance to fix:
- The SMEM dS r2s path uses `make_tiled_copy_D(atom, thr_copy_t2r)` — must
  produce a partition compatible with `sdS_epi`. Need to update sdS_layout
  to match the new tile_copy's tiler rank (rank 3 instead of rank 1).
- Inner loop `tSrS_t2r[None, stage, 0, 0]` indexing patterns may shift
  (3-mode vs 4-mode). bf16 path can stay on `copy_utils` to avoid
  regression — fork the path on dtype.

### B. Replace LSE/D partition with t2r-direct (skip partition_C)

Instead of `partition_D(partition_C(sLSE_2D))`, use a t2r-direct lookup so the
LSE thread mapping is built from t2r's identity coordinates, not from MMA.
Conceptually:

```python
# For each S-element at coord (m_v, n_v) in thread t,
# lookup LSE[m_v] using t2r's identity tensor tScS_t2r.
m_per_thread = tScS_t2r.layout[..., COL=1]  # compile-time constexpr
tSrLSE_full = ...  # all 64 LSE values per stage
tSrLSE[v] = tSrLSE_full[m_per_thread[v]]
```

The challenge: cute_dsl's gather/indirect indexing API. May or may not be
directly expressible.

## What I'd do first next time

1. Reproduce Q=0 test to confirm dV path is correct.
2. Add LSE=0 case (would require modifying forward to return zero LSE) to
   isolate LSE-pairing specifically. (Or check by setting Q tiny so |S|<<LSE
   so P stays uniform — should give correct dV.)
3. If LSE-pairing is confirmed, attack option A (tcgen05 t2r) with the
   downstream SMEM r2s adaptation as a planned line item rather than an
   afterthought.

## Code state at end of this session

bf16 baseline preserved (test_bwd 9/9 PASS). Fp32 partial fixes from earlier
sessions remain (cvt_f16 size assertion, postprocess fp32 layout). The actual
LSE-pairing fix is NOT applied — that's the next step.
