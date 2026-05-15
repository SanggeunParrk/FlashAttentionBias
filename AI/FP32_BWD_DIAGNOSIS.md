# fp32 bwd diagnosis — current state and bug catalog

## Executive summary

`FA4_fp32` backward pass produces **wrong dQ/dK/dV under fp32**, while
**bf16 passes 9/9** end-to-end. After 7 debugging sessions, two distinct
fp32-specific bugs were located:

| Bug | Location | Status |
|---|---|---|
| **#1 — `dQacc_reduce` reshape over-strides for fp32** | `flash_bwd_sm100.py` ~L 2024 | **FIXED** (in tree) |
| **#2 — dP TMEM read returns garbage for fp32** | `compute_loop` dP load | LOCATED, **NOT FIXED** |

bf16 9/9 PASS preserved across all attempts.

The **original diagnosis (LSE-pairing) is wrong** — verified via
`cute.printf` traces. LSE values pair correctly with the per-thread S
elements; the bug is in dP read (Bug #2), with secondary corruption from
Bug #1 in dQ write-back.

---

## Reproducing & verifying

```bash
# Full regression. bf16 9/9 PASS, fp32 9/9 FAIL.
CUDA_VISIBLE_DEVICES=0 /opt/miniforge3/bin/python FA4_fp32/bwd/test_bwd.py

# Uniform-P diagnostic. fp32 dV correct (uniform P → permutation-invariant).
CUDA_VISIBLE_DEVICES=0 python FA4_fp32/bwd/test_q0.py

# Structured-input probes (zero_q, small_q, single_v, eye_dout):
CUDA_VISIBLE_DEVICES=0 python FA4_fp32/bwd/debug_probe.py small_q
```

Current fp32 error magnitudes after Bug #1 fix:

```
[B=1 L=128 D=32]  dQ=1.29  dK=1.74  dV=0.90
[B=1 L=256 D=64]  dQ=0.86  dK=1.07  dV=0.93
[B=1 L=512 D=64]  dQ=0.69  dK=0.83  dV=0.60
```

(atol target is 5e-3; all configs fail.)

---

## Bug #1 — `dQacc_reduce` reshape over-stride (FIXED)

`FA4_fp32/kernels/flash_bwd_sm100.py:dQacc_reduce`. Original code:

```python
tdQrdQ_shape = (self.dQ_reduce_ncol,                # = 16 for fp32
                self.tile_hdim // self.dQ_reduce_ncol)
tdQrdQ = cute.make_tensor(tdQrdQ_t2r.iterator, tdQrdQ_shape)
# (16, 2):(1, 16) col-major  → tdQrdQ[None, stage=1] reads register offset 16
```

The register fragment for fp32 has only **8 elements per chunk × 2 chunks
= 16 total per thread**, with chunk 1 at register offset 8 (the layout
shows `(((2,4),1),1,1,2):(((1,2),0),0,0,8)` — chunk stride 8). The
reshape's mode-1 stride 16 over-shoots → stage=1 reads past the
fragment → returns zeros → GMEM `dq_accum`'s second half stays zero →
final dQ has cols 16..31 universally zero.

For bf16, V-per-chunk happens to equal `dQ_reduce_ncol` so the shape
matches by coincidence. For fp32 (TF32 m64nNk8 C-frag), TF32 packs N
across threads more tightly, halving the per-thread V count.

**Fix** (in tree):

```python
num_chunks_reshape = self.tile_hdim // self.dQ_reduce_ncol
if const_expr(fp32_dQ):
    per_thread_V_per_chunk = cute.size(tdQrdQ_t2r) // num_chunks_reshape
    tdQrdQ_shape = (per_thread_V_per_chunk, num_chunks_reshape)
else:
    tdQrdQ_shape = (self.dQ_reduce_ncol, num_chunks_reshape)
```

Confirmed: dq_accum's second half went `0 → 2.59` max; dQ output cols
16..31 now mirror cols 0..15.

---

## Bug #2 — dP TMEM read returns garbage (NOT FIXED)

`FA4_fp32/kernels/flash_bwd_sm100.py:compute_loop`. After the dP MMA
(V @ dO^T) completes and `pipeline_dP.consumer_wait` returns, reading
`tdPtdP` via `cute.copy(thr_copy_t2r, tdPtdP_t2r[None, stage, None, None],
tdPrdP_t2r)` produces **wrong values for tidx=0 V[1..]** under fp32.
bf16 same setup reads correctly.

### Empirical evidence — V=e_0 probe

Set `V[0,0,0,0]=1`, V elsewhere = 0. Then `dP[m, n=0] = dout[m, 0]` and
`dP[m, n>0] = 0`. Per-thread dump from inside the kernel:

```
[dP] tidx=0  V[0] coord=(0,0) dP=-0.924   ← ✓ matches dout[m=0, 0]=-0.925
[dP] tidx=0  V[1] coord=(0,1) dP=-0.180   ← ✗ expected -0.597 (dout[m=1, 0])
[dP] tidx=0  V[2] coord=(0,2) dP= 0.224   ← ✗ expected -0.033
[dP] tidx=0  V[3] coord=(0,3) dP=-0.262   ← ✗ expected -0.086

[dP] tidx=1  V[0..3]  all 0  ✓ (n=1 column → all zero, V=e_0)
[dP] tidx=2  V[0..3]  all 0  ✓
[dP] tidx=32 V[0..3]  all 0  ✓
```

The dP MMA correctly produces zero for n>0 columns (tidx=1..32 verified).
For the n=0 column, only V[0] of tidx=0 reads the correct value. V[1..] of
tidx=0 read garbage that does not match `dP[m, n]` at any (m, n).

### Comparison with tStS read (which works)

Set `Q[i,d] = i/sqrt(D), K = ones` so `S[m, n] = m·sqrt(D)` (varies in m).
For tidx=0 V[0..7] kernel reads:

```
[S] tidx=0 V[0..7]  coord=(0,0)..(0,7)  S=0, 5.66, 11.31, ..., 39.59
                                          (= m * sqrt(D) for m=0..7)  ✓ all correct
```

Same partition (whether `copy_utils.make_tmem_copy` or
`tcgen05.make_tmem_copy(atom, tStS)`), same per-thread V-strides
`(1, 65536)` — and tStS reads land at the correct cells.

### Why this is mysterious

- `tStS` (QK^T MMA's C-frag) and `tdPtdP` (V@dO^T MMA's C-frag) have
  identical layout `((128,64),1,1):((65536,1),0,0)`.
- The kernel's `partition_S` on both with the same `thr_copy_t2r` gives
  identical per-thread layouts `(((32,32),1),1,1,1):(((1,65536),0),0,0,0)`.
- Both MMAs are built via
  `make_trivial_tiled_mma(fp32, K, K, fp32, ONE, (tile_n, tile_m))` —
  byte-identical setup.
- bf16 reads both correctly.

**So the TMEM bytes at addresses 1..7 of `tdPtdP` are GARBAGE for fp32 even
though tStS at those same addresses is CORRECT.** The partition isn't
deciding what to read incorrectly; it's the underlying TMEM cells that
hold wrong data.

### Working hypothesis

TF32 MMA's actual cell-to-TMEM-address write pattern for the V@dO^T MMA
differs from what `make_fragment_C` claims (and differs from the QK^T
MMA's pattern, despite identical setup). Either:

1. **MMA-private cell layout for some operand combinations.** SM100 TF32
   m64nNk8 may have a non-`make_fragment_C` write pattern for certain
   operand source / N values.
2. **Pipeline/sync gap** specific to TF32 + V@dO^T. `pipeline_dP.consumer_wait`
   may signal before all per-warp scratch reaches main TMEM. (Less
   likely because the dump shows *consistent* wrong values, not jittery
   ones.)
3. **A bug in `make_trivial_tiled_mma` for one of the two MMAs.** The
   setup is identical so this would be a cute_dsl issue.

### Why A1+ fix attempt didn't work

Session 7 tried: switch t2r/r2t to `tcgen05.make_tmem_copy(atom, tensor)`
+ per-element write to sdS via `tScS_t2r` coords.

Mechanical pieces all worked:
- Compile succeeds (when `get_slice` is inlined into the constructor — a
  named intermediate variable triggers MLIR `scf.yield` "live user"
  errors).
- Marker probe (write `100.0` everywhere) propagates to dQ output
  (`dQ abs max = 371`), confirming the write path functions.

But the register `dS` values fed to the writes come from the **dP read
that's still broken**. So per-element writes faithfully propagate
garbage. Fixing Bug #2 requires either a correct dP TMEM partition or
sidestepping the TMEM A-operand altogether.

---

## What was disproved (don't chase these)

### LSE pairing is NOT the bug

The original session-1 diagnosis blamed
`thr_copy_t2r.partition_D(thr_mma_S.partition_C(sLSE_2D))`. Verified
incorrect via in-kernel `cute.printf`:

Setup `Q[i,d] = i/sqrt(D), K=ones` so `LSE[i] = log(L)+i` (varies per i):

```
[LSE] tidx=0 stage=0  coord[0]=(0,0) coord[1]=(0,1)
                       S[0]=0      LSE[0]=6.000  (= log2(L) + 0)  ✓
                       S[1]=5.656  LSE[1]=7.443  (= log2(L) + 1)  ✓
```

Per-V LSE values pair correctly with the S element at coord (n, m=v).
P values come out uniform `1/L = 0.015625` for Q=0. The S → LSE → P
pipeline is functioning end-to-end.

### Option A "tcgen05.make_tmem_copy everywhere" doesn't help by itself

The tcgen05-derived partition has the same V-stride `(1, 65536)` as the
copy_utils one — it adds the 2 m-stage outer iteration but doesn't change
which cells V[0..31] of stage 0 read. So switching the partition reads
the same garbage values. Visible only when paired with per-element sdS
writes (where the actual mismatch is upstream).

---

## Next-step options

### Option A — empirical TMEM cell mapping

Dump tdPtdP via a battery of read partitions (different atoms, different
thread layouts) under the V=e_0 probe until one matches `dout[m, 0]`. The
mapping that works is the dP MMA's actual hardware pattern. Use that
partition.

This is mechanical search — viable but slow.

### Option B — A3 (P/dS through SMEM end-to-end)

Restructure the bwd kernel so that:
- P is computed and written directly to SMEM (skip TMEM r2t)
- dV/dK MMAs use SMEM A-operand (skip TMEM read)
- dP MMA still produces tdPtdP, but it's consumed immediately into dS
  (compute dS = P·(dP−D)) without reading dP through the broken partition

The dP MMA→register read might still use the broken partition, but if
the dS computation overlaps the dP correctness issue with something else
(e.g., always reads only the diagonal where pattern is correct), maybe
the bug only affects parts that the new path avoids.

Bigger refactor. Cleanest end-state.

### Option C — upstream cutlass-dsl query

Ask the cutlass-dsl team: for SM100 TF32 m64nNk8 MMA with output
(M=128, N=64), is there a documented but non-`make_fragment_C`
cell-to-address pattern? Compare with bf16 m64nNk16 to confirm the
hypothesis.

Fast if upstream knows the answer; otherwise wait time.

### Option D — SASS / hardware-level inspection

Dump the SASS of the fp32 bwd kernel and trace the MMA instruction's
actual store pattern. Compare with bf16's SASS. Confirm whether the cell
permutation is in the hardware or in cute_dsl's layout claim.

Investigative; uses `compute-sanitizer` + `nvdisasm`.

---

## Code state in tree

- `FA4_fp32/kernels/flash_bwd_sm100.py` — **Bug #1 fix applied**
  (`dQacc_reduce` reshape uses `cute.size(tdQrdQ_t2r) // num_chunks` for
  fp32). No other changes from the pre-debugging baseline.
- `FA4_fp32/bwd/test_q0.py` — Q=0 uniform-P regression. dV under uniform
  P is permutation-invariant, so this verifies the dV path (it passes).
- `FA4_fp32/bwd/debug_probe.py` — structured-input probes:
  `probe_zero_q`, `probe_small_q` (uniform-P regimes), `probe_single_v`
  (V=e_0 sparse), `probe_eye_dout` (eye-like dout). Use to surface fp32
  errors in a controlled regime.

bf16 9/9 PASS preserved.

---

## Session history (chronological detail)

Detailed trace of every hypothesis tried, evidence gathered, and
dead-ends hit, kept for posterity. Most of this is now superseded by the
bug catalog above — but useful for understanding what *not* to retry.

### Session 1 — original "LSE pairing" hypothesis (DISPROVED)

Initial diagnosis (commit `70ac606`). Set Q=0 to make P uniform; dV
matched the closed-form `(1/L)·Σᵢ dO[i, d]` within TF32 noise. Concluded
that dV path was correct and the bug was in P-computation, specifically
LSE pairing under
`thr_copy_t2r.partition_D(thr_mma_S.partition_C(sLSE_2D))` composition.

Session 5 disproved this with direct `cute.printf` traces (LSE values
pair correctly).

### Session 2 — Q=0 verification + option-B half-fixes

`test_q0.py` written; confirmed fp32 dV correct under uniform P with bf16
partition. Tried partition_D-only (no partition_C composition) for fp32;
gave V-stride 0 → broadcast same LSE to all V slots → semantically broken.

### Session 3 — Option A (tcgen05 partition) r2s blocker

Tried `tcgen05.make_tmem_copy(atom, tStS)` for t2r/r2t. fp32 r2s downstream
broke: `get_smem_store_op` returned a 1×1 atom paired with a rank-3 tiler
that wouldn't tile sdS_epi cleanly. Various atom widths (1×1, 128-bit,
get_tmem_load_op-derived) all blew out per-thread iter count to thousands
with OOB write addresses.

### Session 4 — TMEM-side-only Option A (partial)

Applied tcgen05 partition to t2r/r2t only; kept bf16-style r2s. Compiled
and ran. Random-Q test: dV error 0.9→0.5 (~40% improvement), dK similar
improvement, dQ unchanged (~1.5). LSE *appeared* to be the fix, but with
the Q=0 test the apparently-fixed dV now turned to all-zeros for L≤128
(LSE saturation under Q=0 + new partition). Documentation accepted at
the time.

### Session 5 — cute.printf traces; LSE disproved

Instrumented compute_loop with per-thread `cute.printf` dumps. Verified
LSE pairing is correct (see [LSE] trace above). Located Bug #1
(`dQacc_reduce` reshape over-stride) by following the dQ cols-16..31=0
pattern through the kernel.

### Session 6 — Bug #2 located via S-vs-dP discrepancy

Used `S = m·n` probe to verify tStS reads correctly for fp32 (V[v] reads
the correct cell at varying m for fixed n_t). Used V=e_0 probe to expose
that tdPtdP reads return garbage for tidx=0 V[1..] in fp32. Confirmed
bf16 same setup reads correctly. Updated documentation with the new
hypothesis: cell-to-address mapping mismatch on dP MMA's TMEM output.

Committed Bug #1 fix (commit `eb7245d`).

### Session 7 — A1+ attempted; mechanical pieces work; deeper bug remains

Implemented tcgen05 t2r/r2t + per-element sdS write via `tScS_t2r` coords.
Discovered:
- Storing `tiled_t2r = tcgen05.make_tmem_copy(...)` as a named variable
  before `.get_slice(tidx)` triggers MLIR `scf.yield` "live user" errors.
  Inlining `.get_slice` into the constructor call fixes the compile.
- sdS_2d layout
  `((32, 2), (8, 16)):((1, 4096), (32, 256))` (for fp32 D=32) is a clean
  2D view of sdS's native rank-3 layout, byte-compatible with what the
  dQ MMA reads.
- A marker write (`100.0` to every cell) propagates to dQ output
  (`abs max = 371`) — write path functions.

But dQ values otherwise match the unfixed baseline because **the dS
register values being written are themselves garbage** (read from broken
tdPtdP). A1+ is blocked on the same Bug #2 root cause.

Committed session 7 docs (commit `e1daee4`).
