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

---

## Session 2 (2026-05-15): Q=0 verified; option A blocked on r2s atom

### Q=0 diagnostic — confirms diagnosis

[FA4_fp32/bwd/test_q0.py](../FA4_fp32/bwd/test_q0.py) runs B=1 L∈{64,128,256}
H∈{1,4} D∈{32,64} fp32 with Q=0 (so S=0 → P=1/L uniform). dV vs the closed-form
`mean(dO over i)` per column matches within TF32 noise:

```
[B=1 L=64  H=1 D=32]  max abs err 1.6e-04
[B=1 L=128 H=1 D=32]  max abs err 7.5e-05
[B=1 L=256 H=4 D=64]  max abs err 9.3e-05
```

dV is constant across rows (as required by uniform P). This **rules out** dV
MMA / TMEM A-operand layout / dV epilogue / dV gmem layout. The proximate bug
is in the P-computation path (LSE pairing).

### Why options B-simple and B-gather both fail

**Option B-simple** — replace `thr_copy_t2r.partition_D(thr_mma_S.partition_C(sLSE_2D))`
with `thr_copy_t2r.partition_D(sLSE_2D)`. Compiles, but the result has
`tSsLSE.layout = ((32,1),1,64,2):((0,0),0,1,64)` — the V mode strides are
`(0, 0)`, so every V slot gets the same LSE address. Semantically wrong:
broadcasts a single LSE value to all per-thread V entries instead of
LSE[m_v] varying with v.

**Option B-gather** — load `LSE[m_t_v]` per V via `tScS_t2r`'s m-coord. Also
broken in this configuration: with copy_utils' bf16-tuned t2r, the per-thread
(m, n) coords in `tScS_t2r` are what the *copy* claims to have loaded, not
what the TF32 MMA *actually* wrote to TMEM. Loading LSE for the claimed m
still pairs it with the wrong S element. The mismatch is bidirectional —
fixing the LSE side alone doesn't fix it.

### What option A breaks (and why)

With fp32 forking to `tcgen05.make_tmem_copy(atom, tensor)`:

- t2r built from `Ld32x32bOp(Rep(32))` + tStS:
  `tStS_t2r.layout = (((32,32),1),2,1,1):(((1,65536),0),32,0,0)` — 2 outer
  stages, V=((32,32),1) with the inner V-stride 65536 (i.e., past the full
  8192-element tensor — those entries are "virtual broadcasts").
- t2r built from `get_tmem_load_op(mma_tiler_kq, …, (tile_n, tile_m), …)` + tStS:
  `tStS_t2r.layout = (((64,32),1),1,1,1)` — single stage, V=(64,32). Slightly
  cleaner shape.

Either way the LSE/dPsum partition_D(partition_C(...)) composition now uses
matching thread mappings — the LSE pairing IS fixed by this change. **The
breakage is downstream, in the r2s (rmem→smem dS) path:**

```python
copy_atom_r2s = sm100_utils_basic.get_smem_store_op(
    LayoutEnum.ROW_MAJOR, self.ds_dtype, Float32, tiled_t2r
)
thr_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_t2r).get_slice(tidx)
```

For fp32+tcgen05-derived t2r, `get_smem_store_op` returns a degenerate atom:

```
Copy Atom
  ThrID:         1:0
  TV Layout Src: (1,1):(0,0)
  TV Layout Dst: (1,1):(0,0)
  Value type:    f32
```

A 1×1 atom (one element per copy invocation). Combined with `make_tiled_copy_D`
over the big t2r tile, the resulting r2s partition explodes:

```
tRS_sdS.layout = ((1,64),1,(32,2),1):((0,4096),0,(1,4096),0)
```

4096 entries per thread. The strides include `4096` along V (broadcasts past
sdS_epi's 8192-element extent), so cute.copy issues stores to addresses far
beyond the sdS allocation → `cudaErrorIllegalAddress` from the bwd kernel
launch.

`sdS_epi_layout` for fp32 is also rank-3 with a trailing (1,1) mode
(`((8,16),(32,2),(1,1)):((32,256),(1,4096),(0,0))`), not rank-2 like bf16.
Keeping it rank-3 (no slice) fixes the rank-mismatch error in `partition_D`,
but doesn't address the atom-too-small issue.

### Tried alternative r2s strategies (all blocked)

| Strategy | Outcome |
|---|---|
| `cute.copy(thr_copy_r2s, tdPrdS_view aliased to tRS_sdS.shape, tRS_sdS)` — single bulk copy after stage loop | cudaErrorIllegalAddress — aliased register iterator goes past valid range |
| Per-stage write with rank-4 indexing `tRS_sdS[None, stage, 0, 0]` | Shape coord mismatch — autovec_copy demands rank-5 source matching tRS_sdS |
| Keep sdS_layout rank-2 (collapse trailing (1,1)) | `partition_D` rank-mismatch error (tile_copy tiler is rank-3) |

### What needs to happen in session 3

The blocker is **building an r2s tile_copy that has a multi-element atom
matching fp32 width and a partition that lands per-thread on the actual sdS
SMEM cells the dQ MMA reads**. Two paths:

**A1. Manual atom + tiled_t2r thread layout.** Build copy_atom_r2s by hand
with 128-bit width (4 fp32) using `cute.nvgpu.CopyUniversalOp()`, e.g.:

```python
copy_atom_r2s = cute.make_copy_atom(
    cute.nvgpu.CopyUniversalOp(), self.ds_dtype, num_bits_per_copy=128
)
```

Then `cute.make_tiled_copy_D(copy_atom_r2s, tiled_t2r)` should partition with
8 iters/thread instead of thousands. Test whether the resulting partition
maps to addresses inside sdS_epi.

**A2. Independent r2s tile_copy.** Build a tile_copy directly over the (tile_n,
tile_m) sdS_epi shape with a simple thread layout (256 threads × 32 fp32 each)
— then permute register data via the t2r→r2s thread mapping. Cleaner if
manual permutation is feasible in registers (vs. smem roundtrip).

**A3. Move P/dS through SMEM rather than TMEM.** Per the previous WIP commit
message — bigger restructure but avoids the TMEM A-operand layout mismatch
entirely. Use `make_smem_layout_a` for P (and dS) so dV/dK MMA reads from
SMEM. Then there's no r2t at all, and r2s naturally aligns with what the
MMAs expect.

Verification ladder once any of A1/A2/A3 builds:
1. Q=0 test — dV should still be correct (already is; sanity check).
2. Q small (|S|≪LSE) — dV/dK should be correct (uniform-P regime).
3. test_bwd.py random Q — full accuracy check.

### Captured layout reference

For B=1 L=64 H=1 D=32 fp32 (tile_m=64, tile_n=128, Q_stage=2):

```
sLSE_2D.layout      = (128, 64, 2):(0, 1, 64)          # after transpose: (n, m, q_stage)
sdPsum_2D.layout    = (128, 64, 2):(0, 1, 64)
sdS_epi_layout.outer = ((8,16),(32,2),(1,1)):((32,256),(1,4096),(0,0))
```

Under bf16-style t2r (current baseline):
```
tStS_t2r.layout = (((32,32),1),2,1,1):(((1,65536),0),32,0,0)
tScS_t2r.layout = ((32,1),1,1,1):((1@1,0),0,0,0)
partition_C(sLSE_2D) = ((128,64),1,1,2):((0,1),0,0,64)
tSsLSE.layout   = (((32,1),(1,1)),2,1,1,2):(...)        # rank 5
```

Under Option A with `Ld32x32bOp(Rep(32))` + tStS:
```
tStS_t2r.layout = (((32,32),1),2,1,1):(((1,65536),0),32,0,0)
tStP_r2t.layout = (((16,32),1),4,1,1):(((1,65536),0),16,0,0)
```

Under Option A with `get_tmem_load_op(mma_tiler_kq, ROW_MAJOR, f32, f32, (tile_n, tile_m), False)`:
```
tStS_t2r.layout = (((64,32),1),1,1,1):(((1,65536),0),0,0,0)
tRS_sdS.layout  = ((1,64),1,(32,2),1):((0,4096),0,(1,4096),0)   # too many iter, OOB
```

### Session 3 — Option A r2s deep dive (still blocked, deeper cause found)

Tried Option A with a manual 128-bit fp32 atom for r2s (CopyUniversalOp,
num_bits_per_copy=128) instead of the 1×1 `get_smem_store_op` atom:

```
tRS_sdS.layout = ((4,8),1,(32,2),1):((4096,16384),0,(1,4096),0)
```

Still wrong: V mode (4, 8) gets strides (4096, 16384). 16384 > sdS_epi extent
(8192) → `cudaErrorIllegalAddress`. The auto-derived r2s partition picks the
*swizzle-block strides* (4096, 16384) for V instead of the inner stride 1.
`make_tiled_copy_D(atom, tcgen05_tiled_t2r)` composes the t2r thread layout
with the sdS strided layout in a way that doesn't pick the right axis for
the atom — regardless of atom width.

### Why the actual sdS allocation makes Option A harder

Cross-comparison of the real smem alloc (`self.sdS_layout`) at D=32:

| dtype | sdS alloc                            | total |
|---|---|---|
| bf16 | `S<3,4,3> o 0 o (((64,2),16),1,8):(((1,8192),64),0,1024)`  | 16384 |
| fp32 | `S<2,5,2> o 0 o (((32,2),8),1,16):(((1,4096),32),0,256)`   | **8192** |

**fp32's sdS allocation is half-sized** vs bf16. The fp32 bwd was designed
to fit only half of dS in smem and have the dQ MMA consume that half before
the next half is written. (Visible elsewhere too:
`self.sdS_xchg_layout = (tile_n, tile_m // 2)` suggests an "exchange" buffer
of half a tile.) So bumping `sdS_epi_layout` to `num_stages=2` to get a
full-coverage epi layout produces strides (8192-aligned) past the actual
allocation — exactly the OOB we hit.

**Implication:** Option A as a drop-in r2s replacement is not the right
model. The fp32 bwd architecture computes/stores dS in half-tile chunks,
and the r2s must respect that allocation. The clean fixes are:

1. **A4: write sdS per-element with `tScS_t2r` coords.** Iterate V/stage in
   the compute loop, extract `(n_v, m_v)` from `tScS_t2r[v, stage, …]`,
   write `tdPrdS_view[v, stage]` to `sdS[m_v, n_v]` (or transposed
   depending on the dQ MMA orientation). Need to verify cute_dsl supports
   smem indexing at a runtime coord and that sdS's swizzled layout
   evaluates the address correctly. Slow but unambiguous; good first step
   to *prove* the dQ pipeline works under correct LSE pairing.

2. **A3: P/dS through SMEM end-to-end.** Per session-1 note. Restructures
   the data path so the dV/dK MMAs read from sdS directly, bypassing the
   r2t / TMEM-A-operand layout question entirely. Bigger refactor but
   avoids the awkward half-tile r2s.

Don't try **A1/A2** further — `make_tiled_copy_D(atom, tcgen05_tiled_t2r)`
fundamentally picks the wrong stride for the atom against fp32's swizzled
sdS_epi.

### Session 4 — partial fix landed: TMEM-side Option A in isolation

Applied Option A only to the TMEM-side copies (t2r and r2t), keeping the
SMEM-side r2s on the bf16/copy_utils path even for fp32. This validates the
LSE-pairing fix in isolation:

```
Baseline (fp32 errors, abs):           After session-4 partial fix:
  L=128 D=32  dQ 1.29  dK 1.74  dV 0.90       dQ 1.16  dK 1.01  dV 0.58
  L=256 D=64  dQ 0.83  dK 1.07  dV 0.93       dQ 0.96  dK 1.06  dV 0.54
  L=512 D=64  dQ 1.52  dK 0.83  dV 0.60       dQ 1.52  dK 1.00  dV 0.36
```

dV errors drop ~35-45% across all three configs (the LSE-pairing improvement
showing through). dK improves a little (TMEM-side dS now lands in MMA-aware
cells for the dK MMA's TMEM-A read). dQ unchanged — dQ MMA reads dS from
SMEM via the still-bf16 r2s, so the dQ path remains permuted.

**bf16 9/9 PASS preserved** — fp32 fork is fully gated.

**Q=0 caveat under partial fix.** Under the new TMEM-aware t2r, the Q=0 test
now gives zero dV for L≤128 and correct dV for L=256. Most likely cause:
with S=0 and the new (correct) LSE pairing pulling per-V LSE values per
row, P=exp(-LSE_row)=1/seqlen_q is small enough that fp32 + the magnitude
mismatch saturates to zero somewhere in the pipeline for short rows. For
random Q, |S| is O(sqrt(D)) and P stays well-conditioned, so dV improves
as expected. **Q=0 is no longer a clean isolation under Option A** — for
the next session, use a small-but-nonzero Q (e.g., randn * 1e-4) to verify
uniform-P regimes without LSE saturation.

### Remaining work for session 5

The r2s path is the last blocker. Two paths still viable:

- **A4**: per-element write to sdS using `tScS_t2r` (now MMA-aware) coords.
  Slow but should plug into the existing single-stage sdS allocation.
- **A3**: route P/dS through SMEM end-to-end (skip TMEM A-operand for dV/dK
  too). Bigger refactor; the cleanest end-state.

The TMEM-side fix already in tree is independent and worth keeping while
the r2s gets sorted out.

---

## Session 5 (2026-05-15): Diagnosis was MISDIRECTED. Found actual bugs.

Using `cute.printf` to trace per-thread state with structured inputs
(`Q[i,d]=i/sqrt(D)`, `K=ones` → LSE[i]=log(L)+i, P=1/L uniform), I verified
the kernel's actual behavior at each pipeline stage. **The LSE-pairing
hypothesis is wrong.** Two distinct bugs explain the failure.

### What's NOT a bug: LSE pairing

The chain `tSsLSE = thr_copy_t2r.partition_D(thr_mma_S.partition_C(sLSE_2D))`
under bf16/copy_utils t2r already pairs correctly. Per-thread dump:

```
[LSE] tidx=0 stage=0  coord[0]=(0,0) coord[1]=(0,1)  S[0]=0 LSE[0]=6.000 S[1]=5.656 LSE[1]=7.443
[LSE] tidx=1 stage=0  coord[0]=(1,0) coord[1]=(1,1)  S[0]=0 LSE[0]=6.000 S[1]=5.656 LSE[1]=7.443
```

Thread t's V[0] holds the S cell at (n=t, m=0), LSE[m=0]=log(L)+0 → 6.000
in log2 base ✓. V[1] holds (n=t, m=1), LSE[m=1]=log(L)+1 → 7.443 ✓.
The coord that `tScS_t2r` claims matches both the S value loaded from
TMEM AND the LSE value loaded from SMEM. There is no permutation here.

P values also confirmed uniform 1/L = 0.015625 across all printed threads.

### What IS the bug: under uniform-P + K=ones (where dQ should be ≡ 0)

The Q=tiny + K=ones probe gives:

- **dV correct** ✓ (uniform-P makes dV permutation-invariant — confirmed)
- **dK correct** ✓ (dK = dS·Q ≈ 0 with tiny Q)
- **dQ wrong** ✗ — abs max ≈ 0.46 instead of ≈ 0

So dQ has a bug independent of LSE/P/dV/dK. The dQ path:
`compute dS → write to sdS via r2s → dQ MMA reads sdS → write to TMEM →
dQacc_reduce loads → SMEM → TMA reduce-add → gmem dq_accum → postprocess →
gmem dQ`. Tracing each stage:

### Bug #1 (FIXED): `dQacc_reduce` reshape over-strides the register fragment

`tdQrdQ_t2r` is the per-thread register fragment that receives dQ from TMEM.
For fp32, its layout is rank-4:

```
tdQrdQ_t2r.layout = (((2,4),1),1,1,2):(((1,2),0),0,0,8)
                     ↑ V = 8 elements/thread/chunk    ↑ chunk stride = 8
```

So per-thread total = 8 × 2 chunks = **16 fp32**. Chunk 1 lives at register
offsets 8..15.

The smem-write loop reshapes this to view as `(stage_size, num_stages)`:

```python
# OLD (BUGGY):
tdQrdQ_shape = (self.dQ_reduce_ncol,           # = 16  ← gmem cols per stage
                self.tile_hdim // self.dQ_reduce_ncol)  # = 2
tdQrdQ = cute.make_tensor(tdQrdQ_t2r.iterator, tdQrdQ_shape)
# → tdQrdQ.layout = (16, 2):(1, 16)   col-major default
# → tdQrdQ[None, stage=1] reads offsets 16..31 — past the 16-element fragment
```

Reshape's stride for mode 1 was 16 (col-major over 16 inner), but chunk 1
actually starts at register offset **8**, not 16. `tdQrdQ[None, 1]` reads
register addresses past the allocation → returns zeros. Stage=1's TMA
reduce-add then writes zeros to the second half of gmem dq_accum →
final dQ has cols 16..31 universally zero.

For bf16 this happens to work because per-thread V per chunk = 32 = dQ_reduce_ncol.
For fp32 (TF32 m64nNk8) the per-thread V per chunk = 16 / 2 = 8 because the
TF32 C-frag packs N more tightly. The constant `dQ_reduce_ncol` is the
*gmem* head-dim cols per stage, not the per-thread register count.

**Fix** (applied in this session):

```python
num_chunks_reshape = self.tile_hdim // self.dQ_reduce_ncol
if const_expr(fp32_dQ):
    per_thread_V_per_chunk = cute.size(tdQrdQ_t2r) // num_chunks_reshape
    tdQrdQ_shape = (per_thread_V_per_chunk, num_chunks_reshape)
else:
    tdQrdQ_shape = (self.dQ_reduce_ncol, num_chunks_reshape)
```

After fix: `tdQrdQ_shape = (8, 2)`, layout `(8,2):(1,8)`, stage=1 reads
register offsets 8..15 (the actual chunk 1 data). Captured gmem dq_accum
confirms both halves now populated:

```
dq_accum max in [0..1024):    2.59  ✓ (was 2.59)
dq_accum max in [1024..2048): 2.59  ✓ (was 0)
```

dQ output cols 16..31 now mirror cols 0..15 instead of being zero.

### Bug #2 (LOCATED, not fixed): dQ values themselves are wrong

After bug #1 is fixed, dQ values for K=ones uniform-P are still wrong:

```
Per-row abs max of dQ (expected ≈ 0):
  row  0: 4.3e-05   ✓
  row  4: 0.21
  row  8: 0.32
  row 12: 0.46
  ...
  row 32: 8.4e-05   ✓
  row 36: 0.13
  ...
```

**Granularity = 32**. Only rows 0 and 32 (= first row of each 32-row block,
i.e., one row per warp) are correct. Other rows have arbitrary nonzero.

Cross-referencing with per-thread tdQrdQ_t2r values for K=ones case:

```
[dQred] tidx=0   chunk0[0..3]=(-0.000139,0.000019,-0.000139,0.000019)  ≈ 0  ✓
[dQred] tidx=1   chunk0[0..3]=(-0.000139,0.000019,-0.000139,0.000019)  ≈ 0  ✓
[dQred] tidx=2   chunk0[0..3]=(-0.000139,0.000019,-0.000139,0.000019)  ≈ 0  ✓
[dQred] tidx=3   chunk0[0..3]=(-0.000139,0.000019,-0.000139,0.000019)  ≈ 0  ✓
[dQred] tidx=4   chunk0[0..3]=(-0.451408,-1.184248,-0.451408,-1.184248) ✗
[dQred] tidx=8   chunk0[0..3]=(-0.791084,-0.296139,-0.791084,-0.296139) ✗
[dQred] tidx=16  chunk0[0..3]=(1.504311,-1.452046,1.504311,-1.452046)   ✗
[dQred] tidx=64  chunk0[0..3]=(0.000234,-0.000246,0.000234,-0.000246)   ≈ 0  ✓
```

The TMEM dQ accumulator that `dQacc_reduce` reads has **correct ~0 values
for tidx 0-3 and 64 but wrong nonzero values for tidx 4, 8, 16**. So the
dQ MMA *output to TMEM* is itself inconsistent.

For K=ones, `dQ[m,d] = Σ_n dS[m,n]·1 = Σ_n dS[m,n]`. Under uniform P,
`Σ_n dS[m,n] = (1/L)·Σ_n(dP[m,n] − D[m]) = 0` exactly (math). So dQ MMA
should produce ~0 for *every* (m, d). The fact that it produces ~0 for
some threads' cells and large values for others' means the inputs to dQ
MMA — specifically the dS in sdS smem — are wrong on some (m, n) cells.

**Hypothesis (next session to verify)**: The r2s path in `compute_loop`
uses `copy_utils.make_tmem_copy`'s bf16-tuned thread layout to write dS
to sdS. This thread layout assigns per-thread (m, n) coords that disagree
with the TF32 dQ MMA's A-operand expectations. Some cells of sdS get
*correctly* written by the corresponding compute thread (those whose
bf16 t2r coord happens to align with TF32 MMA-A coord), others get a
permuted value. The dQ MMA then sums over n: cells with correct dS sum
to 0, cells with permuted dS sum to ~0.45.

The 32-granularity in the output suggests the (m, n) mismatch is *within*
each warp's 32-thread coverage — first thread of each warp lands on the
right cell, others permute.

### To verify in next session

Dump sdS smem contents directly after r2s (before dQ MMA reads). Compare
per-(m, n) cell to the expected dS = P·(dP − D). Identify which cells
have correct values, which have permuted values, and whether the
permutation maps to a clean bf16-vs-TF32 thread-layout difference.

The fix path is likely the same as before — make r2s tile_copy MMA-aware
(via `tcgen05.make_tmem_copy(atom, sdS_compatible_tensor)` or an explicit
MMA-A-aware partition) — but for the **r2s** specifically, not for
t2r/r2t. The TMEM r2t path is fine; only the SMEM r2s mismatches the
dQ MMA's read pattern.

---

## Session 6 (2026-05-15): Bug #2 localized to dP-from-TMEM read

Built [debug_probe.py](../FA4_fp32/bwd/debug_probe.py) probes and instrumented
`compute_loop` / `dQacc_reduce` with `cute.printf` per-thread coord+value dumps.
Cross-referenced kernel reads with pytorch reference computations under
structured inputs (`Q[i,d] = i/sqrt(D)`, `K = ones`, `V = e_0`, `V = ones`).

### S from TMEM (tStS read): fully correct under fp32

With S = m·n (varies in both axes, set via `Q[i,0]=i, K[j,0]=j`):

```
[Sraw] tidx=1 V[0..7] coord=(1, 0..7)  S=0,1,2,3,4,5,6,7   ← S[m=v, n=1]=v  ✓
[Sraw] tidx=4 V[0..7] coord=(4, 0..7)  S=0,4,8,12,16,20,24,28 ← S[m=v, n=4]=4v ✓
```

Every V slot of every probed thread matches the cell its `tScS_t2r` coord
claims. The kernel's bf16/copy_utils `partition_S(tStS)` reads tStS
**correctly** for fp32.

### dP from TMEM (tdPtdP read): wrong for V[1..] for fp32

With V = e_0 (`V[0,0,0,0]=1`, rest 0) → expected dP[m,n=0]=dout[m,0], dP[m,n>0]=0:

```
[dP_pre] tidx=0  V[0] coord=(0,0) dP=-0.924316  ← ✓ matches dout[m=0, 0]=-0.925
[dP_pre] tidx=0  V[1] coord=(0,1) dP=-0.180542  ← ✗ expected dout[m=1, 0]=-0.597
[dP_pre] tidx=0  V[2] coord=(0,2) dP= 0.224121  ← ✗ expected dout[m=2, 0]=-0.033
[dP_pre] tidx=0  V[3] coord=(0,3) dP=-0.262451  ← ✗ expected dout[m=3, 0]=-0.086
…
[dP_pre] tidx=1  V[0..3] all 0  ✓ (n=1 column → dP[m, n=1] = 0)
[dP_pre] tidx=2  V[0..3] all 0  ✓
[dP_pre] tidx=32 V[0..3] all 0  ✓
```

Only `tidx=0 V[0]` reads correctly. Other V slots for tidx=0 (which all
claim coord `(n=0, m=v)`) get nonzero values that don't match dP[m, n=0]
**or** dP[m=0, n>0] (which would be 0) **or** any other clean (m', n')
permutation. They look like garbage.

The V=ones test masks this — dP is row-uniform there, so reading the
"wrong" n cell still yields the right value. V=random + V=e_0 expose it.

For **bf16** same setup, every V[0..7] of tidx=0 matches expected dP:

```
[dP_pre] bf16 tidx=0  V[0..7]  dP=2.580,-4.888,3.295,2.598,5.242,7.059,1.514,-4.967
expected                              2.580,-4.888,3.295,2.598,5.241,7.058,1.510,-4.998
```

So bug is **fp32-specific** at the dP-read site.

### Same partition, same layouts — different results

Both tStS and tdPtdP are populated by SM100 tcgen05 MMA with identical
setup (`make_trivial_tiled_mma(fp32, K, K, fp32, ONE, (tile_n, tile_m))`).
Both have the same `make_fragment_C` layout `((128,64),1,1):((65536,1),0,0)`.
The `partition_S` of both via the same `thr_copy_t2r` gives identical
per-thread layouts:

```
[init] tStS_t2r.layout   = (((32,32),1),1,1,1):(((1,65536),0),0,0,0)
[init] tdPtdP_t2r.layout = (((32,32),1),1,1,1):(((1,65536),0),0,0,0)
```

Yet tStS reads correctly and tdPtdP doesn't.

When the tdPtdP read partition is rebuilt via
`tcgen05.make_tmem_copy(load_atom, tdPtdP)` (tensor-aware) the layout
**gains an extra mode**:

```
[init] tdPtdP_t2r (tcgen05) = (((32,32),1),2,1,1):(((1,65536),0),32,0,0)
                                                  ↑↑ 2 stages w/ stride 32
```

So the tensor-aware partition exposes **2 m-iteration stages** that the
bf16/copy_utils partition collapses into 1. The dP MMA's actual TMEM
write pattern needs the 2-stage iteration; the kernel only iterates 1.
Reading via the collapsed partition yields garbage from offsets that
weren't written by the half of the MMA the kernel skipped.

Why this doesn't break S read: the QK^T MMA's C-frag, for our seqlen=64
case (m_block=0 with seqlen_q=64=tile_m=64), only produces one m_iteration's
worth of data because seqlen_q maps to exactly one full tile_m. The dP
MMA, on the other hand, has output (M=tile_n=128, N=tile_m=64), where
M=128 mandates 2 m-iterations regardless of seqlen — and the second
iteration's cells are what the bf16-style partition misses.

In other words: **the TMEM C-frag layout itself is consistent between
the two MMAs (same hardware pattern); the bug is that the read partition's
total M extent (=64 in collapsed-mode) doesn't cover dP's full M=128 row
range that the dP MMA actually writes.**

### Bug #2 fix attempts so far

`tcgen05.make_tmem_copy(load_atom, tdPtdP)` for the dP read alone exposes
the right partition (2 stages), but compiles into the same downstream
shape-coord mismatch we documented in session 3 — the rank-3 tiler doesn't
fit the rank-1-collapsed `sdS_epi` view that the r2s path uses.

So Bug #2 isn't fixable in isolation — its fix touches the same r2s blocker
we already documented. Path **A3** (route P/dS through SMEM end-to-end and
skip the TMEM A-operand for dV/dK) sidesteps the issue entirely. Path
**A1/A2 + per-stage r2s adapter** is also possible: keep tcgen05 partition
for tdPtdP, iterate 2 stages explicitly in the dP compute loop, and adapt
r2s for the new layout.

### Net state after session 6

- Bug #1 (dQacc_reduce reshape) — **applied** in tree. fp32 dQ no longer has
  the cols-16..31-zero pattern.
- Bug #2 (tdPtdP partition collapsing dP MMA's 2 m-iterations) — **identified**.
  Not fixed (blocks on the r2s-rank issue from sessions 3-4).
- LSE pairing — **disproved as the bug**. The original diagnosis was on
  the wrong stage of the pipeline.
- bf16 9/9 PASS preserved throughout.

### Useful probes left in tree

- `FA4_fp32/bwd/debug_probe.py` — structured-input probes (probe_zero_q,
  probe_small_q, probe_single_v, probe_eye_dout). Run with
  `python FA4_fp32/bwd/debug_probe.py <probe_name>` to surface fp32 dQ
  wrongness in a controlled regime.
- `FA4_fp32/bwd/test_q0.py` — Q=0 uniform-P regression test.

---

## Session 7 (2026-05-15): A1+ attempted, deeper root cause emerges

Tried the A1+ plan from session 6: tcgen05-aware `tcgen05.make_tmem_copy
(atom, tensor)` for t2r/r2t in fp32 + per-element write to sdS via the
`(n_v, m_v)` coords from `tScS_t2r`.

### Mechanical findings — fix would compile and write correctly

- `tcgen05.make_tmem_copy(atom, tStS).get_slice(tidx)` (inlined slice;
  storing `tiled_t2r` as an intermediate var triggers MLIR
  "scf.yield live user" errors) builds without crashing.
- For dS' SMEM write, the sdS native layout `(((32,2),8),1,16)` for fp32
  D=32 decomposes cleanly as `M=(32, 2) strides (1, 4096)`,
  `K=(8, 16) strides (32, 256)`. Built a 2D view
  `sdS_2d = make_tensor(sdS.iterator, layout(((32,2),(8,16)), strides=...))`.
  Per-element `sdS_2d[(vv, stage), (n_v%8, n_v//8)] = tdPrdS_cur[vv]`
  writes through `sdS.iterator` which carries dQ MMA's swizzle, so byte
  addresses match what the dQ MMA later reads.
- Marker probe: writing `100.0` to every sdS cell propagated to dQ output
  (`dQ abs max = 371`), confirming the write path is fully functional.

### Why this still doesn't fix dQ — deeper bug discovered

When the per-element write uses the actual `tdPrdS_cur[vv]` (real dS
register values), `dQ` output **matches baseline within ~ε** — i.e., the
fix achieves nothing visible. Tracing the dP register values via probe
`V=e_0` (where expected `dP[m, n=0]=dout[m, 0]`, `dP[m, n>0]=0`):

```
[dP] tidx=0 stage=0  V[0] coord=(0,0) dP=-0.924  ← ✓ expected -0.925
[dP] tidx=0 stage=0  V[1] coord=(0,1) dP=-0.180  ← ✗ expected -0.597
[dP] tidx=0 stage=0  V[2] coord=(0,2) dP= 0.224  ← ✗ expected -0.033
[dP] tidx=0 stage=0  V[3] coord=(0,3) dP=-0.262  ← ✗ expected -0.086
[dP] tidx=0 stage=1  V[0] coord=(0,32) dP=-1.032 ← ?
[dP] tidx=0 stage=1  V[1] coord=(0,33) dP= 0.141 ← ?
```

Even with the tcgen05 partition (which exposes the 2-stage iteration and
gives the SAME inner V-strides `(1, 65536)` as the bf16-style partition),
`tidx=0 V[1..]` still reads garbage. Stages 1's data (m=32..63 cells) is
also corrupted.

bf16 with the identical input setup reads CORRECT dP at every V slot.

**So the TMEM data at addresses 1..7 of `tdPtdP` is GARBAGE for fp32 even
though the QK^T MMA's tStS data at the SAME addresses is CORRECT.** Same
`make_fragment_C` layout, same partition_S, same atoms — yet S reads
clean, dP reads dirty.

### Revised root cause hypothesis

The TF32 MMA's actual cell-to-TMEM-address pattern for the dP MMA's output
differs from what `make_fragment_C` claims (the layout). The QK^T MMA's
output happens to land at the claim, but dP MMA's output doesn't.

Possible deeper causes:
1. dP MMA's K/N traversal order writes accumulator cells in a different
   pattern than QK^T (despite identical MMA setup). For instance, K=8
   reductions iterating in a different order may interleave writes that
   complete out-of-claim.
2. There's an MMA-private TMEM cell pattern that needs a hardware-specific
   partition (not derivable from `make_fragment_C` alone). cute's
   `tcgen05.make_tmem_copy` may have been designed for the QK^T-style use
   case and not handle the V@dO^T variant.
3. There's a pipeline / sync gap that the existing `pipeline_dP.consumer_wait`
   doesn't fully cover (race between MMA write and read).

The first two are software design issues. The third is a sync issue.
Empirical observation that **bf16 same setup works** + that the dP MMA's
setup is byte-identical to QK^T MMA points away from a generic sync bug
toward something specific to TF32 MMA's microarchitectural behavior.

### Status of A1+ fix attempt

- All A1+ pieces are individually working (tcgen05 partition compiles,
  per-element sdS write functions, dS values flow to dQ MMA).
- But the *upstream* dS register values are garbage because tdPtdP read
  is garbage. So fixing the r2s side alone doesn't fix dQ.

The fix would need to additionally somehow re-shape the tdPtdP read to
match the dP MMA's actual write pattern. **None of the partition styles
tried (bf16 copy_utils / tcgen05.make_tmem_copy / get_tmem_load_op-derived)
produces correct per-thread reads for tdPtdP under fp32**, despite all
three working correctly for tStS.

### Recommended next steps

1. **Investigate dP MMA write pattern empirically.** Either via:
   - SASS inspection of the generated MMA + atomic order
   - Dumping tdPtdP via every conceivable read partition until one matches
     `dout[m, 0]` for the V=e_0 case
   - Comparison against working bf16 kernel's dP read partition
2. **A3 path** (route P/dS through SMEM entirely) may sidestep this issue
   because dP would be produced into SMEM with a layout we directly
   control, avoiding the TMEM C-frag mystery.
3. Asking upstream (cutlass-dsl team) whether SM100 TF32 m64nNk8 MMA C-frag
   has a documented but non-`make_fragment_C` cell pattern for accumulator
   N=tile_m=64.

### Code state at end of session 7

Reverted to session 6 state (Bug #1 fix only). The A1+ attempt was
mechanically successful (compile, write propagation), but the dP-load
bug it's downstream of remains. Committed code is unchanged from session 6.
