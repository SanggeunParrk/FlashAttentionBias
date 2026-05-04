# FA4_fp32 — B200 Code Overview

> **B200-only, SM100-only, MHA-only, fixed-length, non-causal happy path**, with **fp16 / bf16 / fp32 (TF32 MMA)** I/O. The `B200` branch is a stripped-down variant of upstream FA4 (`flash_attn/cute/`).
>
> Build assumptions baked into the code (do not violate without re-auditing dead branches):
> - **Arch**: SM100 (B100/B200) only. SM103 (Blackwell Ultra), SM110 (GB10) not supported.
> - **Q = K = V** in shape, dtype, head_dim, num_heads, seqlen. No GQA, no separate `head_dim_v`.
> - **head_dim ∈ {32, 64}** (kernel still validates `8 ≤ head_dim ≤ 128`, but config heuristics assume `head_dim_padded < 96`).
> - **Non-causal**, no sliding window, no SplitKV, no varlen, no paged KV, no block sparsity, no score_mod / mask_mod / softcap / learnable_sink.
> - fwd: `is_persistent=True` always. `use_clc_scheduler` toggleable via `FA_CLC=1`.
> - bwd: always non-persistent (`SingleTileScheduler`).
>
> Total: **11,096 LOC** (vs ~28K LOC on `main`).
>
> Commit history of the simplification on this branch:
> ```
> 050d14a Add CLC scheduler (FA_CLC=1) vs static persistent fwd benchmark
> 598bda1 Strip dead branches/vars assuming SM100, head_dim ∈ {32,64}, Q=K=V
> e22fef8 Add persistent-vs-non-persistent forward benchmark
> 84247e2 Strip dead args from shared dataclasses and propagate through fwd/bwd
> 259d6d7 Add FA4_fp32 OVERVIEW.md for B200 build
> 6cb23fe 정리2
> 1846e68 정리
> 2cad329 Strip FA4_fp32 down to SM100 MHA-only happy path
> b41ae28 Add FA4_fp32 implementation, benchmarks, and pixi env
> ```

---

## 1. What was stripped from upstream FA4

Removed entirely (vs `flash_attn/cute/`):

| Removed | Why |
|---------|-----|
| SM80 / SM90 / SM103 / SM110 / SM120 paths (`flash_fwd.py`, `flash_fwd_sm{80,90,120}.py`, `flash_bwd.py`, `flash_bwd_sm{90,120}.py`, `arch/ampere_helpers.py`) | B200 (SM100) only |
| `flash_fwd_combine.py` (SplitKV combiner) | no SplitKV / FlashDecoding |
| `core/pack_gqa.py`, `core/paged_kv.py` | MHA only, no paged cache |
| `sparsity/` (entire dir) | no block-sparse attention |
| `flash_attn_varlen_func`, `cu_seqlens_*`, `seqused_*`, `max_seqlen_*` | fixed-length only |
| `causal`, `window_size_*`, `is_local`, `learnable_sink`, `softcap`, `score_mod`, `mask_mod`, `aux_tensors` | non-causal, no FlexAttention, no softcap |
| `num_splits`, `deterministic`, `use_2cta_instrs`, `q_subtile_factor`, `pack_gqa` | dead-coded; later batches removed the constants too |
| `apply_score_mod_inner`, `apply_score_mod_bwd_inner` | follows `score_mod` removal |
| Hopper R2P helpers (`sm90_col_to_r2p_idx`, `row_to_r2p_idx`) | SM100 only |
| `SingleTileVarlenScheduler`, `SingleTileLPTBwdScheduler` | varlen / deterministic removed |

Subsequent simplification on this branch (commits 84247e2 + 598bda1) deleted the dead-but-still-present constants/branches:
- All compile-time-False class flags on `FlashAttentionBackwardSm100` (`is_causal`, `is_local`, `pack_gqa`, `deterministic`, `use_2cta_instrs`, …) and the function kwargs that tunneled them through `compute_loop` / `mainloop` / dQ-reduce loop (`dS_cluster_*_mbar_ptr`, `pipeline_Qt`/`Kt`, `is_leader_cta`, `aux_tensors`, `mdK_semaphore`, etc.).
- 11 dead fields on `TileSchedulerArguments` (`num_splits`, `mCuSeqlensQ`, `lpt`, `is_split_kv`, `is_persistent`, `cluster_shape_mn`, `tile_shape_mn`, `total_q`, …).
- `BlockInfo` 6 dead fields + 2 dead methods.
- `SeqlenInfoQK.create` `**_ignored` kwargs.
- `AttentionMask` 3 dead fields (kept `swap_AB` — used by bwd).
- `compute_softmax_scale_log2` `score_mod` arg.
- `flash_bwd_postprocess.py`: `_Removed("ampere_helpers")` sentinel + `AtomLayoutMdQ` / `use_2cta_instrs` / `cluster_size` / `arch` ctor args + the ~115-line 2-CTA dQ-accumulator branch.
- `flash_bwd_preprocess.py`: `use_pdl` arch fork (always True on SM100), `check_hdim_v_oob` predication (head_dim ∈ {32,64} → padded ≡ hdim), Float16/BF16/FP32 dtype guards.
- `flash_fwd_sm100.py`: `is_sm103` / `enable_ex2_emu` fork (always emulate exp2), `head_dim_padded < 96` register-budget fork, Q/K/V dtype-mismatch raises, `split_P_arrive > 0` branches (always > 0).

---

## 2. What's still supported

- dtype: `float16`, `bfloat16`, **`float32` (TF32 MMA)**
- arch: SM100 only (`_SUPPORTED_ARCHES = (100,)`)
- shape: `(batch, seqlen, num_heads, head_dim)`, **same for Q/K/V**, last dim contiguous, 16-byte aligned
- `head_dim` ∈ [8, 128] permitted by `_validate_head_dim`, but config heuristics target {32, 64}
- forward + backward, with `return_lse`/`dlse` for differentiable-LSE losses
- `softmax_scale` override
- optional CLC dynamic-persistent scheduler for fwd via `FA_CLC=1`
- fwd `is_persistent=True` (default; non-persistent path is dead but option stays for safety)

---

## 3. Directory layout

```
FA4_fp32/
├── __init__.py          # exports flash_attn_func, attention_fp32, compare, Shape, DEFAULT_SHAPES
├── interface.py         # public API + autograd Function + 3 JIT compile entry points
├── reference.py         # PyTorch fp32 reference (TF32 internal); 5D (A, B, L, H, D) layout
├── verify.py            # compare(candidate) -> bool harness; default shape grid
├── bench_fp32.py        # fwd benchmark: FA fp32 / FA bf16 / torch ref
│
├── arch/                # blackwell_helpers.py + mma_sm100_desc.py only — no Ampere
├── core/                # 11 arch-agnostic primitives
├── kernels/             # only 4 kernels: flash_fwd_sm100, flash_bwd_{preprocess, sm100, postprocess}
├── infra/               # JIT cache, env-var plumbing, logging, testing helpers, benchmarks
└── bwd/                 # PyTorch reference + tests for the backward (bwd_ref, preprocess_ref, test_*)
```

`bench_results/` (sibling, not in package): canonical benchmark scripts and outputs (`bench_persistent.{py,txt}`, `bench_clc.{py,txt}`, plus the older fp32-vs-bf16 results).

---

## 4. Public API ([interface.py](interface.py), 429 LOC)

```python
from FA4_fp32 import flash_attn_func

out, lse = flash_attn_func(q, k, v, softmax_scale=None, return_lse=False)
# or with autograd:
out, lse = flash_attn_func(q.requires_grad_(), k.requires_grad_(), v.requires_grad_())
out.sum().backward()
```

That's the entire public surface. The autograd `FlashAttnFunc.forward` signature has 5 positional inputs (`q, k, v, softmax_scale, return_lse`).

Three JIT-compile sites, each backed by `compile_cache = get_jit_cache(...)`:
1. **`_flash_attn_fwd`** → [`FlashAttentionForwardSm100`](kernels/flash_fwd_sm100.py)
2. **`_bwd_preprocess`** → [`FlashAttentionBackwardPreprocess`](kernels/flash_bwd_preprocess.py) — computes `D = (O*dO).sum(-1) [- dLSE]`, `lse_log2 = LSE * log2(e)`, zeros `dq_accum`
3. **`_flash_attn_bwd`** → [`FlashAttentionBackwardSm100`](kernels/flash_bwd_sm100.py) → then **`_bwd_postprocess_convert`** → [`FlashAttentionBackwardPostprocess`](kernels/flash_bwd_postprocess.py)

dK/dV postprocess from upstream is gone (no GQA → no head reduction → kernels write dK/dV directly).

---

## 5. The two fp32-specific code paths

The whole "fp32" delta lives in two places:

### A. fp32 forces a smaller tile (SMEM/TMEM is 2× per element)

[`interface.py:133-139`](interface.py#L133-L139) (forward):
```python
if q.dtype == torch.float32:
    tile_m, tile_n = 128, 64       # vs (128, 128) for fp16/bf16
    q_stage = 1                    # vs 2 when seqlen_q > tile_m
```

[`interface.py:273-278`](interface.py#L273-L278) (backward):
```python
if q.dtype == torch.float32:
    m_block_size = 64              # vs 128 for fp16/bf16
n_block_size = 128
```

### B. TF32 MMA produces half the columns per instruction → smaller dQ TMEM tile

[`kernels/flash_bwd_sm100.py`](kernels/flash_bwd_sm100.py): `dQ_reduce_ncol = 16` for fp32, `32` for fp16/bf16; `sdQaccum_stage = 64 // dQ_reduce_ncol`. Plus a chunked TMEM-load atom in the bwd dQ epilogue (search `self.q_dtype is Float32`).

> **fp32 bwd is currently broken** at [`flash_bwd_sm100.py`](kernels/flash_bwd_sm100.py) (`utils.cvt_f16` size mismatch in compute_loop). User is developing the fix separately. fp32 fwd works; bf16/fp16 fwd+bwd works.

Everything else is dtype-agnostic and runs through `q_dtype.width // 8` SMEM-size math.

---

## 6. `core/` — arch-agnostic primitives

| File | LOC | Purpose |
|------|-----|---------|
| [softmax.py](core/softmax.py) | 164 | `Softmax` (multi-row online softmax via `fmax_reduce`/`fadd_reduce`), `SoftmaxSm100` (single-row + `scale_apply_exp2_convert` + polynomial `ex2_emu` fast path). No `apply_score_mod_*` |
| [mask.py](core/mask.py) | 125 | `AttentionMask(tile_m, tile_n, seqlen_info, swap_AB)` — only fields. `r2p_bitmask_below`/`above` (32-column R2P bitmasks via inline-PTX shifts). `apply_mask_sm100[_transposed]` — minimal seqlen-edge masking only (no causal/local/mask_mod kwargs) |
| [block_info.py](core/block_info.py) | 35 | `BlockInfo(tile_m, tile_n)` — only fields. Two methods: `get_n_block_min_max` returns `(0, ceil_div(seqlen_k, tile_n))`, `get_m_block_min_max` symmetric. Causal/local/split-KV/pack-GQA branches all gone |
| [seqlen_info.py](core/seqlen_info.py) | 24 | `SeqlenInfoQK(seqlen_q, seqlen_k)` — 2 fields only. `offset_batch_Q`/`_K` do dense indexing. No `has_cu_seqlens_*` flags anymore |
| [pipeline.py](core/pipeline.py) | 402 | `PipelineStateSimple` + `_PipelineIndexPhaseMixin` wrappers around upstream `cutlass.pipeline.Pipeline*Async` |
| [tile_scheduler.py](core/tile_scheduler.py) | 480 | `TileSchedulerArguments` (now: `num_block`, `num_head`, `num_batch`, `seqlen_k`, `headdim`, `headdim_v`, `element_size` only); `SchedulingMode`, `ClcState`, `WorkTileInfo` (4-tuple, split slot fixed to 0); 3 schedulers: `SingleTileScheduler`, `StaticPersistentTileScheduler`, `SingleTileLPTScheduler` (LPT block-reverse logic also removed) |
| [copy_utils.py](core/copy_utils.py) | 372 | TMA / cp.async copy atoms, `tiled_copy_1d`/`2d`, `cvt_copy`, `make_tmem_copy` |
| [named_barrier.py](core/named_barrier.py) | 24 | `NamedBarrierFwdSm100` and `NamedBarrierBwdSm100` IntEnums |
| [barrier.py](core/barrier.py) | 71 | Cross-CTA semaphore primitives (`ld_acquire`, `red_release`, `wait_eq`, `arrive_inc`) |
| [fast_math.py](core/fast_math.py) | 21 | `clz` only |
| [utils.py](core/utils.py) | 662 | `hash_callable`, `compute_softmax_scale_log2(softmax_scale)` (always folds log2(e); `score_mod` arg removed), `convert_from_dlpack*`, `warp_reduce`, `smid`, `fmax`, `fmax_reduce`/`fadd_reduce` (3-input fmax + packed f32x2 add), `atomic_add_fp32`, `predicate_k`, `shuffle_sync`, `shl_u32`/`shr_u32` (PTX shifts to dodge LLVM shift-by-width UB), `cvt_f16x2_f32`/`cvt_f16` (Float32→fp16/bf16), `evaluate_polynomial`, `add_round_down`, `combine_int_frac_ex2`, `ex2_emulation`/`ex2_emulation_2`/`e2e_asm2` (polynomial-based exp2 emu) |

---

## 7. `kernels/` — the 4 kernels

| File | Class | LOC | Notes |
|------|-------|-----|-------|
| [flash_fwd_sm100.py](kernels/flash_fwd_sm100.py) | `FlashAttentionForwardSm100` | 1531 | Persistent or CLC-scheduled; 16-warp layout (4 softmax0 + 4 softmax1 + 4 correction + mma + epilogue + load + empty + clc). `enable_ex2_emu = True` always (SM100 SFU is slower than the polynomial). Single register budget (head_dim_padded < 96). No Q/K/V dtype-mismatch raises. `split_P_arrive > 0` branches collapsed (always true) |
| [flash_bwd_preprocess.py](kernels/flash_bwd_preprocess.py) | `FlashAttentionBackwardPreprocess` | 297 | Computes `D = (O*dO).sum(-1) - dLSE`, `lse_log2 = LSE * log2(e)`, zeros `dq_accum`. `use_pdl=True` always (SM100). MHA only (`head_dim_v = head_dim` hardcoded). No dtype-mismatch raises, no `check_hdim_v_oob` predication |
| [flash_bwd_sm100.py](kernels/flash_bwd_sm100.py) | `FlashAttentionBackwardSm100` | 2125 | 16 warps (4 reduce + 8 compute + mma + load + empty + 1 unused). All "compile-time False" class constants are gone — `is_causal`/`is_local`/`pack_gqa`/`deterministic`/etc. are no longer attributes; the kernel body just doesn't have those branches anymore. Cluster=1 hardcoded; `cta_group=tcgen05.CtaGroup.ONE` baked into MMA atom construction. dK/dV major-mode raise guards removed. **fp32 → `dQ_reduce_ncol = 16`** (vs 32). Writes dV/dK directly + dQ to `dq_accum` (fp32) |
| [flash_bwd_postprocess.py](kernels/flash_bwd_postprocess.py) | `FlashAttentionBackwardPostprocess` | 322 | Converts `dq_accum` (fp32) → `dq` (input dtype) with `softmax_scale` applied. SM100 only (`cta_group = ONE`). Constructor: `(dtype, head_dim, tile_m, num_threads, dQ_swapAB)` only — `AtomLayoutMdQ` / `use_2cta_instrs` / `cluster_size` / `arch` removed. 2-CTA dQ-accumulator branch (~115 lines) removed |

---

## 8. `arch/` — vendor MMA helpers

| File | LOC | Purpose |
|------|-----|---------|
| [blackwell_helpers.py](arch/blackwell_helpers.py) | 1089 | SM100 UMMA `gemm_*` variants and inline-PTX register-declaration helpers |
| [mma_sm100_desc.py](arch/mma_sm100_desc.py) | 296 | UMMA descriptor enums + bit-pack helpers |

---

## 9. `infra/` — host-side plumbing

| File | LOC | Purpose |
|------|-----|---------|
| [cute_dsl_utils.py](infra/cute_dsl_utils.py) | 129 | `cute_compile_patched` (dump SASS when `CUTE_CUBIN_PATH` set), `to_cute_tensor`, `get_broadcast_dims` (per-dim stride-0 detection for compile key) |
| [cache_utils.py](infra/cache_utils.py) | 282 | `JITCache` / `JITPersistentCache` (sha256 of all `.py` + cutlass/tvm_ffi versions as fingerprint, pickle to `/tmp/$USER/flash_attention_cute_dsl_cache/<fp>/<name>/`); `get_jit_cache(name)`. Pre-loads cute runtime libs with `RTLD_GLOBAL` |
| [cute_dsl_ptxas.py](infra/cute_dsl_ptxas.py) | 151 | `CUTE_DSL_PTXAS_PATH` to dump PTX and re-compile with user `ptxas` |
| [fa_logging.py](infra/fa_logging.py) | 97 | `FA_LOG_LEVEL` env var (0=off / 1=host / 2=kernel / 3=max). `fa_printf` is `cute.printf` wrapped in `const_expr` — zero-cost when level too low |
| [testing.py](infra/testing.py) | 456 | `attention_ref`, `is_fake_mode()`, `maybe_fake_tensor_mode`, varlen test fixtures (mostly dead in this build) |
| [bench_utils.py](infra/bench_utils.py) | 196 | FLOP counters, cuDNN setup helpers |
| [benchmark.py](infra/benchmark.py) | 268 | `benchmark_forward`, `benchmark_backward`, etc. |
| [sm90_config_search.py](infra/sm90_config_search.py) | 402 | Brute-force tile-config search — **dead code in this build** but not deleted |

**Env vars:**
- `FLASH_ATTENTION_ARCH` — override compute-capability detection
- `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` / `FLASH_ATTENTION_CUTE_DSL_CACHE_DIR` — persistent cache
- `CUTE_CUBIN_PATH`, `CUTE_DSL_KEEP_PTX=1`, `CUTE_DSL_PTXAS_PATH` — debug compile artefacts
- `FA_CLC=1` — enable CLC dynamic-persistent scheduler (fwd only)
- `FA_LOG_LEVEL` — log verbosity

---

## 10. `bwd/` — backward correctness reference

| File | Purpose |
|------|---------|
| [bwd/bwd_ref.py](bwd/bwd_ref.py) | `bwd_ref(q, k, v, dout, causal=False, softmax_scale)` runs PyTorch autograd on `attention_fp32` to produce ground-truth dQ/dK/dV |
| [bwd/preprocess_ref.py](bwd/preprocess_ref.py) | Naive PyTorch reproduction of the preprocess kernel; documents the `D - dLSE` math |
| [bwd/test_bwd.py](bwd/test_bwd.py) | End-to-end FA4 backward correctness vs `bwd_ref` |
| [bwd/test_preprocess.py](bwd/test_preprocess.py) | Two-layer correctness: naive vs einsum, then naive vs FA4 preprocess kernel |

---

## 11. Math invariants worth preserving

- **Online softmax.** Each new tile contributes `exp(scores * scale - new_row_max)`; `acc_O` is rescaled by `exp(old_row_max - new_row_max)`. `scale_log2 = scale * log2(e)` so the kernel uses hardware/poly `exp2`.
- **Backward formula.** `dS_ij = P_ij * (dP_ij - D_i)` where `D_i = sum_d O[i,d] * dO[i,d]`. When LSE is differentiable (`return_lse=True` path), an extra `dLSE_i * P_ij` term is folded into `D` by the preprocess kernel: `D' = D - dLSE`, leaving the main bwd kernel unchanged. See the docstring at the top of [flash_bwd_preprocess.py](kernels/flash_bwd_preprocess.py).
- **R2P bitmasks need element indices.** `r2p_bitmask_below`/`above` operate on element positions, not column positions. SM100 fwd uses them only via `AttentionMask` for seqlen-edge padding — there is no causal/local masking in this build.
- **fp32 TMEM granularity.** TF32 MMA produces half the columns of fp16/bf16 MMA. The dQ accumulator in TMEM is therefore tiled at `dQ_reduce_ncol = 16` for fp32 vs 32 for fp16/bf16, and `sdQaccum_stage` is doubled to 4. Don't change one without changing the other.

---

## 12. Quick start

```bash
# Compile kernels without GPU memory and cache to disk:
FLASH_ATTENTION_FAKE_TENSOR=1 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
    pytest -n 64 -x tests/cute/test_flash_attn.py

# Run with cached binaries:
FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
    pytest -x tests/cute/test_flash_attn.py

# Self-check the harness (reference vs reference; errors should be 0):
python -m FA4_fp32.verify

# End-to-end fp32 bwd vs PyTorch autograd: (currently broken, fp32 bwd in dev)
python FA4_fp32/bwd/test_bwd.py

# fp32 vs bf16 fwd benchmark:
python FA4_fp32/bench_fp32.py    # NOTE: edit the hard-coded sys.path at the top

# Persistent vs non-persistent fwd benchmark (B=48 sweep):
python bench_results/bench_persistent.py

# CLC vs static persistent fwd benchmark:
python bench_results/bench_clc.py
```

`compare(my_kernel, shapes=DEFAULT_SHAPES)` from [verify.py](verify.py) is the cheapest way to sanity-check a candidate. Tolerances `atol_fwd=5e-4`, `atol_bwd=2e-3` are the cost of TF32 matmuls.

---

## 13. Bench results (B=48, B200 / 148 SM)

Scripts in `bench_results/`. Headlines:

| Comparison | bf16 | fp32 |
|-----------|------|------|
| persistent vs non-persistent fwd | persistent **+10–20%** (waves 11→167) | persistent **+5–20%** |
| CLC (FA_CLC=1) vs static persistent fwd | CLC **−1%** avg (compute-bound, low variance) | CLC **+2%** avg (memory-bound, more variance) |
| LSE-write vs no-LSE-write fwd (D=64) | LSE-write **+30%** slower → keep `mLSE=Optional` | (not yet measured for fp32) |

Persistent mode is hard-coded `True` in fwd because the win is always positive across the user's workload regime. CLC is left as an env-var toggle (`FA_CLC=1`) — only worth flipping on for fp32-heavy pipelines.

---

## 14. Where to start when extending

The fp32 surface is genuinely tiny — three places:

1. **[interface.py:133-139](interface.py#L133-L139)** + **[interface.py:273-278](interface.py#L273-L278)** — fp32 tile sizing (fwd + bwd `m_block_size`).
2. **`flash_bwd_sm100.py` `_setup_attributes` near `dQ_reduce_ncol`** + the chunked TMEM-load atom in the bwd dQ epilogue — the `dQ_reduce_ncol = 16` branch.
3. **[reference.py](reference.py)** + **[verify.py](verify.py)** — TF32 numerical baseline.

If you're adding a new feature (e.g. re-enabling causal or adding a bias load):
- `interface.py` (parameter + compile key)
- `BlockInfo` if it changes the (n_block_min, n_block_max) range
- `core/mask.py` for masking bitmask plumbing
- `core/tile_scheduler.py` if the scheduler needs to know about the new feature
- bwd persistent: not currently supported and is upstream-untested — would require rewriting TMEM dQ-accum init, pipeline-state reset, and mbarrier carry-over before swapping the scheduler.
