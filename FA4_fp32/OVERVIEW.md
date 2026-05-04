# FA4_fp32 — B200 Code Overview

> **B200-only, SM100-only, MHA-only, fixed-length, non-causal happy path**, with **fp16 / bf16 / fp32 (TF32 MMA)** I/O. This branch (`B200`) is a stripped-down variant of upstream FA4 (`flash_attn/cute/`) — `git log` shows the cuts:
> ```
> 6cb23fe 정리2
> 1846e68 정리
> 2cad329 Strip FA4_fp32 down to SM100 MHA-only happy path
> b41ae28 Add FA4_fp32 implementation, benchmarks, and pixi env
> ```
>
> Total: **11,738 LOC** (vs ~28K LOC on `main`). Read this file first; it maps every module and calls out the tiny set of fp32-specific deviations from upstream FA4.

---

## 1. What was stripped

Removed from upstream FA4:

| Removed | Why |
|---------|-----|
| SM80 / SM90 / SM120 paths (`flash_fwd.py`, `flash_fwd_sm90.py`, `flash_fwd_sm120.py`, `flash_bwd.py`, `flash_bwd_sm90.py`, `flash_bwd_sm120.py`, `arch/ampere_helpers.py`) | B200 only |
| `flash_fwd_combine.py` (SplitKV combiner) | no SplitKV / FlashDecoding |
| `core/pack_gqa.py` | MHA only — Q/K/V share shape |
| `core/paged_kv.py` | no paged KV cache |
| `sparsity/` (entire dir) | no block-sparse attention |
| `flash_attn_varlen_func`, `cu_seqlens_*`, `seqused_*`, `max_seqlen_*` | fixed-length only |
| `causal`, `window_size_*`, `is_local`, `learnable_sink`, `softcap`, `score_mod`, `mask_mod`, `aux_tensors` | non-causal, no FlexAttention, no softcap |
| `num_splits`, `deterministic`, `use_2cta_instrs`, `q_subtile_factor`, `pack_gqa` | dead-coded to `False`/`1` |
| `apply_score_mod_inner`, `apply_score_mod_bwd_inner` from `softmax.py` | follows `score_mod` removal |
| `create_softcap_scoremod*`, `compute_fastdiv_mods`, `warp_prefix_sum`, `domain_offset_aligned`, `scalar_to_ssa`, `ssa_to_scalar`, `make_tiled_copy_A/B(swapAB)`, `mma_make_fragment_A/B(swapAB)`, `get_smem_store_atom`, `canonical_warp_group_idx` from `core/utils.py` | not used by SM100 path |
| `SeqlenInfoQKNewK` | no append-KV |
| Hopper R2P helpers (`sm90_col_to_r2p_idx`, `row_to_r2p_idx`) from `core/mask.py` | SM90 only |
| `SingleTileVarlenScheduler` from `core/tile_scheduler.py` | varlen removed |

Some flags survived only as **compile-time `False`** to keep `if const_expr(...)` branches dead-folded. E.g. `BlockInfo.is_causal` field still exists, `FlashAttentionBackwardSm100` declares `is_causal = False`, `is_local = False`, `pack_gqa = False`, `is_varlen_q = False`, `use_2cta_instrs = False`, `deterministic = False`, `score_mod = None`, `mask_mod = None` as class attributes. Don't accidentally re-enable these — they're hardcoded constants now, not configurable.

A sentinel `_Removed` class lives at the top of `flash_bwd_postprocess.py` — it raises if anything still tries to call into the deleted SM80 helper module.

---

## 2. What's still supported

- dtype: `float16`, `bfloat16`, **`float32` (TF32 MMA)**
- arch: SM100, SM101, SM103, SM110 (the four Blackwell variants enumerated in `_SM100_ARCHES`)
- shape: `(batch, seqlen, num_heads, head_dim)`, **same for Q/K/V** (no GQA), last dim contiguous, 16-byte aligned
- `head_dim` ∈ [8, 128], multiple of `16 // element_size` (i.e. 8 for fp16/bf16, 4 for fp32)
- forward + backward, with `return_lse`/`dlse` for differentiable-LSE losses
- `softmax_scale` override
- optional CLC dynamic-persistent scheduler (`FA_CLC=1`)
- `is_persistent=True` static-persistent scheduler by default

---

## 3. Directory layout

```
FA4_fp32/
├── __init__.py          # exports flash_attn_func, attention_fp32, compare, Shape, DEFAULT_SHAPES
├── interface.py         # 429 LOC, public API + autograd Function + 3 JIT compile entry points
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

`__init__.py` files inside subdirs are empty (flat namespaces).

---

## 4. Public API ([interface.py](interface.py), 429 LOC)

```python
from FA4_fp32 import flash_attn_func

out, lse = flash_attn_func(q, k, v, softmax_scale=None, return_lse=False)
# or with autograd:
out, lse = flash_attn_func(q.requires_grad_(), k.requires_grad_(), v.requires_grad_())
out.sum().backward()
```

**That's the entire public surface.** No `flash_attn_varlen_func`, no `causal=`, no `window_size=`, no `softcap=`, no `score_mod=`, no GQA, no SplitKV. The autograd `FlashAttnFunc.forward` signature has 5 positional inputs total (`q, k, v, softmax_scale, return_lse`).

Internally `interface.py` wires up three JIT-compile sites, each backed by a `compile_cache = get_jit_cache(...)`:
1. **`_flash_attn_fwd`** → [`FlashAttentionForwardSm100`](kernels/flash_fwd_sm100.py)
2. **`_bwd_preprocess`** → [`FlashAttentionBackwardPreprocess`](kernels/flash_bwd_preprocess.py) — computes `D = (O*dO).sum(-1) [- dLSE]`, `lse_log2 = LSE * log2(e)`, zeros `dq_accum`
3. **`_flash_attn_bwd`** → [`FlashAttentionBackwardSm100`](kernels/flash_bwd_sm100.py) — writes dV, dK, and dQ-accum (fp32). Then calls **`_bwd_postprocess_convert`** → [`FlashAttentionBackwardPostprocess`](kernels/flash_bwd_postprocess.py) to convert dQ-accum (fp32) → dQ (input dtype) with `softmax_scale` applied.

dK/dV postprocess from upstream is **gone** (no GQA → no head reduction → kernels write dK/dV directly into the output).

---

## 5. The two fp32-specific code paths (this is the whole "fp32" delta)

If you want to know what makes this build different from a hypothetical `bf16-only` build, **only these two branches matter**:

### A. fp32 forces a smaller tile (SMEM/TMEM is 2× per element)

[`interface.py:133-139`](interface.py#L133-L139) (forward):
```python
if q.dtype == torch.float32:
    tile_m, tile_n = 128, 64       # vs (128, 128) for fp16/bf16
    q_stage = 1                    # vs 2 when seqlen_q > tile_m
else:
    tile_m, tile_n = 128, 128
    q_stage = 2 if seqlen_q > tile_m else 1
```

[`interface.py:273-278`](interface.py#L273-L278) (backward):
```python
if q.dtype == torch.float32:
    m_block_size = 64              # vs 128 for fp16/bf16
else:
    m_block_size = 128
n_block_size = 128
```

### B. TF32 MMA produces half the columns per instruction → smaller dQ TMEM tile

[`kernels/flash_bwd_sm100.py:178-186`](kernels/flash_bwd_sm100.py#L178-L186):
```python
# fp32 I/O path: TMEM dQ accumulator is tiled with 16-col granularity because
# TF32 MMA produces half the columns-per-instr of fp16/bf16 MMA.
if self.q_dtype is Float32:
    self.dQ_reduce_ncol = 16
else:
    self.dQ_reduce_ncol = 32
self.sdQaccum_stage = 64 // self.dQ_reduce_ncol   # 4 for fp32, 2 for fp16/bf16
```

Plus a chunked TMEM-load atom in the bwd dQ epilogue (search for `self.q_dtype is Float32` in `flash_bwd_sm100.py` — there are a handful of related branches).

That's it. Everything else is dtype-agnostic and runs through `q_dtype.width // 8` SMEM-size math.

---

## 6. `core/` — arch-agnostic primitives

| File | LOC | Purpose |
|------|-----|---------|
| [softmax.py](core/softmax.py) | 164 | `Softmax` (multi-row online softmax via `fmax_reduce`/`fadd_reduce`), `SoftmaxSm100` (single-row + `scale_apply_exp2_convert` + polynomial `ex2_emu` fast path). **No `apply_score_mod_*`** — score_mod was removed |
| [mask.py](core/mask.py) | 152 | `AttentionMask` + `r2p_bitmask_below`/`above` (32-column R2P bitmasks via inline-PTX shifts). **No SM90 helpers** |
| [block_info.py](core/block_info.py) | 108 | `BlockInfo` — `get_n_block_min_max`, `get_m_block_min_max`. The `is_causal` / `is_local` / `is_split_kv` / `qhead_per_kvhead_packgqa` fields still exist but `interface.py` always passes `False`/`1` |
| [seqlen_info.py](core/seqlen_info.py) | 58 | Degenerated to a 2-scalar dataclass `(seqlen_q, seqlen_k)`. `has_cu_seqlens_*` / `has_seqused_*` are compile-time `False`. `offset_batch_Q`/`_K` do dense indexing |
| [pipeline.py](core/pipeline.py) | 402 | `PipelineStateSimple` (1-Int32 phase+index) + `_PipelineIndexPhaseMixin` wrappers around upstream `cutlass.pipeline.Pipeline*Async` |
| [tile_scheduler.py](core/tile_scheduler.py) | 620 | `SchedulingMode`, `ClcState`, `TileSchedulerArguments`/`Protocol`, **only 3 schedulers**: `SingleTileScheduler`, `StaticPersistentTileScheduler`, `SingleTileLPTScheduler` (varlen scheduler removed) |
| [copy_utils.py](core/copy_utils.py) | 372 | TMA / cp.async copy atoms, `tiled_copy_1d`/`2d`, `cvt_copy`, `make_tmem_copy` |
| [named_barrier.py](core/named_barrier.py) | 24 | Just `NamedBarrierFwdSm100` and `NamedBarrierBwdSm100` IntEnums |
| [barrier.py](core/barrier.py) | 71 | Cross-CTA semaphore primitives (`ld_acquire`, `red_release`, `wait_eq`, `arrive_inc`) |
| [fast_math.py](core/fast_math.py) | 21 | `clz` only |
| [utils.py](core/utils.py) | 657 | `hash_callable`, `compute_softmax_scale_log2` (always folds log2(e) — score_mod path removed), `convert_from_dlpack*`, `warp_reduce`, `smid`, `fmax`, `fmax_reduce`/`fadd_reduce` (3-input fmax + packed f32x2 add on SM100), `atomic_add_fp32`, `predicate_k`, `shuffle_sync`, `shl_u32`/`shr_u32` (PTX shifts to dodge LLVM shift-by-width UB), `cvt_f16x2_f32`/`cvt_f16` (Float32→fp16/bf16), `evaluate_polynomial`, `add_round_down`, `combine_int_frac_ex2`, `ex2_emulation`/`ex2_emulation_2`/`e2e_asm2` (polynomial-based exp2 emu) |

---

## 7. `kernels/` — the 4 kernels

| File | Class | LOC | Notes |
|------|-------|-----|-------|
| [flash_fwd_sm100.py](kernels/flash_fwd_sm100.py) | `FlashAttentionForwardSm100` | 1601 | Single class; persistent or CLC-scheduled; 16-warp layout (4 softmax0 + 4 softmax1 + 4 correction + mma + epilogue + load + empty + clc); polynomial `ex2_emu` enabled on non-SM103. `is_causal=False`, `is_local=False`, `is_split_kv=False` are hardcoded at all call sites |
| [flash_bwd_preprocess.py](kernels/flash_bwd_preprocess.py) | `FlashAttentionBackwardPreprocess` | 326 | Computes `D = (O*dO).sum(-1) - dLSE` (the `dLSE` branch is alive — it's how `return_lse=True` differentiable-LSE training works), `lse_log2 = LSE * log2(e)`, zeros `dq_accum`. **MHA only** (`head_dim_v = head_dim` hardcoded) |
| [flash_bwd_sm100.py](kernels/flash_bwd_sm100.py) | `FlashAttentionBackwardSm100` | 2248 | 16 warps (4 reduce + 8 compute + mma + load + empty + 1 unused). All feature flags (`is_causal`, `is_local`, `pack_gqa`, `is_varlen_q`, `deterministic`, `use_2cta_instrs`, `score_mod`, `mask_mod`, `dKV_postprocess`, `qhead_per_kvhead`) are class-level constants forced to `False`/`1`/`None`. Cluster size = 1, `cta_group_size = 1`. **fp32 → `dQ_reduce_ncol = 16`** (vs 32). Writes dV/dK directly + dQ to `dq_accum` (fp32) |
| [flash_bwd_postprocess.py](kernels/flash_bwd_postprocess.py) | `FlashAttentionBackwardPostprocess` | 473 | Converts `dq_accum` (fp32) → `dq` (input dtype) with `softmax_scale` applied. SM100 only (`cta_group = ONE`). Top-of-file has a `_Removed` sentinel for the deleted `sm80_utils` import — raises if hit |

---

## 8. `arch/` — vendor MMA helpers

| File | LOC | Purpose |
|------|-----|---------|
| [blackwell_helpers.py](arch/blackwell_helpers.py) | 1089 | SM100 UMMA `gemm_*` variants (`gemm_w_idx`, `gemm_ptx_w_idx`, `gemm_ptx`, `gemm_ptx_loop`, `gemm_ptx_partial`, `gemm_ptx_partial1`, `gemm_ptx_precomputed[_varname]`); `declare_ptx_smem_desc` and `declare_ptx_idesc` emit inline-PTX register declarations for hand-tuned hot paths |
| [mma_sm100_desc.py](arch/mma_sm100_desc.py) | 296 | UMMA descriptor enums (`Major`, `ScaleIn`, `Saturate`, `CFormat`, `F16F32Format`, `S8Format`, `MXF8F6F4Format`, `MaxShift`, `LayoutType`); `make_instr_desc`, `mma_op_to_idesc`, `make_smem_desc_base` etc. |

---

## 9. `infra/` — host-side plumbing

| File | LOC | Purpose |
|------|-----|---------|
| [cute_dsl_utils.py](infra/cute_dsl_utils.py) | 129 | `cute_compile_patched` (dump SASS when `CUTE_CUBIN_PATH` set), `assume_tensor_aligned`, `to_cute_tensor`, `get_broadcast_dims` (per-dim stride-0 detection for compile key) |
| [cache_utils.py](infra/cache_utils.py) | 282 | `JITCache`, `JITPersistentCache` (sha256 of all `.py` + cutlass/tvm_ffi versions as fingerprint, pickle to `/tmp/$USER/flash_attention_cute_dsl_cache/<fp>/<name>/`); `get_jit_cache(name)`. Pre-loads cute runtime libs with `RTLD_GLOBAL` so cached `.so` modules can dlopen |
| [cute_dsl_ptxas.py](infra/cute_dsl_ptxas.py) | 151 | If `CUTE_DSL_PTXAS_PATH` is set, dumps PTX and re-compiles with the user `ptxas` |
| [fa_logging.py](infra/fa_logging.py) | 97 | `FA_LOG_LEVEL` env var (0=off / 1=host / 2=kernel / 3=max). `fa_printf` is `cute.printf` wrapped in `const_expr` — zero-cost when level is too low |
| [testing.py](infra/testing.py) | 456 | `attention_ref` (full-featured PyTorch reference), `is_fake_mode()`, `maybe_fake_tensor_mode`, varlen test fixtures (mostly dead in this build) |
| [bench_utils.py](infra/bench_utils.py) | 196 | FLOP counters, cuDNN setup helpers |
| [benchmark.py](infra/benchmark.py) | 268 | `benchmark_forward`, `benchmark_backward`, `benchmark_combined`, etc. |
| [sm90_config_search.py](infra/sm90_config_search.py) | 402 | Brute-force tile-config search — **dead code in this build** (no SM90 path) but not deleted |

**Env vars:**
- `FLASH_ATTENTION_ARCH` — override compute-capability detection
- `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` / `FLASH_ATTENTION_CUTE_DSL_CACHE_DIR` — persistent cache
- `CUTE_CUBIN_PATH`, `CUTE_DSL_KEEP_PTX=1`, `CUTE_DSL_PTXAS_PATH` — debug compile artefacts
- `FA_CLC=1` — enable CLC dynamic-persistent scheduler
- `FA_LOG_LEVEL` — log verbosity

(`FA_DISABLE_2CTA` is gone — 2-CTA was removed.)

---

## 10. `bwd/` — backward correctness reference

| File | Purpose |
|------|---------|
| [bwd/bwd_ref.py](bwd/bwd_ref.py) | `bwd_ref(q, k, v, dout, causal=False, softmax_scale)` runs PyTorch autograd on `attention_fp32` to produce ground-truth dQ/dK/dV. **No hand-written dS/dP math** |
| [bwd/preprocess_ref.py](bwd/preprocess_ref.py) | Naive PyTorch reproduction of the preprocess kernel; documents the `D - dLSE` mathematics |
| [bwd/test_bwd.py](bwd/test_bwd.py) | End-to-end FA4 fp32 backward correctness vs `bwd_ref`; hard-coded `CASES` grid |
| [bwd/test_preprocess.py](bwd/test_preprocess.py) | Two-layer correctness: naive vs einsum, then naive vs FA4 preprocess kernel |

---

## 11. Math invariants worth preserving

- **Online softmax.** Each new tile contributes `exp(scores * scale - new_row_max)`; `acc_O` is rescaled by `exp(old_row_max - new_row_max)`. `scale_log2 = scale * log2(e)` so the kernel uses hardware/poly `exp2`.
- **Backward formula.** `dS_ij = P_ij * (dP_ij - D_i)` where `D_i = sum_d O[i,d] * dO[i,d]`. When LSE is differentiable (`return_lse=True` path), an extra `dLSE_i * P_ij` term is folded into `D` by the preprocess kernel: `D' = D - dLSE`, leaving the main bwd kernel unchanged. See the docstring at the top of [flash_bwd_preprocess.py](kernels/flash_bwd_preprocess.py).
- **R2P bitmasks need element indices.** `r2p_bitmask_below`/`above` operate on element positions, not column positions. SM100 fwd uses them only via `AttentionMask` for padding — there's no causal/local masking in this build.
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

# End-to-end fp32 bwd vs PyTorch autograd:
python FA4_fp32/bwd/test_bwd.py

# fp32 vs bf16 fwd benchmark:
python FA4_fp32/bench_fp32.py    # NOTE: edit the hard-coded sys.path at the top
```

`compare(my_kernel, shapes=DEFAULT_SHAPES)` from [verify.py](verify.py) is the cheapest way to sanity-check a candidate. Tolerances `atol_fwd=5e-4`, `atol_bwd=2e-3` are the cost of TF32 matmuls.

---

## 13. Where to start when extending

The fp32 surface is genuinely tiny — three places:

1. **[interface.py:133-139](interface.py#L133-L139)** + **[interface.py:273-278](interface.py#L273-L278)** — fp32 tile sizing
2. **[kernels/flash_bwd_sm100.py:178-186](kernels/flash_bwd_sm100.py#L178-L186)** + the chunked TMEM-load atom near it — the `dQ_reduce_ncol = 16` branch
3. **[reference.py](reference.py)** + **[verify.py](verify.py)** — TF32 numerical baseline

If you're adding a new feature (e.g. re-enabling causal), expect to touch:
- `interface.py` (parameter + compile key)
- The `class*Sm100` `__init__` flag overrides (currently `is_causal = False` is class-level)
- `BlockInfo` call sites (the field is still there, just always passed `False`)
- `core/mask.py` for the masking bitmask plumbing
- `core/tile_scheduler.py` if the scheduler needs to know about the change
