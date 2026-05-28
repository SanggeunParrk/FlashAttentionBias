# FA4_fp32 — H100 Build Overview

> H100 (SM90) MHA forward, with fp16 / bf16 (working) and fp32 / TF32-MMA (WIP scaffolding). Stripped down from upstream FA4 in the spirit of the B200 branch, but targeting Hopper instead of Blackwell.

## Environment (CUDA 12.9 driver)

The cluster's H100 nodes run **driver 575.51.03 (CUDA 12.9 capable)** — too old for the
cu13 wheels that the FA4 pixi env pulls. The aug_attn_bench venv (`.venv` at the repo
root area) is the supported runtime instead:

| Package | Version | Notes |
|---|---|---|
| torch | 2.8.0+cu126 | CUDA 12.6 wheel, works with driver R570+ |
| triton | 3.4.0 | Bundled with torch 2.8 |
| nvidia-cutlass-dsl | 4.5.2 | **Base only** — do NOT install `[cu13]` extra |
| nvidia-cutlass-dsl-libs-base | 4.5.2 | CUDA-12-compatible runtime libs |
| cuda-python | 12.9.7 | Pinned to 12.9 (default resolution picks 13.3 → driver too old) |
| cuda-bindings | 12.9.7 | Same |
| apache-tvm-ffi | 0.1.11 | cute FFI |
| torch-c-dlpack-ext | 0.1.5 | DLPack bridge |
| quack-kernels | 0.4.1 | sm90 helpers |

Install script: [`FA4_fp32/run_cu12_install_test.sh`](run_cu12_install_test.sh).

Verified on node01 (NVIDIA H100 80GB HBM3, driver 575.51.03):

```
$ srun ... bash FA4_fp32/run_cu12_install_test.sh
torch 2.8.0+cu126 cuda 12.6 avail True
device NVIDIA H100 80GB HBM3
cuda.bindings 12.9.7
FA4_fp32 ok / arch.* ok
fp32 path → NotImplementedError (expected, integration WIP)
bf16 fwd OK, out shape (2, 512, 4, 64) dtype torch.bfloat16
```

## What was stripped (vs. main / pre-strip H100 branch)

Same spirit as the B200 branch (`origin/B200`), but keeping SM90 instead of SM100.

### Removed kernels / files

| Path | Reason |
|---|---|
| `FA4_fp32/kernels/flash_fwd_sm100.py` | Blackwell-only |
| `FA4_fp32/kernels/flash_fwd_sm120.py` | SM120-only |
| `FA4_fp32/kernels/flash_bwd_sm100.py` | Blackwell-only |
| `FA4_fp32/kernels/flash_bwd_sm120.py` | SM120-only |
| `FA4_fp32/kernels/flash_fwd_combine.py` | SplitKV combiner (we disallow split-KV on SM90) |
| `FA4_fp32/kernels/flash_bwd.py` | SM80 backward, dead on Hopper |
| `FA4_fp32/kernels/flash_fwd.py::FlashAttentionForwardSm80` | SM80 forward class (the base `FlashAttentionForwardBase` stays — SM90 inherits it). The file shrank from 1232 → 581 LOC. |

### Kept but **not** referenced by SM90 fwd happy path

- `arch/blackwell_helpers.py` (project-local), `arch/mma_sm100_desc.py` —
  imported by `core/copy_utils.py` for TMEM helpers and by
  `flash_bwd_postprocess.py`. Not in SM90 fwd execution path.
- `arch/ampere_helpers.py` — used by `flash_bwd_postprocess.py::get_smem_layout_atom`.

### Kept for SM90 fwd

| Path | Why |
|---|---|
| `FA4_fp32/kernels/flash_fwd_sm90.py` | SM90 fwd kernel (still uses sparsity / pack_gqa / paged_kv — full strip is future work) |
| `FA4_fp32/kernels/flash_bwd_sm90.py`, `flash_bwd_preprocess.py`, `flash_bwd_postprocess.py` | Bwd path (untouched in this strip) |
| `FA4_fp32/core/{mask, softmax, block_info, seqlen_info, pipeline, tile_scheduler, copy_utils, named_barrier, barrier, fast_math, utils}.py` | arch-agnostic primitives |
| `FA4_fp32/core/{pack_gqa, paged_kv}.py` | Still referenced by `flash_fwd_sm90.py`; cleanup deferred |
| `FA4_fp32/sparsity/` | Same — referenced by the SM90 kernel body |
| `FA4_fp32/arch/{hopper_helpers, wgmma_tf32, sm90_utils_tf32}.py` | TF32 WGMMA scaffolding (new, this branch) |

### interface.py changes

- `dtype` assertion relaxed to allow `torch.float32` (TF32 MMA path) alongside fp16/bf16.
- `assert arch // 10 == 9` (was `in [8,9,10,11,12]`) — SM90 only for both fwd and bwd.
- Fwd dispatch: only `FlashAttentionForwardSm90` (SM80/100/120 branches deleted).
- Bwd dispatch: only `FlashAttentionBackwardSm90`.
- `FlashAttentionForwardCombine` reduced to a `NotImplementedError` stub (dead path, gated by `is_split_kv`).
- `BlockSparseTensorsTorch` / `get_sparse_q_block_size` / `to_cute_block_sparse_tensors` / `normalize_block_sparse_config{,_bwd}` reduced to stubs (block sparsity rejected at call time).

## What still needs stripping (next sessions)

The SM90 kernel body itself still contains code for features that the H100 build
doesn't need. Per B200's `2cad329 Strip FA4_fp32 down to SM100 MHA-only happy path`
and subsequent dead-arg cleanups, the targets are:

1. **`kernels/flash_fwd_sm90.py`** — strip:
   - `pack_gqa`, `paged_kv_non_tma`, `mPageTable`, `PagedKVManager`
   - `blocksparse_tensors`, `BlockSparseTensors`, `use_block_sparsity`
   - `cu_seqlens_q/k`, `seqused_q/k` (varlen)
   - `is_causal`, `is_local`, `window_size_*`
   - `learnable_sink`, `softcap`, `score_mod`, `mask_mod`, `aux_tensors`
   - `q_subtile_factor`
2. **`kernels/flash_fwd.py::FlashAttentionForwardBase`** — same args.
3. **`core/pack_gqa.py`, `core/paged_kv.py`, `sparsity/`** — fully delete once SM90 stops referencing them.
4. **`core/tile_scheduler.py`** — strip `SingleTileVarlenScheduler`, LPT scheduler, persistent variants we don't use.
5. **`core/mask.py`** — strip causal/local masking; keep only seqlen-edge masking.
6. **`interface.py`** — strip ~1400 LOC of kwargs and validation tied to the above features. Target ~440 LOC like B200's.

## TF32 WGMMA fp32 path (status from earlier commits on this branch)

* `FA4_fp32/arch/wgmma_tf32.py` — PTX inline_asm primitives (fence / commit / wait / SMEM descriptor pack / `wgmma.mma_async ... m64n{N}k8.f32.tf32.tf32.f32` for N ∈ {32, 64, 96, 128}).
* `FA4_fp32/arch/hopper_helpers.py` — `MmaTF32WgmmaOp` probe; MLIR `MmaAtomSM90Type` lowering does not emit TF32 (verified, see commit message of `7f8f26f`).
* `FA4_fp32/arch/sm90_utils_tf32.py` — TF32 counterparts of `quack.sm90_utils.{partition_fragment_ABC, gemm, gemm_zero_init, gemm_w_idx}`, scaffolded with TODOs.
* `FA4_fp32/kernels/flash_fwd_sm90.py::FlashAttentionForwardSm90.mma` — fp32 entry raises `NotImplementedError` pointing at the scaffolds.

Integration is gated on filling in the fragment-partition math + iterating on H100.
