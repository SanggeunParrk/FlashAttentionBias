# `flash_bwd_sm100.py` 깊이 이해하기

> 이 문서는 fp32 bwd를 직접 손볼 수 있도록 **CuTeDSL 기초 → Blackwell SM100 아키텍처 → 이 커널의 한 줄 한 줄**까지 빠짐없이 풀어 씁니다. 기본 FA2 알고리즘은 이미 알고 있다고 가정합니다. 모르는 부분이 있으면 그 절을 끝까지 읽고 다시 와도 됩니다 — 각 절은 self-contained하게 쓰여 있습니다.

---

## 목차

0. [사전 지식 점검](#0-사전-지식-점검)
1. [CuTeDSL 5분 요약 — 큰 그림](#1-cutedsl-5분-요약--큰-그림)
2. [Layout: 모든 것의 시작](#2-layout-모든-것의-시작)
3. [Tensor: pointer + Layout](#3-tensor-pointer--layout)
4. [Layout 변환: composition, divide, slice, select](#4-layout-변환-composition-divide-slice-select)
5. [Memory hierarchy: GMEM / SMEM / RMEM / TMEM](#5-memory-hierarchy-gmem--smem--rmem--tmem)
6. [TiledCopy와 partition_S / partition_D](#6-tiledcopy와-partition_s--partition_d)
7. [TiledMma와 partition_A / B / C, make_fragment_A/B/C](#7-tiledmma와-partition_a--b--c-make_fragment_abc)
8. [recast_tensor — 같은 메모리, 다른 dtype 시점](#8-recast_tensor--같은-메모리-다른-dtype-시점)
9. [Pipelines (mbarrier 기반 비동기 동기화)](#9-pipelines-mbarrier-기반-비동기-동기화)
10. [Named barriers와 warp specialization](#10-named-barriers와-warp-specialization)
11. [Blackwell SM100 — 무엇이 새로운가](#11-blackwell-sm100--무엇이-새로운가)
12. [FA backward 알고리즘 ↔ 커널 구조 매핑](#12-fa-backward-알고리즘--커널-구조-매핑)
13. [`flash_bwd_sm100.py` — `__init__` 풀이](#13-flash_bwd_sm100py--__init__-풀이)
14. [`_setup_attributes` 풀이](#14-_setup_attributes-풀이)
15. [`_get_tiled_mma` — 5개의 MMA](#15-_get_tiled_mma--5개의-mma)
16. [`_setup_smem_layout` — SMEM 레이아웃 카탈로그](#16-_setup_smem_layout--smem-레이아웃-카탈로그)
17. [`__call__` — host-side 준비와 kernel launch](#17-__call__--host-side-준비와-kernel-launch)
18. [`kernel` — warp specialization과 pipeline 생성](#18-kernel--warp-specialization과-pipeline-생성)
19. [`load` — TMA producer warp](#19-load--tma-producer-warp)
20. [`mma` — UMMA producer warp](#20-mma--umma-producer-warp)
21. [`compute_loop` — softmax + dS 계산 (8 warp)](#21-compute_loop--softmax--ds-계산-8-warp)
22. [`dQacc_reduce` — dQ 누적 epilogue (4 warp)](#22-dqacc_reduce--dq-누적-epilogue-4-warp)
23. [`epilogue_dK_or_dV_tma` — dK/dV 출력](#23-epilogue_dk_or_dv_tma--dkdv-출력)
24. [P / dS TMEM overlap trick — 왜 fp32에서 깨지는가](#24-p--ds-tmem-overlap-trick--왜-fp32에서-깨지는가)
25. [fp32 bwd 구현 시작점 — 무엇을 어디서 손대야 하나](#25-fp32-bwd-구현-시작점--무엇을-어디서-손대야-하나)

---

## 0. 사전 지식 점검

이 문서는 이런 걸 알고 있다고 가정합니다.

- **FA2 backward 수식**: `D_i = sum_d O[i,d] * dO[i,d]`, `dS_ij = P_ij * (dP_ij - D_i)`, `dQ = dS @ K`, `dK = dS.T @ Q`, `dV = P.T @ dO`
- **CUDA 모델**: thread, warp(32 lanes), warpgroup(4 warps = 128 threads), CTA, SMEM
- **Python 기본 + PyTorch tensor**

**필요하지만 이 문서에서 다 설명하는 것**:
- CuTeDSL Layout / Tensor / TiledMma / TiledCopy
- TMA (Tensor Memory Accelerator)
- Blackwell tcgen05 / TMEM / UMMA
- TF32 MMA의 특수한 K-depth
- mbarrier / Pipeline 계열

---

## 1. CuTeDSL 5분 요약 — 큰 그림

CUTLASS는 NVIDIA가 만든 GEMM/Convolution 라이브러리입니다. 그 안의 **CuTe**는 "tensor의 layout과 tile partitioning을 표현하는 작은 DSL"이고, **CuTeDSL**은 이걸 **Python에서** 쓰는 frontend입니다. 코드는 Python처럼 보이지만 `@cute.jit` 데코레이터가 붙은 함수는 **MLIR로 컴파일되어 PTX/CUBIN으로 떨어집니다**. C++ CUTLASS와 같은 어셈블리를 만든다고 봐도 됩니다.

핵심 추상화 4가지:

| 추상화 | 정체 | 비유 |
|--------|------|------|
| **Layout** | `(shape, stride)` 쌍. coord → linear offset 함수 | NumPy의 `strides`+`shape` |
| **Tensor** | `pointer + Layout` | `numpy.ndarray` |
| **TiledMma** | "이 모양의 행렬을 이 dtype으로 곱한다"는 명세 + 어떤 thread가 어떤 element를 다루는지 | "warpgroup 단위 GEMM 한 판의 청사진" |
| **TiledCopy** | "이 모양의 데이터를 GMEM↔SMEM↔RMEM↔TMEM 사이로 옮긴다"는 명세 + thread 분배 | "복사 한 판의 청사진" |

이 4개가 결합해서 **커널 = "Tensor를 TiledMma/TiledCopy로 처리하는 코드"**가 됩니다. 거의 모든 CuTeDSL 코드는 이 골격이에요.

추가로 알아야 할 거시적 개념:

- **Warp specialization**: 한 CTA의 warp들을 역할별로 나눔(예: 일부는 데이터 로드만, 일부는 MMA만, 일부는 softmax만). 이 커널은 **16 warps를 4역할로 나눕니다**.
- **Async pipelines**: producer warp와 consumer warp가 mbarrier로 신호 주고받으며 비동기 진행.
- **TMA (Tensor Memory Accelerator)**: Hopper에서 도입된 "한 큐에 박스 단위(=tile) 메모리 복사" 가속기. SMEM↔GMEM의 비동기 복사를 한 명령으로 처리.

---

## 2. Layout: 모든 것의 시작

CuTe에서 **Layout = `(shape, stride)`**. shape는 차원 크기, stride는 차원별 stride입니다. coord → offset 함수예요.

```python
import cutlass.cute as cute

# Layout((4, 8), stride=(8, 1))은 행 우선(row-major) 4x8.
# coord (i, j) → offset i*8 + j
L = cute.make_layout((4, 8), stride=(8, 1))
print(L)  # (4,8):(8,1)
```

### 2.1 모드 (mode)

shape는 **계층적**일 수 있어요. `((2,2), 4)` 같은 것은 "첫 번째 모드는 (2,2)로 분해된 4, 두 번째 모드는 4"입니다. 이렇게 nested된 거를 **mode**라고 부릅니다. mode-0의 shape는 `(2,2)`, mode-1은 `4`.

```python
L = cute.make_layout(((2, 2), 4), stride=((4, 16), 1))
# coord ((i0, i1), j) → i0*4 + i1*16 + j*1
```

이게 왜 중요하냐면, **MMA 결과의 thread별 element index**가 보통 nested layout으로 나옵니다. 예를 들어 한 thread가 (4 elements, 2 packs) 식으로 받으면 mode-0이 `(4, 2)`로 표현됩니다.

### 2.2 Layout 함수 호출

`L(coord) → offset` 으로 직접 부를 수 있어요.

```python
L = cute.make_layout((4, 8), stride=(8, 1))
print(L((1, 3)))  # 1*8 + 3*1 = 11
```

### 2.3 cosize / size

- `cute.size(L)` = 모든 element 개수 = shape의 모든 차원 곱 = `4 * 8 = 32`
- `cute.cosize(L)` = 가장 큰 offset + 1 = "이 layout이 차지하는 메모리 크기"

대개 row/col-major면 `size == cosize`지만, broadcasted layout(stride=0이 섞인 것)은 `cosize < size` 가능.

### 2.4 ComposedLayout (with swizzle)

SMEM tile에 효율적으로 접근하려면 **swizzle**(메모리 bank conflict 회피)이 필요해요. Swizzle을 layout에 합친 게 `ComposedLayout`입니다. `layout.outer`(원본 layout) + `layout.inner`(swizzle 함수)로 분해되어 저장됩니다. 우리 코드에서 `sQ_layout.outer`, `sQ_layout.inner`로 자주 쓰입니다.

```python
sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner, dtype=self.q_dtype)
```

---

## 3. Tensor: pointer + Layout

```python
T = cute.make_tensor(iterator, layout)
```

- `iterator` = 메모리 시작점 (pointer 비슷). GMEM/SMEM/TMEM/RMEM 어디든 가리킬 수 있음.
- `layout` = 위에서 본 Layout.
- `T[(i, j)]` = element at coord (i, j) — `iterator + layout((i,j))`.

자주 쓰는 helper:

| 호출 | 역할 |
|------|------|
| `cute.make_tensor(it, layout)` | 직접 생성 |
| `cute.make_fragment(shape, dtype)` | **register memory(rmem)** 텐서 생성. 컴파일 타임에 알 수 있는 size여야 함 |
| `cute.make_rmem_tensor(shape, dtype)` | `make_fragment`와 거의 같음 |
| `T.iterator` | 메모리 포인터 |
| `T.layout` | Layout |
| `T.shape`, `T.stride` | shape/stride 튜플 |
| `T.element_type` | dtype (e.g. `Float32`, `BFloat16`) |
| `T.load() / T.store(ssa)` | rmem 텐서를 SSA value로 읽기/쓰기 |
| `T[None, i, j]` | "axis 0 전체, axis 1=i, axis 2=j" — slicing |

### 3.1 SSA vs in-place

CuTeDSL은 MLIR 위에서 동작하므로 register 값이 **SSA** 형태입니다. tensor element를 직접 `T[i] = ...`로 쓸 수 있는 건 syntactic sugar이고 내부적으로 SSA value로 쪼개집니다.

```python
val = T.load()        # T 전체를 한 SSA value로
val = val + 1.0       # SSA op
T.store(val)          # 다시 적기

# 또는 인덱싱
T[0] = T[0] + 1.0     # element 단위
```

### 3.2 RMEM 텐서 = register, SMEM/TMEM 텐서 = shared / tensor memory

`make_fragment`는 **register**에 살게 됩니다. shape는 컴파일 타임에 fix되어야 하고, 보통 한 thread당 처리하는 element 수 (per-thread tile)에 해당해요.

SMEM/TMEM 텐서는 `storage.sX.get_tensor(layout, ...)`처럼 SharedStorage struct에서 받아옵니다.

---

## 4. Layout 변환: composition, divide, slice, select

이걸 모르면 이 커널 못 읽어요. 차근차근.

### 4.1 `cute.slice_(layout, coord)` — partial 슬라이싱

shape 일부에 고정값을 넣고 나머지는 그대로. NumPy의 `[0, :, :]`과 비슷.

```python
L = cute.make_layout((4, 8, 16), stride=(128, 16, 1))
L2 = cute.slice_(L, (None, 3, None))  # shape (4, 16), stride (128, 1)
```

### 4.2 `layout_utils.select(tensor, mode=[1, 0, 2])` — 모드 재배열

axis 순서를 바꿈. 우리 코드에서 `(b, s, n, h)` 텐서를 `(s, h, n, b)`로 바꿀 때 자주 등장.

```python
# (b=2, s=128, n=4, h=64) → (s=128, h=64, n=4, b=2)
mQ = layout_utils.select(mQ, mode=[1, 3, 2, 0])
```

### 4.3 `cute.composition(A, B)` — A를 B로 재해석

`B`가 "어떤 좌표를 어떤 (sub-)좌표로 변환하는지" 정의하면, `composition(A, B)`는 그 변환을 통과한 결과 layout. 예를 들어 `tStP = composition(tStS, ((tile_n, tileP_f32_like), 1, 1))` 은 **tStS의 일부를 새로운 (tile_n, tileP_f32_like) 모양으로 재해석**합니다.

```python
# 큰 layout의 sub-region을 재해석
tStS  # ((128, 128), 1, 1) — 128x128 acc
tStP = cute.composition(tStS, (cute.make_layout((128, 64)), 1, 1))
# tStP: ((128, 64), 1, 1) — 첫 64열만 보는 view (memory는 같음)
```

### 4.4 `cute.local_tile(t, tile_shape, coord)` — 큰 텐서를 타일로 나누고 한 타일 선택

```python
gK = cute.local_tile(mK_cur, (tile_n, tile_hdim), (n_block, 0))
# mK_cur 전체에서 (n_block, 0)번째 타일을 가져옴
```

`coord`에 `None`을 넣으면 그 axis 전체가 살아남아 stage 차원이 됩니다:

```python
gQ = cute.local_tile(mQ_cur, (tile_m, tile_hdim), (None, 0))
# 결과 shape: (tile_m, tile_hdim, num_m_blocks)
# m_block 차원이 외부 축으로 보존됨
```

### 4.5 `cute.flat_divide(t, tile_shape)` — divide & flatten

`local_tile`과 비슷한데, divide된 결과를 flat하게 펼침. 결과가 더 풀린 (rank 증가) 형태.

### 4.6 `cute.logical_divide(t, sub_layout)` — 차원을 sub-layout으로 쪼갬

기존 axis를 (inner, outer)로 분해.

```python
# 분량 128을 (16, 8) 식으로 나누기
chunked = cute.logical_divide(t, cute.make_layout(16))
# t.shape: (128, ...) → chunked.shape: ((16, 8), ...)
```

`fp32 dQ epilogue`에서 이 함수가 핵심으로 쓰여요 — 16-col 청크로 나누고 chunk 단위 loop를 도는 패턴입니다.

### 4.7 `cute.group_modes(t, start, end)` — 여러 모드를 하나로 묶기

`(M, N, K)` 모양을 `((M*N), K)`로 만드는 것 같은 작업. TMA partitioning 직전에 자주 등장.

---

## 5. Memory hierarchy: GMEM / SMEM / RMEM / TMEM

CuTe 텐서는 어디에 사는지에 따라 처리법이 달라요.

| 메모리 | 크기 (B200) | 접근 방식 |
|--------|------------|----------|
| **GMEM** (global) | HBM 192 GB | TMA, cp.async, 일반 load/store |
| **SMEM** (shared) | 228 KB / SM | 워프 협력 LD/ST, async copy |
| **RMEM** (register) | thread별 ~256 vector reg | 직접 사용 (`make_fragment`) |
| **TMEM** (tensor memory, **SM100 신규**) | 512 cols × 32 rows / SM | UMMA의 input/output, 전용 atom으로 LD/ST |

### 5.1 TMEM이란

Blackwell SM100이 새로 도입한 메모리 영역. **MMA accumulator가 register가 아니라 TMEM에 저장**됩니다 (Hopper UMMA에서 acc는 register였음). TMEM은 SM 안의 별도 메모리고, 32-bit word 기반.

- 크기: 512 columns × 32 rows = 16384 fp32 word ≈ 64 KB / SM
- column 기준으로 분할: e.g. dV가 hdim=64라면 64 cols 사용
- 접근: `tcgen05.copy.Ld32x32bOp` (load) / `St32x32bOp` (store) atom으로만

이 커널의 TMEM 레이아웃 (한 SM, 1-CTA):

```
column 0 ────────────────────────────────────────── 511
[ S/P (128 cols) ][ dV (hdimv) ][ dP/dS/dQ (128 cols) ][ dK (hdim) ]
                                                       ↑
                                            tmem_dK_offset = tmem_dP_offset + tile_m
```

`tmem_S_offset = 0`, `tmem_P_offset = 0` (S와 같은 영역, S 다 쓴 후 P 덮어씀).

### 5.2 RMEM (`make_fragment`)

```python
tSrS = cute.make_fragment((32, 1, 1), Float32)
# 한 thread당 32개 fp32 register 차지
# .load() / .store() 또는 [i] indexing으로 사용
```

shape는 thread별 view. 예: `partition_S` 결과의 per-thread 부분.

### 5.3 SMEM 가져오기

```python
# In SharedStorage struct (compile time):
sX: cute.struct.Align[
    cute.struct.MemRange[dtype, cute.cosize(layout)],
    1024,
]

# In kernel body:
storage = smem.allocate(self.shared_storage)
sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner, dtype=self.q_dtype)
```

`get_tensor`는 SMEM iterator + layout으로 cute.Tensor를 만들어 줍니다.

### 5.4 TMEM 가져오기

```python
tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
# offset 0부터 시작하는 TMEM pointer

# 특정 영역 (acc 단위) 텐서:
tStS = thr_mma_S.make_fragment_C(Sacc_shape)
tStS = cute.make_tensor(tmem_ptr + self.tmem_S_offset, tStS.layout)
```

`make_fragment_C`는 MMA의 acc(C) 모양을 알아서 만들어 줍니다. 그걸 TMEM 주소에 붙여서 fictional한 acc tensor를 만드는 거예요.

---

## 6. TiledCopy와 partition_S / partition_D

### 6.1 TiledCopy 만들기

데이터를 옮길 때 (예: SMEM→RMEM), 어떤 atom으로 어떤 thread가 어떻게 분할해서 옮길지를 정의:

```python
# 단순한 universal copy
copy_atom = cute.make_copy_atom(
    cute.nvgpu.CopyUniversalOp(),
    dtype,
    num_bits_per_copy=128,  # 한 번에 128bit씩 (8 bf16 또는 4 fp32)
)
tiled_copy = cute.make_tiled_copy_tv(
    copy_atom,
    thr_layout,  # threads의 (M, N) 배치
    val_layout,  # thread당 처리하는 element의 (M, N) 모양
)
```

`thr_layout × val_layout = tile shape`. 즉 thread 배치 × thread당 elements = 한 번에 옮기는 tile 모양.

### 6.2 `get_slice(tidx)` → ThrCopy

특정 thread `tidx` 시각에서 본 copy:

```python
thr_copy = tiled_copy.get_slice(tidx)
```

### 6.3 `partition_S(tensor)` / `partition_D(tensor)`

이 thread가 source(S)/destination(D) 텐서에서 **어떤 부분을 다루는지**를 반환:

```python
tXgX = thr_copy.partition_S(global_tensor)
tXsX = thr_copy.partition_D(smem_tensor)
# 둘 다 thread 시각의 sub-tensor.
# cute.copy(tiled_copy, tXgX, tXsX)로 GMEM→SMEM 복사
```

shape는 보통 `(CPY_atom, CPY_M, CPY_N, ...stages)` 같은 nested 형태. 예: `((4,8), 2, 1, 4)`는 "thread당 32 elements를 (4×8 atom × 2 atom_M × 1 atom_N) 형태로 4 stage 처리".

### 6.4 TMA atom

TMA는 Hopper에서 도입된 "tile 단위 비동기 copy" 가속기. 일반 copy보다 큰 단위로 한 번에 여러 KB 이동.

```python
tma_atom_K, mK_tma = cute.nvgpu.make_tiled_tma_atom_A(
    cpasync.CopyBulkTensorTileG2SOp(cta_group),
    mK,                              # GMEM 텐서
    cute.select(sK_layout, [0,1,2]), # SMEM 단일 stage layout
    mma_tiler,
    tiled_mma,
    cluster_layout_vmnk.shape,
)
# tma_atom_K: TMA descriptor + atom
# mK_tma: tma 설정에 맞춰 재해석된 GMEM tensor (실제 데이터는 같음)
```

TMA는 1 thread (보통 elect_one)가 발사하면 hardware가 알아서 끝까지 이동시킵니다. 완료는 mbarrier로 통보됨.

### 6.5 `cute.copy(atom, src, dst)` 종류

- 일반 copy: `cute.copy(thr_copy, src_partitioned, dst_partitioned)` — RMEM ↔ SMEM
- TMA copy: `cute.copy(tma_atom, src, dst, tma_bar_ptr=...)` — GMEM ↔ SMEM, mbarrier로 완료 신호
- `cute.autovec_copy(src, dst)` — 가장 큰 vector load/store로 자동 매칭

---

## 7. TiledMma와 partition_A / B / C, make_fragment_A/B/C

### 7.1 TiledMma 만들기

GEMM `C = A @ B`를 한 번 수행하는 단위 + thread 분배.

```python
tiled_mma = sm100_utils_basic.make_trivial_tiled_mma(
    a_dtype,
    a_major_mode,    # OperandMajorMode.K or .MN
    b_major_mode,
    acc_dtype,       # 보통 Float32
    cta_group,       # CtaGroup.ONE or .TWO
    mma_tiler,       # (M, N) tuple
    a_source=tcgen05.OperandSource.SMEM,  # 또는 .TMEM
)
```

- `a_dtype` = A operand의 dtype. fp16/bf16/fp32(=TF32). MMA의 K-depth가 dtype에 따라 결정됨 (16 for fp16/bf16, 8 for TF32).
- `a_source`/`b_source` = A/B가 어디 메모리에 사는지. Hopper는 둘 다 SMEM, SM100은 A를 TMEM에서도 읽을 수 있음.
- `mma_tiler = (M, N)` = 한 MMA 결과의 모양.

### 7.2 `get_slice(coord)` → ThrMma

```python
thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
# coord_v는 cluster 안에서의 위치. 1-CTA면 0.
```

### 7.3 `partition_A / B / C(tensor)`

- `partition_A(sA)` = thread가 A operand에서 읽을 부분 (shape는 `(MMA_atom, MMA_M, MMA_K, ...)` 식)
- `partition_B(sB)` = 마찬가지로 B
- `partition_C(cC)` = thread가 C(=acc) 결과를 받는 부분

`cC`는 보통 `cute.make_identity_tensor(mma_tiler)` — coord만 추적하는 가짜 tensor. partition_C로 thread별 좌표 view를 얻는 게 목적입니다.

### 7.4 `make_fragment_A / B / C(...)` — register accumulator

```python
tStS = thr_mma_S.make_fragment_C(acc_shape)
# thread별 acc fragment를 만듦. shape는 partition_C 결과의 모양.
# SM100에서는 이 fragment를 TMEM에 매핑함
```

`make_fragment_A(sA)`처럼 SMEM tensor를 넘기면 그 SMEM에서 읽어올 형식의 register fragment를 만듭니다 (실제 데이터는 cute.copy 시점에 채워짐).

### 7.5 `cute.gemm(tiled_mma, acc, tCrA, tCrB)` — 한 GEMM 발사

```python
cute.gemm(tiled_mma, tStS, tSrK, tSrQ)
# tStS는 누적 destination, tSrK/tSrQ는 A/B fragments
```

SM100에서는 이게 UMMA 명령 한 번 발사로 컴파일됩니다 (asynchronous). 이 명령 끝났는지는 다른 곳에서 mbarrier로 확인.

이 커널에서는 `gemm_w_idx`, `gemm_ptx_w_idx`, `gemm_ptx_partial` 같은 wrapper도 쓰는데, 이는 UMMA를 inline-PTX로 직접 발사하는 fastpath입니다 (CuTe DSL의 자동 생성보다 미세 튜닝 가능).

### 7.6 mma_tiler vs mma_tile_size vs MMA atom

용어가 헷갈려서 정리:

- **MMA atom shape**: HW가 한 명령에 처리하는 (M, N, K) — fp16/bf16: M=64 N=8~256 K=16, TF32: K=8.
- **mma_tiler**: 우리가 한 *논리적* GEMM에서 처리하려는 (M, N, K). e.g. `(128, 128, 128)`. 이건 atom보다 클 수 있고, 그러면 atom을 여러 번 부르도록 컴파일됨.

이 차이가 fp32 bwd의 핵심 어려움 — TF32는 atom K=8이라 같은 (M,N,K) 처리에 atom 호출 횟수가 fp16/bf16의 2배.

---

## 8. recast_tensor — 같은 메모리, 다른 dtype 시점

```python
# fp32 fragment를 bf16 view로 재해석 (recast)
tSrP_r2t_f32 = cute.make_fragment(shape, Float32)  # 16 fp32 elements
tSrP_r2t = cute.recast_tensor(tSrP_r2t_f32, BFloat16)
# tSrP_r2t는 같은 메모리지만 32 bf16 element view (4 bytes ÷ 2 bytes = 2x)
```

bytes 수를 보존한 채 element 개수가 dtype 폭에 따라 변함:
- Float32 → BFloat16: 2배 elements
- Float32 → Float32: 1배 (no-op, 그래도 컴파일러는 새 view 생성)
- Float32 → Float16: 2배

이 커널은 P/dS를 fp16/bf16에 packing하는 트릭에 이 함수를 씁니다 (§24 참고).

---

## 9. Pipelines (mbarrier 기반 비동기 동기화)

CUTLASS의 `pipeline` 모듈이 producer-consumer 모델을 mbarrier로 추상화합니다. 이 커널에서는 **8개 pipeline**을 사용해요.

### 9.1 Pipeline 종류

| Pipeline 클래스 | 의미 |
|----------------|------|
| `PipelineTmaAsync` | TMA producer + async (compute) consumer |
| `PipelineTmaUmma` | TMA producer + **UMMA** consumer (Hopper에서는 Async, SM100 신규는 Umma) |
| `PipelineUmmaAsync` | UMMA producer + async consumer |
| `PipelineAsyncUmma` | Async producer + UMMA consumer |

뒤의 두 개가 SM100 전용 — UMMA가 비동기 issue→완료 모델이라 명시적으로 표시.

### 9.2 Producer / Consumer 패턴

```python
# Producer side
producer_state = make_pipeline_state(PipelineUserType.Producer, num_stages)
pipeline.producer_acquire(producer_state)  # 이 stage 비어있을 때까지 대기
# 데이터 채워넣기 (TMA copy 등)
pipeline.producer_commit(producer_state)   # "이 stage 다 채웠다" 신호
producer_state.advance()

# Consumer side
consumer_state = make_pipeline_state(PipelineUserType.Consumer, num_stages)
pipeline.consumer_wait(consumer_state)     # producer가 commit할 때까지 대기
# 데이터 사용
pipeline.consumer_release(consumer_state)  # "이 stage 다 썼다" 신호
consumer_state.advance()
```

`num_stages`는 circular buffer 깊이. e.g. `Q_stage=2`면 producer가 항상 2 stage 앞서 갈 수 있음.

### 9.3 phase, index

PipelineState 내부엔 `phase` (0/1)와 `index` (0..num_stages-1)가 있습니다. `advance()`마다 index 증가, num_stages 한 바퀴 돌 때마다 phase flip.

### 9.4 sync_object_full / sync_object_empty

저수준 mbarrier 직접 접근. 이 커널에서 `pipeline_S_P.sync_object_full.arrive(0, mask, cta_group)` 식으로 종종 등장.

### 9.5 `cta_layout_vmnk`

cluster 내 CTA들의 layout. `(v, m, n, k)` = (mode_v, cluster_m, cluster_n, K=1). 현재 build는 cluster=(1,1)이라 다 1.

---

## 10. Named barriers와 warp specialization

CTA 내 warp들이 이름 붙여진 barrier로 동기화. CUDA의 `__syncthreads()` 한 번 = barrier 0 전체. 다른 ID(1~15)를 명시적으로 쓰면 일부 warp만 동기화 가능.

```python
named_barrier = cutlass.pipeline.NamedBarrier(
    barrier_id=int(NamedBarrierBwdSm100.Compute),
    num_threads=8 * 32,  # 8 warps만 참여
)
named_barrier.arrive_and_wait()  # 이 8 warps 모두 도달 → 통과
```

이 커널의 named barrier 카탈로그 ([core/named_barrier.py](core/named_barrier.py#L19-L24)):
- `EpilogueWG1`, `EpilogueWG2`: dK/dV epilogue 동기화 (warp group 별)
- `Compute`: 8 compute warps 동기화
- `dQaccReduce`: 4 reduce warps 동기화
- `TmemPtr`: TMEM allocation 핸드셰이크

### 10.1 Warp 역할 분배 (이 커널)

```
warp_idx | 역할
---------+-------------------
0..3     | reduce  (dQ accumulator gmem 누적)
4..11    | compute (softmax, dS 계산, dK/dV epilogue)
12       | mma     (UMMA 명령 발사)
13       | load    (TMA Q/K/V/dO/LSE/dPsum 로드)
14       | empty   (idle)
15       | empty   (idle)
```

각 warp는 `if warp_idx == ...` 분기로 자기 역할 코드 실행.

### 10.2 Register budget

각 역할별 register 한도를 `setmaxregister_decrease/_increase`로 선언:
- reduce: 152 reg/thread
- compute: 136 reg/thread
- mma: 88 reg/thread
- load: 88 reg/thread
- empty: 24 reg/thread

총합이 SM당 register file (256 KB / SM = 65536 reg / SM, 512 thread/CTA → 128 reg/thread average) 안에 들어가야 하고, 더 많은 CTA를 SM에 쌓고 싶으면 더 줄여야.

---

## 11. Blackwell SM100 — 무엇이 새로운가

Hopper(SM90) 대비 핵심 차이:

### 11.1 TMEM (Tensor Memory)

이미 §5에서 다뤘지만 강조: **MMA acc가 TMEM에 살고, register가 아닙니다.** 그래서 `make_fragment_C` 결과를 TMEM 주소에 매핑.

장점:
- Acc가 register file을 안 먹음 → softmax 같은 후속 연산이 더 많은 register 사용 가능
- TMEM은 dual-bank이라 동시에 여러 MMA의 acc 존재 가능

단점:
- TMEM 명시적 load/store 필요 (Ld32x32bOp atom 등)
- 새 동기화 모델 학습 필요

### 11.2 UMMA (`tcgen05.mma`)

SM100의 새 MMA. 비동기 issue:
1. 어느 thread (보통 한 명) 가 `tcgen05.mma` PTX 명령 발사
2. HW가 비동기로 실행
3. 완료 시 mbarrier 통보 (혹은 auto-sync)
4. 다른 warp가 acc 읽기 가능

UMMA atom shape: M (16/64/128/256) × N (8~256) × K (8/16/32, dtype 의존).

### 11.3 TF32 MMA의 K=8

이게 핵심:
- fp16/bf16 atom: K=16 (e.g. M64N128K16)
- **TF32 atom: K=8** (e.g. M64N128K8)

같은 reduction을 하려면 TF32는 atom 호출 2배. **결과 acc 한 instance가 TMEM에서 차지하는 column 수도 다릅니다** (자세한 건 §24).

### 11.4 1-CTA / 2-CTA

UMMA는 "이 한 CTA가 한다" / "두 CTA가 한 cluster로 협력한다" 두 모드. 후자는 TMEM도 cluster-wide로 분배. 이 커널은 **1-CTA only** (`cluster_size = 1`).

### 11.5 `cp.async.bulk` (TMA 후속)

Hopper TMA의 확장. SMEM↔GMEM 외에 **gmem reduce-add** 도 지원 (atomic reduce). `cpasync_reduce_bulk_add_f32`로 dQ accumulator 누적 시 사용.

---

## 12. FA backward 알고리즘 ↔ 커널 구조 매핑

기본 FA2 bwd는 (n_block, head, batch) tile 하나당 다음을 함:

```
Pre-loop (이미 끝남, preprocess kernel이 함):
  D[i] = sum_d O[i,d] * dO[i,d]    # (B, H, L)
  lse_log2[i] = LSE[i] * log2(e)
  zero out dq_accum

Main loop over m_block:
  Load Q[m_block], dO[m_block], LSE[m_block], D[m_block]
  S  = K @ Q.T          (tile_n × tile_m, fp32 acc)
  P  = exp(S * scale - LSE) (Float32 → narrow dtype)
  dP = V @ dO.T         (tile_n × tile_m, fp32 acc)
  dS = P * (dP - D)     (Float32 → narrow dtype)
  dV += P.T @ dO        (tile_n × tile_hdimv, fp32 acc accumulate)
  dK += dS.T @ Q        (tile_n × tile_hdim, fp32 acc accumulate)
  dQ_partial = dS @ K   (tile_m × tile_hdim, fp32) → atomic add to dq_accum

After loop:
  Write dV, dK to gmem (typed, narrow dtype)

Postprocess kernel:
  dq = dq_accum * softmax_scale  (typed)
```

### 12.1 5개 GEMM과 매핑

| GEMM | Acc | A operand | B operand | A source | B source |
|------|-----|-----------|-----------|----------|----------|
| `S` | TMEM 0..127 | K | Q.T | SMEM | SMEM |
| `dP` | TMEM 192..319 | V | dO.T | SMEM | SMEM |
| `dV += P.T @ dO` | TMEM 128..(128+hdimv) | P | dO | **TMEM** | SMEM |
| `dK += dS.T @ Q` | TMEM 320..(320+hdim) | dS | Q | **TMEM** | SMEM |
| `dQ = dS @ K` | TMEM 192.. | dS | K.T | SMEM | SMEM |

P와 dS는 acc로 처음 만들어진 게 아니라 **softmax로 계산한 결과를 TMEM에 다시 적어 넣은 것**입니다. UMMA의 A operand로 쓰려면 TMEM이거나 SMEM이어야 하니까요.

### 12.2 Warp 역할과 5개 GEMM

| Warp 역할 | 무엇을 하나 |
|----------|-----------|
| **Load** (1 warp) | Q/K/V/dO/LSE/dPsum을 GMEM→SMEM (TMA) |
| **MMA** (1 warp) | 5개 UMMA 명령을 순서대로 발사 |
| **Compute** (8 warps) | S→P, dP→dS 계산. P/dS를 TMEM에 다시 적기. dK/dV epilogue (TMEM→RMEM→SMEM→GMEM via TMA) |
| **Reduce** (4 warps) | dQ acc(TMEM)→SMEM→GMEM atomic-add |

### 12.3 Pipeline 의존성

```
Q (TMA) ────────┐
K (TMA, single) ┴──→ S MMA ──→ S in TMEM ────→ Compute reads S, writes P
                                                       │
dO (TMA) ──┐                                           │
V (TMA, single) ────→ dP MMA ──→ dP in TMEM ──→ Compute reads dP, writes dS
                                                       │
                            P in TMEM ─→ dV MMA (acc in TMEM) ─→ Epilogue
                                                       │
                            dS in TMEM ─→ dK MMA (acc in TMEM) ─→ Epilogue
                                       └→ dQ MMA (acc in TMEM) ─→ dQ Reduce
```

각 화살표마다 mbarrier로 producer-consumer 동기화. `pipeline_Q`, `pipeline_dO`, `pipeline_S_P`, `pipeline_dP`, `pipeline_dS`, `pipeline_dKV`, `pipeline_dQ`, `pipeline_LSE`, `pipeline_dPsum` — **9개 pipeline**.

---

## 13. `flash_bwd_sm100.py` — `__init__` 풀이

```python
def __init__(
    self,
    head_dim: int,
    tile_m: int = 128,
    tile_n: int = 128,
    subtile_factor: cutlass.Constexpr[int] = 1,
    q_dtype: Optional[type] = None,
):
    self._init_q_dtype = q_dtype
    hdim_multiple_of = 16
    self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
    self.tile_hdimv = self.tile_hdim
```

- **`head_dim`**: 입력 head dim. 사용자가 32 또는 64를 쓴다고 하셨음.
- **`tile_m`, `tile_n`**: Q-축 / K-축 블록 크기. interface.py에서 fp32일 때 `(64, 128)`로 강제, fp16/bf16에서 `(128, 128)`. **여기서 m은 Q쪽, n은 K쪽**임에 주의 (FA fwd와 의미가 다름 — bwd는 "K-축으로 반복"이므로 outer loop 단위가 n_block).
- **`subtile_factor`**: dQ accumulator subdivide. 항상 2로 호출됨.
- **`q_dtype`**: __init__ 시점에 알면 dtype-dependent layout 결정 가능. 안 주면 `__call__`에서 `mQ.element_type`으로 set.
- **`tile_hdim`**: hdim을 16배수로 padding. e.g. head_dim=32 → 32, head_dim=64 → 64, head_dim=48 → 64.
- **`tile_hdimv = tile_hdim`**: Q=K=V 가정으로 같음.

### 13.1 MMA tiler 정의

```python
self.cta_tiler = (tile_n, tile_m, self.tile_hdim)
self.mma_tiler_kq  = (tile_n, tile_m, self.tile_hdim)   # S = K @ Q.T
self.mma_tiler_vdo = (tile_n, tile_m, self.tile_hdimv)  # dP = V @ dO.T
self.mma_tiler_pdo = (tile_n, self.tile_hdimv, tile_m)  # dV = P.T @ dO
self.mma_tiler_dsq = (tile_n, self.tile_hdim, tile_m)   # dK = dS.T @ Q
self.mma_tiler_dsk = (tile_m, self.tile_hdim, tile_n)   # dQ = dS @ K
```

각 GEMM `C = A @ B`의 (M, N, K). **M은 acc의 첫 axis, N은 두 번째 axis, K는 reduction 축**.

`mma_tiler_dsk`의 첫 axis가 `tile_m`인 게 흥미로워요 — dQ는 Q-축 방향(=tile_m)이 acc의 M축. 다른 4개는 모두 K-축(=tile_n)이 M축.

### 13.2 16-warp 분배

```python
self.reduce_warp_ids = (0, 1, 2, 3)
self.compute_warp_ids = (4, 5, 6, 7, 8, 9, 10, 11)
self.mma_warp_id = 12
self.load_warp_id = 13
self.empty_warp_id = 15
self.threads_per_cta = cute.arch.WARP_SIZE * 16   # = 512
```

15 warps live + 1 idle (warp 14). 총 512 thread/CTA.

### 13.3 NamedBarrier

```python
self.compute_sync_barrier = cutlass.pipeline.NamedBarrier(
    barrier_id=int(NamedBarrierBwdSm100.Compute),
    num_threads=len(self.compute_warp_ids) * cute.arch.WARP_SIZE,  # 8 * 32 = 256
)
self.reduce_sync_barrier = cutlass.pipeline.NamedBarrier(
    barrier_id=int(NamedBarrierBwdSm100.dQaccReduce),
    num_threads=4 * 32 = 128,
)
```

8 compute warps 자체 sync용, 4 reduce warps 자체 sync용. mma/load warp는 자기 역할에 다른 warp가 없으므로 barrier 불필요.

### 13.4 TMEM 영역 배치

```python
self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")  # 512
self.tmem_S_offset = 0
self.tmem_P_offset = 0  # S와 같은 영역, S 다 쓴 후 덮어씀
self.tmem_dV_offset = self.tmem_S_offset + self.tile_n           # = 128
self.tmem_dP_offset = self.tmem_dV_offset + self.tile_hdimv      # = 128 + 64 = 192 (hdim 64일 때)
self.tmem_dQ_offset = self.tmem_dP_offset                        # dP와 같은 영역
self.tmem_dK_offset = self.tmem_dP_offset + self.tile_m          # = 192 + 128 = 320
self.tmem_dS_offset = self.tmem_dP_offset                        # dP와 같은 영역
```

**주목**: P=S, dQ=dP, dS=dP 영역 overlap. 시간적으로 안 겹치게 (S 다 쓴 다음 P 적기) 동기화로 보장. 자세히는 §24.

총 사용량 (hdim=64 기준): 0..128 (S/P) + 128..192 (dV) + 192..320 (dP/dQ/dS) + 320..384 (dK) = **384 cols**. 512 한도 안에.

(hdim=128일 때: dV가 128 쓰고, dK가 128 쓰면 0..128 + 128..256 + 256..384 + 384..512 = 정확히 512. 빡빡함.)

### 13.5 Register budget

```python
self.num_regs_reduce = 152
self.num_regs_compute = 136
self.num_regs_load = 96 - 8
self.num_regs_mma = self.num_regs_load
self.num_regs_empty = 24

assert self.num_regs_reduce + self.num_regs_compute * 2 + max(self.num_regs_load, self.num_regs_mma) <= 512
```

`compute * 2`는 SM100에서 register slot이 한 warp group(=4 warps=128 threads) 단위로 잡히므로, 8 warp = 2 wg → 2배. assert가 SM당 register 한도 (512 reg/thread × 0 ... 사실은 주석의 의미는 unique stake). 어쨌든 budget 검증.

---

## 14. `_setup_attributes` 풀이

```python
def _setup_attributes(self):
    self.Q_stage = 2
    self.dO_stage = 1
    self.single_stage = 1
    self.sdKVaccum_stage = 2
    if self.q_dtype is Float32:
        self.dQ_reduce_ncol = 16
    else:
        self.dQ_reduce_ncol = 32
    self.sdQaccum_stage = 64 // self.dQ_reduce_ncol
    self.dQ_reduce_ncol_t2r = self.dQ_reduce_ncol
    assert self.tile_hdim % self.dQ_reduce_ncol == 0
    self.dQaccum_reduce_stage = self.tile_hdim // self.dQ_reduce_ncol
    self.dQaccum_reduce_stage_t2r = self.tile_hdim // self.dQ_reduce_ncol_t2r
    self.dK_reduce_ncol = math.gcd(32, self.tile_hdim // 2)
```

### 14.1 stage 수

- `Q_stage = 2`: Q는 2 stage 미리 받기 (load와 mma overlap)
- `dO_stage = 1`: dO는 1 stage씩 (느리게 해도 됨)
- `single_stage = 1`: S/P, dP, dS 등 in-flight 1개씩
- `sdKVaccum_stage = 2`: dV/dK epilogue가 2 stage TMEM→SMEM→GMEM 파이프

### 14.2 fp32 분기 (dQ TMEM tiling)

`dQ_reduce_ncol`은 dQ acc를 TMEM에서 RMEM으로 옮길 때 한 chunk의 column 수. **TF32 MMA의 acc layout이 fp16/bf16과 달라서 fp32 시 16, 아니면 32**.

이유: TF32 MMA는 한 atom call로 반쪽 column만 채움. dQ acc 전체 hdim cols = 32 (TF32) × 2 chunks 또는 64 cols × 1 chunk (fp16/bf16). 다음 절(§15)에서 더 자세히.

`sdQaccum_stage = 64 / dQ_reduce_ncol`은 dQ smem buffer 단계 수 (4 vs 2).

### 14.3 dK reduce ncol

`dK_reduce_ncol = gcd(32, tile_hdim // 2)`. dK epilogue 단계 수 결정. tile_hdim=64면 gcd(32, 32)=32. tile_hdim=32면 gcd(32, 16)=16.

---

## 15. `_get_tiled_mma` — 5개의 MMA

```python
def _get_tiled_mma(self):
    tiled_mma_S = sm100_utils_basic.make_trivial_tiled_mma(
        self.q_dtype,                        # A operand dtype
        tcgen05.OperandMajorMode.K,           # A is K-major
        tcgen05.OperandMajorMode.K,           # B is K-major
        self.acc_dtype,                       # acc = Float32
        tcgen05.CtaGroup.ONE,
        self.mma_tiler_kq[:2],                # (M, N) = (tile_n, tile_m)
    )
    # ... 4 more
```

- **`q_dtype`** = A operand의 element type. fp32 → TF32 MMA, bf16 → bf16 MMA.
- **`OperandMajorMode.K`** = inner stride가 K 방향 (행 우선이 K). e.g. SMEM에 `(M, K)` 모양으로 K가 contiguous.
- **`OperandMajorMode.MN`** = MN이 contiguous. transpose된 view.

### 15.1 5개의 MMA 의미

| 변수 | 식 | A | B | acc 위치 |
|------|---|---|---|---------|
| `tiled_mma_S` | S = K @ Q.T | K | Q | S (TMEM 0..) |
| `tiled_mma_dP` | dP = V @ dO.T | V | dO | dP (TMEM 192..) |
| `tiled_mma_dV` | dV += P.T @ dO | P | dO | dV (TMEM 128..) |
| `tiled_mma_dK` | dK += dS.T @ Q | dS | Q | dK (TMEM 320..) |
| `tiled_mma_dQ` | dQ = dS @ K | dS | K.T | dQ (TMEM 192..) |

`a_source=tcgen05.OperandSource.TMEM`이 dV와 dK에만 붙어 있어요 — A operand인 P/dS가 다른 MMA의 acc에서 변형된 (TMEM에 살아 있는) 데이터라서.

### 15.2 fp32 (TF32)에서 atom shape

`make_trivial_tiled_mma`는 dtype을 보고 atom 자동 선택:
- bf16 input: `tcgen05.mma.MmaSm100Atom(M64N128K16)` 같은 거
- fp32 input: `tcgen05.mma.MmaSm100Atom(M64N128K8)` 같은 거 (TF32 atom)

mma_tiler가 (M=128, N=128, K=128)인데 atom이 K=8이면, 컴파일러가 **K dim을 atom 16번 적용**해서 누적. fp16/bf16이면 K=16이라 8번. 이게 fp32 inner-loop 시간이 2배인 원인 중 하나.

### 15.3 acc shape

`thr_mma.partition_shape_C(mma_tiler[:2])`로 thread별 acc fragment shape 얻기. 이 fragment를 `make_fragment_C`로 register fragment 만들고, TMEM 주소에 매핑하는 게 §13.4의 `tStS` 등.

---

## 16. `_setup_smem_layout` — SMEM 레이아웃 카탈로그

```python
sK_layout = sm100_utils_basic.make_smem_layout_a(
    self.tiled_mma_S,
    self.mma_tiler_kq,
    self.k_dtype,
    1,  # num_stages
)
self.sK_layout = cute.slice_(sK_layout, (None, None, None, 0))
```

`make_smem_layout_a/b`는 "이 MMA의 A/B operand가 SMEM에 어떻게 살아야 하는지"의 layout을 dtype-aware하게 만들어 줍니다. swizzle 포함된 ComposedLayout 반환.

`num_stages=1`로 만든 후 `slice_(..., (None, None, None, 0))`로 마지막 stage 차원 제거 = 1-stage layout. 우리가 직접 `1` 를 마지막에 넘긴 이유는 helper API 호환 때문.

`Q_stage=2`로 만들면:
```python
self.sQ_layout = make_smem_layout_b(tiled_mma_S, mma_tiler_kq, q_dtype, self.Q_stage)
```
이건 stage 차원이 남아있고, `[..., stage_idx]`로 stage select 가능.

### 16.1 P (TMEM)의 layout

```python
tP_layout = sm100_utils_basic.make_smem_layout_a(
    self.tiled_mma_dV,
    self.mma_tiler_pdo,
    self.do_dtype,
    1,
)
self.tP_layout = cute.slice_(tP_layout, (None, None, None, 0))
```

이름은 "smem_layout_a"지만 실제로는 **TMEM에 사는 P를 dV MMA가 어떻게 읽기 원하는지**의 layout. SMEM helper와 같은 함수를 쓰는 이유는 TMEM A operand layout 명세가 SMEM A와 비슷해서.

**중요**: `do_dtype` (= q_dtype)이 fp32면 TF32 MMA's A-operand layout, bf16이면 bf16 MMA's A-operand layout. 두 layout은 **다릅니다**. TF32는 K=8이라 column 분할이 다름.

### 16.2 dS (TMEM) layout

```python
sdSt_layout = make_smem_layout_a(self.tiled_mma_dK, self.mma_tiler_dsq, self.ds_dtype, 1)
self.sdSt_layout = cute.slice_(sdSt_layout, (None, None, None, 0))
tdS_layout = make_smem_layout_a(self.tiled_mma_dK, self.mma_tiler_dsq, self.ds_dtype, 1)
self.tdS_layout = cute.slice_(tdS_layout, (None, None, None, 0))
```

`sdSt_layout`은 SMEM에 있는 dS의 transposed view (사용처: dQ MMA가 SMEM에서 dS를 B operand로 읽음 — 사실은 sdS를 transpose해서 본 게 sdSt이지만 코드상 이름이 그렇게 박힘). `tdS_layout`은 TMEM dS layout.

`ds_dtype = q_dtype`. fp32에서 fp32 layout, bf16에서 bf16 layout.

### 16.3 dK / dV epilogue layout

```python
self.sdK_epi_tile = (
    self.tile_n,
    math.gcd(128 // (self.dk_dtype.width // 8), self.tile_hdim // 2),
)
self.sdK_layout = sm100_utils_basic.make_smem_layout_epi(
    self.dk_dtype, LayoutEnum.ROW_MAJOR, self.sdK_epi_tile, 2,
)
```

dK epilogue tile: (tile_n=128, **64 또는 32**). 두 번째 dim은 한 번에 GMEM으로 보내는 hdim의 chunk. `gcd(128/byte, hdim/2)` — 128 byte align과 half-hdim의 gcd. dk_dtype=bf16 (2 byte): gcd(64, 32)=32. dk_dtype=fp32 (4 byte): gcd(32, 32)=32.

`num_epi_stages = max(1, (tile_hdim/2) / sdK_epi_tile[1])` — chunk 몇 개로 hdim/2를 covering.

`2` (마지막 인자) = "2 wg"라는 의미 — compute warp 8개 = 2 warpgroup이라 둘이 나눠서 epilogue 처리.

---

## 17. `__call__` — host-side 준비와 kernel launch

```python
@cute.jit
def __call__(self, mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV, softmax_scale, stream=None):
    self.q_dtype = mQ.element_type
    self.k_dtype = mK.element_type
    # ... 모든 dtype 추출
    self.ds_dtype = self.q_dtype
```

각 텐서의 element_type 추출. `q_dtype`이 여기서 처음 set되어야 `_setup_attributes`가 fp32 분기 평가 가능.

### 17.1 Layout transpose

```python
mQ, mdO = [layout_utils.select(t, mode=[1, 3, 2, 0]) for t in (mQ, mdO)]
# (b, s, n, h) → (s, h, n, b)
```

PyTorch tensor가 `(B, S, N, D)` 인데 CuTe 쪽에서 처리하기 좋은 `(S, D, N, B)`로 모드 재배열. 데이터 자체는 안 옮김 — layout만 바꿈.

```python
mdO = layout_utils.select(mdO, mode=[1, 0, 2, 3])
# (s, h, n, b) → (h, s, n, b)
```

dO는 dV MMA의 B operand로 transpose된 view를 써야 해서 한 번 더.

### 17.2 _setup_attributes / _get_tiled_mma / _setup_smem_layout 순서대로 호출

여기서 dtype-dependent 모든 layout과 stage 수가 결정.

### 17.3 TMA atom 생성

```python
tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
    cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
    mK,
    cute.select(self.sK_layout, mode=[0, 1, 2]),
    self.mma_tiler_kq,
    self.tiled_mma_S,
    self.cluster_layout_vmnk.shape,
)
```

각 텐서마다 TMA descriptor 생성. `make_tiled_tma_atom_A/B`는 MMA의 A/B operand 위치에 맞춰 TMA를 set up.

### 17.4 SharedStorage 정의

```python
@cute.struct
class SharedStorage:
    Q_mbar_ptr: cute.struct.MemRange[Int64, 2 * self.Q_stage]
    # ... 8개 mbar
    sQ: cute.struct.Align[
        cute.struct.MemRange[cute.Uint8, sQ_alloc_bytes],  # raw bytes (sQ는 sdK와 reuse)
        self.buffer_align_bytes,
    ]
    # ... 다른 SMEM tensors
```

이 struct 인스턴스가 SMEM 전체에 한 번 alloc됨 (`smem.allocate(self.shared_storage)`). 각 필드의 offset은 컴파일 타임 결정.

`sQ_alloc_bytes = max(sQ_bytes, sdK_bytes)` — sQ는 main loop 끝나면 dK epilogue용 sdK로 재사용되므로 둘 중 큰 것으로 alloc.

### 17.5 kernel launch

```python
self.kernel(...).launch(
    grid=grid_dim,
    block=[self.threads_per_cta, 1, 1],
    cluster=None,
    smem=self.shared_storage.size_in_bytes(),
    stream=stream,
    min_blocks_per_mp=1,
)
```

`grid_dim`은 `SingleTileScheduler.get_grid_shape(...)` = `(num_n_blocks, num_heads, num_batches)`. 각 CTA는 1개 (n_block, head, batch) tile 처리.

---

## 18. `kernel` — warp specialization과 pipeline 생성

```python
@cute.kernel
def kernel(self, ...):
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
```

각 thread가 자기 warp index 알기. `make_warp_uniform`은 warp 안 모든 lane이 같은 값 갖도록 보장 (보통 lane 0이 결정).

### 18.1 SMEM/TMEM allocator

```python
smem = cutlass.utils.SmemAllocator()
storage = smem.allocate(self.shared_storage)

tmem = cutlass.utils.TmemAllocator(
    storage.tmem_holding_buf,
    barrier_for_retrieve=tmem_alloc_barrier,
    allocator_warp_id=self.mma_warp_id,  # warp 12가 alloc 담당
    is_two_cta=False,
    two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
)
```

TMEM은 dynamic alloc. `tmem.allocate(num_cols)`는 mma warp가 호출, 결과는 mbarrier로 다른 warp들에 전파.

### 18.2 9개 pipeline 생성

순서대로:

1. **`pipeline_S_P`** (UmmaAsync): MMA producer (S MMA 끝), Compute consumer (S 읽고 P 쓰기). num_stages=1.
2. **`pipeline_dP`** (UmmaAsync): MMA producer (dP MMA 끝), Compute consumer.
3. **`pipeline_dKV`** (UmmaAsync): MMA producer (dV/dK MMA 끝), Compute consumer (epilogue). num_stages=2.
4. **`pipeline_dQ`** (UmmaAsync): MMA producer (dQ MMA 끝), Reduce consumer.
5. **`pipeline_dS`** (AsyncUmma): Compute producer (dS 다 적음), MMA consumer (dK/dQ MMA의 A operand로 사용).
6. **`pipeline_LSE`** (TmaAsync): Load producer (LSE TMA), Compute consumer.
7. **`pipeline_dPsum`** (TmaAsync): Load producer (dPsum TMA), Compute consumer.
8. **`pipeline_Q`** (TmaUmma): Load producer (Q TMA), MMA consumer.
9. **`pipeline_dO`** (TmaUmma): Load producer (dO TMA), MMA consumer.

각 pipeline은 SharedStorage의 mbar slot 차지. `barrier_storage=storage.X_mbar_ptr.data_ptr()`.

### 18.3 SMEM tensor 가져오기

```python
sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner, dtype=self.q_dtype)
sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
# ... 등등

# Transposed views (같은 메모리, 다른 layout)
sQt = storage.sQ.get_tensor(sQt_layout.outer, swizzle=sQt_layout.inner, dtype=self.q_dtype)
sKt = storage.sK.get_tensor(sKt_layout.outer, swizzle=sKt_layout.inner, dtype=self.k_dtype)
```

`sQ` vs `sQt`는 메모리 동일, layout만 다름. dK MMA의 B operand로는 sQt(transposed) 필요.

### 18.4 TMEM tensor 가져오기

```python
tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
# offset 0의 fake pointer. 실제 alloc은 mma warp가 함.

thr_mma_S = tiled_mma_S.get_slice(mma_tile_coord_v)
Sacc_shape = thr_mma_S.partition_shape_C(self.mma_tiler_kq[:2])
tStS = thr_mma_S.make_fragment_C(Sacc_shape)
tStS = cute.make_tensor(tmem_ptr + self.tmem_S_offset, tStS.layout)
```

S acc fragment 만들고 TMEM 주소에 매핑. 이 텐서는 모든 warp가 같이 봄 (TMEM은 SM-wide).

`tP`는 tmem_P_offset(=0, S와 같음) 위치를 do_dtype으로 recast한 view:

```python
tP = cute.make_tensor(
    cute.recast_ptr(tmem_ptr + self.tmem_P_offset, dtype=self.do_dtype),
    tP_layout.outer,
)
```

### 18.5 Warp specialization

```python
if warp_idx == self.empty_warp_id or warp_idx == 14:
    cute.arch.setmaxregister_decrease(self.num_regs_empty)

if warp_idx == self.load_warp_id:
    cute.arch.setmaxregister_decrease(self.num_regs_load)
    self.load(...)

if warp_idx == self.mma_warp_id:
    cute.arch.setmaxregister_decrease(self.num_regs_mma)
    tmem.allocate(self.tmem_alloc_cols)   # 여기서만 TMEM alloc
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(Float32)
    self.mma(...)
    tmem.relinquish_alloc_permit()
    tmem_alloc_barrier.arrive_and_wait()
    tmem.free(tmem_ptr)

if warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
    cute.arch.setmaxregister_increase(self.num_regs_compute)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(Float32)
    self.compute_loop(...)
    tmem_alloc_barrier.arrive()

if warp_idx >= self.reduce_warp_ids[0] and warp_idx <= self.reduce_warp_ids[-1]:
    cute.arch.setmaxregister_increase(self.num_regs_reduce)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(Float32)
    self.dQacc_reduce(...)
    tmem_alloc_barrier.arrive()
```

각 분기는 자기 역할 코드 실행. 다른 warp는 그 분기 안 들어감 (각 warp의 PC가 다른 곳).

`tmem_alloc_barrier.arrive_and_wait`는 모든 사용자(mma, compute×2, reduce)가 TMEM 다 썼다는 핸드셰이크. mma warp만 free 가능.

---

## 19. `load` — TMA producer warp

13번 warp 한 개. 1 warp = 32 lanes지만 TMA는 lane 0만 발사하면 충분.

### 19.1 Producer state 초기화

```python
producer_state_Q_LSE = make_pipeline_state(PipelineUserType.Producer, self.Q_stage)
producer_state_dO_dPsum = make_pipeline_state(PipelineUserType.Producer, self.dO_stage)
```

Q와 LSE는 같은 stage 페이스 (Q load 직후 LSE load), dO와 dPsum도 같이.

### 19.2 work tile loop

```python
tile_scheduler = TileSchedulerCls()
work_tile = tile_scheduler.initial_work_tile_info()
while work_tile.is_valid_tile:
    n_block, head_idx, batch_idx, _ = work_tile.tile_idx
    seqlen = SeqlenInfoCls(batch_idx)
    m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
    # ... TMA load 코드
    tile_scheduler.advance_to_next_work()
    work_tile = tile_scheduler.get_current_work()
```

각 (n_block, head, batch) tile 단위. SingleTileScheduler에서는 한 번만 돌고 끝 (CTA 1개당 tile 1개).

### 19.3 GMEM 텐서 슬라이싱

```python
mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
# (s, h, n, b) → (s, h)에서 head/batch 고정
```

`seqlen.offset_batch_Q`는 batch 차원 슬라이스 (varlen 지원하던 흔적 — 지금은 단순 dense indexing).

### 19.4 TMA copy fn

```python
load_K, _, _ = copy_utils.tma_get_copy_fn(
    tma_atom_K,
    block_in_cluster_coord_vmnk[2],
    a_cta_layout,
    tSgK,    # GMEM source partition
    sK,      # SMEM destination
    single_stage=True,
)
```

`tma_get_copy_fn`은 closure 반환 — 호출하면 TMA copy 발사.

### 19.5 Prologue + main loop

```python
# Prologue: 첫 m_block 미리 로드
pipeline_Q.producer_acquire(producer_state_Q_LSE, extra_tx_count=tma_copy_bytes["K"])
load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_LSE))
load_Q(first_m_block, producer_state=producer_state_Q_LSE)
pipeline_Q.producer_commit(producer_state_Q_LSE)

# Main loop: 다음 m_blocks
for m_block in range(m_block_min + 1, m_block_max):
    pipeline_Q.producer_acquire(producer_state_Q_LSE)
    load_Q(m_block, producer_state=producer_state_Q_LSE)
    pipeline_Q.producer_commit(producer_state_Q_LSE)
    # ... LSE, dO, dPsum도 로드
```

K는 한 번만 (single_stage=True), Q는 매 m_block, V도 한 번만, dO는 매 m_block. LSE/dPsum은 row 통계라 매 m_block.

`producer_acquire(extra_tx_count=...)`는 mbarrier에 expected byte count 추가 — TMA가 보낼 byte 수 미리 알리기. mbarrier는 모든 byte 도착 시 자동 trigger.

### 19.6 Tail

```python
pipeline_Q.producer_tail(producer_state_Q_LSE.clone())
pipeline_LSE.producer_tail(producer_state_Q_LSE)
```

pipeline 마지막 정리.

---

## 20. `mma` — UMMA producer warp

12번 warp 한 개. 5개 GEMM을 m_block마다 발사.

### 20.1 fragment 만들기

```python
tSrK = tiled_mma_S.make_fragment_A(sK)   # SMEM K → A operand fragment
tSrQ = tiled_mma_S.make_fragment_B(sQ)
# ...
tdKrdS = tiled_mma_dK.make_fragment_A(tdS)  # **TMEM** dS → A operand fragment
tdVrP = tiled_mma_dV.make_fragment_A(tP)    # **TMEM** P → A operand fragment
```

`make_fragment_A(smem)`은 SMEM 텐서를 받아 "A operand로 쓸 형식의 register fragment" 만듦. 실제 데이터는 MMA 시점에 SMEM에서 읽힘.

### 20.2 mma 함수 partial 정의

```python
mma_qk_fn = partial(
    gemm_ptx_w_idx,
    tiled_mma_S, tStS, tSrK, tSrQ,
    sA=sK, sB=sQ,
    zero_init=True, cta_group=1,
)
# 호출 시: mma_qk_fn(B_idx=handle_Q.index)
```

`gemm_ptx_w_idx`는 inline-PTX로 UMMA 발사. `B_idx`는 Q의 stage index.

### 20.3 prologue + main loop

```python
# Prologue: S, dP, dV (첫 m_block)
handle_Q = pipeline_Q_consumer.wait_and_advance()
pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
mma_qk_fn(B_idx=handle_Q.index)        # S = K @ Q.T
pipeline_S_P.sync_object_full.arrive(0, ...)

pipeline_dO.consumer_wait(consumer_state_dO)
pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)
pipeline_dQ.sync_object_empty.wait(0, producer_phase_acc)
mma_dov_fn(B_idx=consumer_state_dO.index)  # dP = V @ dO.T
pipeline_dP.sync_object_full.arrive(0, ...)

producer_phase_acc ^= 1
pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=True)  # dV = P.T @ dO
pipeline_dO.consumer_release(consumer_state_dO)

# Main loop: S, dK, dQ, dP, dV (각 m_block)
for _ in range(main_loop_iters):
    handle_Q_next = pipeline_Q_consumer.wait_and_advance()
    mma_qk_fn(B_idx=handle_Q_next.index)
    # ...
    pipeline_dS.consumer_wait(consumer_state_dS)
    mma_dsq_fn(...)            # dK += dS.T @ Q
    mma_dsk_fn()               # dQ = dS @ K
    pipeline_dQ.sync_object_full.arrive(...)
    # ...
    mma_dov_fn(...)            # dP = V @ dO.T
    mma_pdo_fn(...)            # dV += P.T @ dO
```

각 MMA는 비동기 발사 후 mbarrier로 완료 신호. `producer_phase_acc`는 acc TMEM이 비어있는지 추적 (pingpong).

### 20.4 Tail

```python
# 마지막 dK, dQ
mma_dsq_fn(...)
pipeline_dKV.sync_object_full.arrive(1, ...)  # dK ready
mma_dsk_fn()
pipeline_dQ.sync_object_full.arrive(0, ...)
```

---

## 21. `compute_loop` — softmax + dS 계산 (8 warp)

이게 가장 복잡한 함수. 8 warps × 32 lane = 256 thread.

### 21.1 sLSE/sdPsum view 변환

```python
sLSE_2D = cute.make_tensor(sLSE.iterator, cute.make_layout((tile_m, tile_n, Q_stage), stride=(1, 0, ...)))
sLSE_2D = layout_utils.transpose_view(sLSE_2D)
```

sLSE는 (tile_m,) per row. 행 마스크 + 열 broadcast로 (tile_m, tile_n) 모양으로 view (stride에 0 끼움). 그리고 transpose해서 (tile_n, tile_m).

### 21.2 thread idx와 wg idx

```python
tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * 8)  # 0..255
dp_idx = tidx % 128                                            # 0..127 (per WG)
num_wg = 2  # 8 warps = 2 warpgroup
```

8 compute warp = 2 warpgroup. 같은 wg 내 thread끼리는 같은 dp_idx 공유 (e.g. wg0 lane 0 = wg1 lane 128 → dp_idx 0).

### 21.3 P/dS TMEM region 정의 (CRITICAL)

```python
tileP_f32_like = self.cta_tiler[1] // 32 * self.v_dtype.width
# bf16: 128 // 32 * 16 = 64
# fp32: 128 // 32 * 32 = 128

tStP = cute.composition(tStS, (cute.make_layout((tile_n, tileP_f32_like)), 1, 1))
tStP = cute.make_tensor(tStS.iterator, tStP.layout)
tScS = thr_mma_S.partition_C(cute.make_identity_tensor(self.mma_tiler_kq[:2]))
tScP = cute.composition(tScS, (cute.make_layout((tile_n, tileP_f32_like)), 1, 1))

tdPtdS = cute.composition(tdPtdP, (cute.make_layout((tile_n, tileP_f32_like)), 1, 1))
tdPcdP = thr_mma_dP.partition_C(cute.make_identity_tensor(self.mma_tiler_vdo[:2]))
tdPcdS = cute.composition(tdPcdP, (cute.make_layout((tile_n, tileP_f32_like)), 1, 1))
```

핵심: P는 S TMEM 영역의 첫 `tileP_f32_like` cols를 빌려 씀. bf16에서는 128 fp32 cols 중 64만 (P가 packed-bf16이라 절반만 필요), fp32에서는 128 모두.

이 식이 fp32에서도 산식적으로는 맞지만, **§24에서 자세히 설명할 atom mismatch 문제** 때문에 단순 fp32 분기로 안 됨.

### 21.4 TMEM atom

```python
tmem_load_atom = cute.make_copy_atom(
    tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32
)
tmem_store_atom = cute.make_copy_atom(
    tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32
)
```

- **load atom**: stage당 32 fp32 word 로드. S 전체 (128 cols)를 4 stages로 나눠 읽음.
- **store atom**: stage당 16 fp32 word 저장. tileP_f32_like cols를 (tileP_f32_like / 16) stages로 나눠 적기. bf16에서 64/16=4 stages.

`Repetition(N)` = "이 atom 한 호출이 N개 fp32 word 처리". make_tmem_copy 머신이 이걸 보고 thread × stage 분배 결정.

### 21.5 partition

```python
thr_copy_t2r = copy_utils.make_tmem_copy(tmem_load_atom, num_wg).get_slice(tidx)
tStS_t2r = thr_copy_t2r.partition_S(tStS)  # ((32, 32), 1, 1) shape comment
tScS_t2r = thr_copy_t2r.partition_D(tScS)  # ((32, 1), 2, 1, 1)

thr_copy_r2t = copy_utils.make_tmem_copy(tmem_store_atom, num_wg).get_slice(tidx)
tScP_r2t = thr_copy_r2t.partition_S(tScP)
tStP_r2t = thr_copy_r2t.partition_D(tStP)
```

shape의 두 번째 mode (`(32, 1)` 안의 1)이 "atom call repetition 수"이고 세 번째 mode가 "stage 수". 이 커널의 t2r 분포에서는 `num_stages = cute.size(tScS_t2r, mode=[1])`로 가져옴.

### 21.6 mainloop

```python
for iter_idx in range(loop_count):
    m_block = m_block_min + iter_idx

    # 1. LSE load (TMA로 SMEM에)
    pipeline_LSE.consumer_wait(consumer_state_LSE)

    # 2. S TMEM 읽어오기 (UMMA 끝남 대기)
    pipeline_S_P.consumer_wait(consumer_state_S_P_dP)
    tSrS_t2r = cute.make_fragment(tScS_t2r.shape, Float32)
    cute.copy(thr_copy_t2r, tStS_t2r, tSrS_t2r)

    # 3. mask 적용 (seqlen 끝 패딩)
    mask_fn(tSrS_t2r, m_block=m_block)

    num_stages = cute.size(tScS_t2r, mode=[1])  # 4 for tile_n=128

    # 4. P = exp2(S * scale_log2 - LSE * log2_e)
    tSrP_r2t_f32 = cute.make_fragment(tScP_r2t.shape, Float32)
    tSrP_r2t = cute.recast_tensor(tSrP_r2t_f32, self.q_dtype)
    for stage in range(num_stages):
        tSrS_cur = tSrS_t2r[None, stage, 0, 0]
        # LSE 빼기 + exp2
        cute.autovec_copy(tSsLSE[..., stage, ..., consumer_state_LSE.index], tSrLSE_s2r)
        for v in range(...):
            tSrS_cur[2v], tSrS_cur[2v+1] = fma_packed_f32x2(...)
            tSrS_cur[2v]   = exp2(tSrS_cur[2v])
            tSrS_cur[2v+1] = exp2(tSrS_cur[2v+1])
        utils.cvt_f16(tSrS_cur, tSrP_r2t[None, stage, 0, 0])  # ★ fp32 깨지는 라인
        cute.copy(thr_copy_r2t, tSrP_r2t_f32[None, stage, ...], tStP_r2t[None, stage, ...])

    # 5. P TMEM에 다 적었음 → MMA에 신호 (dV MMA 시작 가능)
    pipeline_S_P.consumer_release(consumer_state_S_P_dP)
    pipeline_LSE.consumer_release(consumer_state_LSE)

    # 6. dP 받아와서 dS = P * (dP - D) 계산
    pipeline_dPsum.consumer_wait(consumer_state_dPsum)
    pipeline_dP.consumer_wait(consumer_state_S_P_dP)
    for stage in range(num_stages):
        tdPrdP_t2r = cute.make_fragment(...)
        cute.copy(thr_copy_t2r, tdPtdP_t2r[..., stage, ...], tdPrdP_t2r)
        for v in range(...):
            # dP - dPsum (= D)
            tdPrdP_cur[2v], tdPrdP_cur[2v+1] = sub_packed_f32x2(...)
            # P * (dP - D)
            tdPrdP_cur[2v], tdPrdP_cur[2v+1] = mul_packed_f32x2(...)
        # dS를 narrow dtype으로 변환 후 TMEM/SMEM에 적기
        tdPrdS_cvt = make_fragment_like(tdPrdP_cur, self.ds_dtype)
        utils.cvt_f16(tdPrdP_cur, tdPrdS_cvt)  # ★ fp32 깨지는 라인 #2
        if stage == 0:
            pipeline_dS.producer_acquire(producer_state_dS)
        # RMEM → TMEM
        tdPrdS_r2t_f32 = cute.recast_tensor(tdPrdS_cvt, Float32)
        cute.copy(thr_copy_r2t, tdPrdS_r2t_f32, tdPtdS_r2t[None, stage, 0, 0])
        # RMEM → SMEM (for dQ MMA)
        cute.autovec_copy(tdPrdS_cvt, tRS_sdS[None, stage])

    pipeline_dPsum.consumer_release(consumer_state_dPsum)
    pipeline_dS.producer_commit(producer_state_dS)
    consumer_state_S_P_dP.advance()

# Epilogue (loop 끝나고 dV/dK 출력)
self.epilogue_dK_or_dV_tma(..., "V")
self.epilogue_dK_or_dV_tma(..., "K")
```

### 21.7 핵심 포인트

- `tStS_t2r → tSrS_t2r` (TMEM read): S를 register로
- `tSrS_cur stage by stage`: stage별 element 처리 (LSE 빼기, exp2)
- `cvt_f16`: Float32 → narrow dtype (fp32에서 깨짐)
- `tStP_r2t` (RMEM → TMEM): 변환된 P를 다시 TMEM에 (S 영역 overlap)
- 모든 동기화는 mbarrier 기반

dS도 같은 패턴: dP TMEM read → register 계산 → narrow dtype convert → TMEM(dS) + SMEM(dS) 둘 다 적기.

**왜 SMEM에도 적냐**: dQ MMA의 A operand가 SMEM에서 읽음 (TMEM 아님). dK MMA는 TMEM에서 읽음.

---

## 22. `dQacc_reduce` — dQ 누적 epilogue (4 warp)

dQ accumulator가 TMEM에 fp32로 살아 있고, 이걸 GMEM의 `dq_accum` 텐서에 atomic-add. 4 reduce warps = 128 thread가 협력.

### 22.1 fp32 분기 (이미 구현됨)

```python
fp32_dQ = self.q_dtype is Float32
if const_expr(fp32_dQ):
    tdQtdQ_chunks = cute.logical_divide(
        tdQtdQ, cute.make_layout((self.tile_m, self.dQ_reduce_ncol)),
    )
    tmem_load_atom = sm100_utils_basic.get_tmem_load_op(
        self.mma_tiler_dsk, LayoutEnum.ROW_MAJOR,
        Float32, Float32, (self.tile_m, self.dQ_reduce_ncol),
        use_2cta_instrs=False,
    )
    tiled_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ_chunks[(None, None), 0])
    thr_copy_t2r = tiled_t2r.get_slice(tidx)
    tdQtdQ_t2r = thr_copy_t2r.partition_S(tdQtdQ_chunks[(None, None), None])
else:
    # 기존 fp16/bf16 path
    tmem_load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dQ_reduce_ncol_t2r)), Float32
    )
    thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ).get_slice(tidx)
    tdQtdQ_t2r = thr_copy_t2r.partition_S(tdQtdQ)
```

이 분기가 fp32 dQ에서 핵심. **이게 잘 동작한다는 사실이 fp32 P/dS 구현 시 참고 패턴**입니다.

### 22.2 main loop

```python
for iter_idx in range(loop_count):
    pipeline_dQ.consumer_wait(dQ_consumer_state)

    # TMEM → RMEM
    tdQrdQ_t2r = cute.make_fragment(tdQrdQ_t2r_shape, Float32)
    if fp32_dQ:
        for chunk_idx in range(num_chunks):
            cute.copy(thr_copy_t2r, tdQtdQ_t2r[..., chunk_idx], tdQrdQ_t2r[..., chunk_idx])
    else:
        cute.copy(thr_copy_t2r, tdQtdQ_t2r, tdQrdQ_t2r)

    pipeline_dQ.consumer_release(dQ_consumer_state)

    # RMEM → SMEM (chunked)
    for stage in range(...):
        cute.copy(thr_copy_dQaccum_r2s, tdQrdQ[..., stage], tdQsdQ[..., smem_idx])
        # SMEM → GMEM with atomic add
        copy_utils.cpasync_reduce_bulk_add_f32(
            sdQaccum[..., smem_idx].iterator,
            gdQaccum_cur[..., stage].iterator,
            self.tma_copy_bytes["dQ"],
        )
```

`cpasync_reduce_bulk_add_f32`가 핵심 — TMA reduce-add. SMEM에서 GMEM으로 fp32 atomic add bulk transfer. `dq_accum`은 multiple compute warp/CTA에서 동시에 누적되니까.

후속 postprocess kernel이 이 dq_accum을 다시 읽어 softmax_scale 곱하고 dq dtype으로 변환해서 dq에 적습니다.

---

## 23. `epilogue_dK_or_dV_tma` — dK/dV 출력

compute warp 8개가 wg별로 분할 (wg0: dK 전반부, wg1: dK 후반부 / 또는 V도 마찬가지).

### 23.1 wg split

```python
wg_idx = (tidx_within_compute) // 128
sdKV = sdKV[None, None, wg_idx]   # 한 wg가 자기 절반만
gdKV = self.split_wg(gdKV_p, wg_idx, num_wg)
```

`split_wg`는 hdim 축을 num_wg(=2)로 나눠 wg_idx번째를 select.

### 23.2 epi_stages 루프

```python
for epi_stage in range(num_epi_stages):
    # TMEM → RMEM
    tdKVrdKV_t2r = cute.make_fragment(...)
    cute.copy(thr_copy_t2r, tdKVtdKV_t2r, tdKVrdKV_t2r)

    # scale (dK only) + dtype convert
    if scale is not None:
        for i in range(...):
            tdKVrdKV_t2r[2i:2i+2] = mul_packed_f32x2(..., (scale, scale))
    tdKVrdKV = make_fragment(..., dtype)
    tdKVrdKV.store(tdKVrdKV_t2r.load().to(dtype))

    # RMEM → SMEM
    cute.copy(thr_copy_r2s_dKV, tdKVrdKV_r2s, tdKVsdKV_r2s)

    # SMEM → GMEM (TMA)
    if leader_warp:
        cute.copy(tma_atom_dKV, tdKVsdKV, tdKVgdKV[None, epi_stage])
```

dK는 softmax_scale 곱하기, dV는 안 곱함 (`scale=None`).

`num_epi_stages = max(1, (tile_hdim/2) / sdK_epi_tile[1])` — hdim이 sdK_epi_tile의 두 번째 dim보다 크면 stage 여러 번.

### 23.3 fp32 영향

dK/dV postprocess는 dtype-aware (`tdKVrdKV.store(... .to(dtype))` — to() 변환이 dtype에 맞춰 동작). 그래서 **이 부분은 fp32에서 자동으로 동작**합니다 (테스트로 확인됨 — fp32 fwd/bwd 시도 시 이 단계는 통과했음).

---

## 24. P / dS TMEM overlap trick — 왜 fp32에서 깨지는가

이게 fp32 bwd의 핵심 난제. 차근차근.

### 24.1 fp16/bf16 packing trick (현재 동작 중)

S MMA 결과는 항상 Float32 acc — TMEM에 (tile_n=128, tile_m=128) = 128 fp32 cols 차지.

S를 다 쓴 후, **같은 TMEM 영역에 P를 적습니다**. P는 fp16/bf16 (= q_dtype). TMEM은 32-bit word 기반이므로 한 word에 **2개의 bf16 element를 packing**해서 적음. 그래서 P가 TMEM에서 차지하는 cols 수는 S의 **절반** (64 cols).

- **t2r (S 읽기)**: `Ld32x32bOp(Repetition=32)` atom. atom 한 호출 = 32 fp32 word 이동. 128 cols / 32 cols/stage = **4 stages**.
- **r2t (P 쓰기)**: `St32x32bOp(Repetition=16)` atom. atom 한 호출 = 16 fp32 word = 32 bf16 element 이동 (packed). 64 cols / 16 cols/stage = **4 stages**. Stage 수 일치!

코드에서:
```python
tSrP_r2t_f32 = cute.make_fragment(tScP_r2t.shape, Float32)  # 16 fp32 per stage
tSrP_r2t = cute.recast_tensor(tSrP_r2t_f32, self.q_dtype)   # 32 bf16 per stage (recast)
# tSrS_cur per stage = 32 fp32 (from t2r)
utils.cvt_f16(tSrS_cur, tSrP_r2t[None, stage, 0, 0])
# Float32 32 elements → BFloat16 32 elements (size 일치, 변환만)
```

### 24.2 fp32에서는?

**P가 fp32라 packing 안 됨**. 한 word에 1 element. 그래서 **P가 TMEM에서 차지하는 cols 수도 S와 동일** (128 cols).

이론적으로 이렇게 되어야 함:
- t2r 같음: 32 cols/stage × 4 stages
- r2t: **32 cols/stage × 4 stages** (packing 없으니 stage당 32 fp32 word)

하지만 현재 코드의 store atom은 `Repetition=16` 고정. fp32에서 store atom을 그대로 쓰면:
- r2t: 16 cols/stage × **8 stages** (128 / 16 = 8)
- t2r은 4 stages, r2t는 8 stages → **mismatch**

내가 시도했던 fix: store atom을 fp32에서 `Repetition=32`로 바꾸기. 그러면:
- r2t: 32 cols/stage × 4 stages → t2r과 일치

하지만 **`make_tmem_copy(atom, num_wg)` 내부에서 `tiler_mn = (128 * num_rep * num_wg / 32, 32)` 계산**:
- bf16 (Rep=16, num_wg=2): tiler = (128, 32). tStP shape (128, 64). partition fit OK.
- fp32 (Rep=32, num_wg=2): tiler = (256, 32). tStP shape (128, 128). **256 row > 128 row** → 어딘가 overflow → illegal address.

### 24.3 더 깊은 원인: MMA A-operand layout 의존성

내가 분석한 결론은 **store atom을 단순히 Repetition만 바꾸는 것으로는 부족**. dV MMA가 P를 TMEM에서 읽을 때 **TF32 MMA는 fp16/bf16 MMA와 다른 layout으로 해석**합니다.

비유: bf16 MMA는 P가 TMEM에 "(M, K) = (128, 128) bf16, K=16 stride"로 살아있다고 가정하고 읽음. TF32 MMA는 같은 P를 "(M, K) = (128, 128) fp32, K=8 stride"로 살아있다고 가정. 같은 (M, K)지만 stride가 다름.

따라서 fp32 P를 TMEM에 적을 때는, dV MMA가 읽기 원하는 layout으로 적어야 함. fwd kernel은 이걸 `sm100_utils_basic.make_smem_layout_a(tiled_mma_pv, mma_tiler_pv, q_dtype, s_stage)`로 dtype-aware하게 만듦. bwd의 P 영역도 이 helper로 layout을 받아오는 게 정공법.

### 24.4 dS도 동일

dS도 P와 같은 패턴 — dP acc TMEM에 dS를 packing해서 적음. fp32에서는 dK MMA의 A operand layout에 맞춰 적어야 함.

### 24.5 정공법 시작점

1. `tP_layout = sm100_utils_basic.make_smem_layout_a(tiled_mma_dV, mma_tiler_pdo, do_dtype, 1)` — 이미 정의됨 ([_setup_smem_layout](kernels/flash_bwd_sm100.py)). 이건 SMEM helper지만 TMEM A operand에 그대로 사용 가능.
2. `compute_loop` 안에서 `tStP`를 `composition(tStS, ...)`로 손코딩하지 말고 `cute.make_tensor(tmem_ptr + tmem_P_offset, tP_layout.outer)` 같이 **MMA helper에서 받은 layout 사용**.
3. 그 layout에 맞춰 r2t copy atom + thread mapping을 새로 구성. fwd의 P-storage 코드가 좋은 참고.

이건 단순 patch가 아니라 P/dS의 TMEM 표현을 dtype-aware하게 재작성하는 작업입니다. 1-2일 분량.

---

## 25. fp32 bwd 구현 시작점 — 무엇을 어디서 손대야 하나

### 25.1 핵심 작업

**A. `compute_loop`의 P 저장 path 재작성** ([flash_bwd_sm100.py L1582-L1714](kernels/flash_bwd_sm100.py#L1582-L1714))
- `tileP_f32_like`, `tStP`, `tScP` 손코딩 제거
- `tP_layout`(이미 `__init__`에 정의)에서 TMEM P-side layout 사용
- store atom을 `tiled_mma_dV`의 A operand에 맞춰 (fp32일 때 다른 atom 필요)
- `cvt_f16` 호출은 fp32에서 분기로 직접 store

**B. dS 저장 path 동일 작업** ([flash_bwd_sm100.py L1736-L1779](kernels/flash_bwd_sm100.py#L1736-L1779))
- `tdPtdS`, `tdPcdS` 손코딩 제거
- `tdS_layout`에서 TMEM dS-side layout 사용
- 마찬가지로 store atom + cvt 분기

**C. SMEM dS 경로 (`tRS_sdS`) 검증**
- `sdS_layout`, `sdSt_layout`은 `ds_dtype`에 의존하므로 fp32에서 자동으로 2배 폭. 합계가 SMEM 한도 안에 들어가는지 `_setup_smem_layout`에서 확인.

**D. dV / dK MMA 호출이 fp32 P/dS를 잘 읽는지 확인**
- `tdVrP = tiled_mma_dV.make_fragment_A(tP)`에서 `tP`가 올바른 fp32 layout이면 자동으로 동작해야 함.

### 25.2 디버깅 방법

1. **최소 케이스**: `B=1, H=1, L=128, D=32, dtype=Float32, requires_grad=True`. fwd는 통과 확인 (이미 동작).
2. **`CUDA_LAUNCH_BLOCKING=1`로 실행**해서 정확한 illegal address 위치 좁히기.
3. **`cute.printf`로 thread 0의 partition shape 출력**:
   ```python
   if tidx == 0:
       cute.printf("tStP_r2t shape = %d, %d\n", cute.size(tStP_r2t.shape[0]), cute.size(tStP_r2t.shape[1]))
   ```
4. **bf16과 fp32에서 같은 print 비교**해서 어디서 차이 나는지.
5. **fwd의 P-store 코드와 직접 diff** ([flash_fwd_sm100.py](kernels/flash_fwd_sm100.py)) — fwd는 fp32에서 잘 동작하니까 바로 참고.

### 25.3 정확도 검증

bwd 동작하면 `bwd/test_bwd.py`로 정확도 비교. 초기 atol은 `1e-2` 정도 느슨하게, 점진적으로 좁히기. TF32는 수치 정확도가 fp32보다 낮아서 (`atol_bwd=2e-3` 이내가 목표).

### 25.4 SMEM 한도 확인

fp32 시 `sQ`, `sK`, `sV`, `sdO`, `sdS`, `sdK`, `sdV` 합계가 228 KB 안에 들어가는지:
- bf16: ~120 KB (대략)
- fp32: ~240 KB (2배) → **한도 초과 가능**

대응: `Q_stage`를 1로 줄이거나, tile 더 작게.

interface.py에서 fp32 시 `m_block_size=64` 강제하므로 이미 일부 완화됐지만, 추가 조정 필요할 수 있음.

### 25.5 참고할 코드

| 보고 싶은 것 | 어디 |
|------------|-----|
| fp32 P를 TMEM에 적는 방법 (fwd) | [flash_fwd_sm100.py](kernels/flash_fwd_sm100.py)의 `softmax_step`, P-store 부분 |
| MMA helper로 TMEM A layout 받기 | [flash_fwd_sm100.py](kernels/flash_fwd_sm100.py)의 `tP_layout` 생성부 |
| fp32 dQ acc TMEM → RMEM (chunked) | 본 파일의 `dQacc_reduce` (이미 fp32 지원) |
| make_tmem_copy 내부 동작 | [core/copy_utils.py:52-62](core/copy_utils.py#L52-L62) |
| Repetition / atom shape 매뉴얼 | NVIDIA CUTLASS Blackwell docs ([github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)) |

---

## 마치며

이 커널은 한 마디로 "Hopper 코드를 SM100으로 옮긴 것"이고, **TMEM 도입과 UMMA의 TF32 K=8 atom**이 fp32 bwd 구현의 두 핵심 난제입니다.

읽다가 막히는 부분이 있으면:
- §1-§10이 일반 CuTeDSL 기초. 이걸 모르면 그 위는 다 안 읽힘.
- §11이 SM100 핵심 (TMEM, UMMA).
- §12가 알고리즘 ↔ 코드 매핑.
- §13-§23이 함수 단위 walkthrough.
- §24가 fp32의 핵심 문제.
- §25가 어디서부터 손대야 하는지.

집에서 차분히 §1부터 정독하고, 막히는 부분만 추출해서 다음 세션에서 같이 보면 fp32 구현 들어갈 수 있을 거예요.
