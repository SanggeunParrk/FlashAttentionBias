# Follow-up: fwd bias 병목 + bwd fp32 fix paths

선행 분석 docs 를 review 하고, **(a) fp32 bias forward 의 SMEM read 80% 비용 원인 가설**과
**(b) fp32 backward 의 dP TMEM read 버그 (Bug #2) 의 root-cause 가설 및 fix 경로**를 한 곳에
정리한 문서.

선행 docs:
- [BIAS_FWD_BOTTLENECK_DECOMPOSITION.md](./BIAS_FWD_BOTTLENECK_DECOMPOSITION.md)
- [BIAS_FWD_OPTIMIZATION_PLAN.md](./BIAS_FWD_OPTIMIZATION_PLAN.md)
- [FP32_BWD_DIAGNOSIS.md](./FP32_BWD_DIAGNOSIS.md)
- [TMEM_INIT_BIAS_POSTMORTEM.md](./TMEM_INIT_BIAS_POSTMORTEM.md)

## TL;DR

- **fwd bias**: 측정으로 확정된 80% SMEM read 비용의 *그 다음 단계*. `sBias_layout` 이
  swizzle 없는 plain row-major fp32 stride 64 라는 layout 문제가 가장 유력한 후보.
  K/V 는 같은 커널에서 swizzle 되는데 bias 만 raw 인 점, bf16 isolation 에서 30% 회복되는
  결과와 정합. 단 `Ld32x32bOp(Repetition(32))` 의 정확한 lane↔element 분포를 로컬에서
  확정 못 해서 "32-way conflict deterministic" 이라는 단정까지는 못 감. ncu 1 회로 결판 가능.
- **bwd fp32**: Bug #1 (dQacc_reduce reshape over-stride) 은 fix 됨. Bug #2 (dP TMEM read
  garbage) 의 가장 유력한 가설은 *"TF32 dP MMA 의 C-fragment TMEM cell 매핑이
  `make_fragment_C` 가 보고하는 layout 과 다르다"*. Bug #1 과 같은 뿌리 (TF32 fragment
  packing 이 f16 과 다름) 의 TMEM 버전. 진단 doc 이 빠뜨린 cheap한 검증 실험 3 개 +
  Bug #1 fix 와 같은 dtype-aware 분기 1 개의 경로가 진단 doc 의 next-step 보다 훨씬 짧음.

---

## (a) fwd bias — SMEM read 80% 원인 가설

### 측정으로 이미 확정된 사실

[BIAS_FWD_BOTTLENECK_DECOMPOSITION.md](./BIAS_FWD_BOTTLENECK_DECOMPOSITION.md) 에서:

- bias 오버헤드의 **80%** = `apply_bias_smem` 의 SMEM bias read 경로
- 20% = bias TMA 경로
- kv_stage shrinkage (6→2) 와 fma 비용은 거의 0
- bf16 dtype isolation 에서 30% 회복

원래 doc 이 다음 단계로 남긴 open question:

> "What is the actual SMEM bank pattern for `sBias[q,k,stage]` at the current layout?
> Worth checking with `ncu` or a small bank-conflict micro-probe before committing
> to layout changes."

이 section 은 그 open question 에 대한 가장 유력한 답 + 검증 방법.

### 의심 코드

[flash_bias_fwd_sm100_smem.py:301-308](../FA4_fp32/kernels/flash_bias_fwd_sm100_smem.py#L301-L308):

```python
sBias_layout = cute.make_layout(
    (self.m_block_size, self.n_block_size, self.q_stage),   # (128, 64, 2)
    stride=(
        self.n_block_size,                                   # 64
        1,                                                   # 1
        self.m_block_size * self.n_block_size,               # 8192
    ),
)
```

- **swizzle 없음** (plain row-major)
- fp32 stride 64 = 32 SMEM banks 의 정수배
- 같은 커널의 K/V 는
  [flash_bias_fwd_sm100_smem.py:289](../FA4_fp32/kernels/flash_bias_fwd_sm100_smem.py#L289)
  `sm100_utils_basic.make_smem_layout_b` 로 제대로 swizzle 된 layout. **bias 만 raw**.

### Bank 계산

fp32 element 의 bank = `(byte_offset / 4) % 32`. `sBias[q, k]` 선형 offset = `q*64 + k`
elements, fp32 → 4 bytes/element.

```
bank(sBias[q, k]) = (q*64 + k) % 32
                  = ((q*64) % 32) + (k % 32)
                  = 0 + (k % 32)                    ← 64 는 32 의 배수
                  = k % 32
```

→ **q 에 의존하지 않고 k 만으로 결정됨.** 한 warp 인스턴트에 32 lane 이 *같은 k, 다른 q*
로 동시 접근하면 **32-way bank conflict** 확정.

### 가정 부분 (단정 못 하는 이유)

`Ld32x32bOp(Repetition(32))` 의 lane↔element 분포가 *"lane t → row t, 32 element 는 같은
row 의 다른 k 들"* 이라는 표준 mapping 이라는 가정. 이 가정이면 한 warp 인스턴트에
32 lane 이 같은 k (다른 q) → 32-way conflict.

가정이 뒤집혀서 *"lane 이 col 방향, 32 element 가 row 방향"* 이면 같은 q 다른 k →
bank 0..31 다 다름 → conflict 없음. 그 경우 80% 비용은 다른 원인 (SMEM port 점유율,
latency hiding 실패 등) 이라는 결론으로 바뀜.

로컬에 cutlass DSL 의 atom 정의 (`Ld32x32bOp` 의 lane mapping 명세) 가 없어서 코드/문서
로는 확정 불가.

### 정황 증거 (가정 없이도 남음)

1. **K/V 는 swizzle, bias 만 raw row-major** — 의도적 차이의 합리적 이유 없음.
   "일단 동작" 흔적.
2. **fp32 + stride 64 + no swizzle** 조합은 어떤 합리적 lane 분포에도 일부 인스턴트에서
   conflict 가능. swizzle 을 안 쓴 이상 stride 정렬에서 자유롭지 않음.
3. **bf16 isolation 에서 30% 회복** —
   [BIAS_FWD_OPTIMIZATION_PLAN.md "HBM isolation experiment"](./BIAS_FWD_OPTIMIZATION_PLAN.md).
   bf16 이면 stride 64 bf16 (= 128 bytes/row) 이고 q 마다 bank 16 개씩 회전 → conflict
   가 부분적으로 풀려야 함. 측정과 정합.

### 결판 방법

**ncu 1 커맨드** (가설 확정용):

```bash
ncu --metrics \
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
  smsp__inst_executed_pipe_lsu.sum \
  python FA4_fp32/exp_bias_apply_decompose.py --L 1024
```

K/V 는 swizzle 덕에 conflict 카운터 0 근처, bias 만 크게 튀면 가설 확정. 0 이면 가설
폐기 → SMEM port 점유율 / latency 쪽 다른 원인 탐색.

### Fix 후보

| 옵션 | 메커니즘 | trade-off |
|---|---|---|
| **A. swizzle 적용 (정공법)** | K/V 처럼 `sm100_utils_basic.make_smem_layout_b` 류 사용 | SMEM 추가 없음 → kv_stage 보존. TMA atom 호환 위해 추가 변경 |
| **B. stride 패딩 (64 → 65)** | bank rotation 으로 conflict 해소 | SMEM 늘어남 → kv_stage 더 줄어듦. 실용성 낮음 |
| **C. SMEM 우회 (TMA → register 직스트림)** | bias 를 SMEM 거치지 않고 register 로 받기 | SM100 가능성 불확실. 효과 크지만 구현 비용 큼 |

진단 doc 의 lever B/C 에 해당. **ncu 로 가설 확정 → A 시도** 가 ROI 최선.

---

## (b) bwd fp32 — dP TMEM read root cause 가설 + fix 경로

### 진단 doc 의 현재 상태

[FP32_BWD_DIAGNOSIS.md](./FP32_BWD_DIAGNOSIS.md) 요약:

- **Bug #1** (`dQacc_reduce` reshape over-stride) — fp32 에서 TF32 fragment 패킹 차이를
  reshape 가 못 따라가는 문제. **FIXED**.
- **Bug #2** (dP TMEM read garbage) — V=e_0 probe 로 격리. tStS 는 같은 partition 으로
  정답, tdPtdP 만 깨짐. **NOT FIXED**.

진단 doc 이 next-step 으로 Option A (mechanical search) → B (refactor) → C (upstream)
→ D (SASS) 순서로 적어 놓고 어느 것도 시작 안 한 상태.

### Bug #2 fingerprint (확정된 것)

V=e_0 probe (V[0,0,0,0]=1, V elsewhere=0 → dP[m, n=0] = dout[m, 0], dP[m, n>0] = 0):

- tidx=0 V[0] (cell `m=0, n=0`) → **정답** (-0.924 = dout[0, 0])
- tidx=0 V[1..] (cells `m=1.., n=0`) → **garbage, but deterministic** (예: -0.180, 0.224, -0.262)
- tidx≥1 V[0..] (n≥1 column) → 정답 (다 zero, V=e_0 이므로)
- 동일 partition, 동일 `make_trivial_tiled_mma` 시그니처
- 값이 jitter 없이 일관됨 → pipeline sync gap 가설은 약화

### 가장 유력한 가설

**TF32 tcgen05.mma 의 C-fragment TMEM cell 매핑이 f16 의 그것과 다르고, cute_dsl 의
`make_fragment_C` 가 그 차이를 반영하지 못함.** Bug #1 의 RMEM 쪽 fragment 패킹 차이와
같은 뿌리 — TMEM 버전.

근거:

1. S MMA 와 dP MMA 가 같은 partition 으로 한쪽만 깨짐 → 차이는 **MMA 의 store 단계**.
   read partition 은 무죄.
2. V[0] 만 정답, V[1..] 가 garbage → **base 는 맞고 stride 만 어긋난** 전형적 fingerprint
   (cell 이 claimed row-stride 65536 이 아닌 다른 패턴으로 배치됨).
3. Bug #1 의 *"TF32 packs N across threads more tightly, halving the per-thread V count"*
   가 이미 **RMEM 쪽에서** 확인됨. 같은 종류의 fragment-layout 차이가 TMEM 쪽에도 있다
   고 보는 게 가장 자연스러움.

진단 doc 의 가설 #1 ("MMA-private cell layout for some operand combinations") 과 같은
방향이지만, 더 구체적 형태: **`make_fragment_C` 의 layout 객체가 f16-style 을 가정하고
있어서 TF32 acc 의 실제 TMEM 배치와 불일치**.

### 진단 doc 이 빠뜨린 cheap 한 검증 실험 3 개

이 3 개를 하면 root cause 가 deterministic 하게 좁혀짐. 진단 doc 의 Option A
(mechanical search) 보다 압도적으로 짧음.

**검증 1: layout printf (5 분)**

```python
cute.printf("tStS.layout   = %s\n", str(tStS.layout))
cute.printf("tdPtdP.layout = %s\n", str(tdPtdP.layout))
```

진단 doc 은 두 layout 이 *byte-identical* 이라고 단정만 함. 실제로 컴파일 타임 직접
출력해서 확인 필요. 다르다면 그 차이가 단서.

**검증 2: MMA swap test (30 분)**

같은 V=e_0 probe 입력으로:
- S MMA 를 `tmem_dP_offset` 에 쓰고 같은 partition 으로 읽기
- dP MMA 를 `tmem_S_offset` 에 쓰고 같은 partition 으로 읽기

어느 쪽이 따라가는지로 **MMA store 문제 vs TMEM offset 문제** 가 한 번에 갈림.

**검증 3: SASS diff (1 시간)**

```bash
CUTE_DSL_KEEP_PTX=1 ... # 컴파일
nvdisasm cubin > sass.txt
grep -A5 "tcgen05.mma" sass.txt
```

S MMA 와 dP MMA 의 `tcgen05.mma` 명령 인자 직접 비교. cell 매핑을 hardware 단에서 확정.

세 실험 합쳐 **2 시간 이내**. 진단 doc 은 Option A 의 "mechanical search" 부터 시작
하라고 적었지만 위 3 개가 deterministic 하게 root cause 를 좁혀줌.

### Root cause 별 fix

#### Case A — 두 MMA 의 C-frag store 패턴이 실제 다름 (가능성 가장 높음)

**Bug #1 fix 와 동일한 형태의 dtype-aware 분기**. [flash_bwd_sm100.py:1622-1632](../FA4_fp32/kernels/flash_bwd_sm100.py#L1622-L1632)
근처에:

```python
if const_expr(fp32_bwd):
    # SASS 에서 확인한 TF32-친화적 atom 으로 교체
    # (예: Ld16x256bOp, 또는 관찰된 실제 cell 패턴에 맞는 atom)
    tmem_load_atom_dP = cute.make_copy_atom(
        tcgen05.copy.Ld16x256bOp(...),
        Float32,
    )
    thr_copy_t2r_dP = tcgen05.make_tmem_copy(
        tmem_load_atom_dP, tdPtdP,
    ).get_slice(tidx)
else:
    thr_copy_t2r_dP = thr_copy_t2r   # bf16 기존 경로 유지 (9/9 PASS)
```

S MMA 쪽은 손대지 않음 — fp32 에서도 동작하니까. dP 쪽만 분기. Bug #1 fix
([flash_bwd_sm100.py:dQacc_reduce](../FA4_fp32/kernels/flash_bwd_sm100.py)) 가 이미
같은 패턴 (`if const_expr(fp32_dQ)`) 으로 작성돼 있어서 일관성 있음.

#### Case B — layout 객체는 같은데 hardware store 만 다름 (cute_dsl 버그)

- 손코딩으로 `tdPtdP` 의 layout 재구성: `cute.make_layout` 으로 stride 직접 지정.
  SASS 에서 관찰한 실제 cell stride 를 그대로 입력.
- 상류 cutlass-dsl 팀에 issue 제출. SASS dump 첨부.

#### Case C — MMA swap test 에서 dP slot 이 망가짐 (TMEM 영역 overlap)

- `tmem_dP_offset` 재계산 ([flash_bwd_sm100.py:867](../FA4_fp32/kernels/flash_bwd_sm100.py#L867))
- `tStP` (P storage overlapping S),
  `tdPtdS` (dS storage overlapping dP) 가 fp32 에서 인접 region 침범하는지 점검
- 진단 doc 은 이 가능성을 "less likely" 라고만 했는데 정량 확인 안 됨

### Bug #1 fix 보완 작업

진단 doc 은 fix 후 *"dq_accum's second half went 0 → 2.59 max"* + *"dQ output cols
16..31 now mirror cols 0..15"* 만 확인. **PyTorch reference 와의 수치 일치 비교는
아직 안 됨** (Bug #2 때문에 어차피 전체 fail 이지만, fix 자체의 정합성을 분리해서
검증해두면 Bug #2 fix 직후 즉시 verify 가능).

작은 보강: probe 조건을 PyTorch 로 reference 계산해서 fp32 fix 후의 dq cols 16..31
이 *수치적으로* 일치하는지 확인.

### 우선순위

| 순서 | 작업 | 비용 | 결판력 |
|---|---|---|---|
| 1 | 검증 1: layout printf | 5 분 | doc 의 단정 검증 |
| 2 | 검증 2: MMA swap test | 30 분 | MMA store vs TMEM offset 분리 |
| 3 | 검증 3: SASS diff | 1 시간 | 실제 cell 매핑 확보 |
| 4 | 결과에 따라 Case A/B/C 적용 | 0.5–2 일 | 직접 fix |
| 5 | Bug #1 fix 의 reference 비교 보강 | 1 시간 | 정합성 분리 |
| 6 | 막히면 cutlass-dsl 팀 질의 (3 번 결과 첨부) | 외부 의존 | upstream fix |

진단 doc 이 4 번부터 시작하면서 "어렵다" 고 멈춰 있는 상태. 1–3 번을 안 한 채로 멈춘
게 진짜 문제. **1–3 번 (합쳐 2 시간 이내) 만 돌리면 Case A/B/C 중 어디로 갈지
deterministic 하게 결정됨.**

### 한 줄 요약

> **Bug #1 fix 와 동일한 형태의 dtype-aware 분기를 dP read 경로에 추가하는 게 가장
> 가능성 높은 fix.** 어떤 atom/layout 으로 분기할지는 SASS 1 회 관찰로 결정. 그 전에
> layout printf + MMA swap test 2 개로 가설을 확정.

---

## 공통: 두 분석의 공통점

두 케이스 모두 *"같은 종류의 차이를 cute_dsl/CUTLASS layout helper 가 따라잡지 못한다"*
는 패턴.

- **fwd bias**: K/V 는 `make_smem_layout_b` 로 swizzle 됐는데 bias 는 raw row-major 로
  사람 손으로 만들어진 layout → bank conflict 가능성.
- **bwd fp32**: f16 acc 에 맞춰진 `make_fragment_C` 의 layout 이 TF32 acc 의 실제 TMEM
  배치와 불일치 → dP read 가 잘못된 cell 을 봄.

둘 다 **"DSL helper 가 알아서 해주는 줄 알았는데, dtype/operand 가 바뀌면 helper 가
틀린다"** 라는 동일 형태의 함정. fp32 path 가 main path 가 아니라 추가 path 로 들어간
역사적 흐름에서 자연스럽게 생긴 종류의 버그.

향후 fp32 path 를 손볼 때 디폴트 의심 영역으로 둘 만함.
