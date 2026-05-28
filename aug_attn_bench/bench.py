"""Benchmark: fused augmented-attention-pair-bias (Triton)
                 vs.
              split path = FA4(Q,K,V) + softmax(bias) @ V (torch.compile)

Outputs are NOT expected to match (the fused kernel mixes bias into the QK
softmax, while the split path softmaxes them independently and sums). This
script only compares latency.

Shapes match `token_dit` config (`d_single=768, n_head=16` -> head_dim=48,
`A=n_augment=48`, batch B=1, dtype bf16, L swept 64..384).

Run:
  CUDA_VISIBLE_DEVICES=0 python aug_attn_bench/bench.py
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

# --- Stub team_gm.typecheck so the kernel module imports without the full
# team_gm package (whose __init__ pulls in beartype, jaxtyped, .core, etc.).
_stub = types.ModuleType("team_gm")
_stub.typecheck = lambda f: f
sys.modules["team_gm"] = _stub

ROOT = Path(__file__).resolve().parent.parent

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from einops import rearrange  # noqa: E402

# --- Load the kernel module by file path (avoids touching team_gm.modules
# __init__, which pulls in unrelated kernels and blocks).
import importlib.util  # noqa: E402

_kernel_path = (
    ROOT / ".team-gm-perf-trimul/src/team_gm/modules/kernels"
    / "augmented_attention_pair_bias.py"
)
_spec = importlib.util.spec_from_file_location("_aug_attn_kernel", str(_kernel_path))
_kmod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_kmod)
triton_augmented_attention_pair_bias = _kmod.triton_augmented_attention_pair_bias

# Option 2: QK path. Prefer FA4 cute if it imports cleanly; otherwise fall
# back to torch SDPA (which on H100 dispatches to cuDNN/FA backend for fp16/bf16).
try:
    from flash_attn.cute import flash_attn_func as fa4_func  # noqa: E402

    _QK_BACKEND = "FA4"
except Exception as _e:  # noqa: BLE001
    fa4_func = None
    _QK_BACKEND = "SDPA"
    print(f"[bench] flash_attn.cute unavailable ({_e!r}); using SDPA for QK.")


# ---------------------------------------------------------------------------
# Option 1: fused
# ---------------------------------------------------------------------------
def fused_attn(q, k, v, bias, mask):
    # q,k,v: (A,B,H,L,D)  bias: (B,H,L,L)  mask: (A,B,L) bool
    return triton_augmented_attention_pair_bias(q, k, v, bias, mask)


# ---------------------------------------------------------------------------
# Option 2: split. softmax(bias) @ V path compiled with torch.compile.
# ---------------------------------------------------------------------------
def _bias_path_impl(bias, v, mask):
    # bias: (B,H,L,L)  v: (A,B,H,L,D)  mask: (A,B,L) bool or None
    if mask is None:
        # Broadcast over the A dim in the einsum so the (A,B,H,L,L) attn
        # tensor is never materialized. Critical for large L.
        attn = F.softmax(bias.float(), dim=-1).to(v.dtype)  # (B,H,L,L)
        return torch.einsum("bhij,abhjd->abhid", attn, v)
    b = bias.unsqueeze(0).masked_fill(~mask[:, :, None, None, :], float("-inf"))
    attn = F.softmax(b.float(), dim=-1).to(v.dtype)
    return torch.einsum("abhij,abhjd->abhid", attn, v)


bias_path_compiled = torch.compile(_bias_path_impl, dynamic=False, fullgraph=False)


def qk_only(q, k, v, bias, mask):
    """Only the QK attention path of split (no bias). Same compute as SDPA."""
    A, B, H, L, D = q.shape
    if fa4_func is not None:
        q_fa = rearrange(q, "A B H L D -> (A B) L H D").contiguous()
        k_fa = rearrange(k, "A B H L D -> (A B) L H D").contiguous()
        v_fa = rearrange(v, "A B H L D -> (A B) L H D").contiguous()
        out_qk = fa4_func(q_fa, k_fa, v_fa, softmax_scale=D ** -0.5, causal=False)
        return rearrange(out_qk, "(A B) L H D -> A B H L D", A=A)
    q_s = q.reshape(A * B, H, L, D)
    k_s = k.reshape(A * B, H, L, D)
    v_s = v.reshape(A * B, H, L, D)
    out_qk = F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=False)
    return out_qk.reshape(A, B, H, L, D)


def bias_only(q, k, v, bias, mask):
    """Only the softmax(bias)·V path."""
    return bias_path_compiled(bias, v, mask)


def split_attn(q, k, v, bias, mask):
    return qk_only(q, k, v, bias, mask) + bias_only(q, k, v, bias, mask)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def make_inputs(A, B, H, L, D, dtype, device, requires_grad, use_mask=False):
    q = torch.randn(A, B, H, L, D, dtype=dtype, device=device, requires_grad=requires_grad)
    k = torch.randn(A, B, H, L, D, dtype=dtype, device=device, requires_grad=requires_grad)
    v = torch.randn(A, B, H, L, D, dtype=dtype, device=device, requires_grad=requires_grad)
    bias = torch.randn(B, H, L, L, dtype=dtype, device=device, requires_grad=requires_grad)
    mask = (torch.ones(A, B, L, dtype=torch.bool, device=device) if use_mask else None)
    return q, k, v, bias, mask


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def _zero_grads(args):
    for t in args:
        if isinstance(t, torch.Tensor) and t.requires_grad:
            t.grad = None


def bench_fwd(fn, args, n_warmup, n_iters):
    for _ in range(n_warmup):
        with torch.no_grad():
            fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        with torch.no_grad():
            fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iters


def bench_fwd_bwd(fn, args, n_warmup, n_iters):
    for _ in range(n_warmup):
        _zero_grads(args)
        out = fn(*args)
        out.sum().backward()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        _zero_grads(args)
        out = fn(*args)
        out.sum().backward()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iters


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--A", type=int, default=48, help="augment dim (n_augment)")
    p.add_argument("--B", type=int, default=1, help="batch")
    p.add_argument("--H", type=int, default=16, help="n_head")
    p.add_argument("--D", type=int, default=48, help="head_dim (d_single/n_head)")
    p.add_argument("--lens", type=int, nargs="+", default=[64, 128, 256, 384])
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--mode", choices=["fwd", "fwd_bwd", "both"], default="both")
    p.add_argument("--use-mask", action="store_true",
                   help="materialize (A,B,L) all-True mask (default: pass None, "
                        "lets split path skip the (A,B,H,L,L) attn materialization)")
    p.add_argument("--out", type=str, default=None,
                   help="path to write txt log (parsed by plot_aug_attn.py)")
    args = p.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = "cuda"
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log(f"GPU: {torch.cuda.get_device_name(0)}  dtype={args.dtype}")
    log(f"A={args.A} B={args.B} H={args.H} D={args.D}")
    log(f"QK backend: {_QK_BACKEND}")
    log(f"warmup={args.warmup} iters={args.iters}")
    log()

    if args.mode in ("fwd", "both"):
        log("== FWD (ms / call) ==")
        log(f"{'L':>6} {'fused':>12} {'split':>12} {'qk_only':>12} {'bias_only':>12}")
        for L in args.lens:
            inp = make_inputs(args.A, args.B, args.H, L, args.D, dtype, device, False, args.use_mask)
            t_f = bench_fwd(fused_attn, inp, args.warmup, args.iters)
            t_s = bench_fwd(split_attn, inp, args.warmup, args.iters)
            t_qk = bench_fwd(qk_only, inp, args.warmup, args.iters)
            t_b = bench_fwd(bias_only, inp, args.warmup, args.iters)
            log(f"{L:>6d} {t_f:>12.4f} {t_s:>12.4f} {t_qk:>12.4f} {t_b:>12.4f}")
            del inp
            torch.cuda.empty_cache()
        log()

    if args.mode in ("fwd_bwd", "both"):
        log("== FWDBWD (ms / call) ==")
        log(f"{'L':>6} {'fused':>12} {'split':>12} {'qk_only':>12} {'bias_only':>12}")
        for L in args.lens:
            inp = make_inputs(args.A, args.B, args.H, L, args.D, dtype, device, True, args.use_mask)
            t_f = bench_fwd_bwd(fused_attn, inp, args.warmup, args.iters)
            t_s = bench_fwd_bwd(split_attn, inp, args.warmup, args.iters)
            t_qk = bench_fwd_bwd(qk_only, inp, args.warmup, args.iters)
            t_b = bench_fwd_bwd(bias_only, inp, args.warmup, args.iters)
            log(f"{L:>6d} {t_f:>12.4f} {t_s:>12.4f} {t_qk:>12.4f} {t_b:>12.4f}")
            del inp
            torch.cuda.empty_cache()

    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
