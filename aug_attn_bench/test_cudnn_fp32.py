"""Test cuDNN 9.9+ true-FP32 fused flash attention.

Two paths:
  (1) torch SDPA with CUDNN_ATTENTION backend forced.
  (2) cudnn-frontend Python API directly (cudnn.pygraph SDPA op).
"""

import sys
import time

import torch
import torch.nn.functional as F

print("torch:", torch.__version__)
print("cudnn version (torch):", torch.backends.cudnn.version())
print("device:", torch.cuda.get_device_name(0))
print()


# ---------------------------------------------------------------------------
# (1) Torch SDPA with cuDNN backend forced.
# ---------------------------------------------------------------------------
def try_torch_sdpa_cudnn_fp32():
    print("=" * 70)
    print("Path 1: torch SDPA with CUDNN_ATTENTION backend forced (fp32)")
    print("=" * 70)
    from torch.nn.attention import SDPBackend, sdpa_kernel

    B, H, L, D = 2, 8, 1024, 64
    q = torch.randn(B, H, L, D, dtype=torch.float32, device="cuda")
    k = torch.randn(B, H, L, D, dtype=torch.float32, device="cuda")
    v = torch.randn(B, H, L, D, dtype=torch.float32, device="cuda")

    try:
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        torch.cuda.synchronize()
        print(f"OK: output shape={tuple(out.shape)}, dtype={out.dtype}")
        return True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# (2) cudnn-frontend Python API directly.
# ---------------------------------------------------------------------------
def try_cudnn_frontend_fp32():
    print("=" * 70)
    print("Path 2: cudnn-frontend Python API (cudnn.pygraph SDPA, fp32)")
    print("=" * 70)
    try:
        import cudnn
    except ImportError as e:
        print(f"cudnn-frontend not installed: {e}")
        print("install with: pip install nvidia-cudnn-frontend")
        return False

    print(f"cudnn-frontend version: {cudnn.__version__}")
    print(f"cudnn backend version: {cudnn.backend_version()}")

    B, H, L, D = 2, 8, 1024, 64
    q = torch.randn(B, H, L, D, dtype=torch.float32, device="cuda")
    k = torch.randn(B, H, L, D, dtype=torch.float32, device="cuda")
    v = torch.randn(B, H, L, D, dtype=torch.float32, device="cuda")

    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q_t = g.tensor_like(q)
    k_t = g.tensor_like(k)
    v_t = g.tensor_like(v)
    try:
        o, stats = g.sdpa(
            name="sdpa_fp32",
            q=q_t, k=k_t, v=v_t,
            is_inference=True,
            attn_scale=1.0 / (D ** 0.5),
            use_causal_mask=False,
        )
        o.set_output(True).set_dim(q.shape).set_stride(q.stride())
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A])
        g.check_support()
        g.build_plans()
    except Exception as e:
        print(f"FAILED at graph build: {type(e).__name__}: {e}")
        return False

    out = torch.empty_like(q)
    workspace = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)
    try:
        g.execute({q_t: q, k_t: k, v_t: v, o: out}, workspace=workspace)
        torch.cuda.synchronize()
        print(f"OK: output shape={tuple(out.shape)}, dtype={out.dtype}")

        # Compare to reference.
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        diff = (out - ref).abs()
        print(f"vs reference SDPA(fp32 math): max abs err = {diff.max().item():.3e}, "
              f"mean abs err = {diff.mean().item():.3e}")
        return True
    except Exception as e:
        print(f"FAILED at execute: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# (3) If path 2 works, benchmark fp32 cuDNN SDPA vs math/Triton baselines.
# ---------------------------------------------------------------------------
def bench_path2(B=2, H=8, L=4096, D=64, n_warmup=10, n_iters=50):
    print("=" * 70)
    print(f"Bench: cuDNN fp32 SDPA  B={B} H={H} L={L} D={D}")
    print("=" * 70)
    try:
        import cudnn
    except ImportError:
        print("cudnn-frontend not available, skipping.")
        return

    q = torch.randn(B, H, L, D, dtype=torch.float32, device="cuda")
    k = torch.randn(B, H, L, D, dtype=torch.float32, device="cuda")
    v = torch.randn(B, H, L, D, dtype=torch.float32, device="cuda")

    # Build cuDNN graph.
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q_t = g.tensor_like(q)
    k_t = g.tensor_like(k)
    v_t = g.tensor_like(v)
    o, _ = g.sdpa(
        name="sdpa_fp32_bench",
        q=q_t, k=k_t, v=v_t,
        is_inference=True,
        attn_scale=1.0 / (D ** 0.5),
        use_causal_mask=False,
    )
    o.set_output(True).set_dim(q.shape).set_stride(q.stride())
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    g.check_support()
    g.build_plans()
    out = torch.empty_like(q)
    ws = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def run_cudnn():
        g.execute({q_t: q, k_t: k, v_t: v, o: out}, workspace=ws)

    def run_math_sdpa():
        return F.scaled_dot_product_attention(q, k, v, is_causal=False)

    def time_fn(fn):
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / n_iters

    t_cudnn = time_fn(run_cudnn)
    t_math = time_fn(run_math_sdpa)
    print(f"cuDNN fp32 SDPA : {t_cudnn:.3f} ms")
    print(f"torch SDPA fp32 (math fallback): {t_math:.3f} ms")
    print(f"speedup: {t_math / t_cudnn:.2f}x")


if __name__ == "__main__":
    ok1 = try_torch_sdpa_cudnn_fp32()
    print()
    ok2 = try_cudnn_frontend_fp32()
    print()
    if ok2:
        for L in [1024, 2048, 4096, 8192]:
            bench_path2(B=2, H=8, L=L, D=64)
            print()
