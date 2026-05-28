"""Smoke test: FA4_fp32 forward on H100 (SM90) with float32 inputs.

Tries the speculative TF32 WGMMA path added in
``FA4_fp32/arch/hopper_helpers.py``. If MLIR codegen does NOT support TF32 for
``MmaAtomSM90Type``, this script will fail at compile/lowering time with a
clear error message — that tells us we need to fall back to inline_asm.
"""

import os
import sys

import torch

# Pretend SM90 even if the device reports something else (useful when probing
# without an H100). On real H100 this matches.
os.environ.setdefault("FLASH_ATTENTION_ARCH", "90")

from FA4_fp32 import flash_attn_func, attention_fp32


def reference(q, k, v):
    # q, k, v: (B, L, H, D) float32
    out_ref, _ = attention_fp32(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
    )
    return out_ref.squeeze(0) if out_ref.dim() == 5 else out_ref


def run_case(B=2, H=8, L=1024, D=64):
    print(f"== fp32 SM90 FA4 forward  B={B} H={H} L={L} D={D}")
    torch.manual_seed(0)
    device = "cuda"
    q = torch.randn(B, L, H, D, dtype=torch.float32, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # First call triggers JIT compile — print the timing separately.
    print("  - jit + first call ...")
    out, lse = flash_attn_func(q, k, v, return_lse=True)
    torch.cuda.synchronize()
    print(f"  - out shape={tuple(out.shape)} dtype={out.dtype}")
    print(f"  - lse shape={tuple(lse.shape)} dtype={lse.dtype}")

    print("  - reference (torch fp32 / TF32) ...")
    ref = reference(q, k, v)
    if ref.shape != out.shape:
        print(f"  - WARNING shape mismatch: ref {ref.shape} vs out {out.shape}")
        return

    diff = (out - ref).abs()
    print(f"  - max abs err = {diff.max().item():.3e}")
    print(f"  - mean abs err = {diff.mean().item():.3e}")


def main():
    print(f"torch={torch.__version__}  cuda available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device={torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True

    cases = [
        (2, 8, 1024, 64),
        (2, 8, 2048, 64),
        (1, 4, 4096, 32),
    ]
    for case in cases:
        try:
            run_case(*case)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
