"""Plot fp32 FA4 vs fp32 PyTorch reference, forward pass only.

NOTE: The standing rule is to plot fwd/bwd/fwd+bwd together, but the fp32
FA4 backward is not implemented yet — only forward can be compared here.
"""
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

LOG = Path(__file__).with_name("bench_fp32_fwd.txt")
OUT = Path(__file__).with_name("bench_fp32_fwd.png")

rows_full = []
rows_caus = []
for line in LOG.read_text().splitlines():
    m = re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(True|False)\s+"
                 r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x", line)
    if m:
        B, H, L, D, causal, fa_ms, ref_ms, fa_tf, ref_tf, spd = m.groups()
        row = dict(L=int(L), fa_ms=float(fa_ms), ref_ms=float(ref_ms),
                   fa_tf=float(fa_tf), ref_tf=float(ref_tf),
                   speedup=float(spd))
        (rows_caus if causal == "True" else rows_full).append(row)


def draw(ax_t, ax_ms, rows, title):
    Ls = [r["L"] for r in rows]
    x = np.arange(len(Ls))
    w = 0.35

    fa_tf = [r["fa_tf"] for r in rows]
    rf_tf = [r["ref_tf"] for r in rows]
    ax_t.bar(x - w/2, rf_tf, w, color="#999999", label="PyTorch TF32 ref")
    ax_t.bar(x + w/2, fa_tf, w, color="#E8704C", label="FA4 fp32 (TF32 MMA)")
    for xi, v in zip(x - w/2, rf_tf):
        ax_t.text(xi, v + 10, f"{v:.0f}", ha="center", fontsize=8)
    for xi, v in zip(x + w/2, fa_tf):
        ax_t.text(xi, v + 10, f"{v:.0f}", ha="center", fontsize=8)
    ax_t.set_xticks(x)
    ax_t.set_xticklabels([f"{L//1024}k" if L >= 1024 else str(L) for L in Ls])
    ax_t.set_ylabel("TFLOPS")
    ax_t.set_title(title)
    ax_t.grid(axis="y", alpha=0.3)
    ax_t.set_ylim(0, max(max(fa_tf), max(rf_tf)) * 1.25)
    ax_t.legend(loc="upper left")

    fa_ms = [r["fa_ms"] for r in rows]
    rf_ms = [r["ref_ms"] for r in rows]
    ax_ms.bar(x - w/2, rf_ms, w, color="#999999", label="PyTorch TF32 ref")
    ax_ms.bar(x + w/2, fa_ms, w, color="#E8704C", label="FA4 fp32")
    for xi, v in zip(x - w/2, rf_ms):
        ax_ms.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x + w/2, fa_ms):
        ax_ms.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax_ms.set_xticks(x)
    ax_ms.set_xticklabels([f"{L//1024}k" if L >= 1024 else str(L) for L in Ls])
    ax_ms.set_ylabel("Time (ms)")
    ax_ms.set_yscale("log")
    ax_ms.grid(axis="y", alpha=0.3, which="both")

    # Annotate speedup above FA4 bars on the TFLOPS plot.
    for xi, r in zip(x + w/2, rows):
        ax_t.annotate(f"{r['speedup']:.1f}x", (xi, r["fa_tf"]),
                      xytext=(0, -12), textcoords="offset points",
                      ha="center", fontsize=7, color="white", weight="bold")


fig, axes = plt.subplots(2, 2, figsize=(12, 8))
draw(axes[0, 0], axes[1, 0], rows_full, "Forward — non-causal")
draw(axes[0, 1], axes[1, 1], rows_caus, "Forward — causal")

fig.suptitle(
    "fp32 Forward on B200 — FA4 (TF32 MMA) vs PyTorch reference (TF32)\n"
    "hdim=128, B=2, H=8     (bwd not yet implemented in fp32 path)",
    y=1.00,
)
fig.tight_layout()
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print(f"saved: {OUT}")
