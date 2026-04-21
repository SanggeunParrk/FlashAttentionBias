"""Plot fp32 FA4 vs bf16 FA4 vs PyTorch ref, non-causal only."""
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

LOG = Path(__file__).with_name("bench_fp32_vs_bf16.txt")
OUT = Path(__file__).with_name("bench_fp32_vs_bf16.png")

# data[D][L] = (fa32_tf, fa16_tf, ref_tf, fa32_ms, fa16_ms, ref_ms)
data = defaultdict(dict)
for line in LOG.read_text().splitlines():
    m = re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
                 r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
                 r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+x", line)
    if not m:
        continue
    _B, _H, L, D, fa32_ms, fa16_ms, ref_ms, fa32_tf, fa16_tf, ref_tf = m.groups()
    data[int(D)][int(L)] = (float(fa32_tf), float(fa16_tf), float(ref_tf),
                            float(fa32_ms), float(fa16_ms), float(ref_ms))

HDIMS = [32, 64, 96, 128]
COLORS = {"ref": "#999999", "fp32": "#E8704C", "bf16": "#4C9BE8"}

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for col, D in enumerate(HDIMS):
    points = data[D]
    Ls = sorted(points)
    fa32 = [points[L][0] for L in Ls]
    fa16 = [points[L][1] for L in Ls]
    rf   = [points[L][2] for L in Ls]
    fa32_ms = [points[L][3] for L in Ls]
    fa16_ms = [points[L][4] for L in Ls]
    rf_ms   = [points[L][5] for L in Ls]

    x = np.arange(len(Ls))
    w = 0.27

    # TFLOPS row
    ax = axes[0, col]
    ax.bar(x - w, rf,   w, color=COLORS["ref"],  label="PyTorch TF32")
    ax.bar(x,     fa32, w, color=COLORS["fp32"], label="FA4 fp32")
    ax.bar(x + w, fa16, w, color=COLORS["bf16"], label="FA4 bf16")
    for xi, v in zip(x - w, rf):
        ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x, fa32):
        ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x + w, fa16):
        ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([f"{L // 1024}k" for L in Ls])
    ax.set_title(f"D={D}")
    ax.grid(axis="y", alpha=0.3)
    if col == 0:
        ax.set_ylabel("TFLOPS")
        ax.legend(loc="upper left")

    # Time row (log)
    ax = axes[1, col]
    ax.bar(x - w, rf_ms,   w, color=COLORS["ref"])
    ax.bar(x,     fa32_ms, w, color=COLORS["fp32"])
    ax.bar(x + w, fa16_ms, w, color=COLORS["bf16"])
    for xi, v in zip(x - w, rf_ms):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x, fa32_ms):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x + w, fa16_ms):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([f"{L // 1024}k" for L in Ls])
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.set_xlabel("seqlen")
    if col == 0:
        ax.set_ylabel("Time (ms)")

# share y per row
for row in range(2):
    ymax = max(axes[row, c].get_ylim()[1] for c in range(4))
    for c in range(4):
        if row == 0:
            axes[row, c].set_ylim(0, ymax * 1.12)

fig.suptitle("fp32 FA4 vs bf16 FA4 vs PyTorch ref (non-causal) — B200, B=2, H=8",
             y=1.00, fontsize=12)
fig.tight_layout()
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print(f"saved: {OUT}")
