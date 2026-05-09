"""Plot FA4 fp32 vs FA4 bf16 (upstream) vs FA2 bf16 (SDPA), B=48."""
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

LOG = Path(__file__).with_name("bench_fa2_compare_B48.txt")
OUT = Path(__file__).with_name("bench_fa2_compare_B48.png")

# data[D][L] = (fp32_tf, bf16_tf, fa2_tf, fp32_ms, bf16_ms, fa2_ms)
data = defaultdict(dict)
for line in LOG.read_text().splitlines():
    m = re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
                 r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
                 r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+x", line)
    if not m:
        continue
    _B, _H, L, D, fp32_ms, bf16_ms, fa2_ms, fp32_tf, bf16_tf, fa2_tf = m.groups()
    data[int(D)][int(L)] = (float(fp32_tf), float(bf16_tf), float(fa2_tf),
                            float(fp32_ms), float(bf16_ms), float(fa2_ms))

HDIMS = [32, 64]
COLORS = {"fp32": "#E8704C", "bf16": "#4C9BE8", "fa2": "#7E57C2"}

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for col, D in enumerate(HDIMS):
    points = data[D]
    Ls = sorted(points)
    fp32   = [points[L][0] for L in Ls]
    bf16   = [points[L][1] for L in Ls]
    fa2    = [points[L][2] for L in Ls]
    fp32_ms = [points[L][3] for L in Ls]
    bf16_ms = [points[L][4] for L in Ls]
    fa2_ms  = [points[L][5] for L in Ls]

    x = np.arange(len(Ls))
    w = 0.27

    # TFLOPS row
    ax = axes[0, col]
    ax.bar(x - w, fp32, w, color=COLORS["fp32"], label="FA4 fp32 (q2, tile_n=128)")
    ax.bar(x,     bf16, w, color=COLORS["bf16"], label="FA4 bf16 (upstream)")
    ax.bar(x + w, fa2,  w, color=COLORS["fa2"],  label="FA2 bf16 (SDPA FLASH)")
    for xi, v in zip(x - w, fp32):
        ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x,     bf16):
        ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x + w, fa2):
        ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([f"{L // 1024}k" for L in Ls])
    ax.set_title(f"D={D}")
    ax.grid(axis="y", alpha=0.3)
    if col == 0:
        ax.set_ylabel("TFLOPS")
        ax.legend(loc="upper left", fontsize=8)

    # Time row
    ax = axes[1, col]
    ax.bar(x - w, fp32_ms, w, color=COLORS["fp32"])
    ax.bar(x,     bf16_ms, w, color=COLORS["bf16"])
    ax.bar(x + w, fa2_ms,  w, color=COLORS["fa2"])
    for xi, v in zip(x - w, fp32_ms):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x,     bf16_ms):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x + w, fa2_ms):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([f"{L // 1024}k" for L in Ls])
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlabel("seqlen")
    if col == 0:
        ax.set_ylabel("Time (ms)")

# share y per row
for row in range(2):
    ymax = max(axes[row, c].get_ylim()[1] for c in range(2))
    for c in range(2):
        if row == 0:
            axes[row, c].set_ylim(0, ymax * 1.12)
        else:
            axes[row, c].set_ylim(0, ymax * 1.10)

fig.suptitle("FA4 fp32 vs FA4 bf16 (upstream) vs FA2 bf16 (SDPA) — B200, B=48, H=8, non-causal",
             y=1.00, fontsize=12)
fig.tight_layout()
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print(f"saved: {OUT}")
