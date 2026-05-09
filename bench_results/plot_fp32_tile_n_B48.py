"""Plot fp32 q_stage=2 with tile_n=64 vs tile_n=128 vs upstream bf16, B=48."""
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

LOG = Path(__file__).with_name("bench_fp32_tile_n_B48.txt")
OUT = Path(__file__).with_name("bench_fp32_tile_n_B48.png")

# data[D][L] = (tn64_tf, tn128_tf, bf16_tf, tn64_ms, tn128_ms, bf16_ms)
data = defaultdict(dict)
for line in LOG.read_text().splitlines():
    m = re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
                 r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
                 r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+x", line)
    if not m:
        continue
    _B, _H, L, D, tn64_ms, tn128_ms, bf16_ms, tn64_tf, tn128_tf, bf16_tf = m.groups()
    data[int(D)][int(L)] = (float(tn64_tf), float(tn128_tf), float(bf16_tf),
                            float(tn64_ms), float(tn128_ms), float(bf16_ms))

HDIMS = [32, 64]
COLORS = {"tn64": "#F5B97D", "tn128": "#E8704C", "bf16": "#4C9BE8"}

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for col, D in enumerate(HDIMS):
    points = data[D]
    Ls = sorted(points)
    tn64    = [points[L][0] for L in Ls]
    tn128   = [points[L][1] for L in Ls]
    b16     = [points[L][2] for L in Ls]
    tn64_ms  = [points[L][3] for L in Ls]
    tn128_ms = [points[L][4] for L in Ls]
    b16_ms   = [points[L][5] for L in Ls]

    x = np.arange(len(Ls))
    w = 0.27

    # TFLOPS row
    ax = axes[0, col]
    ax.bar(x - w, tn64,  w, color=COLORS["tn64"],  label="FA4 fp32 (q2, tile_n=64)")
    ax.bar(x,     tn128, w, color=COLORS["tn128"], label="FA4 fp32 (q2, tile_n=128)")
    ax.bar(x + w, b16,   w, color=COLORS["bf16"],  label="FA4 bf16 (upstream)")
    for xi, v in zip(x - w, tn64):
        ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x,     tn128):
        ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x + w, b16):
        ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([f"{L // 1024}k" for L in Ls])
    ax.set_title(f"D={D}")
    ax.grid(axis="y", alpha=0.3)
    if col == 0:
        ax.set_ylabel("TFLOPS")
        ax.legend(loc="upper left", fontsize=8)

    # Time row
    ax = axes[1, col]
    ax.bar(x - w, tn64_ms,  w, color=COLORS["tn64"])
    ax.bar(x,     tn128_ms, w, color=COLORS["tn128"])
    ax.bar(x + w, b16_ms,   w, color=COLORS["bf16"])
    for xi, v in zip(x - w, tn64_ms):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x,     tn128_ms):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x + w, b16_ms):
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

fig.suptitle("FA4 fp32 q_stage=2: tile_n=64 vs 128 vs upstream bf16 — B200, B=48, H=8",
             y=1.00, fontsize=12)
fig.tight_layout()
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print(f"saved: {OUT}")
