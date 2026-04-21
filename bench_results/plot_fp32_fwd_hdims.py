"""Plot fp32 FA4 vs PyTorch ref across hdim ∈ {32,64,96,128} and seqlens.

Layout: 2 rows (non-causal / causal) × 4 cols (hdim).
Only forward — fp32 FA4 backward not yet implemented.
"""
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

LOG = Path(__file__).with_name("bench_fp32_fwd_hdims.txt")
OUT = Path(__file__).with_name("bench_fp32_fwd_hdims.png")

# data[(D, causal)][L] = (fa_tflops, ref_tflops, speedup)
data = defaultdict(dict)
for line in LOG.read_text().splitlines():
    m = re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(True|False)\s+"
                 r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x", line)
    if not m:
        continue
    _B, _H, L, D, causal, _fa, _ref, fa_tf, ref_tf, spd = m.groups()
    data[(int(D), causal == "True")][int(L)] = (float(fa_tf), float(ref_tf), float(spd))

HDIMS = [32, 64, 96, 128]
CAUSALS = [False, True]
COLORS = {"ref": "#999999", "fa4": "#E8704C"}

fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey="row")
for row, causal in enumerate(CAUSALS):
    for col, D in enumerate(HDIMS):
        ax = axes[row, col]
        points = data[(D, causal)]
        Ls = sorted(points)
        fa = [points[L][0] for L in Ls]
        rf = [points[L][1] for L in Ls]
        sp = [points[L][2] for L in Ls]

        x = np.arange(len(Ls))
        w = 0.38
        ax.bar(x - w / 2, rf, w, color=COLORS["ref"], label="PyTorch TF32")
        ax.bar(x + w / 2, fa, w, color=COLORS["fa4"], label="FA4 fp32")
        for xi, v in zip(x - w / 2, rf):
            ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
        for xi, v, s in zip(x + w / 2, fa, sp):
            ax.text(xi, v, f"{v:.0f}\n({s:.1f}x)",
                    ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{L // 1024}k" for L in Ls])
        ax.set_title(f"D={D}, {'causal' if causal else 'non-causal'}")
        ax.grid(axis="y", alpha=0.3)
        if col == 0:
            ax.set_ylabel("TFLOPS")
        if row == 1:
            ax.set_xlabel("seqlen")
        if row == 0 and col == 0:
            ax.legend(loc="upper left")

# Give each row a little headroom so the annotations above tallest bar fit.
for row in range(2):
    ymax = max(axes[row, c].get_ylim()[1] for c in range(4))
    for c in range(4):
        axes[row, c].set_ylim(0, ymax * 1.15)

fig.suptitle(
    "fp32 Forward on B200 — FA4 (TF32 MMA) vs PyTorch ref (TF32) — B=2, H=8\n"
    "(D=48 unsupported: TF32 swizzle requires head_dim*4B divisible by 128; "
    "bwd not yet implemented in fp32)",
    y=1.00,
)
fig.tight_layout()
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print(f"saved: {OUT}")
