"""Plot fp32 vs bf16 vs fp32_bias FA4 forward benchmark at Btot=48,H=16,D=64."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LOG = Path(__file__).with_name("bench_fp32_bf16_fp32bias_B48_H16D64.txt")
OUT = Path(__file__).with_name("bench_fp32_bf16_fp32bias_B48_H16D64.png")

rows = []
for line in LOG.read_text().splitlines():
    m = re.match(
        r"\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
        r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
        r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
        r"([\d.]+)x\s+([\d.]+)x",
        line,
    )
    if m:
        A, B, Btot, H, L, D, fp32_ms, bf16_ms, bias_ms, fp32_tf, bf16_tf, bias_tf, _, _ = m.groups()
        rows.append(
            {
                "L": int(L),
                "fp32_ms": float(fp32_ms),
                "bf16_ms": float(bf16_ms),
                "bias_ms": float(bias_ms),
                "fp32_tf": float(fp32_tf),
                "bf16_tf": float(bf16_tf),
                "bias_tf": float(bias_tf),
            }
        )

labels = [f"{row['L'] // 1024}k" for row in rows]
x = np.arange(len(rows))
w = 0.25
colors = {"fp32": "#E8704C", "bf16": "#4C9BE8", "bias": "#6D9F71"}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
ax.bar(x - w, [r["fp32_ms"] for r in rows], w, color=colors["fp32"], label="fp32")
ax.bar(x, [r["bf16_ms"] for r in rows], w, color=colors["bf16"], label="bf16")
ax.bar(x + w, [r["bias_ms"] for r in rows], w, color=colors["bias"], label="fp32_bias")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("seqlen")
ax.set_ylabel("Time (ms, log)")
ax.grid(axis="y", alpha=0.3, which="both")
ax.legend(fontsize=8)

ax = axes[1]
ax.bar(x - w, [r["fp32_tf"] for r in rows], w, color=colors["fp32"], label="fp32")
ax.bar(x, [r["bf16_tf"] for r in rows], w, color=colors["bf16"], label="bf16")
ax.bar(x + w, [r["bias_tf"] for r in rows], w, color=colors["bias"], label="fp32_bias")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("seqlen")
ax.set_ylabel("TFLOPS")
ax.grid(axis="y", alpha=0.3)

fig.suptitle("FA4 forward: fp32 vs bf16 vs fp32_bias, Btot=48 H=16 D=64")
fig.tight_layout()
fig.savefig(OUT, dpi=120, bbox_inches="tight")
print(f"saved: {OUT}")
