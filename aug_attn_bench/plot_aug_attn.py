"""Plot fused vs split vs qk_only vs bias_only breakdown from bench.py output.

Usage:
    python plot_aug_attn.py bench_aug_attn.txt bench_aug_attn.png
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG = Path(sys.argv[1] if len(sys.argv) > 1 else "bench_aug_attn.txt")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "bench_aug_attn.png")

text = LOG.read_text()

# Order matters: legend & bar order use this list.
BACKENDS = ["fused", "split", "qk_only", "bias_only"]
COLORS = {
    "fused":     "#E8704C",
    "split":     "#4C9BE8",
    "qk_only":   "#7DAE6A",
    "bias_only": "#C77DD0",
}


def parse_section(tag):
    m = re.search(rf"== {tag} .*?\n.*?\n(.*?)(?=\n==|\Z)", text, re.S)
    if not m:
        return None, None
    rows = [r for r in m.group(1).strip().splitlines() if r.strip()]
    seqlens, data = [], {b: [] for b in BACKENDS}
    for row in rows:
        parts = row.split()
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        seqlens.append(int(parts[0]))
        # Columns after L are: fused, split, qk_only, bias_only.
        for i, b in enumerate(BACKENDS):
            col = 1 + i
            data[b].append(float(parts[col]) if col < len(parts) else float("nan"))
    return seqlens, data


sfwd, fwd = parse_section("FWD")
sbwd, bwd = parse_section("FWDBWD")

# Header metadata.
mgpu = re.search(r"GPU:\s*(.+?)\s*dtype=(\S+)", text)
gpu_name, dtype = (mgpu.group(1), mgpu.group(2)) if mgpu else ("?", "?")
mcfg = re.search(r"A=(\d+)\s+B=(\d+)\s+H=(\d+)\s+D=(\d+)", text)
A, B, H, D = (mcfg.group(i) for i in range(1, 5)) if mcfg else ("?",) * 4
mbackend = re.search(r"QK backend:\s*(\S+)", text)
qk_backend = mbackend.group(1) if mbackend else "FA4"

panels = []
if sfwd:
    panels.append(("Forward", sfwd, fwd))
if sbwd:
    panels.append(("Forward + Backward", sbwd, bwd))

fig, axes = plt.subplots(2, len(panels), figsize=(6.5 * len(panels), 9),
                          squeeze=False)

w = 0.21
for col, (title, seqlens, data) in enumerate(panels):
    x = np.arange(len(seqlens))
    labels = [f"{s // 1024}k" if s >= 1024 else str(s) for s in seqlens]

    # Top: absolute ms.
    ax = axes[0, col]
    for i, name in enumerate(BACKENDS):
        vals = data[name]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=name, color=COLORS[name])
        for b, v in zip(bars, vals):
            txt = f"{v:.2f}" if v < 10 else f"{v:.1f}"
            ax.text(b.get_x() + b.get_width() / 2, v, txt,
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Time (ms)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # Bottom: relative contributions of qk_only and bias_only normalized by fused.
    ax2 = axes[1, col]
    fused_vals = data["fused"]
    qk_rel = [q / f for f, q in zip(fused_vals, data["qk_only"])]
    bias_rel = [b / f for f, b in zip(fused_vals, data["bias_only"])]
    split_rel = [s / f for f, s in zip(fused_vals, data["split"])]
    ax2.bar(x - w, qk_rel, w, color=COLORS["qk_only"], label="qk_only / fused")
    ax2.bar(x, bias_rel, w, color=COLORS["bias_only"], label="bias_only / fused")
    ax2.bar(x + w, split_rel, w, color=COLORS["split"], label="split / fused")
    for xi, (q, b, s) in enumerate(zip(qk_rel, bias_rel, split_rel)):
        ax2.text(xi - w, q, f"{q:.2f}", ha="center", va="bottom", fontsize=7)
        ax2.text(xi, b, f"{b:.2f}", ha="center", va="bottom", fontsize=7)
        ax2.text(xi + w, s, f"{s:.2f}", ha="center", va="bottom", fontsize=7)
    ax2.axhline(1.0, ls="--", color="k", lw=0.6, alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("Sequence length")
    ax2.set_ylabel("ratio (vs fused; <1 = faster)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)

suptitle = (
    f"Augmented Attention Pair Bias: fused vs split, with QK-only / bias-only breakdown\n"
    f"split = {qk_backend}(Q,K,V) + softmax(bias)·V    "
    f"{gpu_name} — {dtype}, A={A}, B={B}, H={H}, head_dim={D}"
)
fig.suptitle(suptitle, y=1.00, fontsize=11)
fig.tight_layout()
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print(f"saved: {OUT}")
