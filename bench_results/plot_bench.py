"""Parse benchmark_attn.py output and plot fwd / bwd / fwd+bwd TFLOPS."""
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

LOG = Path(sys.argv[1] if len(sys.argv) > 1 else "bench_fwdbwd.txt")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "bench_fwdbwd.png")

text = LOG.read_text()
BACKENDS = ["Standard", "cuDNN", "FA4"]
COLORS = {"Standard": "#999999", "cuDNN": "#4C9BE8", "FA4": "#E8704C"}


def parse_section(tag):
    m = re.search(rf"{tag} \(ms.*?\n=+\n.*?\n-+\n(.*?)(?=\n=|\Z)", text, re.S)
    if not m:
        return None, None
    rows = [r for r in m.group(1).strip().splitlines() if r.strip()]
    seqlens, data = [], {}
    for row in rows:
        parts = row.split()
        seqlens.append(int(parts[3]))
        for name, cell in zip(BACKENDS, parts[4:]):
            ms, tflops, _ = cell.split("/")
            data.setdefault(name, []).append((float(ms), float(tflops)))
    return seqlens, data


sfwd, fwd = parse_section("FWD")
sbwd, bwd = parse_section("BWD")
assert sfwd == sbwd
seqlens = sfwd

combined = {}
for name in BACKENDS:
    combined[name] = [
        (fm + bm, (fm * ft + bm * bt) / (fm + bm))
        for (fm, ft), (bm, bt) in zip(fwd[name], bwd[name])
    ]


def get_idx(d, name, i):
    return [v[i] for v in d[name]]


fig, axes = plt.subplots(2, 3, figsize=(16, 9))
titles = ["Forward", "Backward", "Forward + Backward"]
sources = [fwd, bwd, combined]
x = np.arange(len(seqlens))
w = 0.27
labels = [f"{s // 1024}k" if s >= 1024 else str(s) for s in seqlens]

# Row 0: TFLOPS (index 1). Row 1: ms (index 0, log scale).
for row, (metric_idx, ylabel, log) in enumerate([
    (1, "TFLOPS", False),
    (0, "Time (ms)", True),
]):
    row_axes = axes[row]
    # share y within row
    for ax in row_axes[1:]:
        ax.sharey(row_axes[0])
    for ax, title, src in zip(row_axes, titles, sources):
        for i, name in enumerate(BACKENDS):
            vals = get_idx(src, name, metric_idx)
            bars = ax.bar(x + (i - 1) * w, vals, w,
                          label=name, color=COLORS[name])
            for b, v in zip(bars, vals):
                txt = f"{v:.0f}" if metric_idx == 1 else (
                    f"{v:.2f}" if v < 10 else f"{v:.1f}")
                ax.text(b.get_x() + b.get_width() / 2, v,
                        txt, ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Sequence length")
        if row == 0:
            ax.set_title(title)
            ax.axhline(2250, ls="--", color="k", lw=0.6, alpha=0.4)
        if log:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3, which="both")
    row_axes[0].set_ylabel(ylabel)

axes[0, 0].set_ylim(0, 2400)
axes[0, -1].text(x[-1] + 0.5, 2260, "B200 fp16 peak ~2.25 PFLOPS",
                 fontsize=7.5, ha="right")
axes[0, 0].legend(loc="upper left")
mcfg = re.search(r"hdim=(\d+),.*?nheads=(\d+)(?:,\s*nheads_kv=(\d+))?", text)
causal = "causal" if "causal=True" in text else "non-causal"
hdim, nh, nhkv = mcfg.group(1), mcfg.group(2), mcfg.group(3) or mcfg.group(2)
gqa = f", {nh}Q/{nhkv}KV" if nh != nhkv else f", {nh} heads"
fig.suptitle(f"FlashAttention on B200 — fp16, hdim={hdim}, {causal}{gqa}",
             y=1.00, fontsize=12)
fig.tight_layout()
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print(f"saved: {OUT}")
