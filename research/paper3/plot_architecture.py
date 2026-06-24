"""Figure 1 — MoE architecture diagram for Paper 3.

Saves:
    research/paper3/plots/paper3_architecture.pdf
    research/paper3/plots/paper3_architecture.png

Run:
    python research/paper3/plot_architecture.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PLOTS = Path(__file__).parent / "plots"
PLOTS.mkdir(exist_ok=True)

# ── colour palette ─────────────────────────────────────────────────────────
C_SIEVE   = "#2196F3"   # blue  — active structured expert
C_BLIND   = "#FF9800"   # amber — learned expert
C_ROUTER  = "#4CAF50"   # green — router
C_PASSIVE = "#9E9E9E"   # grey  — passive module
C_FUTURE  = "#CE93D8"   # lavender — future expert
C_POOL    = "#E3F2FD"   # light blue — module pool background
C_OUT     = "#F5F5F5"   # near-white — output node

def box(ax, x, y, w, h, label, sublabel=None,
        fc="#FFFFFF", ec="#333333", lw=1.5, ls="-",
        fontsize=8, alpha=1.0, radius=0.04):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad={radius}",
                          facecolor=fc, edgecolor=ec,
                          linewidth=lw, linestyle=ls, alpha=alpha,
                          zorder=3)
    ax.add_patch(rect)
    dy = 0.03 if sublabel else 0
    ax.text(x, y + dy, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x, y - 0.07, sublabel, ha="center", va="center",
                fontsize=6.5, color="#555555", zorder=4)

def arrow(ax, x0, y0, x1, y1, color="#333333", lw=1.2,
          ls="-", arrowstyle="-|>", mutation_scale=10):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=arrowstyle, color=color,
                                lw=lw, linestyle=ls),
                zorder=2)

def label(ax, x, y, text, fontsize=7, color="#333333", ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize,
            color=color, zorder=5)

# ── canvas ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 10); ax.set_ylim(0, 5)
ax.axis("off")

# ── polyweave.maths pool background ─────────────────────────────────────────
pool_rect = FancyBboxPatch((0.3, 1.5), 3.2, 2.8,
                           boxstyle="round,pad=0.05",
                           facecolor=C_POOL, edgecolor=C_SIEVE,
                           linewidth=1.0, linestyle="--", zorder=1)
ax.add_patch(pool_rect)
ax.text(1.9, 4.45, "polyweave.maths", ha="center", fontsize=7.5,
        color=C_SIEVE, fontstyle="italic", zorder=5)

# ── module pool: DifferentiableSieve (active) ────────────────────────────────
box(ax, 1.9, 3.7, 2.4, 0.55,
    "DifferentiableSieve", "0 params  |  exact  |  frozen",
    fc="#BBDEFB", ec=C_SIEVE, lw=2.0, fontsize=8)

# ── module pool: BinomialExpansion (passive) ─────────────────────────────────
box(ax, 1.9, 2.85, 2.4, 0.55,
    "BinomialExpansion", "0 params  |  exact  |  frozen",
    fc="#F5F5F5", ec=C_PASSIVE, lw=1.2, ls="--",
    fontsize=8, alpha=0.85)

# ── module pool: PascalTriangle (passive, small) ─────────────────────────────
box(ax, 1.9, 2.1, 2.4, 0.45,
    "PascalTriangle / BernoulliTriangle", "combinatorial primitives  |  frozen",
    fc="#F5F5F5", ec=C_PASSIVE, lw=1.0, ls="--",
    fontsize=7, alpha=0.75)

# ── input node ───────────────────────────────────────────────────────────────
box(ax, 1.9, 0.65, 1.4, 0.45, "Input  n",
    fc="#FFFDE7", ec="#795548", lw=1.5, fontsize=8)

# ── arrow: input -> sieve ────────────────────────────────────────────────────
arrow(ax, 1.9, 0.88, 1.9, 3.42, color=C_SIEVE)
label(ax, 2.28, 2.15, "query", fontsize=6.5, color=C_SIEVE, ha="left")

# ── BlindMLP (learned expert) ────────────────────────────────────────────────
box(ax, 5.8, 3.7, 2.0, 0.55,
    "BlindMLP", "learned  |  2 × 64 units",
    fc="#FFF3E0", ec=C_BLIND, lw=2.0, fontsize=8)

# ── arrow: input -> BlindMLP ─────────────────────────────────────────────────
arrow(ax, 2.6, 0.65, 5.8, 3.42, color=C_BLIND)

# ── Router ───────────────────────────────────────────────────────────────────
box(ax, 5.8, 2.55, 2.0, 0.55,
    "Router", "learned  |  softmax  (w₀, w₁)",
    fc="#E8F5E9", ec=C_ROUTER, lw=2.0, fontsize=8)

# ── arrow: input -> router ───────────────────────────────────────────────────
arrow(ax, 2.6, 0.68, 5.8, 2.27, color=C_ROUTER)

# ── weighted sum node ────────────────────────────────────────────────────────
box(ax, 8.1, 3.15, 1.2, 0.5, "Σ  (weighted)",
    fc=C_OUT, ec="#333333", lw=1.5, fontsize=7.5)

# ── arrows: sieve -> sum ─────────────────────────────────────────────────────
arrow(ax, 3.1, 3.7, 7.5, 3.28, color=C_SIEVE)
label(ax, 5.3, 3.65, "ŷ_sieve · w₀", fontsize=6.5, color=C_SIEVE)

# ── arrows: blind -> sum ─────────────────────────────────────────────────────
arrow(ax, 6.8, 3.7, 7.5, 3.22, color=C_BLIND)
label(ax, 7.12, 3.62, "ŷ_blind · w₁", fontsize=6.0, color=C_BLIND, ha="left")

# ── arrows: router -> sum ────────────────────────────────────────────────────
arrow(ax, 6.8, 2.55, 7.5, 3.05, color=C_ROUTER)
label(ax, 7.22, 2.72, "(w₀, w₁)", fontsize=6.0, color=C_ROUTER, ha="left")

# ── output node ──────────────────────────────────────────────────────────────
box(ax, 8.1, 4.25, 1.2, 0.45, "Output ŷ",
    fc="#FFFDE7", ec="#333333", lw=1.5, fontsize=8)
arrow(ax, 8.1, 3.4, 8.1, 4.0, color="#333333")

# ── ForwardChainer (future expert, dashed) ───────────────────────────────────
box(ax, 5.8, 1.35, 2.0, 0.55,
    "ForwardChainer", "polyweave.reasoning  |  future",
    fc="#F3E5F5", ec=C_FUTURE, lw=1.2, ls="--",
    fontsize=7.5, alpha=0.80)
arrow(ax, 2.6, 0.62, 5.8, 1.07, color=C_FUTURE, ls="dashed")
label(ax, 4.1, 0.72, "future", fontsize=6, color=C_FUTURE)

# ── legend ───────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(fc="#BBDEFB", ec=C_SIEVE,   lw=2, label="Active structured expert (frozen)"),
    mpatches.Patch(fc="#F5F5F5", ec=C_PASSIVE, lw=1, label="Passive module (pool, unused in MoE)"),
    mpatches.Patch(fc="#FFF3E0", ec=C_BLIND,   lw=2, label="Learned expert"),
    mpatches.Patch(fc="#E8F5E9", ec=C_ROUTER,  lw=2, label="Learned router"),
    mpatches.Patch(fc="#F3E5F5", ec=C_FUTURE,  lw=1, label="Future expert (already in library)"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=6.5,
          framealpha=0.9, edgecolor="#CCCCCC", ncol=1,
          bbox_to_anchor=(0.0, 0.0))

fig.suptitle(
    "polyweave.maths module pool and Tier-2 MoE architecture",
    fontsize=10, y=0.97
)
fig.tight_layout()

out_pdf = PLOTS / "paper3_architecture.pdf"
out_png = PLOTS / "paper3_architecture.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out_pdf}")
print(f"Saved -> {out_png}")
