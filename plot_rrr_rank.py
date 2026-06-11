"""Re-plot the RRR effective-rank / per-feature-R2 figure from cached JSON (no recompute).

The headline view: global linear recoverability R2_lin (x) vs per-FEATURE median R2 (y),
point size = effective rank (RRR @90%). Points near the diagonal are *broadly* linear
(many features fit, high rank = Type B); points far below the diagonal at high x are
*low-rank / outlier-concentrated* (high variance-weighted R2 from one dominant direction,
most features poorly fit, rank ~1 = Type A). Reads plots/raw/rrr_rank.json; writes
plots/polyweave_rrr_rank_kinds.{pdf,png}.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {"gpt2": ("C0", "GPT-2 (GELU)"),
         "EleutherAI/pythia-160m": ("C1", "Pythia-160m (GELU)"),
         "JackFram/llama-160m": ("C2", "llama-160m (SwiGLU)")}


def main() -> None:
    rows = json.loads(Path("plots/raw/rrr_rank.json").read_text())
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.plot([0, 1], [0, 1], "--", c="0.6", lw=1, zorder=0,
            label="broadly linear (per-feat = global)")
    for name, (c, lab) in STYLE.items():
        rs = [r for r in rows if r["model"] == name]
        if not rs:
            continue
        xs = [r["r2_lin"] for r in rs]
        ys = [r["r2_perfeat_median"] for r in rs]
        sz = [8 + 1.3 * r["eff_rank_90"] for r in rs]   # size ~ effective rank
        ax.scatter(xs, ys, s=sz, c=c, label=lab, alpha=0.7, edgecolor="k", linewidth=0.4)
    # Annotate a couple of archetypes.
    for r in rows:
        tag = None
        if r["model"] == "gpt2" and r["block"] == 2:
            tag = "GPT-2 blk2\n(rank 1, Type A)"
        if r["model"] == "EleutherAI/pythia-160m" and r["block"] == 0:
            tag = "Pythia blk0\n(rank 376, Type B)"
        if tag:
            ax.annotate(tag, (r["r2_lin"], r["r2_perfeat_median"]),
                        fontsize=7, xytext=(-10, -28), textcoords="offset points",
                        ha="center", arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlabel("linear recoverability  R²_lin  (variance-weighted, global)")
    ax.set_ylabel("per-feature median R²")
    ax.set_title("What kind of linearity? (point size ∝ effective rank)\n"
                 "below-diagonal at high R² = low-rank, outlier-concentrated")
    ax.set_xlim(-0.05, 1.02); ax.set_ylim(-0.05, 1.02)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"plots/polyweave_rrr_rank_kinds.{ext}", dpi=150)
    print("saved plots/polyweave_rrr_rank_kinds.{pdf,png}")


if __name__ == "__main__":
    main()
