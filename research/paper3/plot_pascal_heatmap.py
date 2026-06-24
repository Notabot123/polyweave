"""Figure 2 — Pascal triangle log-scale heatmap.

Standalone script; run:
    python research/paper3/plot_pascal_heatmap.py
Saves:
    research/paper3/plots/paper3_pascal_heatmap.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

from polyweave.maths import PascalTriangle

HERE    = Path(__file__).parent
PLOTS   = HERE / "plots"
PLOTS.mkdir(exist_ok=True)

NUM_ROWS = 16

pascal = PascalTriangle(NUM_ROWS)
C = pascal().numpy()          # (NUM_ROWS, NUM_ROWS)

# Mask lower triangle where C == 0 (entries beyond row k don't exist)
masked = np.where(C > 0, C, np.nan)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(
    masked,
    norm=mcolors.LogNorm(vmin=1, vmax=float(np.nanmax(masked))),
    cmap="viridis",
    origin="upper",
    interpolation="nearest",
    aspect="auto",
)
fig.colorbar(im, ax=ax, label="Coefficient value (log scale)")
ax.set_xlabel("Column $k$")
ax.set_ylabel("Row $n$")
ax.set_title(f"Pascal's Triangle ($n \\leq {NUM_ROWS - 1}$, log scale)\nproduced by PascalTriangle module — 0 learnable parameters")

# Annotate a few key values
for n in range(min(NUM_ROWS, 8)):
    for k in range(n + 1):
        v = int(C[n, k])
        ax.text(k, n, str(v), ha="center", va="center",
                fontsize=5.5, color="white" if v > 10 else "black")

fig.tight_layout()
out = PLOTS / "paper3_pascal_heatmap.png"
fig.savefig(out, dpi=150)
fig.savefig(out.with_suffix(".pdf"))
plt.close(fig)
print(f"Saved -> {out} + .pdf")
