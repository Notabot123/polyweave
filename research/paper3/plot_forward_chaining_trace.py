"""Figure: forward-chaining trace for a representative initial fact assignment.

Uses the same KB as run_tier1_forward_chaining.py to show how truth propagates
step-by-step from base facts to derived atoms. Two panels are shown:
  Left  — a fact assignment that exercises a deep chain (b0=T, b3=T, b4=T, b5=T)
  Right — a simpler assignment (b1=T only)

Run:
    python research/paper3/plot_forward_chaining_trace.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

from polyweave.reasoning import PropKB, ForwardChainer
from polyweave.viz.plots import plot_chaining_trace

HERE      = Path(__file__).parent
PLOTS_DIR = HERE / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Rebuild the same KB as the Tier-1 experiment
# ---------------------------------------------------------------------------

BASE    = [f"b{i}" for i in range(8)]
DERIVED = [f"d{i}" for i in range(14)]

RULE_DEFS = [
    # Chain A: b0 -> d0 -> d1 -> d2 -> d3  (depths 1-4)
    (["b0"], "d0"), (["d0"], "d1"), (["d1"], "d2"), (["d2"], "d3"),
    # Chain B: b1 -> d4 -> d5 -> d6  (depths 1-3)
    (["b1"], "d4"), (["d4"], "d5"), (["d5"], "d6"),
    # Chain C: b2 -> d7 -> d8  (depths 1-2)
    (["b2"], "d7"), (["d7"], "d8"),
    # Conjunction chain: b3, b4 -> d9 -> d10 -> d11  (depths 1-3)
    (["b3", "b4"], "d9"), (["d9"], "d10"), (["d10"], "d11"),
    # Cross-chain conjunction: b5, d0 -> d12 -> d13  (depths 2-3)
    (["b5", "d0"], "d12"), (["d12"], "d13"),
    # Alternative depth-1 paths
    (["b6"], "d0"), (["b7"], "d4"),
]

def build_kb():
    kb = PropKB()
    for premises, conclusion in RULE_DEFS:
        kb.add_rule(premises, conclusion)
    for b in BASE:
        kb.add_fact(b)
    return kb


kb      = build_kb()
chainer = ForwardChainer(kb, max_steps=10)

# Ordered fact names for display: bases first, then derived
FACT_NAMES = BASE + DERIVED   # length 22

# ---------------------------------------------------------------------------
# Two representative examples
# ---------------------------------------------------------------------------

# Example 1: deep chains — b0 (chain A depth 1-4) + b3+b4 (conjunction,
#             depth 1-3) + b5 (cross-chain conjunction with d0, depth 2-3)
ex1_true = ["b0", "b3", "b4", "b5"]

# Example 2: single chain B only — b1 -> d4 -> d5 -> d6
ex2_true = ["b1"]

def run_example(true_facts):
    f0 = kb.initial_facts(true_facts)
    _, history = chainer(f0, return_history=True)
    return history   # list of (1, N) tensors


history1 = run_example(ex1_true)
history2 = run_example(ex2_true)

# ---------------------------------------------------------------------------
# Plot: two side-by-side heatmaps using the library helper
# ---------------------------------------------------------------------------

def trace_array(history):
    rows = [h.detach().cpu().numpy().reshape(-1) for h in history]
    return np.vstack(rows)   # (steps, n_facts)

arr1 = trace_array(history1)
arr2 = trace_array(history2)

# Use a nicer display name for each atom
def nice_names(kb, ordered):
    return [n for n in ordered if n in kb._fact_index]

names = nice_names(kb, FACT_NAMES)

fig = plt.figure(figsize=(13, 4.2))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

for ax, arr, title, init in [
    (fig.add_subplot(gs[0]), arr1,
     "Deep chains: $b_0, b_3, b_4, b_5$ asserted",
     ex1_true),
    (fig.add_subplot(gs[1]), arr2,
     "Single chain: $b_1$ asserted",
     ex2_true),
]:
    n_steps, n_facts = arr.shape
    im = ax.imshow(arr, cmap="YlGn", vmin=0.0, vmax=1.0,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(n_facts))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(n_steps))
    step_labels = ["initial"] + [f"step {i}" for i in range(1, n_steps)]
    ax.set_yticklabels(step_labels, fontsize=8)
    ax.grid(False)
    ax.set_title(title, fontsize=9.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cb.set_label("truth value", fontsize=8)
    cb.ax.tick_params(labelsize=7)


fig.suptitle(
    "ForwardChainer trace — truth propagates from base facts (left columns)\n"
    "to derived atoms (right columns) over chaining steps",
    fontsize=9.5,
)
fig.tight_layout()

out_png = PLOTS_DIR / "paper3_tier1_forward_chaining_trace.png"
out_pdf = PLOTS_DIR / "paper3_tier1_forward_chaining_trace.pdf"
fig.savefig(out_png, dpi=150)
fig.savefig(out_pdf)
plt.close(fig)
print(f"Saved -> {out_png}")
print(f"Saved -> {out_pdf}")
