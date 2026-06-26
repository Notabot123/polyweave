"""Tier-2b ablation: frozen-random experts vs exact structured experts.

Replaces DifferentiableSieve and BinomialExpansion with frozen random modules
of zero learnable parameters but no algorithmic structure.
The router and DualHeadMLP are identical to the main experiment.

Key question: does the router collapse to any frozen branch, or only the one
that encodes the correct algorithm?

Expected result: with random frozen experts the router relies on the MLP,
achieving only MLP-level accuracy (~75% primality, near-0% exact binomial).

Run:
    python research/paper3/run_tier2_ablation.py
    python research/paper3/run_tier2_ablation.py --smoke
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from polyweave.maths import DifferentiableSieve, BinomialExpansion

# Re-use helpers from the multi-expert script
from run_tier2_multi_expert import (
    make_primality_data, make_binomial_data, DualHeadMLP, MultiRouter,
    eval_primality, eval_binomial_top1, routing_confusion, train_multi_expert,
)

HERE        = Path(__file__).parent
RESULTS_DIR = HERE / "results"
PLOTS_DIR   = HERE / "plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

NUM_ROWS = 10


# ---------------------------------------------------------------------------
# Random frozen expert modules (same interface as structured, no algorithm)
# ---------------------------------------------------------------------------

class RandomSieve(nn.Module):
    """Frozen random primality scores — same interface as DifferentiableSieve."""

    def __init__(self, N: int, seed: int = 42):
        super().__init__()
        g = torch.Generator(); g.manual_seed(seed)
        scores = torch.rand(N + 1, generator=g)
        self.register_buffer("scores", scores)

    def forward(self) -> torch.Tensor:
        return self.scores


class RandomBinom(nn.Module):
    """Frozen random coefficient vectors — same interface as BinomialExpansion."""

    def __init__(self, num_rows: int, n_table: int = 512, seed: int = 99):
        super().__init__()
        self.num_rows = num_rows
        g = torch.Generator(); g.manual_seed(seed)
        table = torch.randn(n_table, num_rows, generator=g)
        self.register_buffer("table", table)

    def forward(self, A: float, B: float, n_exp: int) -> torch.Tensor:
        idx = (int(A) * 37 + int(B) * 17 + int(n_exp) * 7) % len(self.table)
        return self.table[idx]


# ---------------------------------------------------------------------------
# Ablation model (same as MultiExpertMoE but with random experts)
# ---------------------------------------------------------------------------

class AblationMoE(nn.Module):
    def __init__(self, prime_n_max: int, num_rows: int, hidden: int = 64):
        super().__init__()
        self.sieve   = RandomSieve(prime_n_max)
        self.binom   = RandomBinom(num_rows)
        self.mlp     = DualHeadMLP(prime_n_max, num_rows, hidden=hidden)
        self.router  = MultiRouter(hidden=32)
        self.num_rows = num_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(smoke: bool = False) -> dict:
    PRIME_TRAIN_MAX = 100  if not smoke else 40
    PRIME_EVAL_MAX  = 200  if not smoke else 80
    A_VALS_TRAIN    = [1, 2, 3, 4] if not smoke else [1, 2]
    B_VALS_TRAIN    = [1, 2, 3, 4] if not smoke else [1, 2]
    N_EXP_TRAIN     = [2, 3, 4, 5] if not smoke else [2, 3]
    N_EXP_OOD       = [6, 7]       if not smoke else [4, 5]
    EPOCHS          = 600  if not smoke else 80
    LOG_EVERY       = 100  if not smoke else 20
    HIDDEN          = 64   if not smoke else 32

    binom_ref  = BinomialExpansion(NUM_ROWS)
    all_coeffs = torch.stack([
        binom_ref(float(A), float(B), int(n))
        for A in A_VALS_TRAIN for B in B_VALS_TRAIN for n in N_EXP_TRAIN
    ])
    coeff_scale = float(all_coeffs.abs().max())

    ns_tr,  feats_p_tr,  lbl_p_tr  = make_primality_data(2, PRIME_TRAIN_MAX)
    ns_ood, feats_p_ood, lbl_p_ood = make_primality_data(PRIME_TRAIN_MAX + 1, PRIME_EVAL_MAX)

    As_tr,  Bs_tr,  Ns_tr,  feats_b_tr,  coeffs_tr  = make_binomial_data(
        A_VALS_TRAIN, B_VALS_TRAIN, N_EXP_TRAIN, binom_ref, coeff_scale)
    As_ood, Bs_ood, Ns_ood, feats_b_ood, coeffs_ood = make_binomial_data(
        A_VALS_TRAIN, B_VALS_TRAIN, N_EXP_OOD, binom_ref, coeff_scale)

    model = AblationMoE(PRIME_TRAIN_MAX, NUM_ROWS, hidden=HIDDEN)

    print("Training ablation MoE (random frozen experts)...")
    hist = train_multi_expert(
        model,
        prime_data=(ns_tr, feats_p_tr, lbl_p_tr),
        binom_data=(As_tr, Bs_tr, Ns_tr, feats_b_tr, coeffs_tr),
        coeff_scale=coeff_scale,
        epochs=EPOCHS,
        lr=1e-3,
        batch_size=32,
        lambda_binom=1.0,
        log_every=LOG_EVERY,
    )

    sieve_eval = DifferentiableSieve(PRIME_EVAL_MAX)
    ss = sieve_eval()

    conf_mat = routing_confusion(model, feats_p_tr, feats_b_tr)

    p_tr  = eval_primality(model, ns_tr,  lbl_p_tr,  PRIME_TRAIN_MAX, ss)
    p_ood = eval_primality(model, ns_ood, lbl_p_ood, PRIME_TRAIN_MAX, ss)
    b_tr,  frac_tr  = eval_binomial_top1(model, As_tr,  Bs_tr,  Ns_tr,  feats_b_tr,  coeff_scale)
    b_ood, frac_ood = eval_binomial_top1(model, As_ood, Bs_ood, Ns_ood, feats_b_ood, coeff_scale)

    print("\n=== Ablation Results (random frozen experts) ===")
    print(f"\nPrimality: train={p_tr:.1%}  OOD={p_ood:.1%}")
    print(f"Binomial top-1: train={b_tr:.1%}  OOD={b_ood:.1%}  "
          f"(frac to random binom: {frac_tr:.3f})")
    print(f"\nRouting confusion matrix:")
    print(f"  {'':14s}  {'Sieve':>8}  {'BinomExp':>8}  {'MLP':>8}")
    for task, row in [("Primality", conf_mat[0]), ("Binomial", conf_mat[1])]:
        print(f"  {task:14s}  {row[0]:8.3f}  {row[1]:8.3f}  {row[2]:8.3f}")

    results = {
        "ablation": "random_frozen_experts",
        "primality": {"train": round(p_tr, 4), "ood": round(p_ood, 4)},
        "binomial_top1": {
            "train": round(b_tr, 4), "ood": round(b_ood, 4),
            "frac_to_rand_binom": round(frac_tr, 4),
        },
        "routing_confusion": {
            "primality": {k: round(float(v), 4)
                          for k, v in zip(["sieve", "binom", "mlp"], conf_mat[0])},
            "binomial":  {k: round(float(v), 4)
                          for k, v in zip(["sieve", "binom", "mlp"], conf_mat[1])},
        },
    }

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    im = ax.imshow(conf_mat, vmin=0, vmax=1, cmap="Oranges", aspect="auto")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Rand.Sieve", "Rand.Binom", "MLP"])
    ax.set_yticks([0, 1]);    ax.set_yticklabels(["Primality", "Binomial"])
    ax.set_xlabel("Expert"); ax.set_ylabel("Task type")
    ax.set_title("Ablation: routing confusion matrix\n(random frozen experts)")
    for i in range(2):
        for j in range(3):
            ax.text(j, i, f"{conf_mat[i, j]:.3f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if conf_mat[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.03)

    ax2 = axes[1]
    labels_bar = ["Primality\n(train)", "Primality\n(OOD)", "Binomial\n(train)", "Binomial\n(OOD)"]
    exact_accs = [p_tr * 100, p_ood * 100, b_tr * 100, b_ood * 100]
    colors = ["#e87c4c", "#e8a04c", "#e87c4c", "#e8a04c"]
    bars = ax2.bar(range(4), exact_accs, color=colors)
    for bar, val in zip(bars, exact_accs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    ax2.axhline(100, color="k", lw=0.8, ls="--", alpha=0.4, label="Exact structured (ref)")
    ax2.set_xticks(range(4)); ax2.set_xticklabels(labels_bar, fontsize=8)
    ax2.set_ylim(0, 115); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Ablation accuracy\n(random frozen experts)")
    ax2.legend(fontsize=7)

    fig.suptitle("Ablation: MoE with random frozen experts instead of algorithmic modules")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "paper3_tier2_ablation.png", dpi=150)
    fig.savefig(PLOTS_DIR / "paper3_tier2_ablation.pdf")
    plt.close(fig)

    out_path = RESULTS_DIR / "paper3_tier2_ablation.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nPlot -> {PLOTS_DIR}/paper3_tier2_ablation.png")
    print(f"JSON -> {out_path}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    run(smoke=args.smoke)
