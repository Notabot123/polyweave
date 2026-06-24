"""Tier-2b experiment — Multi-expert routing with structured and learned experts.

Three experts compete for each query:
  0. DifferentiableSieve  — exact primality      (frozen, 0 params)
  1. BinomialExpansion    — exact coefficients   (frozen, 0 params)
  2. DualHeadMLP          — learned residual for both tasks

A single Router MLP (input: 6-dim query feature) outputs softmax weights over all
three experts. The router receives no direct supervision about expert identity; it must
discover the correct allocation purely from the task losses.

Task 1 — Primality:   n in [2, PRIME_TRAIN_MAX],    label = is_prime(n)
Task 2 — Binomial:    (A, B, n_exp) sweep,           label = coefficient vector

Key result: routing confusion matrix showing the router allocates high weight to the
correct structured expert for each task type — without being told which expert handles
which task.

Run:
    python research/paper3/run_tier2_multi_expert.py
    python research/paper3/run_tier2_multi_expert.py --smoke
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

HERE        = Path(__file__).parent
RESULTS_DIR = HERE / "results"
PLOTS_DIR   = HERE / "plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

NUM_ROWS = 10   # BinomialExpansion rows; supports n_exp up to 9


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _true_primes(N: int) -> torch.Tensor:
    is_p = [True] * (N + 1)
    is_p[0] = is_p[1] = False
    for p in range(2, int(math.isqrt(N)) + 1):
        if is_p[p]:
            for m in range(p * p, N + 1, p):
                is_p[m] = False
    return torch.tensor(is_p, dtype=torch.float32)


def make_primality_data(n_min: int, n_max: int):
    """Feature vector: [1, 0, n/n_max, 0, 0, 0]."""
    ns      = torch.arange(n_min, n_max + 1)
    labels  = _true_primes(n_max)[n_min:]
    feats   = torch.zeros(len(ns), 6)
    feats[:, 0] = 1.0
    feats[:, 2] = ns.float() / n_max
    return ns, feats, labels


def make_binomial_data(A_vals, B_vals, n_exp_vals, binom_module: BinomialExpansion,
                       coeff_scale: float):
    """Feature vector: [0, 1, 0, A/max_A, B/max_B, n_exp/max_n]."""
    max_A   = max(A_vals)
    max_B   = max(B_vals)
    max_n   = max(n_exp_vals)
    rows_As, rows_Bs, rows_Ns, feats_list, coeffs_list = [], [], [], [], []
    for A in A_vals:
        for B in B_vals:
            for n_exp in n_exp_vals:
                c = binom_module(float(A), float(B), int(n_exp))
                rows_As.append(A); rows_Bs.append(B); rows_Ns.append(n_exp)
                feat = torch.tensor([0., 1., 0., A / max_A, B / max_B, n_exp / max_n])
                feats_list.append(feat)
                coeffs_list.append(c / coeff_scale)
    As    = torch.tensor(rows_As, dtype=torch.float32)
    Bs    = torch.tensor(rows_Bs, dtype=torch.float32)
    Ns    = torch.tensor(rows_Ns, dtype=torch.float32)
    feats = torch.stack(feats_list)
    coeffs_norm = torch.stack(coeffs_list)
    return As, Bs, Ns, feats, coeffs_norm


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class DualHeadMLP(nn.Module):
    """Learned residual expert — separate heads for primality and binomial tasks."""

    def __init__(self, prime_n_max: int, num_rows: int, hidden: int = 64):
        super().__init__()
        self.prime_n_max = prime_n_max
        # Primality head
        self.prime_enc  = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(),
                                        nn.Linear(hidden, hidden), nn.ReLU())
        self.prime_head = nn.Linear(hidden, 1)
        # Binomial head
        self.binom_enc  = nn.Sequential(nn.Linear(3, hidden), nn.ReLU(),
                                        nn.Linear(hidden, hidden), nn.ReLU())
        self.binom_head = nn.Linear(hidden, num_rows)

    def forward_prime(self, n: torch.Tensor) -> torch.Tensor:
        x = n.float().unsqueeze(-1) / self.prime_n_max
        return torch.sigmoid(self.prime_head(self.prime_enc(x))).squeeze(-1)

    def forward_binom(self, A: torch.Tensor, B: torch.Tensor,
                      N: torch.Tensor) -> torch.Tensor:
        x = torch.stack([A / 8.0, B / 8.0, N / 8.0], dim=-1)
        return self.binom_head(self.binom_enc(x))


class MultiRouter(nn.Module):
    """Softmax router over 3 experts — takes 6-dim query feature."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 3))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(feat), dim=-1)   # (batch, 3)


class MultiExpertMoE(nn.Module):
    """Multi-expert MoE: sieve + binomial_expansion + dual-head MLP + router."""

    def __init__(self, prime_n_max: int, num_rows: int, hidden: int = 64):
        super().__init__()
        self.sieve   = DifferentiableSieve(prime_n_max)
        self.binom   = BinomialExpansion(num_rows)
        self.mlp     = DualHeadMLP(prime_n_max, num_rows, hidden=hidden)
        self.router  = MultiRouter(hidden=32)
        self.num_rows = num_rows


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_multi_expert(
    model: MultiExpertMoE,
    prime_data: tuple,
    binom_data: tuple,
    coeff_scale: float,
    epochs: int,
    lr: float = 1e-3,
    batch_size: int = 32,
    lambda_binom: float = 1.0,
    log_every: int = 100,
) -> dict:
    """Joint training on mixed primality + binomial batches.

    prime_data: (ns, feats, labels)
    binom_data: (As, Bs, Ns, feats, coeffs_norm)
    """
    ns_p, feats_p, lbl_p = prime_data
    As_b, Bs_b, Ns_b, feats_b, coeffs_b = binom_data

    sieve_scores = model.sieve()   # precompute (fixed)

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    history = {"loss": [], "w_sieve_prime": [], "w_binom_prime": [],
               "w_sieve_binom": [], "w_binom_binom": []}

    n_prime = len(ns_p)
    n_binom = len(As_b)

    for ep in range(epochs):
        model.train()
        # --- shuffle both datasets ---
        perm_p = torch.randperm(n_prime)
        perm_b = torch.randperm(n_binom)
        ep_loss = 0.0; n_steps = 0

        steps = max(n_prime, n_binom) // batch_size + 1
        for step in range(steps):
            # sample a mini-batch from each task
            idx_p = perm_p[step * batch_size % n_prime:
                           step * batch_size % n_prime + batch_size]
            idx_b = perm_b[step * batch_size % n_binom:
                           step * batch_size % n_binom + batch_size]

            # ── Primality loss ──────────────────────────────────────────
            n_batch  = ns_p[idx_p]
            f_batch  = feats_p[idx_p]
            l_batch  = lbl_p[idx_p]
            w        = model.router(f_batch)                    # (B, 3)
            y_sieve  = sieve_scores[n_batch].detach()           # (B,) in (0,1)
            y_mlp    = model.mlp.forward_prime(n_batch)         # (B,) in (0,1) via sigmoid
            y_prime  = w[:, 0] * y_sieve + w[:, 2] * y_mlp
            loss_p   = F.binary_cross_entropy(y_prime.clamp(1e-7, 1-1e-7), l_batch)

            # ── Binomial loss ───────────────────────────────────────────
            A_batch  = As_b[idx_b]
            B_batch  = Bs_b[idx_b]
            N_batch  = Ns_b[idx_b]
            f_b_batch = feats_b[idx_b]
            c_batch  = coeffs_b[idx_b]                          # (B, num_rows) normalised
            w_b      = model.router(f_b_batch)                  # (B, 3)
            # structured expert: loop over batch (frozen, no grad)
            with torch.no_grad():
                y_struct = torch.stack([
                    model.binom(float(A), float(B), int(N))
                    for A, B, N in zip(A_batch, B_batch, N_batch)
                ]) / coeff_scale                                 # (B, num_rows) normalised
            y_mlp_b  = model.mlp.forward_binom(A_batch, B_batch, N_batch)
            y_binom  = w_b[:, 1:2] * y_struct + w_b[:, 2:3] * y_mlp_b
            loss_b   = F.mse_loss(y_binom, c_batch)

            loss = loss_p + lambda_binom * loss_b
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); n_steps += 1

        history["loss"].append(ep_loss / max(n_steps, 1))

        # ── Log mean router weights per task type ──────────────────────
        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                w_all_p = model.router(feats_p)      # (N_prime, 3)
                w_all_b = model.router(feats_b)      # (N_binom, 3)
            wp_mean = w_all_p.mean(0)
            wb_mean = w_all_b.mean(0)
            history["w_sieve_prime"].append(wp_mean[0].item())
            history["w_binom_prime"].append(wp_mean[1].item())
            history["w_sieve_binom"].append(wb_mean[0].item())
            history["w_binom_binom"].append(wb_mean[1].item())
            print(f"  ep {ep+1:4d}  loss={history['loss'][-1]:.4f}"
                  f"  prime:[w_s={wp_mean[0]:.3f} w_b={wp_mean[1]:.3f} w_m={wp_mean[2]:.3f}]"
                  f"  binom:[w_s={wb_mean[0]:.3f} w_b={wb_mean[1]:.3f} w_m={wb_mean[2]:.3f}]")
    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_primality(model: MultiExpertMoE, ns: torch.Tensor,
                   labels: torch.Tensor, prime_n_max: int,
                   eval_sieve_scores: torch.Tensor | None = None) -> float:
    """eval_sieve_scores: precomputed scores from a sieve sized to cover all ns."""
    model.eval()
    if eval_sieve_scores is None:
        eval_sieve_scores = model.sieve()
    feats = torch.zeros(len(ns), 6)
    feats[:, 0] = 1.0
    feats[:, 2] = ns.float() / prime_n_max
    with torch.no_grad():
        w       = model.router(feats)
        y_sieve = eval_sieve_scores[ns].detach()
        y_mlp   = model.mlp.forward_prime(ns)
        y       = w[:, 0] * y_sieve + w[:, 2] * y_mlp
        preds   = y > 0.5
    return (preds == labels.bool()).float().mean().item()


def eval_binomial_top1(model: MultiExpertMoE, As: torch.Tensor, Bs: torch.Tensor,
                       Ns: torch.Tensor, feats: torch.Tensor,
                       coeff_scale: float) -> tuple[float, float]:
    """Top-1 (hard) routing: use the argmax expert's output directly.

    Returns (exact_accuracy, fraction_routed_to_binomial_expert).
    When the router correctly selects expert 1 (BinomialExpansion), output is exact.
    """
    model.eval()
    with torch.no_grad():
        w       = model.router(feats)           # (B, 3)
        top1    = w.argmax(dim=-1)              # (B,)
        y_struct = torch.stack([
            model.binom(float(A), float(B), int(N))
            for A, B, N in zip(As, Bs, Ns)
        ])                                      # (B, num_rows) — exact, unscaled
        y_mlp_b = model.mlp.forward_binom(As, Bs, Ns) * coeff_scale  # unscale MLP
        # Select output: expert 1 → structured (exact); else → MLP
        is_binom_expert = (top1 == 1).unsqueeze(-1).float()
        y_out   = is_binom_expert * y_struct + (1 - is_binom_expert) * y_mlp_b
        exact   = ((y_out - y_struct).abs() < 0.5).all(dim=-1).float()
        frac_to_binom = (top1 == 1).float().mean()
    return exact.mean().item(), frac_to_binom.item()


def routing_confusion(model: MultiExpertMoE, feats_p: torch.Tensor,
                      feats_b: torch.Tensor) -> np.ndarray:
    """Returns (2, 3) matrix: rows=task, cols=[sieve, binom, mlp]."""
    model.eval()
    with torch.no_grad():
        w_p = model.router(feats_p).mean(0).numpy()
        w_b = model.router(feats_b).mean(0).numpy()
    return np.stack([w_p, w_b])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(smoke: bool = False) -> dict:
    PRIME_TRAIN_MAX = 100   if not smoke else 40
    PRIME_EVAL_MAX  = 200   if not smoke else 80
    A_VALS_TRAIN    = [1, 2, 3, 4]   if not smoke else [1, 2]
    B_VALS_TRAIN    = [1, 2, 3, 4]   if not smoke else [1, 2]
    N_EXP_TRAIN     = [2, 3, 4, 5]   if not smoke else [2, 3]
    N_EXP_OOD       = [6, 7]         if not smoke else [4, 5]
    EPOCHS          = 600   if not smoke else 80
    LOG_EVERY       = 100   if not smoke else 20
    HIDDEN          = 64    if not smoke else 32

    # ── Structured modules ──────────────────────────────────────────────────
    binom_ref  = BinomialExpansion(NUM_ROWS)
    # coeff_scale: normalise by max coefficient in training set
    all_coeffs = torch.stack([
        binom_ref(float(A), float(B), int(n))
        for A in A_VALS_TRAIN for B in B_VALS_TRAIN for n in N_EXP_TRAIN
    ])
    coeff_scale = float(all_coeffs.abs().max())
    print(f"Coefficient scale (max |coeff| in training set): {coeff_scale:.1f}")

    # ── Datasets ────────────────────────────────────────────────────────────
    ns_tr, feats_p_tr, lbl_p_tr = make_primality_data(2, PRIME_TRAIN_MAX)
    ns_ood, feats_p_ood, lbl_p_ood = make_primality_data(PRIME_TRAIN_MAX + 1, PRIME_EVAL_MAX)
    ns_all, feats_p_all, lbl_p_all = make_primality_data(2, PRIME_EVAL_MAX)

    As_tr, Bs_tr, Ns_tr, feats_b_tr, coeffs_tr = make_binomial_data(
        A_VALS_TRAIN, B_VALS_TRAIN, N_EXP_TRAIN, binom_ref, coeff_scale)
    As_ood, Bs_ood, Ns_ood, feats_b_ood, coeffs_ood = make_binomial_data(
        A_VALS_TRAIN, B_VALS_TRAIN, N_EXP_OOD, binom_ref, coeff_scale)

    print(f"Prime train: {len(ns_tr)} | Binom train: {len(As_tr)} | "
          f"Prime OOD: {len(ns_ood)} | Binom OOD: {len(As_ood)}")

    # ── Model ───────────────────────────────────────────────────────────────
    model = MultiExpertMoE(PRIME_TRAIN_MAX, NUM_ROWS, hidden=HIDDEN)

    print("\nTraining multi-expert MoE...")
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

    # ── Full-range sieve for evaluation (covers PRIME_EVAL_MAX) ────────────
    sieve_standalone = DifferentiableSieve(PRIME_EVAL_MAX)
    ss = sieve_standalone()
    def sieve_acc(ns, lbl):
        return ((ss[ns] > 0.5) == lbl.bool()).float().mean().item()

    # ── Results ─────────────────────────────────────────────────────────────
    conf_mat = routing_confusion(model, feats_p_tr, feats_b_tr)

    results = {
        "prime": {
            "sieve_train":   round(sieve_acc(ns_tr,  lbl_p_tr),  4),
            "sieve_ood":     round(sieve_acc(ns_ood, lbl_p_ood), 4),
            "moe_train":     round(eval_primality(model, ns_tr,  lbl_p_tr,  PRIME_TRAIN_MAX, ss), 4),
            "moe_ood":       round(eval_primality(model, ns_ood, lbl_p_ood, PRIME_TRAIN_MAX, ss), 4),
        },
        "binomial": {
            "structured_exact": 1.0,
            "moe_top1_train_exact": round(eval_binomial_top1(
                model, As_tr, Bs_tr, Ns_tr, feats_b_tr, coeff_scale)[0], 4),
            "moe_top1_train_frac_binom": round(eval_binomial_top1(
                model, As_tr, Bs_tr, Ns_tr, feats_b_tr, coeff_scale)[1], 4),
            "moe_top1_ood_exact": round(eval_binomial_top1(
                model, As_ood, Bs_ood, Ns_ood, feats_b_ood, coeff_scale)[0], 4),
        },
        "routing_confusion": {
            "primality":  {k: round(float(v), 4)
                           for k, v in zip(["sieve", "binom", "mlp"], conf_mat[0])},
            "binomial":   {k: round(float(v), 4)
                           for k, v in zip(["sieve", "binom", "mlp"], conf_mat[1])},
        },
        "params_learned": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    # ── Print ────────────────────────────────────────────────────────────────
    print("\n=== Multi-Expert Routing Results ===")
    print(f"\nPrimality accuracy:")
    print(f"  Sieve (standalone): train={results['prime']['sieve_train']:.1%}  OOD={results['prime']['sieve_ood']:.1%}")
    print(f"  MoE:                train={results['prime']['moe_train']:.1%}    OOD={results['prime']['moe_ood']:.1%}")
    b_top1_tr, b_frac = eval_binomial_top1(model, As_tr, Bs_tr, Ns_tr, feats_b_tr, coeff_scale)
    b_top1_ood, _    = eval_binomial_top1(model, As_ood, Bs_ood, Ns_ood, feats_b_ood, coeff_scale)
    print(f"\nBinomial exact accuracy (top-1 hard routing):")
    print(f"  Structured (exact): train=100.0%")
    print(f"  MoE top-1:          train={b_top1_tr:.1%}   OOD={b_top1_ood:.1%}   "
          f"(frac routed to BinomExp: {b_frac:.3f})")
    print(f"\nRouting confusion matrix (mean gate weights):")
    print(f"  {'':14s}  {'Sieve':>8}  {'BinomExp':>8}  {'MLP':>8}")
    for task, row in [("Primality", conf_mat[0]), ("Binomial", conf_mat[1])]:
        print(f"  {task:14s}  {row[0]:8.3f}  {row[1]:8.3f}  {row[2]:8.3f}")

    # ── Plots ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Confusion matrix heatmap
    ax = axes[0]
    cmat = conf_mat
    im = ax.imshow(cmat, vmin=0, vmax=1, cmap="Greens", aspect="auto")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Sieve", "BinomialExp", "MLP"])
    ax.set_yticks([0, 1]);    ax.set_yticklabels(["Primality", "Binomial"])
    ax.set_xlabel("Expert"); ax.set_ylabel("Task type")
    ax.set_title("Routing confusion matrix\n(mean gate weight per task)")
    for i in range(2):
        for j in range(3):
            ax.text(j, i, f"{cmat[i, j]:.3f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if cmat[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.03)

    # Router weight evolution per task type
    ax2 = axes[1]
    log_epochs = list(range(LOG_EVERY, EPOCHS + 1, LOG_EVERY))
    if len(log_epochs) < len(hist["w_sieve_prime"]):
        log_epochs.append(EPOCHS)
    log_epochs = log_epochs[:len(hist["w_sieve_prime"])]
    ax2.plot(log_epochs, hist["w_sieve_prime"], "b-o",  ms=4, label="Primality  → Sieve")
    ax2.plot(log_epochs, hist["w_binom_prime"], "b--s", ms=4, label="Primality  → BinomExp")
    ax2.plot(log_epochs, hist["w_sieve_binom"], "g--^", ms=4, label="Binomial   → Sieve")
    ax2.plot(log_epochs, hist["w_binom_binom"], "g-o",  ms=4, label="Binomial   → BinomExp")
    ax2.axhline(0.5, color="k", lw=0.8, ls="--")
    ax2.set_ylim(0, 1); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Mean gate weight")
    ax2.set_title("Router weight evolution\nper task type")
    ax2.legend(fontsize=7)

    fig.suptitle("Tier-2b: Multi-expert MoE routing (sieve + binomial + MLP)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "paper3_tier2_multi_expert.png", dpi=150)
    fig.savefig(PLOTS_DIR / "paper3_tier2_multi_expert.pdf")
    plt.close(fig)

    out_path = RESULTS_DIR / "paper3_tier2_multi_expert.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPlot -> {PLOTS_DIR}/paper3_tier2_multi_expert.png")
    print(f"JSON -> {out_path}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    run(smoke=args.smoke)
