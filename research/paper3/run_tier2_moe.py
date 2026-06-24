"""Tier-2 experiment — MoE range extrapolation on primality.

Three models compared on n in [2, EVAL_MAX]:
  1. DifferentiableSieve  — zero params, zero training, exact everywhere
  2. Blind MLP            — trained on [2, TRAIN_MAX], evaluated on [TRAIN_MAX+1, EVAL_MAX]
  3. MoE (structured + blind + router) — same training range

Headline figures:
  * Accuracy by range (in-range vs out-of-range)
  * Router weight evolution during training (does it learn to trust the sieve?)
  * Router weights at inference stratified by n

Run:
    python research/paper3/run_tier2_moe.py
    python research/paper3/run_tier2_moe.py --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from polyweave.maths import DifferentiableSieve

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
PLOTS_DIR   = HERE / "plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _true_primes(N: int) -> torch.Tensor:
    is_p = [True] * (N + 1)
    is_p[0] = is_p[1] = False
    for p in range(2, int(math.isqrt(N)) + 1):
        if is_p[p]:
            for m in range(p * p, N + 1, p):
                is_p[m] = False
    return torch.tensor(is_p, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class BlindMLP(nn.Module):
    def __init__(self, N: int, hidden: int = 64):
        super().__init__()
        self.N = N
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, n: torch.Tensor) -> torch.Tensor:
        return self.net((n.float() / self.N).unsqueeze(-1)).squeeze(-1)


class Router(nn.Module):
    def __init__(self, N: int, hidden: int = 32):
        super().__init__()
        self.N = N
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, n: torch.Tensor) -> torch.Tensor:
        x = (n.float() / self.N).unsqueeze(-1)
        return F.softmax(self.net(x), dim=-1)  # (batch, 2)


class MoEPrimalityModel(nn.Module):
    """Structured sieve expert + blind MLP expert + learned router.

    The sieve expert contributes a fixed primality score (no gradient flows back
    into it).  The blind expert and router are fully learned.

    Expert 0 = structured sieve.
    Expert 1 = blind MLP.
    """

    def __init__(self, N: int, hidden: int = 64):
        super().__init__()
        self.N = N
        self.sieve = DifferentiableSieve(N)
        self.blind = BlindMLP(N, hidden=hidden)
        self.router = Router(N, hidden=32)

    def forward(self, n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.router(n)                           # (batch, 2)
        y_sieve = self.sieve()[n].detach()           # (batch,) — no grad through sieve
        y_blind = self.blind(n)                      # (batch,)
        y = w[:, 0] * y_sieve + w[:, 1] * y_blind   # weighted mix
        return y, w


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model: nn.Module, ns: torch.Tensor, labels: torch.Tensor,
          epochs: int, lr: float = 1e-3, batch: int = 64,
          log_every: int = 50) -> dict:
    """Train model, returning loss history and per-epoch mean router weights (MoE only)."""
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    idx = torch.arange(len(ns))
    losses, router_w0 = [], []

    for ep in range(epochs):
        model.train()
        perm = idx[torch.randperm(len(idx))]
        ep_loss = 0.0; n_b = 0; ep_w0 = []
        for i in range(0, len(perm), batch):
            b = perm[i : i + batch]
            out = model(ns[b])
            if isinstance(out, tuple):
                logits, w = out
                ep_w0.append(w[:, 0].mean().item())
            else:
                logits = out
            loss = F.binary_cross_entropy_with_logits(logits, labels[b])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); n_b += 1
        losses.append(ep_loss / max(n_b, 1))
        if ep_w0:
            router_w0.append(sum(ep_w0) / len(ep_w0))
        if (ep + 1) % log_every == 0:
            print(f"  ep {ep+1:4d}/{epochs}  loss={losses[-1]:.4f}"
                  + (f"  router[sieve]={router_w0[-1]:.3f}" if ep_w0 else ""))

    return {"losses": losses, "router_w0": router_w0}


def accuracy(model: nn.Module, ns: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        out = model(ns)
        logits = out[0] if isinstance(out, tuple) else out
        preds = torch.sigmoid(logits) > 0.5
    return (preds == labels.bool()).float().mean().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(smoke: bool = False) -> dict:
    TRAIN_MAX = 100  if not smoke else 30
    EVAL_MAX  = 500  if not smoke else 120
    EPOCHS    = 400  if not smoke else 60
    HIDDEN    = 64   if not smoke else 32
    LOG_EVERY = 100  if not smoke else 20

    labels_all = _true_primes(EVAL_MAX)
    ns_train   = torch.arange(2, TRAIN_MAX + 1)
    lbl_train  = labels_all[2 : TRAIN_MAX + 1]
    ns_ood     = torch.arange(TRAIN_MAX + 1, EVAL_MAX + 1)
    lbl_ood    = labels_all[TRAIN_MAX + 1 :]
    ns_all     = torch.arange(2, EVAL_MAX + 1)
    lbl_all    = labels_all[2 :]

    # --- Structured module (no training) ---
    sieve = DifferentiableSieve(EVAL_MAX)
    sieve_scores = sieve()
    def sieve_acc(ns, lbl):
        preds = sieve_scores[ns] > 0.5
        return (preds == lbl.bool()).float().mean().item()

    # --- Blind MLP ---
    print("Training blind MLP...")
    blind = BlindMLP(EVAL_MAX, hidden=HIDDEN)
    t0 = time.time()
    blind_hist = train(blind, ns_train, lbl_train, epochs=EPOCHS, log_every=LOG_EVERY)
    blind_time = time.time() - t0

    # --- MoE ---
    print("\nTraining MoE...")
    moe = MoEPrimalityModel(EVAL_MAX, hidden=HIDDEN)
    t1 = time.time()
    moe_hist = train(moe, ns_train, lbl_train, epochs=EPOCHS, log_every=LOG_EVERY)
    moe_time = time.time() - t1

    # --- Evaluate ---
    results = {
        "train_max": TRAIN_MAX, "eval_max": EVAL_MAX, "epochs": EPOCHS,
        "sieve": {
            "acc_train": round(sieve_acc(ns_train, lbl_train), 4),
            "acc_ood":   round(sieve_acc(ns_ood,   lbl_ood),   4),
            "acc_all":   round(sieve_acc(ns_all,   lbl_all),   4),
            "params": 0,
        },
        "blind_mlp": {
            "acc_train": round(accuracy(blind, ns_train, lbl_train), 4),
            "acc_ood":   round(accuracy(blind, ns_ood,   lbl_ood),   4),
            "acc_all":   round(accuracy(blind, ns_all,   lbl_all),   4),
            "params":    sum(p.numel() for p in blind.parameters()),
            "train_sec": round(blind_time, 2),
        },
        "moe": {
            "acc_train": round(accuracy(moe, ns_train, lbl_train), 4),
            "acc_ood":   round(accuracy(moe, ns_ood,   lbl_ood),   4),
            "acc_all":   round(accuracy(moe, ns_all,   lbl_all),   4),
            "params":    sum(p.numel() for p in moe.parameters() if p.requires_grad),
            "train_sec": round(moe_time, 2),
            "final_router_w_sieve": round(moe_hist["router_w0"][-1], 4) if moe_hist["router_w0"] else None,
        },
    }

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Router weight evolution
    if moe_hist["router_w0"]:
        axes[0].plot(moe_hist["router_w0"], label="sieve expert weight")
        axes[0].axhline(0.5, color="k", linestyle="--", linewidth=0.8)
        axes[0].set_ylim(0, 1); axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Mean router weight"); axes[0].set_title("Router weight evolution")
        axes[0].legend(fontsize=8)

    # Per-n router weight at inference
    moe.eval()
    with torch.no_grad():
        _, w_all = moe(ns_all)
    w_sieve_all = w_all[:, 0].numpy()
    true_np = lbl_all.numpy()
    axes[1].scatter(ns_all.numpy(), w_sieve_all, s=4, c=true_np, cmap="coolwarm")
    axes[1].axvline(TRAIN_MAX, color="k", linestyle="--", linewidth=0.8, label="train boundary")
    axes[1].set_xlabel("n"); axes[1].set_ylabel("Router weight (sieve expert)")
    axes[1].set_title("Router allocation by n  (red=prime, blue=composite)")
    axes[1].legend(fontsize=7)

    # Accuracy bar chart
    groups = ["Train [2,{}]".format(TRAIN_MAX), "OOD [{},{}]".format(TRAIN_MAX+1, EVAL_MAX)]
    sieve_bars = [results["sieve"]["acc_train"],   results["sieve"]["acc_ood"]]
    blind_bars = [results["blind_mlp"]["acc_train"], results["blind_mlp"]["acc_ood"]]
    moe_bars   = [results["moe"]["acc_train"],       results["moe"]["acc_ood"]]
    x = range(len(groups)); w = 0.25
    axes[2].bar([i - w for i in x], sieve_bars, width=w, label="Sieve (0 params)", color="steelblue")
    axes[2].bar([i     for i in x], moe_bars,   width=w, label="MoE",              color="darkorange")
    axes[2].bar([i + w for i in x], blind_bars, width=w, label="Blind MLP",        color="grey")
    axes[2].set_xticks(list(x)); axes[2].set_xticklabels(groups)
    axes[2].set_ylim(0, 1.05); axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Accuracy: in-range vs out-of-range")
    axes[2].legend(fontsize=8)

    fig.suptitle(f"Tier-2 MoE range extrapolation  (train [2,{TRAIN_MAX}], eval [2,{EVAL_MAX}])")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "paper3_tier2_moe.png", dpi=150)
    fig.savefig(PLOTS_DIR / "paper3_tier2_moe.pdf")
    plt.close(fig)

    out_path = RESULTS_DIR / "paper3_tier2_moe.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Tier-2 MoE Range Extrapolation Results ===")
    print(f"{'':30s} {'Sieve':>8} {'MoE':>8} {'BlindMLP':>10}")
    for split, key in [("In-range", "acc_train"), ("OOD", "acc_ood"), ("All", "acc_all")]:
        print(f"{split:30s} {results['sieve'][key]:>8.1%} {results['moe'][key]:>8.1%} {results['blind_mlp'][key]:>10.1%}")
    print(f"{'Params (learned)':30s} {'0':>8} {results['moe']['params']:>8,} {results['blind_mlp']['params']:>10,}")
    if results["moe"]["final_router_w_sieve"] is not None:
        print(f"{'Final router->sieve weight':30s} {'---':>8} {results['moe']['final_router_w_sieve']:>8.3f}")
    print(f"\nPlot  -> {PLOTS_DIR}/paper3_tier2_moe.png")
    print(f"JSON  -> {out_path}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    run(smoke=args.smoke)
