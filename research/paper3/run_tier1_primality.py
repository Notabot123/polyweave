"""Tier-1 experiment — primality detection.

Compares:
  * DifferentiableSieve (zero params, zero training) — the structured module
  * MLP baseline trained on n in [2, TRAIN_MAX] and evaluated on [2, EVAL_MAX]

Key claim: the MLP can learn the *training range* but fails to generalise beyond it;
the sieve is exact everywhere with no training at all.

Run:
    python research/paper3/run_tier1_primality.py
    python research/paper3/run_tier1_primality.py --smoke
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
    """Boolean tensor of length N+1; True at prime positions."""
    is_p = [True] * (N + 1)
    is_p[0] = is_p[1] = False
    for p in range(2, int(math.isqrt(N)) + 1):
        if is_p[p]:
            for m in range(p * p, N + 1, p):
                is_p[m] = False
    return torch.tensor(is_p, dtype=torch.float32)


class PrimalityMLP(nn.Module):
    def __init__(self, hidden: int = 128, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, n: torch.Tensor) -> torch.Tensor:
        return self.net(n.float().unsqueeze(-1)).squeeze(-1)


def train_mlp(model: nn.Module, n_train: torch.Tensor, labels: torch.Tensor,
              epochs: int, lr: float = 1e-3, batch: int = 64) -> list[float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    idx = torch.arange(len(n_train))
    for _ in range(epochs):
        perm = idx[torch.randperm(len(idx))]
        ep_loss = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i : i + batch]
            logits = model(n_train[b])
            loss = F.binary_cross_entropy_with_logits(logits, labels[b])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        losses.append(ep_loss / max(1, len(perm) // batch))
    return losses


def eval_accuracy(model: nn.Module, ns: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(ns)) > 0.5
    return (preds == labels.bool()).float().mean().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(smoke: bool = False) -> dict:
    TRAIN_MAX = 100 if not smoke else 30
    EVAL_MAX  = 500 if not smoke else 80
    EPOCHS    = 300 if not smoke else 40
    HIDDEN    = 128 if not smoke else 32

    labels_all = _true_primes(EVAL_MAX)  # (EVAL_MAX+1,)
    ns_all = torch.arange(2, EVAL_MAX + 1, dtype=torch.float32)
    labels_eval = labels_all[2:]

    # Training set: n in [2, TRAIN_MAX]
    ns_train  = torch.arange(2, TRAIN_MAX + 1, dtype=torch.float32)
    lbl_train = labels_all[2 : TRAIN_MAX + 1]

    # --- Structured module (no training) ---
    sieve = DifferentiableSieve(EVAL_MAX)
    scores = sieve()
    sieve_preds = (scores[2:] > 0.5)
    sieve_acc_all   = (sieve_preds == labels_eval.bool()).float().mean().item()
    sieve_acc_train = (sieve_preds[: TRAIN_MAX - 1] == labels_eval[: TRAIN_MAX - 1].bool()).float().mean().item()
    sieve_acc_ood   = (sieve_preds[TRAIN_MAX - 1 :] == labels_eval[TRAIN_MAX - 1 :].bool()).float().mean().item()

    # --- MLP baseline ---
    mlp = PrimalityMLP(hidden=HIDDEN)
    t0 = time.time()
    train_mlp(mlp, ns_train, lbl_train, epochs=EPOCHS)
    train_time = time.time() - t0

    mlp.eval()
    mlp_acc_all   = eval_accuracy(mlp, ns_all,                     labels_eval)
    mlp_acc_train = eval_accuracy(mlp, ns_train,                   lbl_train)
    mlp_acc_ood   = eval_accuracy(mlp, torch.arange(TRAIN_MAX + 1, EVAL_MAX + 1, dtype=torch.float32),
                                  labels_all[TRAIN_MAX + 1 :])

    results = {
        "train_max": TRAIN_MAX, "eval_max": EVAL_MAX, "epochs": EPOCHS,
        "sieve_acc_all":   round(sieve_acc_all, 4),
        "sieve_acc_train": round(sieve_acc_train, 4),
        "sieve_acc_ood":   round(sieve_acc_ood, 4),
        "mlp_acc_all":     round(mlp_acc_all, 4),
        "mlp_acc_train":   round(mlp_acc_train, 4),
        "mlp_acc_ood":     round(mlp_acc_ood, 4),
        "mlp_params":      sum(p.numel() for p in mlp.parameters()),
        "sieve_params":    0,
        "mlp_train_sec":   round(train_time, 2),
    }

    # --- per-number accuracy plot ---
    mlp.eval()
    with torch.no_grad():
        mlp_scores = torch.sigmoid(mlp(ns_all)).numpy()
    sieve_scores = scores[2:].numpy()
    true_np = labels_eval.numpy()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    x = range(2, EVAL_MAX + 1)
    axes[0].scatter(x, sieve_scores, s=4, c=true_np, cmap="coolwarm", label="sieve score")
    axes[0].axvline(TRAIN_MAX, color="k", linestyle="--", linewidth=0.8, label="train boundary")
    axes[0].set_ylabel("Primality score"); axes[0].set_title("DifferentiableSieve (0 params)")
    axes[0].legend(fontsize=7)

    axes[1].scatter(x, mlp_scores, s=4, c=true_np, cmap="coolwarm", label="MLP score")
    axes[1].axvline(TRAIN_MAX, color="k", linestyle="--", linewidth=0.8, label="train boundary")
    axes[1].set_ylabel("Sigmoid output"); axes[1].set_xlabel("n")
    axes[1].set_title(f"MLP baseline ({results['mlp_params']} params, trained on [2,{TRAIN_MAX}])")
    axes[1].legend(fontsize=7)

    fig.suptitle("Primality detection: structured module vs MLP baseline")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "paper3_tier1_primality.png", dpi=150)
    plt.close(fig)

    out_path = RESULTS_DIR / "paper3_tier1_primality.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Tier-1 Primality Results ===")
    print(f"{'':30s} {'Sieve':>8} {'MLP':>8}")
    print(f"{'Overall accuracy':30s} {sieve_acc_all:8.1%} {mlp_acc_all:8.1%}")
    print(f"In-range [2,{TRAIN_MAX}]{'':13s} {sieve_acc_train:8.1%} {mlp_acc_train:8.1%}")
    print(f"OOD [{TRAIN_MAX+1},{EVAL_MAX}]{'':19s} {sieve_acc_ood:8.1%} {mlp_acc_ood:8.1%}")
    print(f"{'Learnable params':30s} {'0':>8} {results['mlp_params']:>8,}")
    print(f"{'Training time (s)':30s} {'0.00':>8} {train_time:>8.2f}")
    print(f"\nPlot  -> {PLOTS_DIR}/paper3_tier1_primality.png")
    print(f"JSON  -> {out_path}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="Quick smoke test")
    args = p.parse_args()
    run(smoke=args.smoke)
