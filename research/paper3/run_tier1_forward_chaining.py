"""Tier-1: Differentiable Forward Chaining vs MLP baseline.

Knowledge base: 8 base facts (b0..b7), 14 derived facts (d0..d13), 15 rules.
Chains of depth 1-4 plus conjunction rules give natural train/OOD split.

Dataset: all 2^8 = 256 initial fact assignments paired with entailment queries
for each of 14 derived atoms -> 3,584 (facts, query, label) examples.
Ground truth: ForwardChainer (0 parameters, exact by construction).
MLP baseline: trained on queries whose minimum derivation depth <= 2.
OOD: queries requiring derivation depth >= 3.

Run:
    python research/paper3/run_tier1_forward_chaining.py
    python research/paper3/run_tier1_forward_chaining.py --smoke
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from polyweave.reasoning import PropKB, ForwardChainer

HERE        = Path(__file__).parent
RESULTS_DIR = HERE / "results"
PLOTS_DIR   = HERE / "plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Knowledge base definition
# ---------------------------------------------------------------------------

BASE    = [f"b{i}" for i in range(8)]   # b0..b7 — externally assertable
DERIVED = [f"d{i}" for i in range(14)]  # d0..d13 — derived by rules

#  Rules and their minimum chain depth from any base-fact assignment:
#
#  Chain A  (b0 as source, depth 1-4):
#    b0 -> d0  [1]   d0 -> d1  [2]   d1 -> d2  [3]   d2 -> d3  [4]
#  Chain B  (b1, depth 1-3):
#    b1 -> d4  [1]   d4 -> d5  [2]   d5 -> d6  [3]
#  Chain C  (b2, depth 1-2):
#    b2 -> d7  [1]   d7 -> d8  [2]
#  Conjunction chain  ({b3,b4}, depth 1-3):
#    b3,b4 -> d9  [1]   d9 -> d10  [2]   d10 -> d11  [3]
#  Cross-chain conjunction  ({b5,d0}, depth 2-3):
#    b5,d0 -> d12  [2]   d12 -> d13  [3]
#  Alternative depth-1 paths:
#    b6 -> d0  [1]   b7 -> d4  [1]

RULE_DEFS = [
    # Chain A
    (["b0"], "d0"),
    (["d0"], "d1"),
    (["d1"], "d2"),
    (["d2"], "d3"),
    # Chain B
    (["b1"], "d4"),
    (["d4"], "d5"),
    (["d5"], "d6"),
    # Chain C
    (["b2"], "d7"),
    (["d7"], "d8"),
    # Conjunction chain
    (["b3", "b4"], "d9"),
    (["d9"],  "d10"),
    (["d10"], "d11"),
    # Cross-chain conjunction (d12 requires b5 AND something that derives d0)
    (["b5", "d0"], "d12"),
    (["d12"], "d13"),
    # Alternative paths
    (["b6"], "d0"),
    (["b7"], "d4"),
]

# Minimum derivation depth for each derived atom (over any single initial assignment):
# used to define the train/OOD split as a property of the query atom, not the data point.
ATOM_DEPTH = {
    "d0": 1, "d1": 2, "d2": 3, "d3": 4,
    "d4": 1, "d5": 2, "d6": 3,
    "d7": 1, "d8": 2,
    "d9": 1, "d10": 2, "d11": 3,
    "d12": 2, "d13": 3,
}
TRAIN_ATOMS = [a for a in DERIVED if ATOM_DEPTH[a] <= 2]   # depth 1-2
OOD_ATOMS   = [a for a in DERIVED if ATOM_DEPTH[a] >= 3]   # depth 3-4


def build_kb() -> PropKB:
    kb = PropKB()
    for premises, conclusion in RULE_DEFS:
        kb.add_rule(premises, conclusion)
    # Register any base facts not yet seen as premises/conclusions
    for b in BASE:
        kb.add_fact(b)
    return kb


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_dataset(kb: PropKB, chainer: ForwardChainer, query_atoms: list[str]):
    """Enumerate all 2^8 initial fact assignments; label = entailment of query_atom.

    Returns:
        feats:   (N, 8 + len(query_atoms)) float tensor  [initial_facts | query_one_hot]
        labels:  (N,) binary float tensor
        meta:    list of (initial_facts_dict, query_atom)
    """
    n_base = len(BASE)
    n_qa   = len(query_atoms)
    feats_list, labels_list, meta = [], [], []

    for bits in itertools.product([0, 1], repeat=n_base):
        true_facts = [b for b, v in zip(BASE, bits) if v]
        f0 = kb.initial_facts(true_facts)
        with torch.no_grad():
            closure = chainer(f0)          # (1, N_atoms)
        base_vec = torch.tensor(list(bits), dtype=torch.float32)  # (8,)

        for q_idx, q_atom in enumerate(query_atoms):
            label = (closure[0, kb.idx(q_atom)] >= 0.5).float()
            q_oh  = torch.zeros(n_qa)
            q_oh[q_idx] = 1.0
            feats_list.append(torch.cat([base_vec, q_oh]))
            labels_list.append(label)
            meta.append((dict(zip(BASE, bits)), q_atom))

    return torch.stack(feats_list), torch.stack(labels_list), meta


# ---------------------------------------------------------------------------
# MLP baseline
# ---------------------------------------------------------------------------

class EntailmentMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def train_mlp(model: EntailmentMLP, feats: torch.Tensor, labels: torch.Tensor,
              epochs: int, lr: float = 1e-3, batch_size: int = 64) -> list:
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    n       = len(feats)
    losses  = []
    for ep in range(epochs):
        perm  = torch.randperm(n)
        ep_loss = 0.0; steps = 0
        for i in range(0, n, batch_size):
            idx   = perm[i:i + batch_size]
            logit = model(feats[idx])
            loss  = F.binary_cross_entropy(logit.clamp(1e-7, 1 - 1e-7), labels[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); steps += 1
        losses.append(ep_loss / max(steps, 1))
    return losses


def accuracy(model: EntailmentMLP, feats: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        preds = model(feats) > 0.5
    return (preds == labels.bool()).float().mean().item()


# ---------------------------------------------------------------------------
# Chainer accuracy (always 100% by construction)
# ---------------------------------------------------------------------------

def chainer_accuracy(kb: PropKB, chainer: ForwardChainer,
                     feats: torch.Tensor, labels: torch.Tensor,
                     query_atoms: list[str]) -> float:
    """Evaluate ForwardChainer on the (feats, labels) dataset."""
    n_base = len(BASE)
    n_qa   = len(query_atoms)
    correct = 0
    for i in range(len(feats)):
        base_vec = feats[i, :n_base]
        q_idx    = feats[i, n_base:].argmax().item()
        q_atom   = query_atoms[q_idx]
        true_facts = [b for b, v in zip(BASE, base_vec.tolist()) if v > 0.5]
        f0 = kb.initial_facts(true_facts)
        with torch.no_grad():
            closure = chainer(f0)
        pred = (closure[0, kb.idx(q_atom)] >= 0.5).item()
        correct += int(pred == bool(labels[i].item()))
    return correct / len(feats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(smoke: bool = False) -> dict:
    EPOCHS     = 300 if not smoke else 40
    HIDDEN     = 64  if not smoke else 32

    kb      = build_kb()
    chainer = ForwardChainer(kb, max_steps=10)

    # Build full dataset split by atom depth
    print("Building dataset...")
    feats_tr, labels_tr, _ = build_dataset(kb, chainer, TRAIN_ATOMS)
    feats_ood, labels_ood, _ = build_dataset(kb, chainer, OOD_ATOMS)

    n_train_pos = labels_tr.sum().int().item()
    n_ood_pos   = labels_ood.sum().int().item()
    print(f"  Train (depth<=2): {len(feats_tr):>5} examples  ({n_train_pos} positive)")
    print(f"  OOD   (depth>=3): {len(feats_ood):>5} examples  ({n_ood_pos} positive)")

    # ForwardChainer is exact by construction; verify numerically
    fc_acc_tr  = chainer_accuracy(kb, chainer, feats_tr,  labels_tr,  TRAIN_ATOMS)
    fc_acc_ood = chainer_accuracy(kb, chainer, feats_ood, labels_ood, OOD_ATOMS)
    print(f"\nForwardChainer  train={fc_acc_tr:.1%}  OOD={fc_acc_ood:.1%}  (0 params)")

    # MLP baseline: train on depth<=2 examples only
    input_dim = feats_tr.shape[1]   # 8 + len(TRAIN_ATOMS)
    # OOD features have a different query one-hot width; evaluate on a shared
    # 8+14-dim space (all derived atoms as query vocabulary).
    all_atoms = DERIVED
    feats_tr_all,  labels_tr_all,  _ = build_dataset(kb, chainer, all_atoms)
    feats_ood_all, labels_ood_all, _ = build_dataset(kb, chainer, all_atoms)
    # Filter train set to depth<=2 queries
    depth_col = torch.tensor(
        [ATOM_DEPTH[a] for a in all_atoms] * (len(feats_tr_all) // len(all_atoms)),
        dtype=torch.float32
    )
    # feats_tr_all is ordered (assignment, query), depth repeats over assignments
    n_assignments = 2 ** len(BASE)
    depth_per_row = torch.tensor(
        [ATOM_DEPTH[a] for a in all_atoms] * n_assignments, dtype=torch.float32
    )
    train_mask = depth_per_row <= 2
    ood_mask   = depth_per_row >= 3

    feats_train_mlp  = feats_tr_all[train_mask]
    labels_train_mlp = labels_tr_all[train_mask]
    feats_ood_mlp    = feats_tr_all[ood_mask]
    labels_ood_mlp   = labels_tr_all[ood_mask]

    mlp = EntailmentMLP(input_dim=len(all_atoms) + len(BASE), hidden=HIDDEN)
    print(f"\nTraining MLP baseline ({sum(p.numel() for p in mlp.parameters())} params)...")
    losses = train_mlp(mlp, feats_train_mlp, labels_train_mlp, epochs=EPOCHS)

    mlp_acc_tr  = accuracy(mlp, feats_train_mlp, labels_train_mlp)
    mlp_acc_ood = accuracy(mlp, feats_ood_mlp,   labels_ood_mlp)
    print(f"MLP baseline     train={mlp_acc_tr:.1%}  OOD={mlp_acc_ood:.1%}")

    # Accuracy on OOD positive examples only (where entailment requires deep chains)
    ood_pos_mask  = labels_ood_mlp.bool()
    mlp_acc_ood_pos = accuracy(mlp, feats_ood_mlp[ood_pos_mask],
                                labels_ood_mlp[ood_pos_mask]) if ood_pos_mask.any() else float("nan")
    fc_pos_correct = labels_ood_mlp[ood_pos_mask].float().mean().item() if ood_pos_mask.any() else 1.0
    print(f"  (OOD positives only) MLP={mlp_acc_ood_pos:.1%}  Chainer=100.0%")

    n_params_chainer = 0   # all buffers, no nn.Parameters
    n_params_mlp     = sum(p.numel() for p in mlp.parameters())

    results = {
        "forward_chainer": {
            "params": n_params_chainer,
            "train_acc": round(fc_acc_tr, 4),
            "ood_acc":   round(fc_acc_ood, 4),
        },
        "mlp_baseline": {
            "params": n_params_mlp,
            "train_acc":     round(mlp_acc_tr, 4),
            "ood_acc":       round(mlp_acc_ood, 4),
            "ood_pos_acc":   round(mlp_acc_ood_pos, 4),
        },
        "dataset": {
            "train_examples": int(len(feats_train_mlp)),
            "ood_examples":   int(len(feats_ood_mlp)),
            "train_positive": int(labels_train_mlp.sum().item()),
            "ood_positive":   int(labels_ood_mlp.sum().item()),
        },
    }

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Training loss curve
    ax = axes[0]
    ax.plot(losses, color="steelblue", lw=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
    ax.set_title(f"MLP training loss\n(depth ≤ 2 queries only, {n_params_mlp:,} params)")
    ax.set_yscale("log")

    # Accuracy bar chart
    ax2 = axes[1]
    methods = ["ForwardChainer\n(0 params)", f"MLP\n({n_params_mlp:,} params)"]
    train_accs = [fc_acc_tr * 100, mlp_acc_tr * 100]
    ood_accs   = [fc_acc_ood * 100, mlp_acc_ood * 100]
    x = np.arange(len(methods))
    w = 0.35
    bars1 = ax2.bar(x - w/2, train_accs, w, label="Train (depth<=2)", color="#4c9be8")
    bars2 = ax2.bar(x + w/2, ood_accs,   w, label="OOD   (depth>=3)", color="#e8844c")
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}%",
                 ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(methods)
    ax2.set_ylim(0, 115); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Entailment accuracy by derivation depth\n(Horn-clause propositional KB)")
    ax2.legend(fontsize=8)
    ax2.axhline(100, color="k", lw=0.8, ls="--", alpha=0.4)

    fig.suptitle("Tier-1: Forward Chaining (exact) vs MLP baseline")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "paper3_tier1_forward_chaining.png", dpi=150)
    fig.savefig(PLOTS_DIR / "paper3_tier1_forward_chaining.pdf")
    plt.close(fig)

    out_path = RESULTS_DIR / "paper3_tier1_forward_chaining.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nPlot -> {PLOTS_DIR}/paper3_tier1_forward_chaining.png")
    print(f"JSON -> {out_path}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    run(smoke=args.smoke)
