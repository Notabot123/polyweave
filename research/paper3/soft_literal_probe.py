"""Probe: can a *signed-exponent* geometric product induce a rule WITH NEGATION?

The thesis (Paper 3 candidate mechanism): a plain product t-norm AND can only build
*monotone* conjunctions of positive literals. But polyweave's Sigma-Pi product has
**signed** exponents, which — applied to the pair of features ``[log t, log(1-t)]`` —
turns one learnable weight per premise into a continuous knob over its role:

    contribution_i = t_i ** [w_i]+  ·  (1 - t_i) ** [w_i]-          ( [w]+ = max(w,0) )
    rule_fires     = prod_i contribution_i                          (product = AND)

    w_i > 0  -> premise REQUIRED (positive literal)
    w_i = 0  -> premise IRRELEVANT (t^0 = 1, drops out)
    w_i < 0  -> premise INHIBITORY (negated literal)

So a single layer can *induce* a non-monotonic rule like ``fly <- bird & not penguin``
and you can read the rule straight off the exponents. A plain positive-only product
cannot represent the ``not penguin`` exception at all.

This script tests exactly that, head to head against the library's other layers.

Run:  python research/paper3/soft_literal_probe.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from polyweave import PolyLinear, SigmaPiLinear  # noqa: E402

EPS = 1e-6
FEATURES = ["bird", "penguin", "d2", "d3", "d4", "d5", "d6", "d7"]
N_FEAT = len(FEATURES)


def make_data(n: int, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Random boolean attributes; label = bird AND NOT penguin."""
    X = (torch.rand(n, N_FEAT, generator=rng) < 0.5).float()
    y = (X[:, 0] * (1.0 - X[:, 1])).unsqueeze(1)          # bird & not penguin
    return X, y


class SoftSignedLiteral(nn.Module):
    """A single conjunction with signed log-space exponents (the candidate mechanism).

    ``signed=False`` disables the negation term -> a plain positive-only product AND.
    """

    def __init__(self, n_feat: int, signed: bool = True) -> None:
        super().__init__()
        self.signed = signed
        self.w = nn.Parameter(torch.randn(n_feat) * 0.1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.clamp(EPS, 1 - EPS)
        wp = self.w.clamp(min=0.0)
        logc = wp * torch.log(t)
        if self.signed:
            wn = (-self.w).clamp(min=0.0)
            logc = logc + wn * torch.log(1.0 - t)
        return torch.exp(logc.sum(-1, keepdim=True))       # rule firing in (0, 1]

    @torch.no_grad()
    def exponent_abs_mean(self) -> float:
        return self.w.abs().mean().item()


@dataclass
class Config:
    n_train: int = 6000
    n_val: int = 2000
    epochs: int = 150
    batch_size: int = 256
    lr: float = 0.05
    seed: int = 0


def _accuracy(prob: torch.Tensor, y: torch.Tensor) -> float:
    return ((prob > 0.5).float() == y).float().mean().item()


def train(model: nn.Module, X: torch.Tensor, y: torch.Tensor, cfg: Config,
          prob_from_logit: bool) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n = X.shape[0]
    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        for b in range(0, n, cfg.batch_size):
            idx = perm[b : b + cfg.batch_size]
            opt.zero_grad()
            out = model(X[idx])
            if prob_from_logit:
                loss = F.binary_cross_entropy_with_logits(out, y[idx])
            else:
                loss = F.binary_cross_entropy(out.clamp(EPS, 1 - EPS), y[idx])
            loss.backward()
            opt.step()


def predict_prob(model: nn.Module, X: torch.Tensor, prob_from_logit: bool) -> torch.Tensor:
    with torch.no_grad():
        out = model(X)
        return torch.sigmoid(out) if prob_from_logit else out


def run(cfg: Config) -> None:
    rng = torch.Generator().manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    Xtr, ytr = make_data(cfg.n_train, rng)
    Xva, yva = make_data(cfg.n_val, rng)

    base_rate = max(yva.mean().item(), 1 - yva.mean().item())
    print(f"task: fly = bird AND NOT penguin   ({N_FEAT} features, "
          f"{N_FEAT-2} distractors)")
    print(f"majority-class baseline = {base_rate:.3f}\n")

    # (name, model, prob_from_logit)
    models = [
        ("soft signed literal (ours)", SoftSignedLiteral(N_FEAT, signed=True), False),
        ("positive-only product AND",  SoftSignedLiteral(N_FEAT, signed=False), False),
        ("PolyLinear (rank 2)",        PolyLinear(N_FEAT, 1, rank=2), True),
        ("SigmaPiLinear",              SigmaPiLinear(N_FEAT, 1), True),
        ("nn.Linear",                  nn.Linear(N_FEAT, 1), True),
    ]

    print(f"{'model':<28} {'val acc':>8} {'params':>8}")
    print("-" * 48)
    results = {}
    for name, model, from_logit in models:
        train(model, Xtr, ytr, cfg, from_logit)
        acc = _accuracy(predict_prob(model, Xva, from_logit), yva)
        results[name] = (acc, model)
        print(f"{name:<28} {acc:>8.3f} {sum(p.numel() for p in model.parameters()):>8}")

    # --- interpretability: read the induced rule off the signed exponents ---
    soft = results["soft signed literal (ours)"][1]
    print("\nInduced rule (signed exponents w_i):")
    for f, wv in zip(FEATURES, soft.w.detach().tolist()):
        role = "REQUIRED" if wv > 0.3 else ("INHIBITORY" if wv < -0.3 else "irrelevant")
        print(f"  {f:<9} w = {wv:+.2f}   {role}")
    print(f"\nrecruitment (mean|w|) = {soft.exponent_abs_mean():.3f}")


ary = ["bird", "penguin", "bat", "broken", "d4", "d5", "d6", "d7"]


def make_dnf_data(n: int, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """fly = (bird & not penguin) OR (bat & not broken)  -- a 2-term DNF, NOT linearly
    separable, so a single linear unit cannot represent it."""
    X = (torch.rand(n, len(ary), generator=rng) < 0.5).float()
    term1 = X[:, 0] * (1 - X[:, 1])      # bird & not penguin
    term2 = X[:, 2] * (1 - X[:, 3])      # bat  & not broken
    y = (1 - (1 - term1) * (1 - term2)).unsqueeze(1)   # OR
    return X, y


class SoftRuleLayer(nn.Module):
    """K signed-literal conjunctions combined by a probabilistic OR (a soft DNF)."""

    def __init__(self, n_feat: int, n_rules: int) -> None:
        super().__init__()
        self.rules = nn.ModuleList(SoftSignedLiteral(n_feat, signed=True)
                                   for _ in range(n_rules))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        not_fire = torch.cat([1 - r(t) for r in self.rules], dim=-1)  # [B, K]
        return (1 - not_fire.prod(dim=-1, keepdim=True))              # OR


class MLP(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)


def run_dnf(cfg: Config) -> None:
    rng = torch.Generator().manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    nf = len(ary)
    Xtr, ytr = make_dnf_data(cfg.n_train, rng)
    Xva, yva = make_dnf_data(cfg.n_val, rng)
    base = max(yva.mean().item(), 1 - yva.mean().item())
    print("\n" + "=" * 56)
    print("DNF task: fly = (bird & NOT penguin) OR (bat & NOT broken)")
    print(f"  NOT linearly separable.  majority baseline = {base:.3f}\n")

    models = [
        ("nn.Linear",                  nn.Linear(nf, 1), True),
        ("soft RULE LAYER (2, ours)",  SoftRuleLayer(nf, 2), False),
        ("PolyLinear (rank 4)",        PolyLinear(nf, 1, rank=4), True),
        ("MLP (hidden 32)",            MLP(nf), True),
    ]
    print(f"{'model':<28} {'val acc':>8} {'params':>8}")
    print("-" * 48)
    res = {}
    for name, model, from_logit in models:
        train(model, Xtr, ytr, cfg, from_logit)
        acc = _accuracy(predict_prob(model, Xva, from_logit), yva)
        res[name] = model
        print(f"{name:<28} {acc:>8.3f} {sum(p.numel() for p in model.parameters()):>8}")

    print("\nInduced rules (soft rule layer):")
    for k, r in enumerate(res["soft RULE LAYER (2, ours)"].rules):
        parts = []
        for f, wv in zip(ary, r.w.detach().tolist()):
            if wv > 0.3:
                parts.append(f"{f}")
            elif wv < -0.3:
                parts.append(f"NOT {f}")
        print(f"  rule {k}: {' & '.join(parts) if parts else '(empty)'}")


if __name__ == "__main__":
    run(Config())
    run_dnf(Config())
