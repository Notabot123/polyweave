"""Tests for polyweave.logic soft signed literals (differentiable rule induction)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from polyweave.logic import SoftRuleLayer, SoftSignedLiteral

FEATS = ["bird", "penguin", "d2", "d3"]


def _xor_free_data(n, rng):
    X = (torch.rand(n, 4, generator=rng) < 0.5).float()
    y = (X[:, 0] * (1 - X[:, 1])).unsqueeze(1)        # bird & not penguin
    return X, y


def _fit(model, X, y, steps=120, lr=0.05):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        F.binary_cross_entropy(model(X).clamp(1e-6, 1 - 1e-6), y).backward()
        opt.step()
    with torch.no_grad():
        return ((model(X) > 0.5).float() == y).float().mean().item()


def test_forward_shape_and_range():
    layer = SoftSignedLiteral(4)
    out = layer(torch.rand(16, 4))
    assert out.shape == (16, 1)
    assert torch.all(out > 0) and torch.all(out <= 1.0 + 1e-6)


def test_signed_literal_induces_negation():
    rng = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    X, y = _xor_free_data(4000, rng)
    layer = SoftSignedLiteral(4, signed=True)
    acc = _fit(layer, X, y)
    w = layer.w.detach()
    assert acc > 0.95
    assert w[0] > 0.3                      # bird REQUIRED
    assert w[1] < -0.3                     # penguin INHIBITORY
    assert w[2].abs() < 0.3 and w[3].abs() < 0.3   # distractors ignored
    roles = {n: r for n, r, _ in layer.literals(FEATS)}
    assert roles == {"bird": "required", "penguin": "inhibitory"}


def test_positive_only_cannot_negate():
    rng = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    X, y = _xor_free_data(4000, rng)
    # No negation term -> cannot represent the "not penguin" exception.
    acc = _fit(SoftSignedLiteral(4, signed=False), X, y)
    assert acc < 0.85                      # capped well below the signed version


def test_rule_layer_induces_dnf():
    rng = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    feats = ["bird", "penguin", "bat", "broken"]
    X = (torch.rand(6000, 4, generator=rng) < 0.5).float()
    t1 = X[:, 0] * (1 - X[:, 1])           # bird & not penguin
    t2 = X[:, 2] * (1 - X[:, 3])           # bat & not broken
    y = (1 - (1 - t1) * (1 - t2)).unsqueeze(1)
    layer = SoftRuleLayer(4, n_rules=2, signed=True)
    acc = _fit(layer, X, y, steps=200)
    assert acc > 0.95
    rules = set(layer.rules_text(feats))
    assert rules == {"bird & not penguin", "bat & not broken"}


def test_exponent_abs_mean_tracks_recruitment():
    layer = SoftSignedLiteral(4)
    base = layer.exponent_abs_mean()
    with torch.no_grad():
        layer.w.add_(2.0)
    assert layer.exponent_abs_mean() > base
