"""Tests for polyweave.metrics — pi-scale and ensemble (dis)agreement."""

from __future__ import annotations

import math

import torch

from polyweave.layers import ConvSigmaPi2d
from polyweave.metrics import (
    disagreement_rate,
    ensemble_accuracy,
    majority_vote,
    mean_accuracy,
    pairwise_disagreement,
    pi_scale_mean,
)


# ---------------------------------------------------------------------------
# pi-scale
# ---------------------------------------------------------------------------

def test_pi_scale_mean_from_block():
    block = ConvSigmaPi2d(channels=8)
    assert math.isclose(pi_scale_mean(block), math.exp(-2.0), rel_tol=1e-5)


def test_pi_scale_mean_from_tensor():
    t = torch.zeros(4, 1, 1)
    assert math.isclose(pi_scale_mean(t), 1.0, rel_tol=1e-6)


def test_pi_scale_mean_finds_nested_param():
    import torch.nn as nn

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = ConvSigmaPi2d(channels=4)

    assert math.isclose(pi_scale_mean(Wrapper()), math.exp(-2.0), rel_tol=1e-5)


# ---------------------------------------------------------------------------
# ensemble disagreement
# ---------------------------------------------------------------------------

def test_disagreement_zero_when_unanimous():
    preds = [torch.tensor([0, 1, 2, 3]) for _ in range(3)]
    assert disagreement_rate(preds) == 0.0
    assert pairwise_disagreement(preds) == 0.0


def test_disagreement_full_when_all_differ():
    preds = [torch.tensor([0, 0, 0, 0]), torch.tensor([1, 1, 1, 1])]
    assert disagreement_rate(preds) == 1.0
    assert pairwise_disagreement(preds) == 1.0


def test_disagreement_partial():
    preds = [torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 9])]
    assert math.isclose(disagreement_rate(preds), 0.25, rel_tol=1e-6)


def test_majority_vote_breaks_ties_to_lowest_index():
    # one vote each for class 0 and 1 -> tie -> argmax picks 0
    preds = [torch.tensor([0]), torch.tensor([1])]
    assert majority_vote(preds, num_classes=3).item() == 0


def test_ensemble_beats_mean_when_errors_decorrelate():
    targets = torch.tensor([0, 1, 2, 3])
    # each member wrong on a different single sample; majority recovers all.
    preds = [
        torch.tensor([9, 1, 2, 3]),
        torch.tensor([0, 9, 2, 3]),
        torch.tensor([0, 1, 9, 3]),
    ]
    assert ensemble_accuracy(preds, targets, num_classes=10) == 1.0
    assert mean_accuracy(preds, targets) < 1.0


def test_single_member_no_disagreement():
    preds = [torch.tensor([0, 1, 2])]
    assert disagreement_rate(preds) == 0.0
    assert pairwise_disagreement(preds) == 0.0
