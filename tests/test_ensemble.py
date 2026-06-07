"""Tests for ensemble metrics, the population helper, and ensemble plots."""

from __future__ import annotations

import torch

from polyweave.evaluation import (
    accuracy_from_probs,
    disagreement_matrix,
    ensemble_accuracy,
    ensemble_gain,
    ensemble_probs,
    member_accuracies,
    pairwise_disagreement,
)
from polyweave.experiments.cifar_conv1 import CONV1_IN, CONV1_KERNEL, CONV1_OUT
from polyweave.experiments.ensemble import (
    evaluate_population,
    population_probs,
)
from polyweave.hypernets import ConvFilterTeacher
from polyweave.students import make_cnn_student
from polyweave.targets import Conv2dTargetSpec
from polyweave.utils import set_seed
from polyweave.viz import plot_diversity_hist, plot_ensemble_bars


# ---------------------------------------------------------------------------
# Pure metrics
# ---------------------------------------------------------------------------

def _onehot_probs(preds, num_classes):
    """[M, N] integer predictions -> [M, N, C] hard one-hot probabilities."""
    M, N = preds.shape
    p = torch.zeros(M, N, num_classes)
    for m in range(M):
        p[m, torch.arange(N), preds[m]] = 1.0
    return p


def test_ensemble_probs_is_member_mean():
    probs = torch.rand(3, 5, 4)
    assert torch.allclose(ensemble_probs(probs), probs.mean(0))


def test_member_and_ensemble_accuracy():
    labels = torch.tensor([0, 1, 2, 0])
    preds = torch.tensor([[0, 1, 2, 0],   # perfect
                          [0, 1, 2, 1],   # 3/4
                          [1, 1, 2, 0]])  # 3/4
    probs = _onehot_probs(preds, 3)
    accs = member_accuracies(probs, labels)
    assert torch.allclose(accs, torch.tensor([1.0, 0.75, 0.75]))
    # Majority vote recovers the correct label on every example -> perfect.
    assert ensemble_accuracy(probs, labels) == 1.0
    assert abs(ensemble_gain(probs, labels) - (1.0 - (1.0 + 0.75 + 0.75) / 3)) < 1e-6


def test_accuracy_from_probs_matches_argmax():
    probs = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    assert accuracy_from_probs(probs, torch.tensor([1, 0])) == 1.0
    assert accuracy_from_probs(probs, torch.tensor([0, 1])) == 0.0


def test_pairwise_disagreement_bounds():
    # Identical members -> 0 disagreement.
    same = _onehot_probs(torch.tensor([[0, 1, 2], [0, 1, 2]]), 3)
    assert pairwise_disagreement(same) == 0.0
    # Fully disjoint predictions -> disagree everywhere.
    diff = _onehot_probs(torch.tensor([[0, 0, 0], [1, 1, 1]]), 3)
    assert pairwise_disagreement(diff) == 1.0
    # Single member is undefined -> 0.
    assert pairwise_disagreement(diff[:1]) == 0.0


def test_disagreement_matrix_symmetric_zero_diag():
    preds = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 0], [3, 2, 1, 0]])
    probs = _onehot_probs(preds, 4)
    m = disagreement_matrix(probs)
    assert m.shape == (3, 3)
    assert torch.allclose(m, m.T)
    assert torch.allclose(torch.diagonal(m), torch.zeros(3))


# ---------------------------------------------------------------------------
# Population helper (tiny real models, fabricated data)
# ---------------------------------------------------------------------------

NUM_CLASSES = 3


def _tiny_teacher(sigma_pi):
    spec = Conv2dTargetSpec(CONV1_OUT, CONV1_IN, CONV1_KERNEL)
    return ConvFilterTeacher(spec, proto_channels=4, width=8, sigma_pi=sigma_pi)


def _tiny_students(n=3):
    return [make_cnn_student("A", feature_dim=16, num_classes=NUM_CLASSES,
                             in_ch=CONV1_IN, conv1_out=CONV1_OUT,
                             kernel_size=CONV1_KERNEL) for _ in range(n)]


def _batches(n=2, bs=8):
    return [(torch.randn(bs, 3, 32, 32), torch.randint(0, NUM_CLASSES, (bs,)))
            for _ in range(n)]


def test_population_probs_shape_and_normalised():
    set_seed(0)
    students = _tiny_students(3)
    support, eval_b = _batches(2), _batches(2)
    probs = population_probs(
        students, _tiny_teacher(False), support, eval_b,
        num_classes=NUM_CLASSES, proto_grid=4, bn_reset_batches=1,
    )
    M, N = 3, sum(b[0].shape[0] for b in eval_b)
    assert probs.shape == (M, N, NUM_CLASSES)
    assert torch.allclose(probs.sum(-1), torch.ones(M, N), atol=1e-5)  # softmax rows


def test_evaluate_population_bundle_keys():
    set_seed(0)
    students = _tiny_students(3)
    support, eval_b = _batches(2), _batches(2)
    labels = torch.cat([y for _x, y in eval_b])
    probs = population_probs(
        students, _tiny_teacher(True), support, eval_b,
        num_classes=NUM_CLASSES, proto_grid=4, bn_reset_batches=1,
    )
    r = evaluate_population(probs, labels)
    assert set(r) >= {"single_mean", "single_std", "ensemble_acc",
                      "ensemble_gain", "mean_disagreement", "pairwise_disagreements"}
    assert len(r["member_accs"]) == 3
    assert len(r["pairwise_disagreements"]) == 3  # C(3, 2)
    assert 0.0 <= r["mean_disagreement"] <= 1.0


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def test_ensemble_plots_write_pdfs(tmp_path):
    w1 = plot_ensemble_bars(
        single_means={"additive": 0.6, "sigmapi": 0.62},
        ensemble={"additive": 0.66, "sigmapi": 0.70},
        single_stds={"additive": 0.03, "sigmapi": 0.04},
        name="ens_bars", plots_dir=tmp_path,
    )
    w2 = plot_diversity_hist(
        {"additive": [0.1, 0.2, 0.15], "sigmapi": [0.3, 0.35, 0.28]},
        name="ens_div", plots_dir=tmp_path,
    )
    assert (tmp_path / "ens_bars.pdf").exists()
    assert (tmp_path / "ens_div.pdf").exists()
    assert any(p.suffix == ".pdf" for p in w1 + w2)
