"""Tests for polyweave.evaluation — zero-shot, recovery, BN reset, baselines."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from polyweave.evaluation import (
    average_weights,
    centroids_to_fc,
    class_centroids,
    evaluate_accuracy,
    evaluate_macro_f1,
    generate_averaged,
    mean_curves,
    ncc_fc,
    random_like,
    recovery_curve,
    reset_bn_stats,
)
from polyweave.utils import set_seed

K, D = 4, 8


# ---------------------------------------------------------------------------
# average_weights
# ---------------------------------------------------------------------------

def test_average_weights_dict():
    a = {"weight": torch.zeros(2, 3), "bias": torch.ones(2)}
    b = {"weight": torch.ones(2, 3) * 2, "bias": torch.ones(2) * 3}
    out = average_weights([a, b])
    assert torch.allclose(out["weight"], torch.ones(2, 3))
    assert torch.allclose(out["bias"], torch.full((2,), 2.0))


def test_average_weights_list_of_dicts():
    s1 = [{"q_weight": torch.zeros(2, 2)}, {"q_weight": torch.zeros(2, 2)}]
    s2 = [{"q_weight": torch.ones(2, 2) * 4}, {"q_weight": torch.ones(2, 2) * 2}]
    out = average_weights([s1, s2])
    assert torch.allclose(out[0]["q_weight"], torch.full((2, 2), 2.0))
    assert torch.allclose(out[1]["q_weight"], torch.ones(2, 2))


# ---------------------------------------------------------------------------
# NCC baseline
# ---------------------------------------------------------------------------

def test_class_centroids_are_per_class_means():
    feats = torch.tensor([[1.0, 1.0], [3.0, 3.0], [10.0, 0.0]])
    y = torch.tensor([0, 0, 1])
    c = class_centroids(feats, y, num_classes=2)
    assert torch.allclose(c[0], torch.tensor([2.0, 2.0]))
    assert torch.allclose(c[1], torch.tensor([10.0, 0.0]))


def test_ncc_fc_classifies_well_separated_clusters():
    set_seed(0)
    centroids = torch.tensor([[5.0, 0.0], [-5.0, 0.0], [0.0, 5.0]])
    y = torch.randint(0, 3, (300,))
    feats = centroids[y] + 0.2 * torch.randn(300, 2)
    head = ncc_fc(feats, y, num_classes=3)
    logits = F.linear(feats, head["weight"], head["bias"])
    acc = (logits.argmax(1) == y).float().mean().item()
    assert acc > 0.95


def test_centroids_to_fc_bias_formula():
    c = torch.tensor([[3.0, 4.0]])  # norm^2 = 25
    head = centroids_to_fc(c)
    assert torch.allclose(head["bias"], torch.tensor([-12.5]))


# ---------------------------------------------------------------------------
# random_like
# ---------------------------------------------------------------------------

def test_random_like_matches_shapes_and_zeros_bias():
    ref = {"weight": torch.randn(5, 7), "bias": torch.randn(5)}
    out = random_like(ref)
    assert out["weight"].shape == (5, 7)
    assert out["bias"].shape == (5,)
    assert torch.count_nonzero(out["bias"]) == 0  # zero_bias default


def test_random_like_recurses_over_list():
    ref = [{"q_weight": torch.randn(3, 3), "q_bias": torch.randn(3)}]
    out = random_like(ref)
    assert out[0]["q_weight"].shape == (3, 3)
    assert torch.count_nonzero(out[0]["q_bias"]) == 0


def test_random_like_uses_fan_in_scaling_by_default():
    """Default init must be Kaiming-linear (std ~ 1/sqrt(fan_in)), not unit variance.

    A unit-variance head saturates the softmax and stalls recovery fine-tuning;
    fan-in scaling keeps the random baseline well-conditioned.
    """
    torch.manual_seed(0)
    fan_in = 256
    ref = {"weight": torch.empty(10, fan_in), "bias": torch.empty(10)}
    w = random_like(ref)["weight"]
    assert abs(w.std().item() - 1.0 / fan_in ** 0.5) < 0.02   # ~0.0625
    # Opt-out recovers raw unit-variance behaviour.
    w_raw = random_like(ref, fan_in_scale=False)["weight"]
    assert abs(w_raw.std().item() - 1.0) < 0.1


# ---------------------------------------------------------------------------
# evaluate_accuracy + generate_averaged
# ---------------------------------------------------------------------------

class _LinearStudent(nn.Module):
    def forward(self, x, gen=None):
        return F.linear(x, gen["weight"], gen["bias"])


def _forward(student, batch, gen):
    x, y = batch
    return student(x, gen), y


def test_evaluate_accuracy_perfect_and_chance():
    centroids = torch.tensor([[5.0, 0.0], [-5.0, 0.0], [0.0, 5.0]])
    y = torch.randint(0, 3, (200,))
    x = centroids[y] + 0.1 * torch.randn(200, 2)
    student = _LinearStudent()
    good = ncc_fc(x, y, 3)
    acc = evaluate_accuracy(student, [(x, y)], good, _forward)
    assert acc > 0.95
    bad = {"weight": torch.zeros(3, 2), "bias": torch.zeros(3)}
    assert evaluate_accuracy(student, [(x, y)], bad, _forward) < 0.6


def test_macro_f1_perfect_and_matches_accuracy_when_balanced():
    centroids = torch.tensor([[5.0, 0.0], [-5.0, 0.0], [0.0, 5.0]])
    y = torch.arange(3).repeat(100)  # perfectly balanced, 100 per class
    x = centroids[y] + 0.1 * torch.randn(y.numel(), 2)
    student = _LinearStudent()
    good = ncc_fc(x, y, 3)
    f1 = evaluate_macro_f1(student, [(x, y)], good, _forward, num_classes=3)
    acc = evaluate_accuracy(student, [(x, y)], good, _forward)
    assert f1 > 0.95
    assert abs(f1 - acc) < 0.05  # balanced -> macro-F1 tracks accuracy


def test_macro_f1_penalises_ignoring_a_class():
    # A degenerate head that always predicts class 0 gets 1/3 accuracy on a
    # balanced 3-class set but a much lower macro-F1 (two classes score 0).
    x = torch.randn(150, 2)
    y = torch.arange(3).repeat(50)
    student = _LinearStudent()
    always0 = {"weight": torch.tensor([[0.0, 0.0], [-9.0, 0.0], [-9.0, 0.0]]),
               "bias": torch.tensor([9.0, -9.0, -9.0])}
    f1 = evaluate_macro_f1(student, [(x, y)], always0, _forward, num_classes=3)
    acc = evaluate_accuracy(student, [(x, y)], always0, _forward)
    assert abs(acc - 1 / 3) < 0.05
    assert f1 < acc  # macro-F1 exposes the collapsed-prediction failure


def test_generate_averaged_reduces_to_mean_over_support():
    class _ConstTeacher(nn.Module):
        """Returns a fixed head regardless of prototype; lets us check averaging."""

        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, proto):
            self.calls += 1
            return {"weight": proto.new_full((2, 2), float(self.calls)),
                    "bias": proto.new_zeros(2)}

    teacher = _ConstTeacher()
    student = _LinearStudent()
    batches = [(torch.randn(4, 2), torch.randint(0, 2, (4,))) for _ in range(3)]
    avg = generate_averaged(teacher, student, batches, build_prototype=lambda s, b: torch.zeros(1, 4, 2, 2))
    # calls return 1,2,3 -> mean 2
    assert torch.allclose(avg["weight"], torch.full((2, 2), 2.0))


# ---------------------------------------------------------------------------
# reset_bn_stats
# ---------------------------------------------------------------------------

def test_reset_bn_stats_updates_running_mean():
    bn = nn.BatchNorm1d(3)
    bn.eval()
    before = bn.running_mean.clone()
    batches = [(torch.randn(16, 3) + 5.0,) for _ in range(5)]
    reset_bn_stats(bn, batches, run=lambda m, b: m(b[0]), max_batches=5)
    assert not torch.allclose(bn.running_mean, before)
    assert not bn.training  # restored to eval


# ---------------------------------------------------------------------------
# recovery_curve
# ---------------------------------------------------------------------------

def test_recovery_curve_improves_from_bad_init():
    set_seed(0)
    centroids = torch.tensor([[5.0, 0.0], [-5.0, 0.0], [0.0, 5.0]])

    def sample_batch():
        y = torch.randint(0, 3, (128,))
        return centroids[y] + 0.2 * torch.randn(128, 2), y

    student = _LinearStudent()
    # Give the student its own trainable head, initialised badly (zeros).
    student.head = nn.Linear(2, 3)
    nn.init.zeros_(student.head.weight)
    nn.init.zeros_(student.head.bias)

    def init(s):
        return [s.head.weight, s.head.bias]

    def forward(s, batch, gen):
        x, y = batch
        # gen is None during recovery -> use the student's own head
        return s.head(x), y

    def eval_fn(s):
        return evaluate_accuracy(s, [sample_batch()], None, forward)

    curve = recovery_curve(
        student, init=init, sample_batch=sample_batch, forward=forward,
        eval_fn=eval_fn, steps=100, lr=0.1, eval_every=20,
    )
    assert curve[0][0] == 0
    assert curve[-1][0] == 100
    assert curve[-1][1] > curve[0][1] + 0.3  # learns from the zero init


def test_mean_curves_averages_on_shared_grid():
    c1 = [(0, 0.2), (20, 0.5), (40, 0.9)]
    c2 = [(0, 0.4), (20, 0.7), (40, 1.0)]
    out = mean_curves([c1, c2])
    assert [s for s, _ in out] == [0, 20, 40]
    assert all(abs(a - b) < 1e-9 for (_, a), b in zip(out, [0.3, 0.6, 0.95]))
