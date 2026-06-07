"""End-to-end regression smoke tests for the teacher-training loop.

These stand in for the paper experiments at miniature scale: they assert the
generic ``train_teacher`` loop drives a teacher to *learn* (loss falls, batch
accuracy rises), that the Sigma-Pi pi-scale diagnostic is recorded and finite,
that a random-weight baseline sits near chance, and that checkpoints round-trip.

A deliberately trivial setup keeps them fast and deterministic: an identity
"student" exposes raw inputs as features, and inputs are Gaussian clusters
around well-separated class centroids, so the optimal linear head is essentially
the per-class mean — exactly the channel-0 signal the prototype carries.
"""

from __future__ import annotations

import math
import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

from polyweave.hypernets import FCMapTeacher
from polyweave.prototypes import feature_class_stats
from polyweave.training import load_checkpoint, save_checkpoint, train_teacher
from polyweave.utils import set_seed

K, D, B = 5, 16, 128


class _IdentityStudent(nn.Module):
    """Features are the raw inputs; the head can be externally supplied."""

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x, generated_fc=None):
        if generated_fc is None:
            raise ValueError("expects a generated head")
        return F.linear(x, generated_fc["weight"], generated_fc["bias"])


def _make_task(seed: int = 0):
    set_seed(seed)
    centroids = torch.randn(K, D) * 3.0

    def sample_batch():
        y = torch.randint(0, K, (B,))
        x = centroids[y] + 0.5 * torch.randn(B, D)
        return x, y

    def build_prototype(student, batch):
        x, y = batch
        return feature_class_stats(student.extract_features(x), y, num_classes=K)

    def forward(student, batch, gen):
        x, y = batch
        return student(x, generated_fc=gen), y

    return sample_batch, build_prototype, forward


def test_teacher_learns_and_records_pi_scale():
    set_seed(0)
    sample_batch, build_prototype, forward = _make_task(seed=0)
    student = _IdentityStudent()
    teacher = FCMapTeacher(num_classes=K, feature_dim=D, sigma_pi=True)

    result = train_teacher(
        teacher, [student],
        sample_batch=sample_batch, build_prototype=build_prototype, forward=forward,
        steps=200, lr=1e-3,
    )

    losses = result.losses
    first = sum(losses[:20]) / 20
    last = sum(losses[-20:]) / 20
    assert last < first, f"teacher did not learn: first={first:.3f} last={last:.3f}"

    # pi-scale diagnostic present, finite, and positive.
    assert len(result.pi_scales) == 200
    assert result.final_pi_scale is not None
    assert math.isfinite(result.final_pi_scale) and result.final_pi_scale > 0


def test_trained_teacher_beats_chance_and_random_does_not():
    set_seed(1)
    sample_batch, build_prototype, forward = _make_task(seed=1)
    student = _IdentityStudent()
    teacher = FCMapTeacher(num_classes=K, feature_dim=D, sigma_pi=False)
    train_teacher(
        teacher, [student],
        sample_batch=sample_batch, build_prototype=build_prototype, forward=forward,
        steps=250, lr=1e-3,
    )

    @torch.no_grad()
    def accuracy(gen_fn):
        correct = total = 0
        for _ in range(10):
            x, y = sample_batch()
            gen = gen_fn(x, y)
            pred = student(x, generated_fc=gen).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
        return correct / total

    teacher.eval()
    trained_acc = accuracy(lambda x, y: teacher(feature_class_stats(x, y, K)))
    random_acc = accuracy(
        lambda x, y: {"weight": torch.randn(K, D), "bias": torch.zeros(K)}
    )

    chance = 1.0 / K
    assert trained_acc > 0.6, f"trained teacher only reached {trained_acc:.3f}"
    assert random_acc < 0.4, f"random head should sit near chance ({chance:.2f}), got {random_acc:.3f}"


def test_checkpoint_roundtrip():
    teacher = FCMapTeacher(num_classes=K, feature_dim=D, sigma_pi=True)
    proto = torch.randn(1, 4, K, D)
    teacher.eval()
    with torch.no_grad():
        before = teacher(proto)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "teacher.pt")
        save_checkpoint(path, teacher, meta={"steps": 200})
        fresh = FCMapTeacher(num_classes=K, feature_dim=D, sigma_pi=True)
        meta = load_checkpoint(path, fresh)

    assert meta == {"steps": 200}
    fresh.eval()
    with torch.no_grad():
        after = fresh(proto)
    assert torch.allclose(before["weight"], after["weight"], atol=1e-6)
    assert torch.allclose(before["bias"], after["bias"], atol=1e-6)
