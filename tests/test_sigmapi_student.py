"""Smoke tests for the Sigma-Pi *student* prototype (further-work experiment).

These cover the new pieces that let a teacher generate the weights of a
multiplicative target layer:

* :class:`~polyweave.targets.SigmaPiConvTargetSpec` — pack/unpack round-trip,
  ``num_params`` accounting, and install/extract against a real ``ConvSigmaPi2d``.
* :class:`~polyweave.students.SigmaPiStudent` — forward shapes with and without
  generated weights, and crucially that the *generated* path is differentiable so
  gradients flow back to the teacher.
* An end-to-end miniature: a ``ConvFilterTeacher`` parameterised by the Sigma-Pi
  conv spec learns to classify Gaussian image clusters, loss falls, and the
  pi-scale diagnostic is recorded.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from polyweave.hypernets import ConvFilterTeacher
from polyweave.layers import ConvSigmaPi2d
from polyweave.prototypes import image_grid_stats
from polyweave.students import SigmaPiStudent
from polyweave.targets import SigmaPiConvTargetSpec
from polyweave.training import train_teacher
from polyweave.utils import set_seed


# ---------------------------------------------------------------------------
# Target spec
# ---------------------------------------------------------------------------

def test_sigmapi_spec_num_params_and_roundtrip():
    spec = SigmaPiConvTargetSpec(channels=8, kernel_size=3)
    # two conv weight matrices + two biases
    assert spec.num_params == 2 * (8 * 8 * 3 * 3) + 2 * 8

    flat = torch.randn(spec.num_params)
    w = spec.unpack(flat)
    assert w["sigma_weight"].shape == (8, 8, 3, 3)
    assert w["pi_weight"].shape == (8, 8, 3, 3)
    assert w["sigma_bias"].shape == (8,)
    assert w["pi_bias"].shape == (8,)
    assert torch.allclose(spec.pack(w), flat)


def test_sigmapi_spec_install_extract_roundtrip():
    spec = SigmaPiConvTargetSpec(channels=6, kernel_size=3)
    block = ConvSigmaPi2d(6)
    flat = torch.randn(spec.num_params)
    weights = spec.unpack(flat)
    spec.install(block, weights)
    out = spec.extract(block)
    for k in ("sigma_weight", "sigma_bias", "pi_weight", "pi_bias"):
        assert torch.allclose(out[k], weights[k], atol=1e-6)


# ---------------------------------------------------------------------------
# Student forward
# ---------------------------------------------------------------------------

def test_sigmapi_student_forward_shapes():
    s = SigmaPiStudent(width=16, num_classes=10, in_ch=3, img_size=32)
    s.eval()
    x = torch.randn(4, 3, 32, 32)
    assert s(x).shape == (4, 10)

    spec = SigmaPiConvTargetSpec(channels=16)
    gen = spec.unpack(torch.randn(spec.num_params) * 0.05)
    assert s(x, gen_sigmapi=gen).shape == (4, 10)


def test_sigmapi_student_pi_scale_diagnostic():
    s = SigmaPiStudent(width=16)
    assert math.isclose(s.pi_scale_mean(), math.exp(-2.0), rel_tol=1e-4)


def test_generated_path_is_differentiable_to_teacher():
    """The generated weights must carry gradient back to the teacher."""
    set_seed(0)
    spec = SigmaPiConvTargetSpec(channels=8)
    teacher = ConvFilterTeacher(spec, proto_channels=4, width=16, sigma_pi=True)
    student = SigmaPiStudent(width=8, num_classes=5, img_size=16)

    proto = torch.randn(1, 4, 8, 8)
    gen = teacher(proto)
    x = torch.randn(4, 3, 16, 16)
    logits = student(x, gen_sigmapi=gen)
    loss = F.cross_entropy(logits, torch.randint(0, 5, (4,)))
    loss.backward()

    grads = [p.grad for p in teacher.parameters() if p.grad is not None]
    assert grads, "no gradient reached the teacher through the generated weights"
    assert all(torch.isfinite(g).all() for g in grads)


# ---------------------------------------------------------------------------
# End-to-end miniature
# ---------------------------------------------------------------------------

def test_teacher_learns_to_drive_sigmapi_student():
    set_seed(0)
    K, width, B = 4, 8, 32
    centroids = torch.randn(K, 3, 1, 1) * 2.0

    def sample_batch():
        y = torch.randint(0, K, (B,))
        x = centroids[y] + 0.5 * torch.randn(B, 3, 16, 16)
        return x, y

    def build_prototype(student, batch):
        x, y = batch
        return image_grid_stats(x, y, num_classes=K, grid=4)

    def forward(student, batch, gen):
        x, y = batch
        return student(x, gen_sigmapi=gen), y

    spec = SigmaPiConvTargetSpec(channels=width)
    # proto from image_grid_stats is [1, 4, K, grid^2 * in_ch]; ConvFilterTeacher
    # is generic over the spatial proto shape.
    teacher = ConvFilterTeacher(spec, proto_channels=4, width=16, sigma_pi=True)
    student = SigmaPiStudent(width=width, num_classes=K, img_size=16)

    result = train_teacher(
        teacher, [student],
        sample_batch=sample_batch, build_prototype=build_prototype, forward=forward,
        steps=150, lr=1e-3,
    )

    first = sum(result.losses[:20]) / 20
    last = sum(result.losses[-20:]) / 20
    assert last < first, f"teacher did not learn: first={first:.3f} last={last:.3f}"
    assert len(result.pi_scales) == 150
    assert result.final_pi_scale is not None and math.isfinite(result.final_pi_scale)
