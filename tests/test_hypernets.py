"""Tests for polyweave.hypernets — teacher output contracts and pi diagnostic."""

from __future__ import annotations

import torch

from polyweave.hypernets import ConvFilterTeacher, FCMapTeacher, QKMapTeacher
from polyweave.targets import AttentionQKTargetSpec, Conv2dTargetSpec, FCTargetSpec


# ---------------------------------------------------------------------------
# Vector-head teacher (conv filter)
# ---------------------------------------------------------------------------

def test_conv_filter_teacher_output_matches_spec():
    spec = Conv2dTargetSpec(out_channels=32, in_channels=3, kernel_size=3)
    teacher = ConvFilterTeacher(spec, proto_channels=4)
    proto = torch.randn(1, 4, 10, 48)
    gen = teacher(proto)
    assert gen["weight"].shape == (32, 3, 3, 3)
    assert gen["bias"].shape == (32,)
    # Round-trips through the spec's pack.
    assert spec.pack(gen).numel() == spec.num_params


def test_conv_filter_teacher_works_with_fc_spec():
    spec = FCTargetSpec(in_features=64, out_features=10)
    teacher = ConvFilterTeacher(spec, proto_channels=4)
    gen = teacher(torch.randn(1, 4, 10, 64))
    assert gen["weight"].shape == (10, 64)
    assert gen["bias"].shape == (10,)


# ---------------------------------------------------------------------------
# FC map-head teacher
# ---------------------------------------------------------------------------

def test_fc_map_teacher_shapes():
    teacher = FCMapTeacher(num_classes=10, feature_dim=256, proto_channels=4)
    gen = teacher(torch.randn(1, 4, 10, 256))
    assert gen["weight"].shape == (10, 256)
    assert gen["bias"].shape == (10,)


# ---------------------------------------------------------------------------
# QK map-head teacher
# ---------------------------------------------------------------------------

def test_qk_map_teacher_shapes():
    teacher = QKMapTeacher(d_model=64, n_layers=2, proto_channels=4)
    layers = teacher(torch.randn(1, 4, 64, 64))
    assert len(layers) == 2
    for layer in layers:
        assert layer["q_weight"].shape == (64, 64)
        assert layer["k_weight"].shape == (64, 64)
        assert layer["q_bias"].shape == (64,)
        assert layer["k_bias"].shape == (64,)


def test_qk_teacher_output_installs_via_spec():
    import torch.nn as nn

    teacher = QKMapTeacher(d_model=8, n_layers=2, proto_channels=4)
    layers = teacher(torch.randn(1, 4, 8, 8))
    spec = AttentionQKTargetSpec(d_model=8, n_layers=2)
    attns = [nn.MultiheadAttention(8, 2, batch_first=True) for _ in range(2)]
    spec.install(attns, layers)  # should not raise


# ---------------------------------------------------------------------------
# pi-scale diagnostic
# ---------------------------------------------------------------------------

def test_vanilla_teacher_has_no_pi_scale():
    spec = Conv2dTargetSpec(out_channels=8, in_channels=3, kernel_size=3)
    assert ConvFilterTeacher(spec, sigma_pi=False).pi_scale_mean() is None
    assert FCMapTeacher(5, 16, sigma_pi=False).pi_scale_mean() is None
    assert QKMapTeacher(8, 1, sigma_pi=False).pi_scale_mean() is None


def test_sigmapi_teacher_reports_pi_scale_near_init():
    import math

    spec = Conv2dTargetSpec(out_channels=8, in_channels=3, kernel_size=3)
    for teacher in (
        ConvFilterTeacher(spec, sigma_pi=True),
        FCMapTeacher(5, 16, sigma_pi=True),
        QKMapTeacher(8, 1, sigma_pi=True),
    ):
        assert math.isclose(teacher.pi_scale_mean(), math.exp(-2.0), rel_tol=1e-4)
