"""Tests for polyweave.layers.ConvSigmaPi2d."""

from __future__ import annotations

import math

import torch

from polyweave.layers import ConvSigmaPi2d, SigmaPiLinear


def test_forward_preserves_shape():
    block = ConvSigmaPi2d(channels=8)
    block.eval()  # avoid BatchNorm batch-size-1 issues
    x = torch.randn(4, 8, 16, 16)
    y = block(x)
    assert y.shape == x.shape


def test_pi_scale_initialises_subdominant():
    block = ConvSigmaPi2d(channels=8)
    # init -2 -> exp(-2) ~= 0.1353
    assert math.isclose(block.pi_scale_mean(), math.exp(-2.0), rel_tol=1e-5)
    assert block.pi_scale.shape == (8, 1, 1)


def test_custom_pi_scale_init():
    block = ConvSigmaPi2d(channels=4, pi_scale_init=0.0)
    assert math.isclose(block.pi_scale_mean(), 1.0, rel_tol=1e-6)


def test_pi_scale_receives_gradient():
    block = ConvSigmaPi2d(channels=8)
    block.train()
    x = torch.randn(4, 8, 16, 16)
    block(x).sum().backward()
    assert block.pi_scale.grad is not None
    assert torch.isfinite(block.pi_scale.grad).all()


def test_output_is_nonnegative_after_relu():
    block = ConvSigmaPi2d(channels=8)
    block.eval()
    y = block(torch.randn(4, 8, 16, 16))
    assert (y >= 0).all()


# ---------------------------------------------------------------------------
# SigmaPiLinear — dense analog
# ---------------------------------------------------------------------------

def test_linear_default_is_channels_preserving():
    layer = SigmaPiLinear(in_features=16)
    assert layer.out_features == 16
    x = torch.randn(4, 16)
    assert layer(x).shape == (4, 16)


def test_linear_custom_out_features_and_extra_dims():
    layer = SigmaPiLinear(in_features=16, out_features=8)
    # works on [..., in_features] with leading batch/sequence dims preserved.
    x = torch.randn(4, 10, 16)
    assert layer(x).shape == (4, 10, 8)


def test_linear_pi_scale_initialises_subdominant():
    layer = SigmaPiLinear(in_features=16, out_features=8)
    assert math.isclose(layer.pi_scale_mean(), math.exp(-2.0), rel_tol=1e-5)
    assert layer.pi_scale.shape == (8,)


def test_linear_custom_pi_scale_init():
    layer = SigmaPiLinear(in_features=4, pi_scale_init=0.0)
    assert math.isclose(layer.pi_scale_mean(), 1.0, rel_tol=1e-6)


def test_linear_no_baked_activation_allows_negative_outputs():
    # Unlike the conv block (ReLU baked in), the dense layer emits a raw
    # pre-activation so it can regress to continuous (incl. negative) targets.
    torch.manual_seed(0)
    layer = SigmaPiLinear(in_features=32, out_features=32)
    y = layer(torch.randn(64, 32))
    assert (y < 0).any()


def test_linear_branches_and_pi_scale_receive_gradient():
    layer = SigmaPiLinear(in_features=16, out_features=8)
    layer(torch.randn(4, 16)).sum().backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all()
