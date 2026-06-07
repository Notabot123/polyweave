"""Tests for polyweave.ops.signed_log."""

from __future__ import annotations

import torch

from polyweave.ops import signed_log, signed_log1p


def test_signed_log_is_odd():
    x = torch.randn(1000)
    assert torch.allclose(signed_log(x), -signed_log(-x), atol=1e-6)


def test_signed_log1p_is_odd_and_zero_at_origin():
    x = torch.randn(1000)
    assert torch.allclose(signed_log1p(x), -signed_log1p(-x), atol=1e-6)
    assert signed_log1p(torch.zeros(1)).item() == 0.0


def test_signed_log_matches_closed_form():
    x = torch.tensor([2.0, -3.0, 0.5])
    expected = torch.sign(x) * torch.log(torch.abs(x) + 1e-8)
    assert torch.allclose(signed_log(x), expected, atol=1e-7)


def test_signed_log1p_matches_closed_form():
    x = torch.tensor([2.0, -3.0, 0.5])
    expected = torch.sign(x) * torch.log1p(torch.abs(x))
    assert torch.allclose(signed_log1p(x), expected, atol=1e-7)


def test_signed_log_is_differentiable():
    x = torch.randn(64, requires_grad=True)
    signed_log(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
