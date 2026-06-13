"""Tests for the Sigma-Pi layers' ablation flags and recruitment diagnostics.

The core forward/shape/gradient behaviour is covered in ``test_layers.py``; this
file guards the *optional* public surface that the experiments rely on:

* the flagged ablations ``signed_products`` and ``center_product``,
* the ``ConvSigmaPi2d.pi_enabled`` runtime toggle,
* the recruitment metrics ``exponent_abs_mean`` (metric A) and
  ``branch_energy`` (metric B), and
* ``extra_repr``.
"""

from __future__ import annotations

import torch

from polyweave.layers import ConvSigmaPi2d, SigmaPiLinear


# ---------------------------------------------------------------------------
# SigmaPiLinear
# ---------------------------------------------------------------------------

def test_linear_signed_products_forward_and_grad():
    layer = SigmaPiLinear(in_features=16, out_features=8, signed_products=True)
    x = torch.randn(4, 16)
    y = layer(x)
    assert y.shape == (4, 8)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(layer.pi_weight_raw.grad).all()


def test_linear_center_product_starts_near_silent():
    # With center_product=True the pi branch uses expm1(u); at init exponents are
    # ~0 so the product is ~the identity and expm1(~0) ~ 0 => branch starts quiet.
    torch.manual_seed(0)
    x = torch.randn(64, 32)
    centred = SigmaPiLinear(in_features=32, out_features=32, center_product=True)
    plain = SigmaPiLinear(in_features=32, out_features=32, center_product=False)
    assert centred.branch_energy(x)["pi_rms"] < plain.branch_energy(x)["pi_rms"]


def test_linear_exponent_abs_mean_grows_with_weights():
    layer = SigmaPiLinear(in_features=16, out_features=8)
    base = layer.exponent_abs_mean()
    assert base >= 0.0
    with torch.no_grad():
        layer.pi_weight_raw.add_(5.0)  # push exponents toward the tanh cap
    assert layer.exponent_abs_mean() > base


def test_linear_branch_energy_keys_and_range():
    layer = SigmaPiLinear(in_features=16, out_features=8)
    e = layer.branch_energy(torch.randn(4, 16))
    assert set(e) == {"sigma_rms", "pi_rms", "pi_share"}
    assert 0.0 <= e["pi_share"] <= 1.0


def test_linear_extra_repr_mentions_flags():
    r = repr(SigmaPiLinear(in_features=8, signed_products=True))
    assert "signed_products=True" in r
    assert "center_product=False" in r


# ---------------------------------------------------------------------------
# ConvSigmaPi2d
# ---------------------------------------------------------------------------

def test_conv_signed_products_forward_and_grad():
    block = ConvSigmaPi2d(channels=8, signed_products=True)
    block.eval()
    x = torch.randn(4, 8, 16, 16)
    y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    block.train()
    block(x).sum().backward()
    assert torch.isfinite(block.pi_weight_raw.grad).all()


def test_conv_pi_enabled_toggle_changes_output():
    torch.manual_seed(0)
    block = ConvSigmaPi2d(channels=8, pi_scale_init=0.0)  # non-trivial pi amplitude
    block.eval()
    x = torch.randn(4, 8, 16, 16)
    y_on = block(x)
    block.pi_enabled = False
    y_off = block(x)
    assert not torch.allclose(y_on, y_off)


def test_conv_branch_energy_includes_postbn_effect():
    block = ConvSigmaPi2d(channels=8)
    e = block.branch_energy(torch.randn(4, 8, 16, 16))
    assert set(e) == {"sigma_rms", "pi_rms", "pi_share", "pi_effect_postbn"}
    assert e["pi_effect_postbn"] >= 0.0
    # branch_energy must not leave BatchNorm in eval mode if it began training.
    assert block.bn.training


def test_conv_exponent_abs_mean_nonnegative():
    assert ConvSigmaPi2d(channels=8).exponent_abs_mean() >= 0.0


def test_conv_extra_repr_mentions_flags():
    r = repr(ConvSigmaPi2d(channels=8, center_product=True))
    assert "center_product=True" in r
    assert "channels=8" in r
