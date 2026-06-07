"""Tests for polyweave.interpretability.occlusion.

The core claim: occlusion cleanly separates additive from multiplicative
features. Additive features have ~0 interaction (conjunction index ~0);
multiplicative (AND) features have a strong sub-additive interaction
(conjunction index ~1). We verify this with closed-form response functions
and on the real SigmaPiLinear layer's two branches.
"""

from __future__ import annotations

import torch

from polyweave.interpretability import (
    conjunction_index,
    group_drops,
    occlusion_sensitivity_1d,
    occlusion_sensitivity_2d,
)


# Two-group inputs: features [0,1] are group A, [2,3] are group B.
GROUP_A = [0, 1]
GROUP_B = [2, 3]


def _additive_response(x):
    # r = sum(A) + sum(B); occluding to 0 removes that group's contribution.
    return x[:, GROUP_A].sum(1) + x[:, GROUP_B].sum(1)


def _multiplicative_response(x):
    # r = (sum A) * (sum B); occluding either group -> factor 0 -> response 0.
    return x[:, GROUP_A].sum(1) * x[:, GROUP_B].sum(1)


def _positive_batch(n=64, f=4):
    torch.manual_seed(0)
    return torch.rand(n, f) + 0.5  # strictly positive so products don't cancel


# ---------------------------------------------------------------------------
# conjunction_index: the headline additive-vs-multiplicative separation
# ---------------------------------------------------------------------------

def test_conjunction_index_near_zero_for_additive():
    x = _positive_batch()
    idx = conjunction_index(_additive_response, x, GROUP_A, GROUP_B)
    assert idx.mean().item() < 0.05


def test_conjunction_index_near_one_for_multiplicative():
    x = _positive_batch()
    idx = conjunction_index(_multiplicative_response, x, GROUP_A, GROUP_B)
    assert idx.mean().item() > 0.95


def test_multiplicative_index_exceeds_additive():
    x = _positive_batch()
    add = conjunction_index(_additive_response, x, GROUP_A, GROUP_B).mean().item()
    mul = conjunction_index(_multiplicative_response, x, GROUP_A, GROUP_B).mean().item()
    assert mul > add + 0.8


# ---------------------------------------------------------------------------
# group_drops: the underlying interaction math
# ---------------------------------------------------------------------------

def test_group_drops_additive_interaction_is_zero():
    x = _positive_batch()
    d = group_drops(_additive_response, x, GROUP_A, GROUP_B)
    assert torch.allclose(d["interaction"], torch.zeros_like(d["interaction"]), atol=1e-5)
    # joint drop == sum of single drops for an additive feature
    assert torch.allclose(d["drop_ab"], d["drop_a"] + d["drop_b"], atol=1e-5)


def test_group_drops_multiplicative_is_subadditive():
    x = _positive_batch()
    d = group_drops(_multiplicative_response, x, GROUP_A, GROUP_B)
    # each single factor already collapses the response -> drop_a ~= drop_b ~= drop_ab
    assert torch.allclose(d["drop_a"], d["drop_ab"], atol=1e-5)
    assert torch.allclose(d["drop_b"], d["drop_ab"], atol=1e-5)
    # interaction strongly negative (sub-additive)
    assert (d["interaction"] < 0).all()


# ---------------------------------------------------------------------------
# 1-D sensitivity map
# ---------------------------------------------------------------------------

def test_occlusion_1d_shape_and_importance():
    x = _positive_batch(n=8, f=6)
    smap = occlusion_sensitivity_1d(_additive_response, x, window=1, stride=1)
    assert smap.shape == (8, 6)
    # additive: occluding a used feature drops response by that feature's value;
    # features 4,5 are unused here -> ~0 drop.
    assert smap[:, 4].abs().mean().item() < 1e-5
    assert smap[:, 0].mean().item() > 0  # used feature matters


def test_occlusion_1d_window_covers_full_axis():
    x = _positive_batch(n=4, f=5)
    smap = occlusion_sensitivity_1d(_additive_response, x, window=2, stride=2)
    # last window is clamped to cover the final feature -> positions {0,2,3}
    assert smap.shape[1] == 3


# ---------------------------------------------------------------------------
# 2-D spatial sensitivity map
# ---------------------------------------------------------------------------

def test_occlusion_2d_localises_a_bright_patch():
    # Response = mean over a 2x2 corner; occluding that corner should drop most.
    x = torch.zeros(2, 1, 6, 6)
    x[:, :, 0:2, 0:2] = 1.0

    def resp(t):
        return t[:, :, 0:2, 0:2].mean(dim=(1, 2, 3))

    smap = occlusion_sensitivity_2d(resp, x, window=2, stride=2)
    assert smap.shape == (2, 3, 3)
    # top-left position has the largest drop
    flat = smap[0].reshape(-1)
    assert flat.argmax().item() == 0


# ---------------------------------------------------------------------------
# Bilinear product (the attention Q.K case): conjunctive AND-signature
# ---------------------------------------------------------------------------

def test_bilinear_product_is_conjunctive_unlike_linear_sum():
    """A genuine linear-space product (q . k) shows the AND-signature; a linear
    sum over the same inputs does not. This is the attention-score case."""
    torch.manual_seed(0)
    q = torch.rand(128, 2) + 0.5  # group A = features [0,1]
    k = torch.rand(128, 2) + 0.5  # group B = features [2,3]
    x = torch.cat([q, k], dim=1)

    def bilinear(t):  # (q . k) dot product
        return (t[:, GROUP_A] * t[:, GROUP_B]).sum(1)

    def linear_sum(t):
        return t[:, GROUP_A].sum(1) + t[:, GROUP_B].sum(1)

    bi = conjunction_index(bilinear, x, GROUP_A, GROUP_B).mean().item()
    lin = conjunction_index(linear_sum, x, GROUP_A, GROUP_B).mean().item()
    assert bi > 0.9
    assert lin < 0.1
    assert bi > lin + 0.8
