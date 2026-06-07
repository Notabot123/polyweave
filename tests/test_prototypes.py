"""Tests for polyweave.prototypes — statistical builders + learnable encoder."""

from __future__ import annotations

import torch

from polyweave.prototypes import (
    LearnablePrototypeEncoder,
    feature_class_stats,
    image_grid_stats,
    normalize_prototype,
    relation_cross_moments,
)


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

def test_feature_class_stats_shape():
    feats = torch.randn(64, 32)
    y = torch.randint(0, 5, (64,))
    proto = feature_class_stats(feats, y, num_classes=5)
    assert proto.shape == (1, 4, 5, 32)


def test_image_grid_stats_shape():
    x = torch.randn(40, 3, 16, 16)
    y = torch.randint(0, 4, (40,))
    proto = image_grid_stats(x, y, num_classes=4, grid=4)
    assert proto.shape == (1, 4, 4, 4 * 4 * 3)


def test_relation_cross_moments_shape():
    emb = torch.randn(32, 10, 16)
    y = torch.randint(0, 5, (32,))
    proto = relation_cross_moments(emb, y, num_key_slots=5)
    assert proto.shape == (1, 4, 16, 16)


# ---------------------------------------------------------------------------
# Normalisation behaviour
# ---------------------------------------------------------------------------

def test_normalize_is_per_channel_zero_mean_unit_std():
    proto = torch.randn(1, 4, 8, 8) * 5 + 3
    out = normalize_prototype(proto)
    # Each channel standardised across its spatial dims.
    assert torch.allclose(out.mean(dim=(-2, -1)), torch.zeros(1, 4), atol=1e-5)
    assert torch.allclose(out.std(dim=(-2, -1), unbiased=False), torch.ones(1, 4), atol=1e-3)


def test_feature_stats_empty_class_is_zero_rows():
    feats = torch.randn(20, 8)
    y = torch.zeros(20, dtype=torch.long)  # only class 0 populated
    proto = feature_class_stats(feats, y, num_classes=3, normalize=False)
    # classes 1 and 2 have no support -> zero mean/var/kurtosis (channels 0-2).
    # The contrast channel (3) stays nonzero: it measures |mean - global_mean|,
    # and an empty class's zero mean still differs from the global mean.
    assert torch.count_nonzero(proto[:, :3, 1, :]) == 0
    assert torch.count_nonzero(proto[:, :3, 2, :]) == 0


# ---------------------------------------------------------------------------
# Learnable encoder
# ---------------------------------------------------------------------------

def test_learnable_encoder_shape_and_grad():
    enc = LearnablePrototypeEncoder(in_dim=16, num_classes=5, out_channels=4)
    feats = torch.randn(40, 16)
    y = torch.randint(0, 5, (40,))
    proto = enc(feats, y)
    assert proto.shape == (1, 4, 5, 16)
    proto.sum().backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert grads, "encoder parameters received no gradient"
    assert all(torch.isfinite(g).all() for g in grads)


def test_learnable_encoder_custom_embed_dim():
    enc = LearnablePrototypeEncoder(in_dim=16, num_classes=3, out_channels=2, embed_dim=7)
    proto = enc(torch.randn(12, 16), torch.randint(0, 3, (12,)))
    assert proto.shape == (1, 2, 3, 7)
