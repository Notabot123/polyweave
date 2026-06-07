"""Tests for polyweave.targets — pack/unpack/install/extract round trips."""

from __future__ import annotations

import torch
import torch.nn as nn

from polyweave.targets import (
    AttentionQKTargetSpec,
    Conv2dTargetSpec,
    FCTargetSpec,
)


# ---------------------------------------------------------------------------
# FC
# ---------------------------------------------------------------------------

def test_fc_num_params():
    spec = FCTargetSpec(in_features=256, out_features=10)
    assert spec.num_params == 256 * 10 + 10


def test_fc_pack_unpack_roundtrip():
    spec = FCTargetSpec(in_features=16, out_features=4)
    flat = torch.randn(spec.num_params)
    again = spec.pack(spec.unpack(flat))
    assert torch.allclose(flat, again, atol=1e-6)


def test_fc_install_extract_roundtrip():
    spec = FCTargetSpec(in_features=16, out_features=4)
    weights = spec.unpack(torch.randn(spec.num_params))
    linear = nn.Linear(16, 4)
    spec.install(linear, weights)
    got = spec.extract(linear)
    assert torch.allclose(got["weight"], weights["weight"], atol=1e-6)
    assert torch.allclose(got["bias"], weights["bias"], atol=1e-6)


# ---------------------------------------------------------------------------
# Conv2d
# ---------------------------------------------------------------------------

def test_conv_num_params():
    spec = Conv2dTargetSpec(out_channels=32, in_channels=3, kernel_size=3)
    assert spec.num_params == 32 * 3 * 3 * 3 + 32


def test_conv_pack_unpack_roundtrip():
    spec = Conv2dTargetSpec(out_channels=8, in_channels=3, kernel_size=3)
    flat = torch.randn(spec.num_params)
    again = spec.pack(spec.unpack(flat))
    assert torch.allclose(flat, again, atol=1e-6)


def test_conv_install_extract_roundtrip():
    spec = Conv2dTargetSpec(out_channels=8, in_channels=3, kernel_size=3)
    weights = spec.unpack(torch.randn(spec.num_params))
    conv = nn.Conv2d(3, 8, 3, padding=1)
    spec.install(conv, weights)
    got = spec.extract(conv)
    assert torch.allclose(got["weight"], weights["weight"], atol=1e-6)
    assert torch.allclose(got["bias"], weights["bias"], atol=1e-6)


# ---------------------------------------------------------------------------
# Attention Q/K
# ---------------------------------------------------------------------------

def test_qk_num_params():
    spec = AttentionQKTargetSpec(d_model=64, n_layers=2)
    assert spec.num_params == 2 * 2 * (64 * 64 + 64)


def test_qk_pack_unpack_roundtrip():
    spec = AttentionQKTargetSpec(d_model=8, n_layers=3)
    flat = torch.randn(spec.num_params)
    again = spec.pack(spec.unpack(flat))
    assert torch.allclose(flat, again, atol=1e-6)


def test_qk_install_extract_roundtrip_and_v_untouched():
    spec = AttentionQKTargetSpec(d_model=8, n_layers=2)
    weights = spec.unpack(torch.randn(spec.num_params))
    attns = [nn.MultiheadAttention(8, 2, batch_first=True) for _ in range(2)]
    # Snapshot the V slice of each module to confirm install leaves it untouched.
    v_before = [a.in_proj_weight[2 * 8 : 3 * 8].detach().clone() for a in attns]
    spec.install(attns, weights)
    got = spec.extract(attns)
    for layer_got, layer_exp in zip(got, weights):
        for key in ("q_weight", "q_bias", "k_weight", "k_bias"):
            assert torch.allclose(layer_got[key], layer_exp[key], atol=1e-6)
    for a, vb in zip(attns, v_before):
        assert torch.allclose(a.in_proj_weight[2 * 8 : 3 * 8], vb, atol=1e-7)


def test_unpack_rejects_wrong_length():
    spec = FCTargetSpec(in_features=4, out_features=2)
    try:
        spec.unpack(torch.randn(spec.num_params + 1))
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong-length flat vector")
