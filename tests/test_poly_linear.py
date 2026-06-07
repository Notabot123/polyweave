"""Tests for the factorized degree-2 PolyLinear layer."""

from __future__ import annotations

import torch

from polyweave import PolyLinear
from polyweave.layers.poly_linear import QUAD_SCALE_INIT


def test_shapes_and_default_square():
    layer = PolyLinear(16)
    assert layer.out_features == 16
    x = torch.randn(4, 7, 16)  # leading dims preserved
    assert layer(x).shape == (4, 7, 16)

    rect = PolyLinear(16, 8, rank=4)
    assert rect(torch.randn(3, 16)).shape == (3, 8)


def test_param_count_matches_formula():
    in_f, out_f, rank = 16, 12, 5
    layer = PolyLinear(in_f, out_f, rank=rank, bias=True, symmetric=False)
    expected = (out_f * in_f + out_f)        # linear weight + bias
    expected += 2 * rank * in_f              # U, V
    expected += out_f * rank                 # mix
    expected += out_f                        # quad_scale
    assert sum(p.numel() for p in layer.parameters()) == expected


def test_symmetric_ties_factors_and_saves_params():
    asym = PolyLinear(16, 16, rank=4, symmetric=False)
    sym = PolyLinear(16, 16, rank=4, symmetric=True)
    assert sym.V is sym.U                       # tied
    assert asym.V is not asym.U
    p_asym = sum(p.numel() for p in asym.parameters())
    p_sym = sum(p.numel() for p in sym.parameters())
    assert p_sym == p_asym - 4 * 16             # one fewer factor matrix


def test_rank_zero_is_pure_linear():
    layer = PolyLinear(8, 8, rank=0)
    assert layer.quad_scale is None
    assert layer.quad_scale_mean() == 0.0
    x = torch.randn(5, 8)
    # With rank 0 the output must equal the linear branch exactly.
    assert torch.allclose(layer(x), layer.linear(x))


def test_gate_starts_subdominant():
    layer = PolyLinear(8, 8, rank=4)
    assert abs(layer.quad_scale_mean() - torch.tensor(QUAD_SCALE_INIT).exp().item()) < 1e-6
    # pi_scale_mean alias maps to the same diagnostic (uniform tracking).
    assert layer.pi_scale_mean() == layer.quad_scale_mean()


def test_recovers_known_bilinear_target():
    """A pure rank-1 bilinear target is fit well by PolyLinear, poorly by linear."""
    torch.manual_seed(0)
    d, n = 12, 4000
    u = torch.randn(d)
    v = torch.randn(d)
    X = torch.randn(n, d)
    Y = ((X @ u) * (X @ v)).unsqueeze(1)  # [n, 1], zero linear correlation in expectation

    lin = torch.nn.Linear(d, 1)
    poly = PolyLinear(d, 1, rank=2)

    def fit(model, steps=1500, lr=1e-2):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            torch.nn.functional.mse_loss(model(X), Y).backward()
            opt.step()
        with torch.no_grad():
            ss_res = ((model(X) - Y) ** 2).sum()
            ss_tot = ((Y - Y.mean()) ** 2).sum()
            return (1 - ss_res / ss_tot).item()

    r2_lin = fit(lin)
    r2_poly = fit(poly)
    assert r2_poly > 0.9          # polynomial layer captures the bilinear map
    assert r2_poly > r2_lin + 0.5  # and decisively beats the linear baseline


def test_gradients_flow_to_all_branches():
    layer = PolyLinear(10, 6, rank=3)
    out = layer(torch.randn(8, 10)).sum()
    out.backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
