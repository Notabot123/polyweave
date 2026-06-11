"""Tests for the activation-space distillation harness (capture + fit + metrics)."""

from __future__ import annotations

import torch
import torch.nn as nn

from polyweave import PolyLinear, SigmaPiLinear
from polyweave.distill import (
    IOCapture,
    collect_io,
    cosine_similarity,
    fit_layer,
    r2_score,
    relative_mse,
    rmse,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_metrics_perfect_and_mean_predictors():
    y = torch.randn(100, 4)
    assert relative_mse(y, y) < 1e-7
    assert r2_score(y, y) > 1 - 1e-6
    # The global-mean predictor (matching r2_score's SS_tot) gives R^2 == 0.
    mean_pred = y.mean().expand_as(y)
    assert abs(r2_score(y, mean_pred)) < 1e-5


def test_rmse_and_cosine_metrics():
    y = torch.randn(100, 8)
    # Perfect prediction: zero RMSE, unit cosine.
    assert rmse(y, y) < 1e-7
    assert cosine_similarity(y, y) > 1 - 1e-6
    # RMSE is sqrt of plain MSE in raw units.
    pred = y + 0.5
    assert abs(rmse(y, pred) - 0.5) < 1e-5
    # Cosine reads direction only: a positive rescale leaves it ~unchanged.
    assert cosine_similarity(y, 3.0 * y) > 1 - 1e-6
    # Sign-flipped prediction is anti-aligned.
    assert cosine_similarity(y, -y) < -1 + 1e-5


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def test_iocapture_collects_flattened_pairs():
    mod = nn.Linear(16, 8)
    with IOCapture(mod) as cap:
        for _ in range(3):
            mod(torch.randn(4, 5, 16))  # [B, T, D] -> flattened to rows
    X, Y = cap.pairs()
    assert X.shape == (3 * 4 * 5, 16)
    assert Y.shape == (3 * 4 * 5, 8)
    assert cap.num_rows == 60


def test_iocapture_respects_max_rows():
    mod = nn.Linear(8, 8)
    with IOCapture(mod, max_rows=10) as cap:
        for _ in range(5):
            mod(torch.randn(8, 8))
    X, _ = cap.pairs()
    assert X.shape[0] == 10


def test_collect_io_helper_matches_target():
    mod = nn.Linear(8, 3)
    X = torch.randn(20, 8)
    captured_x, captured_y = collect_io(mod, lambda: mod(X))
    assert torch.allclose(captured_x, X)
    assert torch.allclose(captured_y, mod(X), atol=1e-6)


# ---------------------------------------------------------------------------
# fit_layer
# ---------------------------------------------------------------------------

def _linear_pairs(n=2000, d_in=12, d_out=6, seed=0):
    torch.manual_seed(seed)
    A = torch.randn(d_out, d_in)
    b = torch.randn(d_out)
    X = torch.randn(n, d_in)
    Y = X @ A.T + b
    return X, Y


def test_fit_layer_recovers_linear_map():
    X, Y = _linear_pairs()
    res = fit_layer(nn.Linear(12, 6), X, Y, steps=800, lr=1e-2, eval_every=200)
    assert res.val_r2 > 0.99
    assert res.val_rel_mse < 1e-2
    assert res.num_params == 12 * 6 + 6
    # No recruitment gate on a plain Linear.
    assert res.recruit_curve == []
    assert res.recruit_delta is None


def _bilinear_pairs(n=4000, d=12, seed=1):
    torch.manual_seed(seed)
    u, v = torch.randn(d), torch.randn(d)
    X = torch.randn(n, d)
    Y = ((X @ u) * (X @ v)).unsqueeze(1)
    return X, Y


def test_polylinear_beats_linear_on_bilinear_target():
    """PolyLinear's home turf: an explicit bilinear dot-product form."""
    X, Y = _bilinear_pairs()
    lin = fit_layer(nn.Linear(12, 1), X, Y, steps=1500, lr=1e-2, eval_every=500)
    poly = fit_layer(PolyLinear(12, 1, rank=2), X, Y, steps=1500, lr=1e-2, eval_every=500)
    assert lin.val_r2 < 0.3                 # linear can't model a product
    assert poly.val_r2 > lin.val_r2 + 0.4   # explicit polynomial wins big


def _geometric_product_pairs(n=4000, d=8, seed=2):
    """A target in SigmaPiLinear's function class: a weighted geometric product.

    The rewritten ``SigmaPiLinear`` realises genuine multiplication as
    ``exp(pi_scale) * prod_i |x_i| ** w_i`` with the inputs geometric-mean
    normalised (so the product is scale-free). We therefore build a *scale-free*
    monomial — exponents summing to zero, each within the layer's ``±max_exponent``
    bound — which the layer can represent exactly (set ``w_i = a_i``). This is the
    structure it is actually built to model, distinct from PolyLinear's bilinear
    forms.
    """
    torch.manual_seed(seed)
    a = torch.empty(d).uniform_(-0.3, 0.3)
    a = a - a.mean()                          # sum(a) == 0  ->  scale-free product
    X = torch.randn(n, d)
    log_mag = torch.log(X.abs() + 1e-8)
    Y = torch.exp(log_mag @ a).unsqueeze(1)   # = prod_i |x_i| ** a_i
    return X, Y


def test_sigmapi_fits_its_geometric_product_function_class():
    X, Y = _geometric_product_pairs()
    lin = fit_layer(nn.Linear(8, 1), X, Y, steps=2500, lr=1e-2, eval_every=500)
    sigmapi = fit_layer(SigmaPiLinear(8, 1), X, Y, steps=2500, lr=1e-2, eval_every=500)
    assert lin.val_r2 < 0.5                  # a linear map can't see the log-space product
    assert sigmapi.val_r2 > 0.8              # Sigma-Pi recovers its own monomial form


def test_recruitment_gate_tracked_for_gated_layers():
    X, Y = _bilinear_pairs()
    res = fit_layer(PolyLinear(12, 1, rank=2), X, Y, steps=1000, lr=1e-2, eval_every=250)
    assert len(res.recruit_curve) >= 2
    assert res.recruit_curve[0][0] == 0
    # Fitting a strongly multiplicative target should recruit the gate upward.
    assert res.recruit_delta is not None
    assert res.recruit_delta > 0
