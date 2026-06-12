"""Regression-quality metrics for activation-space distillation."""

from __future__ import annotations

import torch


@torch.no_grad()
def relative_mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Squared error normalised by target energy: ``||y - ŷ||² / ||y||²``.

    Scale-free and ``1.0`` for the trivial all-zeros predictor (when the target
    has zero mean), so it reads as a fraction of the signal left unexplained.
    """
    num = torch.sum((y_true - y_pred) ** 2)
    den = torch.sum(y_true ** 2).clamp_min(torch.finfo(y_true.dtype).eps)
    return (num / den).item()


@torch.no_grad()
def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Coefficient of determination ``R² = 1 - SS_res / SS_tot``.

    Computed over all elements against the global target mean. ``1.0`` is a
    perfect fit; ``0.0`` matches the constant-mean predictor; negative is worse
    than predicting the mean.
    """
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2).clamp_min(
        torch.finfo(y_true.dtype).eps
    )
    return (1.0 - ss_res / ss_tot).item()


@torch.no_grad()
def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Root mean squared error in the raw activation units.

    Scale-dependent (unlike ``relative_mse`` / ``r2_score``), so it is only
    comparable across candidates fitted to the *same* target block — which is
    exactly how the distillation experiment uses it.
    """
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()


@torch.no_grad()
def cosine_similarity(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Mean per-row cosine similarity between predicted and target vectors.

    Each row (token) contributes the cosine of the angle between its predicted
    and true activation vectors; we average over rows. Complementary to R²: it
    ignores per-token magnitude and reads only *directional* agreement, so a
    layer can track the shape of the output manifold even where it mis-scales.
    ``1.0`` is perfect alignment, ``0.0`` orthogonal.
    """
    return torch.nn.functional.cosine_similarity(
        y_pred, y_true, dim=-1, eps=torch.finfo(y_true.dtype).eps
    ).mean().item()
