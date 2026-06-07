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
