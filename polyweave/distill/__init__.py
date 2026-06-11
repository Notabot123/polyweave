"""Activation-space distillation: fit a single layer to a module's I/O map.

The future GPT-2 / Pythia / SwiGLU experiment replaces a transformer feed-forward
sub-block with one position-wise layer, trained by regressing the block's cached
``(input, output)`` activation pairs. This sub-package is the *model-agnostic*
machinery for that:

    capture    — forward-hook collection of a submodule's (input, output) pairs
    metrics    — relative MSE and R^2 (coefficient of determination)
    regression — ``fit_layer``: MSE-regress any nn.Module onto cached pairs,
                 tracking the multiplicative-recruitment gate if the layer has one

Nothing here imports ``transformers``; ``capture`` works on any ``nn.Module``, so
the same harness fits a synthetic target in the tests and a real FFN later.
"""

from __future__ import annotations

from .capture import IOCapture, collect_io
from .metrics import cosine_similarity, r2_score, relative_mse, rmse
from .regression import DistillResult, fit_closed_form_linear, fit_layer

__all__ = [
    "IOCapture",
    "collect_io",
    "cosine_similarity",
    "r2_score",
    "relative_mse",
    "rmse",
    "DistillResult",
    "fit_layer",
    "fit_closed_form_linear",
]
