"""Fit a single layer to cached activation pairs by MSE regression.

This is the comparison engine for the FFN-distillation experiment: given the same
``(X, Y)`` pairs and a parameter budget, fit a vanilla MLP, a ``SigmaPiLinear``,
and a ``PolyLinear`` and compare their held-out fit + recruitment. The loop is
deliberately generic — any ``nn.Module`` mapping ``[N, in] -> [N, out]`` works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import r2_score, relative_mse


@dataclass
class DistillResult:
    """Outcome of fitting one layer to activation pairs."""

    train_losses: List[Tuple[int, float]] = field(default_factory=list)  # (step, train MSE)
    val_mse: float = float("nan")
    val_rel_mse: float = float("nan")
    val_r2: float = float("nan")
    recruit_curve: List[Tuple[int, float]] = field(default_factory=list)  # (step, gate mean)
    num_params: int = 0

    @property
    def recruit_delta(self) -> Optional[float]:
        """Final − initial recruitment gate, or ``None`` if not tracked."""
        if len(self.recruit_curve) < 2:
            return None
        return self.recruit_curve[-1][1] - self.recruit_curve[0][1]


def _recruit_fn(layer: nn.Module) -> Optional[Callable[[], float]]:
    """Return the layer's recruitment diagnostic (``pi_scale_mean``), if any."""
    fn = getattr(layer, "pi_scale_mean", None) or getattr(layer, "quad_scale_mean", None)
    return fn if callable(fn) else None


def fit_layer(
    layer: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    steps: int = 2000,
    lr: float = 1e-3,
    batch_size: int = 256,
    weight_decay: float = 0.0,
    val_frac: float = 0.2,
    eval_every: int = 100,
    device: str = "cpu",
    seed: int = 0,
    log_fn: Optional[Callable[[str], None]] = None,
) -> DistillResult:
    """MSE-regress ``layer`` onto ``(X, Y)`` and report held-out fit + recruitment.

    The last ``val_frac`` of the rows form a fixed validation split (the inputs
    are assumed already shuffled at capture time — tokens are independent rows).

    Args:
        layer: module mapping ``[batch, in] -> [batch, out]``; trained in place.
        X, Y: activation pairs, shapes ``[N, in]`` and ``[N, out]``.
        steps: optimisation steps (minibatches).
        lr, weight_decay: AdamW hyperparameters.
        batch_size: minibatch size sampled (with replacement) from the train split.
        val_frac: fraction of rows held out for validation metrics.
        eval_every: record train loss + recruitment gate every this many steps.
        device: compute device.
        seed: seed for minibatch sampling.
        log_fn: optional progress sink.

    Returns:
        A :class:`DistillResult` with the train-loss curve, final validation
        ``rel_mse`` / ``R²``, the recruitment-gate curve (if the layer exposes
        ``pi_scale_mean`` / ``quad_scale_mean``), and the layer's parameter count.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    layer = layer.to(device)
    X = X.to(device)
    Y = Y.to(device)

    n = X.shape[0]
    n_val = max(1, int(round(n * val_frac)))
    n_train = max(1, n - n_val)
    Xtr, Ytr = X[:n_train], Y[:n_train]
    Xva, Yva = X[n_train:], Y[n_train:]

    recruit = _recruit_fn(layer)
    result = DistillResult(num_params=sum(p.numel() for p in layer.parameters()))

    opt = torch.optim.AdamW(layer.parameters(), lr=lr, weight_decay=weight_decay)

    def record(step: int) -> None:
        layer.eval()
        with torch.no_grad():
            train_mse = F.mse_loss(layer(Xtr), Ytr).item()
        result.train_losses.append((step, train_mse))
        if recruit is not None:
            result.recruit_curve.append((step, recruit()))
        if log_fn is not None:
            msg = f"step {step:5d}  train_mse={train_mse:.5e}"
            if recruit is not None:
                msg += f"  gate={result.recruit_curve[-1][1]:.5f}"
            log_fn(msg)
        layer.train()

    record(0)
    layer.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n_train, (min(batch_size, n_train),), generator=g)
        xb, yb = Xtr[idx], Ytr[idx]
        opt.zero_grad()
        loss = F.mse_loss(layer(xb), yb)
        loss.backward()
        opt.step()
        if eval_every and step % eval_every == 0:
            record(step)

    if not result.train_losses or result.train_losses[-1][0] != steps:
        record(steps)

    layer.eval()
    with torch.no_grad():
        pred_va = layer(Xva)
    result.val_mse = F.mse_loss(pred_va, Yva).item()
    result.val_rel_mse = relative_mse(Yva, pred_va)
    result.val_r2 = r2_score(Yva, pred_va)
    return result
