"""Ensemble metrics: combine member predictions and quantify their diversity.

Pure tensor utilities (no models, no I/O) shared by the ensemble experiment and
reusable by any caller. The unit of currency is a *member-probability stack*
``probs`` of shape ``[M, N, C]`` — ``M`` ensemble members, ``N`` examples, ``C``
classes — typically the per-member softmax outputs over a fixed evaluation set.

Two questions these answer:

* **How good is the ensemble?** :func:`ensemble_probs` averages members into a
  soft vote; :func:`accuracy_from_probs` / :func:`member_accuracies` give the
  ensemble and per-member top-1 accuracy. The *ensemble gain* (ensemble minus
  mean-member accuracy) is the quantity an ensemble is built to buy.
* **How diverse are the members?** A soft vote only helps if members make
  *different* mistakes. :func:`pairwise_disagreement` is the mean fraction of
  examples on which two members' top-1 predictions differ — the simplest
  error-diversity measure (Kuncheva & Whitaker, 2003).
"""

from __future__ import annotations

import torch


def ensemble_probs(probs: torch.Tensor) -> torch.Tensor:
    """Mean soft vote over members: ``[M, N, C] -> [N, C]``."""
    if probs.ndim != 3:
        raise ValueError(f"expected [M, N, C], got shape {tuple(probs.shape)}")
    return probs.mean(dim=0)


def accuracy_from_probs(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy of a ``[N, C]`` probability/logit tensor against ``[N]`` labels."""
    pred = probs.argmax(dim=-1).reshape(-1)
    labels = labels.reshape(-1)
    return (pred == labels).float().mean().item()


def member_accuracies(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-member top-1 accuracy: ``[M, N, C], [N] -> [M]``."""
    preds = probs.argmax(dim=-1)  # [M, N]
    labels = labels.reshape(1, -1)
    return (preds == labels).float().mean(dim=1)


def ensemble_accuracy(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy of the mean soft vote over members."""
    return accuracy_from_probs(ensemble_probs(probs), labels)


def ensemble_gain(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Ensemble accuracy minus mean single-member accuracy (the diversity payoff)."""
    return ensemble_accuracy(probs, labels) - member_accuracies(probs, labels).mean().item()


def pairwise_disagreement(probs: torch.Tensor) -> float:
    """Mean over distinct member pairs of the fraction of examples they disagree on.

    ``probs`` is ``[M, N, C]``; predictions are taken as the per-member argmax.
    Returns a scalar in ``[0, 1]``: 0 when every member predicts identically,
    higher when members make different predictions (error diversity). Undefined
    for a single member (returns 0.0).
    """
    if probs.ndim != 3:
        raise ValueError(f"expected [M, N, C], got shape {tuple(probs.shape)}")
    preds = probs.argmax(dim=-1)  # [M, N]
    M = preds.shape[0]
    if M < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(M):
        for j in range(i + 1, M):
            total += (preds[i] != preds[j]).float().mean().item()
            pairs += 1
    return total / pairs


def disagreement_matrix(probs: torch.Tensor) -> torch.Tensor:
    """Symmetric ``[M, M]`` matrix of pairwise prediction-disagreement rates.

    Entry ``(i, j)`` is the fraction of examples where members ``i`` and ``j``
    differ; the diagonal is 0. Useful for a heatmap of which members are
    redundant vs complementary.
    """
    preds = probs.argmax(dim=-1)  # [M, N]
    M = preds.shape[0]
    out = probs.new_zeros(M, M)
    for i in range(M):
        for j in range(i + 1, M):
            d = (preds[i] != preds[j]).float().mean()
            out[i, j] = d
            out[j, i] = d
    return out
