"""Diagnostics: the pi-scale gate and ensemble (dis)agreement.

These are the two measurement stories of the project:

* ``pi_scale_mean`` — how strongly a Sigma-Pi block recruits its multiplicative
  branch. Growth of this scalar across training is the paper's central diagnostic.
* the ``ensemble_*`` helpers — for comparing a *vanilla*-teacher student ensemble
  against a *Sigma-Pi*-teacher student ensemble. Disagreement rate captures whether
  the two teachers induce qualitatively different students, beyond mean accuracy.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Pi-scale diagnostic
# ---------------------------------------------------------------------------

@torch.no_grad()
def pi_scale_mean(obj: object) -> float:
    """``exp(pi_scale).mean()`` for a Sigma-Pi block or any module containing one.

    Accepts a raw ``pi_scale`` Parameter/Tensor, an object with a ``.pi_scale``
    attribute, or an ``nn.Module`` somewhere inside which a ``pi_scale`` parameter
    lives (the first one found is used).
    """
    if isinstance(obj, torch.Tensor):
        return obj.exp().mean().item()
    pi = getattr(obj, "pi_scale", None)
    if isinstance(pi, torch.Tensor):
        return pi.exp().mean().item()
    if isinstance(obj, nn.Module):
        for name, param in obj.named_parameters():
            if name.split(".")[-1] == "pi_scale":
                return param.exp().mean().item()
    raise ValueError("no pi_scale parameter found on the given object")


# ---------------------------------------------------------------------------
# Ensemble (dis)agreement
# ---------------------------------------------------------------------------

def _stack_preds(preds: Sequence[torch.Tensor]) -> torch.Tensor:
    """Stack a list of [N] integer prediction tensors into an [M, N] long tensor."""
    if len(preds) == 0:
        raise ValueError("need at least one prediction tensor")
    stacked = torch.stack([p.reshape(-1).long() for p in preds], dim=0)
    return stacked  # [M models, N samples]


def disagreement_rate(preds: Sequence[torch.Tensor]) -> float:
    """Fraction of samples on which the ensemble members are not unanimous.

    Args:
        preds: list of ``M`` prediction tensors, each ``[N]`` of class indices.

    Returns:
        Scalar in ``[0, 1]``; 0 means all models always agree.
    """
    stacked = _stack_preds(preds)
    if stacked.shape[0] == 1:
        return 0.0
    # A sample is "disagreed" if any member differs from the first member.
    differs = (stacked != stacked[0:1]).any(dim=0)
    return differs.float().mean().item()


def pairwise_disagreement(preds: Sequence[torch.Tensor]) -> float:
    """Mean fraction of differing predictions averaged over all model pairs."""
    stacked = _stack_preds(preds)
    M = stacked.shape[0]
    if M < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(M):
        for j in range(i + 1, M):
            total += (stacked[i] != stacked[j]).float().mean().item()
            pairs += 1
    return total / pairs


def majority_vote(preds: Sequence[torch.Tensor], num_classes: int) -> torch.Tensor:
    """Hard-voting ensemble prediction. Ties are broken by lowest class index."""
    stacked = _stack_preds(preds)  # [M, N]
    N = stacked.shape[1]
    votes = torch.zeros(N, num_classes, dtype=torch.long, device=stacked.device)
    for m in range(stacked.shape[0]):
        votes.scatter_add_(1, stacked[m].unsqueeze(1), torch.ones(N, 1, dtype=torch.long, device=stacked.device))
    return votes.argmax(dim=1)


def ensemble_accuracy(
    preds: Sequence[torch.Tensor], targets: torch.Tensor, num_classes: int
) -> float:
    """Majority-vote ensemble accuracy against ``targets`` (``[N]`` class indices)."""
    vote = majority_vote(preds, num_classes)
    return (vote == targets.reshape(-1).long()).float().mean().item()


def mean_accuracy(preds: Sequence[torch.Tensor], targets: torch.Tensor) -> float:
    """Mean per-member accuracy (the baseline the ensemble is compared against)."""
    t = targets.reshape(-1).long()
    accs: List[float] = [(p.reshape(-1).long() == t).float().mean().item() for p in preds]
    return sum(accs) / len(accs)
