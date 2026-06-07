"""FC (linear classification head) target spec — Experiment 1."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import TargetSpec

FCWeights = Dict[str, torch.Tensor]  # {"weight": [out, in], "bias": [out]}


class FCTargetSpec(TargetSpec):
    """A linear head ``y = W x + b`` with ``W in R^{out x in}``, ``b in R^{out}``.

    In the paper this is the few-shot classification head whose optimal solution
    is (to first order) the class-centroid NCC rule — the regime where the pi
    branch stays dormant.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self._weight_n = out_features * in_features

    @property
    def num_params(self) -> int:
        return self._weight_n + self.out_features

    def pack(self, weights: FCWeights) -> torch.Tensor:
        return torch.cat([weights["weight"].reshape(-1), weights["bias"].reshape(-1)])

    def unpack(self, flat: torch.Tensor) -> FCWeights:
        flat = self._check_flat(flat)
        weight = flat[: self._weight_n].reshape(self.out_features, self.in_features)
        bias = flat[self._weight_n :].reshape(self.out_features)
        return {"weight": weight, "bias": bias}

    def install(self, into: nn.Linear, weights: FCWeights) -> None:
        with torch.no_grad():
            into.weight.copy_(weights["weight"])
            into.bias.copy_(weights["bias"])

    @torch.no_grad()
    def extract(self, frm: nn.Linear) -> FCWeights:
        return {
            "weight": frm.weight.detach().clone(),
            "bias": frm.bias.detach().clone(),
        }
