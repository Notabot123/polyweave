"""Conv2d filter target spec — Experiment 2 (conv1 generation)."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import TargetSpec

ConvWeights = Dict[str, torch.Tensor]  # {"weight": [O, I, k, k], "bias": [O]}


class Conv2dTargetSpec(TargetSpec):
    """A 2-D convolution kernel + bias.

    Matches the CIFAR conv1 target: ``Conv2d(in_ch, out_ch, kernel)``. The mapping
    from class-conditional image statistics to useful conv1 filters is substantially
    more nonlinear than the FC mapping, and the pi branch grows accordingly.
    """

    def __init__(self, out_channels: int, in_channels: int, kernel_size: int) -> None:
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self._w_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self._weight_n = out_channels * in_channels * kernel_size * kernel_size

    @property
    def num_params(self) -> int:
        return self._weight_n + self.out_channels

    def pack(self, weights: ConvWeights) -> torch.Tensor:
        return torch.cat([weights["weight"].reshape(-1), weights["bias"].reshape(-1)])

    def unpack(self, flat: torch.Tensor) -> ConvWeights:
        flat = self._check_flat(flat)
        weight = flat[: self._weight_n].reshape(self._w_shape)
        bias = flat[self._weight_n :].reshape(self.out_channels)
        return {"weight": weight, "bias": bias}

    def install(self, into: nn.Conv2d, weights: ConvWeights) -> None:
        with torch.no_grad():
            into.weight.copy_(weights["weight"])
            into.bias.copy_(weights["bias"])

    @torch.no_grad()
    def extract(self, frm: nn.Conv2d) -> ConvWeights:
        return {
            "weight": frm.weight.detach().clone(),
            "bias": frm.bias.detach().clone(),
        }
