"""Sigma-Pi conv target spec — generating the weights of a *multiplicative* layer.

Where :class:`Conv2dTargetSpec` (Experiment 2) targets an ordinary additive
convolution, this spec targets a :class:`~polyweave.layers.sigmapi_conv.ConvSigmaPi2d`
block. It is the substrate for the "Sigma-Pi student" further-work experiment:
the teacher no longer generates conventional filters but the two convolutional
weight matrices *inside* a multiplicative block — the additive ``sigma_conv`` and
the log-space ``pi_conv``.

What is generated vs. left learnable
------------------------------------
We generate the two conv pathways (``sigma_conv``/``pi_conv`` weight + bias). We
deliberately do **not** generate:

* ``pi_scale`` — the per-channel multiplicative gate. It is the paper's central
  *diagnostic* and belongs to the student so its recruitment can be measured on
  the student side; generating it would couple the diagnostic to the teacher.
* ``bn`` — BatchNorm carries running statistics that are meaningless to emit from
  a hypernetwork; it stays a normal learnable/buffered layer.

Both decisions keep ``num_params`` a clean function of the conv shapes and leave
the recruitment gate as a free, observable student parameter.

The block is channels-preserving (``in == out == channels``), matching
``ConvSigmaPi2d``.
"""

from __future__ import annotations

from typing import Dict

import torch

from ..layers.sigmapi_conv import ConvSigmaPi2d
from .base import TargetSpec

SigmaPiConvWeights = Dict[str, torch.Tensor]
# {"sigma_weight": [C,C,k,k], "sigma_bias": [C],
#  "pi_weight":    [C,C,k,k], "pi_bias":    [C]}


class SigmaPiConvTargetSpec(TargetSpec):
    """The two conv weight matrices (additive + log-space) of a Sigma-Pi block.

    Args:
        channels: input == output channels of the target ``ConvSigmaPi2d``.
        kernel_size: conv kernel size for both branches (default 3).
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        self.channels = channels
        self.kernel_size = kernel_size
        self._w_shape = (channels, channels, kernel_size, kernel_size)
        self._weight_n = channels * channels * kernel_size * kernel_size

    @property
    def num_params(self) -> int:
        # two conv weight matrices + two biases
        return 2 * self._weight_n + 2 * self.channels

    def pack(self, weights: SigmaPiConvWeights) -> torch.Tensor:
        return torch.cat(
            [
                weights["sigma_weight"].reshape(-1),
                weights["sigma_bias"].reshape(-1),
                weights["pi_weight"].reshape(-1),
                weights["pi_bias"].reshape(-1),
            ]
        )

    def unpack(self, flat: torch.Tensor) -> SigmaPiConvWeights:
        flat = self._check_flat(flat)
        C, n = self.channels, self._weight_n
        i = 0
        sigma_weight = flat[i : i + n].reshape(self._w_shape); i += n
        sigma_bias = flat[i : i + C].reshape(C); i += C
        pi_weight = flat[i : i + n].reshape(self._w_shape); i += n
        pi_bias = flat[i : i + C].reshape(C); i += C
        return {
            "sigma_weight": sigma_weight,
            "sigma_bias": sigma_bias,
            "pi_weight": pi_weight,
            "pi_bias": pi_bias,
        }

    def install(self, into: ConvSigmaPi2d, weights: SigmaPiConvWeights) -> None:
        with torch.no_grad():
            into.sigma_conv.weight.copy_(weights["sigma_weight"])
            into.sigma_conv.bias.copy_(weights["sigma_bias"])
            into.pi_conv.weight.copy_(weights["pi_weight"])
            into.pi_conv.bias.copy_(weights["pi_bias"])

    @torch.no_grad()
    def extract(self, frm: ConvSigmaPi2d) -> SigmaPiConvWeights:
        return {
            "sigma_weight": frm.sigma_conv.weight.detach().clone(),
            "sigma_bias": frm.sigma_conv.bias.detach().clone(),
            "pi_weight": frm.pi_conv.weight.detach().clone(),
            "pi_bias": frm.pi_conv.bias.detach().clone(),
        }
