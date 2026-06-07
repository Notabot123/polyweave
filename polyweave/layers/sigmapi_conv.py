"""Convolutional Sigma-Pi block — the core multiplicative layer of the paper.

Two branches are summed, batch-normalised, and passed through ReLU:

    sigma (additive)        : Conv2d over the zero-centred input.
    pi    (multiplicative)  : exp(pi_scale) * tanh( Conv2d( signed_log(x) ) ).

`pi_scale` is a per-channel learnable gate initialised to ``-2`` so the pi branch
starts subdominant (``exp(-2) ~= 0.135``). Its growth during training,
``exp(pi_scale).mean()``, is the central diagnostic of the paper: it measures how
strongly the model *recruits* multiplicative computation for a given target.

This is a verbatim extraction of the block validated in the CIFAR conv1 and
synthetic attention Q/K experiments; ``eps`` and ``pi_scale_init`` are exposed as
constructor arguments but default to the paper values.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops.signed_log import DEFAULT_EPS, signed_log

PI_SCALE_INIT: float = -2.0


class ConvSigmaPi2d(nn.Module):
    """Signed-log Sigma-Pi convolutional block.

    Args:
        channels: number of input == output channels.
        kernel_size: conv kernel size for both branches (default 3).
        padding: conv padding (default 1, i.e. "same" for kernel 3).
        pi_scale_init: initial value of the per-channel ``pi_scale`` gate
            (default ``-2.0`` so ``exp(pi_scale) ~= 0.135``).
        eps: stabiliser inside ``signed_log`` (default ``1e-8``).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        pi_scale_init: float = PI_SCALE_INIT,
        eps: float = DEFAULT_EPS,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.sigma_conv = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.pi_conv = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.pi_scale = nn.Parameter(torch.full((channels, 1, 1), float(pi_scale_init)))
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sigma: zero-centred additive branch.
        sigma = self.sigma_conv(x - x.mean(dim=(-2, -1), keepdim=True))
        # Pi: signed log-space multiplicative branch.
        z = signed_log(x, self.eps)
        pi = torch.exp(self.pi_scale) * torch.tanh(self.pi_conv(z))
        return F.relu(self.bn(sigma + pi))

    @torch.no_grad()
    def pi_scale_mean(self) -> float:
        """Current value of the pi-branch diagnostic ``exp(pi_scale).mean()``."""
        return self.pi_scale.exp().mean().item()

    def extra_repr(self) -> str:
        return f"channels={self.channels}, eps={self.eps}"
