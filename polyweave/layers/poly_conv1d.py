"""Factorized degree-2 polynomial 1-D convolutional block.

Extends :class:`~polyweave.layers.poly_linear.PolyLinear`'s sum-of-products idea to
1-D temporal convolution, making it applicable to EEG, vibration, and other
time-series signals.

For each output channel ``k``, the block computes:

    sigma(x) = Conv1d(x)                                   (linear branch)
    quad(x)  = A [ (U_conv * x) ⊙ (V_conv * x) ]          (quadratic branch)
    y        = BN( sigma(x) + exp(quad_scale) * quad(x) )  (gated sum → BN → ReLU)

Both ``U_conv`` and ``V_conv`` are ``Conv1d(channels, rank, kernel_size)``.  The rank-R
factorization mirrors :class:`PolyLinear`: full degree-2 over a receptive field would
need ``O(C^2 k^2)`` parameters per output; the factorization keeps capacity linear in
``C`` and ``k`` and makes ``rank`` an explicit capacity knob.

The ``quad_scale`` gate is a per-channel scalar initialised to ``-2.0``
(``exp(-2) ≈ 0.135``), so the quadratic branch starts subdominant and
``exp(quad_scale).mean()`` is a directly comparable **polynomial-recruitment
diagnostic** consistent with ``PolyLinear.quad_scale_mean()`` and
``ConvSigmaPi2d.pi_scale_mean()``.

Structural choices:

* **Causal padding option.** Setting ``causal=True`` pads the left by
  ``(kernel_size - 1)`` and drops the right end, making the block
  non-anticipatory for online/streaming scenarios.  Default ``causal=False``
  uses symmetric "same" padding (matching ``ConvSigmaPi2d``).
* **No bias in factor convolutions.** The linear branch carries a bias; the
  factor convolutions ``U_conv`` / ``V_conv`` have no bias so the quadratic
  term is zero when ``x == 0`` (a cleaner multiplicative-identity analogue).
* **BatchNorm + ReLU baked in.** Matches ``ConvSigmaPi2d`` so the block is a
  drop-in replacement in temporal CNN backbones.
* **``quad_enabled`` runtime switch.** Toggle off to measure the quadratic
  branch's functional contribution on a trained block, identical to
  ``ConvSigmaPi2d.pi_enabled``.

Typical use::

    block = PolyConv1d(channels=64, kernel_size=7, rank=16)
    y = block(x)   # x: (B, 64, T) → y: (B, 64, T)

For a classifier stack on EEG or suspension signals, compose several blocks
then global-average-pool::

    import torch.nn as nn
    model = nn.Sequential(
        PolyConv1d(in_ch, kernel_size=7, rank=8),
        PolyConv1d(in_ch, kernel_size=5, rank=8),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(in_ch, n_classes),
    )
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

QUAD_SCALE_INIT: float = -2.0


class PolyConv1d(nn.Module):
    """Linear + low-rank quadratic 1-D convolutional block.

    Args:
        channels: number of input (and output) channels.  The block is
            channels-preserving so it can be stacked without projection layers.
        kernel_size: temporal receptive field for all convolutions (default 7).
        rank: number of rank-1 bilinear factors in the quadratic branch.
            ``rank=0`` disables the quadratic branch (pure linear conv + BN + ReLU).
        stride: stride for the linear (sigma) convolution; factor convolutions
            always stride 1 so the quadratic term stays at the original resolution
            before the gate-weighted mix (default 1).
        causal: if ``True``, apply left-only padding so the block is
            non-anticipatory (default ``False``).
        quad_scale_init: initial per-channel gate (default ``-2.0``).
        eps: BatchNorm epsilon (default ``1e-5``).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        rank: int = 8,
        stride: int = 1,
        causal: bool = False,
        quad_scale_init: float = QUAD_SCALE_INIT,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.rank = rank
        self.stride = stride
        self.causal = causal

        # Symmetric "same" padding or causal left-padding.
        self._sym_pad = kernel_size // 2
        self._causal_pad = kernel_size - 1  # left-only; trim right in forward

        # Linear (sigma) branch — carries bias and the "sign" information.
        sigma_pad = 0 if causal else self._sym_pad
        self.sigma_conv = nn.Conv1d(
            channels, channels, kernel_size,
            stride=stride, padding=sigma_pad, bias=True,
        )

        if rank > 0:
            # Factor convolutions: no bias, no stride (operate at input resolution).
            self.U_conv = nn.Conv1d(channels, rank, kernel_size,
                                    padding=self._sym_pad, bias=False)
            self.V_conv = nn.Conv1d(channels, rank, kernel_size,
                                    padding=self._sym_pad, bias=False)
            # Mix rank-1 products back to channels.
            self.mix = nn.Conv1d(rank, channels, kernel_size=1, bias=False)
            # Per-channel gating / recruitment diagnostic.
            self.quad_scale = nn.Parameter(
                torch.full((1, channels, 1), float(quad_scale_init))
            )
            self._reset_factor_parameters()
        else:
            self.register_parameter("quad_scale", None)

        self.bn = nn.BatchNorm1d(channels, eps=eps)
        self.quad_enabled: bool = True

    def _reset_factor_parameters(self) -> None:
        import math
        bound = 1.0 / math.sqrt(self.channels * self.kernel_size)
        nn.init.uniform_(self.U_conv.weight, -bound, bound)
        nn.init.uniform_(self.V_conv.weight, -bound, bound)
        nn.init.uniform_(
            self.mix.weight,
            -1.0 / max(self.rank, 1) ** 0.5,
            1.0 / max(self.rank, 1) ** 0.5,
        )

    def _apply_causal_pad(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        """Pad left by (kernel - 1), apply conv with no internal padding."""
        x = F.pad(x, (self._causal_pad, 0))
        return conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        if self.causal:
            sigma = self._apply_causal_pad(x, self.sigma_conv)
        else:
            sigma = self.sigma_conv(x)

        if self.rank > 0 and self.quad_enabled:
            # Factor convolutions always at input resolution (symmetric padding).
            ux = self.U_conv(x)   # (B, rank, T)
            vx = self.V_conv(x)   # (B, rank, T)
            quad = self.mix(ux * vx)  # (B, C, T)

            # Align spatial length to sigma in case stride != 1.
            if sigma.shape[-1] != quad.shape[-1]:
                quad = quad[..., : sigma.shape[-1]]

            sigma = sigma + torch.exp(self.quad_scale) * quad

        return F.relu(self.bn(sigma))

    # ------------------------------------------------------------------
    # Diagnostics — same interface as PolyLinear and ConvSigmaPi2d
    # ------------------------------------------------------------------

    @torch.no_grad()
    def quad_scale_mean(self) -> float:
        """Polynomial-recruitment diagnostic ``exp(quad_scale).mean()``.

        Returns ``0.0`` when the quadratic branch is disabled (``rank == 0``).
        Directly comparable to ``PolyLinear.quad_scale_mean()`` and
        ``ConvSigmaPi2d.pi_scale_mean()``.
        """
        if self.rank == 0:
            return 0.0
        return self.quad_scale.exp().mean().item()

    pi_scale_mean = quad_scale_mean  # uniform alias for generic tracking code

    @torch.no_grad()
    def branch_energy(self, x: torch.Tensor) -> dict:
        """Pre- and post-BN energy split between linear and quadratic branches.

        Returns ``{"sigma_rms", "quad_rms", "quad_share", "quad_effect_postbn"}``.

        ``quad_effect_postbn`` is the honest post-BN functional contribution:
        relative L2 change in the block output when the quadratic branch is
        removed, measured in eval mode so running stats are unchanged.
        """
        # Compute branches separately.
        if self.causal:
            sigma = self._apply_causal_pad(x, self.sigma_conv)
        else:
            sigma = self.sigma_conv(x)

        quad = torch.zeros_like(sigma)
        if self.rank > 0:
            ux = self.U_conv(x)
            vx = self.V_conv(x)
            q = self.mix(ux * vx)
            if sigma.shape[-1] != q.shape[-1]:
                q = q[..., : sigma.shape[-1]]
            quad = torch.exp(self.quad_scale) * q

        sigma_rms = sigma.float().pow(2).mean().sqrt().item()
        quad_rms = quad.float().pow(2).mean().sqrt().item()
        denom = sigma_rms + quad_rms

        was_training = self.bn.training
        self.bn.eval()
        y_on = F.relu(self.bn(sigma + quad)).float()
        y_off = F.relu(self.bn(sigma)).float()
        if was_training:
            self.bn.train()

        on_rms = y_on.pow(2).mean().sqrt()
        effect = ((y_on - y_off).pow(2).mean().sqrt() / on_rms).item() if on_rms > 0 else 0.0

        return {
            "sigma_rms": sigma_rms,
            "quad_rms": quad_rms,
            "quad_share": (quad_rms / denom) if denom > 0 else 0.0,
            "quad_effect_postbn": effect,
        }

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, kernel_size={self.kernel_size}, "
            f"rank={self.rank}, stride={self.stride}, causal={self.causal}"
        )
