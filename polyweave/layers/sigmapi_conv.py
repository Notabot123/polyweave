"""Convolutional Sigma-Pi block — the core multiplicative layer of the paper.

Two branches are summed, batch-normalised, and passed through ReLU:

    sigma (additive)        : Conv2d over the spatially-centred input.
    pi    (multiplicative)  : a genuine *geometric product* over the receptive
                              field, formed in log space and exponentiated back.

The two branches are deliberately **structurally parallel**: the pi branch is the
sigma branch with ``x`` replaced by ``log|x|`` and a final ``exp`` —

    sigma = Conv2d( x  - spatial_mean(x) )
    pi    = exp(pi_scale) * exp( clamp( Conv2d( log|x| - spatial_mean(log|x|) ) ) )

so for output channel ``k`` the pi branch realises a weighted geometric product
``prod over (input channel c, kernel offset) |x_c|**w`` over the local receptive
field — a true higher-order (multiplicative) response. This **exponentiates back to
a product**, unlike the deprecated ``tanh(Conv2d(signed_log(x)))`` formulation,
which never exp'd and therefore could not multiply (see
``archive/DEPRECATED_sigmapi_tanh_logspace.md``).

Design choices (agreed 2026-06-07; mirror the dense ``SigmaPiLinear``):

* **Bounded signed exponents.** The pi conv weights are ``max_exponent *
  tanh(pi_weight_raw)`` (default ``max_exponent=0.5``), so each exponent lies in
  ``(-0.5, +0.5)``: signed exponents allow both amplification and division, and the
  ``tanh`` cap keeps the log-space sum well-conditioned. The pi branch has **no
  bias** — an additive offset in log space is a multiplicative constant, already
  supplied by the ``pi_scale`` gate.
* **Geometric-mean normalisation.** ``log|x|`` is centred by its per-channel spatial
  mean (mirroring sigma centring ``x``), making each channel's log-magnitude
  relative to its own spatial average — scale-free under (per-channel) input
  rescaling. (Channel-wise centring is an alternative; spatial centring was chosen
  for exact parallelism with sigma, and BatchNorm absorbs residual cross-channel
  scale.)
* **Clamp.** The log-space accumulation is clamped to ``[-max_log, max_log]``
  (default ``6``) before ``exp`` as an overflow guard.
* **Learnable amplitude gate / recruitment diagnostic.** ``pi_scale`` is a
  per-channel learnable gate initialised to ``-2`` (``exp(-2) ≈ 0.135``);
  ``exp(pi_scale).mean()`` is the multiplicative-recruitment diagnostic of the
  paper. Unchanged from the deprecated block, so the recruitment story carries over.

**Sign handling.** Magnitude-only by default (sign carried by the signed sigma
branch); ``signed_products=True`` enables a flagged, documented sign-vote surrogate.

``eps``, ``pi_scale_init``, ``max_exponent``, ``max_log`` are exposed but default to
the chosen values.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops.signed_log import DEFAULT_EPS

PI_SCALE_INIT: float = -2.0
DEFAULT_MAX_EXPONENT: float = 0.5
DEFAULT_MAX_LOG: float = 6.0


class ConvSigmaPi2d(nn.Module):
    """Sigma-Pi convolutional block with a genuine geometric-product pi branch.

    Args:
        channels: number of input == output channels.
        kernel_size: conv kernel size for both branches (default 3).
        padding: conv padding (default 1, i.e. "same" for kernel 3).
        pi_scale_init: initial value of the per-channel ``pi_scale`` gate
            (default ``-2.0`` so ``exp(pi_scale) ~= 0.135``).
        max_exponent: bound on each log-space exponent; the pi conv weight is
            ``max_exponent * tanh(raw)`` (default ``0.5``).
        max_log: the log-space accumulation is clamped to ``[-max_log, max_log]``
            before exponentiation (default ``6.0``).
        signed_products: if ``True``, multiply the magnitude product by a sign-vote
            surrogate (a flagged ablation; default ``False`` is magnitude-only).
        eps: stabiliser inside ``log(|x| + eps)`` (default ``1e-8``).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        pi_scale_init: float = PI_SCALE_INIT,
        max_exponent: float = DEFAULT_MAX_EXPONENT,
        max_log: float = DEFAULT_MAX_LOG,
        signed_products: bool = False,
        center_product: bool = False,
        eps: float = DEFAULT_EPS,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.padding = padding
        self.max_exponent = float(max_exponent)
        self.max_log = float(max_log)
        self.signed_products = bool(signed_products)
        self.center_product = bool(center_product)
        self.eps = eps

        # Sigma: signed additive branch (carries sign information).
        self.sigma_conv = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        # Pi: raw log-space exponent kernel, squashed to (-max_exponent, max_exponent).
        # No bias (a log-space offset is a multiplicative constant -> the gate's job).
        self.pi_weight_raw = nn.Parameter(
            torch.zeros(channels, channels, kernel_size, kernel_size)
        )
        nn.init.normal_(self.pi_weight_raw, std=0.1)
        # Per-channel learnable amplitude gate / recruitment diagnostic.
        self.pi_scale = nn.Parameter(torch.full((channels, 1, 1), float(pi_scale_init)))
        self.bn = nn.BatchNorm2d(channels)

    def pi_weight(self) -> torch.Tensor:
        """Bounded signed log-space exponent kernel ``max_exponent * tanh(raw)``."""
        return self.max_exponent * torch.tanh(self.pi_weight_raw)

    def _branches(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the (sigma, pi) branch outputs *before* they are summed.

        These are the raw pre-sum, pre-BatchNorm/ReLU branch activations. Shared by
        :meth:`forward` and :meth:`branch_energy` so the two never drift.
        """
        # Sigma: spatially-centred additive branch.
        sigma = self.sigma_conv(x - x.mean(dim=(-2, -1), keepdim=True))

        # Pi: genuine weighted geometric product over the receptive field.
        log_mag = torch.log(x.abs() + self.eps)
        # Geometric-mean normalisation: centre log|x| by its per-channel spatial mean
        # (mirrors sigma centring x); scale-free under per-channel input rescaling.
        log_mag = log_mag - log_mag.mean(dim=(-2, -1), keepdim=True)
        w = self.pi_weight()
        u = F.conv2d(log_mag, w, padding=self.padding)
        u = torch.clamp(u, -self.max_log, self.max_log)
        # ``center_product`` uses expm1(u) = product - 1, exactly 0 when the exponents
        # are 0 (the multiplicative identity), so the branch starts SILENT and the
        # ``pi_scale`` gate recovers its "volume knob" recruitment meaning. The default
        # exp(u) starts at 1 (a constant BatchNorm / the sigma bias can absorb).
        pi_mag = torch.expm1(u) if self.center_product else torch.exp(u)

        if self.signed_products:
            # Flagged ablation: an exponent-magnitude-weighted sign vote
            # sign( Conv2d(sign(x), |w|) ) — an explicit surrogate for the
            # (ill-defined-for-real-exponents) true product of signs. OFF by default.
            sign_x = torch.sign(x)
            sign_x = torch.where(sign_x == 0, torch.ones_like(sign_x), sign_x)
            pi_sign = torch.sign(F.conv2d(sign_x, w.abs(), padding=self.padding))
            pi_sign = torch.where(pi_sign == 0, torch.ones_like(pi_sign), pi_sign)
            pi_mag = pi_mag * pi_sign

        pi = torch.exp(self.pi_scale) * pi_mag
        return sigma, pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma, pi = self._branches(x)
        return F.relu(self.bn(sigma + pi))

    @torch.no_grad()
    def exponent_abs_mean(self) -> float:
        """**Recruitment metric A** — ``mean(|exponent|)`` over the pi weights.

        The geometric product is ``prod |x| ** w``; ``w = 0`` means that factor is
        ``|x| ** 0 = 1`` (a no-op). So the mean absolute exponent measures *how far the
        learned product departs from doing nothing* — it starts ~0 and grows as the
        block recruits multiplicative structure. Unlike ``pi_scale`` this is meaningful
        even when the product starts at the identity (``exp(u)`` form), because it reads
        the product's SHAPE, not its amplitude/volume.
        """
        return self.pi_weight().abs().mean().item()

    @torch.no_grad()
    def branch_energy(self, x: torch.Tensor) -> dict:
        """**Recruitment metric B** — how much each branch moves the output.

        Returns ``{"sigma_rms", "pi_rms", "pi_share"}`` where ``pi_share =
        pi_rms / (sigma_rms + pi_rms)`` on the given batch: the fraction of the
        (pre-BN) output scale carried by the multiplicative branch. ~0 = pi branch
        idle, toward 1 = output dominated by the product. A direct, if noisier,
        complement to metric A (which reads the weights; this reads activations).
        """
        sigma, pi = self._branches(x)
        sigma_rms = sigma.float().pow(2).mean().sqrt().item()
        pi_rms = pi.float().pow(2).mean().sqrt().item()
        denom = sigma_rms + pi_rms
        return {
            "sigma_rms": sigma_rms,
            "pi_rms": pi_rms,
            "pi_share": (pi_rms / denom) if denom > 0 else 0.0,
        }

    @torch.no_grad()
    def pi_scale_mean(self) -> float:
        """Current value of the pi-branch diagnostic ``exp(pi_scale).mean()``."""
        return self.pi_scale.exp().mean().item()

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, max_exponent={self.max_exponent}, "
            f"max_log={self.max_log}, signed_products={self.signed_products}, "
            f"center_product={self.center_product}, eps={self.eps}"
        )
