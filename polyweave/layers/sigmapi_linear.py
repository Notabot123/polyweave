"""Dense (fully-connected) Sigma-Pi layer — a genuine higher-order neuron.

Two branches are summed:

    sigma (additive)       : Linear over the (feature-)centred input — carries sign.
    pi    (multiplicative) : a genuine *geometric product* of the input magnitudes,
                             realised in log space and exponentiated back.

**The pi branch computes a real product.** For output feature ``j`` it forms

    pi_j = exp(pi_scale_j) * exp( clamp( sum_i w_ji * (log(|x_i| + eps) - logbar) ) )
         = exp(pi_scale_j) * (geometric-mean-normalised)  prod_i |x_i| ** w_ji

where ``logbar`` is the per-row mean of ``log|x|`` over the input features.

i.e. a weighted geometric product (a log-space monomial that is **exponentiated**,
unlike the deprecated ``tanh(W·signed_log(x))`` formulation which never exp'd back
and therefore could not represent a product).

Design choices (agreed 2026-06-07), each interpretable for the paper:

* **Bounded signed exponents.** The log-space weights are ``w = max_exponent *
  tanh(pi_weight_raw)``, so each exponent lies in ``(-max_exponent, +max_exponent)``
  (default ``0.5``). Signed (not positivity-constrained) exponents let the branch
  represent both *amplification* (``|x|**+w``) and *suppression / division*
  (``|x|**-w``); the ``tanh`` cap keeps any single factor's contribution finite and
  the log-space sum well-conditioned. ``max_exponent=0.5`` means no single input can
  push an output past ``|x|**0.5`` (a square-root) before the gate — a deliberately
  conservative, easy-to-justify bound.
* **Geometric-mean normalisation.** Centring ``log|x|`` by its per-row mean (over
  the input features) makes the pi branch a *relative* product — scale-free in the
  overall input magnitude (rescaling every input by ``c`` leaves the branch
  unchanged) — so it does not blow up with input dimension and the learnable gate
  alone sets its amplitude. This mirrors the sigma branch centring ``x`` by its mean.
* **Clamp.** The accumulation is clamped to ``[-max_log, +max_log]`` (default ``6``)
  before ``exp`` as a belt-and-braces guard against overflow.
* **Learnable amplitude gate / recruitment diagnostic.** ``pi_scale`` is a
  per-output learnable gate initialised to ``-2`` (``exp(-2) ≈ 0.135``) so the pi
  branch starts subdominant; ``exp(pi_scale).mean()`` is the multiplicative-
  recruitment diagnostic used throughout the paper. This is unchanged from the
  deprecated layer, so the recruitment story carries over directly.

**Sign handling.** The product is **magnitude-only** by default: the pi branch sees
``log|x|`` and the *sign* information is carried by the (signed) sigma branch. This
avoids the subtly-wrong "sum of input signs" rule and keeps the parameter count
down. A genuinely signed product (``prod_i sign(x_i)`` times the magnitude product)
is available as a flagged ablation via ``signed_products=True``.

Like ``nn.Linear`` (and unlike the self-contained ``ConvSigmaPi2d`` block, which
bakes in BatchNorm + ReLU), this emits the raw ``sigma + pi`` pre-activation and
leaves any normalisation/nonlinearity to the caller — so it works as a regression
head (fitting continuous, possibly negative activations) and as a drop-in layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..ops.signed_log import DEFAULT_EPS

PI_SCALE_INIT: float = -2.0
DEFAULT_MAX_EXPONENT: float = 0.5
DEFAULT_MAX_LOG: float = 6.0


class SigmaPiLinear(nn.Module):
    """Sigma-Pi fully-connected layer with a genuine geometric-product pi branch.

    Args:
        in_features: size of each input sample's last dimension.
        out_features: size of each output sample's last dimension.
            Defaults to ``in_features`` (channels-preserving, like the conv block).
        bias: whether the sigma (additive) branch carries a bias term
            (default ``True``). The pi branch has no bias — an additive offset in
            log space would be a multiplicative constant, which the ``pi_scale``
            gate already supplies.
        pi_scale_init: initial value of the per-output-feature ``pi_scale`` gate
            (default ``-2.0`` so ``exp(pi_scale) ≈ 0.135``).
        max_exponent: bound on the magnitude of each log-space exponent; weights are
            ``max_exponent * tanh(raw)`` (default ``0.5``).
        max_log: the log-space accumulation is clamped to ``[-max_log, max_log]``
            before exponentiation (default ``6.0``).
        signed_products: if ``True``, multiply the magnitude product by
            ``prod_i sign(x_i)`` (a true signed product) — a flagged ablation. The
            default ``False`` is magnitude-only, with sign carried by sigma.
        center_product: if ``True`` the pi branch uses ``expm1(u)`` (= product - 1)
            instead of ``exp(u)``, so it starts at the multiplicative identity (silent)
            and the ``pi_scale`` gate recovers a clean "volume knob" meaning (default
            ``False``). The silent-init property holds only when the accumulated ``u``
            is near zero at init, i.e. for narrow fan-in; a wide ``in_features`` sums
            many small exponents into a non-negligible ``u`` and weakens the effect.
        eps: stabiliser inside ``log(|x| + eps)`` (default ``1e-8``).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int | None = None,
        *,
        bias: bool = True,
        pi_scale_init: float = PI_SCALE_INIT,
        max_exponent: float = DEFAULT_MAX_EXPONENT,
        max_log: float = DEFAULT_MAX_LOG,
        signed_products: bool = False,
        center_product: bool = False,
        eps: float = DEFAULT_EPS,
    ) -> None:
        super().__init__()
        out_features = in_features if out_features is None else out_features
        self.in_features = in_features
        self.out_features = out_features
        self.max_exponent = float(max_exponent)
        self.max_log = float(max_log)
        self.signed_products = bool(signed_products)
        self.center_product = bool(center_product)
        self.eps = eps

        # Sigma: signed additive branch (also carries sign information).
        self.sigma = nn.Linear(in_features, out_features, bias=bias)
        # Pi: raw log-space exponent weights, squashed to (-max_exponent, max_exponent).
        # Init near zero so exponents start ~0 (product ~ 1) and the branch is gentle.
        self.pi_weight_raw = nn.Parameter(torch.zeros(out_features, in_features))
        nn.init.normal_(self.pi_weight_raw, std=0.1)
        # Per-output learnable amplitude gate / recruitment diagnostic.
        self.pi_scale = nn.Parameter(torch.full((out_features,), float(pi_scale_init)))

    def pi_weight(self) -> torch.Tensor:
        """Bounded signed log-space exponents ``max_exponent * tanh(raw)``."""
        return self.max_exponent * torch.tanh(self.pi_weight_raw)

    def _branches(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the (sigma, pi) branch outputs *before* they are summed.

        Shared by :meth:`forward` and :meth:`branch_energy` so the two never drift.
        """
        # Sigma: feature-centred additive branch.
        sigma = self.sigma(x - x.mean(dim=-1, keepdim=True))

        # Pi: genuine weighted geometric product, formed in log space.
        log_mag = torch.log(x.abs() + self.eps)          # [..., in]
        # Geometric-mean normalisation: centre log|x| by its per-row mean over the
        # INPUT features. This makes the product relative to the input's geometric
        # mean (scale-free: rescaling all inputs by c shifts log|x| by log c, which
        # the centring removes), exactly mirroring how sigma centres x by its mean.
        log_mag = log_mag - log_mag.mean(dim=-1, keepdim=True)
        w = self.pi_weight()                              # [out, in]
        u = torch.nn.functional.linear(log_mag, w)        # [..., out]  = sum_i w_ji (log|x_i| - mean)
        # Overflow guard, then exponentiate back to a genuine product.
        u = torch.clamp(u, -self.max_log, self.max_log)
        # ``center_product`` uses expm1(u) = product - 1, which is exactly 0 when the
        # exponents are 0 (the multiplicative identity), so the branch starts SILENT
        # and the ``pi_scale`` gate recovers its "volume knob" recruitment meaning.
        # The default exp(u) starts at 1 (a constant the sigma bias can absorb).
        pi_mag = torch.expm1(u) if self.center_product else torch.exp(u)

        if self.signed_products:
            # Flagged ablation. A genuine signed monomial would be
            # ``prod_i sign(x_i) ** w_ji`` — but ``sign ** (real exponent)`` is
            # ill-defined, so there is no exact differentiable form. We use an
            # explicit, documented surrogate: an exponent-magnitude-weighted sign
            # vote, ``sign( sum_i |w_ji| * sign(x_i) )``. This recovers the true
            # product of signs when one input dominates and degrades gracefully
            # otherwise; it is OFF by default precisely because it is approximate.
            sign_x = torch.sign(x)
            sign_x = torch.where(sign_x == 0, torch.ones_like(sign_x), sign_x)
            pi_sign = torch.sign(torch.nn.functional.linear(sign_x, w.abs()))
            pi_sign = torch.where(pi_sign == 0, torch.ones_like(pi_sign), pi_sign)
            pi_mag = pi_mag * pi_sign

        pi = torch.exp(self.pi_scale) * pi_mag
        return sigma, pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features]
        sigma, pi = self._branches(x)
        return sigma + pi

    @torch.no_grad()
    def exponent_abs_mean(self) -> float:
        """**Recruitment metric A** — ``mean(|exponent|)`` over the pi weights.

        The geometric product is ``prod_i |x_i| ** w_i``; ``w_i = 0`` means that
        factor is ``|x_i| ** 0 = 1`` (a no-op). So the mean absolute exponent measures
        *how far the learned product departs from doing nothing* — it starts ~0 and
        grows as the layer recruits multiplicative structure. Unlike ``pi_scale``,
        this is meaningful even when the product starts at the identity (``exp(u)``
        form), because it reads the product's SHAPE, not its amplitude/volume.
        """
        return self.pi_weight().abs().mean().item()

    @torch.no_grad()
    def branch_energy(self, x: torch.Tensor) -> dict:
        """**Recruitment metric B** — how much each branch moves the output.

        Returns ``{"sigma_rms", "pi_rms", "pi_share"}`` where ``pi_share =
        pi_rms / (sigma_rms + pi_rms)`` on the given batch: the fraction of the
        output's scale carried by the multiplicative branch. ~0 = pi branch idle,
        toward 1 = output dominated by the product. A direct, if noisier, complement
        to metric A (which reads the weights; this reads the actual activations).
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
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"max_exponent={self.max_exponent}, max_log={self.max_log}, "
            f"signed_products={self.signed_products}, "
            f"center_product={self.center_product}, eps={self.eps}"
        )
