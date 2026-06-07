"""Dense (fully-connected) Sigma-Pi layer — the vector analog of ConvSigmaPi2d.

Two branches are summed:

    sigma (additive)       : Linear over the (feature-)centred input.
    pi    (multiplicative) : exp(pi_scale) * tanh( Linear( signed_log(x) ) ).

``pi_scale`` is a per-output-feature learnable gate initialised to ``-2`` so the
pi branch starts subdominant (``exp(-2) ~= 0.135``); its growth during training,
``exp(pi_scale).mean()``, is the same multiplicative-recruitment diagnostic used
throughout the paper.

Unlike :class:`ConvSigmaPi2d` — which is a self-contained *block* baking in
BatchNorm + ReLU — this is a composable *layer* analogous to ``nn.Linear``: it
emits the raw ``sigma + pi`` pre-activation and leaves any normalisation or
nonlinearity to the caller. That keeps it usable as a regression head (e.g.
fitting continuous activations, which a baked-in ReLU would clip) and as a drop-in
inside larger modules. It also need not be channels-preserving: ``out_features``
may differ from ``in_features``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..ops.signed_log import DEFAULT_EPS, signed_log

PI_SCALE_INIT: float = -2.0


class SigmaPiLinear(nn.Module):
    """Signed-log Sigma-Pi fully-connected layer.

    Args:
        in_features: size of each input sample's last dimension.
        out_features: size of each output sample's last dimension.
            Defaults to ``in_features`` (channels-preserving, like the conv block).
        bias: whether the two linear branches carry bias terms (default ``True``).
        pi_scale_init: initial value of the per-output-feature ``pi_scale`` gate
            (default ``-2.0`` so ``exp(pi_scale) ~= 0.135``).
        eps: stabiliser inside ``signed_log`` (default ``1e-8``).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int | None = None,
        *,
        bias: bool = True,
        pi_scale_init: float = PI_SCALE_INIT,
        eps: float = DEFAULT_EPS,
    ) -> None:
        super().__init__()
        out_features = in_features if out_features is None else out_features
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.sigma = nn.Linear(in_features, out_features, bias=bias)
        self.pi = nn.Linear(in_features, out_features, bias=bias)
        self.pi_scale = nn.Parameter(torch.full((out_features,), float(pi_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features]
        # Sigma: feature-centred additive branch.
        sigma = self.sigma(x - x.mean(dim=-1, keepdim=True))
        # Pi: signed log-space multiplicative branch.
        z = signed_log(x, self.eps)
        pi = torch.exp(self.pi_scale) * torch.tanh(self.pi(z))
        return sigma + pi

    @torch.no_grad()
    def pi_scale_mean(self) -> float:
        """Current value of the pi-branch diagnostic ``exp(pi_scale).mean()``."""
        return self.pi_scale.exp().mean().item()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"eps={self.eps}"
        )
