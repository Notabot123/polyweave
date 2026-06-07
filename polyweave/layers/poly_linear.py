"""Factorized degree-2 polynomial layer — an *explicit* multiplicative baseline.

Where :class:`SigmaPiLinear` realises multiplication implicitly in log-space
(``exp(pi_scale) * tanh(W * signed_log(x))``), this layer realises it
*explicitly* as a low-rank bilinear (quadratic) form, in the spirit of
Factorization Machines (Rendle 2010) and higher-order/product networks:

    y = W x + b  +  exp(quad_scale) * A [ (U x) ⊙ (V x) ]

The linear branch ``W x + b`` matches ``nn.Linear``; the quadratic branch sums
``rank`` rank-1 products ``(u_r · x)(v_r · x)`` and mixes them to the outputs via
``A``. A full dense degree-2 layer would need ``O(in^2)`` coefficients per output
(infeasible at transformer width); the rank-``R`` factorization makes capacity a
tunable knob and keeps the parameter count linear in ``in``.

``quad_scale`` is a per-output-feature learnable gate initialised to ``-2`` — the
same convention as ``SigmaPiLinear.pi_scale`` — so the multiplicative branch
starts subdominant (``exp(-2) ~= 0.135``) and its growth, ``exp(quad_scale).mean()``,
is a directly-comparable *polynomial-recruitment* diagnostic. This lets the two
multiplicative layers be compared on equal footing: same gate, same subdominant
init, differing only in the *form* of the multiplicative branch (signed-log·tanh
vs raw low-rank bilinear).

Like ``SigmaPiLinear`` this is a composable *layer* (raw pre-activation out, no
baked normalisation/activation), so it works as a regression head fitting
continuous targets, and ``out_features`` may differ from ``in_features``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

QUAD_SCALE_INIT: float = -2.0


class PolyLinear(nn.Module):
    """Linear + low-rank quadratic (degree-2 factorization-machine) layer.

    Args:
        in_features: size of each input sample's last dimension.
        out_features: size of each output's last dimension. Defaults to
            ``in_features`` (channels-preserving).
        rank: number of rank-1 bilinear factors in the quadratic branch. ``0``
            disables the quadratic branch entirely (pure linear). Higher rank =
            more multiplicative capacity; controls the parameter budget.
        bias: whether the linear branch carries a bias term (default ``True``).
        symmetric: if ``True`` the two factor matrices are tied (``V = U``),
            giving a true symmetric quadratic form ``sum_r (u_r · x)^2``; if
            ``False`` (default) ``U`` and ``V`` are independent, giving a general
            (asymmetric) bilinear form.
        quad_scale_init: initial per-output gate (default ``-2.0``).

    Parameter count: ``out*in + (bias?out:0) + 2*rank*in + out*rank + out``
    (with ``symmetric=True`` the ``2*rank*in`` term becomes ``rank*in``).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int | None = None,
        *,
        rank: int = 8,
        bias: bool = True,
        symmetric: bool = False,
        quad_scale_init: float = QUAD_SCALE_INIT,
    ) -> None:
        super().__init__()
        out_features = in_features if out_features is None else out_features
        if rank < 0:
            raise ValueError(f"rank must be >= 0, got {rank}")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.symmetric = symmetric

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if rank > 0:
            self.U = nn.Parameter(torch.empty(rank, in_features))
            self.V = self.U if symmetric else nn.Parameter(torch.empty(rank, in_features))
            self.mix = nn.Parameter(torch.empty(out_features, rank))
            self.quad_scale = nn.Parameter(torch.full((out_features,), float(quad_scale_init)))
            self.reset_quadratic_parameters()
        else:
            self.register_parameter("U", None)
            self.register_parameter("mix", None)
            self.register_parameter("quad_scale", None)

    def reset_quadratic_parameters(self) -> None:
        # Small init so rank-1 products start near zero; the gate keeps the branch
        # subdominant regardless, but this avoids large products at step 0.
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.U, -bound, bound)
        if not self.symmetric:
            nn.init.uniform_(self.V, -bound, bound)
        nn.init.uniform_(self.mix, -1.0 / math.sqrt(max(self.rank, 1)), 1.0 / math.sqrt(max(self.rank, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features]
        y = self.linear(x)
        if self.rank > 0:
            ux = torch.nn.functional.linear(x, self.U)  # [..., rank]
            vx = ux if self.symmetric else torch.nn.functional.linear(x, self.V)
            prod = ux * vx  # [..., rank]  (rank-1 bilinear products)
            quad = torch.nn.functional.linear(prod, self.mix)  # [..., out]
            y = y + torch.exp(self.quad_scale) * quad
        return y

    @torch.no_grad()
    def quad_scale_mean(self) -> float:
        """Polynomial-recruitment diagnostic ``exp(quad_scale).mean()``.

        Returns ``0.0`` when the quadratic branch is disabled (``rank == 0``),
        matching "no multiplicative branch".
        """
        if self.rank == 0:
            return 0.0
        return self.quad_scale.exp().mean().item()

    # Alias so generic recruitment-tracking code can treat PolyLinear and
    # SigmaPiLinear uniformly.
    pi_scale_mean = quad_scale_mean

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, symmetric={self.symmetric}"
        )
