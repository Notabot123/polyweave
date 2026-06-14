"""Differentiable fuzzy-logic gates — Boolean operations on soft truth values.

Truth values live in ``[0, 1]``. On the Boolean corners ``{0, 1}`` every gate here
reduces to the exact truth table; in between it interpolates smoothly, so the gates
are differentiable and compose into trainable logic circuits.

The **product t-norm** is the default — and it is exactly the multiplicative (Pi)
branch this library is built around: a fuzzy AND ``a * b`` *is* a product neuron.
That is the bridge between logic and Sigma-Pi computation, and it is why a single
multiplicative neuron solves XOR, the textbook problem a linear unit cannot:

    xor(a, b) = or(a, b) - and(a, b) = (a + b - ab) - ab = a + b - 2ab

— a linear term plus a bilinear product, i.e. a degree-2 (Sigma-Pi / poly) neuron.
See :class:`polyweave.layers.PolyLinear` for the learnable version and
:func:`polyweave.ops.radbas` for the radial-basis route.

t-norms
-------
The conjunction ("AND") uses one of:

* ``"product"`` (default): ``and = a*b``, ``or = a + b - a*b`` (probabilistic sum).
  Smooth, with non-zero gradients through *both* inputs — the form that ties to
  Sigma-Pi.
* ``"min"``: ``and = min(a, b)``, ``or = max(a, b)`` (Gödel / Zadeh logic). Crisper,
  but the gradient flows through only one input at a time.

Every other gate is derived by De Morgan duality, so the whole set stays consistent
under either t-norm. XOR is defined as ``or - and`` (``a + b - 2ab`` for the product
t-norm, ``|a - b|`` for ``min``) — both correct on the Boolean corners.
"""

from __future__ import annotations

import torch
import torch.nn as nn

T_NORMS = ("product", "min")


def _check_t_norm(t_norm: str) -> None:
    if t_norm not in T_NORMS:
        raise ValueError(f"t_norm must be one of {T_NORMS}, got {t_norm!r}")


# ---------------------------------------------------------------------------
# Functional gates
# ---------------------------------------------------------------------------

def fuzzy_not(a: torch.Tensor) -> torch.Tensor:
    """Fuzzy negation ``1 - a`` (the standard strong/complement negation)."""
    return 1.0 - a


def fuzzy_and(a: torch.Tensor, b: torch.Tensor, t_norm: str = "product") -> torch.Tensor:
    """Fuzzy conjunction. ``product``: ``a*b`` (a Pi neuron); ``min``: ``min(a, b)``."""
    _check_t_norm(t_norm)
    return a * b if t_norm == "product" else torch.minimum(a, b)


def fuzzy_or(a: torch.Tensor, b: torch.Tensor, t_norm: str = "product") -> torch.Tensor:
    """Fuzzy disjunction (t-conorm dual of :func:`fuzzy_and`).

    ``product``: ``a + b - a*b`` (probabilistic sum); ``min``: ``max(a, b)``.
    """
    _check_t_norm(t_norm)
    return a + b - a * b if t_norm == "product" else torch.maximum(a, b)


def fuzzy_nand(a: torch.Tensor, b: torch.Tensor, t_norm: str = "product") -> torch.Tensor:
    """Fuzzy NAND ``not(and(a, b))``."""
    return fuzzy_not(fuzzy_and(a, b, t_norm))


def fuzzy_nor(a: torch.Tensor, b: torch.Tensor, t_norm: str = "product") -> torch.Tensor:
    """Fuzzy NOR ``not(or(a, b))``."""
    return fuzzy_not(fuzzy_or(a, b, t_norm))


def fuzzy_xor(a: torch.Tensor, b: torch.Tensor, t_norm: str = "product") -> torch.Tensor:
    """Fuzzy exclusive-or ``or(a, b) - and(a, b)``.

    For the product t-norm this is ``a + b - 2ab`` — a linear term plus a bilinear
    product, i.e. exactly a single Sigma-Pi / degree-2 neuron. For ``min`` it is
    ``|a - b|``. Both are correct on the Boolean corners.
    """
    return fuzzy_or(a, b, t_norm) - fuzzy_and(a, b, t_norm)


def fuzzy_xnor(a: torch.Tensor, b: torch.Tensor, t_norm: str = "product") -> torch.Tensor:
    """Fuzzy equivalence ``not(xor(a, b))`` — soft equality of two truth values."""
    return fuzzy_not(fuzzy_xor(a, b, t_norm))


# ---------------------------------------------------------------------------
# Module wrappers — drop logic into nn.Sequential / circuits as parameter-free "neurons"
# ---------------------------------------------------------------------------

class FuzzyNot(nn.Module):
    """Module form of :func:`fuzzy_not` (unary)."""

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return fuzzy_not(a)


class _BinaryGate(nn.Module):
    """Base for the two-input gates, carrying the chosen t-norm."""

    def __init__(self, t_norm: str = "product") -> None:
        super().__init__()
        _check_t_norm(t_norm)
        self.t_norm = t_norm

    def extra_repr(self) -> str:
        return f"t_norm={self.t_norm!r}"


class FuzzyAnd(_BinaryGate):
    """Module form of :func:`fuzzy_and`."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return fuzzy_and(a, b, self.t_norm)


class FuzzyOr(_BinaryGate):
    """Module form of :func:`fuzzy_or`."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return fuzzy_or(a, b, self.t_norm)


class FuzzyNand(_BinaryGate):
    """Module form of :func:`fuzzy_nand`."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return fuzzy_nand(a, b, self.t_norm)


class FuzzyNor(_BinaryGate):
    """Module form of :func:`fuzzy_nor`."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return fuzzy_nor(a, b, self.t_norm)


class FuzzyXor(_BinaryGate):
    """Module form of :func:`fuzzy_xor`."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return fuzzy_xor(a, b, self.t_norm)


class FuzzyXnor(_BinaryGate):
    """Module form of :func:`fuzzy_xnor`."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return fuzzy_xnor(a, b, self.t_norm)
