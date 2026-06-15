"""Differentiable fuzzy logic — Boolean gates as soft, trainable neurons.

Truth values are tensors in ``[0, 1]``. The gates reduce to exact Boolean truth
tables on the corners and interpolate smoothly between, so they are differentiable
and compose into trainable logic circuits.

The default **product t-norm** makes a fuzzy AND a literal product neuron — the
multiplicative (Pi) computation at the heart of this library — which is why a single
Sigma-Pi / :class:`~polyweave.layers.PolyLinear` neuron solves XOR
(``a + b - 2ab``). See :func:`polyweave.ops.radbas` for the radial-basis route to the
same non-linearly-separable problem.

Available as functions (``fuzzy_and``, ``fuzzy_or``, …) and as parameter-free
``nn.Module`` gates (``FuzzyAnd``, ``FuzzyOr``, …) for use in ``nn.Sequential``.
"""

from __future__ import annotations

from .gates import (
    FuzzyAnd,
    FuzzyNand,
    FuzzyNor,
    FuzzyNot,
    FuzzyOr,
    FuzzyXnor,
    FuzzyXor,
    fuzzy_and,
    fuzzy_nand,
    fuzzy_nor,
    fuzzy_not,
    fuzzy_or,
    fuzzy_xnor,
    fuzzy_xor,
)
from .literals import SoftRuleLayer, SoftSignedLiteral

__all__ = [
    # functional gates
    "fuzzy_not",
    "fuzzy_and",
    "fuzzy_or",
    "fuzzy_nand",
    "fuzzy_nor",
    "fuzzy_xor",
    "fuzzy_xnor",
    # module gates
    "FuzzyNot",
    "FuzzyAnd",
    "FuzzyOr",
    "FuzzyNand",
    "FuzzyNor",
    "FuzzyXor",
    "FuzzyXnor",
    # rule induction
    "SoftSignedLiteral",
    "SoftRuleLayer",
]
