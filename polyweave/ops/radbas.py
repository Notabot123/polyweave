"""Radial basis activation — a differentiable fuzzy-equality / locality primitive.

    radbas(x, epsilon) = exp(-(epsilon * x) ** 2)

A Gaussian bump centred at ``x = 0`` (output in ``(0, 1]``, peaking at ``1`` when
``x == 0``). Because it fires only when its input is *near zero*, it acts as a soft
equality test: feed it a difference ``a - b`` and it answers "are a and b equal?" —
sharply (near-binary) for large ``epsilon``, smoothly for small.

That locality is what makes it useful beyond a plain activation:

* **Fuzzy equality / unification.** ``radbas(query - key)`` peaks where they match —
  the basis of radial-basis indexing and content addressing.
* **A second route to XOR.** A single linear neuron cannot separate XOR, but a radial
  bump can: ``radbas(a - b)`` is XNOR (``1`` when the inputs agree), so
  ``1 - radbas(a - b)`` is a fuzzy XOR — the classic RBF answer to the
  non-linearly-separable problem, complementing the Sigma-Pi / poly route
  ``a + b - 2ab`` (see :mod:`polyweave.logic`).

``epsilon`` sets the width: ``epsilon = 1`` is a gentle bell; ``epsilon >= 10`` is
near-binary (``~1`` for ``|x| < 0.1``, ``~0`` beyond), the setting used for
integer-index lookups. The map is smooth and differentiable everywhere — including in
the centre — so an "index" or "centre" fed through it can itself be learned.
"""

from __future__ import annotations

import torch

DEFAULT_EPSILON: float = 1.0


def radbas(x: torch.Tensor, epsilon: float = DEFAULT_EPSILON) -> torch.Tensor:
    """`exp(-(epsilon * x) ** 2)` — a Gaussian bump peaking at ``x == 0``.

    Args:
        x: input tensor (typically a difference / residual).
        epsilon: sharpness. ``1.0`` is a gentle bell; ``>= 10`` is near-binary
            (the value used for integer-index lookups).

    Returns:
        Tensor of the same shape as ``x``, with values in ``(0, 1]``.
    """
    return torch.exp(-((epsilon * x) ** 2))
