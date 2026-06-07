"""Signed logarithmic maps for the multiplicative (pi) branch.

The pi branch operates in log space so that a convolution / linear map over the
transformed inputs approximates the *log of a signed geometric product*. The map
must be **odd** (antisymmetric) so it preserves the sign of the input — this lets
the pi branch represent both amplification and suppression, unlike `log(softplus(x))`.

    signed_log(x)   = sign(x) * log(|x| + eps)
    signed_log1p(x) = sign(x) * log1p(|x|)          # = sign(x) * log(1 + |x|)

`signed_log1p` is numerically gentler near 0 (it is exactly 0 at x=0 and needs no
epsilon), and is offered as a drop-in alternative for experiments that want a
parameter-free, well-behaved variant.
"""

from __future__ import annotations

import torch

DEFAULT_EPS: float = 1e-8


def signed_log(x: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    """`sign(x) * log(|x| + eps)` — the formulation used in the paper experiments.

    Args:
        x: input tensor.
        eps: stabiliser added to ``|x|`` before the log (default ``1e-8``).

    Returns:
        Tensor of the same shape as ``x``.
    """
    return torch.sign(x) * torch.log(torch.abs(x) + eps)


def signed_log1p(x: torch.Tensor) -> torch.Tensor:
    """`sign(x) * log1p(|x|)` — epsilon-free, exactly zero at the origin.

    Args:
        x: input tensor.

    Returns:
        Tensor of the same shape as ``x``.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))
