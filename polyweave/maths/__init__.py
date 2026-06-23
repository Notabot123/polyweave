"""Exact mathematical structures as zero-parameter differentiable neural modules.

Each module encodes a classical algorithm as fixed-weight network layers.  They
compose with learned components and are differentiable with respect to their
continuous hyper-parameters (sharpness, decay, etc.).

    DifferentiableSieve  — soft primality scores via frozen comb buffers
    PascalTriangle       — Pascal's triangle via fixed [[1,1],[0,0]] conv recurrence
    BinomialExpansion    — exact (Ax + By)^n coefficient vectors
    BernoulliTriangle    — partial row sums of Pascal; encodes simplex / cake numbers
"""

from __future__ import annotations

from .bernoulli import BernoulliTriangle
from .pascal import BinomialExpansion, PascalTriangle
from .sieve import DifferentiableSieve

__all__ = [
    "DifferentiableSieve",
    "PascalTriangle",
    "BinomialExpansion",
    "BernoulliTriangle",
]
