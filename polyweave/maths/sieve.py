"""Differentiable Sieve of Eratosthenes — primality as a zero-parameter neural module.

The classical sieve marks composites by sweeping multiples of each known prime p ≤ √N.
Here, each prime p contributes a frozen "comb" buffer — a tensor of length N+1 with 1s
at positions 0, p, 2p, … (position p itself is cleared, since p is prime).  The combs
are combined via the probabilistic OR formula (1 − ∏(1 − cᵢ)), yielding a composite
score in [0, 1].  A final exp(−α · composite_score) maps 0 → 1 (prime) and 1 → 0
(composite).

Zero learnable parameters.  The decay parameter α is tunable (and can be made
learnable) to trade off sharpness vs. smoothness of the prime/composite boundary.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _sieve_up_to(n: int) -> list[int]:
    """Return all primes ≤ n via the classical sieve."""
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(math.isqrt(n)) + 1):
        if is_prime[p]:
            for m in range(p * p, n + 1, p):
                is_prime[m] = False
    return [i for i, v in enumerate(is_prime) if v]


class DifferentiableSieve(nn.Module):
    """Soft primality detector via frozen comb buffers.

    For each prime p ≤ √N, a comb tensor marks positions that are multiples of p
    (excluding p itself, which is prime).  These composite indicators are combined
    via probabilistic OR, then mapped through an exponential decay to produce
    primality scores in (0, 1].

    Args:
        N: upper bound (inclusive). Scores are returned for 0 … N.
        max_p: largest prime used as a sieve factor. Defaults to ⌊√N⌋, which is
            sufficient for exact primality detection up to N.
        decay: exponential decay α. Higher values sharpen the prime/composite
            separation. Default 5.0 gives near-binary output.

    Example::

        >>> sieve = DifferentiableSieve(30)
        >>> scores = sieve()
        >>> [i for i in range(2, 31) if scores[i] > 0.5]
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    """

    def __init__(
        self,
        N: int,
        max_p: int | None = None,
        decay: float = 5.0,
    ) -> None:
        super().__init__()
        self.N = N
        self.decay = decay

        if max_p is None:
            max_p = int(math.isqrt(N))

        base_primes = _sieve_up_to(max_p)

        # Pre-compute the composite mask from all prime combs.
        # composite[n] = P(n is divisible by at least one base prime, and n != that prime)
        composite = torch.zeros(N + 1)
        for p in base_primes:
            comb = torch.zeros(N + 1)
            comb[::p] = 1.0   # mark all multiples of p
            comb[p] = 0.0     # p itself is prime, not composite
            composite = 1.0 - (1.0 - composite) * (1.0 - comb)

        self.register_buffer("composite", composite)
        self.register_buffer(
            "base_primes", torch.tensor(base_primes, dtype=torch.long)
        )

    def forward(self) -> torch.Tensor:
        """Compute soft primality scores for 0 … N.

        Returns:
            1-D tensor of shape ``(N+1,)`` with values in ``[0, 1]``.
            Values near 1 indicate primes; values near 0 indicate composites.
            Indices 0 and 1 are always 0.
        """
        scores = torch.exp(-self.decay * self.composite)
        scores = scores.clone()
        scores[0] = 0.0
        scores[1] = 0.0
        return scores
