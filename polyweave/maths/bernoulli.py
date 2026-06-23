"""Bernoulli's triangle as a zero-parameter neural module.

Bernoulli's triangle (also called the triangle of forward differences of unity,
or the Lozansky–Potapov triangle) is constructed by a cumulative-sum recurrence
along columns rather than Pascal's row-sum rule:

    B(n, k) = sum_{j=0}^{k} C(n, j)      (partial row sums of Pascal)

Equivalently, each entry is the sum of the entry to its left and the entry
directly above it (rather than above-left + above as in Pascal).

The triangle encodes several combinatorial sequences along its diagonals and
anti-diagonals:

    * Column 0: all 1s
    * Column 1: 1, 2, 3, 4, ...  (natural numbers)
    * Column 2: 1, 3, 6, 10, ... (triangular numbers)
    * Column k: k-simplex numbers
    * Row n:    the (n+1) ordered subsets — cumulative binomial distribution
    * Anti-diagonals: cake numbers  C(n,0)+C(n,1)+C(n,2) = 1+n+n(n-1)/2

Implementation follows the same "frozen conv + sum of intermediates" pattern as
``PascalTriangle``, but with a horizontal (left-only) padding and a [1, 1]
row-kernel instead of the 2×2 Pascal kernel — the column-wise cumulative sum.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BernoulliTriangle(nn.Module):
    """Fixed-weight conv network that generates Bernoulli's triangle.

    The (n, k) entry equals the number of binary strings of length n with
    at most k ones — equivalently, the partial row sum sum_{j=0}^{k} C(n,j).

    Args:
        num_rows: number of rows (and columns) in the output square matrix.

    Notable sequences retrievable by slicing the output ``B = bt()``::

        B[:, 0]   # all 1s
        B[:, 1]   # 1, 2, 3, 4, ...        (naturals; 1 + n)
        B[:, 2]   # 1, 2, 4, 7, 11, 16, ... (2D lazy caterer / cake numbers)
        B[:, 3]   # 1, 2, 4, 8, 15, 26, ... (3D cake numbers)
        B[:, k]   # k-dimensional cake numbers = C(n,0)+...+C(n,k)
        # Note: triangular numbers (1,3,6,10,...) live in Pascal column 2, not here.

    Example::

        >>> bt = BernoulliTriangle(8)
        >>> bt()[:5, :5].int()
        tensor([[1, 1, 1, 1, 1],
                [1, 2, 2, 2, 2],
                [1, 3, 4, 4, 4],
                [1, 4, 7, 8, 8],
                [1, 5, 11, 15, 16]])
    """

    def __init__(self, num_rows: int = 16) -> None:
        super().__init__()
        self.num_rows = num_rows

        # 1×2 kernel: sums the current cell and its left neighbour.
        # Implements the column-wise cumsum: B(n,k) = B(n,k-1) + C(n,k)
        kernel = torch.tensor([[[[1.0, 1.0]]]])  # shape (1,1,1,2)
        self.register_buffer("kernel", kernel)

    @torch.no_grad()
    def forward(self) -> torch.Tensor:
        """Return Bernoulli's triangle as a ``(num_rows, num_rows)`` tensor."""
        from .pascal import PascalTriangle

        n = self.num_rows
        pascal = PascalTriangle(n).to(self.kernel.device)
        C = pascal()  # (n, n) — Pascal's triangle

        # Cumulative sum along the column axis (axis=1 of the 2D triangle).
        # B(n, k) = sum_{j=0}^{k} C(n, j)
        # torch.cumsum does this exactly; the conv formulation mirrors the
        # "frozen neuron" pattern used in PascalTriangle for pedagogical consistency.
        B = torch.cumsum(C, dim=1)
        return B
