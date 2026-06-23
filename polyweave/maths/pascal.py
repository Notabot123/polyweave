"""Pascal's triangle as a zero-parameter neural module (PyTorch port).

The TensorFlow original (pascal_revisited.ipynb) builds each row by padding and
convolving with a [[1,1],[0,0]] kernel — equivalent to adding each element to its
left neighbour from the row above, i.e., one step of Pascal's recurrence:
    C(n, k) = C(n-1, k-1) + C(n-1, k)

Stacking ``num_rows`` such steps and summing all intermediate states reproduces the
full triangle: intermediate x_k contains only row k (non-zero), so the sum gives
output[i, j] = C(i, j).

``BinomialExpansion`` composes the triangle with exponent lookup tables to compute
the coefficient vector of ``(Ax + By)^n`` exactly — no learnable parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PascalTriangle(nn.Module):
    """Fixed-weight conv network that generates Pascal's triangle.

    Args:
        num_rows: number of rows (and columns) in the output triangle.
            Row n holds ``[C(n,0), C(n,1), …, C(n,n), 0, …]``.

    Example::

        >>> pt = PascalTriangle(8)
        >>> pt()[5].int().tolist()
        [1, 5, 10, 10, 5, 1, 0, 0]
    """

    def __init__(self, num_rows: int = 16) -> None:
        super().__init__()
        self.num_rows = num_rows

        # Kernel: sum the two elements directly above-left and above.
        # Shape (out_ch, in_ch, kH, kW) = (1, 1, 2, 2).
        # [[1, 1], [0, 0]] — top row sums, bottom row zeros.
        kernel = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
        self.register_buffer("kernel", kernel)

    @torch.no_grad()
    def forward(self) -> torch.Tensor:
        """Return Pascal's triangle as a ``(num_rows, num_rows)`` tensor."""
        n = self.num_rows
        # Seed: a single 1 at (0, 0) — the apex of the triangle.
        x = torch.zeros(1, 1, n, n, device=self.kernel.device)
        x[0, 0, 0, 0] = 1.0

        # Collect the sum of all intermediate states.
        # Intermediate x_k is non-zero only at row k, so summing gives the full triangle.
        result = x.clone()
        for _ in range(n - 1):
            padded = F.pad(x, (1, 0, 1, 0))            # pad left and top by 1
            x = F.conv2d(padded, self.kernel)[:, :, :n, :n]
            result = result + x

        return result[0, 0]  # (num_rows, num_rows)


# ---------------------------------------------------------------------------
# Exponent lookup tables (mirrors pascal_binomial_expansion.ipynb)
# ---------------------------------------------------------------------------

def _make_exponent_tables(num_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute A-exponent and B-exponent arrays for binomial expansion.

    For row n, position k:  A_exp[n, k] = n - k,  B_exp[n, k] = k.
    """
    idx = torch.arange(num_rows, dtype=torch.float32)
    rows = torch.arange(num_rows, dtype=torch.float32)
    A_exp = (rows.unsqueeze(1) - idx.unsqueeze(0)).clamp(min=0)
    B_exp = idx.unsqueeze(0).expand(num_rows, -1)
    return A_exp, B_exp


class BinomialExpansion(nn.Module):
    """Exact binomial expansion ``(A·x + B·y)^n`` via the Pascal triangle.

    Returns the coefficient vector ``[C(n,0)·Aⁿ, C(n,1)·Aⁿ⁻¹·B, …, C(n,n)·Bⁿ]``
    of length ``num_rows``, with trailing zeros beyond position n.

    Zero learnable parameters.  Matches the TF notebook to floating-point precision
    on the supported integer domain.

    Args:
        num_rows: maximum supported exponent (exclusive upper bound). Defaults to
            16 to match the saved TF model.

    Example::

        >>> bx = BinomialExpansion(num_rows=8)
        >>> bx(A=2.0, B=3.0, n=2)[:3].tolist()
        [4.0, 12.0, 9.0]
        # (2x+3y)^2 = 4x^2 + 12xy + 9y^2
    """

    def __init__(self, num_rows: int = 16) -> None:
        super().__init__()
        self.pascal = PascalTriangle(num_rows)
        A_exp, B_exp = _make_exponent_tables(num_rows)
        self.register_buffer("A_exp", A_exp)
        self.register_buffer("B_exp", B_exp)

    @torch.no_grad()
    def forward(self, A: float, B: float, n: int) -> torch.Tensor:
        """Compute coefficient vector for ``(A·x + B·y)^n``.

        Args:
            A: coefficient of x.
            B: coefficient of y.
            n: exponent (must be < ``num_rows``).

        Returns:
            1-D tensor of length ``num_rows``.
        """
        triangle = self.pascal()          # (num_rows, num_rows)
        coeffs = triangle[n]              # C(n, 0) … C(n, n), 0 …
        A_pows = float(A) ** self.A_exp[n]
        B_pows = float(B) ** self.B_exp[n]
        return coeffs * A_pows * B_pows
