"""Occlusion-sensitivity demonstration — the conjunctive AND-signature.

A fast, CPU-only, deterministic demonstration that occlusion sensitivity
distinguishes additive from multiplicative (Sigma-Pi / bilinear) features, and
that the separation is *graded* — mirroring the paper's central pi-scale
recruitment story (FC < conv1 < Q/K).

Produces two paper-ready figures in ``plots/``:

* ``polyweave_occlusion_conjunction_index.{pdf,png}`` — conjunction index vs the
  multiplicative mixing fraction of a feature ``r = (1-a)(p+q) + a*(p*q)``. The
  index rises monotonically 0 -> 1 as the feature becomes more multiplicative.
* ``polyweave_occlusion_heatmaps.{pdf,png}`` — side-by-side spatial occlusion
  heatmaps for an additive vs a multiplicative two-patch detector on a synthetic
  image, sharing a colour scale so the multiplicative panel is visibly "hotter":
  occluding *either* patch collapses the response (each patch alone is critical).

Run:  python -m polyweave.experiments.occlusion_demo
"""

from __future__ import annotations

import torch

from ..interpretability import conjunction_index, occlusion_sensitivity_2d
from ..utils import set_seed
from ..viz import (
    configure_plots,
    plot_conjunction_index,
    plot_occlusion_heatmaps,
)

# Two disjoint input groups for the 1-D demonstration.
GROUP_A = [0, 1]
GROUP_B = [2, 3]


def _mixed_response(alpha: float):
    """Feature interpolating additive -> multiplicative by fraction ``alpha``.

    ``r = (1-alpha)*(sum_A + sum_B) + alpha*(sum_A * sum_B)``. At ``alpha=0`` the
    feature is purely additive; at ``alpha=1`` it is the bilinear product.
    """
    def resp(x: torch.Tensor) -> torch.Tensor:
        a = x[:, GROUP_A].sum(1)
        b = x[:, GROUP_B].sum(1)
        return (1.0 - alpha) * (a + b) + alpha * (a * b)
    return resp


def conjunction_vs_mixing(n: int = 512) -> dict:
    """Conjunction index (mean +/- std over items) for a sweep of mixing fractions."""
    set_seed(0)
    x = torch.rand(n, 4) + 0.5  # strictly positive so products stay informative
    values, errors = {}, {}
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        idx = conjunction_index(_mixed_response(alpha), x, GROUP_A, GROUP_B)
        label = f"{int(alpha * 100)}% mult"
        values[label] = idx.mean().item()
        errors[label] = idx.std(unbiased=True).item()
    return {"values": values, "errors": errors}


def _two_patch_image(n: int = 1) -> torch.Tensor:
    """A 12x12 single-channel image with two informative bright patches."""
    img = torch.zeros(n, 1, 12, 12)
    img[:, :, 1:4, 1:4] = 1.0     # patch A (top-left)
    img[:, :, 8:11, 8:11] = 1.0   # patch B (bottom-right)
    img += 0.05 * torch.randn(n, 1, 12, 12)
    return img


def occlusion_heatmaps() -> dict:
    """Spatial occlusion maps for an additive vs a multiplicative two-patch detector."""
    set_seed(0)
    img = _two_patch_image()

    def patch_a(t):
        return t[:, :, 1:4, 1:4].mean(dim=(1, 2, 3)).clamp(min=0)

    def patch_b(t):
        return t[:, :, 8:11, 8:11].mean(dim=(1, 2, 3)).clamp(min=0)

    def additive(t):       # responds if EITHER patch present
        return patch_a(t) + patch_b(t)

    def multiplicative(t):  # responds only if BOTH patches present (AND gate)
        return patch_a(t) * patch_b(t)

    # Relative drops expose the AND-signature: each patch is ~100% critical for
    # the multiplicative detector but only ~50% critical for the additive one.
    add_map = occlusion_sensitivity_2d(additive, img, window=3, stride=1, relative=True)[0]
    mul_map = occlusion_sensitivity_2d(multiplicative, img, window=3, stride=1, relative=True)[0]
    return {"Additive (p + q)": add_map, "Multiplicative (p . q)": mul_map}


def main() -> None:
    configure_plots(dark=False)

    sweep = conjunction_vs_mixing()
    plot_conjunction_index(
        sweep["values"], name="polyweave_occlusion_conjunction_index",
        errors=sweep["errors"],
        title="Occlusion conjunction index rises with multiplicative content",
    )
    print("Conjunction index vs mixing fraction:")
    for k, v in sweep["values"].items():
        print(f"  {k:>9}: {v:.3f} +/- {sweep['errors'][k]:.3f}")

    maps = occlusion_heatmaps()
    plot_occlusion_heatmaps(
        maps, name="polyweave_occlusion_heatmaps",
        title="Occlusion sensitivity: additive vs multiplicative two-patch detector",
        cbar_label="fraction of response lost",
    )
    for label, m in maps.items():
        print(f"  {label:>22}: max drop {float(m.max()):.3f}, "
              f"mean drop {float(m.mean()):.3f}")


if __name__ == "__main__":
    main()
