"""Plotting with arXiv-friendly defaults.

Conventions (all configurable):
* Figures are written to a dedicated ``plots/`` directory to keep the project
  root clean.
* Vector **PDF** is the primary output (crisp at any scale in the paper); a PNG
  is also written for quick preview.
* Larger default fonts than matplotlib's, sized for a single-column figure.
* A colourblind-safe palette (Okabe-Ito) is used as the default colour cycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # file-saving only; avoids interactive-backend (Tk) issues
import matplotlib.pyplot as plt

# Okabe-Ito colourblind-safe qualitative palette.
OKABE_ITO: List[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

DEFAULT_PLOTS_DIR = Path("plots")


def configure_plots(dark: bool = False, font_scale: float = 1.3) -> None:
    """Apply global matplotlib styling.

    Args:
        dark: use a dark background theme (for slides/blog) instead of light.
        font_scale: multiplier applied to a 10pt base; default ``1.3`` (~13pt body)
            reads well in a single-column arXiv figure.
    """
    plt.style.use("dark_background" if dark else "default")
    base = 10.0 * font_scale
    rc: Dict[str, object] = {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": base,
        "axes.titlesize": base * 1.15,
        "axes.labelsize": base,
        "xtick.labelsize": base * 0.9,
        "ytick.labelsize": base * 0.9,
        "legend.fontsize": base * 0.9,
        "lines.linewidth": 2.0,
        "axes.grid": True,
        "grid.alpha": 0.35 if not dark else 0.25,
        "axes.prop_cycle": matplotlib.cycler(color=OKABE_ITO),
    }
    if dark:
        rc.update(
            {
                "axes.facecolor": "#1c1c1c",
                "figure.facecolor": "#1c1c1c",
                "grid.color": "#444444",
            }
        )
    plt.rcParams.update(rc)


def save_figure(
    fig: "matplotlib.figure.Figure",
    name: str,
    plots_dir: Path = DEFAULT_PLOTS_DIR,
    formats: Sequence[str] = ("pdf", "png"),
    close: bool = True,
) -> List[Path]:
    """Save ``fig`` as ``name.<ext>`` for each format into ``plots_dir``.

    Returns the list of written paths. The directory is created if missing.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for ext in formats:
        path = plots_dir / f"{name}.{ext}"
        fig.savefig(path)
        written.append(path)
        print(f"saved {path}")
    if close:
        plt.close(fig)
    return written


def plot_lines(
    data: Dict[str, Sequence[float]],
    title: str,
    ylabel: str,
    name: str,
    xlabel: str = "evaluation point",
    plots_dir: Path = DEFAULT_PLOTS_DIR,
    figsize: Tuple[float, float] = (7.0, 4.5),
) -> List[Path]:
    """Line plot of one or more series, saved as PDF + PNG."""
    fig, ax = plt.subplots(figsize=figsize)
    for label, values in data.items():
        ax.plot(list(values), label=label, marker="o", markersize=3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, name, plots_dir=plots_dir)


def plot_occlusion_heatmaps(
    maps: Dict[str, "object"],
    name: str,
    title: str = "Occlusion sensitivity",
    cbar_label: str = "response drop",
    plots_dir: Path = DEFAULT_PLOTS_DIR,
    figsize: Tuple[float, float] | None = None,
) -> List[Path]:
    """Side-by-side occlusion sensitivity heatmaps, one panel per named map.

    Each value in ``maps`` is a 2-D array-like (e.g. an ``[H, W]`` numpy array or
    torch tensor) of response drops. Panels share a common colour scale so the
    additive vs multiplicative fingerprints are directly comparable.
    """
    import numpy as np

    items = list(maps.items())
    n = len(items)
    arrs = [np.asarray(v, dtype=float) for _, v in items]
    vmax = max((a.max() for a in arrs), default=1.0)
    vmin = min((a.min() for a in arrs), default=0.0)
    figsize = figsize or (3.6 * n + 1.0, 3.8)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    im = None
    for ax, (label, _), arr in zip(axes[0], items, arrs):
        im = ax.imshow(arr, cmap="magma", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle(title)
    if im is not None:
        cbar = fig.colorbar(im, ax=list(axes[0]), fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)
    return save_figure(fig, name, plots_dir=plots_dir)


def _to_hwc(image) -> "object":
    """Coerce an image to an ``[H, W, 3]`` float array in ``[0, 1]`` for display.

    Accepts a torch tensor or numpy array shaped ``[3, H, W]`` (CHW) or
    ``[H, W, 3]`` (HWC). Values are clamped to ``[0, 1]`` (so a de-normalised
    CIFAR image with slight out-of-range values renders cleanly).
    """
    import numpy as np

    arr = np.asarray(image, dtype=float)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = arr.transpose(1, 2, 0)  # CHW -> HWC
    return np.clip(arr, 0.0, 1.0)


def plot_occlusion_overlay(
    image,
    maps: Dict[str, "object"],
    name: str,
    title: str = "Occlusion sensitivity over input",
    cbar_label: str = "relative response drop",
    alpha: float = 0.55,
    plots_dir: Path = DEFAULT_PLOTS_DIR,
    figsize: Tuple[float, float] | None = None,
) -> List[Path]:
    """Overlay occlusion heatmaps on a real input image, one panel per map.

    The first panel shows the bare ``image``; each subsequent panel re-draws the
    image with a named occlusion map alpha-blended on top (the map is stretched
    to the image resolution by bilinear interpolation, so a coarse strided map
    still aligns spatially). All overlays share a common colour scale so two
    methods — e.g. an additive-teacher vs a $\\Sigma\\Pi$-teacher generated
    ``conv1`` — are directly comparable.

    Args:
        image: the input image, ``[3, H, W]`` or ``[H, W, 3]`` (any range; it is
            clamped to ``[0, 1]`` for display — pass a *de-normalised* image).
        maps: label -> 2-D occlusion map (``[h, w]`` array/tensor); typically
            ``relative=True`` maps so the scale is a response *fraction*.
        alpha: overlay opacity for the heatmap.
    """
    import numpy as np

    def _np(v):
        if hasattr(v, "detach"):  # torch tensor (possibly on GPU)
            v = v.detach().cpu()
        return np.asarray(v, dtype=float)

    img = _to_hwc(image)
    H, W = img.shape[:2]
    items = list(maps.items())
    arrs = [_np(v) for _, v in items]
    vmax = max((a.max() for a in arrs), default=1.0)
    vmin = min((a.min() for a in arrs), default=0.0)

    n = len(items) + 1  # +1 for the bare input
    figsize = figsize or (3.4 * n + 1.0, 3.8)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    ax0 = axes[0][0]
    # ``nearest`` keeps the low-res CIFAR pixels crisp when the panel is upscaled
    # for the page (bilinear would blur a 32x32 image into mush).
    ax0.imshow(img, aspect="equal", interpolation="nearest")
    ax0.set_title("input")
    ax0.set_xticks([]); ax0.set_yticks([]); ax0.grid(False)

    im = None
    extent = (0, W, H, 0)  # map heatmap onto the image's pixel grid
    for ax, (label, _), arr in zip(axes[0][1:], items, arrs):
        ax.imshow(img, aspect="equal", interpolation="nearest")
        im = ax.imshow(arr, cmap="magma", vmin=vmin, vmax=vmax, alpha=alpha,
                       extent=extent, interpolation="bilinear", aspect="equal")
        ax.set_title(label)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle(title)
    if im is not None:
        cbar = fig.colorbar(im, ax=list(axes[0]), fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)
    return save_figure(fig, name, plots_dir=plots_dir)


def plot_conjunction_index(
    values: Dict[str, float],
    name: str,
    errors: Dict[str, float] | None = None,
    title: str = "Conjunction (AND-signature) index",
    ylabel: str = "conjunction index",
    plots_dir: Path = DEFAULT_PLOTS_DIR,
    figsize: Tuple[float, float] = (6.0, 4.2),
) -> List[Path]:
    """Bar chart of per-feature/per-branch conjunction indices in ``[0, 1]``.

    Optional ``errors`` (same keys) draws symmetric error bars. A dashed guide at
    0.5 separates the additive-leaning from multiplicative-leaning regime.
    """
    labels = list(values.keys())
    x = list(range(len(labels)))
    heights = [values[k] for k in labels]
    yerr = [errors[k] for k in labels] if errors else None
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, heights, 0.6, yerr=yerr, capsize=4,
           color=[OKABE_ITO[i % len(OKABE_ITO)] for i in range(len(labels))])
    ax.axhline(0.5, ls="--", lw=1.2, color="grey", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # Direction cues at the two ends of the index range.
    ax.text(0.012, 0.02, "additive", transform=ax.transAxes, fontsize="small",
            color="grey", ha="left", va="bottom")
    ax.text(0.012, 0.98, "multiplicative", transform=ax.transAxes, fontsize="small",
            color="grey", ha="left", va="top")
    fig.tight_layout()
    return save_figure(fig, name, plots_dir=plots_dir)


def plot_ensemble_bars(
    single_means: Dict[str, float],
    ensemble: Dict[str, float],
    name: str,
    single_stds: Dict[str, float] | None = None,
    title: str = "Single member vs ensemble accuracy",
    ylabel: str = "test accuracy",
    plots_dir: Path = DEFAULT_PLOTS_DIR,
    figsize: Tuple[float, float] = (6.5, 4.5),
) -> List[Path]:
    """Grouped bars: mean single-member vs full-ensemble accuracy, per method.

    ``single_means``/``ensemble``/``single_stds`` share keys (the methods, e.g.
    ``additive teacher`` / ``$\\Sigma\\Pi$ teacher``). Single-member bars carry
    optional ±std error bars across members; the gap to the ensemble bar is the
    diversity payoff.
    """
    methods = list(single_means.keys())
    x = list(range(len(methods)))
    w = 0.38
    yerr = [single_stds[m] for m in methods] if single_stds else None
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar([i - w / 2 for i in x], [single_means[m] for m in methods], w,
           yerr=yerr, capsize=4, label="mean single member", color=OKABE_ITO[0])
    ax.bar([i + w / 2 for i in x], [ensemble[m] for m in methods], w,
           label="ensemble", color=OKABE_ITO[1])
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, name, plots_dir=plots_dir)


def plot_diversity_hist(
    disagreements: Dict[str, Sequence[float]],
    name: str,
    title: str = "Ensemble member diversity",
    xlabel: str = "pairwise prediction disagreement",
    ylabel: str = "pair count",
    bins: int = 12,
    plots_dir: Path = DEFAULT_PLOTS_DIR,
    figsize: Tuple[float, float] = (6.5, 4.5),
) -> List[Path]:
    """Overlaid histograms of per-pair disagreement rates, one series per method.

    Each value in ``disagreements`` is a flat list of pairwise disagreement rates
    (one per member pair). A right-shifted distribution means a more diverse
    population — the property that makes an ensemble worth more than its members.

    All series share a single set of bin edges (spanning the pooled data range) so
    the overlaid histograms are directly comparable bar-for-bar.
    """
    # Shared bin edges across every series, derived from the pooled value range,
    # so the two distributions line up bin-for-bin instead of each picking its own.
    pooled = [v for vals in disagreements.values() for v in vals]
    if pooled:
        lo, hi = min(pooled), max(pooled)
        if hi <= lo:  # all values identical -> pad to a tiny non-zero width
            hi = lo + 1e-6
        edges = [lo + (hi - lo) * k / bins for k in range(bins + 1)]
    else:
        edges = bins  # nothing to plot; fall back to the integer bin count

    fig, ax = plt.subplots(figsize=figsize)
    for i, (label, vals) in enumerate(disagreements.items()):
        color = OKABE_ITO[i % len(OKABE_ITO)]
        ax.hist(list(vals), bins=edges, alpha=0.55, label=label, color=color,
                edgecolor="black", linewidth=0.7)
        mean = sum(vals) / len(vals) if len(vals) else 0.0
        ax.axvline(mean, color=color, ls="--", lw=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, name, plots_dir=plots_dir)


def plot_zeroshot_bar(
    seen_means: Dict[str, float],
    unseen_means: Dict[str, float],
    name: str,
    title: str = "Zero-shot generation: seen vs unseen architectures",
    ylabel: str = "zero-shot accuracy",
    plots_dir: Path = DEFAULT_PLOTS_DIR,
    figsize: Tuple[float, float] = (6.5, 4.5),
) -> List[Path]:
    """Grouped bar chart of seen vs unseen accuracy per method, saved as PDF + PNG."""
    methods = list(seen_means.keys())
    x = list(range(len(methods)))
    w = 0.38
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar([i - w / 2 for i in x], [seen_means[m] for m in methods], w,
           label="seen arch", color=OKABE_ITO[0])
    ax.bar([i + w / 2 for i in x], [unseen_means[m] for m in methods], w,
           label="unseen arch", color=OKABE_ITO[1])
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, name, plots_dir=plots_dir)
