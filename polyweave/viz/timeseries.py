"""Time-series visualisation helpers.

Two families of plots:

**Occlusion heatmap overlay on stacked lineplots**
    The headline diagnostic for 1-D conv interpretability.  Each channel of a
    multivariate time series is drawn on its own axis (stacked vertically with a
    shared time axis), and a semi-transparent heatmap of occlusion-sensitivity
    values is overlaid behind the line.  The heatmap is linearly upsampled from the
    coarser window-strided resolution back to the original time resolution, so the
    spatial alignment is exact regardless of the occlusion stride.

    Inspired by standard MATLAB ``stackedplot`` + heatmap overlays used in BCI /
    EEG research.  The function is general: it works for any 1-D conv model over
    any multivariate signal (EEG, vibration, suspension, ECG, …).

**RUL / health-index prediction panels**
    Visualise predicted vs ground-truth remaining useful life and health index for
    a single degradation trajectory, optionally with a shaded degradation onset
    window.  Designed for the SimWeave suspension and predictive-maintenance
    datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from .plots import OKABE_ITO, DEFAULT_PLOTS_DIR, save_figure


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy(x) -> np.ndarray:
    """Coerce tensor or array-like to a float64 NumPy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x, dtype=np.float64)


def _upsample_1d(values: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly interpolate a 1-D array from its current length to *target_len*.

    This is used to bring the coarse, stride-averaged occlusion map back to the
    original signal resolution so the heatmap overlay aligns exactly with the
    plotted line.
    """
    src_len = len(values)
    if src_len == target_len:
        return values
    src_x = np.linspace(0.0, 1.0, src_len)
    dst_x = np.linspace(0.0, 1.0, target_len)
    return np.interp(dst_x, src_x, values)


# ---------------------------------------------------------------------------
# Occlusion heatmap overlaid on stacked lineplots
# ---------------------------------------------------------------------------

def plot_occlusion_stacked(
    signal: "np.ndarray | torch.Tensor",
    sensitivity: "np.ndarray | torch.Tensor",
    *,
    channel_names: Optional[Sequence[str]] = None,
    time: Optional["np.ndarray | torch.Tensor"] = None,
    xlabel: str = "time",
    cmap: str = "YlOrRd",
    alpha: float = 0.55,
    line_color: str = OKABE_ITO[0],
    cbar_label: str = "occlusion sensitivity",
    title: str = "Occlusion sensitivity — stacked channels",
    name: str = "occlusion_stacked",
    plots_dir: Path = DEFAULT_PLOTS_DIR,
    figsize: Optional[Tuple[float, float]] = None,
    panel_height: float = 1.6,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> List[Path]:
    """Stacked lineplots with per-channel occlusion sensitivity heatmap overlay.

    Each channel of *signal* is plotted on its own axis (rows, top-to-bottom),
    sharing the same time axis.  Behind each line, the corresponding row of
    *sensitivity* is rendered as a semi-transparent ``pcolormesh`` heatmap —
    hot colours mark time regions the model most relies on.

    The sensitivity map is linearly upsampled from its (typically coarser)
    window-strided resolution to the full signal length before plotting, so the
    overlay aligns exactly with the signal line regardless of occlusion stride.

    Args:
        signal: multivariate signal, shape ``[C, T]`` (channels × time-steps).
            A 1-D input ``[T]`` is treated as a single-channel signal.
        sensitivity: occlusion-sensitivity map, shape ``[C, P]`` (one row per
            channel, ``P`` window positions).  A 1-D input ``[P]`` is broadcast
            to all channels.  Values should be non-negative drops (as returned
            by :func:`~polyweave.interpretability.occlusion.occlusion_sensitivity_1d`).
        channel_names: optional list of ``C`` channel labels (e.g. EEG electrode
            names or suspension DOF names).  Defaults to ``ch_0``, ``ch_1``, …
        time: optional 1-D time axis of length ``T`` for the x-axis.  Defaults to
            ``0, 1, …, T-1``.
        xlabel: label for the shared x-axis (default ``"time"``).
        cmap: matplotlib colourmap for the heatmap (default ``"YlOrRd"``).
        alpha: opacity of the heatmap overlay (default ``0.55``; lower = more
            transparent, easier to read the signal line).
        line_color: colour of the signal line (default Okabe-Ito blue).
        cbar_label: colour-bar axis label.
        title: figure suptitle.
        name: filename stem passed to :func:`~polyweave.viz.plots.save_figure`.
        plots_dir: directory for saved figures.
        figsize: explicit ``(width, height)`` in inches; computed automatically
            from *panel_height* and the channel count if ``None``.
        panel_height: height per channel panel in inches when *figsize* is
            auto-computed (default ``1.6``).
        vmin: lower bound for the heatmap colour scale; defaults to the global
            minimum of *sensitivity*.
        vmax: upper bound for the heatmap colour scale; defaults to the global
            maximum of *sensitivity*.

    Returns:
        List of written file paths (PDF + PNG).

    Example::

        from polyweave.viz.timeseries import plot_occlusion_stacked
        from polyweave.interpretability.occlusion import occlusion_sensitivity_1d
        import torch

        # model: nn.Module with 1-D conv backbone
        def response_fn(x):           # (N, C, T) -> (N,)
            return model(x).squeeze(-1)

        sensitivity = occlusion_sensitivity_1d(
            response_fn, x_batch, window=10, stride=5
        )  # (N, P)

        # Plot channel 0 of the first sample with a per-channel overlay.
        # For per-channel maps compute sensitivity per channel separately,
        # or pass a (C, P) array built from per-channel response functions.
        plot_occlusion_stacked(
            signal=x_batch[0],          # (C, T)
            sensitivity=sensitivity[0], # (P,) — broadcast to all channels
            channel_names=["Cz", "C3", "C4", "Pz"],
            xlabel="time (samples)",
        )
    """
    sig = _to_numpy(signal)
    if sig.ndim == 1:
        sig = sig[None, :]       # treat as single channel
    C, T = sig.shape

    sens = _to_numpy(sensitivity)
    if sens.ndim == 1:
        sens = np.broadcast_to(sens[None, :], (C, len(sens)))
    if sens.shape[0] != C:
        raise ValueError(
            f"sensitivity has {sens.shape[0]} rows but signal has {C} channels"
        )

    t_axis = _to_numpy(time) if time is not None else np.arange(T, dtype=float)

    names = list(channel_names) if channel_names is not None else [f"ch_{i}" for i in range(C)]
    if len(names) != C:
        raise ValueError(f"channel_names has {len(names)} entries, expected {C}")

    _vmin = float(np.nanmin(sens)) if vmin is None else vmin
    _vmax = float(np.nanmax(sens)) if vmax is None else vmax
    if _vmax <= _vmin:
        _vmax = _vmin + 1e-8

    norm = mcolors.Normalize(vmin=_vmin, vmax=_vmax)
    cm = plt.get_cmap(cmap)

    fig_w = 9.0
    fig_h = panel_height * C + 1.0
    fig = plt.figure(figsize=figsize or (fig_w, fig_h))

    # Reserve the rightmost ~5 % of the figure for the colour bar.
    gs = GridSpec(
        C, 2,
        figure=fig,
        width_ratios=[1.0, 0.04],
        hspace=0.06,
        wspace=0.04,
        left=0.10, right=0.88,
        top=0.90, bottom=0.10,
    )

    axes = [fig.add_subplot(gs[i, 0]) for i in range(C)]
    cbar_ax = fig.add_subplot(gs[:, 1])

    for i, (ax, name_ch) in enumerate(zip(axes, names)):
        # Upsample sensitivity row to signal resolution.
        heat = _upsample_1d(sens[i], T)

        # Heatmap: use pcolormesh over two dummy y-rows spanning the signal range.
        sig_min, sig_max = float(sig[i].min()), float(sig[i].max())
        pad = max((sig_max - sig_min) * 0.15, 1e-6)
        y_lo, y_hi = sig_min - pad, sig_max + pad

        # pcolormesh expects (2, T+1) shaped vertices for a (1, T) cell grid.
        t_edges = np.empty(T + 1)
        t_edges[:-1] = t_axis
        t_edges[-1] = t_axis[-1] + (t_axis[-1] - t_axis[-2]) if T > 1 else t_axis[-1] + 1.0
        y_edges = np.array([y_lo, y_hi])

        # Reshape heat to (1, T) for pcolormesh.
        ax.pcolormesh(
            t_edges, y_edges, heat[None, :],
            cmap=cm, norm=norm, alpha=alpha,
            shading="flat", zorder=1,
        )

        # Signal line drawn on top.
        ax.plot(t_axis, sig[i], color=line_color, lw=1.3, zorder=2)
        ax.set_xlim(t_axis[0], t_axis[-1])
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel(name_ch, fontsize="small", rotation=0, labelpad=40, va="center")
        ax.yaxis.set_label_position("left")
        ax.tick_params(axis="y", labelsize="x-small")

        # Only the bottom panel gets x-tick labels; others share the axis silently.
        if i < C - 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel(xlabel, fontsize="small")
            ax.tick_params(axis="x", labelsize="x-small")

        ax.grid(True, axis="x", alpha=0.25, lw=0.6)
        ax.grid(False, axis="y")

    # Shared colour bar on the right strip.
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label(cbar_label, fontsize="small")
    cb.ax.tick_params(labelsize="x-small")

    fig.suptitle(title, fontsize="medium", y=0.97)
    # GridSpec + colorbar axis is not tight_layout-compatible; suppress the warning.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save_figure(fig, name, plots_dir=plots_dir)


# ---------------------------------------------------------------------------
# RUL / health-index prediction panel
# ---------------------------------------------------------------------------

def plot_rul_prediction(
    time: "np.ndarray | torch.Tensor",
    rul_true: "np.ndarray | torch.Tensor",
    rul_pred: "np.ndarray | torch.Tensor",
    *,
    health_true: Optional["np.ndarray | torch.Tensor"] = None,
    onset_time: Optional[float] = None,
    failure_time: Optional[float] = None,
    title: str = "RUL prediction",
    xlabel: str = "time",
    rul_ylabel: str = "remaining useful life",
    health_ylabel: str = "health index",
    name: str = "rul_prediction",
    plots_dir: Path = DEFAULT_PLOTS_DIR,
    figsize: Optional[Tuple[float, float]] = None,
) -> List[Path]:
    """Predicted vs ground-truth RUL, with optional health index and degradation shading.

    Produces a one- or two-panel figure:

    * **Top panel** (always): ground-truth RUL and predicted RUL on the same axis.
    * **Bottom panel** (if *health_true* supplied): ground-truth health index.

    Shaded regions mark the degradation window (onset → failure, amber) and the
    post-failure period (red), matching SimWeave's ``plot_health_index`` convention
    so both plots can be placed side-by-side in the paper.

    Args:
        time: 1-D time axis ``[T]``.
        rul_true: ground-truth RUL ``[T]``; ``inf`` values before fault onset are
            clipped to the maximum finite value for display.
        rul_pred: model-predicted RUL ``[T]``.
        health_true: optional ground-truth health index ``[T]`` in ``[0, 1]``.
        onset_time: start of the degradation window for shading (optional).
        failure_time: end of the degradation window / start of failure for shading.
        title: figure suptitle.
        name: filename stem for :func:`~polyweave.viz.plots.save_figure`.
        plots_dir: directory for saved figures.
        figsize: explicit figure size; auto-computed if ``None``.

    Returns:
        List of written file paths (PDF + PNG).
    """
    t = _to_numpy(time)
    gt = _to_numpy(rul_true).copy()
    pred = _to_numpy(rul_pred)

    # Clip infinite RUL to the max finite value for display.
    finite_mask = np.isfinite(gt)
    if finite_mask.any():
        gt[~finite_mask] = gt[finite_mask].max()

    n_panels = 2 if health_true is not None else 1
    figsize = figsize or (9.0, 2.8 * n_panels + 0.6)
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True,
                              gridspec_kw={"hspace": 0.12})
    if n_panels == 1:
        axes = [axes]

    def _shade(ax):
        """Add degradation onset and failure shading."""
        if onset_time is not None and failure_time is not None:
            ax.axvspan(onset_time, failure_time, color="#E69F00", alpha=0.18,
                       label="degradation window", zorder=0)
        if failure_time is not None:
            ax.axvspan(failure_time, t[-1], color="#D55E00", alpha=0.18,
                       label="failed", zorder=0)

    # RUL panel.
    ax_rul = axes[0]
    _shade(ax_rul)
    ax_rul.plot(t, gt, lw=1.8, color=OKABE_ITO[0], label="true RUL")
    ax_rul.plot(t, pred, lw=1.6, color=OKABE_ITO[1], ls="--", label="predicted RUL")
    ax_rul.set_ylabel(rul_ylabel, fontsize="small")
    ax_rul.legend(fontsize="x-small", loc="upper right")
    ax_rul.tick_params(labelsize="x-small")
    ax_rul.grid(True, alpha=0.3)

    # Health index panel (optional).
    if health_true is not None:
        hi = _to_numpy(health_true)
        ax_hi = axes[1]
        _shade(ax_hi)
        ax_hi.plot(t, hi, lw=1.8, color=OKABE_ITO[2], label="health index")
        ax_hi.set_ylim(-0.05, 1.08)
        ax_hi.set_ylabel(health_ylabel, fontsize="small")
        ax_hi.legend(fontsize="x-small", loc="upper right")
        ax_hi.tick_params(labelsize="x-small")
        ax_hi.grid(True, alpha=0.3)

    axes[-1].set_xlabel(xlabel, fontsize="small")
    axes[-1].tick_params(axis="x", labelsize="x-small")
    fig.suptitle(title, fontsize="medium")
    fig.tight_layout()
    return save_figure(fig, name, plots_dir=plots_dir)


# ---------------------------------------------------------------------------
# Multi-channel timeseries comparison (model vs baseline)
# ---------------------------------------------------------------------------

def plot_timeseries_predictions(
    time: "np.ndarray | torch.Tensor",
    targets: "np.ndarray | torch.Tensor",
    predictions: Dict[str, "np.ndarray | torch.Tensor"],
    *,
    channel_names: Optional[Sequence[str]] = None,
    onset_time: Optional[float] = None,
    failure_time: Optional[float] = None,
    xlabel: str = "time",
    ylabel: str = "value",
    title: str = "Model predictions vs ground truth",
    name: str = "ts_predictions",
    plots_dir: Path = DEFAULT_PLOTS_DIR,
    figsize: Optional[Tuple[float, float]] = None,
    panel_height: float = 2.2,
) -> List[Path]:
    """Stacked panels comparing model predictions against ground truth.

    Each output channel gets its own panel (shared time axis).  Ground truth is
    drawn as a solid line; each model in *predictions* gets a dashed line.  Useful
    for comparing ``PolyConv1d`` vs MLP vs SigmaPi on a regression task (e.g. RUL
    per suspension DOF).

    Args:
        time: 1-D time axis ``[T]``.
        targets: ground-truth output ``[C, T]`` or ``[T]`` for single channel.
        predictions: ``{model_name: array [C, T]}`` — one entry per model to compare.
        channel_names: optional output channel names.
        onset_time / failure_time: optional degradation-window shading.
        panel_height: height per channel panel when figsize is auto-computed.

    Returns:
        List of written file paths (PDF + PNG).
    """
    t = _to_numpy(time)
    tgt = _to_numpy(targets)
    if tgt.ndim == 1:
        tgt = tgt[None, :]
    C, T = tgt.shape

    preds = {k: _to_numpy(v) for k, v in predictions.items()}
    for k, v in preds.items():
        if v.ndim == 1:
            preds[k] = v[None, :]

    names = list(channel_names) if channel_names is not None else [f"out_{i}" for i in range(C)]

    figsize = figsize or (9.0, panel_height * C + 0.8)
    fig, axes = plt.subplots(C, 1, figsize=figsize, sharex=True,
                              gridspec_kw={"hspace": 0.10})
    if C == 1:
        axes = [axes]

    def _shade(ax):
        if onset_time is not None and failure_time is not None:
            ax.axvspan(onset_time, failure_time, color="#E69F00", alpha=0.18, zorder=0)
        if failure_time is not None:
            ax.axvspan(failure_time, t[-1], color="#D55E00", alpha=0.18, zorder=0)

    for i, ax in enumerate(axes):
        _shade(ax)
        ax.plot(t, tgt[i], lw=1.8, color="black", label="ground truth", zorder=3)
        for j, (model_name, pred_arr) in enumerate(preds.items()):
            ax.plot(t, pred_arr[i], lw=1.5, ls="--",
                    color=OKABE_ITO[j % len(OKABE_ITO)], label=model_name, zorder=2)
        ax.set_ylabel(names[i], fontsize="small", rotation=0, labelpad=44, va="center")
        ax.tick_params(labelsize="x-small")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize="x-small", loc="upper right")

    axes[-1].set_xlabel(xlabel, fontsize="small")
    axes[-1].tick_params(axis="x", labelsize="x-small")
    fig.suptitle(title, fontsize="medium")
    fig.tight_layout()
    return save_figure(fig, name, plots_dir=plots_dir)
