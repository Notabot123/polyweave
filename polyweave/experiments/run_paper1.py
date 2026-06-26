"""Paper 1 experiment orchestrator — single-seed fast pass or multi-seed final run.

Runs all Paper 1 experiments in sequence and collects results into a single
JSON summary and a combined accuracy/metrics bar chart.

Experiments
-----------
1. ``suspension`` — SimWeave FullCarModel RUL regression (PolyConv1d vs MLP vs PolyLinear)
2. ``eeg``        — BCI-IV-2a 4-class motor imagery (PolyConv1d vs ShallowConv vs MLP)
3. ``cifar``      — CIFAR-10 conv1 filter generation (existing hypernetwork experiment)

Design philosophy
-----------------
* **Single-seed first** (default ``--seeds 42``): all three experiments run once,
  quickly, so you can check plots and tweak hyperparameters before committing to
  a multi-seed run.
* **Multi-seed final** (``--seeds 42 43 44``): each experiment runs independently
  per seed; results are cached to ``plots/raw/paper1_<exp>_seed<N>.json`` so a
  crash never discards finished work.
* **Selective re-runs**: ``--experiments suspension eeg`` skips CIFAR if you only
  want to re-run the new experiments.

Usage::

    # Quick pilot: single seed, 5 epochs each, 6 MC suspension runs
    python -m polyweave.experiments.run_paper1 \\
        --seeds 42 --suspension-n-runs 6 --suspension-epochs 5 \\
        --eeg-subjects 1 2 --eeg-epochs 5

    # Standard single-seed pass
    python -m polyweave.experiments.run_paper1 --seeds 42

    # Full multi-seed run (paper-quality)
    python -m polyweave.experiments.run_paper1 --seeds 42 43 44 \\
        --suspension-n-runs 200 --suspension-epochs 80 \\
        --eeg-epochs 150

Output
------
* ``plots/paper1_results.json``        — all raw + aggregated numbers
* ``plots/paper1_summary.{pdf,png}``   — combined accuracy/metrics bar chart
* Per-experiment plots saved by each sub-experiment into ``plots/``
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from polyweave.viz.plots import configure_plots, save_figure, OKABE_ITO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW_DIR = Path("plots/raw")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to plain Python for JSON."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _load_cache(path: Path) -> Optional[Dict]:
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_cache(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_safe(data), f, indent=2)


# ---------------------------------------------------------------------------
# Suspension runner
# ---------------------------------------------------------------------------

def run_suspension(seed: int, cfg_args) -> Dict:
    """Run suspension RUL experiment for one seed; return metrics dict."""
    from polyweave.experiments.suspension_rul import run, DEFAULTS
    import argparse as ap

    cache_path = RAW_DIR / f"paper1_suspension_seed{seed}.json"
    if not cfg_args.force:
        cached = _load_cache(cache_path)
        if cached is not None:
            print(f"  [suspension seed={seed}] loaded from cache")
            return cached

    args = ap.Namespace(
        seed=seed,
        n_runs=cfg_args.suspension_n_runs,
        train_frac=0.80,
        epochs=cfg_args.suspension_epochs,
        batch_size=256,
        lr=1e-3,
        seq_len=50,
        train_stride=5,
        test_stride=50,
        channels=32,
        n_blocks=3,
        kernel_size=7,
        rank=8,
        mlp_hidden=128,
        max_rul=None,
        damping_attr="c_s",
        dt=0.01,
        road_amp=0.004,
        nominal_damping=1500.0,
        max_delta=-0.80,
        plots_dir=cfg_args.plots_dir,
        device=cfg_args.device,
    )

    t0 = time.perf_counter()
    from polyweave.experiments.suspension_rul import run as _run
    result = _run(args)
    elapsed = time.perf_counter() - t0

    metrics = {
        "seed": seed,
        "elapsed_s": elapsed,
        "mae":  {k: ev.mae  for k, ev in result.metrics.items()},
        "rmse": {k: ev.rmse for k, ev in result.metrics.items()},
        "max_rul": result.max_rul,
    }
    _save_cache(cache_path, metrics)
    return metrics


# ---------------------------------------------------------------------------
# EEG runner
# ---------------------------------------------------------------------------

def run_eeg(seed: int, cfg_args) -> Dict:
    """Run EEG experiment for one seed; return per-subject + mean accuracy."""
    cache_path = RAW_DIR / f"paper1_eeg_seed{seed}.json"
    if not cfg_args.force:
        cached = _load_cache(cache_path)
        if cached is not None:
            print(f"  [eeg seed={seed}] loaded from cache")
            return cached

    import argparse as ap
    args = ap.Namespace(
        subjects=cfg_args.eeg_subjects,
        seed=seed,
        epochs=cfg_args.eeg_epochs,
        batch_size=64,
        lr=1e-3,
        channels=32,
        n_blocks=3,
        kernel_size=25,
        rank=8,
        mlp_hidden=256,
        fmin=4.0,
        fmax=40.0,
        tmin=0.0,
        tmax=4.0,
        occlusion_window=25,
        occlusion_stride=10,
        plots_dir=cfg_args.plots_dir,
        device=cfg_args.device,
    )

    t0 = time.perf_counter()
    from polyweave.experiments.eeg_bciiv2a import run as _run
    result = _run(args)
    elapsed = time.perf_counter() - t0

    metrics = {
        "seed": seed,
        "elapsed_s": elapsed,
        "mean_acc":  result.mean_accs,
        "std_acc":   result.std_accs,
        "per_subject": [
            {"subject": sr.subject, "test_accs": sr.test_accs}
            for sr in result.per_subject
        ],
    }
    _save_cache(cache_path, metrics)
    return metrics


# ---------------------------------------------------------------------------
# CIFAR runner (existing hypernetwork experiments)
# ---------------------------------------------------------------------------

def run_cifar(seed: int, cfg_args) -> Dict:
    """Run existing CIFAR conv1 hypernetwork experiment for one seed."""
    cache_path = RAW_DIR / f"paper1_cifar_seed{seed}.json"
    if not cfg_args.force:
        cached = _load_cache(cache_path)
        if cached is not None:
            print(f"  [cifar seed={seed}] loaded from cache")
            return cached

    t0 = time.perf_counter()
    from polyweave.experiments.cifar_conv1 import Config, run as _run
    cfg = Config(seed=seed, plots_dir=cfg_args.plots_dir)
    result = _run(cfg)
    elapsed = time.perf_counter() - t0

    # Extract the key numbers: zero-shot seen/unseen accuracy per method
    metrics = {
        "seed": seed,
        "elapsed_s": elapsed,
        "zeroshot_seen":   _json_safe(getattr(result, "zeroshot_seen",   {})),
        "zeroshot_unseen": _json_safe(getattr(result, "zeroshot_unseen", {})),
        "pi_scale_mean":   _json_safe(getattr(result, "pi_scale_mean",   {})),
    }
    _save_cache(cache_path, metrics)
    return metrics


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _agg(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def aggregate_suspension(all_metrics: List[Dict]) -> Dict:
    model_names = list(all_metrics[0]["mae"].keys())
    return {
        "mae":  {n: _agg([m["mae"][n]  for m in all_metrics]) for n in model_names},
        "rmse": {n: _agg([m["rmse"][n] for m in all_metrics]) for n in model_names},
    }


def aggregate_eeg(all_metrics: List[Dict]) -> Dict:
    model_names = list(all_metrics[0]["mean_acc"].keys())
    return {
        "mean_acc": {n: _agg([m["mean_acc"][n] for m in all_metrics]) for n in model_names},
    }


# ---------------------------------------------------------------------------
# Summary plot
# ---------------------------------------------------------------------------

def plot_summary(
    susp_agg: Optional[Dict],
    eeg_agg:  Optional[Dict],
    seeds: List[int],
    plots_dir: Path,
) -> None:
    """Combined two-panel summary: suspension RMSE and EEG accuracy."""
    n_panels = sum([susp_agg is not None, eeg_agg is not None])
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    panel = 0

    if susp_agg is not None:
        ax = axes[panel]; panel += 1
        names  = list(susp_agg["rmse"].keys())
        means  = [susp_agg["rmse"][n]["mean"] for n in names]
        stds   = [susp_agg["rmse"][n]["std"]  for n in names]
        x = np.arange(len(names))
        ax.bar(x, means, 0.6, yerr=stds if len(seeds) > 1 else None,
               capsize=5,
               color=[OKABE_ITO[i % len(OKABE_ITO)] for i in range(len(names))],
               edgecolor="black", linewidth=0.7)
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_ylabel("test RMSE (s)")
        ax.set_title(f"Suspension RUL\n(seeds {seeds})")
        ax.grid(True, axis="y", alpha=0.3)

    if eeg_agg is not None:
        ax = axes[panel]; panel += 1
        names  = list(eeg_agg["mean_acc"].keys())
        means  = [eeg_agg["mean_acc"][n]["mean"] for n in names]
        stds   = [eeg_agg["mean_acc"][n]["std"]  for n in names]
        x = np.arange(len(names))
        ax.bar(x, means, 0.6, yerr=stds if len(seeds) > 1 else None,
               capsize=5,
               color=[OKABE_ITO[i % len(OKABE_ITO)] for i in range(len(names))],
               edgecolor="black", linewidth=0.7)
        ax.axhline(0.25, ls="--", lw=1.2, color="grey", label="chance (25 %)")
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_ylim(0, 1)
        ax.set_ylabel("mean test accuracy")
        ax.set_title(f"EEG BCI-IV-2a\n(seeds {seeds})")
        ax.legend(fontsize="x-small")
        ax.grid(True, axis="y", alpha=0.3)

    seed_str = "_".join(str(s) for s in seeds)
    fig.suptitle(f"Paper 1 — experiment summary  (seeds: {seeds})", fontsize="medium")
    fig.tight_layout()
    save_figure(fig, f"paper1_summary_seeds{seed_str}", plots_dir=plots_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg) -> Dict:
    configure_plots()
    plots_dir = Path(cfg.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    seeds = list(cfg.seeds)
    exps  = set(cfg.experiments)
    t_total = time.perf_counter()

    all_results: Dict[str, List[Dict]] = {
        "suspension": [], "eeg": [], "cifar": [],
    }

    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"# Seed {seed}")
        print(f"{'#'*60}")

        if "suspension" in exps:
            print(f"\n[suspension seed={seed}]")
            m = run_suspension(seed, cfg)
            all_results["suspension"].append(m)
            print(f"  MAE:  { {k: f'{v:.2f}s' for k, v in m['mae'].items()} }")
            print(f"  RMSE: { {k: f'{v:.2f}s' for k, v in m['rmse'].items()} }")

        if "eeg" in exps:
            print(f"\n[eeg seed={seed}]")
            m = run_eeg(seed, cfg)
            all_results["eeg"].append(m)
            print(f"  Mean acc: { {k: f'{v:.3f}' for k, v in m['mean_acc'].items()} }")

        if "cifar" in exps:
            print(f"\n[cifar seed={seed}]")
            m = run_cifar(seed, cfg)
            all_results["cifar"].append(m)

    # Aggregate across seeds
    print(f"\n{'='*60}")
    print("Aggregated results")
    print(f"{'='*60}")

    susp_agg = eeg_agg = None

    if all_results["suspension"]:
        susp_agg = aggregate_suspension(all_results["suspension"])
        print("\nSuspension RUL (RMSE, s):")
        for name, v in susp_agg["rmse"].items():
            print(f"  {name:15s}  {v['mean']:.2f} ± {v['std']:.2f}")

    if all_results["eeg"]:
        eeg_agg = aggregate_eeg(all_results["eeg"])
        print("\nEEG accuracy:")
        for name, v in eeg_agg["mean_acc"].items():
            print(f"  {name:15s}  {v['mean']:.3f} ± {v['std']:.3f}")

    # Save JSON
    summary = {
        "seeds": seeds,
        "experiments": list(exps),
        "raw": _json_safe(all_results),
        "aggregated": _json_safe({
            "suspension": susp_agg,
            "eeg":        eeg_agg,
        }),
    }
    json_path = plots_dir / "paper1_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Summary plot
    plot_summary(susp_agg, eeg_agg, seeds, plots_dir)

    print(f"\nTotal wall time: {time.perf_counter() - t_total:.1f}s")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Paper 1 experiment orchestrator")
    p.add_argument("--seeds",         type=int, nargs="+", default=[42],
                   help="Random seeds to run (default: 42 for single-seed pass)")
    p.add_argument("--experiments",   type=str, nargs="+",
                   default=["suspension", "eeg", "cifar"],
                   choices=["suspension", "eeg", "cifar"],
                   help="Which experiments to run")
    # Suspension
    p.add_argument("--suspension-n-runs",  type=int, default=50)
    p.add_argument("--suspension-epochs",  type=int, default=30)
    # EEG
    p.add_argument("--eeg-subjects",  type=int, nargs="+", default=list(range(1, 10)))
    p.add_argument("--eeg-epochs",    type=int, default=30)
    # Shared
    p.add_argument("--plots-dir",     type=str, default="plots")
    p.add_argument("--device",        type=str, default=None)
    p.add_argument("--force",         action="store_true",
                   help="Ignore cached results and re-run everything")
    args = p.parse_args()
    args.experiments = list(args.experiments)
    return args


if __name__ == "__main__":
    cfg = _parse_args()
    run(cfg)
