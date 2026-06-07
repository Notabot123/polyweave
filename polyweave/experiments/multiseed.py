"""Multi-seed driver for the three paper experiments.

Runs each experiment across several *paired* seeds (the same seed list drives the
student population, teacher init, and evaluation for every experiment), then
aggregates the results into paper-ready figures with mean +/- 1 std bands / error
bars:

* ``<prefix>_recovery_multiseed.pdf`` -- recovery curves, mean line + std band.
* ``<prefix>_zeroshot_multiseed.pdf`` -- seen/unseen zero-shot bars + error bars.
* ``polyweave_pi_ordering.pdf``       -- the headline cross-experiment chart of
  delta_pi (Sigma-Pi recruitment) per target type, FC -> conv1 -> Q/K.

It also writes ``plots/multiseed_results.json`` with every raw and aggregated
number so the paper tables can be filled without re-running.

Run (all three, default seeds 42/43/44)::

    python -m polyweave.experiments.multiseed

Subset / custom seeds::

    python -m polyweave.experiments.multiseed --experiments fc conv1 --seeds 42 43
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

from . import _common, cifar_conv1, cifar_fc, synthetic_attention

# key -> (module, short label, plot prefix, friendly title fragment)
EXPERIMENTS = {
    "fc": (cifar_fc, "FC", "polyweave_cifar_fc", "FC head"),
    "conv1": (cifar_conv1, "conv1", "polyweave_cifar_conv1", "conv1 filters"),
    "qk": (synthetic_attention, "Q/K", "polyweave_synthetic_attention", "attention Q/K"),
}

DEFAULT_SEEDS = (42, 43, 44)

# Per-seed result caches live here so a crash never discards a finished seed.
RAW_DIR = Path("plots/raw")


def _base_config(key: str, save_models_dir: str | None):
    """Construct the default Config for an experiment, with optional model save."""
    module = EXPERIMENTS[key][0]
    cfg = module.Config()
    if key == "conv1" and save_models_dir is not None:
        cfg = dataclasses.replace(cfg, save_models_dir=save_models_dir)
    return cfg


def run_experiment(
    key: str,
    seeds: Sequence[int],
    *,
    save_models_dir: str | None = None,
) -> List[_common.RunResult]:
    """Run one experiment across ``seeds``; models (conv1) saved on first seed only."""
    module, label, _prefix, frag = EXPERIMENTS[key]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    results: List[_common.RunResult] = []
    for i, seed in enumerate(seeds):
        cache = RAW_DIR / f"{key}_seed{seed}.json"
        if cache.exists():
            res = _common.RunResult.from_dict(json.loads(cache.read_text()))
            print(f"[{label} seed {seed}] resumed from cache {cache} "
                  f"(pi_delta={res.pi_delta})")
            results.append(res)
            continue
        # Only save models for the very first seed to keep disk use modest.
        save_dir = save_models_dir if (i == 0) else None
        cfg = _base_config(key, save_dir)
        cfg = dataclasses.replace(cfg, seed=seed)
        print("\n" + "#" * 70)
        print(f"# {frag}  --  seed {seed}  ({i + 1}/{len(seeds)})")
        print("#" * 70)
        t0 = time.time()
        res = module.run(cfg, make_plots=False)
        # Persist immediately so a later crash never discards this seed.
        cache.write_text(json.dumps(res.to_dict(), indent=2))
        print(f"[{label} seed {seed}] finished in {time.time() - t0:.0f}s "
              f"(pi_delta={res.pi_delta}) -> cached {cache}")
        results.append(res)
    return results


def aggregate_experiment(key: str, results: List[_common.RunResult]) -> dict:
    """Make the per-experiment multi-seed plots; return aggregated numbers."""
    _module, label, prefix, frag = EXPERIMENTS[key]
    n = len(results)

    bands = _common.aggregate_recovery(results)
    _common.plot_recovery_band(
        bands, name=f"{prefix}_recovery_multiseed",
        title=f"{frag}: recovery after zero-shot init "
              f"(unseen archs, mean$\\pm$std over {n} seeds)",
    )

    seen, unseen = _common.aggregate_zeroshot(results)
    _common.plot_zeroshot_grouped_std(
        seen, unseen, name=f"{prefix}_zeroshot_multiseed",
        title=f"{frag}: zero-shot accuracy (mean$\\pm$std over {n} seeds)",
    )

    pi_mean, pi_std = _common.aggregate_pi_delta(results)
    print(f"\n[{label}] Sigma-Pi delta_pi over {n} seeds: "
          f"{pi_mean:+.5f} +/- {pi_std:.5f}")
    return {
        "label": label,
        "n_seeds": n,
        "pi_delta_mean": pi_mean,
        "pi_delta_std": pi_std,
        "zeroshot_seen": {m: {"mean": v[0], "std": v[1]} for m, v in seen.items()},
        "zeroshot_unseen": {m: {"mean": v[0], "std": v[1]} for m, v in unseen.items()},
        "per_seed": [r.to_dict() for r in results],
    }


def main(argv: Sequence[str] | None = None) -> None:
    # Force UTF-8 on stdout/stderr so no non-ASCII log line can ever kill a run
    # (Windows defaults to cp1252 when stdout is redirected to a file).
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS),
                    choices=list(EXPERIMENTS), help="which experiments to run")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
                    help="paired seeds shared across all experiments")
    ap.add_argument("--save-models-dir", default="models",
                    help="dir for conv1 model checkpoints (ensemble reuse); '' to disable")
    ap.add_argument("--results-json", default="plots/multiseed_results.json",
                    help="where to write the aggregated numbers")
    args = ap.parse_args(argv)

    _common.configure_plots(dark=False)
    save_models_dir = args.save_models_dir or None

    summary: Dict[str, dict] = {"seeds": list(args.seeds), "experiments": {}}
    pi_ordering: Dict[str, tuple] = {}

    out = Path(args.results_json)
    out.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    for key in args.experiments:
        results = run_experiment(key, args.seeds, save_models_dir=save_models_dir)
        agg = aggregate_experiment(key, results)
        summary["experiments"][key] = agg
        pi_ordering[agg["label"]] = (agg["pi_delta_mean"], agg["pi_delta_std"])
        # Checkpoint after every experiment so a late crash keeps finished work.
        out.write_text(json.dumps(summary, indent=2))
        print(f"[checkpoint] wrote partial results after '{key}' -> {out}")

    # Headline cross-experiment ordering chart (only if every core experiment ran,
    # in the canonical FC -> conv1 -> Q/K order).
    canonical = [EXPERIMENTS[k][1] for k in ("fc", "conv1", "qk") if k in args.experiments]
    if len(canonical) >= 2:
        ordered = {lab: pi_ordering[lab] for lab in canonical}
        _common.plot_pi_ordering(ordered, name="polyweave_pi_ordering")
        print("\n=== Sigma-Pi recruitment ordering (delta_pi, mean +/- std) ===")
        for lab in canonical:
            m, s = pi_ordering[lab]
            print(f"  {lab:<6} {m:+.5f} +/- {s:.5f}")

    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote aggregated results -> {out}")
    print(f"Total wall time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
