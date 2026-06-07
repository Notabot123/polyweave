"""Random-baseline recovery only — regenerate corrected random recovery curves.

Background
----------
The refactored ``random_like`` baseline (``polyweave.evaluation.baselines``) drew
weights from a *unit-variance* normal instead of the original Kaiming-linear scale
(std ~= 1/sqrt(fan_in)). A unit-variance head saturates the softmax and stalls
fine-tuning, so the multi-seed *random* recovery finals for Experiments 1 (FC) and
2 (conv1) came out far too low. That bug is now fixed (fan-in scaling is the
default). This driver regenerates *just* the corrected random recovery numbers
**without retraining any teacher** — the random baseline never touches a teacher.

Scope
-----
Only **FC** and **conv1** use ``random_like`` and are therefore affected. The
attention (Q/K) random baseline uses a Xavier-uniform init (``_random_qk``), not
``random_like``, so it is *not* affected and is intentionally excluded here.

What it does (per experiment, per seed)
---------------------------------------
1. Rebuilds the student population exactly as ``run()`` does (this trains the
   student *trunks*; it does **not** train teachers).
2. Runs the module's own ``_recovery`` restricted to ``{"random": None}`` — using
   the fixed ``random_like``.
3. Aggregates the per-seed mean recovery curves into mean +/- std bands across
   seeds, writes ``plots/random_recovery_results.json`` and a recovery-band PDF.

These numbers are a throwaway estimate to confirm the fix at scale / fill the
recovery tables in the interim; a full multi-seed re-run would regenerate them
(and everything else) consistently.

Run:  python -m polyweave.experiments.random_recovery --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from . import _common, cifar_conv1, cifar_fc

RAW_DIR = Path("plots/raw_random")
RANDOM_ONLY: Dict[str, None] = {"random": None}


def _fc_random_curve(cfg: cifar_fc.Config):
    train_loader, test_loader = _common.cifar10_loaders(cfg.batch_size)
    groups = cifar_fc._make_population(cfg, train_loader)
    _, unseen = _common.split_seen_unseen(groups, cfg.num_train_groups)
    eval_batches = _common.collect_batches(
        test_loader, cfg.eval_max_batches or len(test_loader), cfg.device
    )
    rec = cifar_fc._recovery(unseen, RANDOM_ONLY, eval_batches, train_loader, cfg)
    return rec["random"]


def _conv1_random_curve(cfg: cifar_conv1.Config):
    train_loader, test_loader = _common.cifar10_loaders(cfg.batch_size)
    groups = cifar_conv1._make_population(cfg, train_loader)
    _, unseen = _common.split_seen_unseen(groups, cfg.num_train_groups)
    eval_batches = _common.collect_batches(
        test_loader, cfg.eval_max_batches or len(test_loader), cfg.device
    )
    ref_gen = {
        "weight": torch.zeros(
            cifar_conv1.CONV1_OUT, cifar_conv1.CONV1_IN,
            cifar_conv1.CONV1_KERNEL, cifar_conv1.CONV1_KERNEL, device=cfg.device,
        ),
        "bias": torch.zeros(cifar_conv1.CONV1_OUT, device=cfg.device),
    }
    rec = cifar_conv1._recovery(unseen, RANDOM_ONLY, eval_batches, train_loader, cfg, ref_gen)
    return rec["random"]


# key -> (module, label, plot prefix, curve fn)
EXPERIMENTS = {
    "fc": (cifar_fc, "FC", "polyweave_cifar_fc", _fc_random_curve),
    "conv1": (cifar_conv1, "conv1", "polyweave_cifar_conv1", _conv1_random_curve),
}

DEFAULT_SEEDS = (42, 43, 44)


def run_experiment(key: str, seeds: Sequence[int]) -> List[_common.RunResult]:
    module, label, _prefix, curve_fn = EXPERIMENTS[key]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    results: List[_common.RunResult] = []
    for i, seed in enumerate(seeds):
        cache = RAW_DIR / f"{key}_seed{seed}.json"
        if cache.exists():
            res = _common.RunResult.from_dict(json.loads(cache.read_text()))
            print(f"[{label} seed {seed}] resumed from cache {cache}")
            results.append(res)
            continue
        from ..utils import set_seed
        set_seed(seed)
        cfg = dataclasses.replace(module.Config(), seed=seed)
        print("\n" + "#" * 70)
        print(f"# {label} random-recovery  --  seed {seed}  ({i + 1}/{len(seeds)})")
        print("#" * 70)
        t0 = time.time()
        curve = curve_fn(cfg)
        res = _common.RunResult(seed=seed, label=label, recovery={"random": curve})
        cache.write_text(json.dumps(res.to_dict(), indent=2))
        final = curve[-1][1] if curve else float("nan")
        print(f"[{label} seed {seed}] random recovery final={final:.4f} "
              f"in {time.time() - t0:.0f}s -> cached {cache}")
        results.append(res)
    return results


def main(argv: Sequence[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS),
                    choices=list(EXPERIMENTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument("--results-json", default="plots/random_recovery_results.json")
    args = ap.parse_args(argv)

    _common.configure_plots(dark=False)
    summary: Dict[str, dict] = {"seeds": list(args.seeds), "experiments": {}}
    out = Path(args.results_json)
    out.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    for key in args.experiments:
        _module, label, prefix, _fn = EXPERIMENTS[key]
        results = run_experiment(key, args.seeds)
        bands = _common.aggregate_recovery(results)
        steps, mean, std = bands["random"]
        _common.plot_recovery_band(
            {"random": bands["random"]},
            name=f"{prefix}_random_recovery_fixed",
            title=f"{label}: corrected random recovery "
                  f"(fan-in init, mean$\\pm$std over {len(results)} seeds)",
        )
        finals = [r.recovery["random"][-1][1] for r in results]
        fmean = sum(finals) / len(finals)
        fstd = (sum((f - fmean) ** 2 for f in finals) / max(len(finals) - 1, 1)) ** 0.5
        summary["experiments"][key] = {
            "label": label,
            "per_seed_final": {r.seed: r.recovery["random"][-1][1] for r in results},
            "final_mean": fmean,
            "final_std": fstd,
            "steps": steps,
            "mean_curve": mean,
            "std_curve": std,
        }
        out.write_text(json.dumps(summary, indent=2))
        print(f"\n[{label}] corrected random recovery final: "
              f"{fmean:.4f} +/- {fstd:.4f}  (per seed: "
              f"{', '.join(f'{f:.3f}' for f in finals)})")

    print(f"\nWrote {out}")
    print(f"Total wall time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
