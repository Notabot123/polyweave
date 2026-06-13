"""CORRECTED multi-seed GPT-2 run (v2). Supersedes run_gpt2_multiseed.py, whose
numbers were confounded by an under-converged linear baseline on ill-conditioned
activations (see project_paper2.md 2026-06-09).

Changes vs v1:
  * linear_closed_form=True  -> the "dense" baseline is the EXACT least-squares linear
    map (the true linear ceiling), not a 3k-step Adam fit that badly underfit it.
  * include_sigma_pi=False   -> the single geometric-product layer is dropped from this
    paper (unstable; adds no value on either GELU or SwiGLU; deferred to future work).
  * steps=8000               -> ample budget so the trained candidates (poly, dense 2x)
    converge; verify each block's poly/dense2x R2 >= the closed-form "dense" ceiling.

Candidates: dense (= closed-form linear ceiling) | poly | dense (2x) depth control.
Run:  venv/Scripts/python.exe -u run_gpt2_multiseed_v2.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, List

from polyweave.experiments.gpt2_mlp_distill import Config, run

MODEL = "gpt2"
SEEDS = (42, 43, 44)
RAW = Path("plots/raw")
AGG_PATH = RAW / "gpt2_mlp_distill_wikitext2_v2_multiseed.json"
CAND_METRICS = ["val_r2", "val_cosine", "val_rmse", "dppl_swap", "dppl_heal"]


def _cfg(seed: int) -> Config:
    return Config(
        model_name=MODEL,
        block_indices=(1, 10),
        block_labels=("early block", "deep block"),
        dataset="wikitext2",
        seq_len=128,
        batch_size=4,
        max_tokens=30_000,
        poly_rank=16,
        equal_budget=True,
        include_sigma_pi=False,
        linear_closed_form=True,
        steps=8000,
        lr=1e-3,
        seed=seed,
        eval_perplexity=True,
        ppl_split="test",
        ppl_max_batches=50,
        heal_steps=200,
        heal_lr=1e-4,
        results_path=str(RAW / f"gpt2_mlp_distill_wikitext2_v2_seed{seed}.json"),
        plot_prefix=f"polyweave_gpt2_mlp_distill_wikitext2_v2_seed{seed}",
    )


def _mean_std(values: List[float]) -> Dict[str, float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {"mean": statistics.fmean(vals), "std": std, "n": len(vals)}


def main() -> None:
    for seed in SEEDS:
        path = RAW / f"gpt2_mlp_distill_wikitext2_v2_seed{seed}.json"
        if path.exists():
            print(f"[seed {seed}] cached -> {path}")
            continue
        print(f"\n########## SEED {seed} ##########")
        run(_cfg(seed), make_plots=False)

    per_seed = [
        json.loads((RAW / f"gpt2_mlp_distill_wikitext2_v2_seed{s}.json").read_text(encoding="utf-8"))
        for s in SEEDS
    ]
    n_blocks = len(per_seed[0])
    agg = []
    for bi in range(n_blocks):
        b0 = per_seed[0][bi]
        block_agg = {
            "label": b0["label"], "block_index": b0["block_index"],
            "mlp_params": b0["mlp_params"],
            "ppl_base": _mean_std([s[bi].get("ppl_base") for s in per_seed]),
            "dppl_heal_original": _mean_std([s[bi].get("dppl_heal_original") for s in per_seed]),
            "candidates": {},
        }
        for name in b0["candidates"]:
            cand_agg = {"num_params": b0["candidates"][name]["num_params"],
                        "compression": b0["candidates"][name]["compression"]}
            for m in CAND_METRICS:
                cand_agg[m] = _mean_std([s[bi]["candidates"][name].get(m) for s in per_seed])
            block_agg["candidates"][name] = cand_agg
        agg.append(block_agg)
    AGG_PATH.write_text(json.dumps(agg, indent=2), encoding="utf-8")

    print(f"\n{'=' * 96}")
    print(f"CORRECTED MULTI-SEED  {MODEL}  (seeds {SEEDS}; dense = closed-form linear ceiling)")
    print("=" * 96)
    print(f"  {'block':<12} {'layer':<12} {'params':>10} {'R2':>15} "
          f"{'cosine':>15} {'dPPL_swap':>15} {'dPPL_heal':>15}")
    print("  " + "-" * 94)
    for b in agg:
        for name, c in b["candidates"].items():
            def f(m):
                d = c[m]
                return f"{d['mean']:.3f}+/-{d['std']:.3f}" if d["mean"] is not None else "-"
            print(f"  {b['label']:<12} {name:<12} {c['num_params']:>10,} "
                  f"{f('val_r2'):>15} {f('val_cosine'):>15} {f('dppl_swap'):>15} {f('dppl_heal'):>15}")
        ho = b["dppl_heal_original"]
        if ho["mean"] is not None:
            ho_str = f"{ho['mean']:.3f}+/-{ho['std']:.3f}"
            print(f"  {b['label']:<12} {'ORIG(heal)':<12} {b['mlp_params']:>10,} "
                  f"{'-':>15} {'-':>15} {'-':>15} {ho_str:>15}")
    print("=" * 96)
    print(f"\nsaved {AGG_PATH}")


if __name__ == "__main__":
    main()
