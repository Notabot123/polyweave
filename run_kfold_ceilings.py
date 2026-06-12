"""Blocked k-fold cross-validation of the closed-form linear ceiling.

Addresses a real question about the reported +/- std: across random seeds the cached
activations and the train/val split are FIXED, so seed-std measures only optimisation
reproducibility, not data/split uncertainty. k-fold CV of the (deterministic) closed-form
ceiling gives an honest variance over DATA SPLITS instead. Because our rows are sequential
token windows (adjacent windows are correlated), we use BLOCKED (contiguous) folds rather
than random folds, so a fold's held-out block is not leaked via neighbours in train.

Cheap (k closed-form solves per block; no training). All 12 blocks x 3 models, WikiText-2.
Run:  venv/Scripts/python.exe -u run_kfold_ceilings.py
Writes plots/raw/kfold_ceilings_wikitext2.{json,csv}.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch
import torch.nn as nn

from polyweave.experiments.gpt2_mlp_distill import (
    Config, load_model, token_batches, _blocks,
)
from run_residual_gain_clean import capture_all_blocks

MODELS = ["gpt2", "EleutherAI/pythia-160m", "JackFram/llama-160m"]
N_BLOCKS = 12
MAX_TOKENS = 15_000
K = 5


def closed_form_r2(Xtr, Ytr, Xva, Yva) -> float:
    Xtr, Ytr, Xva, Yva = (t.double() for t in (Xtr, Ytr, Xva, Yva))
    ones = torch.ones(Xtr.shape[0], 1, dtype=torch.float64, device=Xtr.device)
    W = torch.linalg.lstsq(torch.cat([Xtr, ones], 1), Ytr).solution
    onev = torch.ones(Xva.shape[0], 1, dtype=torch.float64, device=Xva.device)
    pred = torch.cat([Xva, onev], 1) @ W
    ss_res = ((Yva - pred) ** 2).sum()
    ss_tot = ((Yva - Yva.mean(0)) ** 2).sum().clamp_min(1e-12)
    return float(1.0 - ss_res / ss_tot)


def blocked_kfold_r2(X, Y, dev, k=K) -> list[float]:
    n = X.shape[0]
    X, Y = X.to(dev), Y.to(dev)
    out = []
    for j in range(k):
        lo, hi = n * j // k, n * (j + 1) // k
        val_idx = torch.arange(lo, hi)
        tr_idx = torch.cat([torch.arange(0, lo), torch.arange(hi, n)])
        out.append(closed_form_r2(X[tr_idx], Y[tr_idx], X[val_idx], Y[val_idx]))
    return out


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for name in MODELS:
        print(f"\n## {name}", flush=True)
        cfg = Config(model_name=name, dataset="wikitext2", seq_len=128, batch_size=4,
                     max_tokens=MAX_TOKENS, device=dev)
        model, tok = load_model(cfg)
        blocks = list(range(min(N_BLOCKS, len(_blocks(model)))))
        caps = capture_all_blocks(model, blocks, token_batches(cfg, tok, split="train"), cfg)
        for i in blocks:
            X, Y = caps[i]
            folds = blocked_kfold_r2(X, Y, dev)
            mean = statistics.fmean(folds)
            std = statistics.stdev(folds)
            rows.append({"model": name, "block": i, "r2_folds": folds,
                         "r2_mean": mean, "r2_std": std,
                         "r2_min": min(folds), "r2_max": max(folds)})
            print(f"  blk {i:2d}  r2={mean:.3f} +/- {std:.3f}  "
                  f"(min {min(folds):.3f}, max {max(folds):.3f})", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    Path("plots/raw").mkdir(parents=True, exist_ok=True)
    Path("plots/raw/kfold_ceilings_wikitext2.json").write_text(json.dumps(rows, indent=2))
    csv = ["model,block,r2_mean,r2_std,r2_min,r2_max"]
    csv += [f"{r['model']},{r['block']},{r['r2_mean']:.5f},{r['r2_std']:.5f},"
            f"{r['r2_min']:.5f},{r['r2_max']:.5f}" for r in rows]
    Path("plots/raw/kfold_ceilings_wikitext2.csv").write_text("\n".join(csv) + "\n")
    worst = max(rows, key=lambda r: r["r2_std"])
    print(f"\nmax fold-std across all 36 blocks = {worst['r2_std']:.4f} "
          f"({worst['model']} blk {worst['block']})")
    print(f"mean fold-std = {statistics.fmean(r['r2_std'] for r in rows):.4f}")
    print("saved plots/raw/kfold_ceilings_wikitext2.{json,csv}")


if __name__ == "__main__":
    main()
