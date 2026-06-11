"""Diagnostic: why does the trained `dense` (R2~0.25 GPT-2 early) score so far below
a closed-form least-squares linear map (R2~0.97 in-sample)? Mirror the experiment
EXACTLY (30k tokens, contiguous 80/20 split, the experiment's own r2_score) and
decompose the gap into: metric convention, train/val distribution shift (the split is
a contiguous tail — rows are NOT shuffled at capture), and optimisation (trained
nn.Linear vs closed-form optimum on the same split).

Prints, per model + early block:
  dc_frac            energy fraction in the all-ones (feature-mean) direction of X
  lstsq tr/va  (exp) closed-form linear, experiment r2_score, on train / val splits
  lstsq va  (perfeat) closed-form linear, per-feature R2, on val  (standard convention)
  trained va  (exp)  nn.Linear trained by the experiment's fit_layer, exp r2_score, val

Run:  venv/Scripts/python.exe -u run_sigmapi_diag.py
"""

from __future__ import annotations

import torch

from polyweave.distill.metrics import r2_score
from polyweave.distill.regression import fit_layer
from polyweave.experiments.gpt2_mlp_distill import (
    Config, load_model, token_batches, mlp_of, capture_block_io,
)

MODELS = ["gpt2", "JackFram/llama-160m"]
BLOCK = 1
MAX_TOKENS = 30_000
VAL_FRAC = 0.2


def _lstsq_fit(Xtr, Ytr):
    ones = torch.ones(Xtr.shape[0], 1, dtype=torch.float64)
    Xb = torch.cat([Xtr.double(), ones], dim=1)
    return torch.linalg.lstsq(Xb, Ytr.double()).solution  # [d+1, out]


def _apply(W, X):
    ones = torch.ones(X.shape[0], 1, dtype=torch.float64)
    return (torch.cat([X.double(), ones], dim=1) @ W)


def _r2_perfeat(y, p):
    ss_res = (y - p).pow(2).sum()
    ss_tot = (y - y.mean(dim=0, keepdim=True)).pow(2).sum()
    return float(1 - ss_res / ss_tot)


def _dc_fraction(X):
    d = X.shape[1]
    return float(((X.mean(dim=1).pow(2) * d) / X.pow(2).sum(dim=1).clamp_min(1e-12)).mean())


def main() -> None:
    hdr = f"{'model':<20} {'dc_frac':>8} {'lstsq_tr':>9} {'lstsq_va':>9} {'lstsq_va_pf':>12} {'trained_va':>11}"
    print(hdr)
    print("-" * len(hdr))
    for name in MODELS:
        cfg = Config(model_name=name, block_indices=(BLOCK,), dataset="wikitext2",
                     seq_len=128, batch_size=4, max_tokens=MAX_TOKENS,
                     steps=3000, lr=1e-3, fit_batch_size=256, device="cuda"
                     if torch.cuda.is_available() else "cpu")
        model, tok = load_model(cfg)
        batches = token_batches(cfg, tok, split="train")
        X, Y = capture_block_io(model, mlp_of(model, BLOCK), batches, cfg)
        X, Y = X.float().cpu(), Y.float().cpu()

        n = X.shape[0]
        n_tr = n - max(1, round(n * VAL_FRAC))
        Xtr, Ytr, Xva, Yva = X[:n_tr], Y[:n_tr], X[n_tr:], Y[n_tr:]

        W = _lstsq_fit(Xtr, Ytr)
        ptr, pva = _apply(W, Xtr), _apply(W, Xva)
        ls_tr = r2_score(Ytr.double(), ptr)
        ls_va = r2_score(Yva.double(), pva)
        ls_va_pf = _r2_perfeat(Yva.double(), pva)

        lin = torch.nn.Linear(X.shape[1], Y.shape[1])
        res = fit_layer(lin, X, Y, steps=cfg.steps, lr=cfg.lr,
                        batch_size=cfg.fit_batch_size, val_frac=VAL_FRAC,
                        device=cfg.device, seed=42)

        print(f"{name:<20} {_dc_fraction(X):>8.4f} {ls_tr:>9.4f} {ls_va:>9.4f} "
              f"{ls_va_pf:>12.4f} {res.val_r2:>11.4f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
