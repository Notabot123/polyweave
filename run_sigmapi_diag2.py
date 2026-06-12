"""Confirm the GPT-2 `dense` underfit is an ill-conditioning/scale artifact.

If GPT-2's FFN-output target has a few huge-variance (outlier) features, MSE is
dominated by them and Adam at a single lr underfits the rest -> low R2 even though a
closed-form linear map reaches ~0.95. Test: standardize X and Y per-feature (train
stats) and re-fit the SAME nn.Linear; per-feature R2 is invariant to that affine, so
if the trained R2 jumps toward the closed-form ceiling, ill-conditioning is the cause.

Prints, per model + early block:
  y_outlier   max per-feature std / median per-feature std of the target Y
  raw_va      trained nn.Linear, per-feature val R2, RAW targets (current pipeline)
  std_va      trained nn.Linear, per-feature val R2, STANDARDIZED targets (proposed)
  lstsq_va    closed-form linear ceiling, per-feature val R2

Run:  venv/Scripts/python.exe -u run_sigmapi_diag2.py
"""

from __future__ import annotations

import torch

from polyweave.distill.regression import fit_layer
from polyweave.experiments.gpt2_mlp_distill import (
    Config, load_model, token_batches, mlp_of, capture_block_io,
)

MODELS = ["gpt2", "JackFram/llama-160m"]
BLOCK = 1
MAX_TOKENS = 30_000
VAL_FRAC = 0.2


def _r2_perfeat(y, p):
    ss_res = (y - p).pow(2).sum()
    ss_tot = (y - y.mean(dim=0, keepdim=True)).pow(2).sum()
    return float(1 - ss_res / ss_tot)


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hdr = f"{'model':<20} {'y_outlier':>10} {'raw_va':>9} {'std_va':>9} {'lstsq_va':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name in MODELS:
        cfg = Config(model_name=name, block_indices=(BLOCK,), dataset="wikitext2",
                     seq_len=128, batch_size=4, max_tokens=MAX_TOKENS, device=dev)
        model, tok = load_model(cfg)
        batches = token_batches(cfg, tok, split="train")
        X, Y = capture_block_io(model, mlp_of(model, BLOCK), batches, cfg)
        X, Y = X.float().cpu(), Y.float().cpu()

        n = X.shape[0]
        n_tr = n - max(1, round(n * VAL_FRAC))

        std_per_feat = Y[:n_tr].std(dim=0)
        y_outlier = float(std_per_feat.max() / std_per_feat.median())

        # RAW fit (current pipeline).
        lin_raw = torch.nn.Linear(X.shape[1], Y.shape[1])
        r_raw = fit_layer(lin_raw, X, Y, steps=3000, lr=1e-3, batch_size=256,
                          val_frac=VAL_FRAC, device=dev, seed=42)

        # STANDARDIZED fit: z-score X and Y by TRAIN stats, fit, eval per-feature R2
        # (invariant to the per-feature affine, so comparable to the raw fit).
        xm, xs = X[:n_tr].mean(0), X[:n_tr].std(0).clamp_min(1e-6)
        ym, ys = Y[:n_tr].mean(0), Y[:n_tr].std(0).clamp_min(1e-6)
        Xs, Ys = (X - xm) / xs, (Y - ym) / ys
        lin_std = torch.nn.Linear(X.shape[1], Y.shape[1])
        fit_layer(lin_std, Xs, Ys, steps=3000, lr=1e-3, batch_size=256,
                  val_frac=VAL_FRAC, device=dev, seed=42)
        lin_std.eval()
        with torch.no_grad():
            pva_std = lin_std(Xs[n_tr:].to(dev)).cpu()
        std_va = _r2_perfeat(Ys[n_tr:], pva_std)

        # Closed-form ceiling.
        ones = torch.ones(n_tr, 1, dtype=torch.float64)
        Xb = torch.cat([X[:n_tr].double(), ones], dim=1)
        W = torch.linalg.lstsq(Xb, Y[:n_tr].double()).solution
        onev = torch.ones(n - n_tr, 1, dtype=torch.float64)
        pva = torch.cat([X[n_tr:].double(), onev], dim=1) @ W
        lstsq_va = _r2_perfeat(Y[n_tr:].double(), pva)

        print(f"{name:<20} {y_outlier:>10.1f} {r_raw.val_r2:>9.4f} {std_va:>9.4f} {lstsq_va:>9.4f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
