"""Decisive test: is the GPT-2 `dense` underfit cross-feature ill-conditioning?

Per-feature standardization barely helped (0.25 -> 0.29) yet closed-form linear hits
0.95. Per-feature scaling does NOT decorrelate inputs, so if the input covariance is
ill-conditioned (correlated dims + a 106x outlier), first-order Adam crawls along
low-curvature directions. PCA-whitening the input (train SVD) removes that conditioning
without adding capacity (it is an invertible linear reparam, so the closed-form ceiling
is unchanged). If Adam on whitened input now reaches ~0.95, ill-conditioning is proven
the cause -> the experiment's R2 numbers are an optimisation artifact, not FFN geometry.

Prints for GPT-2 early block:
  cond(Xtr)     condition number of the centered train input (eigval spread)
  raw_va        Adam, raw input, val per-feature R2 (current pipeline)
  whiten_va     Adam, PCA-whitened input, val per-feature R2 (well-conditioned)
  lstsq_va      closed-form linear ceiling
"""

from __future__ import annotations

import torch

from polyweave.distill.regression import fit_layer
from polyweave.experiments.gpt2_mlp_distill import (
    Config, load_model, token_batches, mlp_of, capture_block_io,
)

BLOCK = 1
MAX_TOKENS = 30_000
VAL_FRAC = 0.2


def _r2_perfeat(y, p):
    ss_res = (y - p).pow(2).sum()
    ss_tot = (y - y.mean(dim=0, keepdim=True)).pow(2).sum()
    return float(1 - ss_res / ss_tot)


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config(model_name="gpt2", block_indices=(BLOCK,), dataset="wikitext2",
                 seq_len=128, batch_size=4, max_tokens=MAX_TOKENS, device=dev)
    model, tok = load_model(cfg)
    batches = token_batches(cfg, tok, split="train")
    X, Y = capture_block_io(model, mlp_of(model, BLOCK), batches, cfg)
    X, Y = X.float().cpu(), Y.float().cpu()

    n = X.shape[0]
    n_tr = n - max(1, round(n * VAL_FRAC))

    # PCA-whiten using TRAIN statistics (invertible linear reparam of the input).
    xm = X[:n_tr].mean(0)
    Xc = X - xm
    U, S, Vh = torch.linalg.svd(Xc[:n_tr], full_matrices=False)
    cond = float(S[0] / S[S.shape[0] - 1].clamp_min(1e-12))
    scale = (S / (n_tr ** 0.5)).clamp_min(1e-6)
    Xw = (Xc @ Vh.T) / scale                       # whitened: ~identity covariance

    # raw fit (current pipeline)
    lin_raw = torch.nn.Linear(X.shape[1], Y.shape[1])
    r_raw = fit_layer(lin_raw, X, Y, steps=3000, lr=1e-3, batch_size=256,
                      val_frac=VAL_FRAC, device=dev, seed=42)

    # whitened fit (same optimiser budget)
    lin_w = torch.nn.Linear(X.shape[1], Y.shape[1])
    r_w = fit_layer(lin_w, Xw, Y, steps=3000, lr=1e-3, batch_size=256,
                    val_frac=VAL_FRAC, device=dev, seed=42)

    # closed-form ceiling
    ones = torch.ones(n_tr, 1, dtype=torch.float64)
    Xb = torch.cat([X[:n_tr].double(), ones], dim=1)
    W = torch.linalg.lstsq(Xb, Y[:n_tr].double()).solution
    onev = torch.ones(n - n_tr, 1, dtype=torch.float64)
    pva = torch.cat([X[n_tr:].double(), onev], dim=1) @ W
    lstsq_va = _r2_perfeat(Y[n_tr:].double(), pva)

    print(f"\ncond(Xtr)={cond:.3e}  raw_va={r_raw.val_r2:.4f}  "
          f"whiten_va={r_w.val_r2:.4f}  lstsq_va={lstsq_va:.4f}")


if __name__ == "__main__":
    main()
