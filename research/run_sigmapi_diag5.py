"""Corrected core comparison with CONVERGED fits + the closed-form linear ceiling.

The experiment's 3000-step/lr-1e-3 budget left `dense` grossly underfit on GPT-2
(0.25 vs a 0.95 closed-form ceiling). Re-run both blocks of both models with a
generous budget and report, per (model, block):

  lin_ceiling : closed-form least-squares LINEAR val R2 (exact best linear map)
  dense_conv  : nn.Linear trained to convergence (30k steps)  -> should ~= ceiling
  poly_conv   : PolyLinear trained to convergence
  sigpi_conv  : SigmaPiLinear trained to convergence
  poly_gain   : poly_conv - lin_ceiling  (>0 = multiplication genuinely beats linear)

All val R2 use the experiment's r2_score (global-mean) for comparability with the
paper tables. Single seed (42) — this is a diagnostic, not the final multi-seed table.
"""

from __future__ import annotations

import torch

from polyweave.distill.metrics import r2_score
from polyweave.distill.regression import fit_layer
from polyweave.experiments.gpt2_mlp_distill import (
    Config, load_model, token_batches, mlp_of, capture_block_io, build_candidates,
)

MODELS = ["gpt2", "JackFram/llama-160m"]
BLOCKS = [1, 10]
MAX_TOKENS = 30_000
STEPS = 30_000
LR = 3e-3
LR_SIGPI = 1e-3   # gentler for the log/exp branch


def _lstsq_va(X, Y, n_tr):
    ones = torch.ones(n_tr, 1, dtype=torch.float64)
    W = torch.linalg.lstsq(torch.cat([X[:n_tr].double(), ones], 1), Y[:n_tr].double()).solution
    onev = torch.ones(X.shape[0] - n_tr, 1, dtype=torch.float64)
    pva = torch.cat([X[n_tr:].double(), onev], 1) @ W
    return r2_score(Y[n_tr:].double(), pva)


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hdr = (f"{'model':<20} {'blk':>3} {'lin_ceil':>9} {'dense_cv':>9} "
           f"{'poly_cv':>9} {'sigpi_cv':>9} {'poly_gain':>10}")
    print(hdr); print("-" * len(hdr))
    for name in MODELS:
        cfg = Config(model_name=name, dataset="wikitext2", seq_len=128, batch_size=4,
                     max_tokens=MAX_TOKENS, device=dev, poly_rank=16)
        model, tok = load_model(cfg)
        batches = token_batches(cfg, tok, split="train")
        for blk in BLOCKS:
            X, Y = capture_block_io(model, mlp_of(model, blk), batches, cfg)
            X, Y = X.float().cpu(), Y.float().cpu()
            n_tr = X.shape[0] - max(1, round(X.shape[0] * 0.2))

            ceil = _lstsq_va(X, Y, n_tr)
            d = fit_layer(torch.nn.Linear(768, 768), X, Y, steps=STEPS, lr=LR,
                          batch_size=256, val_frac=0.2, device=dev, seed=42).val_r2
            p = fit_layer(build_candidates(768, cfg)["poly"], X, Y, steps=STEPS, lr=LR,
                          batch_size=256, val_frac=0.2, device=dev, seed=42).val_r2
            s = fit_layer(build_candidates(768, cfg)["sigma-pi"], X, Y, steps=STEPS,
                          lr=LR_SIGPI, batch_size=256, val_frac=0.2, device=dev, seed=42).val_r2
            print(f"{name:<20} {blk:>3} {ceil:>9.4f} {d:>9.4f} {p:>9.4f} {s:>9.4f} {p - ceil:>10.4f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
