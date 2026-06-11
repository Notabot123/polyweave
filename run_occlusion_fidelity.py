"""Occlusion-fidelity: does a distilled layer reproduce the ORIGINAL FFN's input-
dependency *structure*, not just its outputs? A mechanistic complement to R²/ΔPPL.

For each model + block we occlude input-feature groups (Zeiler-Fergus style) and read:
  conj_*   : conjunction (AND-signature) index in [0,1] — ~0 additive, ~1 multiplicative
             — for {original FFN, fitted linear, fitted poly}, mean over tokens.
  corr_*   : Pearson corr between the original FFN's per-feature occlusion-sensitivity
             profile and each candidate's — "does it depend on the same inputs?"

Expected (and the point of the figure): on GPT-2's near-linear GELU FFN everything is
additive (low conjunction). On llama's SwiGLU FFN the original is genuinely conjunctive,
a linear map is additive by construction (conj ~0, cannot reproduce it), and poly
*partially* recovers the conjunctive structure — mirroring its R²/ΔPPL gain there.

Run:  venv/Scripts/python.exe -u run_occlusion_fidelity.py
"""

from __future__ import annotations

import torch

from polyweave.interpretability.occlusion import conjunction_index, occlusion_sensitivity_1d
from polyweave.distill.regression import fit_closed_form_linear, fit_layer
from polyweave.layers import PolyLinear
from polyweave.experiments.gpt2_mlp_distill import (
    Config, load_model, token_batches, mlp_of, capture_block_io,
)

CASES = [("gpt2", [1, 10]), ("JackFram/llama-160m", [1, 10])]
MAX_TOKENS = 30_000
OCC_ROWS = 256
WINDOW = 16          # occlude 16-feature windows -> 48 positions over d=768


def _resp(layer):
    return lambda b: layer(b).mean(dim=1)   # mean output per token (scalar response)


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a @ b) / denom)


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = 768
    ga, gb = list(range(d // 2)), list(range(d // 2, d))
    hdr = (f"{'model':<20} {'blk':>3} | {'conj_orig':>9} {'conj_lin':>9} {'conj_poly':>9} "
           f"| {'corr_lin':>8} {'corr_poly':>9}")
    print(hdr); print("-" * len(hdr))
    for name, blocks in CASES:
        cfg = Config(model_name=name, dataset="wikitext2", seq_len=128, batch_size=4,
                     max_tokens=MAX_TOKENS, device=dev, poly_rank=16)
        model, tok = load_model(cfg)
        batches = token_batches(cfg, tok, split="train")
        for blk in blocks:
            original = mlp_of(model, blk)
            X, Y = capture_block_io(model, original, batches, cfg)
            X, Y = X.float(), Y.float()
            rows = X[:OCC_ROWS].to(dev)

            linear = torch.nn.Linear(d, d).to(dev)
            fit_closed_form_linear(linear, X, Y, val_frac=0.2, device=dev)
            poly = PolyLinear(d, d, rank=16).to(dev)
            fit_layer(poly, X, Y, steps=8000, lr=1e-3, batch_size=256,
                      val_frac=0.2, device=dev, seed=42)

            layers = {"orig": original, "lin": linear, "poly": poly}
            conj, sens = {}, {}
            for k, L in layers.items():
                L.eval()
                conj[k] = conjunction_index(_resp(L), rows, ga, gb).mean().item()
                sens[k] = occlusion_sensitivity_1d(_resp(L), rows, window=WINDOW,
                                                   stride=WINDOW).mean(dim=0)  # [P]
            print(f"{name:<20} {blk:>3} | {conj['orig']:>9.3f} {conj['lin']:>9.3f} "
                  f"{conj['poly']:>9.3f} | {_corr(sens['orig'], sens['lin']):>8.3f} "
                  f"{_corr(sens['orig'], sens['poly']):>9.3f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
