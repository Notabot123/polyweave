"""2-layer poly vs 2-layer dense: does *multiplicative* depth beat *additive* depth?

`dense (2x)` is the additive-depth control already in the paper (Linear -> GELU ->
Linear, ~2d^2). Its natural multiplicative analog is `poly (2x)` = PolyLinear -> GELU
-> PolyLinear at the same shape. Comparing the two at matched budget separates the
benefit of *depth* (a hidden layer) from the benefit of an explicit *multiplicative*
branch within each layer. Most informative on the genuinely-nonlinear SwiGLU target
(llama-160m); GPT-2 (near-linear early block) is the control.

Protocol matches run_{gpt2,llama}_multiseed_v2.py so numbers slot beside §5.4:
30k tokens/block, closed-form linear ceiling, trained candidates 8k AdamW steps,
seeds 42/43/44. Activation fit only (R2/cosine) — the headline question here is
representational, and dense(2x)/poly already have downstream PPL in §5.5.

Run:  venv/Scripts/python.exe -u run_poly2x.py
Writes plots/raw/poly2x_wikitext2.{json,csv}.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch
import torch.nn as nn

from polyweave.layers import PolyLinear
from polyweave.distill import fit_closed_form_linear, fit_layer
from polyweave.experiments.gpt2_mlp_distill import (
    Config, load_model, token_batches, mlp_of, capture_block_io,
)
from polyweave.utils import count_params, set_seed

MODELS = ["gpt2", "JackFram/llama-160m"]
BLOCKS = [(1, "early"), (10, "deep")]
SEEDS = (42, 43, 44)
MAX_TOKENS = 30_000
STEPS = 8000
RANK = 16
RAW = Path("plots/raw")


class Poly2x(nn.Module):
    """Multiplicative analog of `dense (2x)`: PolyLinear -> GELU -> PolyLinear."""

    def __init__(self, d: int, rank: int = RANK) -> None:
        super().__init__()
        self.fc1 = PolyLinear(d, d, rank=rank)
        self.act = nn.GELU()
        self.fc2 = PolyLinear(d, d, rank=rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def build(d: int) -> dict:
    return {
        "poly": PolyLinear(d, d, rank=RANK),
        "dense (2x)": nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)),
        "poly (2x)": Poly2x(d, rank=RANK),
    }


def _ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"mean": None, "std": None}
    return {"mean": statistics.fmean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for name in MODELS:
        print(f"\n########## {name} ##########", flush=True)
        cfg = Config(model_name=name, dataset="wikitext2", seq_len=128, batch_size=4,
                     max_tokens=MAX_TOKENS, device=dev, poly_rank=RANK,
                     steps=STEPS, lr=1e-3, fit_batch_size=256)
        model, tok = load_model(cfg)
        batches = token_batches(cfg, tok)
        for idx, depth in BLOCKS:
            mlp = mlp_of(model, idx)
            mlp_params = count_params(mlp)
            X, Y = capture_block_io(model, mlp, batches, cfg)

            # Exact linear ceiling (deterministic).
            lin = nn.Linear(X.shape[1], X.shape[1])
            r_lin = fit_closed_form_linear(lin, X, Y, val_frac=0.2, device=dev)
            print(f"  [{depth} blk{idx}] linear(ceil) R2={r_lin.val_r2:.4f} "
                  f"cos={r_lin.val_cosine:.4f}", flush=True)
            rows.append({"model": name, "block": idx, "depth": depth, "layer": "linear",
                         "num_params": r_lin.num_params,
                         "compression": mlp_params / r_lin.num_params,
                         "r2": {"mean": r_lin.val_r2, "std": 0.0},
                         "cosine": {"mean": r_lin.val_cosine, "std": 0.0}})

            # Trained candidates, per seed.
            per = {k: {"r2": [], "cos": [], "params": 0} for k in build(X.shape[1])}
            for seed in SEEDS:
                set_seed(seed)
                for k, layer in build(X.shape[1]).items():
                    res = fit_layer(layer, X, Y, steps=STEPS, lr=1e-3, batch_size=256,
                                    val_frac=0.2, eval_every=0, device=dev, seed=seed)
                    per[k]["r2"].append(res.val_r2)
                    per[k]["cos"].append(res.val_cosine)
                    per[k]["params"] = res.num_params
            for k, agg in per.items():
                r2, cos = _ms(agg["r2"]), _ms(agg["cos"])
                print(f"  [{depth} blk{idx}] {k:<11} params={agg['params']:>9,} "
                      f"compress x{mlp_params / agg['params']:4.1f}  "
                      f"R2={r2['mean']:.4f}+/-{r2['std']:.4f}  "
                      f"cos={cos['mean']:.4f}", flush=True)
                rows.append({"model": name, "block": idx, "depth": depth, "layer": k,
                             "num_params": agg["params"],
                             "compression": mlp_params / agg["params"],
                             "r2": r2, "cosine": cos})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "poly2x_wikitext2.json").write_text(json.dumps(rows, indent=2))
    csv = ["model,block,depth,layer,num_params,compression,r2_mean,r2_std,cos_mean,cos_std"]
    for r in rows:
        csv.append(f"{r['model']},{r['block']},{r['depth']},{r['layer']},{r['num_params']},"
                   f"{r['compression']:.3f},{r['r2']['mean']:.5f},{r['r2']['std']:.5f},"
                   f"{r['cosine']['mean']:.5f},{r['cosine']['std']:.5f}")
    (RAW / "poly2x_wikitext2.csv").write_text("\n".join(csv) + "\n")
    print(f"\nsaved {RAW / 'poly2x_wikitext2.csv'}")


if __name__ == "__main__":
    main()
