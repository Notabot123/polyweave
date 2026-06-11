"""Depth sweep: linear-ceiling (and poly) vs block index, per model.

Motivated by the finding that FFN linearity varies across models AND depth and is not
predicted by the activation family (GPT-2 GELU early ~0.95 linear; Pythia GELU early
~0.51). This sweeps EVERY block of each model and reports, per block:
  * the EXACT closed-form linear ceiling (R^2) and its zero-shot swap ΔPPL, and
  * a trained poly layer (R^2 + zero-shot swap ΔPPL),
then plots both vs depth. The picture *shows* the per-model linearity profile instead
of claiming a single trend.

Unattended-friendly: captures all blocks in one forward pass; writes JSON after each
model; wraps poly per-block in try/except so one failure can't sink the linear sweep.

Run:  venv/Scripts/python.exe -u run_depth_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from polyweave.layers import PolyLinear
from polyweave.distill.regression import fit_closed_form_linear, fit_layer
from polyweave.experiments.gpt2_mlp_distill import (
    Config, load_model, token_batches, mlp_of, _blocks, _perplexity,
    block_swap_perplexity,
)

MODELS = ["gpt2", "EleutherAI/pythia-160m", "JackFram/llama-160m"]
N_BLOCKS = 12
MAX_TOKENS = 15_000
POLY_STEPS = 12_000
RAW = Path("plots/raw")
OUT = RAW / "depth_sweep_wikitext2.json"


def capture_all_blocks(model, blocks, batches, cfg):
    """One forward pass over the corpus, capturing (X, Y) for every block at once."""
    store = {i: {"x": [], "y": [], "n": 0} for i in blocks}

    def mk_hook(i):
        def hook(_m, inp, out):
            s = store[i]
            if s["n"] >= MAX_TOKENS:
                return
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            y = (out[0] if isinstance(out, (tuple, list)) else out).detach()
            y = y.reshape(-1, y.shape[-1])
            room = MAX_TOKENS - s["n"]
            s["x"].append(x[:room].float().cpu())
            s["y"].append(y[:room].float().cpu())
            s["n"] += x[:room].shape[0]
        return hook

    handles = [mlp_of(model, i).register_forward_hook(mk_hook(i)) for i in blocks]
    try:
        with torch.no_grad():
            for ids in batches:
                if all(store[i]["n"] >= MAX_TOKENS for i in blocks):
                    break
                model(ids.to(cfg.device))
    finally:
        for h in handles:
            h.remove()
    return {i: (torch.cat(store[i]["x"]), torch.cat(store[i]["y"])) for i in blocks}


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for name in MODELS:
        print(f"\n########## {name} ##########", flush=True)
        cfg = Config(model_name=name, dataset="wikitext2", seq_len=128, batch_size=4,
                     max_tokens=MAX_TOKENS, device=dev, poly_rank=16,
                     eval_perplexity=True, ppl_split="test", ppl_max_batches=30,
                     heal_steps=0)
        model, tok = load_model(cfg)
        n_blocks = min(N_BLOCKS, len(_blocks(model)))
        blocks = list(range(n_blocks))
        eval_batches = token_batches(cfg, tok, split="test")
        base_ppl = _perplexity(model, eval_batches, cfg)
        train_batches = token_batches(cfg, tok, split="train")
        caps = capture_all_blocks(model, blocks, train_batches, cfg)
        d = caps[0][0].shape[1]

        rows = []
        for i in blocks:
            X, Y = caps[i]
            lin = nn.Linear(d, d)
            r_lin = fit_closed_form_linear(lin, X, Y, val_frac=0.2, device=dev)
            ppl_lin = block_swap_perplexity(model, lin, i, eval_batches, cfg,
                                            heal_batches=None, ppl_base=base_ppl)["ppl_swap"]
            row = {"block": i, "r2_lin": r_lin.val_r2, "dppl_lin": ppl_lin - base_ppl}
            try:
                poly = PolyLinear(d, d, rank=16)
                # Seed poly's linear branch from the closed-form solution so it STARTS
                # at the linear ceiling; fine-tuning can then only add the (small)
                # multiplicative gain — no divergence/underfit confound in the overlay.
                fit_closed_form_linear(poly.linear, X, Y, val_frac=0.2, device=dev)
                r_poly = fit_layer(poly, X, Y, steps=POLY_STEPS, lr=1e-3, batch_size=256,
                                   val_frac=0.2, device=dev, seed=42)
                ppl_poly = block_swap_perplexity(model, poly, i, eval_batches, cfg,
                                                 heal_batches=None, ppl_base=base_ppl)["ppl_swap"]
                row["r2_poly"] = r_poly.val_r2
                row["dppl_poly"] = ppl_poly - base_ppl
            except Exception as e:  # never let one poly fit sink the sweep
                row["r2_poly"] = None
                row["dppl_poly"] = None
                print(f"  [block {i}] poly failed: {e}", flush=True)
            print(f"  block {i:2d}  r2_lin={row['r2_lin']:.3f}  dppl_lin={row['dppl_lin']:+.2f}"
                  f"  r2_poly={row.get('r2_poly')}", flush=True)
            rows.append(row)

        results[name] = {"base_ppl": base_ppl, "d_model": d, "blocks": rows}
        OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")  # incremental save
        _write_csv(results)
        print(f"  saved -> {OUT}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Plotting lives in plot_depth_sweep.py and reads the JSON, so the plot can be
    # reformatted by re-running that script alone — no need to recompute the sweep.
    from plot_depth_sweep import plot_from_json
    plot_from_json(OUT)


def _write_csv(results) -> None:
    """Flat tabular dump (one row per model×block) for easy inspection / re-plotting."""
    def fmt(v):
        return "" if v is None else f"{v:.5f}"
    lines = ["model,base_ppl,d_model,block,r2_lin,dppl_lin,r2_poly,dppl_poly"]
    for name, data in results.items():
        for r in data["blocks"]:
            lines.append(
                f"{name},{data['base_ppl']:.4f},{data['d_model']},{r['block']},"
                f"{r['r2_lin']:.5f},{r['dppl_lin']:.5f},"
                f"{fmt(r.get('r2_poly'))},{fmt(r.get('dppl_poly'))}"
            )
    (RAW / "depth_sweep_wikitext2.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
