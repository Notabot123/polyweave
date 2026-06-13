"""Full depth sweep on a DIFFERENT-domain corpus (Project Gutenberg, Moby-Dick).

Mirrors run_depth_sweep.py (closed-form linear ceiling + seeded-poly, each with zero-shot
swap dPPL, every block x 3 models) but on literary prose instead of WikiText-2 — to check
that the poly-gain and dPPL findings (not just the linear ceiling, which run_corpus_
robustness.py already covers) are robust to corpus domain.

Crucial difference from pointing cfg.text_paths at the book: we split the corpus's OWN
token stream into train (capture + fit) and a held-out test tail (perplexity), so dPPL is
measured on unseen text rather than the captured text.

Run:  venv/Scripts/python.exe -u run_depth_sweep_gutenberg.py
Writes plots/raw/depth_sweep_gutenberg.{json,csv}.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from polyweave.layers import PolyLinear
from polyweave.distill.regression import fit_closed_form_linear, fit_layer
from polyweave.experiments.gpt2_mlp_distill import (
    Config, load_model, mlp_of, _blocks, _perplexity, block_swap_perplexity,
)
from run_depth_sweep import capture_all_blocks, MAX_TOKENS, POLY_STEPS

MODELS = ["gpt2", "EleutherAI/pythia-160m", "JackFram/llama-160m"]
N_BLOCKS = 12
CORPUS = "data/gutenberg_mobydick.txt"
SEQ_LEN = 128
BATCH = 4
TRAIN_FRAC = 0.85
RAW = Path("plots/raw")
OUT = RAW / "depth_sweep_gutenberg.json"


def split_batches(text, tok):
    """Tokenise once, split the stream into train/test, window each into [B, seq] batches."""
    ids = tok(text, return_tensors="pt").input_ids[0]
    n_split = int(len(ids) * TRAIN_FRAC)
    def windows(stream):
        nw = max(1, len(stream) // SEQ_LEN)
        stream = stream[: nw * SEQ_LEN].reshape(nw, SEQ_LEN)
        return [stream[i:i + BATCH] for i in range(0, nw, BATCH)]
    return windows(ids[:n_split]), windows(ids[n_split:])


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    text = Path(CORPUS).read_text(encoding="utf-8", errors="ignore")
    results = {}
    for name in MODELS:
        print(f"\n########## {name} ##########", flush=True)
        cfg = Config(model_name=name, seq_len=SEQ_LEN, batch_size=BATCH,
                     max_tokens=MAX_TOKENS, device=dev, poly_rank=16,
                     eval_perplexity=True, ppl_max_batches=30, heal_steps=0)
        model, tok = load_model(cfg)
        train_batches, eval_batches = split_batches(text, tok)
        n_blocks = min(N_BLOCKS, len(_blocks(model)))
        blocks = list(range(n_blocks))
        base_ppl = _perplexity(model, eval_batches, cfg)
        caps = capture_all_blocks(model, blocks, train_batches, cfg)
        d = caps[0][0].shape[1]
        print(f"  base_ppl={base_ppl:.3f}  d={d}  "
              f"train_batches={len(train_batches)} test_batches={len(eval_batches)}", flush=True)

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
                fit_closed_form_linear(poly.linear, X, Y, val_frac=0.2, device=dev)
                r_poly = fit_layer(poly, X, Y, steps=POLY_STEPS, lr=1e-3, batch_size=256,
                                   val_frac=0.2, device=dev, seed=42)
                ppl_poly = block_swap_perplexity(model, poly, i, eval_batches, cfg,
                                                 heal_batches=None, ppl_base=base_ppl)["ppl_swap"]
                row["r2_poly"] = r_poly.val_r2
                row["dppl_poly"] = ppl_poly - base_ppl
            except Exception as e:
                row["r2_poly"] = None
                row["dppl_poly"] = None
                print(f"  [block {i}] poly failed: {e}", flush=True)
            print(f"  block {i:2d}  r2_lin={row['r2_lin']:.3f}  dppl_lin={row['dppl_lin']:+.2f}"
                  f"  r2_poly={row.get('r2_poly')}", flush=True)
            rows.append(row)

        results[name] = {"base_ppl": base_ppl, "d_model": d, "blocks": rows}
        OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
        _write_csv(results)
        print(f"  saved -> {OUT}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _write_csv(results) -> None:
    def fmt(v):
        return "" if v is None else f"{v:.5f}"
    lines = ["model,base_ppl,d_model,block,r2_lin,dppl_lin,r2_poly,dppl_poly"]
    for name, data in results.items():
        for r in data["blocks"]:
            lines.append(
                f"{name},{data['base_ppl']:.4f},{data['d_model']},{r['block']},"
                f"{r['r2_lin']:.5f},{r['dppl_lin']:.5f},"
                f"{fmt(r.get('r2_poly'))},{fmt(r.get('dppl_poly'))}")
    (RAW / "depth_sweep_gutenberg.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
