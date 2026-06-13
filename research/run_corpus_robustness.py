"""Cross-domain robustness of the linear-recoverability profile.

Because R2_lin is measured over the activation distribution a corpus induces, a fair
robustness check is a *different-domain* corpus (not merely a bigger same-domain one like
WikiText-103, which is still Wikipedia). We recompute the exact closed-form ceilings on a
given corpus and compare per block to the WikiText-2 depth sweep.

Closed-form only (no training) -> fast. Generic over corpus:
  venv/Scripts/python.exe -u run_corpus_robustness.py <corpus.txt> <tag>
e.g.  ... data/gutenberg_mobydick.txt mobydick
      ... data/gutenberg_math_puzzles.txt math_puzzles
Writes plots/raw/corpus_robustness_<tag>.{json,csv} (key r2_lin_corpus). Plot via
plot_corpus_robustness.py (overlays WikiText-2 + every corpus_robustness_*.json).
"""

from __future__ import annotations

import csv as csvmod
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

from polyweave.distill import fit_closed_form_linear
from polyweave.experiments.gpt2_mlp_distill import (
    Config, load_model, token_batches, _blocks,
)
from run_residual_gain_clean import capture_all_blocks

MODELS = ["gpt2", "EleutherAI/pythia-160m", "JackFram/llama-160m"]
N_BLOCKS = 12
MAX_TOKENS = 15_000


def load_wikitext_ceilings() -> dict:
    out = {}
    with open("plots/raw/depth_sweep_wikitext2.csv") as f:
        for row in csvmod.DictReader(f):
            out[(row["model"], int(row["block"]))] = float(row["r2_lin"])
    return out


def _pearson(a, b):
    a, b = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


def _spearman(a, b):
    ra = torch.tensor(a).argsort().argsort().float()
    rb = torch.tensor(b).argsort().argsort().float()
    return _pearson(ra.tolist(), rb.tolist())


def main() -> None:
    corpus = sys.argv[1] if len(sys.argv) > 1 else "data/gutenberg_mobydick.txt"
    tag = sys.argv[2] if len(sys.argv) > 2 else "mobydick"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"corpus={corpus}  tag={tag}", flush=True)
    rows = []
    for name in MODELS:
        print(f"\n## {name}", flush=True)
        cfg = Config(model_name=name, seq_len=128, batch_size=4, max_tokens=MAX_TOKENS,
                     device=dev, text_paths=(corpus,))
        model, tok = load_model(cfg)
        blocks = list(range(min(N_BLOCKS, len(_blocks(model)))))
        caps = capture_all_blocks(model, blocks, token_batches(cfg, tok), cfg)
        for i in blocks:
            X, Y = caps[i]
            lin = nn.Linear(X.shape[1], X.shape[1])
            r2 = fit_closed_form_linear(lin, X, Y, val_frac=0.2, device=dev).val_r2
            rows.append({"model": name, "block": i, "r2_lin_corpus": r2})
            print(f"  blk {i:2d}  r2_lin({tag})={r2:.3f}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    wiki = load_wikitext_ceilings()
    Path("plots/raw").mkdir(parents=True, exist_ok=True)
    Path(f"plots/raw/corpus_robustness_{tag}.json").write_text(json.dumps(rows, indent=2))
    out = ["model,block,r2_lin_wikitext,r2_lin_corpus,abs_diff"]
    for r in rows:
        w = wiki.get((r["model"], r["block"]))
        out.append(f"{r['model']},{r['block']},{w:.5f},{r['r2_lin_corpus']:.5f},"
                   f"{abs(w - r['r2_lin_corpus']):.5f}")
    Path(f"plots/raw/corpus_robustness_{tag}.csv").write_text("\n".join(out) + "\n")

    print(f"\n{'=' * 70}\nPROFILE STABILITY (WikiText-2 vs {tag}), per model:")
    for name in MODELS:
        w = [wiki[(name, i)] for i in range(N_BLOCKS)]
        g = [r["r2_lin_corpus"] for r in rows if r["model"] == name]
        maxd = max(abs(a - b) for a, b in zip(w, g))
        meand = sum(abs(a - b) for a, b in zip(w, g)) / len(w)
        print(f"  {name:<24} pearson={_pearson(w, g):+.3f}  spearman={_spearman(w, g):+.3f}"
              f"  max|d|={maxd:.3f}  mean|d|={meand:.3f}")
    print(f"saved plots/raw/corpus_robustness_{tag}.{{json,csv}}")


if __name__ == "__main__":
    main()
