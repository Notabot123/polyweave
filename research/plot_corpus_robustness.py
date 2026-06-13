"""Overlay the per-block linear-recoverability profile across corpus domains.

Reads WikiText-2 ceilings (plots/raw/depth_sweep_wikitext2.csv) and every
plots/raw/corpus_robustness_<tag>.json, and overlays R2_lin vs block per model, one
linestyle per corpus. Re-plot only (no recompute). Writes
plots/polyweave_corpus_robustness.{pdf,png}.
"""

from __future__ import annotations

import csv as csvmod
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["gpt2", "EleutherAI/pythia-160m", "JackFram/llama-160m"]
N_BLOCKS = 12
COLOR = {"gpt2": "C0", "EleutherAI/pythia-160m": "C1", "JackFram/llama-160m": "C2"}
LABEL = {"gpt2": "GPT-2", "EleutherAI/pythia-160m": "Pythia-160m", "JackFram/llama-160m": "llama-160m"}
# corpus tag -> (linestyle, marker, display name)
CORPUS_STYLE = {
    "wikitext2": ("-", "o", "WikiText-2"),
    "mobydick": ("--", "s", "Moby-Dick"),
    "math_puzzles": (":", "^", "Math puzzles"),
}


def load_wikitext():
    out = {}
    with open("plots/raw/depth_sweep_wikitext2.csv") as f:
        for row in csvmod.DictReader(f):
            out[(row["model"], int(row["block"]))] = float(row["r2_lin"])
    return out


def main() -> None:
    corpora = {"wikitext2": load_wikitext()}
    for path in sorted(glob.glob("plots/raw/corpus_robustness_*.json")):
        tag = Path(path).stem.replace("corpus_robustness_", "")
        rows = json.loads(Path(path).read_text())
        corpora[tag] = {(r["model"], r["block"]): r["r2_lin_corpus"] for r in rows}

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for name in MODELS:
        for tag, data in corpora.items():
            ls, mk, _ = CORPUS_STYLE.get(tag, (":", "x", tag))
            ys = [data.get((name, i)) for i in range(N_BLOCKS)]
            if any(v is None for v in ys):
                continue
            ax.plot(range(N_BLOCKS), ys, ls, marker=mk, c=COLOR[name], ms=4, alpha=0.8)
    # Two legends: color = model, linestyle = corpus.
    from matplotlib.lines import Line2D
    model_handles = [Line2D([0], [0], color=COLOR[m], lw=2, label=LABEL[m]) for m in MODELS]
    corpus_handles = [Line2D([0], [0], color="0.3", ls=ls, marker=mk, label=nm)
                      for ls, mk, nm in CORPUS_STYLE.values()]
    leg1 = ax.legend(handles=model_handles, fontsize=8, loc="lower left", title="model")
    ax.add_artist(leg1)
    ax.legend(handles=corpus_handles, fontsize=8, loc="lower right", title="corpus")
    ax.set_xlabel("block index"); ax.set_ylabel("linear recoverability  R2_lin")
    ax.set_title("Per-block linear recoverability is stable across corpus domain")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"plots/polyweave_corpus_robustness.{ext}", dpi=150)
    print("saved plots/polyweave_corpus_robustness.{pdf,png}")


if __name__ == "__main__":
    main()
