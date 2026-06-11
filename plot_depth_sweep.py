"""Plot the depth sweep from saved results — re-run this alone to tweak formatting
WITHOUT recomputing the sweep.

    venv/Scripts/python.exe plot_depth_sweep.py            # uses the default JSON
    venv/Scripts/python.exe plot_depth_sweep.py path.json  # or a specific file

Reads plots/raw/depth_sweep_wikitext2.json (written by run_depth_sweep.py) and writes
plots/polyweave_depth_sweep_wikitext2.{pdf,png}. Two panels:
  left  — linear-ceiling R² vs block depth (solid), poly R² overlay (dashed)
  right — zero-shot ΔPPL when the layer is swapped in, vs depth (solid linear, dashed poly)
one colour per model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_JSON = Path("plots/raw/depth_sweep_wikitext2.json")
OUT_STEM = "plots/polyweave_depth_sweep_wikitext2"

# Per-model colour + display label. Add new models here.
STYLE = {
    "gpt2": ("C0", "GPT-2 (GELU)"),
    "EleutherAI/pythia-160m": ("C1", "Pythia-160m (GELU)"),
    "JackFram/llama-160m": ("C2", "llama-160m (SwiGLU)"),
}


def plot_from_json(json_path: Path = DEFAULT_JSON) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = json.loads(Path(json_path).read_text(encoding="utf-8"))
    fig, (axr, axp) = plt.subplots(1, 2, figsize=(12, 4.5))

    for name, data in results.items():
        color, label = STYLE.get(name, ("C3", name))
        bl = [r["block"] for r in data["blocks"]]
        r2_lin = [r["r2_lin"] for r in data["blocks"]]
        dppl_lin = [r["dppl_lin"] for r in data["blocks"]]
        r2_poly = [r.get("r2_poly") for r in data["blocks"]]
        dppl_poly = [r.get("dppl_poly") for r in data["blocks"]]
        nan = float("nan")

        axr.plot(bl, r2_lin, "-o", color=color, label=label)
        if any(v is not None for v in r2_poly):
            axr.plot(bl, [v if v is not None else nan for v in r2_poly], "--s",
                     color=color, alpha=0.55)
        axp.plot(bl, dppl_lin, "-o", color=color, label=label)
        if any(v is not None for v in dppl_poly):
            axp.plot(bl, [v if v is not None else nan for v in dppl_poly], "--s",
                     color=color, alpha=0.55)

    axr.set_xlabel("block index (depth)")
    axr.set_ylabel("linear-ceiling R²  (closed form)")
    axr.set_title("How linear is each FFN, by depth")
    axr.grid(alpha=0.3)
    axr.legend(fontsize=8, title="solid = linear, dashed = poly", title_fontsize=8)

    axp.set_xlabel("block index (depth)")
    axp.set_ylabel("zero-shot ΔPPL when swapped in")
    axp.set_title("Perplexity penalty of single-layer replacement")
    axp.grid(alpha=0.3)
    axp.set_yscale("symlog")
    axp.legend(fontsize=8)

    fig.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_STEM}.{ext}", dpi=150)
    print(f"saved {OUT_STEM}.{{pdf,png}}")


def plot_delta_from_json(json_path: Path = DEFAULT_JSON) -> None:
    """Secondary figure: the poly-vs-linear *delta* per block, same x-axis (depth).

    Left  — R² gain (R²_poly − R²_linear). NOTE this is noisy where the whole-layer
            poly fit drifted below its linear seed on ill-conditioned blocks.
    Right — perplexity reduction (ΔPPL_linear − ΔPPL_poly): how much the bilinear term
            shrinks the single-layer swap penalty. The clean, striking panel.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = json.loads(Path(json_path).read_text(encoding="utf-8"))
    fig, (axr, axp) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, data in results.items():
        color, label = STYLE.get(name, ("C3", name))
        bl, dr2, dppl = [], [], []
        for r in data["blocks"]:
            if r.get("r2_poly") is None:
                continue
            bl.append(r["block"])
            dr2.append(r["r2_poly"] - r["r2_lin"])
            dppl.append(r["dppl_lin"] - (r["dppl_poly"] if r.get("dppl_poly") is not None else r["dppl_lin"]))
        axr.plot(bl, dr2, "-o", color=color, label=label)
        axp.plot(bl, dppl, "-o", color=color, label=label)

    for ax in (axr, axp):
        ax.axhline(0, color="k", lw=0.8, alpha=0.5)
        ax.set_xlabel("block index (depth)")
        ax.grid(alpha=0.3)
    axr.set_ylabel("R² gain  (poly − linear)")
    axr.set_title("Activation-fit gain from the bilinear term")
    axr.legend(fontsize=8)
    axp.set_ylabel("ΔPPL reduction  (linear − poly)")
    axp.set_title("Perplexity-penalty reduction from the bilinear term")
    fig.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_STEM}_delta.{ext}", dpi=150)
    print(f"saved {OUT_STEM}_delta.{{pdf,png}}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON
    plot_from_json(path)
    plot_delta_from_json(path)
