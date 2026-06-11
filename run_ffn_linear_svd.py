"""Spectral / effective-rank analysis of the closed-form linear FFN maps.

For each block of each model we already solve the exact least-squares linear map
W* (the linear ceiling, §5.1). This asks what that map *looks like*: is a near-linear
FFN block a *low-rank* linear operator (consistent with FFN-as-key-value-memory, Geva
et al. 2021), or full-rank? We SVD each W* (d x d) and report:

  * stable rank      = ||W||_F^2 / ||W||_2^2  = sum(s^2) / s_max^2
  * participation r. = (sum s)^2 / sum(s^2)             (effective # of directions)
  * rank @90 / @99   = # singular values for 90% / 99% of the spectral energy (sum s^2)
  * s_max, and the normalized spectrum (for the spectrum plot)

and pairs each with the block's linear recoverability R2_lin, to test "are the
linearly-recoverable blocks the low-rank ones?".

Cheap (one closed-form solve + SVD per block; no training). Run:
  venv/Scripts/python.exe -u run_ffn_linear_svd.py
Writes plots/raw/ffn_linear_svd.{json,csv} and plots/polyweave_ffn_linear_svd.{pdf,png}.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from polyweave.distill import fit_closed_form_linear
from polyweave.experiments.gpt2_mlp_distill import (
    Config, load_model, token_batches, mlp_of, _blocks,
)
from run_residual_gain_clean import capture_all_blocks  # reuse the multi-block hook

MODELS = ["gpt2", "EleutherAI/pythia-160m", "JackFram/llama-160m"]
N_BLOCKS = 12
MAX_TOKENS = 15_000
STYLE = {"gpt2": ("C0", "GPT-2 (GELU)"),
         "EleutherAI/pythia-160m": ("C1", "Pythia-160m (GELU)"),
         "JackFram/llama-160m": ("C2", "llama-160m (SwiGLU)")}


def _spectral(W: torch.Tensor) -> dict:
    s = torch.linalg.svdvals(W.double())            # descending singular values
    energy = s ** 2
    total = energy.sum()
    cum = torch.cumsum(energy, 0) / total
    return {"s_max": float(s[0]),
            "stable_rank": float(total / energy[0]),       # ||W||_F^2 / ||W||_2^2
            "participation_ratio": float((s.sum() ** 2) / total),
            "rank90": int((cum < 0.90).sum().item()) + 1,
            "rank99": int((cum < 0.99).sum().item()) + 1,
            "spectrum": (s / s[0]).tolist()}


def spectral_stats(X: torch.Tensor, Y: torch.Tensor, lin: nn.Linear, dev: str) -> dict:
    """Effective rank of the closed-form linear FFN map.

    The RAW map W* is dominated by the transformer's outlier features (one input/output
    feature carries ~100x the variance), so its spectrum is ~rank-1 and uninformative
    about the operator's intrinsic complexity. We therefore also fit and SVD the
    *standardized* operator — the closed-form map between per-feature z-scored
    activations — which removes the outlier-scale artifact and reveals how many
    directions the FFN's linear component genuinely uses. We report both: the raw
    (outlier-dominated) stable rank and the standardized one (the meaningful figure).
    """
    d = X.shape[1]
    raw = _spectral(lin.weight.detach().cpu())
    # Standardized operator: z-score X and Y per feature on the train split, re-solve.
    n_tr = X.shape[0] - max(1, round(X.shape[0] * 0.2))
    mx, sx = X[:n_tr].mean(0), X[:n_tr].std(0).clamp_min(1e-6)
    my, sy = Y[:n_tr].mean(0), Y[:n_tr].std(0).clamp_min(1e-6)
    Xs, Ys = (X - mx) / sx, (Y - my) / sy
    lin_s = nn.Linear(d, d)
    fit_closed_form_linear(lin_s, Xs, Ys, val_frac=0.2, device=dev)
    std = _spectral(lin_s.weight.detach().cpu())
    return {"d": d, "s_max": raw["s_max"],
            "stable_rank_raw": raw["stable_rank"],
            "stable_rank": std["stable_rank"],
            "participation_ratio": std["participation_ratio"],
            "rank90": std["rank90"], "rank99": std["rank99"],
            "spectrum": std["spectrum"], "spectrum_raw": raw["spectrum"]}


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for name in MODELS:
        print(f"\n## {name}", flush=True)
        cfg = Config(model_name=name, dataset="wikitext2", seq_len=128, batch_size=4,
                     max_tokens=MAX_TOKENS, device=dev)
        model, tok = load_model(cfg)
        blocks = list(range(min(N_BLOCKS, len(_blocks(model)))))
        caps = capture_all_blocks(model, blocks, token_batches(cfg, tok, split="train"), cfg)
        for i in blocks:
            X, Y = caps[i]
            lin = nn.Linear(X.shape[1], X.shape[1])
            r2 = fit_closed_form_linear(lin, X, Y, val_frac=0.2, device=dev).val_r2
            st = spectral_stats(X, Y, lin, dev)
            rows.append({"model": name, "block": i, "r2_lin": r2, **st})
            print(f"  blk {i:2d}  r2_lin={r2:.3f}  stable_rank(std)={st['stable_rank']:6.1f}"
                  f"  rank90={st['rank90']:3d}  rank99={st['rank99']:3d}"
                  f"  raw_sr={st['stable_rank_raw']:5.2f}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    Path("plots/raw").mkdir(parents=True, exist_ok=True)
    Path("plots/raw/ffn_linear_svd.json").write_text(json.dumps(rows, indent=2))
    csv = ["model,block,r2_lin,d,stable_rank_std,stable_rank_raw,participation_ratio,rank90,rank99"]
    csv += [f"{r['model']},{r['block']},{r['r2_lin']:.5f},{r['d']},"
            f"{r['stable_rank']:.3f},{r['stable_rank_raw']:.3f},"
            f"{r['participation_ratio']:.3f},{r['rank90']},{r['rank99']}"
            for r in rows]
    Path("plots/raw/ffn_linear_svd.csv").write_text("\n".join(csv) + "\n")
    _plot(rows)


def _pearson(a, b):
    a, b = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


def _plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    # Left: stable rank vs depth, per model.
    for name in MODELS:
        c, lab = STYLE.get(name, ("C3", name))
        rs = [r for r in rows if r["model"] == name]
        axL.plot([r["block"] for r in rs], [r["stable_rank"] for r in rs],
                 "o-", c=c, label=lab, ms=4)
    axL.set_xlabel("block index")
    axL.set_ylabel("stable rank of standardized W*  (sum s^2 / s_max^2)")
    axL.set_title("Effective rank of the (standardized) linear FFN map vs depth")
    axL.grid(alpha=0.3); axL.legend(fontsize=8)
    # Right: is linear recoverability tied to low rank? r2_lin vs stable rank.
    allx, ally = [r["r2_lin"] for r in rows], [r["stable_rank"] for r in rows]
    for name in MODELS:
        c, lab = STYLE.get(name, ("C3", name))
        rs = [r for r in rows if r["model"] == name]
        axR.scatter([r["r2_lin"] for r in rs], [r["stable_rank"] for r in rs],
                    c=c, label=lab, s=42, alpha=0.85, edgecolor="k", linewidth=0.4)
    axR.set_xlabel("linear recoverability  R2_lin")
    axR.set_ylabel("stable rank of standardized W*")
    axR.set_title(f"Recoverability vs effective rank\n(Pearson r = {_pearson(allx, ally):+.2f})")
    axR.grid(alpha=0.3); axR.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"plots/polyweave_ffn_linear_svd.{ext}", dpi=150)
    print(f"\nALL pearson(r2_lin, stable_rank) = {_pearson(allx, ally):+.3f}")
    print("saved plots/polyweave_ffn_linear_svd.{pdf,png}")


if __name__ == "__main__":
    main()
