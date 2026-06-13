"""Effective rank of the linear FFN map (reduced-rank regression) + per-feature R2.

A metric-careful answer to "how many directions does the linear component of an FFN use?"
The naive SVD of the closed-form weight W* is scale-dominated by outlier features (rank-1)
and standardising the inputs is invalid (near-constant post-LayerNorm dims). Reduced-rank
regression (RRR) sidesteps both: it measures rank by *predictive content*. With the OLS
fit B (centered), let Yhat = Xc B and V = right singular vectors of Yhat (ordered by
explained variance). The optimal rank-k linear map predicts Yhat projected onto the top-k
of V; the effective rank is the smallest k whose held-out R2 reaches a fraction (90 / 95%)
of the full closed-form R2.

Also reports PER-FEATURE R2 (median / mean over the d output features) of the full
closed-form fit — a stricter companion to the variance-weighted global R2_lin, which a few
high-variance outlier features can flatter.

All 12 blocks x {gpt2, pythia-160m, llama-160m}. Cheap (no training). Run:
  venv/Scripts/python.exe -u run_rrr_rank.py
Writes plots/raw/rrr_rank.{json,csv} and plots/polyweave_rrr_rank.{pdf,png}.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from polyweave.distill import fit_closed_form_linear
from polyweave.distill.metrics import r2_score
from polyweave.experiments.gpt2_mlp_distill import (
    Config, load_model, token_batches, _blocks,
)
from run_residual_gain_clean import capture_all_blocks

MODELS = ["gpt2", "EleutherAI/pythia-160m", "JackFram/llama-160m"]
N_BLOCKS = 12
MAX_TOKENS = 15_000
STYLE = {"gpt2": ("C0", "GPT-2 (GELU)"),
         "EleutherAI/pythia-160m": ("C1", "Pythia-160m (GELU)"),
         "JackFram/llama-160m": ("C2", "llama-160m (SwiGLU)")}


def per_feature_r2(Y: torch.Tensor, P: torch.Tensor) -> tuple[float, float]:
    ss_res = ((Y - P) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0).clamp_min(1e-12)
    r2j = 1.0 - ss_res / ss_tot
    return float(r2j.median()), float(r2j.mean())


def rrr_effective_rank(X: torch.Tensor, Y: torch.Tensor, dev: str):
    """Return (r2_curve[d], r2_full, eff_rank@90, eff_rank@95) via reduced-rank regression."""
    n = X.shape[0]
    n_tr = n - max(1, round(n * 0.2))
    Xtr, Ytr = X[:n_tr].to(dev).double(), Y[:n_tr].to(dev).double()
    Xva, Yva = X[n_tr:].to(dev).double(), Y[n_tr:].to(dev).double()
    mx, my = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - mx, Ytr - my
    Xcv, Ycv = Xva - mx, Yva - my
    B = torch.linalg.lstsq(Xc, Yc).solution            # [d, d] OLS map
    Yhat_tr = Xc @ B
    _, _, Vh = torch.linalg.svd(Yhat_tr, full_matrices=False)
    V = Vh.T                                           # [d, d] output directions
    Yhat_va = Xcv @ B
    Z = Yhat_va @ V                                    # coords of val prediction
    G = Ycv @ V
    sstot = (Ycv ** 2).sum()
    ip = (G * Z).sum(0)                                # <Y, Yhat_k> per component
    nm = (Z * Z).sum(0)                                # ||Yhat_k||^2 per component
    ssres_k = sstot - 2 * torch.cumsum(ip, 0) + torch.cumsum(nm, 0)
    r2_k = (1.0 - ssres_k / sstot).cpu()               # R2 at rank 1..d
    r2_full = float(r2_k[-1])
    def first_k(frac):
        thr = frac * r2_full
        hits = (r2_k >= thr).nonzero()
        return int(hits[0].item()) + 1 if len(hits) else len(r2_k)
    return r2_k.tolist(), r2_full, first_k(0.90), first_k(0.95)


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
            d = X.shape[1]
            lin = nn.Linear(d, d)
            r = fit_closed_form_linear(lin, X, Y, val_frac=0.2, device=dev)
            n_tr = X.shape[0] - max(1, round(X.shape[0] * 0.2))
            with torch.no_grad():
                Pva = lin(X[n_tr:].to(dev))
            pf_med, pf_mean = per_feature_r2(Y[n_tr:].to(dev), Pva)
            curve, r2_full, k90, k95 = rrr_effective_rank(X, Y, dev)
            rows.append({"model": name, "block": i, "d": d, "r2_lin": r.val_r2,
                         "r2_perfeat_median": pf_med, "r2_perfeat_mean": pf_mean,
                         "rrr_r2_full": r2_full, "eff_rank_90": k90, "eff_rank_95": k95,
                         "rrr_curve": curve})
            print(f"  blk {i:2d}  r2_lin={r.val_r2:.3f}  perfeat(med)={pf_med:+.3f}"
                  f"  eff_rank@90={k90:3d}/{d}  @95={k95:3d}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    Path("plots/raw").mkdir(parents=True, exist_ok=True)
    Path("plots/raw/rrr_rank.json").write_text(json.dumps(rows, indent=2))
    csv = ["model,block,d,r2_lin,r2_perfeat_median,r2_perfeat_mean,eff_rank_90,eff_rank_95"]
    csv += [f"{r['model']},{r['block']},{r['d']},{r['r2_lin']:.5f},"
            f"{r['r2_perfeat_median']:.5f},{r['r2_perfeat_mean']:.5f},"
            f"{r['eff_rank_90']},{r['eff_rank_95']}" for r in rows]
    Path("plots/raw/rrr_rank.csv").write_text("\n".join(csv) + "\n")
    _plot(rows)


def _plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for name in MODELS:
        c, lab = STYLE.get(name, ("C3", name))
        rs = [r for r in rows if r["model"] == name]
        axL.plot([r["block"] for r in rs], [r["eff_rank_90"] for r in rs],
                 "o-", c=c, label=lab, ms=4)
        axR.scatter([r["r2_lin"] for r in rs], [r["eff_rank_90"] for r in rs],
                    c=c, label=lab, s=42, alpha=0.85, edgecolor="k", linewidth=0.4)
    axL.set_xlabel("block index")
    axL.set_ylabel("effective rank  (RRR, 90% of closed-form R²)")
    axL.set_title("Effective rank of the linear FFN map vs depth")
    axL.grid(alpha=0.3); axL.legend(fontsize=8)
    axR.set_xlabel("linear recoverability  R²_lin")
    axR.set_ylabel("effective rank @90%")
    axR.set_title("Is linear recoverability tied to low rank?")
    axR.grid(alpha=0.3); axR.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"plots/polyweave_rrr_rank.{ext}", dpi=150)
    print("saved plots/polyweave_rrr_rank.{pdf,png}")


if __name__ == "__main__":
    main()
