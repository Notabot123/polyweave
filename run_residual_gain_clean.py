"""Clean test of "poly gain scales with residual nonlinearity beyond the linear
ceiling" — with a STABLE poly so the gain is meaningful.

Per block: solve the exact closed-form linear map (the ceiling), FREEZE it as poly's
linear branch, and train ONLY the low-rank quad branch, keeping the best held-out R²
(early stopping). Gain = best_poly_R2 - R2_linear is then >= 0 by construction and
isolates what a low-rank bilinear adds *on top of the optimal linear map*. Then:
    x = 1 - R2_linear   (residual nonlinearity)
    y = gain            (recovered by poly)
and we correlate across all blocks / models.

Reads nothing; recomputes from the models (needs GPU). Writes
plots/raw/residual_gain_clean.{json,csv} and plots/polyweave_residual_gain_clean.{pdf,png}.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from polyweave.layers import PolyLinear
from polyweave.distill.regression import fit_closed_form_linear
from polyweave.distill.metrics import r2_score
from polyweave.experiments.gpt2_mlp_distill import (
    Config, load_model, token_batches, mlp_of, _blocks,
)

MODELS = ["gpt2", "EleutherAI/pythia-160m", "JackFram/llama-160m"]
N_BLOCKS = 12
MAX_TOKENS = 15_000
STYLE = {"gpt2": ("C0", "GPT-2 (GELU)"),
         "EleutherAI/pythia-160m": ("C1", "Pythia-160m (GELU)"),
         "JackFram/llama-160m": ("C2", "llama-160m (SwiGLU)")}


def capture_all_blocks(model, blocks, batches, cfg):
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
            s["x"].append(x[:room].float().cpu()); s["y"].append(y[:room].float().cpu())
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


def quad_only_gain(X, Y, r2_lin, dev, steps=4000, lr=1e-3, patience=8, eval_every=200):
    """Freeze poly's linear branch at the closed-form optimum; train only the quad
    branch; return the BEST held-out R2 reached (>= r2_lin)."""
    d = X.shape[1]
    n_tr = X.shape[0] - max(1, round(X.shape[0] * 0.2))
    Xtr, Ytr = X[:n_tr].to(dev), Y[:n_tr].to(dev)
    Xva, Yva = X[n_tr:].to(dev), Y[n_tr:].to(dev)
    poly = PolyLinear(d, d, rank=16).to(dev)
    fit_closed_form_linear(poly.linear, X, Y, val_frac=0.2, device=dev)  # seed linear
    for p in poly.linear.parameters():
        p.requires_grad_(False)                                          # freeze it
    quad_params = [p for n, p in poly.named_parameters() if not n.startswith("linear.")]
    opt = torch.optim.Adam(quad_params, lr=lr)
    g = torch.Generator().manual_seed(0)
    best = r2_lin
    bad = 0
    poly.eval()
    for step in range(1, steps + 1):
        poly.train()
        idx = torch.randint(0, n_tr, (256,), generator=g)
        opt.zero_grad()
        F.mse_loss(poly(Xtr[idx]), Ytr[idx]).backward()
        opt.step()
        if step % eval_every == 0:
            poly.eval()
            with torch.no_grad():
                v = r2_score(Yva, poly(Xva))
            if v > best + 1e-4:
                best, bad = v, 0
            else:
                bad += 1
                if bad >= patience:
                    break
    return max(best, r2_lin)


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for name in MODELS:
        print(f"\n## {name}", flush=True)
        cfg = Config(model_name=name, dataset="wikitext2", seq_len=128, batch_size=4,
                     max_tokens=MAX_TOKENS, device=dev, poly_rank=16)
        model, tok = load_model(cfg)
        blocks = list(range(min(N_BLOCKS, len(_blocks(model)))))
        caps = capture_all_blocks(model, blocks, token_batches(cfg, tok, split="train"), cfg)
        for i in blocks:
            X, Y = caps[i]
            lin = nn.Linear(X.shape[1], X.shape[1])
            r2_lin = fit_closed_form_linear(lin, X, Y, val_frac=0.2, device=dev).val_r2
            r2_poly = quad_only_gain(X, Y, r2_lin, dev)
            rows.append({"model": name, "block": i, "r2_lin": r2_lin,
                         "r2_poly": r2_poly, "residual": 1 - r2_lin, "gain": r2_poly - r2_lin})
            print(f"  block {i:2d}  r2_lin={r2_lin:.3f}  gain={r2_poly - r2_lin:+.4f}"
                  f"  residual={1 - r2_lin:.3f}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    Path("plots/raw").mkdir(parents=True, exist_ok=True)
    Path("plots/raw/residual_gain_clean.json").write_text(json.dumps(rows, indent=2))
    csv = ["model,block,r2_lin,r2_poly,residual,gain"]
    csv += [f"{r['model']},{r['block']},{r['r2_lin']:.5f},{r['r2_poly']:.5f},"
            f"{r['residual']:.5f},{r['gain']:.5f}" for r in rows]
    Path("plots/raw/residual_gain_clean.csv").write_text("\n".join(csv) + "\n")

    _report_and_plot(rows)


def _pearson(a, b):
    a, b = torch.tensor(a), torch.tensor(b)
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


def _report_and_plot(rows):
    allx = [r["residual"] for r in rows]
    ally = [r["gain"] for r in rows]
    print(f"\nALL n={len(rows)}  pearson(residual, gain) = {_pearson(allx, ally):+.3f}")
    for name in MODELS:
        xs = [r["residual"] for r in rows if r["model"] == name]
        ys = [r["gain"] for r in rows if r["model"] == name]
        print(f"  {name:<24} pearson={_pearson(xs, ys):+.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for name in MODELS:
        c, lab = STYLE.get(name, ("C3", name))
        xs = [r["residual"] for r in rows if r["model"] == name]
        ys = [r["gain"] for r in rows if r["model"] == name]
        ax.scatter(xs, ys, c=c, label=lab, s=44, alpha=0.85, edgecolor="k", linewidth=0.4)
    ax.set_xlabel("residual nonlinearity  (1 − R²_linear)")
    ax.set_ylabel("poly gain over linear ceiling  (R²_poly − R²_linear)")
    ax.set_title(f"Does multiplication recover the residual?\n(per block; Pearson r = {_pearson(allx, ally):+.2f})")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"plots/polyweave_residual_gain_clean.{ext}", dpi=150)
    print("saved plots/polyweave_residual_gain_clean.{pdf,png}")


if __name__ == "__main__":
    main()
