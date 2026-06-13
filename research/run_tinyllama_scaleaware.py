"""Scale-aware TinyLlama-1.1B (SwiGLU, d=2048) distillation probe.

The first TinyLlama run (run_tinyllama_zeroshot.py) was messy: trained poly/dense2x
DIVERGED (negative R2) because rank-1 bilinear products of d=2048 outlier features blow
up at lr1e-3, and the closed-form linear ceiling read only ~0.05 (global variance-
weighted R2 tanked by a few unpredictable outlier output features). This re-runs with
scale-aware fitting to get a trustworthy datapoint:

  * linear  : exact closed-form ceiling + PER-FEATURE R2 (median/mean), to see whether
              the low global R2 is an outlier-feature artifact.
  * poly    : the §5.3 stable protocol — seed the linear branch from the closed-form
              solution, FREEZE it, train ONLY the low-rank quad branch with held-out
              early stopping (>= linear by construction, cannot diverge).
  * dense2x : trained with gradient clipping + lower lr, verified converged (R2 >= linear).

Reports global R2, per-feature R2 (median/mean), and zero-shot swap dPPL per candidate.
Single seed 42 first (probe); multi-seed only if it comes out clean.

Run:  venv/Scripts/python.exe -u run_tinyllama_scaleaware.py
Writes plots/raw/tinyllama_scaleaware_wikitext2.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from polyweave.layers import PolyLinear
from polyweave.distill import fit_closed_form_linear
from polyweave.distill.metrics import r2_score, cosine_similarity
from polyweave.experiments.gpt2_mlp_distill import (
    Config, load_model, token_batches, mlp_of, capture_block_io,
    block_swap_perplexity, _perplexity,
)
from polyweave.utils import count_params, set_seed

MODEL = "TinyLlama/TinyLlama_v1.1"
BLOCKS = [(2, "early"), (20, "deep")]
SEED = 42
MAX_TOKENS = 20_000
RANK = 16
RAW = Path("plots/raw")


def per_feature_r2(Y: torch.Tensor, P: torch.Tensor) -> tuple[float, float]:
    """Median and mean of the per-output-feature R2 (un-variance-weighted view)."""
    ss_res = ((Y - P) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0).clamp_min(1e-12)
    r2j = 1.0 - ss_res / ss_tot
    return float(r2j.median()), float(r2j.mean())


def seeded_poly(X, Y, r2_lin, dev, steps=8000, lr=1e-3, patience=8, eval_every=200):
    """PolyLinear with linear branch frozen at the closed-form optimum; train only the
    quad branch (early-stopped on held-out R2). Returns (best_layer, r2, cos)."""
    d = X.shape[1]
    n_tr = X.shape[0] - max(1, round(X.shape[0] * 0.2))
    Xtr, Ytr = X[:n_tr].to(dev), Y[:n_tr].to(dev)
    Xva, Yva = X[n_tr:].to(dev), Y[n_tr:].to(dev)
    poly = PolyLinear(d, d, rank=RANK).to(dev)
    fit_closed_form_linear(poly.linear, X, Y, val_frac=0.2, device=dev)
    for p in poly.linear.parameters():
        p.requires_grad_(False)
    quad = [p for n, p in poly.named_parameters() if not n.startswith("linear.")]
    opt = torch.optim.Adam(quad, lr=lr)
    g = torch.Generator().manual_seed(0)
    best, bad = r2_lin, 0
    best_state = {k: v.detach().clone() for k, v in poly.state_dict().items()}
    for step in range(1, steps + 1):
        poly.train()
        idx = torch.randint(0, n_tr, (256,), generator=g)
        opt.zero_grad()
        F.mse_loss(poly(Xtr[idx]), Ytr[idx]).backward()
        torch.nn.utils.clip_grad_norm_(quad, 1.0)
        opt.step()
        if step % eval_every == 0:
            poly.eval()
            with torch.no_grad():
                v = r2_score(Yva, poly(Xva))
            if v > best + 1e-4:
                best, bad = v, 0
                best_state = {k: val.detach().clone() for k, val in poly.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    poly.load_state_dict(best_state)
    poly.eval()
    with torch.no_grad():
        P = poly(Xva)
    return poly, float(r2_score(Yva, P)), float(cosine_similarity(Yva, P))


def stable_train(layer, X, Y, dev, steps=8000, lr=3e-4, clip=1.0, patience=10, eval_every=200):
    """Scale-aware training: grad clipping + lower lr + early stopping on held-out R2."""
    d = X.shape[1]
    n_tr = X.shape[0] - max(1, round(X.shape[0] * 0.2))
    Xtr, Ytr = X[:n_tr].to(dev), Y[:n_tr].to(dev)
    Xva, Yva = X[n_tr:].to(dev), Y[n_tr:].to(dev)
    layer = layer.to(dev)
    opt = torch.optim.AdamW(layer.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    best, bad = -1e9, 0
    best_state = {k: v.detach().clone() for k, v in layer.state_dict().items()}
    for step in range(1, steps + 1):
        layer.train()
        idx = torch.randint(0, n_tr, (256,), generator=g)
        opt.zero_grad()
        F.mse_loss(layer(Xtr[idx]), Ytr[idx]).backward()
        torch.nn.utils.clip_grad_norm_(layer.parameters(), clip)
        opt.step()
        if step % eval_every == 0:
            layer.eval()
            with torch.no_grad():
                v = r2_score(Yva, layer(Xva))
            if v > best + 1e-4:
                best, bad = v, 0
                best_state = {k: val.detach().clone() for k, val in layer.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    layer.load_state_dict(best_state)
    layer.eval()
    with torch.no_grad():
        P = layer(Xva)
    return layer, float(r2_score(Yva, P)), float(cosine_similarity(Yva, P))


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(SEED)
    cfg = Config(model_name=MODEL, dataset="wikitext2", seq_len=128, batch_size=2,
                 max_tokens=MAX_TOKENS, device=dev, poly_rank=RANK,
                 eval_perplexity=True, ppl_split="test", ppl_max_batches=30, seed=SEED)
    model, tok = load_model(cfg)
    batches = token_batches(cfg, tok)
    eval_batches = token_batches(cfg, tok, split="test")
    ppl_base = _perplexity(model, eval_batches, cfg)
    print(f"base PPL = {ppl_base:.3f}", flush=True)

    results = []
    for idx, depth in BLOCKS:
        mlp = mlp_of(model, idx)
        mlp_params = count_params(mlp)
        X, Y = capture_block_io(model, mlp, batches, cfg)
        d = X.shape[1]
        n_tr = X.shape[0] - max(1, round(X.shape[0] * 0.2))
        Yva = Y[n_tr:].to(dev)
        print(f"\n=== {depth} block {idx}  d={d}  rows={X.shape[0]}  "
              f"FFN {mlp_params:,} params ===", flush=True)

        block = {"depth": depth, "block": idx, "mlp_params": mlp_params,
                 "rows": X.shape[0], "ppl_base": ppl_base, "candidates": {}}

        # ---- linear (closed-form) ----
        lin = nn.Linear(d, d)
        r = fit_closed_form_linear(lin, X, Y, val_frac=0.2, device=dev)
        with torch.no_grad():
            Pva = lin(X[n_tr:].to(dev))
        pf_med, pf_mean = per_feature_r2(Yva, Pva)
        ppl = block_swap_perplexity(model, lin, idx, eval_batches, cfg, ppl_base=ppl_base)
        block["candidates"]["linear"] = {
            "num_params": r.num_params, "compression": mlp_params / r.num_params,
            "r2": r.val_r2, "r2_perfeat_median": pf_med, "r2_perfeat_mean": pf_mean,
            "cosine": r.val_cosine, "dppl_swap": ppl["ppl_swap"] - ppl_base}
        print(f"  linear     R2(global)={r.val_r2:+.4f}  R2(perfeat med)={pf_med:+.4f} "
              f"mean={pf_mean:+.4f}  cos={r.val_cosine:.4f}  "
              f"dPPL={ppl['ppl_swap'] - ppl_base:+.3f}", flush=True)

        # ---- poly (seeded + quad-only) ----
        poly, r2p, cosp = seeded_poly(X, Y, r.val_r2, dev)
        with torch.no_grad():
            Pva = poly(X[n_tr:].to(dev))
        pf_med, pf_mean = per_feature_r2(Yva, Pva)
        ppl = block_swap_perplexity(model, poly, idx, eval_batches, cfg, ppl_base=ppl_base)
        block["candidates"]["poly"] = {
            "num_params": count_params(poly), "compression": mlp_params / count_params(poly),
            "r2": r2p, "r2_perfeat_median": pf_med, "r2_perfeat_mean": pf_mean,
            "cosine": cosp, "dppl_swap": ppl["ppl_swap"] - ppl_base}
        print(f"  poly       R2(global)={r2p:+.4f}  R2(perfeat med)={pf_med:+.4f}  "
              f"cos={cosp:.4f}  dPPL={ppl['ppl_swap'] - ppl_base:+.3f}", flush=True)

        # ---- dense (2x), stabilized ----
        d2x = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        d2x, r2d, cosd = stable_train(d2x, X, Y, dev)
        with torch.no_grad():
            Pva = d2x(X[n_tr:].to(dev))
        pf_med, pf_mean = per_feature_r2(Yva, Pva)
        ppl = block_swap_perplexity(model, d2x, idx, eval_batches, cfg, ppl_base=ppl_base)
        block["candidates"]["dense (2x)"] = {
            "num_params": count_params(d2x), "compression": mlp_params / count_params(d2x),
            "r2": r2d, "r2_perfeat_median": pf_med, "r2_perfeat_mean": pf_mean,
            "cosine": cosd, "dppl_swap": ppl["ppl_swap"] - ppl_base}
        print(f"  dense (2x) R2(global)={r2d:+.4f}  R2(perfeat med)={pf_med:+.4f}  "
              f"cos={cosd:.4f}  dPPL={ppl['ppl_swap'] - ppl_base:+.3f}", flush=True)

        results.append(block)

    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "tinyllama_scaleaware_wikitext2.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved {RAW / 'tinyllama_scaleaware_wikitext2.json'}")


if __name__ == "__main__":
    main()
