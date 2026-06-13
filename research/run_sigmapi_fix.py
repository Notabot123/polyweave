"""Validate a sigma-pi fix: seed the additive branch from the closed-form linear
solution and start the product branch silent, so the layer BEGINS at the linear
ceiling and the multiplicative branch can only add (or the gate suppresses it).

Tests, on the two diagnostic-critical blocks (GPT-2 early, llama early):
  ceiling     closed-form linear val R2 (target to match-or-beat)
  seed_only   seeded sigma-pi, NO training (should ~= ceiling: pi starts silent)
  C0_joint    current defaults (exp, max_log6, scale-2), joint fine-tune  [shows breakage]
  C1_joint    stabilised (expm1, max_log2, scale-4), joint fine-tune
  C2_pi_only  stabilised, FREEZE seeded sigma, train pi branch only (guaranteed >= linear)
  C3_pi_signed C2 + signed_products (the sign-in-logspace idea)

Each trained config reports val R2 at 2k and at 10k steps to expose instability
(the current layer DEGRADES with more training: 0.948@3k -> 0.776@30k).
"""

from __future__ import annotations

import torch

from polyweave.distill.metrics import r2_score
from polyweave.layers import SigmaPiLinear
from polyweave.experiments.gpt2_mlp_distill import (
    Config, load_model, token_batches, mlp_of, capture_block_io,
)

CASES = [("gpt2", 1), ("JackFram/llama-160m", 1)]
MAX_TOKENS = 30_000


def _seed_sigma(layer, Xtr, Ytr):
    """Seed the (centered) sigma branch from the closed-form lstsq fit it would see."""
    Xc = (Xtr - Xtr.mean(dim=1, keepdim=True)).double()
    ones = torch.ones(Xc.shape[0], 1, dtype=torch.float64)
    W = torch.linalg.lstsq(torch.cat([Xc, ones], 1), Ytr.double()).solution  # [d+1,out]
    with torch.no_grad():
        layer.sigma.weight.copy_(W[:-1].T.float())
        layer.sigma.bias.copy_(W[-1].float())


def _val_r2(layer, Xva, Yva, dev):
    layer.eval()
    with torch.no_grad():
        p = layer(Xva.to(dev)).cpu()
    return r2_score(Yva, p)


def _train(layer, X, Y, n_tr, dev, *, lr, steps, freeze_sigma, probe=2000):
    Xtr, Ytr, Xva, Yva = X[:n_tr].to(dev), Y[:n_tr].to(dev), X[n_tr:], Y[n_tr:]
    if freeze_sigma:
        for p in layer.sigma.parameters():
            p.requires_grad_(False)
    params = [p for p in layer.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    g = torch.Generator().manual_seed(42)
    r2_probe = None
    for step in range(1, steps + 1):
        idx = torch.randint(0, n_tr, (256,), generator=g)
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(layer(Xtr[idx]), Ytr[idx])
        loss.backward()
        opt.step()
        if step == probe:
            r2_probe = _val_r2(layer, Xva, Yva, dev)
            layer.train()
    return r2_probe, _val_r2(layer, Xva, Yva, dev)


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for name, blk in CASES:
        cfg = Config(model_name=name, dataset="wikitext2", seq_len=128, batch_size=4,
                     max_tokens=MAX_TOKENS, device=dev)
        model, tok = load_model(cfg)
        batches = token_batches(cfg, tok, split="train")
        X, Y = capture_block_io(model, mlp_of(model, blk), batches, cfg)
        X, Y = X.float().cpu(), Y.float().cpu()
        n_tr = X.shape[0] - max(1, round(X.shape[0] * 0.2))
        d = X.shape[1]

        # ceiling
        Xc = (X[:n_tr] - X[:n_tr].mean(1, keepdim=True)).double()
        ones = torch.ones(n_tr, 1, dtype=torch.float64)
        W = torch.linalg.lstsq(torch.cat([Xc, ones], 1), Y[:n_tr].double()).solution
        Xcv = (X[n_tr:] - X[n_tr:].mean(1, keepdim=True)).double()
        onev = torch.ones(X.shape[0] - n_tr, 1, dtype=torch.float64)
        ceiling = r2_score(Y[n_tr:].double(), torch.cat([Xcv, onev], 1) @ W)

        print(f"\n=== {name}  block {blk}   ceiling={ceiling:.4f} ===")
        print(f"{'config':<14} {'r2@2k':>8} {'r2@10k':>8}")

        # seed_only
        L = SigmaPiLinear(d, d, center_product=True, max_log=2.0, pi_scale_init=-4.0).to(dev)
        _seed_sigma(L, X[:n_tr], Y[:n_tr])
        print(f"{'seed_only':<14} {'-':>8} {_val_r2(L, X[n_tr:], Y[n_tr:], dev):>8.4f}")

        specs = [
            ("C0_joint", dict(center_product=False, max_log=6.0, pi_scale_init=-2.0), 3e-4, False),
            ("C1_joint", dict(center_product=True, max_log=2.0, pi_scale_init=-4.0), 3e-4, False),
            ("C2_pi_only", dict(center_product=True, max_log=2.0, pi_scale_init=-4.0), 1e-3, True),
            ("C3_pi_signed", dict(center_product=True, max_log=2.0, pi_scale_init=-4.0,
                                  signed_products=True), 1e-3, True),
        ]
        for label, kw, lr, freeze in specs:
            L = SigmaPiLinear(d, d, **kw).to(dev)
            _seed_sigma(L, X[:n_tr], Y[:n_tr])
            r2k, r10k = _train(L, X, Y, n_tr, dev, lr=lr, steps=10000, freeze_sigma=freeze)
            print(f"{label:<14} {r2k:>8.4f} {r10k:>8.4f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
