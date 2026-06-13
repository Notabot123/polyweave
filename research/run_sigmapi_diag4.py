"""Does the `poly` >> `dense` gap on GPT-2 early survive a fair optimisation budget?

The target is ~95% linear (closed-form val R2 0.954). `poly`'s linear branch is a plain
nn.Linear on raw x — identical to `dense` — yet trained `poly` hits 0.955 while trained
`dense` stalls at 0.246 under the experiment's 3000-step/lr-1e-3 budget. If `dense`
climbs toward 0.95 with more steps / higher lr, the headline "multiplication helps"
on GPT-2 is an under-optimised-baseline artifact. If it plateaus far below `poly`,
the poly branch genuinely eases optimisation. Reports val R2 (experiment metric).
"""

from __future__ import annotations

import torch

from polyweave.distill.regression import fit_layer
from polyweave.experiments.gpt2_mlp_distill import (
    Config, load_model, token_batches, mlp_of, capture_block_io, build_candidates,
)

BLOCK = 1
MAX_TOKENS = 30_000


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config(model_name="gpt2", block_indices=(BLOCK,), dataset="wikitext2",
                 seq_len=128, batch_size=4, max_tokens=MAX_TOKENS, device=dev,
                 poly_rank=16)
    model, tok = load_model(cfg)
    batches = token_batches(cfg, tok, split="train")
    X, Y = capture_block_io(model, mlp_of(model, BLOCK), batches, cfg)
    X, Y = X.float().cpu(), Y.float().cpu()

    runs = [
        ("dense  3k  lr1e-3", torch.nn.Linear(768, 768), 3000, 1e-3),
        ("dense 20k  lr1e-3", torch.nn.Linear(768, 768), 20000, 1e-3),
        ("dense 20k  lr1e-2", torch.nn.Linear(768, 768), 20000, 1e-2),
        ("dense 50k  lr3e-3", torch.nn.Linear(768, 768), 50000, 3e-3),
        ("poly   3k  lr1e-3", build_candidates(768, cfg)["poly"], 3000, 1e-3),
    ]
    print(f"{'run':<20} {'val_r2':>8}")
    print("-" * 30)
    for label, layer, steps, lr in runs:
        r = fit_layer(layer, X, Y, steps=steps, lr=lr, batch_size=256,
                      val_frac=0.2, device=dev, seed=42)
        print(f"{label:<20} {r.val_r2:>8.4f}")


if __name__ == "__main__":
    main()
