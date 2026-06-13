"""TinyLlama-1.1B zero-shot external-validity check (SwiGLU at larger scale).

A cheap single-seed, ZERO-SHOT (no heal) probe: does a *larger* SwiGLU model's FFN also
resist single-layer (and linear) compression, like llama-160m? Reports the closed-form
linear ceiling + trained poly / dense(2x), activation fit and zero-shot ΔPPL, for an
early and a deep block. Runs on GPU: fp32 weights are ~4.4 GB and inference + the
(no-backprop) zero-shot fitting fit a 6 GB card; if it OOMs, set device="cpu" below.
Delete the HF cache afterward to reclaim disk (the model is ~4 GB).

TinyLlama_v1.1: d=2048, 22 layers, SiLU SwiGLU, intermediate 5632.
Run:  venv/Scripts/python.exe -u run_tinyllama_zeroshot.py
"""

from __future__ import annotations

import json
from pathlib import Path

from polyweave.experiments.gpt2_mlp_distill import Config, run

RAW = Path("plots/raw")
OUT = RAW / "tinyllama_mlp_distill_wikitext2_zeroshot.json"


def main() -> None:
    cfg = Config(
        model_name="TinyLlama/TinyLlama_v1.1",
        block_indices=(2, 20),
        block_labels=("early block", "deep block"),
        dataset="wikitext2",
        seq_len=128,
        batch_size=4,
        max_tokens=20_000,
        poly_rank=16,
        equal_budget=True,
        include_sigma_pi=False,
        linear_closed_form=True,
        steps=8000,
        lr=1e-3,
        seed=42,
        device="cuda",                # fp32 1.1B (~4.4 GB) fits a 6 GB card for
                                      # inference + zero-shot fit; set "cpu" if it OOMs
        eval_perplexity=True,
        ppl_split="test",
        ppl_max_batches=30,
        heal_steps=0,                 # ZERO-SHOT only (no heal)
        results_path=str(OUT),
        plot_prefix="polyweave_tinyllama_mlp_distill_wikitext2_zeroshot",
    )
    results = run(cfg, make_plots=False)

    print(f"\n{'=' * 84}")
    print("TINYLLAMA-1.1B ZERO-SHOT  (SwiGLU, d=2048; dense = closed-form linear ceiling)")
    print("=" * 84)
    print(f"  {'block':<12} {'layer':<12} {'R2':>8} {'cosine':>8} {'dPPL_swap':>10}")
    print("  " + "-" * 52)
    for b in results:
        for name, c in b.candidates.items():
            print(f"  {b.label:<12} {name:<12} {c.val_r2:>8.3f} {c.val_cosine:>8.3f} "
                  f"{(c.dppl_swap if c.dppl_swap is not None else float('nan')):>10.3f}")
    print("=" * 84)
    print(f"saved {OUT}")
    print("\nReclaim disk with:  huggingface-cli delete-cache  (or remove the "
          "models--TinyLlama--TinyLlama_v1.1 dir under ~/.cache/huggingface/hub)")


if __name__ == "__main__":
    main()
