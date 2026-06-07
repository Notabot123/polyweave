"""Experiment — compressing a GPT-2 MLP block into one position-wise layer.

A GPT-2 transformer block's feed-forward sub-network is two dense layers with a
GELU between them (``c_fc``: 768->3072, GELU, ``c_proj``: 3072->768) — a
position-wise map ``[.., 768] -> [.., 768]`` applied independently per token.
This experiment asks: can that ~4.7M-parameter two-layer MLP be *distilled* into a
single position-wise layer, and does a **multiplicative** layer (Sigma-Pi or the
factorized polynomial) capture the residual structure that a plain dense layer
cannot — at a fraction of the parameters?

Method (activation-space distillation, model-agnostic machinery in
``polyweave.distill``):

1. Load a pretrained GPT-2 and run a text corpus through it, tapping a chosen
   block's ``.mlp`` submodule with a forward hook to cache its ``(input, output)``
   activation pairs (one row per token).
2. Fit three single-layer candidates to those pairs by MSE regression at a chosen
   parameter budget:
     ``dense``     nn.Linear                  — additive baseline
     ``sigma-pi``  SigmaPiLinear              — log-space multiplicative branch
     ``poly``      PolyLinear (low-rank quad) — explicit bilinear branch
3. Report held-out relative-MSE / R^2, the multiplicative-recruitment gate
   (``exp(pi_scale)`` / ``exp(quad_scale)``) and its drift over training, the
   parameter count, and the compression ratio vs the original MLP.
4. Probe mechanism with occlusion: the conjunction (AND-signature) index over two
   disjoint halves of the input features — near 0 for an additive map, higher for
   a conjunctive/multiplicative one.

Repeated for an *early* and a *deep* block to see whether recruitment / fit
differs with depth (early FFNs are often described as more "key-value memory"
like, deeper ones more abstractive).

End-to-end perplexity after re-inserting the fitted layer into the network is the
natural next step but is deliberately *out of scope* here (it needs model surgery
plus a held-out text eval and more compute); it is flagged as further work.

Run:  python -m polyweave.experiments.gpt2_mlp_distill
(requires the optional ``transformers`` dependency: ``pip install polyweave[distill]``)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..distill import IOCapture, DistillResult, fit_layer
from ..interpretability import conjunction_index
from ..layers import PolyLinear, SigmaPiLinear
from ..utils import count_params, default_device, set_seed
from ..viz import (
    configure_plots,
    plot_conjunction_index,
    plot_grouped_bars,
    plot_lines,
)

# A small built-in corpus so ``run`` is self-contained for a quick demo. Real
# runs should pass ``cfg.text_paths`` pointing at a larger plain-text file
# (e.g. a WikiText slice) for activation pairs that actually cover the manifold.
_DEMO_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "In the beginning the universe was created. This has made a lot of people "
    "very angry and been widely regarded as a bad move. "
    "It was the best of times, it was the worst of times, it was the age of "
    "wisdom, it was the age of foolishness. "
    "All happy families are alike; each unhappy family is unhappy in its own way."
) * 8


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    seed: int = 42
    device: str = default_device()

    # Model + blocks to distil. ``block_labels`` names them for tables/plots.
    model_name: str = "gpt2"          # any HF causal-LM with a .mlp per block
    block_indices: Tuple[int, ...] = (1, 10)        # early, deep (gpt2 has 12)
    block_labels: Tuple[str, ...] = ("early block", "deep block")

    # Activation capture.
    seq_len: int = 128
    batch_size: int = 4
    max_tokens: int = 20_000          # cap captured rows (memory/disk bound)
    text_paths: Tuple[str, ...] = ()  # plain-text files; empty -> built-in demo text

    # Candidate layers.
    poly_rank: int = 16
    equal_budget: bool = False        # if True also fit a dense bottleneck MLP
                                      # matched to the Sigma-Pi parameter count

    # Regression.
    steps: int = 3000
    lr: float = 1e-3
    fit_batch_size: int = 256
    weight_decay: float = 0.0
    val_frac: float = 0.2
    eval_every: int = 100

    # Occlusion AND-signature probe.
    occlusion_rows: int = 256         # tokens used for the conjunction index

    dark_plots: bool = False
    plot_prefix: str = "polyweave_gpt2_mlp_distill"
    results_path: str = "plots/raw/gpt2_mlp_distill.json"


# ---------------------------------------------------------------------------
# Per-candidate / per-block result containers
# ---------------------------------------------------------------------------

@dataclass
class CandidateResult:
    name: str
    num_params: int
    compression: float                # original MLP params / candidate params
    val_rel_mse: float
    val_r2: float
    recruit_start: Optional[float]
    recruit_final: Optional[float]
    conjunction: Optional[float] = None
    recruit_curve: List[Tuple[int, float]] = field(default_factory=list)

    @property
    def recruit_delta(self) -> Optional[float]:
        if self.recruit_start is None or self.recruit_final is None:
            return None
        return self.recruit_final - self.recruit_start


@dataclass
class BlockResult:
    label: str
    block_index: int
    mlp_params: int
    num_rows: int
    candidates: Dict[str, CandidateResult] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model / data plumbing (monkeypatched by the offline smoke test)
# ---------------------------------------------------------------------------

def load_model(cfg: Config):
    """Load a pretrained causal-LM + tokenizer via ``transformers`` (lazy import).

    Returns ``(model, tokenizer)`` with the model in eval mode on ``cfg.device``.
    Kept as a module-level function so tests can monkeypatch it with a stub.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy, optional dep

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    model.to(cfg.device).eval()
    return model, tokenizer


def mlp_of(model, block_index: int) -> nn.Module:
    """Resolve a block's feed-forward submodule across common HF layouts.

    Handles GPT-2 (``model.transformer.h[i].mlp``) and the flatter ``model.h[i]``
    layout used by some stubs/architectures.
    """
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h
    elif hasattr(model, "h"):
        blocks = model.h
    else:
        raise AttributeError(
            "could not locate transformer blocks; expected model.transformer.h "
            "or model.h"
        )
    block = blocks[block_index]
    if not hasattr(block, "mlp"):
        raise AttributeError(f"block {block_index} has no .mlp submodule")
    return block.mlp


def token_batches(cfg: Config, tokenizer) -> List[torch.Tensor]:
    """Tokenise the corpus into ``[batch_size, seq_len]`` input-id tensors.

    Concatenates all text, then chops the id stream into non-overlapping windows
    of ``seq_len`` and packs them into batches — enough windows to cover roughly
    ``max_tokens`` tokens.
    """
    if cfg.text_paths:
        texts = [Path(p).read_text(encoding="utf-8") for p in cfg.text_paths]
        corpus = "\n".join(texts)
    else:
        corpus = _DEMO_TEXT
    ids = tokenizer(corpus, return_tensors="pt").input_ids[0]
    n_windows = max(1, min(len(ids) // cfg.seq_len, cfg.max_tokens // cfg.seq_len))
    ids = ids[: n_windows * cfg.seq_len].reshape(n_windows, cfg.seq_len)
    return [ids[i : i + cfg.batch_size] for i in range(0, n_windows, cfg.batch_size)]


# ---------------------------------------------------------------------------
# Capture + candidate building
# ---------------------------------------------------------------------------

@torch.no_grad()
def capture_block_io(
    model, mlp: nn.Module, batches: List[torch.Tensor], cfg: Config
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drive forward passes and return the block's ``(X, Y)`` activation pairs."""
    with IOCapture(mlp, max_rows=cfg.max_tokens, device="cpu") as cap:
        for input_ids in batches:
            model(input_ids.to(cfg.device))
            if cap.num_rows >= cfg.max_tokens:
                break
    return cap.pairs()


def build_candidates(d_model: int, cfg: Config) -> Dict[str, nn.Module]:
    """The single-layer candidates that compete to replace the MLP block."""
    candidates: Dict[str, nn.Module] = {
        "dense": nn.Linear(d_model, d_model),
        "sigma-pi": SigmaPiLinear(d_model, d_model),
        "poly": PolyLinear(d_model, d_model, rank=cfg.poly_rank),
    }
    if cfg.equal_budget:
        # A two-layer dense bottleneck whose param count matches SigmaPiLinear's
        # (~2 * d_model^2): 768->768->768 is ~2 * d^2, the same budget split
        # additively instead of into a multiplicative branch.
        candidates["dense (2x)"] = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
    return candidates


def _recruit_value(layer: nn.Module) -> Optional[float]:
    fn = getattr(layer, "pi_scale_mean", None) or getattr(layer, "quad_scale_mean", None)
    return fn() if callable(fn) else None


@torch.no_grad()
def _conjunction_for_layer(layer: nn.Module, X: torch.Tensor, cfg: Config) -> float:
    """Mean AND-signature index of a fitted layer over two disjoint feature halves.

    The response is the layer's mean output per token; features are split into the
    first and second halves of the input dimension. Additive maps score ~0;
    conjunctive (multiplicative) maps score higher because occluding either half
    alone already collapses much of the response.
    """
    d = X.shape[1]
    device = next(layer.parameters()).device
    rows = X[: cfg.occlusion_rows].to(device)
    group_a = list(range(d // 2))
    group_b = list(range(d // 2, d))
    layer.eval()

    def response(batch: torch.Tensor) -> torch.Tensor:
        return layer(batch).mean(dim=1)

    return conjunction_index(response, rows, group_a, group_b).mean().item()


# ---------------------------------------------------------------------------
# Core: distil one block
# ---------------------------------------------------------------------------

def distill_block(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    label: str,
    block_index: int,
    mlp_params: int,
    cfg: Config,
    log: Callable[[str], None] = print,
) -> BlockResult:
    """Fit every candidate to one block's activation pairs and collect metrics."""
    d_model = X.shape[1]
    block = BlockResult(
        label=label, block_index=block_index, mlp_params=mlp_params,
        num_rows=X.shape[0],
    )
    log(f"\n=== {label} (block {block_index}) - {X.shape[0]} token rows, "
        f"d_model={d_model}, original MLP {mlp_params:,} params ===")

    for name, layer in build_candidates(d_model, cfg).items():
        res: DistillResult = fit_layer(
            layer, X, Y,
            steps=cfg.steps, lr=cfg.lr, batch_size=cfg.fit_batch_size,
            weight_decay=cfg.weight_decay, val_frac=cfg.val_frac,
            eval_every=cfg.eval_every, device=cfg.device, seed=cfg.seed,
        )
        recruit_start = res.recruit_curve[0][1] if res.recruit_curve else None
        recruit_final = res.recruit_curve[-1][1] if res.recruit_curve else None
        cand = CandidateResult(
            name=name,
            num_params=res.num_params,
            compression=mlp_params / max(res.num_params, 1),
            val_rel_mse=res.val_rel_mse,
            val_r2=res.val_r2,
            recruit_start=recruit_start,
            recruit_final=recruit_final,
            conjunction=_conjunction_for_layer(layer, X, cfg),
            recruit_curve=res.recruit_curve,
        )
        block.candidates[name] = cand
        gate = (
            f"  gate {recruit_start:.4f}->{recruit_final:.4f} "
            f"(d{cand.recruit_delta:+.4f})" if recruit_final is not None else ""
        )
        log(f"  {name:<12} params={res.num_params:>9,}  "
            f"compress x{cand.compression:5.1f}  rel_mse={res.val_rel_mse:.4f}  "
            f"R2={res.val_r2:.4f}  AND={cand.conjunction:.3f}{gate}")
    return block


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg: Config, make_plots: bool = True) -> List[BlockResult]:
    configure_plots(cfg.dark_plots)
    set_seed(cfg.seed)
    print("=" * 64)
    print(cfg)
    print("=" * 64)

    model, tokenizer = load_model(cfg)
    batches = token_batches(cfg, tokenizer)
    print(f"corpus: {len(batches)} batches of "
          f"<= {cfg.batch_size}x{cfg.seq_len} tokens")

    results: List[BlockResult] = []
    for label, idx in zip(cfg.block_labels, cfg.block_indices):
        mlp = mlp_of(model, idx)
        mlp_params = count_params(mlp)
        X, Y = capture_block_io(model, mlp, batches, cfg)
        results.append(distill_block(
            X, Y, label=label, block_index=idx, mlp_params=mlp_params, cfg=cfg,
        ))

    _save_results(cfg, results)
    if make_plots:
        _make_plots(cfg, results)
    print("\nDone.")
    return results


def _save_results(cfg: Config, results: List[BlockResult]) -> None:
    path = Path(cfg.results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "label": b.label,
            "block_index": b.block_index,
            "mlp_params": b.mlp_params,
            "num_rows": b.num_rows,
            "candidates": {
                name: {
                    "num_params": c.num_params,
                    "compression": c.compression,
                    "val_rel_mse": c.val_rel_mse,
                    "val_r2": c.val_r2,
                    "recruit_start": c.recruit_start,
                    "recruit_final": c.recruit_final,
                    "recruit_delta": c.recruit_delta,
                    "conjunction": c.conjunction,
                }
                for name, c in b.candidates.items()
            },
        }
        for b in results
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {path}")


def _make_plots(cfg: Config, results: List[BlockResult]) -> None:
    # Held-out fit (R^2) grouped by block depth, one bar per candidate.
    r2 = {b.label: {n: c.val_r2 for n, c in b.candidates.items()} for b in results}
    plot_grouped_bars(
        r2, name=f"{cfg.plot_prefix}_r2",
        title="MLP-block distillation fit (held-out R²)",
        ylabel="R²", xlabel="GPT-2 block",
    )
    # Relative MSE, same grouping (lower is better).
    rel = {b.label: {n: c.val_rel_mse for n, c in b.candidates.items()} for b in results}
    plot_grouped_bars(
        rel, name=f"{cfg.plot_prefix}_relmse",
        title="MLP-block distillation residual (held-out relative MSE)",
        ylabel="relative MSE", xlabel="GPT-2 block",
    )
    # AND-signature (conjunction index) per candidate, per block.
    andsig = {
        b.label: {n: (c.conjunction or 0.0) for n, c in b.candidates.items()}
        for b in results
    }
    plot_grouped_bars(
        andsig, name=f"{cfg.plot_prefix}_conjunction",
        title="Occlusion AND-signature (conjunction index)",
        ylabel="conjunction index", xlabel="GPT-2 block",
    )
    # Recruitment-gate trajectories for the gated candidates, per block.
    for b in results:
        curves = {
            n: [v for _, v in c.recruit_curve]
            for n, c in b.candidates.items() if c.recruit_curve
        }
        if curves:
            plot_lines(
                curves, title=f"Multiplicative recruitment — {b.label}",
                ylabel="gate mean exp(scale)", xlabel="evaluation point",
                name=f"{cfg.plot_prefix}_recruit_{b.block_index}",
            )


if __name__ == "__main__":
    run(Config())
