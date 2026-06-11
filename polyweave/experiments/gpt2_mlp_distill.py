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

End-to-end perplexity (opt-in via ``cfg.eval_perplexity``) re-inserts each fitted
layer into the live model and measures held-out LM perplexity — the downstream
truth that activation R² only proxies. We report the zero-shot swap ΔPPL and,
optionally (``cfg.heal_steps > 0``), the ΔPPL after *healing the swapped layer
only* (fine-tuning just the new layer with the rest of the model frozen — a fair,
cheap probe of the layer's standalone capacity, not full-model retraining). Use a
real corpus for this: ``cfg.dataset="wikitext2"`` (WikiText-2 raw, cached locally)
or ``cfg.text_paths``; the built-in demo text is for the self-contained smoke run.

Run:  python -m polyweave.experiments.gpt2_mlp_distill
(requires the optional ``transformers`` dependency: ``pip install polyweave[distill]``;
WikiText-2 additionally needs ``datasets`` on first fetch)
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..distill import IOCapture, DistillResult, fit_closed_form_linear, fit_layer
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

    # Corpus. ``dataset`` selects the source when ``text_paths`` is empty:
    #   "demo"      -> built-in _DEMO_TEXT (self-contained quick run / smoke test)
    #   "wikitext2" -> WikiText-2 raw via _wikitext.wikitext2_text (needs `datasets`
    #                  on first fetch; cached to wikitext_cache_dir thereafter).
    # ``text_paths`` (if set) always wins, for arbitrary custom corpora.
    dataset: str = "demo"
    wikitext_cache_dir: str = "data"

    # Activation capture.
    seq_len: int = 128
    batch_size: int = 4
    max_tokens: int = 20_000          # cap captured rows (memory/disk bound)
    text_paths: Tuple[str, ...] = ()  # plain-text files; empty -> ``dataset`` source

    # Candidate layers.
    poly_rank: int = 16
    equal_budget: bool = False        # if True also fit a dense bottleneck MLP
                                      # matched to the Sigma-Pi parameter count
    include_sigma_pi: bool = True     # if False, drop the Sigma-Pi candidate (its
                                      # log/exp branch needs a redesign; excluded from
                                      # the FFN-distillation paper, kept for other runs)
    linear_closed_form: bool = False  # if True, solve the "dense" linear baseline in
                                      # closed form (exact least squares) instead of
                                      # training it — the true linear ceiling, immune to
                                      # the optimiser-underfit confound on ill-conditioned
                                      # activations. The multiplicative/depth candidates
                                      # are still trained.

    # Regression.
    steps: int = 3000
    lr: float = 1e-3
    fit_batch_size: int = 256
    weight_decay: float = 0.0
    val_frac: float = 0.2
    eval_every: int = 100

    # Occlusion AND-signature probe.
    occlusion_rows: int = 256         # tokens used for the conjunction index

    # End-to-end perplexity (re-insert the fitted layer into the live model and
    # measure LM perplexity on a held-out split). OFF by default so the demo /
    # offline smoke test never need a real causal-LM head. Real runs set True.
    eval_perplexity: bool = False
    ppl_split: str = "test"           # held-out split for PPL (wikitext2)
    ppl_max_batches: int = 50         # cap eval batches (compute bound)
    heal_steps: int = 0               # >0: fine-tune the SWAPPED LAYER ONLY (LM loss)
                                      # before the post-swap PPL read (heal-the-layer,
                                      # NOT full-model FT). Same steps for every cand.
    heal_lr: float = 1e-4

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
    val_rmse: float
    val_cosine: float
    recruit_start: Optional[float]
    recruit_final: Optional[float]
    conjunction: Optional[float] = None
    recruit_curve: List[Tuple[int, float]] = field(default_factory=list)
    # End-to-end perplexity (populated only when cfg.eval_perplexity): the live
    # model's PPL with this candidate swapped in for the block's MLP, zero-shot and
    # (if cfg.heal_steps>0) after healing the swapped layer. ``ppl_base`` is the
    # untouched-model PPL, repeated per candidate for convenience.
    ppl_base: Optional[float] = None
    ppl_swap: Optional[float] = None
    ppl_heal: Optional[float] = None
    # The fitted layer itself, kept so ``run`` can swap it into the model for PPL.
    # Not serialised (excluded from repr/compare and skipped in _save_results).
    layer: Optional[nn.Module] = field(default=None, repr=False, compare=False)

    @property
    def recruit_delta(self) -> Optional[float]:
        if self.recruit_start is None or self.recruit_final is None:
            return None
        return self.recruit_final - self.recruit_start

    @property
    def dppl_swap(self) -> Optional[float]:
        """Zero-shot perplexity increase from the swap (lower = better fit)."""
        if self.ppl_base is None or self.ppl_swap is None:
            return None
        return self.ppl_swap - self.ppl_base

    @property
    def dppl_heal(self) -> Optional[float]:
        """Perplexity increase after healing the swapped layer only."""
        if self.ppl_base is None or self.ppl_heal is None:
            return None
        return self.ppl_heal - self.ppl_base


@dataclass
class BlockResult:
    label: str
    block_index: int
    mlp_params: int
    num_rows: int
    candidates: Dict[str, CandidateResult] = field(default_factory=dict)
    # End-to-end perplexity references (populated only when cfg.eval_perplexity):
    # the untouched-model PPL, and — when cfg.heal_steps>0 — the PPL after healing
    # the ORIGINAL block's MLP with the *same* budget as the candidates. The latter
    # is the fair "equally-adapted original" baseline: a candidate's healed ΔPPL is
    # only meaningful against an original given the same in-domain fine-tuning, since
    # healing on the train split adapts to a corpus the base model never saw.
    ppl_base: Optional[float] = None
    ppl_heal_original: Optional[float] = None

    @property
    def dppl_heal_original(self) -> Optional[float]:
        if self.ppl_base is None or self.ppl_heal_original is None:
            return None
        return self.ppl_heal_original - self.ppl_base


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
    # Force float32: some checkpoints (e.g. Pythia) are stored in fp16, but the whole
    # distillation pipeline (closed-form solve, fitted fp32 candidates, layer swap) is
    # fp32 — a half-precision model would dtype-mismatch the swapped layer.
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.float32)
    model.to(cfg.device).eval()
    return model, tokenizer


def _blocks(model):
    """Resolve the list of transformer blocks across common HF layouts.

    Handles GPT-2 (``model.transformer.h``), the flatter ``model.h`` used by some
    stubs, the Llama/Mistral family (``model.model.layers`` — the SwiGLU-FFN decoders),
    and GPT-NeoX / Pythia (``model.gpt_neox.layers`` — a second GELU-FFN model).
    """
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers  # GPT-NeoX / Pythia (GELU FFN, second GELU model)
    if hasattr(model, "h"):
        return model.h
    raise AttributeError(
        "could not locate transformer blocks; expected model.transformer.h, "
        "model.model.layers, model.gpt_neox.layers, or model.h"
    )


def mlp_of(model, block_index: int) -> nn.Module:
    """Resolve a block's feed-forward submodule across common HF layouts.

    Handles GPT-2 (``model.transformer.h[i].mlp``) and the flatter ``model.h[i]``
    layout used by some stubs/architectures.
    """
    block = _blocks(model)[block_index]
    if not hasattr(block, "mlp"):
        raise AttributeError(f"block {block_index} has no .mlp submodule")
    return block.mlp


def corpus_text(cfg: Config, split: str = "train") -> str:
    """Resolve the raw corpus string for a split.

    Precedence: explicit ``cfg.text_paths`` (any custom files) > ``cfg.dataset``
    (``"wikitext2"`` -> the cached WikiText-2 split) > built-in ``_DEMO_TEXT``.
    The demo corpus is split-agnostic (the same text for every split).
    """
    if cfg.text_paths:
        return "\n".join(Path(p).read_text(encoding="utf-8") for p in cfg.text_paths)
    if cfg.dataset == "wikitext2":
        from ._wikitext import wikitext2_text  # local import keeps datasets optional
        return wikitext2_text(split, cfg.wikitext_cache_dir)
    return _DEMO_TEXT


def token_batches(cfg: Config, tokenizer, split: str = "train") -> List[torch.Tensor]:
    """Tokenise the corpus into ``[batch_size, seq_len]`` input-id tensors.

    Concatenates the ``split`` corpus, chops the id stream into non-overlapping
    ``seq_len`` windows and packs them into batches — enough windows to cover
    roughly ``max_tokens`` tokens.
    """
    ids = tokenizer(corpus_text(cfg, split), return_tensors="pt").input_ids[0]
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
    }
    if cfg.include_sigma_pi:
        candidates["sigma-pi"] = SigmaPiLinear(d_model, d_model)
    candidates["poly"] = PolyLinear(d_model, d_model, rank=cfg.poly_rank)
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
# End-to-end perplexity (re-insert the fitted layer into the live model)
# ---------------------------------------------------------------------------

def _lm_loss(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy for a causal LM on ``input_ids`` [B, T].

    Uses the HF convention ``model(input_ids, labels=input_ids).loss`` (the model
    shifts internally, averaging over the B*(T-1) predicted positions).
    """
    return model(input_ids, labels=input_ids).loss


@torch.no_grad()
def _perplexity(model, batches: List[torch.Tensor], cfg: Config) -> float:
    """Token-weighted perplexity over ``batches`` (``exp`` of the mean LM loss)."""
    model.eval()
    total_loss, total_tok = 0.0, 0
    for input_ids in batches[: cfg.ppl_max_batches]:
        input_ids = input_ids.to(cfg.device)
        n_pred = input_ids.shape[0] * max(input_ids.shape[1] - 1, 1)
        total_loss += _lm_loss(model, input_ids).item() * n_pred
        total_tok += n_pred
    return math.exp(total_loss / max(total_tok, 1))


def _heal_layer(
    model, layer: nn.Module, block_index: int,
    heal_batches: List[torch.Tensor], cfg: Config,
) -> None:
    """Fine-tune ONLY the swapped ``layer`` (LM loss) with the rest of the model
    frozen — the fair, cheap "heal" that isolates the layer's representational
    capacity without re-training the whole network. Mutates ``layer`` in place.
    Assumes ``layer`` is already installed as ``block.mlp``.
    """
    frozen = [p for p in model.parameters() if p.requires_grad]
    for p in frozen:
        p.requires_grad_(False)
    for p in layer.parameters():
        p.requires_grad_(True)
    layer.train()
    opt = torch.optim.Adam(layer.parameters(), lr=cfg.heal_lr)
    done = 0
    while done < cfg.heal_steps:
        for input_ids in heal_batches:
            input_ids = input_ids.to(cfg.device)
            opt.zero_grad()
            _lm_loss(model, input_ids).backward()
            opt.step()
            done += 1
            if done >= cfg.heal_steps:
                break
    layer.eval()
    for p in frozen:  # restore the rest of the model's grad flags
        p.requires_grad_(True)


def block_swap_perplexity(
    model, layer: nn.Module, block_index: int,
    eval_batches: List[torch.Tensor], cfg: Config,
    heal_batches: Optional[List[torch.Tensor]] = None,
    ppl_base: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """Perplexity with ``layer`` swapped in for block ``block_index``'s MLP.

    Returns ``{"ppl_base", "ppl_swap", "ppl_heal"}``. ``ppl_base`` is the untouched
    model (pass a precomputed value to skip recomputation — it is invariant across
    candidates); ``ppl_swap`` is zero-shot after the swap; ``ppl_heal`` is after
    healing the swapped layer (``None`` unless ``cfg.heal_steps > 0`` and heal
    batches are given). The original MLP is always restored before returning.
    """
    blocks = _blocks(model)
    original = blocks[block_index].mlp
    layer = layer.to(cfg.device)
    try:
        if ppl_base is None:
            ppl_base = _perplexity(model, eval_batches, cfg)
        blocks[block_index].mlp = layer
        layer.eval()
        ppl_swap = _perplexity(model, eval_batches, cfg)
        ppl_heal: Optional[float] = None
        if cfg.heal_steps > 0 and heal_batches:
            _heal_layer(model, layer, block_index, heal_batches, cfg)
            ppl_heal = _perplexity(model, eval_batches, cfg)
        return {"ppl_base": ppl_base, "ppl_swap": ppl_swap, "ppl_heal": ppl_heal}
    finally:
        blocks[block_index].mlp = original


def heal_original_perplexity(
    model, block_index: int,
    eval_batches: List[torch.Tensor], heal_batches: List[torch.Tensor], cfg: Config,
) -> float:
    """PPL after healing a *copy of the original* block MLP with the heal budget.

    The fair baseline for a candidate's healed ΔPPL: the original two-layer MLP
    given the **same** in-domain fine-tuning the candidates receive. If a small
    candidate heals to roughly this, it matches an equally-adapted original. Works
    on a ``deepcopy`` so the live model's weights are never mutated; the original
    submodule is always restored before returning.
    """
    blocks = _blocks(model)
    original = blocks[block_index].mlp
    clone = copy.deepcopy(original).to(cfg.device)
    try:
        blocks[block_index].mlp = clone
        _heal_layer(model, clone, block_index, heal_batches, cfg)
        return _perplexity(model, eval_batches, cfg)
    finally:
        blocks[block_index].mlp = original


def _evaluate_perplexity(
    model, tokenizer, results: List[BlockResult], cfg: Config,
    heal_batches: List[torch.Tensor], log: Callable[[str], None] = print,
) -> None:
    """Populate each candidate's PPL fields by swapping it into the live model."""
    eval_batches = token_batches(cfg, tokenizer, split=cfg.ppl_split)
    do_heal = cfg.heal_steps > 0 and bool(heal_batches)
    ppl_base = _perplexity(model, eval_batches, cfg)  # invariant across candidates
    log(f"\n=== end-to-end perplexity ({cfg.ppl_split} split, "
        f"{len(eval_batches[: cfg.ppl_max_batches])} batches, base PPL {ppl_base:.3f}"
        f"{f', heal {cfg.heal_steps} steps' if do_heal else ''}) ===")
    for block in results:
        block.ppl_base = ppl_base
        if do_heal:
            block.ppl_heal_original = heal_original_perplexity(
                model, block.block_index, eval_batches, heal_batches, cfg,
            )
        ho = (f"  [orig healed={block.ppl_heal_original:.3f} "
              f"(d{block.dppl_heal_original:+.3f})]"
              if block.ppl_heal_original is not None else "")
        log(f"  {block.label} (block {block.block_index}):{ho}")
        for name, cand in block.candidates.items():
            if cand.layer is None:
                continue
            ppl = block_swap_perplexity(
                model, cand.layer, block.block_index, eval_batches, cfg,
                heal_batches=heal_batches, ppl_base=ppl_base,
            )
            cand.ppl_base = ppl["ppl_base"]
            cand.ppl_swap = ppl["ppl_swap"]
            cand.ppl_heal = ppl["ppl_heal"]
            heal = (f" heal={cand.ppl_heal:.3f} (d{cand.dppl_heal:+.3f})"
                    if cand.ppl_heal is not None else "")
            log(f"    {name:<12} base={cand.ppl_base:.3f}  "
                f"swap={cand.ppl_swap:.3f} (d{cand.dppl_swap:+.3f}){heal}")


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
        if cfg.linear_closed_form and name == "dense":
            # Exact linear ceiling — no optimiser, no underfit confound.
            res: DistillResult = fit_closed_form_linear(
                layer, X, Y, val_frac=cfg.val_frac, device=cfg.device,
            )
        else:
            res = fit_layer(
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
            val_rmse=res.val_rmse,
            val_cosine=res.val_cosine,
            recruit_start=recruit_start,
            recruit_final=recruit_final,
            conjunction=_conjunction_for_layer(layer, X, cfg),
            recruit_curve=res.recruit_curve,
            layer=layer,
        )
        block.candidates[name] = cand
        gate = (
            f"  gate {recruit_start:.4f}->{recruit_final:.4f} "
            f"(d{cand.recruit_delta:+.4f})" if recruit_final is not None else ""
        )
        log(f"  {name:<12} params={res.num_params:>9,}  "
            f"compress x{cand.compression:5.1f}  rel_mse={res.val_rel_mse:.4f}  "
            f"R2={res.val_r2:.4f}  cos={res.val_cosine:.4f}  rmse={res.val_rmse:.4f}  "
            f"AND={cand.conjunction:.3f}{gate}")
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

    if cfg.eval_perplexity:
        # Re-insert each fitted layer into the live model and measure held-out PPL.
        # The distillation batches double as heal batches (train-split text).
        _evaluate_perplexity(model, tokenizer, results, cfg, heal_batches=batches)

    _print_summary_table(results)
    _save_results(cfg, results)
    if make_plots:
        _make_plots(cfg, results)
    print("\nDone.")
    return results


def _print_summary_table(results: List[BlockResult], log: Callable[[str], None] = print) -> None:
    """One compact ASCII table: layer type x (params, compression, fidelity, PPL).

    Columns: parameters, compression vs the original MLP, held-out R² / cosine /
    RMSE, and — when perplexity was evaluated — the zero-shot swap ΔPPL and the
    post-heal ΔPPL. Grouped by block depth so the early/deep contrast is legible.
    ASCII only (the Windows cp1252 console can't encode ²/Δ).
    """
    has_ppl = any(c.dppl_swap is not None for b in results for c in b.candidates.values())
    has_heal = any(c.dppl_heal is not None for b in results for c in b.candidates.values())
    header = (
        f"  {'block':<12} {'layer':<12} {'params':>10} {'compr':>7} "
        f"{'R2':>8} {'cosine':>8} {'rmse':>9}"
    )
    if has_ppl:
        header += f" {'dPPL':>9}"
    if has_heal:
        header += f" {'dPPL_heal':>10}"
    log("\n" + "=" * len(header))
    log("SUMMARY  (held-out fit; lower rel/rmse/dPPL = better, higher R2/cosine = better)")
    log("=" * len(header))
    log(header)
    log("  " + "-" * (len(header) - 2))
    for b in results:
        for name, c in b.candidates.items():
            row = (
                f"  {b.label:<12} {name:<12} {c.num_params:>10,} "
                f"x{c.compression:>5.1f} {c.val_r2:>8.4f} {c.val_cosine:>8.4f} "
                f"{c.val_rmse:>9.4f}"
            )
            if has_ppl:
                row += f" {c.dppl_swap:>+9.3f}" if c.dppl_swap is not None else f" {'-':>9}"
            if has_heal:
                row += f" {c.dppl_heal:>+10.3f}" if c.dppl_heal is not None else f" {'-':>10}"
            log(row)
        # Reference row: the original block's MLP given the same heal budget — the
        # fair baseline a candidate's healed ΔPPL should be read against.
        if has_heal and b.dppl_heal_original is not None:
            row = (
                f"  {b.label:<12} {'ORIG(heal)':<12} {b.mlp_params:>10,} "
                f"x{1.0:>5.1f} {'-':>8} {'-':>8} {'-':>9}"
            )
            if has_ppl:
                row += f" {'-':>9}"
            row += f" {b.dppl_heal_original:>+10.3f}"
            log(row)
    log("=" * len(header) + "\n")


def _save_results(cfg: Config, results: List[BlockResult]) -> None:
    path = Path(cfg.results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "label": b.label,
            "block_index": b.block_index,
            "mlp_params": b.mlp_params,
            "num_rows": b.num_rows,
            "ppl_base": b.ppl_base,
            "ppl_heal_original": b.ppl_heal_original,
            "dppl_heal_original": b.dppl_heal_original,
            "candidates": {
                name: {
                    "num_params": c.num_params,
                    "compression": c.compression,
                    "val_rel_mse": c.val_rel_mse,
                    "val_r2": c.val_r2,
                    "val_rmse": c.val_rmse,
                    "val_cosine": c.val_cosine,
                    "recruit_start": c.recruit_start,
                    "recruit_final": c.recruit_final,
                    "recruit_delta": c.recruit_delta,
                    "conjunction": c.conjunction,
                    "ppl_base": c.ppl_base,
                    "ppl_swap": c.ppl_swap,
                    "ppl_heal": c.ppl_heal,
                    "dppl_swap": c.dppl_swap,
                    "dppl_heal": c.dppl_heal,
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
    # End-to-end perplexity increase from the swap (only if PPL was evaluated).
    if any(c.dppl_swap is not None for b in results for c in b.candidates.values()):
        dppl = {
            b.label: {
                n: (c.dppl_swap or 0.0) for n, c in b.candidates.items()
                if c.dppl_swap is not None
            }
            for b in results
        }
        plot_grouped_bars(
            dppl, name=f"{cfg.plot_prefix}_dppl",
            title="Perplexity increase from MLP-block swap (zero-shot, lower=better)",
            ylabel="Δ perplexity", xlabel="GPT-2 block",
        )
        if any(c.dppl_heal is not None for b in results for c in b.candidates.values()):
            dppl_h = {
                b.label: {
                    n: (c.dppl_heal or 0.0) for n, c in b.candidates.items()
                    if c.dppl_heal is not None
                }
                for b in results
            }
            plot_grouped_bars(
                dppl_h, name=f"{cfg.plot_prefix}_dppl_heal",
                title="Perplexity increase after healing the swapped layer",
                ylabel="Δ perplexity", xlabel="GPT-2 block",
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
