"""Offline smoke test for the GPT-2 MLP-block distillation experiment.

``transformers`` is an optional dependency and downloading GPT-2 is too heavy for
CI, so we monkeypatch ``load_model`` / ``token_batches`` with a tiny stub model
whose blocks expose a real ``.mlp`` (a two-layer GELU MLP, exactly the shape the
experiment distils). The test asserts the pipeline runs end to end on a CPU and
writes its figures + results JSON; it does not check fit quality (that needs the
real network and the full regime).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from polyweave.experiments import gpt2_mlp_distill as exp


class _StubMLP(nn.Module):
    """Mimics GPT2MLP: position-wise [.., d] -> [.., d] with a GELU bottleneck."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.c_fc = nn.Linear(d, 4 * d)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(4 * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))


class _StubBlock(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.mlp = _StubMLP(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class _StubTransformer(nn.Module):
    def __init__(self, d: int, n_layer: int, vocab: int) -> None:
        super().__init__()
        self.wte = nn.Embedding(vocab, d)
        self.h = nn.ModuleList([_StubBlock(d) for _ in range(n_layer)])


class _StubModel(nn.Module):
    """Tiny GPT-2-shaped stub: model.transformer.h[i].mlp, callable on input_ids."""

    def __init__(self, d: int = 16, n_layer: int = 4, vocab: int = 64) -> None:
        super().__init__()
        self.transformer = _StubTransformer(d, n_layer, vocab)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.transformer.wte(input_ids)
        for block in self.transformer.h:
            h = block(h)  # drives each .mlp forward hook
        return h


def test_gpt2_mlp_distill_runs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(exp, "load_model", lambda cfg: (_StubModel(), None))
    monkeypatch.setattr(
        exp, "token_batches",
        lambda cfg, tok: [torch.randint(0, 64, (cfg.batch_size, cfg.seq_len)) for _ in range(3)],
    )

    cfg = exp.Config(
        device="cpu",
        block_indices=(0, 3),
        block_labels=("early block", "deep block"),
        seq_len=8,
        batch_size=4,
        max_tokens=200,
        poly_rank=2,
        equal_budget=True,
        steps=4,
        eval_every=2,
        fit_batch_size=16,
        occlusion_rows=8,
    )
    results = exp.run(cfg)

    assert len(results) == 2
    for block in results:
        # All four candidates fitted (dense, sigma-pi, poly, dense (2x)).
        assert set(block.candidates) == {"dense", "sigma-pi", "poly", "dense (2x)"}
        # Gated layers expose a recruitment curve; the plain dense layer does not.
        assert block.candidates["sigma-pi"].recruit_delta is not None
        assert block.candidates["dense"].recruit_delta is None
        # Compression is reported relative to the original MLP.
        assert block.candidates["dense"].compression > 0

    assert (tmp_path / "plots" / f"{cfg.plot_prefix}_r2.pdf").exists()
    assert (tmp_path / "plots" / f"{cfg.plot_prefix}_conjunction.pdf").exists()
    assert (tmp_path / "plots" / "raw" / "gpt2_mlp_distill.json").exists()


class _LMOutput:
    """Minimal stand-in for a HF ``CausalLMOutput`` (just the ``.loss`` field)."""

    def __init__(self, loss: torch.Tensor) -> None:
        self.loss = loss


class _StubLM(_StubModel):
    """``_StubModel`` plus an LM head, so ``model(ids, labels=ids).loss`` works —
    enough to exercise the end-to-end perplexity swap/heal path offline."""

    def __init__(self, d: int = 16, n_layer: int = 4, vocab: int = 64) -> None:
        super().__init__(d, n_layer, vocab)
        self.lm_head = nn.Linear(d, vocab)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        h = self.transformer.wte(input_ids)
        for block in self.transformer.h:
            h = block(h)
        logits = self.lm_head(h)
        if labels is None:
            return logits
        shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
        shift_labels = labels[:, 1:].reshape(-1)
        loss = nn.functional.cross_entropy(shift_logits, shift_labels)
        return _LMOutput(loss)


def test_gpt2_mlp_distill_perplexity(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    model = _StubLM()
    monkeypatch.setattr(exp, "load_model", lambda cfg: (model, None))
    monkeypatch.setattr(
        exp, "token_batches",
        lambda cfg, tok, split="train": [
            torch.randint(0, 64, (cfg.batch_size, cfg.seq_len)) for _ in range(3)
        ],
    )

    cfg = exp.Config(
        device="cpu", block_indices=(0, 3), seq_len=8, batch_size=4,
        max_tokens=200, poly_rank=2, equal_budget=True, steps=4, eval_every=2,
        fit_batch_size=16, occlusion_rows=8,
        eval_perplexity=True, ppl_max_batches=2, heal_steps=2, heal_lr=1e-4,
    )
    orig_mlps = [exp.mlp_of(model, i) for i in cfg.block_indices]
    results = exp.run(cfg)

    for block in results:
        for cand in block.candidates.values():
            assert cand.ppl_base is not None and cand.ppl_base > 0
            assert cand.ppl_swap is not None
            assert cand.ppl_heal is not None          # heal_steps > 0
            assert cand.dppl_swap is not None

    # The original MLPs must be restored after every swap (finally-block contract).
    for i, orig in zip(cfg.block_indices, orig_mlps):
        assert exp.mlp_of(model, i) is orig
    # Healing must not have left the rest of the model with grads disabled.
    assert all(p.requires_grad for p in model.lm_head.parameters())
    assert (tmp_path / "plots" / f"{cfg.plot_prefix}_dppl.pdf").exists()


def test_mlp_of_resolves_both_layouts():
    model = _StubModel(d=8, n_layer=2)
    assert exp.mlp_of(model, 0) is model.transformer.h[0].mlp

    # Flatter layout: model.h[i].mlp (no .transformer wrapper).
    class _Flat(nn.Module):
        def __init__(self):
            super().__init__()
            self.h = nn.ModuleList([_StubBlock(8)])

    flat = _Flat()
    assert exp.mlp_of(flat, 0) is flat.h[0].mlp
