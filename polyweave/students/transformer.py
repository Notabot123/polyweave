"""Tiny attention-only transformer student for Q/K weight generation.

Used in Experiment 3. The student solves a synthetic relational-lookup task; a
teacher generates the query/key projection weights for *every* attention layer
(value/output projections and the classifier are the student's own, frozen).

Token and positional embeddings are deterministic (seeded by ``emb_seed``) and
frozen so all students share the same matching geometry; architectural diversity
comes from a per-architecture pointwise activation and from per-student
value/output/classifier initialisation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

LayerQK = Dict[str, torch.Tensor]


def make_activation(name: str) -> nn.Module:
    table = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "leaky_relu": lambda: nn.LeakyReLU(0.1),
        "swish": nn.SiLU,
    }
    if name not in table:
        raise ValueError(f"unknown activation {name!r}")
    return table[name]()


def multihead_attention_with_generated_qk(
    attn: nn.MultiheadAttention, x: torch.Tensor, gen_qk: LayerQK
) -> torch.Tensor:
    """Manual MHA forward using generated Wq/bq/Wk/bk and the module's V/O."""
    B, L, D = x.shape
    H = attn.num_heads
    Dh = D // H
    assert D % H == 0, "d_model must be divisible by n_heads"

    W = attn.in_proj_weight
    b = attn.in_proj_bias
    Wv = W[2 * D : 3 * D]
    bv = b[2 * D : 3 * D] if b is not None else None

    q = F.linear(x, gen_qk["q_weight"], gen_qk["q_bias"])
    k = F.linear(x, gen_qk["k_weight"], gen_qk["k_bias"])
    v = F.linear(x, Wv, bv)

    def split_heads(t: torch.Tensor) -> torch.Tensor:
        return t.view(B, L, H, Dh).transpose(1, 2)

    qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(Dh)
    weights = F.softmax(scores, dim=-1)
    out = torch.matmul(weights, vh).transpose(1, 2).contiguous().view(B, L, D)
    return attn.out_proj(out)


class TinySelfAttentionBlock(nn.Module):
    """Attention-only block (no feed-forward) with a pointwise activation."""

    def __init__(self, d_model: int, n_heads: int, activation: str, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.act = make_activation(activation)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, gen_qk: Optional[LayerQK] = None) -> torch.Tensor:
        if gen_qk is None:
            h, _ = self.attn(x, x, x, need_weights=False)
        else:
            h = multihead_attention_with_generated_qk(self.attn, x, gen_qk)
        return self.ln(x + self.act(h))


class TinyTransformerStudent(nn.Module):
    """Attention-only transformer with shared frozen embeddings.

    Args:
        vocab_size: token vocabulary size.
        seq_len: sequence length (query token is at the last position).
        num_classes: number of key slots / output classes.
        d_model: model width (uniform across students so the generated Q/K target
            is fixed-size).
        n_heads: number of attention heads.
        n_layers: number of transformer blocks.
        activation: per-architecture pointwise activation in each block.
        emb_seed: seed for the shared, frozen token/positional embeddings.
        dropout: attention dropout (default 0).
    """

    def __init__(
        self,
        vocab_size: int = 64,
        seq_len: int = 10,
        num_classes: int = 5,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        activation: str = "relu",
        emb_seed: int = 7,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.activation = activation

        self.token_emb = nn.Embedding(vocab_size, d_model)
        gen = torch.Generator().manual_seed(emb_seed)
        with torch.no_grad():
            self.token_emb.weight.copy_(torch.randn(vocab_size, d_model, generator=gen) * 0.02)
            pos = torch.randn(1, seq_len, d_model, generator=gen) * 0.02
        self.token_emb.weight.requires_grad_(False)
        self.register_buffer("pos_emb", pos)

        self.blocks = nn.ModuleList(
            [TinySelfAttentionBlock(d_model, n_heads, activation, dropout) for _ in range(n_layers)]
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_emb(tokens) + self.pos_emb[:, : tokens.size(1)]

    def forward(
        self, tokens: torch.Tensor, gen_qk: Optional[List[LayerQK]] = None
    ) -> torch.Tensor:
        x = self.embed(tokens)
        for i, block in enumerate(self.blocks):
            block_qk = gen_qk[i] if gen_qk is not None else None
            x = block(x, gen_qk=block_qk)
        pooled = x[:, -1]  # query is always at the last position
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# Q/K parameter helpers (operate over every attention layer)
# ---------------------------------------------------------------------------

def attn_layers(model: TinyTransformerStudent) -> List[nn.MultiheadAttention]:
    return [block.attn for block in model.blocks]


def qk_params(model: TinyTransformerStudent) -> List[nn.Parameter]:
    ps: List[nn.Parameter] = []
    for attn in attn_layers(model):
        ps.append(attn.in_proj_weight)
        ps.append(attn.in_proj_bias)
    return ps


def freeze_except_qk(model: TinyTransformerStudent) -> None:
    """Freeze everything except every block's packed in_proj (Q/K/V)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for attn in attn_layers(model):
        attn.in_proj_weight.requires_grad_(True)
        attn.in_proj_bias.requires_grad_(True)


def mask_qk_grads(model: TinyTransformerStudent) -> None:
    """Zero the V slice of every block's packed projection gradient.

    Lets an optimiser step the Q/K rows while leaving the (frozen) value
    projection untouched, even though Q/K/V share one packed parameter.
    """
    for attn in attn_layers(model):
        D = attn.embed_dim
        if attn.in_proj_weight.grad is not None:
            attn.in_proj_weight.grad[2 * D : 3 * D].zero_()
        if attn.in_proj_bias is not None and attn.in_proj_bias.grad is not None:
            attn.in_proj_bias.grad[2 * D : 3 * D].zero_()


@torch.no_grad()
def reinit_qk(model: TinyTransformerStudent) -> None:
    """Xavier-reinitialise the Q and K slices of every block (V left as-is)."""
    for attn in attn_layers(model):
        D = attn.embed_dim
        nn.init.xavier_uniform_(attn.in_proj_weight[:D])
        nn.init.xavier_uniform_(attn.in_proj_weight[D : 2 * D])
        attn.in_proj_bias[: 2 * D].zero_()
