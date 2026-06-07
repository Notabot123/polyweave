"""Tests for polyweave.students — CNN and tiny-transformer students."""

from __future__ import annotations

import torch

from polyweave.students import (
    TinyTransformerStudent,
    make_cnn_student,
    make_cnn_students,
    reinit_qk,
)
from polyweave.students.transformer import attn_layers


# ---------------------------------------------------------------------------
# CNN student
# ---------------------------------------------------------------------------

def test_cnn_student_forward_shape():
    s = make_cnn_student("A", num_classes=10)
    s.eval()
    out = s(torch.randn(8, 3, 32, 32))
    assert out.shape == (8, 10)


def test_cnn_student_all_archs_same_conv1_shape():
    students = make_cnn_students(["A", "B", "C"], conv1_out=32)
    shapes = {tuple(s.conv1.weight.shape) for s in students}
    assert shapes == {(32, 3, 3, 3)}


def test_cnn_student_accepts_generated_conv1():
    s = make_cnn_student("B")
    s.eval()
    x = torch.randn(4, 3, 32, 32)
    gen = {"weight": torch.randn(32, 3, 3, 3), "bias": torch.randn(32)}
    out = s(x, gen_conv1=gen)
    assert out.shape == (4, s.num_classes)


def test_cnn_student_accepts_generated_fc():
    s = make_cnn_student("A", feature_dim=256, num_classes=10)
    s.eval()
    x = torch.randn(4, 3, 32, 32)
    gen = {"weight": torch.randn(10, 256), "bias": torch.randn(10)}
    out = s(x, generated_fc=gen)
    assert out.shape == (4, 10)


# ---------------------------------------------------------------------------
# Transformer student
# ---------------------------------------------------------------------------

def test_transformer_forward_shape():
    t = TinyTransformerStudent(d_model=64, n_heads=4, n_layers=2, num_classes=5, seq_len=10)
    tokens = torch.randint(0, 64, (16, 10))
    assert t(tokens).shape == (16, 5)


def test_transformer_shared_embeddings_are_frozen_and_deterministic():
    t1 = TinyTransformerStudent(emb_seed=7)
    t2 = TinyTransformerStudent(emb_seed=7)
    assert torch.allclose(t1.token_emb.weight, t2.token_emb.weight)
    assert not t1.token_emb.weight.requires_grad


def test_transformer_generated_qk_path_runs():
    t = TinyTransformerStudent(d_model=64, n_heads=4, n_layers=2, num_classes=5, seq_len=10)
    tokens = torch.randint(0, 64, (8, 10))
    D = 64
    gen_qk = [
        {
            "q_weight": torch.randn(D, D) * 0.1,
            "q_bias": torch.zeros(D),
            "k_weight": torch.randn(D, D) * 0.1,
            "k_bias": torch.zeros(D),
        }
        for _ in range(2)
    ]
    assert t(tokens, gen_qk=gen_qk).shape == (8, 5)


def test_reinit_qk_leaves_value_slice_untouched():
    t = TinyTransformerStudent(d_model=32, n_heads=4, n_layers=2, seq_len=10)
    D = 32
    v_before = [a.in_proj_weight[2 * D : 3 * D].detach().clone() for a in attn_layers(t)]
    reinit_qk(t)
    for a, vb in zip(attn_layers(t), v_before):
        assert torch.allclose(a.in_proj_weight[2 * D : 3 * D], vb)
