"""Runnable paper experiments, re-implemented on the PolyWeave library.

Each module is a self-contained script with a ``Config`` dataclass and a
``run(cfg)`` entry point, wiring library components (students, target specs,
teachers, the generic training loop, and the evaluation primitives) together
with the experiment-specific scaffolding in :mod:`polyweave.experiments._common`.

    cifar_fc            — Experiment 1: linear-head (FC) generation on CIFAR-10.
    cifar_conv1         — Experiment 2: first-conv-filter generation on CIFAR-10.
    synthetic_attention — Experiment 3: query/key projection generation.
    gpt2_mlp_distill    — compress a GPT-2 MLP block into one position-wise layer
                          (dense vs Sigma-Pi vs poly); needs the ``distill`` extra.

These are not imported by ``polyweave/__init__.py``; import the specific module
you want to run. (``gpt2_mlp_distill`` is also kept out of this package's eager
imports so ``transformers`` stays an optional dependency.)
"""

from __future__ import annotations

__all__ = ["cifar_fc", "cifar_conv1", "synthetic_attention"]
