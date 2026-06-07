"""Runnable paper experiments, re-implemented on the PolyWeave library.

Each module is a self-contained script with a ``Config`` dataclass and a
``run(cfg)`` entry point, wiring library components (students, target specs,
teachers, the generic training loop, and the evaluation primitives) together
with the experiment-specific scaffolding in :mod:`polyweave.experiments._common`.

    cifar_fc            — Experiment 1: linear-head (FC) generation on CIFAR-10.
    cifar_conv1         — Experiment 2: first-conv-filter generation on CIFAR-10.
    synthetic_attention — Experiment 3: query/key projection generation.

These are not imported by ``polyweave/__init__.py``; import the specific module
you want to run.
"""

from __future__ import annotations

__all__ = ["cifar_fc", "cifar_conv1", "synthetic_attention"]
