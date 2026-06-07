"""Evaluation: zero-shot accuracy, recovery curves, baselines.

The read-side companion to :mod:`polyweave.training`. Given a trained teacher,
these measure the weights it generates — without fine-tuning (zero-shot) and as a
fine-tuning initialisation (recovery) — and provide the non-hypernetwork
baselines (random, nearest-class-centroid) the teachers are compared against.
"""

from __future__ import annotations

from .baselines import (
    centroids_to_fc,
    class_centroids,
    ncc_fc,
    random_like,
)
from .ensemble import (
    accuracy_from_probs,
    disagreement_matrix,
    ensemble_accuracy,
    ensemble_gain,
    ensemble_probs,
    member_accuracies,
    pairwise_disagreement,
)
from .loops import (
    average_weights,
    evaluate_accuracy,
    evaluate_macro_f1,
    generate_averaged,
    mean_curves,
    recovery_curve,
    reset_bn_stats,
)

__all__ = [
    # loops
    "average_weights",
    "evaluate_accuracy",
    "evaluate_macro_f1",
    "generate_averaged",
    "mean_curves",
    "recovery_curve",
    "reset_bn_stats",
    # baselines
    "centroids_to_fc",
    "class_centroids",
    "ncc_fc",
    "random_like",
    # ensemble
    "accuracy_from_probs",
    "disagreement_matrix",
    "ensemble_accuracy",
    "ensemble_gain",
    "ensemble_probs",
    "member_accuracies",
    "pairwise_disagreement",
]
