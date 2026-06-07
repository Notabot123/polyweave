"""Ensemble experiment — additive vs Sigma-Pi teacher-generated student populations.

A warm-restarted student population is *diverse by construction*: members share an
architecture but reach their target layer by different routes. Here we ask whether
the teacher that generated their ``conv1`` filters affects that diversity — and
whether it buys a better ensemble.

For each teacher (additive ``conv`` and multiplicative ``conv_sigmapi``), loaded
from the ``models/seed{N}/conv1_models.pt`` payload (no retraining), we:

1. generate + install ``conv1`` for every member of the population, re-estimating
   bn1 each time (the zero-shot setting of §4.2);
2. collect each member's softmax over a fixed test set;
3. compare mean single-member accuracy, the soft-vote *ensemble* accuracy, the
   ensemble gain, and the pairwise prediction diversity of the two populations.

The headline figures are a single-vs-ensemble accuracy bar chart and an overlaid
histogram of pairwise member disagreement (the diversity distribution).

Prerequisite: this script does no training — it loads the teachers + student
population from ``models/seed{N}/conv1_models.pt``. That payload is *not* shipped
with the repo (it is large and regenerable); produce it first by running the
conv1 experiment with model-saving enabled::

    python -m polyweave.experiments.multiseed          # saves seed-42 models

(``--save-models-dir`` defaults to ``models`` and the conv1 models are written for
the first seed). For a single seed only, call ``cifar_conv1.run`` with
``Config(save_models_dir="models")``.

Run:  python -m polyweave.experiments.ensemble --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..evaluation import (
    disagreement_matrix,
    ensemble_accuracy,
    ensemble_gain,
    generate_averaged,
    member_accuracies,
    pairwise_disagreement,
)
from ..prototypes import image_grid_stats
from ..students import make_cnn_student
from ..utils import set_seed
from ..viz import plot_diversity_hist, plot_ensemble_bars
from . import _common
from .cifar_conv1 import ARCHS, CONV1_IN, CONV1_KERNEL, CONV1_OUT
from .student_occlusion import _install_conv1, _rebuild_teachers

PLOT_PREFIX = "polyweave_cifar_conv1_ensemble"


# ---------------------------------------------------------------------------
# Rebuild a saved student population (arch per index from the warm-restart scheme)
# ---------------------------------------------------------------------------

def _student_archs(payload: dict, which: str) -> List[str]:
    """Architecture label for each saved student in ``which`` ∈ {seen, unseen}."""
    cfg, arch = payload["cfg"], payload["arch"]
    wr = cfg["warm_restarts"]
    n_train = arch["num_train_groups"]
    archs = ARCHS[: cfg["num_architectures"]]
    groups = archs[:n_train] if which == "seen" else archs[n_train:]
    return [a for a in groups for _ in range(wr)]


def _rebuild_students(payload: dict, which: str, device: str) -> List[nn.Module]:
    cfg = payload["cfg"]
    states = payload[f"{which}_students"]
    archs = _student_archs(payload, which)
    students = []
    for arch, state in zip(archs, states):
        s = make_cnn_student(
            arch, feature_dim=cfg["feature_dim"], num_classes=cfg["num_classes"],
            in_ch=CONV1_IN, conv1_out=CONV1_OUT, kernel_size=CONV1_KERNEL,
        ).to(device)
        s.load_state_dict({k: v.to(device) for k, v in state.items()})
        s.eval()
        students.append(s)
    return students


# ---------------------------------------------------------------------------
# Member probabilities over a fixed evaluation set
# ---------------------------------------------------------------------------

@torch.no_grad()
def _member_probs(model: nn.Module, eval_batches) -> torch.Tensor:
    """Concatenated softmax outputs of ``model`` over ``eval_batches`` -> ``[N, C]``."""
    model.eval()
    return torch.cat([F.softmax(model(x), dim=1) for x, _y in eval_batches], dim=0)


@torch.no_grad()
def _member_probs_gen(model: nn.Module, eval_batches, gen) -> torch.Tensor:
    """Pure zero-shot softmaxes: generated ``conv1`` via ``gen``, original bn1."""
    model.eval()
    return torch.cat(
        [F.softmax(model(x, gen_conv1=gen), dim=1) for x, _y in eval_batches], dim=0
    )


def population_probs(
    students: List[nn.Module],
    teacher: nn.Module,
    support_batches,
    eval_batches,
    *,
    num_classes: int,
    proto_grid: int,
    bn_reset_batches: int,
    bn_reset: bool = False,
) -> torch.Tensor:
    """Stack of per-member softmaxes for ``teacher``-generated conv1 -> ``[M, N, C]``.

    With ``bn_reset=False`` (default) this is the *pure zero-shot* protocol of
    §4.2 — the generated ``conv1`` is used through the student's original bn1, so
    the numbers are directly comparable to the paper's zero-shot table. With
    ``bn_reset=True`` the filters are installed and bn1 is re-estimated over the
    support set (the recovery-style "deployed" setting), which typically shifts
    the absolute accuracy.
    """
    def build_proto(_student, batch):
        x, y = batch
        return image_grid_stats(x, y, num_classes, grid=proto_grid)

    stacks = []
    for student in students:
        gen = generate_averaged(teacher, student, support_batches, build_proto)
        if bn_reset:
            model = _install_conv1(student, gen, support_batches, bn_reset_batches)
            stacks.append(_member_probs(model, eval_batches))
        else:
            stacks.append(_member_probs_gen(student, eval_batches, gen))
    return torch.stack(stacks, dim=0)


# ---------------------------------------------------------------------------
# Metrics bundle for one teacher
# ---------------------------------------------------------------------------

def evaluate_population(probs: torch.Tensor, labels: torch.Tensor) -> dict:
    """Single/ensemble accuracy + diversity summary for one ``[M, N, C]`` stack."""
    accs = member_accuracies(probs, labels)
    dmat = disagreement_matrix(probs)
    M = dmat.shape[0]
    iu = torch.triu_indices(M, M, offset=1)
    pair_vals = dmat[iu[0], iu[1]].tolist()
    return {
        "member_accs": accs.tolist(),
        "single_mean": float(accs.mean()),
        "single_std": float(accs.std(unbiased=True)) if M > 1 else 0.0,
        "ensemble_acc": ensemble_accuracy(probs, labels),
        "ensemble_gain": ensemble_gain(probs, labels),
        "mean_disagreement": pairwise_disagreement(probs),
        "pairwise_disagreements": pair_vals,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows cp1252 safety
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--population", choices=["seen", "unseen"], default="seen",
                    help="seen students have working zero-shot; unseen are near chance.")
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--bn-reset", action="store_true",
                    help="re-estimate bn1 after install (recovery-style); default "
                         "is pure zero-shot to match the §4.2 table.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)  # deterministic support-batch shuffling -> reproducible numbers
    path = f"{args.models_dir}/seed{args.seed}/conv1_models.pt"
    print(f"loading {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload["cfg"]

    teachers = _rebuild_teachers(payload, device)
    students = _rebuild_students(payload, args.population, device)
    print(f"{args.population} population: {len(students)} students "
          f"({'+'.join(_student_archs(payload, args.population))})")

    _, test_loader = _common.cifar10_loaders(cfg["batch_size"])
    train_loader, _ = _common.cifar10_loaders(cfg["batch_size"])
    support = _common.collect_batches(train_loader, cfg["eval_support_batches"], device)
    eval_batches = _common.collect_batches(test_loader, args.eval_batches, device)
    labels = torch.cat([y for _x, y in eval_batches], dim=0).to(device)

    results: Dict[str, dict] = {}
    for kind, teacher in teachers.items():
        probs = population_probs(
            students, teacher, support, eval_batches,
            num_classes=cfg["num_classes"], proto_grid=cfg["proto_grid"],
            bn_reset_batches=cfg.get("bn_reset_batches", 10), bn_reset=args.bn_reset,
        )
        results[kind] = evaluate_population(probs, labels)
        r = results[kind]
        print(f"\n[{kind}] single {r['single_mean']:.4f}±{r['single_std']:.4f}  "
              f"ensemble {r['ensemble_acc']:.4f}  gain {r['ensemble_gain']:+.4f}  "
              f"diversity {r['mean_disagreement']:.4f}")

    _write_outputs(args, results)


def _write_outputs(args, results: Dict[str, dict]) -> None:
    _common.configure_plots(False)
    pretty = {k: _common.method_label(k) for k in results}
    plot_ensemble_bars(
        single_means={pretty[k]: results[k]["single_mean"] for k in results},
        ensemble={pretty[k]: results[k]["ensemble_acc"] for k in results},
        single_stds={pretty[k]: results[k]["single_std"] for k in results},
        name=f"{PLOT_PREFIX}_accuracy_seed{args.seed}",
        title=f"Single vs ensemble accuracy ({args.population} architectures)",
    )
    plot_diversity_hist(
        {pretty[k]: results[k]["pairwise_disagreements"] for k in results},
        name=f"{PLOT_PREFIX}_diversity_seed{args.seed}",
        title=f"Ensemble member diversity ({args.population} architectures)",
    )
    out = Path("plots") / f"{PLOT_PREFIX}_results_seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    protocol = "bn_reset" if args.bn_reset else "zero_shot"
    out.write_text(json.dumps({"seed": args.seed, "population": args.population,
                               "protocol": protocol, "results": results}, indent=2))
    print(f"\nsaved {out}")
    print("Done.")


if __name__ == "__main__":
    main()
