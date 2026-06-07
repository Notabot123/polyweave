"""Shared scaffolding for the three paper experiments.

The core library (``targets``, ``students``, ``hypernets``, ``training``,
``evaluation``) is deliberately data-agnostic. The glue that turns it into a
*specific* paper experiment — building a diverse, warm-restarted student
population, the synthetic relational-lookup task, and a couple of plotting
adapters — lives here rather than in the library proper.

Nothing in this module is imported by ``polyweave/__init__.py``; the experiment
scripts import it explicitly. It is exercised by the smoke tests, not shipped as
public API.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..utils import freeze_all, set_seed
from ..viz import plots as _plots

# Friendly, paper-ready labels for the internal method keys used by the
# experiment scripts. Applied at plot time so figures are self-explanatory.
METHOD_LABELS: Dict[str, str] = {
    "random": "random init",
    "ncc": "NCC baseline",
    "conv": "additive teacher",
    "conv_sigmapi": r"$\Sigma\Pi$ teacher",
}


def method_label(key: str) -> str:
    """Map an internal method key to a human-readable plot label."""
    return METHOD_LABELS.get(key, key)

# ---------------------------------------------------------------------------
# Freeze helpers
# ---------------------------------------------------------------------------

def freeze_except(model: torch.nn.Module, submodule_name: str) -> None:
    """Freeze every parameter of ``model`` except those of ``model.<name>``."""
    freeze_all(model)
    for p in getattr(model, submodule_name).parameters():
        p.requires_grad_(True)


# ---------------------------------------------------------------------------
# CIFAR-10 data (shared by the two CIFAR experiments)
# ---------------------------------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def cifar10_loaders(batch_size: int = 128, root: str = "./data", augment: bool = True):
    """Return ``(train_loader, test_loader)`` for CIFAR-10.

    ``torchvision`` is imported lazily so the rest of the library (and the smoke
    tests, which fabricate tiny tensor datasets) does not depend on it.
    """
    from torchvision import datasets, transforms

    test_tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )
    if augment:
        train_tf = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
    else:
        train_tf = test_tf

    train_ds = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)
    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, drop_last=True, **kw)
    test_loader = torch.utils.data.DataLoader(test_ds, shuffle=False, **kw)
    return train_loader, test_loader


def collect_batches(loader, n: int, device: str = "cpu") -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Pull ``n`` batches off ``loader`` (cycling if needed), moved to ``device``."""
    it = iter(loader)
    out: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(n):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        out.append((x.to(device), y.to(device)))
    return out


# ---------------------------------------------------------------------------
# Warm-restart student population
# ---------------------------------------------------------------------------

BuildBase = Callable[[Any], torch.nn.Module]
InPlace = Callable[[torch.nn.Module], None]


def build_student_groups(
    archs: Sequence[Any],
    *,
    build_base: BuildBase,
    full_train: InPlace,
    freeze_trunk: InPlace,
    reinit_target: InPlace,
    finetune_target: InPlace,
    warm_restarts: int,
    seed: int = 42,
    log_fn: Callable[[str], None] = print,
) -> List[List[torch.nn.Module]]:
    """Build one *group* of diverse students per architecture.

    Each group shares a fully-trained trunk (built once via ``build_base`` then
    ``full_train``) and differs only in the target layer: after freezing the
    trunk, the target is re-initialised and briefly fine-tuned ``warm_restarts``
    times from independent seeds, yielding a population of students that all
    solve the task but reach the target weights by different routes. This is the
    diversity the teacher must generalise across.

    The seed scheme is ``seed + 1000 * arch_idx`` for the base and
    ``seed + 1000 * arch_idx + ri + 1`` for warm restart ``ri`` (matching the
    original experiment scripts so populations are reproducible).

    Args:
        archs: per-architecture identifiers passed to ``build_base`` (their order
            defines the seen/unseen split downstream).
        build_base: ``arch -> student`` (a fresh, untrained model).
        full_train: trains a base student in place.
        freeze_trunk: freezes everything except the generated target layer.
        reinit_target: re-initialises the target layer of a student in place.
        finetune_target: warm-restart fine-tunes the (only trainable) target.
        warm_restarts: number of warm restarts per architecture.
        seed: base seed for the deterministic per-student seed scheme.
        log_fn: sink for progress lines (``print`` by default; pass a no-op to
            silence during smoke tests).

    Returns:
        A list of ``len(archs)`` groups, each a list of ``warm_restarts``
        frozen, ``eval``-mode students.
    """
    groups: List[List[torch.nn.Module]] = []
    for arch_idx, arch in enumerate(archs):
        log_fn(f"=== architecture group {arch_idx + 1}/{len(archs)}: {arch} ===")
        set_seed(seed + 1000 * arch_idx)
        base = build_base(arch)
        full_train(base)
        freeze_trunk(base)

        group: List[torch.nn.Module] = []
        for ri in range(warm_restarts):
            log_fn(f"  warm restart {ri + 1}/{warm_restarts}")
            set_seed(seed + 1000 * arch_idx + ri + 1)
            student = copy.deepcopy(base)
            reinit_target(student)
            finetune_target(student)
            freeze_all(student)
            student.eval()
            group.append(student)
        groups.append(group)
    return groups


def flatten_groups(groups: Sequence[Sequence[torch.nn.Module]]) -> List[torch.nn.Module]:
    """Flatten ``[[s, ...], ...]`` into a single list of students."""
    return [s for group in groups for s in group]


def split_seen_unseen(
    groups: Sequence[Sequence[torch.nn.Module]], num_seen_groups: int
) -> Tuple[List[torch.nn.Module], List[torch.nn.Module]]:
    """Split architecture groups into flat *seen* and *unseen* student lists.

    The first ``num_seen_groups`` architectures train the teacher; the rest are
    held out to measure zero-shot transfer to unseen architectures.
    """
    seen = flatten_groups(groups[:num_seen_groups])
    unseen = flatten_groups(groups[num_seen_groups:])
    return seen, unseen


# ---------------------------------------------------------------------------
# Synthetic relational-lookup task (Experiment 3)
# ---------------------------------------------------------------------------

def sample_relation(vocab_size: int, device: str = "cpu") -> torch.Tensor:
    """A random relation ``pi: vocab -> vocab`` (a permutation), one per episode."""
    return torch.randperm(vocab_size, device=device)


def identity_relation(vocab_size: int, device: str = "cpu") -> torch.Tensor:
    """The identity relation used for base training and warm restarts."""
    return torch.arange(vocab_size, device=device)


def make_relational_batch(
    relation: torch.Tensor,
    *,
    batch_size: int,
    vocab_size: int,
    num_key_slots: int,
    seq_len: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Relational key-lookup batch.

    Sequence layout (``K = num_key_slots``)::

        positions 0..K-1:  key tokens
        positions K..L-2:  noise / distractors
        position  L-1:     query token q

    The "matched" key is the slot whose token equals ``pi(q)``; its slot index
    ``j`` is the label. The matched token ``pi(q)`` is placed at exactly one slot
    (``j``) and removed from every other position, so the label is unambiguous.

    Returns:
        ``(seq, label)`` with ``seq`` shaped ``[B, seq_len]`` and ``label`` in
        ``[0, num_key_slots)``.
    """
    B, V, K, L = batch_size, vocab_size, num_key_slots, seq_len
    assert L >= K + 1, f"seq_len must be >= {K + 1}"

    q = torch.randint(0, V, (B,), device=device)
    target = relation[q]  # pi(q): the token the query matches
    j = torch.randint(0, K, (B,), device=device)

    seq = torch.randint(0, V, (B, L), device=device)
    # Remove the target from every position (replace collisions with a
    # guaranteed non-target token).
    collide = seq == target.unsqueeze(1)
    repl = (target.unsqueeze(1) + torch.randint(1, V, (B, L), device=device)) % V
    seq = torch.where(collide, repl, seq)

    # Place the matched token at slot j and the query at the last position.
    seq[torch.arange(B, device=device), j] = target
    seq[:, -1] = q
    return seq, j


# ---------------------------------------------------------------------------
# Plotting adapters (thin wrappers over polyweave.viz.plots)
# ---------------------------------------------------------------------------

def curve_accuracies(curve: Sequence[Tuple[int, float]]) -> List[float]:
    """Drop the step index from a ``[(step, acc), ...]`` curve, keeping accuracy."""
    return [acc for _step, acc in curve]


def plot_recovery_curves(
    curves: dict,
    *,
    name: str,
    title: str = "Recovery after zero-shot init",
    ylabel: str = "accuracy",
    plots_dir=_plots.DEFAULT_PLOTS_DIR,
) -> List:
    """Plot per-method recovery curves.

    ``curves`` maps a method label to either a ``[(step, acc), ...]`` curve (as
    returned by :func:`polyweave.evaluation.recovery_curve` /
    :func:`mean_curves`) or a bare list of accuracies.
    """
    data = {}
    for label, curve in curves.items():
        friendly = method_label(label)
        if curve and isinstance(curve[0], tuple):
            data[friendly] = curve_accuracies(curve)
        else:
            data[friendly] = list(curve)
    return _plots.plot_lines(
        data, title=title, ylabel=ylabel, name=name, plots_dir=plots_dir,
        xlabel="fine-tuning step",
    )


# Re-export the most-used plotting entry points so experiment scripts can pull
# everything they need from this one module.
configure_plots = _plots.configure_plots
plot_lines = _plots.plot_lines
plot_zeroshot_bar = _plots.plot_zeroshot_bar
save_figure = _plots.save_figure


def silent(_msg: str) -> None:
    """A no-op ``log_fn`` for quiet (e.g. smoke-test) runs."""


# ---------------------------------------------------------------------------
# Structured per-run result + multi-seed aggregation
# ---------------------------------------------------------------------------

Curve = List[Tuple[int, float]]


@dataclass
class RunResult:
    """Everything a single ``run(cfg)`` produces that we want to aggregate.

    ``pi_start`` / ``pi_final`` are the ``exp(pi_scale).mean()`` diagnostic of
    the *Sigma-Pi* teacher at the first and last training step (``None`` for a
    run with no Sigma-Pi teacher). ``recovery`` maps each method to a mean
    ``[(step, acc), ...]`` curve already averaged over the student population.
    """

    seed: int
    label: str  # short experiment label, e.g. "FC", "conv1", "Q/K"
    losses: Dict[str, List[float]] = field(default_factory=dict)
    seen_means: Dict[str, float] = field(default_factory=dict)
    unseen_means: Dict[str, float] = field(default_factory=dict)
    recovery: Dict[str, Curve] = field(default_factory=dict)
    pi_start: Optional[float] = None
    pi_final: Optional[float] = None

    @property
    def pi_delta(self) -> Optional[float]:
        if self.pi_start is None or self.pi_final is None:
            return None
        return self.pi_final - self.pi_start

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "label": self.label,
            "losses": self.losses,
            "seen_means": self.seen_means,
            "unseen_means": self.unseen_means,
            "recovery": {m: [[int(s), float(a)] for s, a in c] for m, c in self.recovery.items()},
            "pi_start": self.pi_start,
            "pi_final": self.pi_final,
            "pi_delta": self.pi_delta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunResult":
        """Reconstruct a RunResult from :meth:`to_dict` output (for resume)."""
        recovery = {
            m: [(int(s), float(a)) for s, a in c] for m, c in d["recovery"].items()
        }
        return cls(
            seed=d["seed"], label=d["label"], losses=d.get("losses", {}),
            seen_means=d["seen_means"], unseen_means=d["unseen_means"],
            recovery=recovery, pi_start=d.get("pi_start"), pi_final=d.get("pi_final"),
        )


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), 0.0
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)  # sample std
    return m, math.sqrt(var)


def aggregate_pi_delta(results: Sequence[RunResult]) -> Tuple[float, float]:
    """Mean and sample-std of the Sigma-Pi Δpi across seeds."""
    deltas = [r.pi_delta for r in results if r.pi_delta is not None]
    return _mean_std(deltas)


def aggregate_zeroshot(
    results: Sequence[RunResult],
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    """Per-method (mean, std) zero-shot accuracy for seen and unseen students."""
    methods = list(results[0].seen_means.keys())
    seen = {m: _mean_std([r.seen_means[m] for r in results]) for m in methods}
    unseen = {m: _mean_std([r.unseen_means[m] for r in results]) for m in methods}
    return seen, unseen


def aggregate_recovery(
    results: Sequence[RunResult],
) -> Dict[str, Tuple[List[int], List[float], List[float]]]:
    """Per-method ``(steps, mean_acc, std_acc)`` averaged over seeds.

    Curves are aligned by index; all seeds use the same recovery schedule.
    """
    methods = list(results[0].recovery.keys())
    out: Dict[str, Tuple[List[int], List[float], List[float]]] = {}
    for m in methods:
        curves = [r.recovery[m] for r in results]
        n_points = min(len(c) for c in curves)
        steps = [curves[0][i][0] for i in range(n_points)]
        means, stds = [], []
        for i in range(n_points):
            accs = [c[i][1] for c in curves]
            mu, sd = _mean_std(accs)
            means.append(mu)
            stds.append(sd)
        out[m] = (steps, means, stds)
    return out


# ---------------------------------------------------------------------------
# Multi-seed paper plots (mean line + std band / error bars)
# ---------------------------------------------------------------------------

def plot_recovery_band(
    bands: Dict[str, Tuple[Sequence[int], Sequence[float], Sequence[float]]],
    *,
    name: str,
    title: str,
    xlabel: str = "fine-tuning step",
    ylabel: str = "test accuracy",
    plots_dir=_plots.DEFAULT_PLOTS_DIR,
    figsize: Tuple[float, float] = (7.0, 4.5),
) -> List:
    """Recovery curves as mean line + shaded ±1 std band across seeds."""
    plt = _plots.plt
    fig, ax = plt.subplots(figsize=figsize)
    for i, (method, (steps, mean, std)) in enumerate(bands.items()):
        color = _plots.OKABE_ITO[i % len(_plots.OKABE_ITO)]
        steps = list(steps)
        mean = list(mean)
        std = list(std)
        ax.plot(steps, mean, label=method_label(method), color=color, marker="o", markersize=3)
        lower = [m - s for m, s in zip(mean, std)]
        upper = [m + s for m, s in zip(mean, std)]
        ax.fill_between(steps, lower, upper, color=color, alpha=0.18, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    return _plots.save_figure(fig, name, plots_dir=plots_dir)


def plot_zeroshot_grouped_std(
    seen: Dict[str, Tuple[float, float]],
    unseen: Dict[str, Tuple[float, float]],
    *,
    name: str,
    title: str = "Zero-shot accuracy: seen vs unseen architectures",
    ylabel: str = "zero-shot test accuracy",
    plots_dir=_plots.DEFAULT_PLOTS_DIR,
    figsize: Tuple[float, float] = (7.0, 4.5),
) -> List:
    """Grouped seen/unseen bars per method with ±1 std error bars across seeds."""
    plt = _plots.plt
    methods = list(seen.keys())
    labels = [method_label(m) for m in methods]
    x = list(range(len(methods)))
    w = 0.38
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        [i - w / 2 for i in x], [seen[m][0] for m in methods], w,
        yerr=[seen[m][1] for m in methods], capsize=4,
        label="seen arch", color=_plots.OKABE_ITO[0],
    )
    ax.bar(
        [i + w / 2 for i in x], [unseen[m][0] for m in methods], w,
        yerr=[unseen[m][1] for m in methods], capsize=4,
        label="unseen arch", color=_plots.OKABE_ITO[1],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return _plots.save_figure(fig, name, plots_dir=plots_dir)


def plot_pi_ordering(
    deltas: Dict[str, Tuple[float, float]],
    *,
    name: str = "polyweave_pi_ordering",
    title: str = r"$\Pi$-branch recruitment by target type",
    ylabel: str = "Change in pi-scale",
    plots_dir=_plots.DEFAULT_PLOTS_DIR,
    figsize: Tuple[float, float] = (6.0, 4.5),
) -> List:
    """Headline cross-experiment chart: Δpi (mean±std over seeds) per target.

    ``deltas`` maps an experiment label (e.g. ``"FC"``) to ``(mean, std)`` and
    is plotted in insertion order, so pass it FC → conv1 → Q/K to show the
    predicted recruitment ordering left-to-right.
    """
    plt = _plots.plt
    labels = list(deltas.keys())
    x = list(range(len(labels)))
    means = [deltas[k][0] for k in labels]
    stds = [deltas[k][1] for k in labels]
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        x, means, yerr=stds, capsize=5, width=0.6,
        color=[_plots.OKABE_ITO[i % len(_plots.OKABE_ITO)] for i in range(len(labels))],
    )
    ax.axhline(0.0, color="0.4", linewidth=1.0, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # Annotate each bar with its mean value, clear of the error-bar cap.
    for bar, m, s in zip(bars, means, stds):
        top = m + s if m >= 0 else m - s
        ax.annotate(
            f"{m:+.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, top),
            xytext=(0, 6 if m >= 0 else -14),
            textcoords="offset points", ha="center", fontsize=9,
        )
    # Headroom so the highest value label clears the top frame.
    top_extent = max((m + s for m, s in zip(means, stds)), default=0.0)
    if top_extent > 0:
        ax.set_ylim(top=top_extent * 1.18)
    fig.tight_layout()
    return _plots.save_figure(fig, name, plots_dir=plots_dir)
