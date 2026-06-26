"""EEG motor imagery classification — PolyConv1d vs baselines on BCI-IV-2a.

Uses the BCI Competition IV Dataset 2a (BNCI2014_001 in MOABB) — the canonical
EEG benchmark for 4-class motor imagery (left hand / right hand / feet / tongue),
22 channels, 250 Hz, 4-second epochs, 9 subjects.

Models compared
---------------
* ``PolyConv1d``   — polynomial 1-D conv backbone (paper contribution)
* ``MLP``          — dense baseline (flattened epoch)
* ``ShallowConv``  — standard 1-D conv (same structure as PolyConv1d, rank=0)
                     isolates the polynomial branch's contribution

Protocol
--------
Within-subject evaluation: per subject, train on the training session (A0xT),
evaluate on the evaluation session (A0xE).  MOABB's paradigm handles band-pass
(4–40 Hz), epoching (0–4 s post-cue), and label encoding automatically.

Single-seed first pass (single ``--seed``).  Multi-seed wrapper is in
``run_paper1.py``.

Usage::

    python -m polyweave.experiments.eeg_bciiv2a                   # all 9 subjects
    python -m polyweave.experiments.eeg_bciiv2a --subjects 1 2 3  # subset
    python -m polyweave.experiments.eeg_bciiv2a --subjects 1 --epochs 50

Occlusion overlay
-----------------
After training, occlusion sensitivity is computed over the evaluation trials and
visualised with ``plot_occlusion_stacked`` — one row per EEG channel, with the
sensitivity heatmap behind the mean epoch trace.  Hot regions mark time intervals
and channels the model relies on most; for motor imagery these should cluster in
the 0.5–3 s post-cue window and lateralise over sensorimotor cortex (C3/C4).
"""

from __future__ import annotations

import argparse
import dataclasses
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery
    _MOABB_OK = True
except ImportError:
    _MOABB_OK = False

from polyweave.layers import PolyConv1d
from polyweave.viz.plots import configure_plots, save_figure, OKABE_ITO
from polyweave.viz.timeseries import plot_occlusion_stacked

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_CLASSES  = 4
LABEL_MAP  = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}
# Standard 10-20 electrode names for BCI-IV-2a (22 channels, no EOG)
CHANNEL_NAMES = [
    "Fz",
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1",  "Pz",  "P2",
    "POz",
]

DEFAULTS = dict(
    subjects=list(range(1, 10)),   # all 9
    seed=42,
    epochs=30,
    batch_size=64,
    lr=1e-3,
    channels=32,
    n_blocks=3,
    kernel_size=25,    # ~100 ms at 250 Hz — spans a sensorimotor burst
    rank=8,
    mlp_hidden=256,
    plots_dir=Path("plots"),
    device=None,
    fmin=4.0,
    fmax=40.0,
    tmin=0.0,
    tmax=4.0,
    occlusion_window=25,   # ~100 ms
    occlusion_stride=10,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_subject(
    subject: int,
    fmin: float = 4.0,
    fmax: float = 40.0,
    tmin: float = 0.0,
    tmax: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one subject's data via MOABB; return (X_train, y_train, X_test, y_test).

    MOABB's ``MotorImagery`` paradigm applies:
    * Band-pass filter: [fmin, fmax] Hz
    * Epoching: [tmin, tmax] s relative to the motor imagery cue onset
    * Label encoding: string class names

    Returns arrays shaped ``[trials, channels, time_samples]``.
    BCI-IV-2a has two sessions per subject; the first is used for training,
    the second for evaluation, matching the competition protocol.
    """
    paradigm = MotorImagery(
        n_classes=4, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax,
    )
    dataset = BNCI2014_001()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, y_str, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])

    y = np.array([LABEL_MAP[label] for label in y_str], dtype=np.int64)

    # Split by session: BCI-IV-2a labels sessions as '0train' and '1test'.
    sessions = metadata["session"].values
    unique   = sorted(set(sessions))
    if len(unique) >= 2:
        train_mask = sessions == unique[0]
        test_mask  = sessions == unique[1]
    else:
        # Fallback: 80/20 split if session info is missing
        n = len(y)
        idx = np.arange(n)
        np.random.default_rng(0).shuffle(idx)
        train_mask = np.zeros(n, dtype=bool); train_mask[idx[:int(0.8*n)]] = True
        test_mask  = ~train_mask

    return (
        X[train_mask].astype(np.float32),
        y[train_mask],
        X[test_mask].astype(np.float32),
        y[test_mask],
    )


def normalise(X_train: np.ndarray, X_test: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Channel-wise z-score on training set; apply same stats to test."""
    mu  = X_train.mean(axis=(0, 2), keepdims=True)   # (1, C, 1)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    return (X_train - mu) / std, (X_test - mu) / std, mu.squeeze(), std.squeeze()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PolyConv1dClassifier(nn.Module):
    """PolyConv1d temporal backbone for EEG classification.

    Input: ``(B, 22, T)`` — one epoch per sample, channels × time.
    Architecture: channel projection → N polynomial conv blocks →
    global average pool → dropout → linear head.
    """

    def __init__(
        self,
        in_channels: int,
        n_classes: int = 4,
        channels: int = 32,
        n_blocks: int = 3,
        kernel_size: int = 25,
        rank: int = 8,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.proj   = nn.Conv1d(in_channels, channels, kernel_size=1, bias=False)
        self.blocks = nn.Sequential(*[
            PolyConv1d(channels, kernel_size=kernel_size, rank=rank)
            for _ in range(n_blocks)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(p=dropout)
        self.head = nn.Linear(channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        x = self.proj(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)   # (B, channels)
        x = self.drop(x)
        return self.head(x)            # (B, n_classes)


class ShallowConvClassifier(nn.Module):
    """Standard 1-D conv classifier (PolyConv1d with rank=0 — no polynomial branch).

    Identical structure to ``PolyConv1dClassifier`` but ``rank=0`` disables the
    quadratic branch entirely, leaving a pure linear conv + BN + ReLU stack.
    This isolates the polynomial branch's contribution: any accuracy gap between
    ``ShallowConv`` and ``PolyConv1d`` is attributable solely to the quadratic term.
    """

    def __init__(
        self,
        in_channels: int,
        n_classes: int = 4,
        channels: int = 32,
        n_blocks: int = 3,
        kernel_size: int = 25,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.proj   = nn.Conv1d(in_channels, channels, kernel_size=1, bias=False)
        self.blocks = nn.Sequential(*[
            PolyConv1d(channels, kernel_size=kernel_size, rank=0)   # rank=0 = linear only
            for _ in range(n_blocks)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(p=dropout)
        self.head = nn.Linear(channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return self.head(x)


class MLPClassifier(nn.Module):
    """Dense MLP baseline: flatten epoch → hidden layers → softmax logits."""

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        n_classes: int = 4,
        hidden: int = 256,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        in_dim = in_channels * seq_len
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TrainResult:
    model: nn.Module
    train_accs: List[float]
    val_accs:   List[float]
    best_val_acc: float


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    label: str = "",
) -> TrainResult:
    model = model.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.CrossEntropyLoss()

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    Xv = torch.tensor(X_val,   dtype=torch.float32).to(device)
    yv = torch.tensor(y_val,   dtype=torch.long).to(device)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size,
                        shuffle=True, drop_last=False)

    train_accs, val_accs = [], []
    best_acc, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        correct = total = 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(Xb)
            loss   = crit(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            correct += (logits.argmax(1) == yb).sum().item()
            total   += len(yb)
        sched.step()

        train_acc = correct / total
        model.eval()
        with torch.no_grad():
            val_acc = (model(Xv).argmax(1) == yv).float().mean().item()
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs:
            print(f"  [{label}] epoch {epoch:3d}/{epochs}  "
                  f"train={train_acc:.3f}  val={val_acc:.3f}  best={best_acc:.3f}")

    model.load_state_dict(best_state)
    return TrainResult(model, train_accs, val_accs, best_acc)


# ---------------------------------------------------------------------------
# Occlusion sensitivity
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_occlusion_map(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    window: int = 25,
    stride: int = 10,
    n_samples: int = 32,
) -> np.ndarray:
    """Per-channel × per-timestep sensitivity map averaged over a trial batch.

    The response function is the log-probability of the *true* class — so a
    large positive drop means that region was helping the correct prediction.

    Returns ``[C, T]`` mean sensitivity (mean over trials).
    """
    model.eval()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    Xs  = torch.tensor(X[idx], dtype=torch.float32).to(device)  # (N, C, T)
    ys  = torch.tensor(y[idx], dtype=torch.long).to(device)     # (N,)

    C, T = Xs.shape[1], Xs.shape[2]

    # Base log-prob of the true class.
    logits_base = model(Xs)                                          # (N, 4)
    log_p_base  = torch.log_softmax(logits_base, dim=1)             # (N, 4)
    base        = log_p_base[torch.arange(len(ys)), ys].cpu().numpy()  # (N,)

    sens_map = np.zeros((C, T), dtype=np.float32)
    counts   = np.zeros((C, T), dtype=np.float32)

    steps = list(range(0, T - window + 1, stride))
    if steps and steps[-1] != T - window:
        steps.append(T - window)

    for ch in range(C):
        for s in steps:
            occ = Xs.clone()
            occ[:, ch, s:s + window] = 0.0
            log_p_occ = torch.log_softmax(model(occ), dim=1)
            drop = base - log_p_occ[torch.arange(len(ys)), ys].cpu().numpy()
            mean_drop = float(drop.mean())
            for t in range(s, min(s + window, T)):
                sens_map[ch, t] += mean_drop
                counts[ch, t]   += 1.0

    counts = np.maximum(counts, 1.0)
    return sens_map / counts


# ---------------------------------------------------------------------------
# Per-subject experiment
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SubjectResult:
    subject: int
    test_accs: Dict[str, float]
    train_results: Dict[str, TrainResult]
    n_channels: int
    seq_len: int


def run_subject(subject: int, cfg) -> SubjectResult:
    """Run the full experiment for one subject."""
    device = torch.device(
        cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(cfg.seed + subject)
    np.random.seed(cfg.seed + subject)

    print(f"\n{'='*55}")
    print(f"Subject {subject:2d}")
    print(f"{'='*55}")

    # Load + normalise
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_train, y_train, X_test, y_test = load_subject(
            subject, cfg.fmin, cfg.fmax, cfg.tmin, cfg.tmax
        )
    X_train, X_test, _, _ = normalise(X_train, X_test)
    print(f"Data loaded in {time.perf_counter()-t0:.1f}s  "
          f"train={X_train.shape}  test={X_test.shape}")

    in_channels = X_train.shape[1]
    seq_len     = X_train.shape[2]
    print(f"Channels={in_channels}  T={seq_len}")

    # Use 20 % of train as val (stratified by class)
    rng = np.random.default_rng(cfg.seed + subject)
    val_frac = 0.20
    val_idx, train_idx = [], []
    for c in range(N_CLASSES):
        ci = np.where(y_train == c)[0]
        rng.shuffle(ci)
        n_val = max(1, int(len(ci) * val_frac))
        val_idx.extend(ci[:n_val].tolist())
        train_idx.extend(ci[n_val:].tolist())
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_tr,  y_tr  = X_train[train_idx], y_train[train_idx]

    # Build models
    models: Dict[str, nn.Module] = {
        "PolyConv1d": PolyConv1dClassifier(
            in_channels, N_CLASSES, channels=cfg.channels,
            n_blocks=cfg.n_blocks, kernel_size=cfg.kernel_size, rank=cfg.rank,
        ),
        "ShallowConv": ShallowConvClassifier(
            in_channels, N_CLASSES, channels=cfg.channels,
            n_blocks=cfg.n_blocks, kernel_size=cfg.kernel_size,
        ),
        "MLP": MLPClassifier(in_channels, seq_len, N_CLASSES, hidden=cfg.mlp_hidden),
    }
    for name, m in models.items():
        n_p = sum(p.numel() for p in m.parameters())
        print(f"  {name:15s}  params={n_p:,}")

    # Train
    train_results: Dict[str, TrainResult] = {}
    for name, model in models.items():
        print(f"\nTraining {name} (subject {subject}) ...")
        tr = train_model(
            model, X_tr, y_tr, X_val, y_val,
            epochs=cfg.epochs, batch_size=cfg.batch_size,
            lr=cfg.lr, device=device, label=f"S{subject}/{name}",
        )
        train_results[name] = tr

    # Test accuracy
    test_accs: Dict[str, float] = {}
    Xte = torch.tensor(X_test, dtype=torch.float32).to(device)
    yte = torch.tensor(y_test, dtype=torch.long).to(device)
    print(f"\nTest accuracies (subject {subject}):")
    for name, tr in train_results.items():
        tr.model.eval()
        with torch.no_grad():
            acc = (tr.model(Xte).argmax(1) == yte).float().mean().item()
        test_accs[name] = acc
        print(f"  {name:15s}  {acc:.3f}")

    # Recruitment diagnostic
    poly_model = train_results["PolyConv1d"].model
    recruitments = [
        b.quad_scale_mean()
        for b in poly_model.blocks
        if hasattr(b, "quad_scale_mean")
    ]
    if recruitments:
        print(f"  Recruitment (quad_scale_mean): "
              f"{[f'{r:.4f}' for r in recruitments]}")

    # Occlusion sensitivity + plot (best model)
    best_name = max(test_accs, key=test_accs.get)
    print(f"\nOcclusion sensitivity ({best_name}, subject {subject}) ...")
    sens_map = compute_occlusion_map(
        train_results[best_name].model, X_test, y_test, device,
        window=cfg.occlusion_window, stride=cfg.occlusion_stride,
    )   # (22, T)

    # Mean epoch trace (all classes) for the overlay signal
    mean_epoch = X_test.mean(axis=0)   # (22, T)
    t_axis     = np.linspace(cfg.tmin, cfg.tmax, seq_len)

    plots_dir = Path(cfg.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_occlusion_stacked(
            mean_epoch,
            sens_map,
            channel_names=CHANNEL_NAMES[:in_channels],
            time=t_axis,
            xlabel="time post-cue (s)",
            cmap="YlOrRd",
            alpha=0.55,
            title=(f"EEG occlusion sensitivity — {best_name} "
                   f"(subject {subject}, acc={test_accs[best_name]:.2f})"),
            name=f"eeg_occlusion_s{subject:02d}_seed{cfg.seed}",
            plots_dir=plots_dir,
            panel_height=1.1,
        )

    return SubjectResult(subject, test_accs, train_results, in_channels, seq_len)


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class EEGResult:
    per_subject: List[SubjectResult]
    mean_accs: Dict[str, float]
    std_accs:  Dict[str, float]


def run(cfg) -> EEGResult:
    if not _MOABB_OK:
        raise ImportError(
            "MOABB is required: pip install moabb"
        )

    configure_plots()
    device = torch.device(
        cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}  |  Subjects: {cfg.subjects}  |  Epochs: {cfg.epochs}")

    plots_dir = Path(cfg.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    per_subject: List[SubjectResult] = []
    for subj in cfg.subjects:
        sr = run_subject(subj, cfg)
        per_subject.append(sr)

    # Aggregate across subjects
    model_names = list(per_subject[0].test_accs.keys())
    all_accs: Dict[str, List[float]] = {n: [] for n in model_names}
    for sr in per_subject:
        for name, acc in sr.test_accs.items():
            all_accs[name].append(acc)

    mean_accs = {n: float(np.mean(v)) for n, v in all_accs.items()}
    std_accs  = {n: float(np.std(v))  for n, v in all_accs.items()}

    print(f"\n{'='*55}")
    print(f"Aggregate results ({len(per_subject)} subjects, seed={cfg.seed})")
    print(f"{'='*55}")
    for name in model_names:
        print(f"  {name:15s}  "
              f"mean={mean_accs[name]:.3f}  std={std_accs[name]:.3f}")

    # --- Summary bar chart ---
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(model_names))
    errs = [std_accs[n] for n in model_names]
    bars = ax.bar(
        x,
        [mean_accs[n] for n in model_names],
        0.55,
        yerr=errs,
        capsize=5,
        color=[OKABE_ITO[i % len(OKABE_ITO)] for i in range(len(model_names))],
        edgecolor="black", linewidth=0.7,
    )
    ax.axhline(0.25, ls="--", lw=1.2, color="grey", label="chance (25 %)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("test accuracy")
    ax.set_title(f"BCI-IV-2a — {len(per_subject)}-subject mean ± std  (seed {cfg.seed})")
    ax.legend(fontsize="x-small")
    fig.tight_layout()
    save_figure(fig, f"eeg_accuracy_bars_seed{cfg.seed}", plots_dir=plots_dir)

    # --- Per-subject accuracy table ---
    fig2, ax2 = plt.subplots(figsize=(max(6.5, 1.0 * len(per_subject)), 4.2))
    subjects_x = np.arange(len(per_subject))
    w = 0.25
    for i, name in enumerate(model_names):
        accs = [sr.test_accs[name] for sr in per_subject]
        ax2.bar(subjects_x + i * w, accs, w,
                color=OKABE_ITO[i % len(OKABE_ITO)],
                label=name, edgecolor="black", linewidth=0.5)
    ax2.axhline(0.25, ls="--", lw=1.0, color="grey", label="chance")
    ax2.set_xticks(subjects_x + w)
    ax2.set_xticklabels([f"S{sr.subject}" for sr in per_subject])
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("test accuracy")
    ax2.set_title(f"BCI-IV-2a — per-subject accuracy  (seed {cfg.seed})")
    ax2.legend(fontsize="x-small", loc="upper right")
    fig2.tight_layout()
    save_figure(fig2, f"eeg_accuracy_per_subject_seed{cfg.seed}", plots_dir=plots_dir)

    # --- Training curves (averaged across subjects) ---
    fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4))
    for i, name in enumerate(model_names):
        color = OKABE_ITO[i % len(OKABE_ITO)]
        # Mean val acc across subjects at each epoch
        curves = [sr.train_results[name].val_accs for sr in per_subject]
        min_len = min(len(c) for c in curves)
        arr = np.array([c[:min_len] for c in curves])
        mu_curve  = arr.mean(axis=0)
        std_curve = arr.std(axis=0)
        ep = np.arange(1, min_len + 1)
        axes3[0].plot(ep, mu_curve, color=color, lw=1.8, label=name)
        axes3[0].fill_between(ep, mu_curve - std_curve, mu_curve + std_curve,
                               color=color, alpha=0.15)
        # Train acc
        curves_tr = [sr.train_results[name].train_accs for sr in per_subject]
        arr_tr    = np.array([c[:min_len] for c in curves_tr])
        axes3[1].plot(ep, arr_tr.mean(axis=0), color=color, lw=1.8, label=name)
    for ax_, title_ in zip(axes3, ["validation accuracy", "train accuracy"]):
        ax_.axhline(0.25, ls="--", lw=1.0, color="grey", alpha=0.6)
        ax_.set_xlabel("epoch")
        ax_.set_ylabel("accuracy")
        ax_.set_title(title_)
        ax_.legend(fontsize="x-small")
    fig3.suptitle(f"BCI-IV-2a training curves — mean ± std ({len(per_subject)} subjects)")
    fig3.tight_layout()
    save_figure(fig3, f"eeg_training_curves_seed{cfg.seed}", plots_dir=plots_dir)

    return EEGResult(per_subject, mean_accs, std_accs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="EEG motor imagery experiment (BCI-IV-2a, PolyConv1d)"
    )
    p.add_argument("--subjects",       type=int, nargs="+", default=DEFAULTS["subjects"])
    p.add_argument("--seed",           type=int,   default=DEFAULTS["seed"])
    p.add_argument("--epochs",         type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--batch-size",     type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--lr",             type=float, default=DEFAULTS["lr"])
    p.add_argument("--channels",       type=int,   default=DEFAULTS["channels"])
    p.add_argument("--n-blocks",       type=int,   default=DEFAULTS["n_blocks"])
    p.add_argument("--kernel-size",    type=int,   default=DEFAULTS["kernel_size"])
    p.add_argument("--rank",           type=int,   default=DEFAULTS["rank"])
    p.add_argument("--mlp-hidden",     type=int,   default=DEFAULTS["mlp_hidden"])
    p.add_argument("--fmin",           type=float, default=DEFAULTS["fmin"])
    p.add_argument("--fmax",           type=float, default=DEFAULTS["fmax"])
    p.add_argument("--tmin",           type=float, default=DEFAULTS["tmin"])
    p.add_argument("--tmax",           type=float, default=DEFAULTS["tmax"])
    p.add_argument("--occlusion-window", type=int, default=DEFAULTS["occlusion_window"])
    p.add_argument("--occlusion-stride", type=int, default=DEFAULTS["occlusion_stride"])
    p.add_argument("--plots-dir",      type=str,   default=str(DEFAULTS["plots_dir"]))
    p.add_argument("--device",         type=str,   default=DEFAULTS["device"])
    args = p.parse_args()
    d = {k.replace("-", "_"): v for k, v in vars(args).items()}
    return argparse.Namespace(**d)


if __name__ == "__main__":
    cfg = _parse_args()
    result = run(cfg)
    print("\n=== Final Summary ===")
    for name in result.mean_accs:
        print(f"  {name:15s}  mean={result.mean_accs[name]:.3f}  "
              f"std={result.std_accs[name]:.3f}")
