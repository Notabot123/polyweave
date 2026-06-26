"""Suspension RUL regression — PolyConv1d vs MLP on SimWeave-generated data.

Generates a physics-based multivariate time-series dataset using SimWeave's
FullCarModel (7 DOF) with a degrading front suspension damper (``c_s``).
A Monte Carlo corpus of varied degradation onset times, severities, and road
profiles is produced, windowed into fixed-length sequences, and used to train
and compare three models:

    * ``PolyConv1d``  — our polynomial 1-D conv (the paper contribution)
    * ``MLP``         — dense feedforward baseline (flattened window)
    * ``PolyLinear``  — dense polynomial (PolyLinear, no conv structure)

Evaluation metrics: MAE and RMSE on held-out runs (run-level train/test split
to avoid leakage across time).  Occlusion sensitivity is computed on the best
model and visualised with the stacked lineplot + heatmap overlay.

Usage::

    # Quick single-seed pilot (default: 50 MC runs, 30 epochs)
    python -m polyweave.experiments.suspension_rul

    # Specify seed / device / number of runs
    python -m polyweave.experiments.suspension_rul --seed 42 --n-runs 50 --epochs 30

    # Multi-seed (called by run_paper1.py)
    python -m polyweave.experiments.suspension_rul --seed 42 --n-runs 200 --epochs 80

Notes
-----
* SimWeave's ``ParameterFault`` targets ``system.c_s`` on the ``FullCarModel``
  instance wrapped inside ``FaultInjector``.  If SimWeave stores the damping
  under a different attribute name, pass ``--damping-attr`` to override.
* Road profiles are sums of sinusoids at 12 frequencies drawn per run, so
  every MC run has a distinct, deterministic, non-trivial excitation — the
  conv polynomial sees genuine multi-frequency dynamics, not just bumps.
* The run-level train/test split (first 80 % of runs = train) prevents any
  leakage between sequences from the same physical trajectory.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# SimWeave imports — hard dependency for this experiment
# ---------------------------------------------------------------------------
try:
    import simweave as sw
    from simweave.continuous.solver import simulate
    from simweave.continuous.systems import FullCarModel
    from simweave.faults import FaultDataset, FaultInjector, FaultProfile, ParameterFault
    from simweave.mc import run_monte_carlo
    _SIMWEAVE_OK = True
except ImportError:
    _SIMWEAVE_OK = False

from polyweave.layers import PolyConv1d, PolyLinear
from polyweave.viz.plots import configure_plots, save_figure, OKABE_ITO
from polyweave.viz.timeseries import (
    plot_occlusion_stacked,
    plot_rul_prediction,
    plot_timeseries_predictions,
)
from polyweave.interpretability.occlusion import occlusion_sensitivity_1d

# ---------------------------------------------------------------------------
# Default hyper-parameters (all overridable via argparse)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    seed=42,
    n_runs=50,
    train_frac=0.80,
    epochs=30,
    batch_size=256,
    lr=1e-3,
    seq_len=50,
    train_stride=5,
    test_stride=50,          # non-overlapping on test runs
    channels=32,
    n_blocks=3,
    kernel_size=7,
    rank=8,
    mlp_hidden=128,
    max_rul=None,            # None → infer from training data
    damping_attr="c_s",
    dt=0.01,                 # integration step (100 Hz)
    road_amp=0.004,          # stochastic road amplitude (m)
    nominal_damping=1500.0,
    max_delta=-0.80,         # c_s degrades to 20 % of nominal at full failure
    plots_dir=Path("plots"),
    device=None,
)

# FullCarModel construction kwargs (shared across all runs)
_FC_BASE = dict(
    sprung_mass=1200.0,
    pitch_inertia=2500.0,
    roll_inertia=2200.0,
    unsprung_mass=60.0,
    k_s=20_000.0,
    k_t=150_000.0,
    a=1.2,
    b=1.6,
    track_width=1.6,
)


# ---------------------------------------------------------------------------
# Road profile
# ---------------------------------------------------------------------------

def make_road_fn(seed: int, amplitude: float = 0.004):
    """4-wheel stochastic road profile as a sum of 12 sinusoids.

    Frequencies span 0.5–20 Hz to excite body bounce (~1 Hz), pitch (~1.5 Hz),
    wheel-hop (~10–15 Hz), and mid-band resonances simultaneously.  Each wheel
    receives the same frequencies but independent random phases, so the input
    excites both pitch and roll — giving the model richer dynamics to learn from
    than a single bump would.

    Returns a callable ``road(t) -> (FL, FR, RL, RR)`` that SimWeave's
    ``simulate()`` can call at arbitrary RK4 sub-steps.
    """
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(0.5, 20.0, size=12)
    amps  = amplitude * rng.uniform(0.4, 1.0, size=12)
    # 4 wheels × 12 components, independent phases per wheel
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(4, 12))

    freqs_  = freqs.copy()
    amps_   = amps.copy()
    phases_ = phases.copy()

    def road(t: float) -> tuple:
        v = [
            float(np.dot(amps_, np.sin(2.0 * np.pi * freqs_ * t + phases_[w])))
            for w in range(4)
        ]
        return (v[0], v[1], v[2], v[3])

    return road


# ---------------------------------------------------------------------------
# SimWeave dataset generation
# ---------------------------------------------------------------------------

def _make_run(
    seed: int,
    nominal_damping: float,
    max_delta: float,
    damping_attr: str,
    dt: float,
    road_amp: float,
) -> Optional["FaultDataset"]:
    """Generate one Monte Carlo run: random onset / severity / road profile."""
    rng     = np.random.default_rng(seed)
    onset   = float(rng.uniform(5.0, 15.0))
    failure = onset + float(rng.uniform(5.0, 15.0))
    t_end   = failure + float(rng.uniform(3.0, 7.0))
    severity = float(rng.uniform(max_delta * 1.2, max_delta * 0.8))  # vary degradation depth
    noise_std = float(rng.uniform(0.001, 0.005))

    profile  = FaultProfile(onset, failure, mode="damper_wear", shape="exponential")
    fault    = ParameterFault(damping_attr, profile, max_delta=severity, relative=True)
    model    = FullCarModel(**_FC_BASE, c_s=nominal_damping)
    injector = FaultInjector(system=model, faults=[fault])
    road_fn  = make_road_fn(seed=seed + 10000, amplitude=road_amp)

    result = simulate(injector, (0.0, t_end), dt=dt, inputs=road_fn)
    ds = FaultDataset.from_result(
        result, injector,
        noise_std=noise_std,
        rng=np.random.default_rng(seed + 20000),
    )
    return ds, onset, failure


def generate_corpus(cfg) -> Tuple[List, List, List]:
    """Generate n_runs MC FaultDatasets; return (datasets, onset_times, failure_times)."""
    print(f"Generating {cfg.n_runs} SimWeave runs (FullCarModel, dt={cfg.dt}s) ...")
    t0 = time.perf_counter()

    datasets, onsets, failures = [], [], []
    for i in range(cfg.n_runs):
        seed_i = cfg.seed * 10000 + i
        ds, onset, failure = _make_run(
            seed_i, cfg.nominal_damping, cfg.max_delta,
            cfg.damping_attr, cfg.dt, cfg.road_amp,
        )
        datasets.append(ds)
        onsets.append(onset)
        failures.append(failure)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{cfg.n_runs}  ({time.perf_counter()-t0:.1f}s)")

    print(f"Done in {time.perf_counter()-t0:.1f}s  |  "
          f"features: {datasets[0].feature_names}  "
          f"n_features={datasets[0].features.shape[1]}")
    return datasets, onsets, failures


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def window_run(
    ds: "FaultDataset",
    seq_len: int,
    stride: int,
    max_rul: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slide a window over one FaultDataset; return (X, y) for degrading window only.

    Only windows whose final timestep falls in the degrading or failed regime
    (``health_index < 1.0``) are kept — pre-onset samples have infinite RUL and
    are not useful regression targets.

    X shape: ``(n_windows, n_features, seq_len)`` — channel-first for Conv1d.
    y shape: ``(n_windows,)`` — RUL clipped to ``max_rul``, normalised to [0, 1].
    """
    features = ds.features          # (N, F)
    rul      = ds.rul.copy()        # (N,)  — inf before onset
    hi       = ds.health_index      # (N,)

    N, F = features.shape
    X_wins, y_wins = [], []

    for start in range(0, N - seq_len + 1, stride):
        end = start + seq_len
        last = end - 1
        if hi[last] >= 1.0:         # pre-onset: skip
            continue
        r = float(np.clip(rul[last], 0.0, max_rul))
        X_wins.append(features[start:end].T)   # (F, seq_len)
        y_wins.append(r / max_rul)             # normalise to [0, 1]

    if not X_wins:
        return np.empty((0, F, seq_len)), np.empty((0,))
    return np.stack(X_wins).astype(np.float32), np.array(y_wins, dtype=np.float32)


def build_dataset(
    datasets: List,
    seq_len: int,
    stride: int,
    max_rul: float,
    feature_mean: Optional[np.ndarray] = None,
    feature_std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Window all runs and concatenate; return X, y and feature stats."""
    all_X, all_y = [], []
    for ds in datasets:
        X, y = window_run(ds, seq_len, stride, max_rul)
        if len(X):
            all_X.append(X)
            all_y.append(y)

    X = np.concatenate(all_X, axis=0)   # (N_total, F, seq_len)
    y = np.concatenate(all_y, axis=0)   # (N_total,)

    if feature_mean is None:
        # Compute per-channel stats over the flattened training corpus.
        flat = X.reshape(len(X), X.shape[1], -1)      # (N, F, T)
        feature_mean = flat.mean(axis=(0, 2))          # (F,)
        feature_std  = flat.std(axis=(0, 2)) + 1e-8   # (F,)

    # Normalise: broadcast over (N, F, seq_len).
    X = (X - feature_mean[None, :, None]) / feature_std[None, :, None]
    return X, y, feature_mean, feature_std


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PolyConv1dRegressor(nn.Module):
    """PolyConv1d backbone → global avg pool → linear head."""

    def __init__(self, in_channels: int, channels: int = 32,
                 n_blocks: int = 3, kernel_size: int = 7, rank: int = 8) -> None:
        super().__init__()
        self.proj   = nn.Conv1d(in_channels, channels, kernel_size=1, bias=False)
        self.blocks = nn.Sequential(*[
            PolyConv1d(channels, kernel_size=kernel_size, rank=rank)
            for _ in range(n_blocks)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = self.proj(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)      # (B, channels)
        return self.head(x).squeeze(-1)   # (B,)


class MLPRegressor(nn.Module):
    """Dense MLP baseline: flatten window → hidden layers → scalar."""

    def __init__(self, in_channels: int, seq_len: int, hidden: int = 128) -> None:
        super().__init__()
        in_dim = in_channels * seq_len
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PolyLinearRegressor(nn.Module):
    """PolyLinear (dense polynomial) baseline: flatten → poly layer → scalar."""

    def __init__(self, in_channels: int, seq_len: int, hidden: int = 128,
                 rank: int = 8) -> None:
        super().__init__()
        in_dim = in_channels * seq_len
        self.net = nn.Sequential(
            nn.Flatten(),
            PolyLinear(in_dim, hidden, rank=rank), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TrainResult:
    model: nn.Module
    train_losses: List[float]
    val_losses: List[float]
    best_val_loss: float
    best_state: dict


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    label: str = "",
) -> TrainResult:
    model = model.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.MSELoss()

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    Xv = torch.tensor(X_val,   dtype=torch.float32).to(device)
    yv = torch.tensor(y_val,   dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size,
                        shuffle=True, drop_last=False)

    train_losses, val_losses = [], []
    best_val, best_state = float("inf"), None

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(Xb)
            loss = crit(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            ep_loss += loss.item() * len(Xb)
        sched.step()

        train_loss = ep_loss / len(Xt)
        model.eval()
        with torch.no_grad():
            val_loss = crit(model(Xv), yv).item()
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs:
            print(f"  [{label}] epoch {epoch:3d}/{epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"best={best_val:.4f}")

    model.load_state_dict(best_state)
    return TrainResult(model, train_losses, val_losses, best_val, best_state)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class EvalResult:
    mae: float
    rmse: float
    pred: np.ndarray
    true: np.ndarray


@torch.no_grad()
def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray,
             max_rul: float, device: torch.device) -> EvalResult:
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    pred_norm = model(Xt).cpu().numpy()
    pred = pred_norm * max_rul
    true = y * max_rul
    mae  = float(np.abs(pred - true).mean())
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    return EvalResult(mae, rmse, pred, true)


# ---------------------------------------------------------------------------
# Occlusion sensitivity
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_occlusion(
    model: nn.Module,
    X_sample: np.ndarray,
    device: torch.device,
    window: int = 5,
    stride: int = 2,
) -> np.ndarray:
    """Per-timestep occlusion sensitivity averaged over a sample batch.

    Returns ``[C, T]`` mean sensitivity map (mean over the sample dimension).
    The response function is the model's scalar RUL prediction.
    """
    model.eval()
    Xs = torch.tensor(X_sample, dtype=torch.float32).to(device)  # (N, C, T)

    # Per-channel occlusion: zero out a sliding window over the time axis
    # for each channel independently.
    C, T = Xs.shape[1], Xs.shape[2]
    sens_map = np.zeros((C, T), dtype=np.float32)
    counts   = np.zeros((C, T), dtype=np.float32)

    with torch.no_grad():
        base = model(Xs).cpu().numpy()  # (N,)

    n_steps = list(range(0, T - window + 1, stride))
    if n_steps and n_steps[-1] != T - window:
        n_steps.append(T - window)

    for ch in range(C):
        for s in n_steps:
            occ = Xs.clone()
            occ[:, ch, s:s + window] = 0.0
            drop = base - model(occ).cpu().numpy()  # (N,)
            mean_drop = float(drop.mean())
            for t in range(s, s + window):
                if t < T:
                    sens_map[ch, t] += mean_drop
                    counts[ch, t]   += 1.0

    counts = np.maximum(counts, 1.0)
    return sens_map / counts


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SuspensionRULResult:
    metrics: Dict[str, EvalResult]
    train_results: Dict[str, TrainResult]
    n_features: int
    seq_len: int
    max_rul: float


def run(cfg) -> SuspensionRULResult:
    if not _SIMWEAVE_OK:
        raise ImportError(
            "SimWeave is required for this experiment: pip install simweave"
        )

    device = torch.device(
        cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    configure_plots()
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    # ------------------------------------------------------------------
    # 1. Generate SimWeave corpus
    # ------------------------------------------------------------------
    datasets, onsets, failures = generate_corpus(cfg)

    n_train = int(len(datasets) * cfg.train_frac)
    train_ds = datasets[:n_train]
    test_ds  = datasets[n_train:]
    print(f"Train runs: {n_train}  |  Test runs: {len(test_ds)}")

    # ------------------------------------------------------------------
    # 2. Infer max_rul from training data
    # ------------------------------------------------------------------
    max_rul = cfg.max_rul
    if max_rul is None:
        finite_ruls = [
            ds.rul[np.isfinite(ds.rul) & (ds.health_index < 1.0)]
            for ds in train_ds
        ]
        flat = np.concatenate(finite_ruls) if finite_ruls else np.array([30.0])
        max_rul = float(flat.max())
        print(f"Inferred max_rul: {max_rul:.2f}s")

    # ------------------------------------------------------------------
    # 3. Build windowed datasets
    # ------------------------------------------------------------------
    print("Windowing training runs ...")
    X_train, y_train, feat_mean, feat_std = build_dataset(
        train_ds, cfg.seq_len, cfg.train_stride, max_rul
    )
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")

    # Validation: last 15 % of training windows.
    n_val    = max(1, int(len(X_train) * 0.15))
    val_idx  = rng.choice(len(X_train), n_val, replace=False)
    train_idx = np.setdiff1d(np.arange(len(X_train)), val_idx)
    X_val, y_val   = X_train[val_idx],   y_train[val_idx]
    X_train, y_train = X_train[train_idx], y_train[train_idx]
    print(f"  Train windows: {len(X_train)}  Val windows: {len(X_val)}")

    print("Windowing test runs ...")
    X_test, y_test, _, _ = build_dataset(
        test_ds, cfg.seq_len, cfg.test_stride, max_rul,
        feature_mean=feat_mean, feature_std=feat_std,
    )
    print(f"  X_test: {X_test.shape}  y_test: {y_test.shape}")

    in_channels = X_train.shape[1]
    print(f"Input channels (features): {in_channels}")

    # ------------------------------------------------------------------
    # 4. Build models
    # ------------------------------------------------------------------
    models = {
        "PolyConv1d": PolyConv1dRegressor(
            in_channels, channels=cfg.channels,
            n_blocks=cfg.n_blocks, kernel_size=cfg.kernel_size, rank=cfg.rank,
        ),
        "MLP": MLPRegressor(in_channels, cfg.seq_len, hidden=cfg.mlp_hidden),
        "PolyLinear": PolyLinearRegressor(in_channels, cfg.seq_len,
                                           hidden=cfg.mlp_hidden, rank=cfg.rank),
    }
    for name, m in models.items():
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  {name:15s}  params={n_params:,}")

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    train_results: Dict[str, TrainResult] = {}
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        result = train_model(
            model, X_train, y_train, X_val, y_val,
            epochs=cfg.epochs, batch_size=cfg.batch_size,
            lr=cfg.lr, device=device, label=name,
        )
        train_results[name] = result

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    print("\n--- Test results ---")
    metrics: Dict[str, EvalResult] = {}
    for name, tr in train_results.items():
        ev = evaluate(tr.model, X_test, y_test, max_rul, device)
        metrics[name] = ev
        print(f"  {name:15s}  MAE={ev.mae:.2f}s  RMSE={ev.rmse:.2f}s")

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    plots_dir = Path(cfg.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 7a. Training loss curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (name, tr) in enumerate(train_results.items()):
        ax.plot(tr.val_losses, label=name, color=OKABE_ITO[i], lw=1.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val MSE (normalised RUL²)")
    ax.set_title("Suspension RUL — validation loss")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, f"suspension_rul_loss_seed{cfg.seed}", plots_dir=plots_dir)

    # 7b. RUL prediction panel — first test run, best model
    best_name = min(metrics, key=lambda k: metrics[k].rmse)
    print(f"\nBest model (test RMSE): {best_name}")

    test_run    = test_ds[0]
    onset_t     = onsets[n_train]
    failure_t   = failures[n_train]
    feat_mean_t = feat_mean[None, :, None]   # for broadcasting
    feat_std_t  = feat_std[None, :, None]

    # Reconstruct prediction over a full test trajectory (non-overlapping windows).
    _X_traj, _y_traj, _, _ = build_dataset(
        [test_run], cfg.seq_len, cfg.seq_len, max_rul,
        feature_mean=feat_mean, feature_std=feat_std,
    )

    if len(_X_traj) > 0:
        _preds = {}
        for name, tr in train_results.items():
            ev_traj = evaluate(tr.model, _X_traj, _y_traj, max_rul, device)
            _preds[name] = ev_traj.pred

        # Reconstruct time axis (centre of each window)
        traj_time = np.arange(len(_y_traj)) * cfg.seq_len * cfg.dt + cfg.seq_len * cfg.dt / 2

        plot_rul_prediction(
            traj_time,
            _y_traj * max_rul,
            _preds[best_name],
            onset_time=onset_t,
            failure_time=failure_t,
            title=f"Suspension RUL prediction — {best_name} (seed {cfg.seed})",
            name=f"suspension_rul_pred_seed{cfg.seed}",
            plots_dir=plots_dir,
        )

        plot_timeseries_predictions(
            traj_time,
            _y_traj * max_rul,
            {k: v for k, v in _preds.items()},
            channel_names=["RUL (s)"],
            onset_time=onset_t,
            failure_time=failure_t,
            title=f"Suspension RUL — all models vs ground truth (seed {cfg.seed})",
            name=f"suspension_rul_all_models_seed{cfg.seed}",
            plots_dir=plots_dir,
        )

    # 7c. Occlusion sensitivity on best model — first test run, degrading window
    print(f"\nOcclusion sensitivity ({best_name}) ...")
    best_model = train_results[best_name].model

    # Build a small batch of windows from the degrading portion of the test run.
    _X_occ, _y_occ, _, _ = build_dataset(
        [test_run], cfg.seq_len, cfg.seq_len * 2, max_rul,
        feature_mean=feat_mean, feature_std=feat_std,
    )

    if len(_X_occ) >= 4:
        occ_batch = _X_occ[:min(16, len(_X_occ))]     # (N, C, T)
        sens_map  = run_occlusion(
            best_model, occ_batch, device, window=5, stride=2
        )                                               # (C, T)

        # Show the first window's raw signal (de-normalised) with occlusion overlay.
        raw_signal = occ_batch[0] * feat_std[:, None] + feat_mean[:, None]  # (C, T)

        n_ch_to_show = min(8, raw_signal.shape[0])
        feat_names   = test_run.feature_names[:n_ch_to_show]

        plot_occlusion_stacked(
            raw_signal[:n_ch_to_show],
            sens_map[:n_ch_to_show],
            channel_names=feat_names,
            time=np.arange(cfg.seq_len) * cfg.dt,
            xlabel="time (s)",
            title=f"Suspension occlusion sensitivity — {best_name} (seed {cfg.seed})",
            name=f"suspension_occlusion_seed{cfg.seed}",
            plots_dir=plots_dir,
        )

    # 7d. Recruitment diagnostic (PolyConv1d only)
    if "PolyConv1d" in train_results:
        poly_model = train_results["PolyConv1d"].model
        recruitments = []
        for block in poly_model.blocks:
            if hasattr(block, "quad_scale_mean"):
                recruitments.append(block.quad_scale_mean())
        if recruitments:
            print(f"\nPolyConv1d quad_scale_mean per block: "
                  f"{[f'{r:.4f}' for r in recruitments]}")
            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            ax2.bar(range(len(recruitments)), recruitments,
                    color=OKABE_ITO[0], edgecolor="black", linewidth=0.8)
            ax2.axhline(0.135, ls="--", lw=1.2, color="grey",
                        label="init (exp(-2) ≈ 0.135)")
            ax2.set_xlabel("block")
            ax2.set_ylabel("exp(quad_scale)")
            ax2.set_title("Polynomial recruitment per block")
            ax2.legend(fontsize="x-small")
            fig2.tight_layout()
            save_figure(fig2, f"suspension_recruitment_seed{cfg.seed}",
                        plots_dir=plots_dir)

    return SuspensionRULResult(
        metrics=metrics,
        train_results=train_results,
        n_features=in_channels,
        seq_len=cfg.seq_len,
        max_rul=max_rul,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Suspension RUL experiment (SimWeave + PolyConv1d)"
    )
    p.add_argument("--seed",            type=int,   default=DEFAULTS["seed"])
    p.add_argument("--n-runs",          type=int,   default=DEFAULTS["n_runs"])
    p.add_argument("--train-frac",      type=float, default=DEFAULTS["train_frac"])
    p.add_argument("--epochs",          type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--batch-size",      type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--lr",              type=float, default=DEFAULTS["lr"])
    p.add_argument("--seq-len",         type=int,   default=DEFAULTS["seq_len"])
    p.add_argument("--train-stride",    type=int,   default=DEFAULTS["train_stride"])
    p.add_argument("--test-stride",     type=int,   default=DEFAULTS["test_stride"])
    p.add_argument("--channels",        type=int,   default=DEFAULTS["channels"])
    p.add_argument("--n-blocks",        type=int,   default=DEFAULTS["n_blocks"])
    p.add_argument("--kernel-size",     type=int,   default=DEFAULTS["kernel_size"])
    p.add_argument("--rank",            type=int,   default=DEFAULTS["rank"])
    p.add_argument("--mlp-hidden",      type=int,   default=DEFAULTS["mlp_hidden"])
    p.add_argument("--max-rul",         type=float, default=DEFAULTS["max_rul"])
    p.add_argument("--damping-attr",    type=str,   default=DEFAULTS["damping_attr"])
    p.add_argument("--dt",              type=float, default=DEFAULTS["dt"])
    p.add_argument("--road-amp",        type=float, default=DEFAULTS["road_amp"])
    p.add_argument("--nominal-damping", type=float, default=DEFAULTS["nominal_damping"])
    p.add_argument("--max-delta",       type=float, default=DEFAULTS["max_delta"])
    p.add_argument("--plots-dir",       type=str,   default=str(DEFAULTS["plots_dir"]))
    p.add_argument("--device",          type=str,   default=DEFAULTS["device"])
    args = p.parse_args()
    # Convert hyphen-separated names to underscore attributes.
    d = {k.replace("-", "_"): v for k, v in vars(args).items()}
    return argparse.Namespace(**d)


if __name__ == "__main__":
    cfg = _parse_args()
    result = run(cfg)
    print("\n=== Summary ===")
    for name, ev in result.metrics.items():
        print(f"  {name:15s}  MAE={ev.mae:.2f}s  RMSE={ev.rmse:.2f}s")
