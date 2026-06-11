"""Test the claim: poly's gain scales with residual nonlinearity beyond the linear
ceiling. For every block of the depth sweep,
    x = 1 - R2_linear         (variance NOT recovered by the exact best linear map)
    y = R2_poly - R2_linear    (extra variance the low-rank bilinear recovers)
If x and y correlate across blocks AND models, "multiplicative benefit scales with
distance from the linear ceiling" — a stronger, testable statement than "multiplication
helps". Also reports the recovery fraction y/x (what share of the residual poly closes).

Reads plots/raw/depth_sweep_wikitext2.json; writes plots/polyweave_residual_gain.{pdf,png}.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

JSON = Path("plots/raw/depth_sweep_wikitext2.json")
STYLE = {"gpt2": ("C0", "GPT-2 (GELU)"),
         "EleutherAI/pythia-160m": ("C1", "Pythia-160m (GELU)"),
         "JackFram/llama-160m": ("C2", "llama-160m (SwiGLU)")}


def _pearson(a, b):
    a, b = torch.tensor(a), torch.tensor(b)
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


def _spearman(a, b):
    ra = torch.tensor(a).argsort().argsort().float()
    rb = torch.tensor(b).argsort().argsort().float()
    return _pearson(ra.tolist(), rb.tolist())


def main() -> None:
    data = json.loads(JSON.read_text(encoding="utf-8"))
    allx, ally = [], []
    per_model = {}
    for name, d in data.items():
        xs, ys, fr = [], [], []
        for r in d["blocks"]:
            if r.get("r2_poly") is None:
                continue
            x = 1.0 - r["r2_lin"]
            y = r["r2_poly"] - r["r2_lin"]
            xs.append(x); ys.append(y); fr.append(y / x if x > 1e-3 else float("nan"))
            allx.append(x); ally.append(y)
        per_model[name] = (xs, ys, fr)
        med_fr = torch.tensor([f for f in fr if f == f]).median().item() if fr else float("nan")
        print(f"{name:<24} n={len(xs):2d}  pearson={_pearson(xs, ys):+.3f}  "
              f"spearman={_spearman(xs, ys):+.3f}  median recovery y/x={med_fr:+.3f}")
    print(f"{'ALL':<24} n={len(allx):2d}  pearson={_pearson(allx, ally):+.3f}  "
          f"spearman={_spearman(allx, ally):+.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for name, (xs, ys, _) in per_model.items():
        c, lab = STYLE.get(name, ("C3", name))
        ax.scatter(xs, ys, c=c, label=lab, s=42, alpha=0.8, edgecolor="k", linewidth=0.4)
    lim = max(allx) * 1.05
    ax.plot([0, lim], [0, lim], "k:", alpha=0.4, label="y = x (poly recovers all residual)")
    ax.set_xlabel("residual nonlinearity  (1 − R²_linear)")
    ax.set_ylabel("poly gain  (R²_poly − R²_linear)")
    ax.set_title(f"Multiplicative benefit vs residual nonlinearity\n"
                 f"(per block; Pearson r = {_pearson(allx, ally):+.2f})")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"plots/polyweave_residual_gain.{ext}", dpi=150)
    print("saved plots/polyweave_residual_gain.{pdf,png}")


if __name__ == "__main__":
    main()
