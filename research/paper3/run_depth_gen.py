"""Phase 1/2 — depth-generalization harness for Paper 3.

Trains a small Transformer classifier (the learned *foil*) on SHALLOW proofs only
(depth <= train cutoff) and evaluates accuracy as a function of test proof depth, well
beyond the training range. The differentiable forward chainer is the reference: given
the rules it is exact at any depth (it needs depth-many chaining steps — an honest,
depth-scaling cost), so it sits at 100% and the question is how far the *learned* model
follows before it falls off.

Headline figure: accuracy vs proof depth, transformer (decaying past the train cutoff)
vs chaining (flat). This is the Phase 2 decision gate — run it and see if the story holds
before investing in ProntoQA / ProofWriter.

A bidirectional Transformer *encoder* with a [CLS] readout is the right (and stronger)
foil for this whole-problem classification task; "decoder" in the scope note was loose
shorthand. Atoms are encoded as random per-instance slots (see encoding.py), so the model
must generalize over structure, not atom identity.

Run (quick signal):  python research/paper3/run_depth_gen.py --epochs 20
     (smoke test):    python research/paper3/run_depth_gen.py --smoke
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from encoding import PAD, encode, vocab_size  # noqa: E402
from horn_kb import make_dataset  # noqa: E402

from polyweave.reasoning import ForwardChainer  # noqa: E402

HERE = Path(__file__).parent


@dataclass
class Config:
    seed: int = 0
    train_depths: Tuple[int, ...] = (1, 2, 3, 4)
    n_train_per_depth: int = 500
    eval_depths: Tuple[int, ...] = field(default_factory=lambda: tuple(range(1, 13)))
    n_eval_per_depth: int = 200
    n_distractor_rules: int = 8
    max_slots: int = 64
    d_model: int = 128
    nhead: int = 4
    nlayers: int = 4
    dim_ff: int = 256
    dropout: float = 0.1
    epochs: int = 25
    batch_size: int = 64
    lr: float = 3e-4
    out_prefix: str = "paper3_depthgen"


class TransformerClassifier(nn.Module):
    def __init__(self, vocab: int, max_len: int, cfg: Config) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, cfg.d_model, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.nhead, cfg.dim_ff, cfg.dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, cfg.nlayers)
        self.head = nn.Linear(cfg.d_model, 2)

    def forward(self, toks: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(toks.shape[1], device=toks.device).unsqueeze(0)
        x = self.tok(toks) + self.pos(pos)
        x = self.enc(x, src_key_padding_mask=pad_mask)
        return self.head(x[:, 0])  # [CLS]


def _pad(seqs: List[List[int]], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    toks = torch.full((len(seqs), max_len), PAD, dtype=torch.long)
    mask = torch.ones(len(seqs), max_len, dtype=torch.bool)
    for i, s in enumerate(seqs):
        s = s[:max_len]
        toks[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = False
    return toks, mask


def chaining_accuracy(instances, max_steps: int) -> float:
    correct = 0
    for inst in instances:
        kb, f0 = inst.build()
        entailed, _ = ForwardChainer(kb, max_steps=max_steps).entails(f0, inst.query)
        correct += int(entailed == inst.label)
    return correct / len(instances)


def run(cfg: Config) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    enc_rng = random.Random(cfg.seed + 1)

    print(f"device={device}  train_depths={cfg.train_depths}  "
          f"eval_depths={cfg.eval_depths[0]}..{cfg.eval_depths[-1]}")

    # --- data (generated + oracle-verified) ---
    train_inst = make_dataset(rng, list(cfg.train_depths), cfg.n_train_per_depth,
                              n_distractor_rules=cfg.n_distractor_rules, verify=True)
    eval_inst = {d: make_dataset(rng, [d], cfg.n_eval_per_depth,
                                 n_distractor_rules=cfg.n_distractor_rules, verify=True)
                 for d in cfg.eval_depths}

    # --- encode (consistent first-appearance ids; shared max_len) ---
    train_enc = [encode(i, cfg.max_slots) for i in train_inst]
    eval_enc = {d: [encode(i, cfg.max_slots) for i in insts]
                for d, insts in eval_inst.items()}
    max_len = max([len(s) for s, _ in train_enc]
                  + [len(s) for d in eval_enc for s, _ in eval_enc[d]])
    print(f"train={len(train_inst)}  max_len={max_len}  "
          f"vocab={vocab_size(cfg.max_slots)}")

    Xtr, Mtr = _pad([s for s, _ in train_enc], max_len)
    ytr = torch.tensor([y for _, y in train_enc])
    Xtr, Mtr, ytr = Xtr.to(device), Mtr.to(device), ytr.to(device)

    model = TransformerClassifier(vocab_size(cfg.max_slots), max_len, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"transformer params: {n_params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lossf = nn.CrossEntropyLoss()

    # --- train on shallow proofs only ---
    t0 = time.time()
    N = Xtr.shape[0]
    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        tot = 0.0
        for b in range(0, N, cfg.batch_size):
            idx = perm[b : b + cfg.batch_size]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx], Mtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"  epoch {ep+1:>2}/{cfg.epochs}  train_loss={tot/N:.4f}")
    print(f"trained in {time.time()-t0:.1f}s")

    # --- evaluate accuracy vs depth ---
    max_steps = max(cfg.eval_depths) + 2
    model.eval()
    results = {"config": cfg.__dict__ | {"n_params": n_params, "device": device},
               "by_depth": {}}
    print("\ndepth |  transformer  chaining   (n)")
    for d in cfg.eval_depths:
        seqs = [s for s, _ in eval_enc[d]]
        ys = torch.tensor([y for _, y in eval_enc[d]]).to(device)
        Xe, Me = _pad(seqs, max_len)
        with torch.no_grad():
            pred = model(Xe.to(device), Me.to(device)).argmax(-1)
        tacc = (pred == ys).float().mean().item()
        cacc = chaining_accuracy(eval_inst[d], max_steps)
        in_dist = "  (train)" if d in cfg.train_depths else ""
        print(f"{d:>5} |  {tacc:8.3f}    {cacc:7.3f}   {len(ys)}{in_dist}")
        results["by_depth"][d] = {"transformer": tacc, "chaining": cacc, "n": len(ys)}

    _save(cfg, results)
    return results


def _save(cfg: Config, results: Dict) -> None:
    out_json = HERE / "results" / f"{cfg.out_prefix}.json"
    out_json.parent.mkdir(exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, default=str))

    depths = list(results["by_depth"])
    tacc = [results["by_depth"][d]["transformer"] for d in depths]
    cacc = [results["by_depth"][d]["chaining"] for d in depths]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(depths, cacc, "s-", label="forward chaining (exact)", color="#2a9d8f")
    ax.plot(depths, tacc, "o-", label="transformer (learned)", color="#e76f51")
    ax.axvline(max(cfg.train_depths) + 0.5, color="grey", ls="--", lw=1)
    ax.text(max(cfg.train_depths) + 0.6, 0.05, "trained up to here", fontsize=8, color="grey")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_xlabel("test proof depth")
    ax.set_ylabel("entailment accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Depth generalization of deductive reasoning")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    png = Path("plots") / f"{cfg.out_prefix}.png"
    png.parent.mkdir(exist_ok=True)
    fig.savefig(png, dpi=120)
    print(f"\nsaved {out_json}  and  {png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=None, help="train instances per depth")
    ap.add_argument("--n-eval", type=int, default=None, help="eval instances per depth")
    ap.add_argument("--layers", type=int, default=None, help="transformer layers")
    ap.add_argument("--dmodel", type=int, default=None, help="model width")
    ap.add_argument("--out", type=str, default=None, help="output prefix")
    ap.add_argument("--smoke", action="store_true", help="tiny config to check it runs")
    args = ap.parse_args()

    cfg = Config()
    if args.smoke:
        cfg = Config(n_train_per_depth=40, n_eval_per_depth=40, epochs=3,
                     d_model=64, nlayers=2, eval_depths=tuple(range(1, 7)),
                     out_prefix="paper3_depthgen_smoke")
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.n_train is not None:
        cfg.n_train_per_depth = args.n_train
    if args.n_eval is not None:
        cfg.n_eval_per_depth = args.n_eval
    if args.layers is not None:
        cfg.nlayers = args.layers
    if args.dmodel is not None:
        cfg.d_model = args.dmodel
    if args.out is not None:
        cfg.out_prefix = args.out
    run(cfg)
