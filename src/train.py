"""Train the multi-head BiGRU.

    ./py src/train.py --name base [--window 5] [--epochs 40] [--no-aug] [--limit-steps N]

Writes outputs/models/<name>/{config.json, norm.json, model.pt, history.csv}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import augment  # noqa: E402
from analysis import ROOT  # noqa: E402
from gaitdata import EVENTS, FS, TARGET_NAMES, TARGETS, load_all, tile_windows, window_labels  # noqa: E402
from model import GaitNet, features  # noqa: E402

MODELS = ROOT / "outputs" / "models"


# ---------------------------------------------------------------- batching
def make_batch(items, W: int, rng, noise_fn):
    """items: [(Trial, i0, i1)] -> dict of tensors, windows padded to W frames."""
    B = len(items)
    x = np.zeros((B, W, 10, 2), np.float32)
    vis = np.zeros((B, W, 10), np.float32)
    mask = np.zeros((B, W), np.float32)
    ev = np.zeros((B, W, len(EVENTS)), np.float32)
    y = np.full((B, len(TARGET_NAMES)), np.nan, np.float32)
    for b, (tr, i0, i1) in enumerate(items):
        n = i1 - i0
        x[b, :n], vis[b, :n] = augment.apply(tr.pose[i0:i1], rng, noise_fn(rng))
        mask[b, :n] = 1
        ev[b, :n] = tr.ev_soft[i0:i1]
        y[b] = window_labels(tr, i0, i1)
    f = features(x, vis) * mask[..., None]
    return {"f": torch.from_numpy(f), "mask": torch.from_numpy(mask), "ev": torch.from_numpy(ev),
            "y": torch.from_numpy(y)}


class Sampler:
    """Random training windows: 50 % Ferber, 25 % Fukuchi treadmill, 25 % Fukuchi overground."""

    def __init__(self, trials, W, rng):
        groups = {"ferber": [], "fukuchi_treadmill": [], "fukuchi_overground": []}
        for tr in trials:
            groups[tr.ds if tr.ds == "ferber" else f"{tr.ds}_{tr.mode}"].append(tr)
        self.groups = [g for g in groups.values() if g]
        w = np.array([0.5, 0.25, 0.25])[[bool(g) for g in groups.values()]]
        self.p = w / w.sum()
        self.W, self.rng = W, rng

    def draw(self, B):
        out = []
        for gi in self.rng.choice(len(self.groups), B, p=self.p):
            g = self.groups[gi]
            tr = g[self.rng.integers(len(g))]
            T = len(tr.pose)
            i0 = int(self.rng.integers(0, T - self.W + 1)) if T > self.W else 0
            out.append((tr, i0, min(T, i0 + self.W)))
        return out


# ---------------------------------------------------------------- loss
def target_norm(trials, W) -> dict[str, list[float]]:
    Y = np.array([window_labels(tr, a, b) for tr in trials for a, b in tile_windows(tr, W)])
    return {"mean": np.nanmean(Y, 0).tolist(), "std": np.nanstd(Y, 0).tolist()}


def losses(model, batch, norm, pos_weight=5.0):
    logits, heads = model(batch["f"], batch["mask"])
    m = batch["mask"][..., None]
    bce = F.binary_cross_entropy_with_logits(logits, batch["ev"], reduction="none",
                                             pos_weight=torch.tensor(pos_weight))
    out = {"events": (bce * m).sum() / (m.sum() * len(EVENTS))}
    yz = (batch["y"] - norm["mean"]) / norm["std"]
    col = 0
    for k, names in TARGETS.items():
        t = yz[:, col:col + len(names)]
        ok = torch.isfinite(t)
        col += len(names)
        if ok.any():
            out[k] = F.smooth_l1_loss(heads[k][ok], t[ok])
        else:
            out[k] = heads[k].sum() * 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="base")
    ap.add_argument("--window", type=float, default=5.0, help="seconds")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=300, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--event-weight", type=float, default=2.0)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--no-aug", action="store_true", help="train on clean mocap only")
    ap.add_argument("--val-noise", type=float, default=0.02, help="noise level for model selection")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    W = int(round(a.window * FS))
    out = MODELS / a.name
    out.mkdir(parents=True, exist_ok=True)

    data = load_all()
    print({k: len(v) for k, v in data.items()})
    norm = target_norm(data["train"], W)
    normt = {k: torch.tensor(v) for k, v in norm.items()}
    cfg = vars(a) | {"W": W, "targets": TARGETS, "events": EVENTS}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "norm.json").write_text(json.dumps(norm | {"names": TARGET_NAMES}, indent=1))

    sampler = Sampler(data["train"], W, rng)
    noise_fn = (lambda r: augment.Noise()) if a.no_aug else augment.Noise.random
    val_items = [(tr, i0, i1) for tr in data["val"] for i0, i1 in tile_windows(tr, W)]
    vrng = np.random.default_rng(1234)
    val_batches = [make_batch(val_items[i:i + 256], W, vrng, lambda r: augment.Noise.level(a.val_noise))
                   for i in range(0, len(val_items), 256)]
    print(f"train trials {len(data['train'])}, val windows {len(val_items)}")

    model = GaitNet(a.hidden, a.layers)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    best, bad, hist = np.inf, 0, []
    for ep in range(a.epochs):
        t0 = time.time()
        model.train()
        tr_l = []
        for _ in range(a.steps):
            L = losses(model, make_batch(sampler.draw(a.batch), W, rng, noise_fn), normt)
            total = a.event_weight * L["events"] + sum(v for k, v in L.items() if k != "events")
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_l.append({k: float(v) for k, v in L.items()} | {"total": float(total)})
        model.eval()
        with torch.no_grad():
            vl = [losses(model, b, normt) for b in val_batches]
        w = np.array([len(b["mask"]) for b in val_batches])
        val = {k: float(np.average([float(v[k]) for v in vl], weights=w)) for k in vl[0]}
        val["total"] = a.event_weight * val["events"] + sum(v for k, v in val.items() if k != "events")
        row = {"epoch": ep, "lr": opt.param_groups[0]["lr"], "sec": time.time() - t0,
               **{f"train_{k}": np.mean([d[k] for d in tr_l]) for k in tr_l[0]},
               **{f"val_{k}": v for k, v in val.items()}}
        hist.append(row)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        sched.step(val["total"])
        flag = ""
        if val["total"] < best - 1e-4:
            best, bad, flag = val["total"], 0, " *"
            torch.save(model.state_dict(), out / "model.pt")
        else:
            bad += 1
        print(f"ep {ep:2d} {row['sec']:5.0f}s train {row['train_total']:.3f} val {val['total']:.3f} | "
              + " ".join(f"{k[:5]} {v:.3f}" for k, v in val.items() if k != "total") + flag, flush=True)
        if bad >= a.patience:
            print("early stop")
            break


if __name__ == "__main__":
    main()
