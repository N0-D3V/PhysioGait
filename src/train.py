"""Train the multi-head BiGRU.

    ./py src/train.py --name bigru_v4 --features v3 --angle-beta 0.1 --asym-p 0.5
    ./py src/train.py --name bigru_v3 --features v3 --angle-beta 0.1
    ./py src/train.py --name bigru_v2 [--window 5] [--epochs 50] [--no-aug] [--swap-p 0.5]
    ./py src/train.py --name bigru_aug --features v1 --no-angle-head --swap-p 0   # first model

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
from gaitdata import EVENTS, FS, TARGET_NAMES, TARGETS, TI, load_all, tile_windows, window_labels  # noqa: E402
from model import GaitNet, features  # noqa: E402

MODELS = ROOT / "outputs" / "models"
SI_IDX = [TI[n] for n in TARGETS["symmetry"]]


# ---------------------------------------------------------------- batching
def by_length(items, size: int):
    """Chunk windows into batches of equal length."""
    groups = {}
    for it in items:
        groups.setdefault(it[2] - it[1], []).append(it)
    return [g[i:i + size] for g in groups.values() for i in range(0, len(g), size)]


def make_batch(items, W: int, rng, noise_fn, swap_p: float = 0.0, feat: str = "v2", asym_p: float = 0.0):
    """items: [(Trial, i0, i1)] -> dict of tensors, padded to the longest window.

    swap_p: probability of mirroring left/right (pose, events, angles, SI sign).
    asym_p: probability of a synthetic asymmetry (one leg phase-shifted and its excursions scaled).
    """
    B = len(items)
    W = max(i1 - i0 for _, i0, i1 in items)
    x = np.zeros((B, W, 10, 2), np.float32)
    vis = np.zeros((B, W, 10), np.float32)
    mask = np.zeros((B, W), np.float32)
    ev = np.zeros((B, W, len(EVENTS)), np.float32)
    y = np.full((B, len(TARGET_NAMES)), np.nan, np.float32)
    ang = np.zeros((B, W, 6), np.float32)
    for b, (tr, i0, i1) in enumerate(items):
        n = i1 - i0
        if asym_p and rng.random() < asym_p:
            pose, evb, angb, yb = augment.asymmetric_window(tr, i0, i1, rng)
        else:
            pose, evb, angb, yb = tr.pose[i0:i1], tr.ev_soft[i0:i1], tr.ang[i0:i1], window_labels(tr, i0, i1)
        if swap_p and rng.random() < swap_p:
            pose, evb, angb, yb = augment.swap_sides(pose, evb, angb, yb, SI_IDX)
        x[b, :n], vis[b, :n] = augment.apply(pose, rng, noise_fn(rng))
        mask[b, :n] = 1
        ev[b, :n], ang[b, :n], y[b] = evb, angb, yb
    f = features(x, vis, feat) * mask[..., None]
    return {"f": torch.from_numpy(f), "mask": torch.from_numpy(mask), "ev": torch.from_numpy(ev),
            "y": torch.from_numpy(y), "ang": torch.from_numpy(ang)}


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
        """One group per batch, all windows cropped to a common length (no padding)."""
        g = self.groups[self.rng.choice(len(self.groups), p=self.p)]
        trs = [g[i] for i in self.rng.integers(len(g), size=B)]
        n = min(self.W, *(len(tr.pose) for tr in trs))
        out = []
        for tr in trs:
            i0 = int(self.rng.integers(0, len(tr.pose) - n + 1))
            out.append((tr, i0, i0 + n))
        return out


# ---------------------------------------------------------------- loss
def target_norm(trials, W) -> dict[str, list[float]]:
    Y = np.array([window_labels(tr, a, b) for tr in trials for a, b in tile_windows(tr, W)])
    A = np.concatenate([tr.ang for tr in trials])
    return {"mean": np.nanmean(Y, 0).tolist(), "std": np.nanstd(Y, 0).tolist(),
            "ang_mean": A.mean(0).tolist(), "ang_std": A.std(0).tolist()}


def losses(model, batch, norm, pos_weight=5.0, angle_beta=1.0):
    logits, heads = model(batch["f"], batch["mask"])
    m = batch["mask"][..., None]
    bce = F.binary_cross_entropy_with_logits(logits, batch["ev"], reduction="none",
                                             pos_weight=torch.tensor(pos_weight))
    out = {"events": (bce * m).sum() / (m.sum() * len(EVENTS))}
    if "angles" in heads:
        az = (batch["ang"] - norm["ang_mean"]) / norm["ang_std"]
        al = F.smooth_l1_loss(heads["angles"], az, reduction="none", beta=angle_beta)
        out["angles"] = (al * m).sum() / (m.sum() * 6)
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
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps", type=int, default=200, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--event-weight", type=float, default=2.0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--features", default="v2", choices=["v1", "v2", "v3"])
    ap.add_argument("--no-angle-head", action="store_true")
    ap.add_argument("--angle-weight", type=float, default=1.0)
    ap.add_argument("--angle-beta", type=float, default=1.0, help="smooth-L1 beta (z units) for angles")
    ap.add_argument("--swap-p", type=float, default=0.5, help="left/right mirror probability")
    ap.add_argument("--asym-p", type=float, default=0.0, help="synthetic asymmetry probability")
    ap.add_argument("--no-aug", action="store_true", help="train on clean mocap only")
    ap.add_argument("--val-noise", type=float, default=0.02, help="noise level for model selection")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
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
    cfg = vars(a) | {"W": W, "targets": TARGETS, "events": EVENTS, "angle_head": not a.no_angle_head}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "norm.json").write_text(json.dumps(norm | {"names": TARGET_NAMES}, indent=1))

    sampler = Sampler(data["train"], W, rng)
    noise_fn = (lambda r: augment.Noise()) if a.no_aug else augment.Noise.random
    val_items = [(tr, i0, i1) for tr in data["val"] for i0, i1 in tile_windows(tr, W)]
    vrng = np.random.default_rng(1234)
    val_batches = [make_batch(c, W, vrng, lambda r: augment.Noise.level(a.val_noise), feat=a.features)
                   for c in by_length(val_items, 256)]
    print(f"train trials {len(data['train'])}, val windows {len(val_items)}")

    model = GaitNet(a.hidden, a.layers, feat=a.features, angle_head=not a.no_angle_head)
    if a.features == "v3":
        model.ang_mean.copy_(normt["ang_mean"]), model.ang_std.copy_(normt["ang_std"])
    weight = {"events": a.event_weight, "angles": a.angle_weight}
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    best, bad, hist = np.inf, 0, []
    for ep in range(a.epochs):
        t0 = time.time()
        model.train()
        tr_l = []
        for _ in range(a.steps):
            L = losses(model, make_batch(sampler.draw(a.batch), W, rng, noise_fn, a.swap_p, a.features, a.asym_p), normt,
                       angle_beta=a.angle_beta)
            total = sum(weight.get(k, 1.0) * v for k, v in L.items())
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_l.append({k: v.item() for k, v in L.items()} | {"total": total.item()})
        model.eval()
        with torch.no_grad():
            vl = [losses(model, b, normt, angle_beta=a.angle_beta) for b in val_batches]
        w = np.array([len(b["mask"]) for b in val_batches])
        val = {k: float(np.average([float(v[k]) for v in vl], weights=w)) for k in vl[0]}
        val["total"] = sum(weight.get(k, 1.0) * v for k, v in val.items())
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
