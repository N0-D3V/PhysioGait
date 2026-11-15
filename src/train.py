"""Train the gait model on mocap made to look like video (pose noise, left/right mirroring, synthetic limps).

    ./py src/train.py [--name bigru_v4] [--epochs 50]   -> outputs/models/<name>/ (~2.5 h on CPU)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import augment
from helpers.analysis import ROOT
from helpers.gaitdata import EVENTS, FS, TARGET_NAMES, TARGETS, TI, load_all, tile_windows, window_labels
from model import GaitNet, features

W = int(5 * FS)                  # 5 s windows
BATCH, STEPS, LR, PATIENCE = 64, 200, 1e-3, 10
EVENT_WEIGHT, ANGLE_BETA = 2.0, 0.1
SI_IDX = [TI[n] for n in TARGETS["symmetry"]]


def make_batch(items, rng, noise, augment_legs=False):
    """items [(trial, i0, i1)] of equal length -> tensors. augment_legs: random synthetic limp and L/R mirror."""
    xs, vs, evs, angs, ys = [], [], [], [], []
    for tr, i0, i1 in items:
        if augment_legs and rng.random() < 0.5:
            pose, ev, ang, y = augment.asymmetric_window(tr, i0, i1, rng)
        else:
            pose, ev, ang, y = tr.pose[i0:i1], tr.ev_soft[i0:i1], tr.ang[i0:i1], window_labels(tr, i0, i1)
        if augment_legs and rng.random() < 0.5:
            pose, ev, ang, y = augment.swap_sides(pose, ev, ang, y, SI_IDX)
        x, v = augment.apply(pose, rng, noise(rng))
        xs.append(x), vs.append(v), evs.append(ev), angs.append(ang), ys.append(y)
    t = lambda a: torch.from_numpy(np.stack(a).astype(np.float32))  # noqa: E731
    return {"f": torch.from_numpy(features(np.stack(xs), np.stack(vs))), "ev": t(evs), "ang": t(angs), "y": t(ys)}


def draw_windows(groups, rng):
    """A training batch: pick a data group (50 % Ferber, 25 % Fukuchi treadmill, 25 % Fukuchi overground),
    then random trials from it, all cropped to the same length."""
    g = groups[rng.choice(len(groups), p=[0.5, 0.25, 0.25])]
    trs = [g[i] for i in rng.integers(len(g), size=BATCH)]
    n = min(W, *(len(tr.pose) for tr in trs))
    return [(tr, i0, i0 + n) for tr in trs for i0 in [int(rng.integers(0, len(tr.pose) - n + 1))]]


def losses(model, batch, norm):
    logits, heads = model(batch["f"])
    out = {"events": F.binary_cross_entropy_with_logits(logits, batch["ev"], pos_weight=torch.tensor(5.0)),
           "angles": F.smooth_l1_loss(heads["angles"], (batch["ang"] - norm["ang_mean"]) / norm["ang_std"], beta=ANGLE_BETA)}
    yz = (batch["y"] - norm["mean"]) / norm["std"]
    col = 0
    for k, names in TARGETS.items():
        t = yz[:, col:col + len(names)]
        ok = torch.isfinite(t)                       # labels can be missing (e.g. no full stride in the window)
        out[k] = F.smooth_l1_loss(heads[k][ok], t[ok]) if ok.any() else heads[k].sum() * 0
        col += len(names)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="bigru_v4")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    out = ROOT / "outputs" / "models" / a.name
    out.mkdir(parents=True, exist_ok=True)

    data = load_all()
    train = data["train"]
    Y = np.array([window_labels(tr, i0, i1) for tr in train for i0, i1 in tile_windows(tr, W)])
    A = np.concatenate([tr.ang for tr in train])
    norm = {"mean": np.nanmean(Y, 0).tolist(), "std": np.nanstd(Y, 0).tolist(),
            "ang_mean": A.mean(0).tolist(), "ang_std": A.std(0).tolist()}
    (out / "config.json").write_text(json.dumps({"hidden": 128, "layers": 2, "W": W, "epochs": a.epochs, "seed": a.seed}))
    (out / "norm.json").write_text(json.dumps(norm | {"names": TARGET_NAMES}, indent=1))
    normt = {k: torch.tensor(v) for k, v in norm.items()}

    groups = [[t for t in train if t.ds == "ferber"], [t for t in train if t.ds == "fukuchi" and t.mode == "treadmill"],
              [t for t in train if t.ds == "fukuchi" and t.mode == "overground"]]
    val_items = {}
    for tr in data["val"]:
        for i0, i1 in tile_windows(tr, W):
            val_items.setdefault(i1 - i0, []).append((tr, i0, i1))
    vrng = np.random.default_rng(1234)
    val_batches = [make_batch(g[i:i + 256], vrng, lambda r: augment.Noise.level(0.02))
                   for g in val_items.values() for i in range(0, len(g), 256)]

    model = GaitNet()
    model.ang_mean.copy_(normt["ang_mean"]), model.ang_std.copy_(normt["ang_std"])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    weight = {"events": EVENT_WEIGHT}
    best, bad, hist = np.inf, 0, []
    for ep in range(a.epochs):
        t0 = time.time()
        model.train()
        train_loss = []
        for _ in range(STEPS):
            L = losses(model, make_batch(draw_windows(groups, rng), rng, augment.Noise.random, augment_legs=True), normt)
            total = sum(weight.get(k, 1.0) * v for k, v in L.items())
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss.append(total.item())
        model.eval()
        with torch.no_grad():
            vl = [losses(model, b, normt) for b in val_batches]
        sizes = [len(b["f"]) for b in val_batches]
        val = {k: float(np.average([float(v[k]) for v in vl], weights=sizes)) for k in vl[0]}
        val_total = sum(weight.get(k, 1.0) * v for k, v in val.items())
        hist.append({"epoch": ep, "train_total": np.mean(train_loss), "val_total": val_total} | {f"val_{k}": v for k, v in val.items()})
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        sched.step(val_total)
        if val_total < best - 1e-4:
            best, bad = val_total, 0
            torch.save(model.state_dict(), out / "model.pt")
        else:
            bad += 1
        print(f"epoch {ep:2d} ({time.time() - t0:.0f}s): train {np.mean(train_loss):.3f}  val {val_total:.3f}"
              + ("  saved" if bad == 0 else ""), flush=True)
        if bad >= PATIENCE:
            break


if __name__ == "__main__":
    main()
