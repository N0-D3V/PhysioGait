"""Evaluate trained models vs the classic pipeline and the mean predictor on held-out subjects.

    ./py src/evaluate.py bigru_v2 bigru_aug bigru_clean [--split test] [--levels 0 0.01 0.02 0.04 0.08]

Methods: each run's pooled heads (`<run>`), and for runs with an angle head `<run>+derived`:
stride outputs computed from the predicted events + angle curves with the label formulas
(speed from the pooled head; pooled value used where no complete stride was decoded).

Every method sees the *same* noisy input at each noise level. Windows are non-overlapping
X-second tiles of each held-out trial (short overground trials = one window). Writes
outputs/eval/<split>/{predictions.parquet, regression.csv, events.csv, angles.csv, symmetry_bins.csv}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import augment  # noqa: E402
import classic  # noqa: E402
from analysis import ROOT  # noqa: E402
from canonical import sagittal_angles  # noqa: E402
from gaitdata import (ANG_NAMES, EVENTS, FS, TARGET_NAMES, TARGETS, TI, derive, load_all,  # noqa: E402
                      tile_windows, window_labels, zeni_offsets)
from model import GaitNet, decode_events, features  # noqa: E402
from train import MODELS, by_length  # noqa: E402

EVAL = ROOT / "outputs" / "eval"
EDGE_S = 0.2      # ignore events this close to a window edge (no method can see both sides)
TOL_S = 0.10      # event detection tolerance


def load_model(name: str):
    cfg = json.loads((MODELS / name / "config.json").read_text())
    norm = json.loads((MODELS / name / "norm.json").read_text())
    m = GaitNet(cfg["hidden"], cfg["layers"], feat=cfg.get("features", "v1"),
                angle_head=cfg.get("angle_head", False))
    m.load_state_dict(torch.load(MODELS / name / "model.pt", weights_only=True))
    return m.eval(), cfg, norm


@torch.no_grad()
def run_model(m, norm, X, V, feat="v2"):
    """X, V: lists of windows -> (event probs list, pooled targets [N, n], angle curves list | None)."""
    probs, ys, angs = [None] * len(X), np.zeros((len(X), len(TARGET_NAMES))), [None] * len(X)
    idx = list(range(len(X)))
    for chunk in by_length([(i, 0, len(X[i])) for i in idx], 256):
        ii = [c[0] for c in chunk]
        f = torch.from_numpy(features(np.stack([X[i] for i in ii]), np.stack([V[i] for i in ii]), feat))
        logits, heads = m(f, torch.ones(f.shape[:2]))
        p = torch.sigmoid(logits).numpy()
        y = torch.cat([heads[k] for k in TARGETS], -1).numpy() * norm["std"] + np.array(norm["mean"])
        a = heads["angles"].numpy() * norm["ang_std"] + np.array(norm["ang_mean"]) if "angles" in heads else None
        for j, i in enumerate(ii):
            probs[i], ys[i] = p[j], y[j]
            if a is not None:
                angs[i] = a[j]
    return probs, ys, (angs if angs[0] is not None else None)


def match(true: np.ndarray, pred: np.ndarray, tol: float):
    """One-to-one greedy matching by distance -> (signed errors of matches, n_fp, n_fn)."""
    pairs = sorted((abs(p - t), i, j) for i, t in enumerate(true) for j, p in enumerate(pred) if abs(p - t) <= tol)
    ut, up, err = set(), set(), []
    for _, i, j in pairs:
        if i not in ut and j not in up:
            ut.add(i), up.add(j), err.append(pred[j] - true[i])
    return err, len(pred) - len(up), len(true) - len(ut)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--split", default="test")
    ap.add_argument("--levels", type=float, nargs="+", default=[0, 0.01, 0.02, 0.04, 0.08])
    ap.add_argument("--out", default=None, help="output dir name under outputs/eval (default: the split)")
    a = ap.parse_args()
    torch.set_num_threads(4)

    trials = load_all()[a.split]
    models = {r: load_model(r) for r in a.runs}
    W = next(iter(models.values()))[1]["W"]
    assert all(c["W"] == W for _, c, _ in models.values()), "runs must share the window length"
    mean_y = np.array(next(iter(models.values()))[2]["mean"])
    offsets = zeni_offsets()
    items = [(tr, i0, i1) for tr in trials for i0, i1 in tile_windows(tr, W)]
    Y = np.array([window_labels(tr, i0, i1) for tr, i0, i1 in items])
    meta = pd.DataFrame({"dataset": [t.ds for t, _, _ in items], "trial": [t.trial for t, _, _ in items],
                         "subject": [t.subject for t, _, _ in items], "mode": [t.mode for t, _, _ in items],
                         "i0": [i for _, i, _ in items], "leg_length": [t.leg_length for t, _, _ in items]})
    print(f"{a.split}: {len(items)} windows from {len(trials)} trials")

    preds, ev_rows, ang_rows, fallback = [], [], [], []
    for lev in a.levels:
        rng = np.random.default_rng(int(lev * 1e4) + 7)
        noisy = [augment.apply(tr.pose[i0:i1], rng, augment.Noise.level(lev)) for tr, i0, i1 in items]
        X, V = [n[0] for n in noisy], [n[1] for n in noisy]
        method_ev, method_y = {}, {}
        cl = [classic.predict(x, v, offsets) for x, v in zip(X, V)]
        method_ev["classic"] = [c[0] for c in cl]
        method_y["classic"] = np.array([c[1] for c in cl])
        method_y["mean"] = np.tile(mean_y, (len(items), 1))
        method_ang = {"classic": [np.stack([sagittal_angles(classic.to_sag(x, v))[n] for n in ANG_NAMES], -1)
                                  for x, v in zip(X, V)]}
        for r, (m, cfg, norm) in models.items():
            probs, y, angs = run_model(m, norm, X, V, cfg.get("features", "v1"))
            method_ev[r] = [decode_events(p) for p in probs]
            method_y[r] = y
            if angs is not None:
                d = np.array([derive(e, a_, len(a_), yy[TI["speed_ll"]])
                              for e, a_, yy in zip(method_ev[r], angs, y)])
                fallback.append(dict(level=lev, run=r, **{n: float(np.mean(~np.isfinite(d[:, i]) & np.isfinite(Y[:, i])))
                                                          for i, n in enumerate(TARGET_NAMES)}))
                method_y[f"{r}+derived"] = np.where(np.isfinite(d), d, y)
                method_ang[r] = angs
        for meth, angs in method_ang.items():
            for (tr, i0, i1), a_ in zip(items, angs):
                err = a_ - tr.ang[i0:i1]
                ang_rows.append(dict(dataset=tr.ds, level=lev, method=meth, n=len(err),
                                     **{n: float((err[:, k] ** 2).sum()) for k, n in enumerate(ANG_NAMES)}))
        for meth, y in method_y.items():
            preds.append(meta.assign(level=lev, method=meth, **{f"true_{n}": Y[:, i] for i, n in enumerate(TARGET_NAMES)},
                                     **{f"pred_{n}": y[:, i] for i, n in enumerate(TARGET_NAMES)}))
        for meth, evs in method_ev.items():
            for (tr, i0, i1), pe in zip(items, evs):
                lo, hi = i0 / FS + EDGE_S, (i1 - 1) / FS - EDGE_S
                for k in EVENTS:
                    t = tr.events[k][(tr.events[k] >= lo) & (tr.events[k] <= hi)]
                    p = pe[k] + i0 / FS
                    p = p[(p >= lo) & (p <= hi)]
                    err, fp, fn = match(t, p, TOL_S)
                    ev_rows.append(dict(dataset=tr.ds, level=lev, method=meth, event=k[:2], tp=len(err), fp=fp,
                                        fn=fn, err_sum=sum(err), abs_err_sum=sum(abs(e) for e in err)))
        print(f"level {lev}: done", flush=True)

    out = EVAL / (a.out or a.split)
    out.mkdir(parents=True, exist_ok=True)
    P = pd.concat(preds, ignore_index=True)
    for c in ("true", "pred"):
        P[f"{c}_speed_ms"] = P[f"{c}_speed_ll"] * P.leg_length
    P.to_parquet(out / "predictions.parquet", index=False)

    names = TARGET_NAMES + ["speed_ms"]
    reg = []
    for (ds, lev, meth), g in P.groupby(["dataset", "level", "method"]):
        for n in names:
            t, p = g[f"true_{n}"].to_numpy(), g[f"pred_{n}"].to_numpy()
            ok = np.isfinite(t)
            cov = np.isfinite(p[ok]).mean() if ok.any() else np.nan
            ok &= np.isfinite(p)
            if ok.sum() < 3:
                continue
            e = p[ok] - t[ok]
            ss = ((t[ok] - t[ok].mean()) ** 2).sum()
            reg.append(dict(dataset=ds, level=lev, method=meth, target=n, n=int(ok.sum()), coverage=cov,
                            mae=np.abs(e).mean(), bias=e.mean(),
                            r2=1 - (e ** 2).sum() / ss if ss > 0 else np.nan,
                            r=np.corrcoef(t[ok], p[ok])[0, 1] if ss > 0 and p[ok].std() > 0 else np.nan,
                            slope=np.polyfit(t[ok], p[ok], 1)[0] if ss > 0 else np.nan))
    pd.DataFrame(reg).to_csv(out / "regression.csv", index=False)

    # symmetry error by size of the true asymmetry: does a large SI get through?
    bins = [0, 3, 7, np.inf]
    sb = []
    for n in TARGETS["symmetry"]:
        q = P[np.isfinite(P[f"true_{n}"]) & np.isfinite(P[f"pred_{n}"])]
        q = q.assign(abs_bin=pd.cut(q[f"true_{n}"].abs(), bins, right=False, labels=["<3", "3-7", ">=7"]),
                     err=(q[f"pred_{n}"] - q[f"true_{n}"]).abs(),
                     under=np.abs(q[f"pred_{n}"]) - np.abs(q[f"true_{n}"]))
        g = q.groupby(["dataset", "level", "method", "abs_bin"], observed=True).agg(
            n=("err", "size"), mae=("err", "mean"), mean_abs_shrink=("under", "mean")).reset_index()
        sb.append(g.assign(target=n))
    pd.concat(sb).to_csv(out / "symmetry_bins.csv", index=False)

    A = pd.DataFrame(ang_rows).groupby(["dataset", "level", "method"]).sum().reset_index()
    for n in ANG_NAMES:
        A[n] = np.sqrt(A[n] / A.n)
    A.to_csv(out / "angles.csv", index=False)
    if fallback:
        pd.DataFrame(fallback).to_csv(out / "derived_fallback.csv", index=False)

    E = pd.DataFrame(ev_rows).groupby(["dataset", "level", "method", "event"]).sum().reset_index()
    E["precision"] = E.tp / (E.tp + E.fp)
    E["recall"] = E.tp / (E.tp + E.fn)
    E["f1"] = 2 * E.tp / (2 * E.tp + E.fp + E.fn)
    E["mae_ms"] = 1000 * E.abs_err_sum / E.tp
    E["bias_ms"] = 1000 * E.err_sum / E.tp
    E.drop(columns=["err_sum", "abs_err_sum"]).to_csv(out / "events.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
