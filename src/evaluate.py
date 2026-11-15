"""Evaluate the model against the classic method and "guess the average" on held-out people, per 5 s window,
at several levels of simulated video-pose noise.

    ./py src/evaluate.py [--levels 0 0.01 0.02 0.04 0.08]   -> outputs/eval/test/

Methods: `bigru_v5+derived` (outputs computed from the predicted events and angles, the main method),
`bigru_v5` (the model's whole-window heads), `classic`, `mean`.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

import augment
import classic
from helpers.analysis import ROOT
from helpers.canonical import sagittal_angles
from helpers.gaitdata import ANG_NAMES, EVENTS, FS, TARGET_NAMES, TARGETS, TI, derive, load_all, tile_windows, \
    window_labels, zeni_offsets
from model import GaitNet, decode_events, features

RUN = os.environ.get("GAIT_MODEL", "bigru_v5")   # the model used everywhere; GAIT_MODEL=bigru_v4 to compare
MODELS = ROOT / "outputs" / "models"
EVAL = ROOT / "outputs" / "eval"
EDGE = 0.2   # events this close to a window edge are not scored
TOL = 0.1    # event matching tolerance (s)


def load_model(name=RUN):
    cfg = json.loads((MODELS / name / "config.json").read_text())
    norm = json.loads((MODELS / name / "norm.json").read_text())
    m = GaitNet(cfg["hidden"], cfg["layers"])
    m.load_state_dict(torch.load(MODELS / name / "model.pt", weights_only=True))
    return m.eval(), cfg, norm


@torch.no_grad()
def run_model(m, norm, X, V):
    """Lists of windows -> (event probabilities, whole-window head outputs [N, targets], angle curves)."""
    probs, ys, angs = [None] * len(X), np.zeros((len(X), len(TARGET_NAMES))), [None] * len(X)
    by_len = {}
    for i, x in enumerate(X):
        by_len.setdefault(len(x), []).append(i)
    for ids in by_len.values():
        for k in range(0, len(ids), 256):
            ii = ids[k:k + 256]
            f = torch.from_numpy(features(np.stack([X[i] for i in ii]), np.stack([V[i] for i in ii])))
            logits, heads = m(f)
            p = torch.sigmoid(logits).numpy()
            y = torch.cat([heads[t] for t in TARGETS], -1).numpy() * norm["std"] + np.array(norm["mean"])
            a = heads["angles"].numpy() * norm["ang_std"] + np.array(norm["ang_mean"])
            for j, i in enumerate(ii):
                probs[i], ys[i], angs[i] = p[j], y[j], a[j]
    return probs, ys, angs


def match(true, pred, tol):
    """Pair each true event with the closest prediction (one-to-one) -> (errors, n false alarms, n missed)."""
    pairs = sorted((abs(p - t), i, j) for i, t in enumerate(true) for j, p in enumerate(pred) if abs(p - t) <= tol)
    used_t, used_p, err = set(), set(), []
    for _, i, j in pairs:
        if i not in used_t and j not in used_p:
            used_t.add(i), used_p.add(j), err.append(pred[j] - true[i])
    return err, len(pred) - len(used_p), len(true) - len(used_t)


def regression_metrics(t, p):
    """Error statistics of predictions p against truth t. slope = predicted vs true (1 = full size)."""
    has_t = np.isfinite(t)
    ok = has_t & np.isfinite(p)
    out = dict(n=int(ok.sum()), coverage=ok[has_t].mean() if has_t.any() else np.nan)
    if ok.sum() < 3:
        return out
    t, p = t[ok], p[ok]
    e, spread = p - t, t.std() > 0
    return out | dict(mae=np.abs(e).mean(), bias=e.mean(), sd=e.std(),
                      r2=1 - (e ** 2).sum() / ((t - t.mean()) ** 2).sum() if spread else np.nan,
                      r=np.corrcoef(t, p)[0, 1] if spread and p.std() > 0 else np.nan,
                      slope=np.polyfit(t, p, 1)[0] if spread else np.nan,
                      mean_pred_mae=np.abs(t - t.mean()).mean())


def regression_table(P, by):
    rows = []
    for key, g in P.groupby(by):
        for n in TARGET_NAMES + ["speed_ms"]:
            m = regression_metrics(g[f"true_{n}"].to_numpy(), g[f"pred_{n}"].to_numpy())
            if "mae" in m:
                rows.append(dict(zip(by, key), target=n, **m))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=float, nargs="+", default=[0, 0.01, 0.02, 0.04, 0.08])
    ap.add_argument("--out", default="test")
    a = ap.parse_args()
    torch.set_num_threads(4)
    m, cfg, norm = load_model()
    trials, offsets, W = load_all()["test"], zeni_offsets(), cfg["W"]
    items = [(tr, i0, i1) for tr in trials for i0, i1 in tile_windows(tr, W)]
    Y = np.array([window_labels(tr, i0, i1) for tr, i0, i1 in items])
    meta = pd.DataFrame([dict(dataset=t.ds, trial=t.trial, subject=t.subject, mode=t.mode, i0=i0, leg_length=t.leg_length)
                         for t, i0, _ in items])
    print(f"{len(items)} windows from {len(trials)} test trials")

    preds, ev_rows = [], []
    for lev in a.levels:
        rng = np.random.default_rng(int(lev * 1e4) + 7)
        noisy = [augment.apply(tr.pose[i0:i1], rng, augment.Noise.level(lev)) for tr, i0, i1 in items]
        X, V = [n[0] for n in noisy], [n[1] for n in noisy]
        classic_out = [classic.predict(x, v, offsets) for x, v in zip(X, V)]
        probs, heads, angs = run_model(m, norm, X, V)
        model_ev = [decode_events(p) for p in probs]
        derived = np.array([derive(e, an, len(an), y[TI["speed_ll"]]) for e, an, y in zip(model_ev, angs, heads)])
        outputs = {"classic": np.array([c[1] for c in classic_out]), "mean": np.tile(norm["mean"], (len(items), 1)),
                   RUN: heads, f"{RUN}+derived": np.where(np.isfinite(derived), derived, heads)}
        for meth, y in outputs.items():
            preds.append(meta.assign(level=lev, method=meth, **{f"true_{n}": Y[:, i] for i, n in enumerate(TARGET_NAMES)},
                                     **{f"pred_{n}": y[:, i] for i, n in enumerate(TARGET_NAMES)}))
        for meth, evs in (("classic", [c[0] for c in classic_out]), (RUN, model_ev)):
            for (tr, i0, i1), pe in zip(items, evs):
                lo, hi = i0 / FS + EDGE, (i1 - 1) / FS - EDGE
                for k in EVENTS:
                    t = tr.events[k][(tr.events[k] >= lo) & (tr.events[k] <= hi)]
                    p = pe[k] + i0 / FS
                    err, fp, fn = match(t, p[(p >= lo) & (p <= hi)], TOL)
                    ev_rows.append(dict(dataset=tr.ds, level=lev, method=meth, event=k[:2], tp=len(err), fp=fp, fn=fn,
                                        err_sum=sum(err), abs_err_sum=sum(abs(e) for e in err)))
        print(f"noise level {lev}: done", flush=True)

    out = EVAL / a.out
    out.mkdir(parents=True, exist_ok=True)
    P = pd.concat(preds, ignore_index=True)
    for c in ("true", "pred"):
        P[f"{c}_speed_ms"] = P[f"{c}_speed_ll"] * P.leg_length
    P.to_parquet(out / "predictions.parquet", index=False)
    regression_table(P, ["dataset", "level", "method"]).to_csv(out / "regression.csv", index=False)
    E = pd.DataFrame(ev_rows).groupby(["dataset", "level", "method", "event"]).sum().reset_index()
    E["recall"], E["precision"] = E.tp / (E.tp + E.fn), E.tp / (E.tp + E.fp)
    E["mae_ms"], E["bias_ms"] = 1000 * E.abs_err_sum / E.tp, 1000 * E.err_sum / E.tp
    E.drop(columns=["err_sum", "abs_err_sum"]).to_csv(out / "events.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
