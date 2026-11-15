"""Evaluate per recording, as a physio would use it: each held-out trial gets one noise draw over its whole
length, and the strides of all its 5 s windows are pooled before computing the outputs.

    ./py src/evaluate_trials.py [--levels 0 0.01 0.02 0.04]   -> outputs/eval/test_trial/ (the demo's error table)
"""
import argparse

import numpy as np
import pandas as pd
import torch

import augment
import classic
from evaluate import EVAL, RUN, load_model, regression_table, run_model
from helpers.canonical import sagittal_angles
from helpers.gaitdata import ANG_NAMES, FS, TARGET_NAMES, TI, aggregate, load_all, strides_from_events, tile_windows, \
    zeni_offsets
from model import decode_events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=float, nargs="+", default=[0, 0.01, 0.02, 0.04])
    a = ap.parse_args()
    torch.set_num_threads(4)
    m, cfg, norm = load_model()
    trials, offsets, W = load_all()["test"], zeni_offsets(), cfg["W"]
    rows = []
    for lev in a.levels:
        rng = np.random.default_rng(int(lev * 1e4) + 11)
        noisy = [augment.apply(tr.pose, rng, augment.Noise.level(lev)) for tr in trials]
        items = [(k, i0, i1) for k, tr in enumerate(trials) for i0, i1 in tile_windows(tr, W)]
        X = [noisy[k][0][i0:i1] for k, i0, i1 in items]
        V = [noisy[k][1][i0:i1] for k, i0, i1 in items]
        probs, heads, angs = run_model(m, norm, X, V)
        per_trial = [dict(classic=[], model=[], heads=[], classic_speed=[], model_speed=[]) for _ in trials]
        for j, ((k, i0, _), x, v) in enumerate(zip(items, X, V)):
            ev, y = classic.predict(x, v, offsets)
            ca = sagittal_angles(classic.to_sag(x, v))
            for key, S in (("classic", strides_from_events(ev, np.stack([ca[n] for n in ANG_NAMES], -1))),
                           ("model", strides_from_events(decode_events(probs[j]), angs[j]))):
                S[:, :2] += i0 / FS                          # window time -> trial time
                per_trial[k][key].append(S)
            per_trial[k]["classic_speed"].append(y[TI["speed_ll"]])
            per_trial[k]["model_speed"].append(heads[j][TI["speed_ll"]])
            per_trial[k]["heads"].append(heads[j])
        for tr, d in zip(trials, per_trial):
            T = len(tr.pose)
            pool = lambda key: aggregate(np.concatenate(d[key]), 0, (T - 1) / FS,  # noqa: E731
                                         np.nanmedian(d[f"{key}_speed"]) if np.isfinite(d[f"{key}_speed"]).any() else np.nan)
            head_mean = np.mean(d["heads"], 0)
            preds = {"mean": np.array(norm["mean"]), "classic": pool("classic"),
                     f"{RUN}+derived": np.where(np.isfinite(pool("model")), pool("model"), head_mean), RUN: head_mean}
            true = aggregate(tr.strides, 0, (T - 1) / FS, tr.speed_ll)
            base = dict(dataset=tr.ds, trial=tr.trial, subject=tr.subject, mode=tr.mode, level=lev, leg_length=tr.leg_length,
                        **{f"true_{n}": true[i] for i, n in enumerate(TARGET_NAMES)})
            for meth, p in preds.items():
                rows.append(base | dict(method=meth) | {f"pred_{n}": p[i] for i, n in enumerate(TARGET_NAMES)})
        print(f"noise level {lev}: done", flush=True)

    out = EVAL / "test_trial"
    out.mkdir(parents=True, exist_ok=True)
    P = pd.DataFrame(rows)
    for c in ("true", "pred"):
        P[f"{c}_speed_ms"] = P[f"{c}_speed_ll"] * P.leg_length
    P.to_parquet(out / "predictions.parquet", index=False)
    regression_table(P, ["dataset", "level", "method"]).to_csv(out / "regression.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
