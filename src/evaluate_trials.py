"""Recording-level evaluation: one output per trial, as a physio would get from one video.

    ./py src/evaluate_trials.py bigru_v4 bigru_v3 [--split test] [--levels 0 0.01 0.02 0.04]

Each trial gets ONE noise draw over its whole length (fixed far leg, keypoint bias, scale and
camera tilt, as in a real recording), is tiled into windows, and every method's per-window strides
are pooled before computing the outputs, so symmetry is measured over all strides of the trial
(30 s treadmill trials ≈ 50 strides; Toronto passes ≈ 9 s; Fukuchi overground = one 3 s pass).
Methods: `<run>+derived` (events + angle curves -> formulas, strides pooled), `<run>` (pooled
heads, averaged over windows), classic (strides pooled), mean.
Writes outputs/eval/<split>_trial/{predictions.parquet, regression.csv, symmetry_bins.csv}.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import augment  # noqa: E402
import classic  # noqa: E402
from canonical import sagittal_angles  # noqa: E402
from evaluate import EVAL, load_model, run_model  # noqa: E402
from gaitdata import (ANG_NAMES, FS, TARGET_NAMES, TARGETS, TI, aggregate, load_all,  # noqa: E402
                      strides_from_events, tile_windows, zeni_offsets)
from model import decode_events  # noqa: E402


def pooled(strides: list[np.ndarray], T: int, speed_ll: float) -> np.ndarray:
    """Aggregate strides from all windows (already in trial time) over the whole trial."""
    S = np.concatenate(strides) if strides else np.zeros((0, 13))
    return aggregate(S, 0, (T - 1) / FS, speed_ll)


def metrics(P: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in P.groupby(by):
        for n in TARGET_NAMES + ["speed_ms"]:
            t, p = g[f"true_{n}"].to_numpy(), g[f"pred_{n}"].to_numpy()
            ok = np.isfinite(t) & np.isfinite(p)
            if ok.sum() < 3:
                continue
            e = p[ok] - t[ok]
            ss = ((t[ok] - t[ok].mean()) ** 2).sum()
            rows.append(dict(zip(by, key), target=n, n=int(ok.sum()), mae=np.abs(e).mean(), bias=e.mean(),
                             r=np.corrcoef(t[ok], p[ok])[0, 1] if ss > 0 and p[ok].std() > 0 else np.nan,
                             slope=np.polyfit(t[ok], p[ok], 1)[0] if ss > 0 else np.nan))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--split", default="test")
    ap.add_argument("--levels", type=float, nargs="+", default=[0, 0.01, 0.02, 0.04])
    a = ap.parse_args()
    torch.set_num_threads(4)

    trials = load_all()[a.split]
    models = {r: load_model(r) for r in a.runs}
    W = next(iter(models.values()))[1]["W"]
    mean_y = np.array(next(iter(models.values()))[2]["mean"])
    offsets = zeni_offsets()
    rows = []
    for lev in a.levels:
        rng = np.random.default_rng(int(lev * 1e4) + 11)
        noisy = [augment.apply(tr.pose, rng, augment.Noise.level(lev)) for tr in trials]
        items = [(k, i0, i1) for k, tr in enumerate(trials) for i0, i1 in tile_windows(tr, W)]
        X = [noisy[k][0][i0:i1] for k, i0, i1 in items]
        V = [noisy[k][1][i0:i1] for k, i0, i1 in items]
        strides = {m: [[] for _ in trials] for m in ["classic"] + [f"{r}+derived" for r in models]}
        heads = {r: [[] for _ in trials] for r in models}
        speed = {m: [[] for _ in trials] for m in strides}
        for (k, i0, _), x, v in zip(items, X, V):
            ev, y = classic.predict(x, v, offsets)
            ang = np.stack([sagittal_angles(classic.to_sag(x, v))[n] for n in ANG_NAMES], -1)
            S = strides_from_events(ev, ang)
            S[:, :2] += i0 / FS
            strides["classic"][k].append(S)
            speed["classic"][k].append(y[TI["speed_ll"]])
        for r, (m, cfg, norm) in models.items():
            probs, ys, angs = run_model(m, norm, X, V, cfg.get("features", "v1"))
            for j, (k, i0, _) in enumerate(items):
                heads[r][k].append(ys[j])
                if angs is not None:
                    S = strides_from_events(decode_events(probs[j]), angs[j])
                    S[:, :2] += i0 / FS
                    strides[f"{r}+derived"][k].append(S)
                    speed[f"{r}+derived"][k].append(ys[j][TI["speed_ll"]])
        for k, tr in enumerate(trials):
            T = len(tr.pose)
            true = aggregate(tr.strides, 0, (T - 1) / FS, tr.speed_ll)
            base = dict(dataset=tr.ds, trial=tr.trial, subject=tr.subject, mode=tr.mode, level=lev,
                        leg_length=tr.leg_length, n_windows=len(tile_windows(tr, W)),
                        **{f"true_{n}": true[i] for i, n in enumerate(TARGET_NAMES)})
            preds = {"mean": mean_y}
            for meth, st in strides.items():
                if meth.endswith("+derived") and not st[k]:
                    continue
                sp = np.nanmedian(speed[meth][k]) if np.isfinite(speed[meth][k]).any() else np.nan
                p = pooled(st[k], T, sp)
                if meth.endswith("+derived"):        # pooled-head fallback where no stride was decoded
                    p = np.where(np.isfinite(p), p, np.mean(heads[meth[:-8]][k], 0))
                preds[meth] = p
            for r in models:
                preds[r] = np.mean(heads[r][k], 0)
            for meth, p in preds.items():
                rows.append(base | dict(method=meth) | {f"pred_{n}": p[i] for i, n in enumerate(TARGET_NAMES)})
        print(f"level {lev}: done", flush=True)

    out = EVAL / f"{a.split}_trial"
    out.mkdir(parents=True, exist_ok=True)
    P = pd.DataFrame(rows)
    for c in ("true", "pred"):
        P[f"{c}_speed_ms"] = P[f"{c}_speed_ll"] * P.leg_length
    P.to_parquet(out / "predictions.parquet", index=False)
    metrics(P, ["dataset", "level", "method"]).to_csv(out / "regression.csv", index=False)

    sb = []
    for n in TARGETS["symmetry"]:
        q = P[np.isfinite(P[f"true_{n}"]) & np.isfinite(P[f"pred_{n}"])]
        q = q.assign(abs_bin=pd.cut(q[f"true_{n}"].abs(), [0, 3, 7, np.inf], right=False, labels=["<3", "3-7", ">=7"]),
                     err=(q[f"pred_{n}"] - q[f"true_{n}"]).abs(),
                     under=np.abs(q[f"pred_{n}"]) - np.abs(q[f"true_{n}"]))
        sb.append(q.groupby(["dataset", "level", "method", "abs_bin"], observed=True).agg(
            n=("err", "size"), mae=("err", "mean"), mean_abs_shrink=("under", "mean")).reset_index().assign(target=n))
    pd.concat(sb).to_csv(out / "symmetry_bins.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
