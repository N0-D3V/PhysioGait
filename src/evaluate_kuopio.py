"""External validation on real video keypoints: Kuopio (OpenPose BODY_25 on sagittal video vs Vicon + force plates).

    ./py src/evaluate_kuopio.py [--clips data/processed/kuopio_clips.pkl] [--out outputs/kuopio]

Every clip (~1.05 s around the middle-force-plate stance) is run through the deployed pipeline steps
(`video_pipeline`: 30 Hz resampling, L/R swap repair, facing direction, mid-hip origin, leg-length
scale, BiGRU v4) and through the classic pipeline, under four inputs on identical frames:
  real       OpenPose keypoints from the video (the actual test)
  mocap      the synchronised mocap joints projected to the camera plane (clean input)
  sim 0.02   mocap + the training noise model at the "realistic video" level
  sim 0.04   mocap + twice that
Truth: force-plate events (gold), mocap joint angles and excursions, mocap pelvis speed.

Writes outputs/kuopio/{keypoint_error.csv, events.csv, timing.csv, angles.csv, rom_subject.csv,
rom_si.csv, speed.csv, summary.md}. Kuopio is never used for training or model selection.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import augment  # noqa: E402
import classic  # noqa: E402
from analysis import ROOT  # noqa: E402
from canonical import J, JOINTS, normalise, sagittal_angles  # noqa: E402
from evaluate import match  # noqa: E402
from gaitdata import ANG_NAMES, FS, IN_IDX, IN_JOINTS, TI, zeni_offsets  # noqa: E402
from video_pipeline import Analyzer, fix_left_right, normalise_pass, resample_30  # noqa: E402

CLIPS = ROOT / "data" / "processed" / "kuopio_clips.pkl"
OUT = ROOT / "outputs" / "kuopio"
CONF = 0.3          # keypoint confidence threshold (as the demo)
TOL = 0.15          # event matching tolerance (s)
EDGE = 0.08         # gold events closer than this to a clip edge are not scored
CONDITIONS = ["real", "mocap", "sim0.02", "sim0.04"]
JOINT_TYPES = ["hip", "knee", "ankle", "heel", "toe"]


# ---------------------------------------------------------------- inputs
def video_input(c: dict):
    """Real keypoints -> model input exactly as video_pipeline.from_keypoints does (one pass)."""
    x = c["px"][:, IN_IDX].copy()
    x[..., 1] *= -1                                         # y up
    s = c["conf"][:, IN_IDX].copy()
    x = np.nan_to_num(x)
    x, s = resample_30(x, s, c["fps"])
    vis = (s > CONF).astype(np.float32)
    swapped = fix_left_right(x, vis)
    x[swapped], vis[swapped] = x[swapped][:, augment.JOINT_SWAP], vis[swapped][:, augment.JOINT_SWAP]
    fx = sum(vis[:, IN_JOINTS.index(f"{sd}_heel")] * vis[:, IN_JOINTS.index(f"{sd}_toe")]
             * np.sign(x[:, IN_JOINTS.index(f"{sd}_toe"), 0] - x[:, IN_JOINTS.index(f"{sd}_heel"), 0]) for sd in "LR")
    d = 1 if fx.sum() >= 0 else -1
    xn, vis, _ = normalise_pass(x, vis, d)
    return xn, vis, dict(swap_frac=float(swapped.mean()), direction_ok=d == -c["direction"],
                         vis_frac=float(vis.mean()))


def truth_30(c: dict) -> np.ndarray:
    """Mocap sagittal joints [T30, 11, 2] (m) at the 30 Hz frame times used for the video input."""
    n = len(c["truth_sag"])
    t = np.arange(n) / c["fps"]
    tn = np.arange(0, t[-1] + 1e-9, 1 / FS)
    flat = c["truth_sag"].reshape(n, -1)
    return np.stack([np.interp(tn, t, flat[:, k]) for k in range(flat.shape[1])], 1).reshape(len(tn), 11, 2)


def mocap_input(sag: np.ndarray):
    xn, _ = normalise(sag)
    return xn[:, IN_IDX].astype(np.float32), np.ones((len(sag), len(IN_IDX)), np.float32)


# ---------------------------------------------------------------- per-clip measures
def gold_window(c, T30):
    tmax = (T30 - 1) / FS
    return {k: np.array([e for e in v if EDGE <= e <= tmax - EDGE]) for k, v in c["events"].items()}, tmax


def event_rows(gold, pred, tmax, base):
    rows = []
    for k, g in gold.items():
        p = np.asarray(pred.get(k, []))
        p = p[(p >= EDGE - TOL) & (p <= tmax - EDGE + TOL)]
        err, fp, fn = match(g, p, TOL)
        rows.append(dict(base, event=k[:2], side=k[-1], n_true=len(g), n_hit=len(err), fp=fp, fn=fn,
                         abs_err_ms=list(np.abs(err) * 1000), err_ms=list(np.array(err) * 1000)))
    return rows


def _nearest(arr, t):
    arr = np.asarray(arr)
    if not len(arr):
        return np.nan
    k = np.argmin(np.abs(arr - t))
    return float(arr[k]) if abs(arr[k] - t) <= TOL else np.nan


def timing_rows(gold, pred, base):
    """Stance (HS->TO same foot), swing (TO->HS same foot) and step (HS->HS other foot) inside the clip."""
    rows = []
    for s in "LR":
        o = "R" if s == "L" else "L"
        for h in gold[f"HS_{s}"]:
            to = gold[f"TO_{s}"][gold[f"TO_{s}"] > h]
            if len(to):
                t = to[0] - h
                p = _nearest(pred[f"TO_{s}"], to[0]) - _nearest(pred[f"HS_{s}"], h)
                rows.append(dict(base, measure="stance_time", side=s, true=t, pred=p))
            nh = gold[f"HS_{o}"][gold[f"HS_{o}"] > h]
            if len(nh):
                rows.append(dict(base, measure="step_time", side=o, true=nh[0] - h,
                                 pred=_nearest(pred[f"HS_{o}"], nh[0]) - _nearest(pred[f"HS_{s}"], h)))
        for t0 in gold[f"TO_{s}"]:
            hs = gold[f"HS_{s}"][gold[f"HS_{s}"] > t0]
            if len(hs):
                rows.append(dict(base, measure="swing_time", side=s, true=hs[0] - t0,
                                 pred=_nearest(pred[f"HS_{s}"], hs[0]) - _nearest(pred[f"TO_{s}"], t0)))
    return rows


def angle_rows(true_ang: dict, pred_ang: np.ndarray, base, near: str):
    rows = []
    for i, n in enumerate(ANG_NAMES):
        t, p = true_ang[n], pred_ang[:, i]
        ok = np.isfinite(t) & np.isfinite(p)
        if ok.sum() < 10:
            continue
        d = p[ok] - t[ok]
        rows.append(dict(base, angle=n, joint=n[2:], side=n[0], leg="near" if n[0] == near else "far",
                         offset=d.mean(), shape_rmse=d.std(), true_exc=np.ptp(t[ok]), pred_exc=np.ptp(p[ok])))
    return rows


def keypoint_rows(xv, vv, xt, base, near, leg_m):
    rows = []
    for j, name in enumerate(IN_JOINTS):
        ok = vv[:, j] > 0
        if ok.sum() < 5:
            rows.append(dict(base, joint=name[2:], leg="near" if name[0] == near else "far", detected=ok.mean()))
            continue
        e = xv[ok, j] - xt[ok, j]
        b = e.mean(0)
        r = e - b
        ac = np.mean([np.corrcoef(r[:-1, k], r[1:, k])[0, 1] for k in range(2)]) if ok.sum() > 6 else np.nan
        rows.append(dict(base, joint=name[2:], leg="near" if name[0] == near else "far", detected=ok.mean(),
                         rms=np.sqrt((e ** 2).mean()), bias=np.sqrt((b ** 2).mean()), jitter=r.std(),
                         bias_x=b[0], bias_y=b[1], lag1_autocorr=ac, rms_cm=np.sqrt((e ** 2).mean()) * leg_m * 100))
    return rows


# ---------------------------------------------------------------- run
def run(clips, an: Analyzer, offsets, seed=0):
    rng = np.random.default_rng(seed)
    ev_rows, tim_rows, ang_rows, kp_rows, sp_rows, meta = [], [], [], [], [], []
    for k, c in enumerate(clips):
        sag = truth_30(c)
        if not np.isfinite(sag).all():
            continue
        T30 = len(sag)
        xv, vv, info = video_input(c)
        T = min(T30, len(xv))
        sag, xv, vv = sag[:T], xv[:T], vv[:T]
        xt, vt = mocap_input(sag)
        true_ang = {n: v for n, v in sagittal_angles(sag).items()}
        gold, tmax = gold_window(c, T)
        leg2d = float(np.median(np.concatenate([np.linalg.norm(sag[:, J[f"{s}_hip"]] - sag[:, J[f"{s}_ankle"]], axis=-1) for s in "LR"])))
        true_speed_ll = c["speed_ms"] / leg2d
        far = "L" if c["near"] == "R" else "R"
        inputs = {"real": (xv, vv), "mocap": (xt, vt)}
        for lv in (0.02, 0.04):
            inputs[f"sim{lv}"] = augment.apply(xt, rng, augment.Noise.level(lv), far=far)
        meta.append(dict(subject=c["subject"], trial=c["trial"], near=c["near"], sync_err_px=c["sync_err_px"],
                         T=T, **info))
        base0 = dict(subject=c["subject"], trial=c["trial"], near=c["near"])
        keypoint_rows_ = keypoint_rows(xv, vv, xt, dict(base0, condition="real"), c["near"], leg2d)
        kp_rows += keypoint_rows_
        for cond, (x, v) in inputs.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                S, ev, ang, speed, _ = an._run_pass(x, v, 0)
                cev, cy = classic.predict(x, v, offsets)
                cang = sagittal_angles(classic.to_sag(x, v))
            for method, pev, pang, psp in (("model", ev, ang, speed),
                                           ("classic", cev, np.stack([cang[n] for n in ANG_NAMES], 1),
                                            float(cy[TI["speed_ll"]]))):
                base = dict(base0, condition=cond, method=method)
                ev_rows += event_rows(gold, pev, tmax, base)
                tim_rows += timing_rows(gold, {q: np.asarray(v_) for q, v_ in pev.items()}, base)
                ang_rows += angle_rows(true_ang, pang, base, c["near"])
                sp_rows.append(dict(base, true_ll=true_speed_ll, pred_ll=psp, true_ms=c["speed_ms"],
                                    pred_ms=psp * leg2d))
        if k % 100 == 0:
            print(f"{k}/{len(clips)}", flush=True)
    return (pd.DataFrame(ev_rows), pd.DataFrame(tim_rows), pd.DataFrame(ang_rows), pd.DataFrame(kp_rows),
            pd.DataFrame(sp_rows), pd.DataFrame(meta))


# ---------------------------------------------------------------- summaries
def event_summary(E):
    g = E.groupby(["condition", "method", "event"])
    rows = []
    for key, d in g:
        errs = np.concatenate([np.array(x, float) for x in d.err_ms]) if len(d) else np.zeros(0)
        n_true, n_hit = d.n_true.sum(), d.n_hit.sum()
        rows.append(dict(zip(["condition", "method", "event"], key), n_true=n_true,
                         recall=n_hit / max(n_true, 1), fp=d.fp.sum(),
                         mae_ms=np.abs(errs).mean() if len(errs) else np.nan,
                         median_ae_ms=np.median(np.abs(errs)) if len(errs) else np.nan,
                         bias_ms=errs.mean() if len(errs) else np.nan))
    return pd.DataFrame(rows)


def reg(t, p):
    ok = np.isfinite(t) & np.isfinite(p)
    t, p = t[ok], p[ok]
    if len(t) < 3:
        return dict(n=len(t))
    return dict(n=len(t), mae=np.abs(p - t).mean(), bias=(p - t).mean(),
                r=np.corrcoef(t, p)[0, 1] if p.std() > 0 and t.std() > 0 else np.nan,
                slope=np.polyfit(t, p, 1)[0] if t.std() > 0 else np.nan,
                mean_pred_mae=np.abs(t - t.mean()).mean())


def rom_subject(A):
    """Per subject, joint, side: mean excursion over clips (all clips, or near-leg clips only)."""
    rows = []
    for (cond, method, sub, joint, side), d in A.groupby(["condition", "method", "subject", "joint", "side"]):
        near = d[d.leg == "near"]
        rows.append(dict(condition=cond, method=method, subject=sub, joint=joint, side=side,
                         true=d.true_exc.mean(), pred_all=d.pred_exc.mean(),
                         pred_near=near.pred_exc.mean() if len(near) else np.nan, n=len(d)))
    return pd.DataFrame(rows)


def rom_si(R):
    rows = []
    for (cond, method, sub, joint), d in R.groupby(["condition", "method", "subject", "joint"]):
        d = d.set_index("side")
        if not {"L", "R"} <= set(d.index):
            continue
        si = lambda col: 100 * (d.loc["L", col] - d.loc["R", col]) / (0.5 * (d.loc["L", col] + d.loc["R", col]))  # noqa: E731
        rows.append(dict(condition=cond, method=method, subject=sub, joint=joint, true_si=si("true"),
                         pred_si_all=si("pred_all"), pred_si_near=si("pred_near")))
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, floatfmt=".2f") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:{floatfmt}}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=str(CLIPS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = pickle.load(open(a.clips, "rb"))
    if a.limit:
        clips = clips[:a.limit]
    an = Analyzer()
    E, Tm, A, KP, SP, META = run(clips, an, zeni_offsets())
    E.to_parquet(out / "events_raw.parquet")
    for name, df in (("timing", Tm), ("angles", A), ("keypoint_error", KP), ("speed", SP), ("clips", META)):
        df.to_csv(out / f"{name}.csv", index=False)
    ES = event_summary(E)
    ES.to_csv(out / "events.csv", index=False)
    TS = pd.DataFrame([dict(condition=c, method=m, measure=ms, **reg(d.true.to_numpy(), d.pred.to_numpy()))
                       for (c, m, ms), d in Tm.groupby(["condition", "method", "measure"])])
    TS.to_csv(out / "timing_summary.csv", index=False)
    AS = A.groupby(["condition", "method", "joint", "leg"]).apply(
        lambda d: pd.Series(dict(shape_rmse=d.shape_rmse.mean(), offset=d.offset.mean(),
                                 exc_mae=np.abs(d.pred_exc - d.true_exc).mean(),
                                 exc_bias=(d.pred_exc - d.true_exc).mean(),
                                 exc_r=np.corrcoef(d.true_exc, d.pred_exc)[0, 1]))).reset_index()
    AS.to_csv(out / "angles_summary.csv", index=False)
    R = rom_subject(A)
    R.to_csv(out / "rom_subject.csv", index=False)
    SI = rom_si(R)
    SI.to_csv(out / "rom_si.csv", index=False)
    SIS = pd.DataFrame([dict(condition=c, method=m, joint=j, protocol=p, **reg(d.true_si.to_numpy(), d[f"pred_si_{p}"].to_numpy()))
                        for (c, m, j), d in SI.groupby(["condition", "method", "joint"]) for p in ("all", "near")])
    SIS.to_csv(out / "rom_si_summary.csv", index=False)
    KS = KP.groupby(["joint", "leg"])[["detected", "rms", "bias", "jitter", "lag1_autocorr", "rms_cm"]].mean().reset_index()
    KS.to_csv(out / "keypoint_summary.csv", index=False)
    SPS = pd.DataFrame([dict(condition=c, method=m, **reg(d.true_ms.to_numpy(), d.pred_ms.to_numpy()))
                        for (c, m), d in SP.groupby(["condition", "method"])])
    SPS.to_csv(out / "speed_summary.csv", index=False)
    with open(out / "summary.md", "w") as f:
        f.write(f"# Kuopio external validation\n\n{len(META)} clips, {META.subject.nunique()} subjects. "
                f"Median sync error {META.sync_err_px.median():.1f} px; L/R swaps repaired in "
                f"{100 * META.swap_frac.mean():.1f} % of frames; facing direction correct in "
                f"{100 * META.direction_ok.mean():.1f} % of clips.\n\n")
        for title, df in (("Keypoint error (real OpenPose vs mocap, leg lengths)", KS), ("Events", ES),
                          ("Timing (per clip)", TS), ("Angles (per clip)", AS),
                          ("ROM symmetry (per subject)", SIS), ("Speed (m/s, per clip)", SPS)):
            f.write(f"## {title}\n\n{md_table(df)}\n\n")
    print(open(out / "summary.md").read())


if __name__ == "__main__":
    main()
