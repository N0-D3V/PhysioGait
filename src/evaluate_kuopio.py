"""Test on real video: Kuopio OpenPose keypoints vs Vicon mocap and force plates.

    ./py src/evaluate_kuopio.py [--clips data/processed/kuopio_clips.pkl] [--out outputs/kuopio] [--half all|test]

By default only the held-out subjects are scored: the pose errors of subjects 01-26 were used for training
(helpers/kuopio_noise.py). --half all scores every subject (fair only for models trained without them, e.g. v4).

Each ~1 s clip is run through the same steps as the demo, for the model and the classic method, with four inputs
on the same frames: real video keypoints, clean mocap, and mocap with simulated noise (σ 0.02 and 0.04).
"""
import argparse
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import augment
import classic
from helpers.analysis import ROOT
from helpers.canonical import J, normalise, resample, sagittal_angles
from evaluate import match, regression_metrics
from helpers.gaitdata import ANG_NAMES, FS, IN_IDX, IN_JOINTS, TI, zeni_offsets
from helpers.kuopio_noise import is_fit
from video_pipeline import Analyzer, facing, normalise_pass, prepare

CLIPS = ROOT / "data" / "processed" / "kuopio_clips.pkl"
OUT = ROOT / "outputs" / "kuopio"
CONF = 0.3       # keypoint confidence threshold (as the demo)
TOL = 0.15       # event matching tolerance (s)
EDGE = 0.08      # gold events closer than this to a clip edge are not scored
GROSS = 0.2      # keypoint error (leg lengths, from the clip's median offset) counted as gross


# ---------------------------------------------------------------- inputs
def video_input(c: dict):
    """Real keypoints -> model input, as video_pipeline.from_keypoints does for one pass."""
    x = np.nan_to_num(c["px"][:, IN_IDX] * [1, -1])                   # y up
    x, _, vis, swapped = prepare(x, c["conf"][:, IN_IDX].copy(), c["fps"], CONF)
    direction = 1 if facing(x, vis).sum() >= 0 else -1
    x, vis, _ = normalise_pass(x, vis, direction)
    return x, vis, dict(swap_frac=swapped.mean(), direction_ok=direction == -c["direction"], vis_frac=vis.mean())


def mocap_input(sag: np.ndarray):
    x, _ = normalise(sag)
    return x[:, IN_IDX].astype(np.float32), np.ones((len(sag), len(IN_IDX)), np.float32)


# ---------------------------------------------------------------- per-clip measures
def event_rows(gold, pred, tmax, base):
    rows = []
    for k, g in gold.items():
        p = np.asarray(pred.get(k, []))
        err, fp, fn = match(g, p[(p >= EDGE - TOL) & (p <= tmax - EDGE + TOL)], TOL)
        rows.append(dict(base, event=k[:2], side=k[-1], n_true=len(g), n_hit=len(err), fp=fp, fn=fn,
                         err_ms=list(np.array(err) * 1000)))
    return rows


def timing_rows(gold, pred, base):
    """Stance (HS -> TO, same foot), swing (TO -> HS, same foot), step (HS -> HS, other foot) in the clip."""
    def nearest(name, t):
        p = np.asarray(pred[name])
        return p[np.argmin(np.abs(p - t))] if len(p) and np.abs(p - t).min() <= TOL else np.nan

    rows = []
    for s, o in (("L", "R"), ("R", "L")):
        for measure, start, end, side in (("stance_time", f"HS_{s}", f"TO_{s}", s),
                                          ("swing_time", f"TO_{s}", f"HS_{s}", s),
                                          ("step_time", f"HS_{s}", f"HS_{o}", o)):
            for t0 in gold[start]:
                later = gold[end][gold[end] > t0]
                if len(later):
                    rows.append(dict(base, measure=measure, side=side, true=later[0] - t0,
                                     pred=nearest(end, later[0]) - nearest(start, t0)))
    return rows


def angle_rows(true_ang, pred_ang, base, near):
    rows = []
    for i, n in enumerate(ANG_NAMES):
        t, p = true_ang[n], pred_ang[:, i]
        ok = np.isfinite(t) & np.isfinite(p)
        if ok.sum() >= 10:
            d = p[ok] - t[ok]
            rows.append(dict(base, joint=n[2:], side=n[0], leg="near" if n[0] == near else "far", offset=d.mean(),
                             shape_rmse=d.std(), true_exc=np.ptp(t[ok]), pred_exc=np.ptp(p[ok])))
    return rows


def keypoint_rows(xv, vv, xt, base, near, leg_m):
    """Real keypoint error split into a constant offset per clip, jitter, and gross errors."""
    rows = []
    for j, name in enumerate(IN_JOINTS):
        ok = vv[:, j] > 0
        row = dict(base, joint=name[2:], leg="near" if name[0] == near else "far", detected=ok.mean())
        if ok.sum() >= 5:
            e = xv[ok, j] - xt[ok, j]
            gross = np.linalg.norm(e - np.median(e, 0), axis=-1) > GROSS
            offset = e.mean(0)
            r = e - offset
            clean = e[~gross] if (~gross).sum() > 3 else e
            mad = np.median(np.abs(clean - np.median(clean, 0)))
            row |= dict(rms=np.sqrt((e ** 2).mean()), bias=np.sqrt((offset ** 2).mean()), jitter=r.std(),
                        jitter_robust=1.4826 * mad, gross_frac=gross.mean(),
                        lag1_autocorr=np.mean([np.corrcoef(r[:-1, k], r[1:, k])[0, 1] for k in range(2)])
                        if ok.sum() > 6 else np.nan,
                        rms_cm=np.sqrt((e ** 2).mean()) * leg_m * 100)
        rows.append(row)
    return rows


# ---------------------------------------------------------------- run
def run(clips, seed=0):
    import torch
    torch.set_num_threads(1)
    an, offsets, rng = Analyzer(), zeni_offsets(), np.random.default_rng(seed)
    tables = {k: [] for k in ("events", "timing", "angles", "keypoint_error", "speed", "clips")}
    for c in clips:
        sag = resample(c["truth_sag"], c["fps"])                       # truth at the 30 Hz frame times
        if not np.isfinite(sag).all():
            continue
        xv, vv, info = video_input(c)
        T = min(len(sag), len(xv))
        sag, xv, vv = sag[:T], xv[:T], vv[:T]
        xt, vt = mocap_input(sag)
        tmax = (T - 1) / FS
        gold = {k: np.array([e for e in v if EDGE <= e <= tmax - EDGE]) for k, v in c["events"].items()}
        true_ang = sagittal_angles(sag)
        leg = float(np.median([np.linalg.norm(sag[:, J[f"{s}_hip"]] - sag[:, J[f"{s}_ankle"]], axis=-1)
                               for s in "LR"]))                         # 2D leg length (m), as the input scale
        far = "L" if c["near"] == "R" else "R"
        inputs = {"real": (xv, vv), "mocap": (xt, vt)} | {
            f"sim{lv}": augment.apply(xt, rng, augment.Noise.level(lv), far=far) for lv in (0.02, 0.04)}

        base0 = dict(subject=c["subject"], trial=c["trial"], near=c["near"])
        hs = sorted(e for k, v in c["events"].items() if k.startswith("HS") for e in v if e >= 0)
        tables["clips"].append(base0 | info | dict(sync_err_px=c["sync_err_px"], T=T,
                                                   first_plate_hs_s=hs[0] if hs else np.nan))
        tables["keypoint_error"] += keypoint_rows(xv, vv, xt, base0, c["near"], leg)
        for cond, (x, v) in inputs.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, model_ev, model_ang, model_speed, _ = an._run_pass(x, v, 0)
                classic_ev, classic_y = classic.predict(x, v, offsets)
                ca = sagittal_angles(classic.to_sag(x, v))
            for method, ev, ang, speed in (("model", model_ev, model_ang, model_speed),
                                           ("classic", classic_ev, np.stack([ca[n] for n in ANG_NAMES], 1),
                                            classic_y[TI["speed_ll"]])):
                base = base0 | dict(condition=cond, method=method)
                tables["events"] += event_rows(gold, ev, tmax, base)
                tables["timing"] += timing_rows(gold, ev, base)
                tables["angles"] += angle_rows(true_ang, ang, base, c["near"])
                tables["speed"].append(base | dict(true_ms=c["speed_ms"], pred_ms=speed * leg))
    return {k: pd.DataFrame(v) for k, v in tables.items()}


# ---------------------------------------------------------------- summaries
def summarise(t: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    def metrics(df, by, true, pred, **extra):
        return pd.DataFrame([dict(zip(by, key)) | extra | regression_metrics(d[true].to_numpy(), d[pred].to_numpy())
                             for key, d in df.groupby(by)])

    ev = []
    for key, d in t["events"].groupby(["condition", "method", "event"]):
        e = np.concatenate([np.array(x, float) for x in d.err_ms])
        ev.append(dict(zip(["condition", "method", "event"], key), n_true=d.n_true.sum(),
                       recall=d.n_hit.sum() / max(d.n_true.sum(), 1), extra_detections=d.fp.sum(),
                       mae_ms=np.abs(e).mean(), median_ae_ms=np.median(np.abs(e)), bias_ms=e.mean(), sd_ms=e.std()))

    A = t["angles"].assign(exc_err=lambda d: d.pred_exc - d.true_exc)
    angles = A.groupby(["condition", "method", "joint", "leg"]).agg(
        shape_rmse=("shape_rmse", "mean"), offset=("offset", "mean"), exc_bias=("exc_err", "mean"),
        exc_mae=("exc_err", lambda e: e.abs().mean())).reset_index()

    # ROM symmetry per subject: mean excursion per side over all clips, or over near-leg clips only
    by = ["condition", "method", "subject", "joint"]
    side_mean = lambda d: d.groupby(by + ["side"])[["true_exc", "pred_exc"]].mean().unstack("side")  # noqa: E731
    si, si_summary = [], []
    for protocol, d in (("all", A), ("near", A[A.leg == "near"])):
        m = side_mean(d)
        for col in ("true_exc", "pred_exc"):
            m[(col[:4] + "_si", "")] = 200 * (m[(col, "L")] - m[(col, "R")]) / (m[(col, "L")] + m[(col, "R")])
        m = m[["true_si", "pred_si"]].droplevel(1, axis=1).dropna().reset_index().assign(protocol=protocol)
        si.append(m)
        si_summary.append(metrics(m, by[:2] + ["joint"], "true_si", "pred_si", protocol=protocol))

    kp_cols = ["detected", "rms", "bias", "jitter", "jitter_robust", "gross_frac", "lag1_autocorr", "rms_cm"]
    return {"keypoint_summary": t["keypoint_error"].groupby(["joint", "leg"])[kp_cols].mean().reset_index(),
            "events_summary": pd.DataFrame(ev),
            "timing_summary": metrics(t["timing"], ["condition", "method", "measure"], "true", "pred"),
            "angles_summary": angles,
            "rom_si": pd.concat(si, ignore_index=True),
            "rom_si_summary": pd.concat(si_summary, ignore_index=True),
            "speed_summary": metrics(t["speed"], ["condition", "method"], "true_ms", "pred_ms")}


def markdown(df: pd.DataFrame) -> str:
    cell = lambda v: f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)  # noqa: E731
    return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * df.shape[1]]
                     + ["| " + " | ".join(map(cell, r)) + " |" for r in df.itertuples(index=False)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=str(CLIPS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--half", choices=["all", "test"], default="test")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = [c for c in pickle.load(open(a.clips, "rb")) if a.half == "all" or not is_fit(c["subject"])]
    subjects = sorted({c["subject"] for c in clips})
    chunks = [[c for c in clips if c["subject"] in subjects[i::a.workers]] for i in range(a.workers)]
    with ProcessPoolExecutor(a.workers) as ex:
        parts = list(ex.map(run, chunks, range(len(chunks))))
    tables = {k: pd.concat([p[k] for p in parts], ignore_index=True) for k in parts[0]}
    tables["events"].to_parquet(out / "events.parquet")
    summaries = summarise(tables)
    for name, df in {**tables, **summaries}.items():
        if name != "events":
            df.to_csv(out / f"{name}.csv", index=False)

    m = tables["clips"]
    hs = m.first_plate_hs_s
    text = [f"# Kuopio external validation\n\n{len(m)} clips, {m.subject.nunique()} subjects. Median sync error "
            f"{m.sync_err_px.median():.1f} px; L/R swaps repaired in {100 * m.swap_frac.mean():.1f} % of frames; "
            f"facing direction correct in {100 * m.direction_ok.mean():.1f} % of clips. Sync check (force plates "
            f"not used by the sync): first plate heel strike {hs.median():.3f} s after the clip start "
            f"(IQR {hs.quantile(.25):.3f}-{hs.quantile(.75):.3f})."]
    text += [f"## {name.replace('_', ' ')}\n\n{markdown(df)}" for name, df in summaries.items() if name != "rom_si"]
    (out / "summary.md").write_text("\n\n".join(text) + "\n")
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
