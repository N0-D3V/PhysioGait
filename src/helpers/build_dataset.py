"""Build the processed dataset: each trial -> our skeleton, side view, 30 Hz, events, labels, quality checks.

    ./py src/helpers/build_dataset.py [fukuchi ferber toronto] [--limit N] [--workers K]

Writes data/processed/<dataset>/{trials.parquet, strides.parquet, npz/<trial>.npz}.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from helpers.canonical import (J, TARGET_FS, detect_events, lowpass, normalise, progression_axis_pca, resample,
                       sagittal_angles, sagittal_project)
from helpers.gaitdata import ANG_NAMES, SIDES, STRIDE_COLS, TARGET_NAMES, TI, aggregate, strides_from_events

OUT = Path(__file__).resolve().parents[2] / "data" / "processed"

# ---- QC thresholds (filtering) --------------------------------------------------------
QC = dict(
    max_missing_frac=0.01,     # frames still NaN after short-gap filling
    leg_len=(0.60, 1.10),      # m, hip->ankle in sagittal plane
    max_cluster_rms_mm=5.0,    # Ferber rigid-fit residual
    walk_speed=(0.3, 2.2),     # m/s; above that the "walking" trial is likely a jog
    min_strides=2,             # valid strides per side (treadmill / Toronto passes)
    max_stride_cv=20.0,        # % ; larger = event detection failure / non-steady gait
    min_clean_s=10.0,          # gappy long trials: keep longest clean stretch if >= this
)


def treadmill_forward(xyz, up):
    """Mean horizontal heel->toe direction snapped to the nearest lab axis (belt direction)."""
    v = np.nanmean(xyz[:, [J["L_toe"], J["R_toe"]]] - xyz[:, [J["L_heel"], J["R_heel"]]], axis=(0, 1))
    v = v - (v @ up) * up
    ax = np.zeros(3)
    k = int(np.argmax(np.abs(v)))
    ax[k] = np.sign(v[k])
    return ax


def process(rec: dict, mode: str) -> tuple[dict, list[dict], dict]:
    """Shared pipeline for one trial. Returns (row, strides, arrays)."""
    xyz, fs, up = rec["xyz"].astype(np.float64), rec["fs"], rec["up"]
    row = {"fs": fs, "n_frames": len(xyz), "duration_s": len(xyz) / fs,
           "missing_frac": float(np.isnan(xyz).any(-1).any(-1).mean())}
    good = ~np.isnan(xyz).any(-1).any(-1)
    crop = (0, len(xyz))
    if row["missing_frac"] > QC["max_missing_frac"]:
        # long trials: salvage the longest gap-free stretch if it is long enough
        d = np.diff(np.r_[0, good.astype(int), 0])
        runs = list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))
        s, e = max(runs, key=lambda r: r[1] - r[0]) if runs else (0, 0)
        if mode == "overground" or (e - s) < QC["min_clean_s"] * fs:
            return row | {"reject_reason": "missing_markers"}, [], {}
        crop = (s, e)
    else:
        crop = (good.argmax(), len(good) - good[::-1].argmax())       # trim NaN ends
    xyz = xyz[crop[0]:crop[1]]
    row |= {"crop_start": int(crop[0]), "duration_used_s": len(xyz) / fs}
    gold = rec.get("gold_events")
    if gold and crop[0] > 0 or gold and crop[1] < row["n_frames"]:
        gold = {k: v[(v >= crop[0]) & (v < crop[1])] - crop[0] for k, v in gold.items()}
    xyz = np.nan_to_num(xyz)  # residual isolated NaNs (<1%) — negligible after filtering
    fwd = treadmill_forward(xyz, up) if mode == "treadmill" else progression_axis_pca(xyz[:, J["pelvis"]], up)
    xyz_f = lowpass(xyz.reshape(len(xyz), -1), fs, fc=6.0).reshape(xyz.shape)
    sag = sagittal_project(xyz_f, fwd, up)
    ang = sagittal_angles(sag)
    sag_n, L = normalise(sag)
    row["leg_length_sag"] = L
    pel = sag[:, J["pelvis"], 0]
    row["speed_pelvis"] = float(abs(np.median(np.gradient(pel) * fs)))
    ev = detect_events(sag, fs)
    use = gold if gold else ev
    row["event_source"] = "grf" if gold else "zeni"
    S = strides_from_events({k: np.asarray(v) / fs for k, v in use.items()},
                            np.stack([ang[n] for n in ANG_NAMES], -1), fs)
    y = aggregate(S, -np.inf, np.inf)                       # per-trial labels, same formulas as the model's
    row |= {"n_strides": len(S)} | {n: y[TI[n]] for n in TARGET_NAMES if n != "speed_ll"}
    strides = [dict(side=SIDES[int(r[2])], start=int(round(r[0] * fs)), end=int(round(r[1] * fs)),
                    **dict(zip(STRIDE_COLS, r[3:]))) for r in S]
    if gold:  # coordinate-based vs force-plate timing error (ms)
        for k in ev:
            g, e = gold[k], ev[k]
            if len(g) and len(e):
                d = np.array([e[np.argmin(abs(e - v))] - v for v in g])
                d = d[np.abs(d) < 0.15 * fs]
                row[f"zeni_err_ms_{k}"] = float(np.mean(d) / fs * 1000) if len(d) else np.nan
    arrays = dict(
        xyz=xyz.astype(np.float32), fwd=fwd, up=up, fs=fs, leg_length=L,
        sag30=resample(sag_n, fs).astype(np.float32),
        ang30=resample(np.stack([ang[n] for n in ANG_NAMES], -1), fs).astype(np.float32),
        pelvis_fwd30=resample(pel - pel[0], fs).astype(np.float32),
        **{f"ev_{k}": v for k, v in ev.items()},
        **({f"gold_{k}": v for k, v in gold.items()} if gold else {}),
        **{f"ev30_{k}": np.round(v * TARGET_FS / fs).astype(int) for k, v in use.items()},
    )
    return row, strides, arrays


def qc_reason(row: dict, mode: str) -> str | None:
    if row.get("reject_reason"):
        return row["reject_reason"]
    lo, hi = QC["leg_len"]
    if not lo <= row["leg_length_sag"] <= hi:
        return "leg_length"
    if row.get("max_cluster_rms_mm", 0) > QC["max_cluster_rms_mm"]:
        return "cluster_fit"
    sp = row.get("speed")
    if sp is not None and np.isfinite(sp) and not QC["walk_speed"][0] <= sp <= QC["walk_speed"][1]:
        return "speed_out_of_range"
    min_str = 1 if mode == "overground" else QC["min_strides"] * 2
    if row.get("n_strides", 0) < min_str:
        return "too_few_strides"
    if mode != "overground" and row.get("stride_time_cv", 0) > QC["max_stride_cv"]:
        return "irregular_events"
    return None


# ---------------------------------------------------------------- dataset drivers
def _run_one(args):
    ds, meta = args
    try:
        if ds == "fukuchi":
            from helpers import load_fukuchi as L
            rec = L.load_trial(pd.Series(meta))
        elif ds == "ferber":
            from helpers import load_ferber as L
            rec = L.load_trial(pd.Series(meta))
            meta["max_cluster_rms_mm"] = max(v for k, v in rec.items() if k.startswith("rms_"))
            meta["toe_source"] = rec["toe_source"]
        row, strides, arrays = process(rec, meta["mode"])
        row = meta | row | {"n_missing_frames_raw": rec.get("n_missing_frames_raw")}
    except ValueError as e:
        if type(e).__name__ == "NoFootLandmarks":
            return meta | {"reject_reason": "no_foot_landmarks"}, [], None
        return meta | {"reject_reason": f"load_error: {type(e).__name__}: {e}"[:200]}, [], None
    except Exception as e:  # noqa: BLE001 — record failure, keep going
        return meta | {"reject_reason": f"load_error: {type(e).__name__}: {e}"[:200]}, [], None
    return row, strides, arrays


def _finish(ds, rows, out):
    df = pd.DataFrame(rows)
    df["reject_reason"] = [qc_reason(r, r["mode"]) for r in rows]
    df["keep"] = df.reject_reason.isna()
    df = df.drop(columns=[c for c in ("path",) if c in df])
    df.to_parquet(out / "trials.parquet", index=False)
    print(f"[{ds}] {len(df)} trials, kept {df.keep.sum()} "
          f"({df[df.keep].subject.nunique()} subjects); rejects: {df.reject_reason.value_counts().to_dict()}")
    return df


def build(ds: str, limit: int | None, workers: int):
    out = OUT / ds
    (out / "npz").mkdir(parents=True, exist_ok=True)
    if ds == "toronto":
        return build_toronto(out)
    if ds == "fukuchi":
        from helpers import load_fukuchi as L
        meta = L.info()
        cols = ["subject", "trial", "FileName", "mode", "speed", "speed_label", "age", "age_group",
                "sex", "height_cm", "mass_kg", "leg_length_m"]
    else:
        from helpers import load_ferber as L
        meta = L.info()
        meta["mode"] = "treadmill"
        meta = meta[meta.path.map(Path.exists)]
        cols = ["subject", "trial", "path", "mode", "speed", "age", "sex", "height_cm", "mass_kg",
                "DominantLeg", "injured", "InjDefn", "InjJoint", "InjSide", "InjSide_clean",
                "SpecInjury_clean", "InjDuration", "Level", "datestring"]
    meta = meta[cols]
    if limit:
        meta = meta.groupby("mode", group_keys=False).head(limit)
    jobs = [(ds, m) for m in meta.to_dict("records")]
    rows, all_strides = [], []
    with ProcessPoolExecutor(workers) as ex:
        for row, strides, arrays in tqdm(ex.map(_run_one, jobs, chunksize=4), total=len(jobs), desc=ds):
            rows.append(row)
            if arrays:
                np.savez_compressed(out / "npz" / f"{row['trial']}.npz", **arrays)
                all_strides += [s | {"trial": row["trial"], "subject": row["subject"]} for s in strides]
    df = _finish(ds, rows, out)
    st = pd.DataFrame(all_strides)
    st = st[st.trial.isin(df.trial[df.keep])]
    st.to_parquet(out / "strides.parquet", index=False)
    return df


def build_toronto(out: Path):
    from helpers import load_toronto as L
    info = L.info().set_index("pid")
    rows, all_strides = [], []
    for pid in info.index:
        xyz = L.load_xsens(pid)
        base = dict(subject=info.loc[pid, "subject"], mode="overground_pass",
                    **info.loc[pid, ["age", "sex", "height_cm", "poma_gait", "poma_balance",
                                     "bbs", "tug_s", "falls_6mo"]].to_dict())
        if xyz is None:
            rows.append(base | {"trial": f"OAW{pid:02d}", "reject_reason": "no_xsens_file"})
            continue
        for i, (s, e) in enumerate(L.split_passes(xyz)):
            rec = dict(xyz=xyz[s:e], fs=L.FS, up=L.UP)
            row, strides, arrays = process(rec, "overground")
            row = base | {"trial": f"OAW{pid:02d}_pass{i}", "pass_start_s": s / L.FS} | row
            row["speed"] = row["speed_pelvis"]
            rows.append(row)
            np.savez_compressed(out / "npz" / f"{row['trial']}.npz", **arrays)
            all_strides += [x | {"trial": row["trial"], "subject": row["subject"]} for x in strides]
    df = _finish("toronto", rows, out)
    st = pd.DataFrame(all_strides)
    st[st.trial.isin(df.trial[df.keep])].to_parquet(out / "strides.parquet", index=False)
    clips = L.pose_clips()
    q = []
    for r in clips.itertuples():
        p = L.load_pose(r.path)
        q.append(dict(r._asdict(), n_frames=len(p["t"]), duration_s=float(p["t"][-1] - p["t"][0]) if len(p["t"]) else 0,
                      mean_conf=float(np.nanmean(p["conf"])),
                      frac_lowconf=float((p["conf"] < 0.3).mean())))
    pd.DataFrame(q).drop(columns=["Index", "path"]).to_parquet(out / "pose_clips.parquet", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=["fukuchi", "toronto", "ferber"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    for d in a.datasets:
        build(d, a.limit, a.workers)
