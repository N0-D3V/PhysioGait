"""Fukuchi 2018 WBDS -> canonical skeleton.

Markers (Leardini-style, ASCII export): mm, Y-up, 150 Hz. Treadmill trials T01..T08
(30 s, fixed belt along lab X) and overground trials O01..O23 x {S,C,F} (~2.5 s).

Joint centres
  pelvis  = mean(ASIS L/R, PSIS L/R)
  hip     = Harrington 2007 regression from pelvis width / depth
  knee    = lateral femoral epicondyle (R.Knee) — medial marker only exists in static;
            its medio-lateral offset is discarded by the sagittal projection anyway.
  ankle   = lateral malleolus (same argument)
  heel    = Heel; toe = mean(MT1, MT5)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from canonical import JOINTS, fill_gaps

ROOT = Path(__file__).resolve().parents[1] / "data" / "fukuchi_wbds"
ASCII = ROOT / "51subjs"
FS = 150.0
FS_GRF = 300.0
UP = np.array([0.0, 1.0, 0.0])


def _m(df, name):
    return df[[f"{name}X", f"{name}Y", f"{name}Z"]].to_numpy(float) / 1000.0


def harrington_hjc(rasis, lasis, rpsis, lpsis):
    """Harrington et al. 2007 hip joint centres [T,3] (metres) for both sides."""
    o = 0.5 * (rasis + lasis)
    z = rasis - lasis
    pw = np.linalg.norm(z, axis=1, keepdims=True)
    z = z / pw                                    # right
    mid_psis = 0.5 * (rpsis + lpsis)
    xa = o - mid_psis
    pd_ = np.linalg.norm(xa, axis=1, keepdims=True)
    y = np.cross(z, xa)
    y /= np.linalg.norm(y, axis=1, keepdims=True)  # superior
    x = np.cross(y, z)                              # anterior
    pw_mm, pd_mm = pw * 1000, pd_ * 1000
    hx = (-0.24 * pd_mm - 9.9) / 1000
    hy = (-0.30 * pw_mm - 10.9) / 1000
    hz = (0.33 * pw_mm + 7.3) / 1000
    r = o + hx * x + hy * y + hz * z
    l = o + hx * x + hy * y - hz * z
    return l, r


def to_canonical(df: pd.DataFrame) -> np.ndarray:
    ra, la, rp, lp = (_m(df, n) for n in ("R.ASIS", "L.ASIS", "R.PSIS", "L.PSIS"))
    lh, rh = harrington_hjc(ra, la, rp, lp)
    pts = {
        "pelvis": (ra + la + rp + lp) / 4,
        "L_hip": lh, "R_hip": rh,
    }
    for s in "LR":
        pts[f"{s}_knee"] = _m(df, f"{s}.Knee")
        pts[f"{s}_ankle"] = _m(df, f"{s}.Ankle")
        pts[f"{s}_heel"] = _m(df, f"{s}.Heel")
        pts[f"{s}_toe"] = 0.5 * (_m(df, f"{s}.MT1") + _m(df, f"{s}.MT5"))
    return np.stack([pts[j] for j in JOINTS], axis=1)


def grf_events(grf_path: Path, heel_z: dict[str, float], thresh: float = 30.0) -> dict[str, np.ndarray]:
    """Gold-standard HS/TO (marker-frame indices) from the instrumented treadmill belts.

    Vertical force is Fy. Belts are assigned to feet by mean COP-z vs mean heel-z.
    """
    g = pd.read_csv(grf_path, sep="\t")
    out = {}
    for b in (1, 2):
        fy = g[f"Fy{b}"].to_numpy(float)
        on = fy > thresh
        d = np.diff(np.r_[0, on.astype(int), 0])
        segs = list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))
        # merge flicker (< 50 ms off) then keep real stance phases (> 200 ms)
        merged = []
        for s, e in segs:
            if merged and s - merged[-1][1] < 0.05 * FS_GRF:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        merged = [(s, e) for s, e in merged if e - s > 0.2 * FS_GRF and s > 0 and e < len(fy)]
        copz = g.loc[on, f"COPz{b}"].median() / 1000
        side = min(heel_z, key=lambda s: abs(heel_z[s] - copz))
        out[f"HS_{side}"] = np.round(np.array([s for s, _ in merged]) * FS / FS_GRF).astype(int)
        out[f"TO_{side}"] = np.round(np.array([e for _, e in merged]) * FS / FS_GRF).astype(int)
    return out


def info() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "WBDSinfo.csv", na_values=["--"])
    d = d[d.FileName.str.endswith("mkr.txt")].copy()
    m = d.FileName.str.extract(r"WBDS(\d+)walk(T|O)(\d+)([SCF]?)mkr")
    d["subject"] = "fukuchi_" + m[0]
    d["mode"] = np.where(m[1] == "T", "treadmill", "overground")
    d["trial"] = d.FileName.str.replace("mkr.txt", "", regex=False)
    d["speed_label"] = np.where(m[1] == "T", "T" + m[2], m[3].map({"S": "slow", "C": "comfortable", "F": "fast"}))
    return d.rename(columns={"GaitSpeed(m/s)": "speed", "Age": "age", "Height": "height_cm",
                             "Mass": "mass_kg", "Gender": "sex", "LegLength": "leg_length_m",
                             "AgeGroup": "age_group"})


def load_trial(row) -> dict:
    df = pd.read_csv(ASCII / row.FileName, sep="\t")
    xyz = to_canonical(df)
    n_missing = int(np.isnan(xyz).any(-1).any(-1).sum())
    xyz = fill_gaps(xyz, max_gap=int(0.1 * FS))
    rec = dict(xyz=xyz.astype(np.float32), fs=FS, up=UP, n_missing_frames_raw=n_missing)
    if row["mode"] == "treadmill":
        grf = ASCII / row.FileName.replace("mkr", "grf")
        heel_z = {s: float(np.nanmean(xyz[:, JOINTS.index(f"{s}_heel"), 2])) for s in "LR"}
        gold = grf_events(grf, heel_z) if grf.exists() else None
        # some sessions (subjects 43-51) have a dead belt -> no usable gold events for one foot
        ok = gold and all(len(gold.get(f"{e}_{s}", [])) >= 5 for e in ("HS", "TO") for s in "LR")
        rec["gold_events"] = gold if ok else None
    return rec
