"""Ferber Running Injury Clinic (walking subset) -> canonical skeleton.

During walking only rigid 4-marker clusters (pelvis, thighs, shanks, feet) + a toe marker
were recorded; anatomical markers exist only in the neutral standing trial. We follow the
dataset's own gait_kinematics.m definitions for the static joint centres:
  hip   = trochanter marker + 25% towards the contralateral one
  knee  = mid(lat, med epicondyle);  ankle = mid(lat, med malleolus)
and carry each into the walking trial with a per-frame least-squares rigid fit (Kabsch)
of its segment cluster (hip->pelvis, knee->thigh, ankle->shank, heel->foot).

Heel has no marker: the foot cluster sits on the shoe heel counter, so heel = the most
posterior foot-cluster marker dropped to the metatarsal-marker height in the neutral pose.
Units mm, Y-up, treadmill.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from canonical import JOINTS

ROOT = Path(__file__).resolve().parents[1] / "data" / "Ferber Running Injury Dataset Full"
RAW = ROOT / "extracted" / "reformat_data"
UP = np.array([0.0, 1.0, 0.0])


def kabsch_track(ref: np.ndarray, dyn: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigidly map `pts` [P,3] (expressed with `ref` [M,3]) into each frame of `dyn` [T,M,3].

    Returns tracked points [T,P,3] and per-frame RMS fit residual [T].
    """
    rc = ref.mean(0)
    dc = dyn.mean(1, keepdims=True)
    H = np.einsum("mi,tmj->tij", ref - rc, dyn - dc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("tij,tjk->tik", U, Vt)))
    D = np.tile(np.eye(3), (len(dyn), 1, 1))
    D[:, 2, 2] = d
    R = np.einsum("tji,tjk,tlk->til", Vt, D, U)       # R = V D U^T
    fit = np.einsum("tij,mj->tmi", R, ref - rc) + dc
    rms = np.sqrt(((fit - dyn) ** 2).sum(-1).mean(-1))
    out = np.einsum("tij,pj->tpi", R, pts - rc) + dc
    return out, rms


def _cluster(d, seg):
    return np.array([d[f"{seg}_{i}"] for i in range(1, 5)], float)


def to_canonical(js: dict) -> tuple[np.ndarray, dict]:
    jn, ne, wk = js["joints"], js["neutral"], js["walking"]
    A = {k: np.asarray(v, float) for k, v in jn.items()}
    W = {k: np.asarray(v, float) for k, v in wk.items()}
    static = {
        "L_hip": A["L_hip"] + (A["R_hip"] - A["L_hip"]) / 4,
        "R_hip": A["R_hip"] + (A["L_hip"] - A["R_hip"]) / 4,
    }
    for s in "LR":
        static[f"{s}_knee"] = (A[f"{s}_lat_knee"] + A[f"{s}_med_knee"]) / 2
        static[f"{s}_ankle"] = (A[f"{s}_lat_ankle"] + A[f"{s}_med_ankle"]) / 2
        foot = _cluster(ne, f"{s}_foot")
        toe = np.asarray(ne[f"{s}_toe"], float)
        fwd = toe - foot.mean(0)
        post = foot[np.argmin(foot @ fwd)]
        heel = post.copy()
        heel[1] = min(A[f"{s}_first"][1], A[f"{s}_fifth"][1])   # floor-ish height (Y-up)
        static[f"{s}_heel"] = heel
        static[f"{s}_toe"] = toe

    T = len(W["pelvis_1"])
    out = {}
    qc = {}
    tracks = [("pelvis", ["L_hip", "R_hip"]),
              ("L_thigh", ["L_knee"]), ("R_thigh", ["R_knee"]),
              ("L_shank", ["L_ankle"]), ("R_shank", ["R_ankle"]),
              ("L_foot", ["L_heel", "L_toe"]), ("R_foot", ["R_heel", "R_toe"])]
    for seg, names in tracks:
        ref = _cluster(ne, seg)
        dyn = np.stack([W[f"{seg}_{i}"] for i in range(1, 5)], axis=1)
        pts, rms = kabsch_track(ref, dyn, np.array([static[n] for n in names]))
        for n, p in zip(names, pts.transpose(1, 0, 2)):
            out[n] = p
        qc[f"rms_{seg}"] = float(np.nanmedian(rms))
    # toe marker is recorded in walking — prefer it over the cluster-tracked estimate
    for s in "LR":
        if f"{s}_toe" in W and np.isfinite(W[f"{s}_toe"]).all():
            out[f"{s}_toe"] = W[f"{s}_toe"]
    out["pelvis"] = 0.5 * (out["L_hip"] + out["R_hip"])
    xyz = np.stack([out[j] for j in JOINTS], axis=1) / 1000.0
    assert xyz.shape[0] == T
    return xyz, qc


def clean_injury(s: pd.Series) -> pd.Series:
    """README's harmonisation of SpecInjury labels (Table 1 of the paper)."""
    s = s.astype("string").str.strip().str.lower()
    s = s.replace({"oa": "osteoarthritis", "hip oa": "osteoarthritis", "knee oa": "osteoarthritis",
                   "osteroarthritis": "osteoarthritis", "itbs": "itb syndrome",
                   "pfps": "patellofemoral pain syndrome"})
    return s.replace({"nan": pd.NA, "null": pd.NA, "n/a": pd.NA, "": pd.NA})


def info() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "walk_data_meta.csv")
    d["subject"] = "ferber_" + d.sub_id.astype(str)
    d["trial"] = d.sub_id.astype(str) + "_" + d.filename.str.replace(".json", "", regex=False)
    d["path"] = [RAW / str(s) / f for s, f in zip(d.sub_id, d.filename)]
    d["SpecInjury_clean"] = clean_injury(d.SpecInjury)
    inj_joint = d.InjJoint.astype("string").str.lower().fillna("no injury")
    d["injured"] = ~(d.InjDefn.astype("string").str.lower().eq("no injury")
                     & inj_joint.str.startswith("no injury")
                     & d.SpecInjury_clean.isna())
    d["InjSide_clean"] = d.InjSide.astype("string").str.strip().str.lower()
    d.loc[~d.injured, "InjSide_clean"] = "none"
    return d.rename(columns={"speed_w": "speed", "Height": "height_cm", "Weight": "mass_kg",
                             "Gender": "sex"})


def load_trial(row) -> dict:
    with open(row.path) as f:
        js = json.load(f)
    fs = float(js["hz_w"])
    xyz, qc = to_canonical(js)
    n_missing = int(np.isnan(xyz).any(-1).any(-1).sum())
    return dict(xyz=xyz.astype(np.float32), fs=fs, up=UP, n_missing_frames_raw=n_missing, **qc)
