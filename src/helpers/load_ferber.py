"""Ferber Running Injury Clinic, walking trials -> our skeleton.

Walking trials only have marker clusters (4 markers per segment); the joint centres come from the standing
calibration trial (hip, knee, ankle as in the dataset's own gait_kinematics.m) and are carried into every walking
frame by fitting each cluster rigidly (Kabsch). ~40 % of sessions have no toe markers and get a virtual toe.
Units mm, Y up, treadmill.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from helpers.canonical import JOINTS

ROOT = Path(__file__).resolve().parents[2] / "data" / "Ferber Running Injury Dataset Full"
RAW = ROOT / "extracted" / "reformat_data"
UP = np.array([0.0, 1.0, 0.0])
# virtual-foot constants, measured on sessions that do have toe + metatarsal markers
VIRTUAL_TOE_RATIO = 0.122   # static ankle-JC -> toe marker (along -Z) / standing height
VIRTUAL_TOE_MM = 208.0      # median of the above when height is missing
VIRTUAL_TOE_Y_MM = 52.0     # median static toe-marker height
VIRTUAL_HEEL_Y_MM = 40.0    # median static 1st-metatarsal marker height


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


class NoFootLandmarks(ValueError):
    pass


def to_canonical(js: dict, height_cm: float | None = None, allow_virtual_foot: bool = True) -> tuple[np.ndarray, dict]:
    jn, ne, wk = js["joints"], js["neutral"], js["walking"]
    virtual_foot = not all(f"{s}_toe" in ne or f"{s}_first" in jn for s in "LR")
    if virtual_foot and not allow_virtual_foot:
        raise NoFootLandmarks("no toe marker and no static metatarsal markers")
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
        # some sessions have no toe marker: fall back to the static metatarsal-head midpoint,
        # and if those are missing too, a virtual toe: subjects stood feet straight ahead (-Z)
        # on a template, ankle->toe = 0.122 x height (median of sessions that have both).
        if f"{s}_toe" in ne:
            toe = np.asarray(ne[f"{s}_toe"], float)
        elif f"{s}_first" in A:
            toe = (A[f"{s}_first"] + A[f"{s}_fifth"]) / 2
        else:
            foot_len = VIRTUAL_TOE_RATIO * height_cm * 10 if height_cm and 120 <= height_cm <= 220 else VIRTUAL_TOE_MM
            toe = static[f"{s}_ankle"] + np.array([0.0, 0.0, -foot_len])
            toe[1] = VIRTUAL_TOE_Y_MM
        fwd = toe - foot.mean(0)
        post = foot[np.argmin(foot @ fwd)]
        heel = post.copy()
        heel[1] = (min(A[f"{s}_first"][1], A[f"{s}_fifth"][1]) if f"{s}_first" in A
                   else VIRTUAL_HEEL_Y_MM)                         # floor-ish height (Y-up)
        static[f"{s}_heel"] = heel
        static[f"{s}_toe"] = toe

    T = len(W["pelvis_1"])
    out = {}
    qc = {}
    tracks = [("pelvis", ["L_hip", "R_hip"]),
              ("L_thigh", ["L_knee"]), ("R_thigh", ["R_knee"]),
              ("L_shank", ["L_ankle"]), ("R_shank", ["R_ankle"]),
              ("L_foot", ["L_heel", "L_toe"]), ("R_foot", ["R_heel", "R_toe"])]
    toe_source = ("virtual" if virtual_foot else
                  "marker" if all(f"{s}_toe" in W for s in "LR") else "metatarsals")
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
    qc["toe_source"] = toe_source
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
    d["InjSide_clean"] = d.InjSide.astype("string").str.strip().str.lower().replace({"bi-lateral": "bilateral"})
    d.loc[~d.injured, "InjSide_clean"] = "none"
    # placeholder values in the metadata (0 / 999 cm, 0 kg, age 255) -> NaN
    d["Height"] = d.Height.where(d.Height.between(120, 220))
    d["Weight"] = d.Weight.where(d.Weight.between(30, 200))
    d["age"] = d.age.where(d.age.between(10, 100))
    return d.rename(columns={"speed_w": "speed", "Height": "height_cm", "Weight": "mass_kg",
                             "Gender": "sex"})


def load_trial(row) -> dict:
    with open(row.path) as f:
        js = json.load(f)
    fs = float(js["hz_w"])
    xyz, qc = to_canonical(js, height_cm=row.get("height_cm"))
    n_missing = int(np.isnan(xyz).any(-1).any(-1).sum())
    return dict(xyz=xyz.astype(np.float32), fs=fs, up=UP, n_missing_frames_raw=n_missing, **qc)
