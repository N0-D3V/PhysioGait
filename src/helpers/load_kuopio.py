"""Kuopio gait dataset (Zenodo 10559504): real side-view video keypoints (OpenPose) recorded together with
motion capture and force plates. Used only to test the model on real video.

Per subject: 10 + 10 walks at comfortable speed, `r_*` walking one way (right leg nearest the camera) and `l_*`
back. Mocap joint centres are Vicon's functional joint-centre markers; heel = rear foot marker, toe = mean of
the two forefoot markers. Video clips are ~1 s around the step on the middle force plate.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from helpers import c3d
from helpers.canonical import JOINTS

ROOT = Path(__file__).resolve().parents[2] / "data" / "kuopio"
RAW = ROOT / "raw"
FS_MOCAP = 100.0
PLATES = (1, 2, 3)                      # floor plates along the walkway (4, 5 are not in the walkway)
SYNC_JOINTS = ["L_hip", "R_hip", "L_knee", "R_knee", "L_ankle", "R_ankle"]

# BODY_25 index of each canonical joint (toe = mean of big and small toe; pelvis = mid-hip)
BODY25 = {"L_hip": [12], "L_knee": [13], "L_ankle": [14], "L_heel": [21], "L_toe": [19, 20],
          "R_hip": [9], "R_knee": [10], "R_ankle": [11], "R_heel": [24], "R_toe": [22, 23], "pelvis": [8]}
MOCAP = {"L_hip": ["Pelvis_LFemur_score"], "R_hip": ["Pelvis_RFemur_score"], "L_knee": ["LKnee"],
         "R_knee": ["RKnee"], "L_ankle": ["LTibia_LFoot_score"], "R_ankle": ["RTibia_RFoot_score"],
         "L_heel": ["LFoot1"], "R_heel": ["RFoot1"], "L_toe": ["LFoot2", "LFoot3"], "R_toe": ["RFoot2", "RFoot3"]}


def info() -> pd.DataFrame:
    d = pd.read_excel(ROOT / "info_participants.xlsx")
    d.columns = [c.strip() for c in d.columns]
    d["subject"] = d["ID"].map(lambda i: f"{int(i):02d}")
    return d


def invalid_trials(row) -> set[str]:
    v = row.get("Invalid_trials")
    return set(re.findall(r"[lr]_\w+?_\d\d", str(v))) if isinstance(v, str) else set()


def trials(subject: str, speed: str = "comf") -> list[str]:
    d = RAW / subject
    return sorted(p.name[5:] for p in (d / "openpose").glob(f"json_[lr]_{speed}_*")
                  if (d / "mocap" / f"{p.name[5:]}.c3d").exists())


# ---------------------------------------------------------------- video keypoints
def openpose(subject: str, trial: str) -> np.ndarray:
    """[n, 25, 3] (x, y, confidence) of the walking subject; NaN where not detected.

    Several people are in view (operators, a seated person); the subject is the tallest skeleton,
    with a penalty on jumps of the mid-hip so the track does not switch person."""
    out, prev = [], None
    for f in sorted((RAW / subject / "openpose" / f"json_{trial}").glob("*_keypoints.json")):
        best, score = np.zeros((25, 3)), -np.inf
        for p in json.loads(f.read_text())["people"]:
            k = np.array(p["pose_keypoints_2d"], float).reshape(25, 3)
            ok = k[:, 2] > 0.1
            if ok.sum() < 6:
                continue
            sc = np.ptp(k[ok, 1]) - (0.5 * abs(k[8, 0] - prev) if prev is not None and k[8, 2] > 0 else 0)
            if sc > score:
                best, score = k, sc
        if best[8, 2] > 0:
            prev = best[8, 0]
        out.append(best)
    K = np.array(out)
    K[K[..., 2] <= 0, :2] = np.nan
    return K


def video_canonical(K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """BODY_25 -> canonical [n, 11, 2] pixels (y down) and confidence [n, 11]."""
    x = np.stack([np.nanmean(K[:, BODY25[j], :2], 1) for j in JOINTS], 1)
    s = np.stack([K[:, BODY25[j], 2].min(1) for j in JOINTS], 1)
    s[~np.isfinite(x).all(-1)] = 0
    return x, s


# ---------------------------------------------------------------- mocap
def mocap(subject: str, trial: str) -> dict:
    """Canonical joints [T, 11, 3] (m, lab frame), force-plate events (s) and walking direction."""
    r = c3d.read(RAW / subject / "mocap" / f"{trial}.c3d")
    L, P = r["labels"], r["points"] / 1000.0
    missing = sorted({m for ms in MOCAP.values() for m in ms} - set(L))
    if missing:                                   # subjects 07, 09, 10, 24 have no ankle joint centres
        raise KeyError(f"{subject}/{trial}: no {', '.join(missing)}")
    pts = {j: np.mean([P[:, L.index(m)] for m in ms], 0) for j, ms in MOCAP.items()}
    pts["pelvis"] = 0.5 * (pts["L_hip"] + pts["R_hip"])
    xyz = np.stack([pts[j] for j in JOINTS], 1)
    return dict(xyz=xyz, events=plate_events(r, xyz), direction=-1 if trial[0] == "r" else 1)


def plate_events(r: dict, xyz: np.ndarray, thresh: float = 20.0) -> dict[str, list[float]]:
    """HS/TO (s) from the floor plates; each contact is assigned to the foot whose ankle is lowest
    and over the plate at mid-contact. Contacts touching the trial start/end are dropped."""
    fs, an, lab = r["analog_rate"], r["analog"], r["analog_labels"]
    corners = np.asarray(r["params"]["FORCE_PLATFORM"]["CORNERS"], float) / 1000.0   # [plates, 4, 3]
    ev = {f"{e}_{s}": [] for e in ("HS", "TO") for s in "LR"}
    ank = {s: xyz[:, JOINTS.index(f"{s}_ankle")] for s in "LR"}
    for p in PLATES:
        fz = np.abs(an[:, lab.index(f"Force.Fz{p}")])
        on = fz > thresh
        d = np.diff(np.r_[0, on.astype(int), 0])
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if b - a < 0.2 * fs or a == 0 or b >= len(fz):
                continue
            mid = int((a + b) / 2 / fs * FS_MOCAP)
            c = corners[p - 1]
            lo, hi = c[:, :2].min(0) - 0.1, c[:, :2].max(0) + 0.1
            cand = [(ank[s][mid, 2], s) for s in "LR" if np.isfinite(ank[s][mid]).all()
                    and (lo <= ank[s][mid, :2]).all() and (ank[s][mid, :2] <= hi).all()]
            if not cand:
                continue
            s = min(cand)[1]
            ev[f"HS_{s}"].append(a / fs)
            ev[f"TO_{s}"].append(b / fs)
    return {k: sorted(v) for k, v in ev.items()}


def sagittal(xyz: np.ndarray, direction: int) -> np.ndarray:
    """Lab [T, J, 3] -> (forward, up) [T, J, 2] in the camera plane (walking is +forward)."""
    return np.stack([direction * xyz[..., 1], xyz[..., 2]], -1)


# ---------------------------------------------------------------- video <-> mocap sync
def _interp(M: np.ndarray, idx: np.ndarray) -> np.ndarray:
    T = len(M)
    flat = M.reshape(T, -1)
    t = np.arange(T)
    out = np.stack([np.interp(idx, t, flat[:, c], left=np.nan, right=np.nan) for c in range(flat.shape[1])], 1)
    return out.reshape((len(idx),) + M.shape[1:])


def _project(M: np.ndarray, geo) -> np.ndarray:
    """Lab (Y, Z) -> pixels for a camera looking along +X: px = -s*Y + bx, py = -s*Z + by."""
    s, bx, by = geo
    return np.stack([-s * M[..., 0] + bx, -s * M[..., 1] + by], -1)


def _cost(V, M, fps, geo, t0s) -> np.ndarray:
    """Median labelled joint distance (px) for each candidate start time (vectorised over t0s)."""
    n = len(V)
    idx = (np.asarray(t0s)[:, None] * FS_MOCAP + np.arange(n) * FS_MOCAP / fps).ravel()
    d = np.linalg.norm(_project(_interp(M, idx), geo).reshape((len(t0s), n) + V.shape[1:]) - V, axis=-1)
    ok = np.isfinite(d)
    med = np.nanmedian(np.where(ok, d, np.nan).reshape(len(t0s), -1), 1)
    return np.where(ok.reshape(len(t0s), -1).sum(1) > 0.5 * V.shape[0] * V.shape[1], med, np.inf)


def _best_t0(V, M, fps, geo, step=0.01, around=None):
    T = len(M)
    t0s = np.arange(0, T / FS_MOCAP - len(V) / fps, step) if around is None else \
        np.arange(around - 3 * step, around + 3 * step + 1e-9, step / 5)
    c = _cost(V, M, fps, geo, t0s)
    k = int(np.argmin(c))
    return float(c[k]), float(t0s[k])


def _fit_geo(pairs, fps, t0s):
    """Least-squares similarity (s, bx, by) given each trial's start time."""
    A, b = [], []
    for (V, M), t0 in zip(pairs, t0s):
        Mi = _interp(M, t0 * FS_MOCAP + np.arange(len(V)) * FS_MOCAP / fps)
        ok = np.isfinite(Mi).all(-1) & np.isfinite(V).all(-1)
        m, v = Mi[ok], V[ok]
        Ax = np.c_[-m[:, 0], np.ones(len(m)), np.zeros(len(m))]
        Ay = np.c_[-m[:, 1], np.zeros(len(m)), np.ones(len(m))]
        A += [Ax, Ay]
        b += [v[:, 0], v[:, 1]]
    A, b = np.concatenate(A), np.concatenate(b)
    r = np.abs(A @ np.linalg.lstsq(A, b, rcond=None)[0] - b)
    keep = r < 3 * np.median(r) + 1e-9                    # robust: drop swapped / missed keypoints
    return np.linalg.lstsq(A[keep], b[keep], rcond=None)[0]


def sync_subject(clips, fps):
    """Find where each video clip sits in its mocap recording, using geometry only.

    clips: trial -> (video hip/knee/ankle keypoints [n, 6, 2] in pixels, mocap joints [T, 6, 2] as lab (Y, Z) in m).
    One fixed side camera per subject (pixels = scale * -lab + offset, shared by all its trials) and one start
    time per clip, chosen so the left/right-labelled keypoints sit on the projected mocap joints.
    Returns (camera (scale, x offset, y offset), {trial: (start time s, median error px)}).
    """
    pairs = list(clips.values())
    # rough camera: typical scale, vertical offset from the hips, horizontal offset by scanning
    V0, M0 = pairs[0]
    scale = 330.0
    by = float(np.nanmedian(V0[:, :2, 1]) + scale * np.nanmedian(M0[:, :2, 1]))
    cost = {bx: np.median([_best_t0(V, M, fps, (scale, bx, by), 0.02)[0] for V, M in pairs[:6]])
            for bx in np.arange(-400, 1600, 30)}
    geo = np.array([scale, min(cost, key=cost.get), by])
    # refine: alternate start times and the shared camera
    for _ in range(4):
        t0s = [_best_t0(V, M, fps, geo, 0.01)[1] for V, M in pairs]
        geo = _fit_geo(pairs, fps, t0s)
    out = {}
    for name, (V, M) in clips.items():
        _, t = _best_t0(V, M, fps, geo, 0.01)
        e, t = _best_t0(V, M, fps, geo, 0.01, around=t)
        out[name] = (t, e)
    return geo, out
