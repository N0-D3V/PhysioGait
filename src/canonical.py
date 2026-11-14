"""Canonical skeleton, sagittal projection, normalisation, events and gait parameters.

Every dataset loader produces `xyz` [T, J, 3] in metres in its lab frame with the
joint order below, plus the lab's progression and up axes. Everything downstream
(projection, events, angles, spatiotemporal labels) is shared and lives here.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import interp1d

JOINTS = [
    "L_hip", "L_knee", "L_ankle", "L_heel", "L_toe",
    "R_hip", "R_knee", "R_ankle", "R_heel", "R_toe",
    "pelvis",
]
J = {n: i for i, n in enumerate(JOINTS)}
SIDES = ("L", "R")
TARGET_FS = 30.0  # video-like rate for model windows


# ---------------------------------------------------------------- filtering / gaps
def fill_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate NaN runs up to `max_gap` frames along axis 0; longer runs stay NaN."""
    x = np.array(x, dtype=np.float64, copy=True)
    flat = x.reshape(len(x), -1)
    t = np.arange(len(x))
    for c in range(flat.shape[1]):
        col = flat[:, c]
        bad = np.isnan(col)
        if not bad.any() or bad.all():
            continue
        filled = np.interp(t, t[~bad], col[~bad])
        # restore long gaps
        edges = np.diff(np.r_[0, bad.astype(int), 0])
        for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            if e - s <= max_gap:
                col[s:e] = filled[s:e]
    return flat.reshape(x.shape)


def lowpass(x: np.ndarray, fs: float, fc: float = 6.0, order: int = 4) -> np.ndarray:
    """Zero-lag Butterworth low-pass along axis 0 (NaN-free input required)."""
    b, a = butter(order, fc / (fs / 2))
    return filtfilt(b, a, x, axis=0)


# ---------------------------------------------------------------- projection
def progression_axis_pca(pelvis_xyz: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Dominant horizontal direction of pelvis travel (overground trials)."""
    h = pelvis_xyz - np.outer(pelvis_xyz @ up, up)
    h = h - np.nanmean(h, axis=0)
    h = h[~np.isnan(h).any(1)]
    _, _, vt = np.linalg.svd(h, full_matrices=False)
    f = vt[0] - (vt[0] @ up) * up
    f /= np.linalg.norm(f)
    # orient along the net direction of travel
    if (pelvis_xyz[-1] - pelvis_xyz[0]) @ f < 0:
        f = -f
    return f


def sagittal_project(xyz: np.ndarray, forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """[T,J,3] -> [T,J,2] (forward, up) in the global/camera plane."""
    up = up / np.linalg.norm(up)
    f = forward - (forward @ up) * up
    f /= np.linalg.norm(f)
    return np.stack([xyz @ f, xyz @ up], axis=-1)


def leg_length(sag: np.ndarray) -> float:
    """Median hip->ankle distance over both legs (same units as `sag`)."""
    ll = [np.linalg.norm(sag[:, J[f"{s}_hip"]] - sag[:, J[f"{s}_ankle"]], axis=-1) for s in SIDES]
    return float(np.nanmedian(np.concatenate(ll)))


def normalise(sag: np.ndarray) -> tuple[np.ndarray, float]:
    """Pelvis-centre each frame and scale by leg length."""
    L = leg_length(sag)
    return (sag - sag[:, J["pelvis"]][:, None, :]) / L, L


def resample(x: np.ndarray, fs: float, fs_new: float = TARGET_FS) -> np.ndarray:
    t = np.arange(len(x)) / fs
    tn = np.arange(0, t[-1] + 1e-9, 1 / fs_new)
    return interp1d(t, x, axis=0)(tn)


# ---------------------------------------------------------------- events (Zeni 2008)
def detect_events(sag_raw: np.ndarray, fs: float, min_stride_s: float = 0.7) -> dict[str, np.ndarray]:
    """Coordinate-based heel-strike / toe-off detection on sagittal coords (metres).

    HS = peak of (heel - pelvis) along progression; TO = trough of (toe - pelvis).
    """
    out = {}
    dist = int(min_stride_s * fs * 0.8)
    for s in SIDES:
        heel = sag_raw[:, J[f"{s}_heel"], 0] - sag_raw[:, J["pelvis"], 0]
        toe = sag_raw[:, J[f"{s}_toe"], 0] - sag_raw[:, J["pelvis"], 0]
        rng_h = np.nanpercentile(heel, 95) - np.nanpercentile(heel, 5)
        rng_t = np.nanpercentile(toe, 95) - np.nanpercentile(toe, 5)
        out[f"HS_{s}"], _ = find_peaks(heel, distance=dist, prominence=0.3 * rng_h)
        out[f"TO_{s}"], _ = find_peaks(-toe, distance=dist, prominence=0.3 * rng_t)
    return out


# ---------------------------------------------------------------- angles
def _seg_angle(p_from, p_to):
    """Angle of segment (from->to) w.r.t. downward vertical, +ve = pointing forward (deg)."""
    v = p_to - p_from
    return np.degrees(np.arctan2(v[..., 0], -v[..., 1]))


def sagittal_angles(sag: np.ndarray) -> dict[str, np.ndarray]:
    """2D sagittal joint angles (deg) from projected coords.

    hip   = thigh angle vs vertical (no trunk/pelvis tilt available) — flexion +ve
    knee  = thigh - shank angle — flexion +ve
    ankle = foot (heel->toe) vs shank-perpendicular — dorsiflexion +ve
    """
    out = {}
    for s in SIDES:
        hip, knee, ankle = (sag[:, J[f"{s}_{n}"]] for n in ("hip", "knee", "ankle"))
        heel, toe = sag[:, J[f"{s}_heel"]], sag[:, J[f"{s}_toe"]]
        thigh = _seg_angle(hip, knee)
        shank = _seg_angle(knee, ankle)
        foot = np.degrees(np.arctan2(toe[:, 1] - heel[:, 1], toe[:, 0] - heel[:, 0]))  # +ve = toes up
        out[f"{s}_hip"] = thigh
        out[f"{s}_knee"] = thigh - shank
        out[f"{s}_ankle"] = foot - shank
    return out


# ---------------------------------------------------------------- per-stride / per-trial labels
def symmetry_index(l: float, r: float) -> float:
    """Robinson SI (%): 0 = perfect symmetry."""
    m = 0.5 * (l + r)
    return float(100 * (l - r) / m) if m else np.nan


def stride_table(ev: dict, angles: dict, fs: float) -> list[dict]:
    """One row per stride (HS->next HS, same side) with timing and ROM."""
    rows = []
    for s in SIDES:
        o = "R" if s == "L" else "L"
        hs, to, hs_o, to_o = ev[f"HS_{s}"], ev[f"TO_{s}"], ev[f"HS_{o}"], ev[f"TO_{o}"]
        for a, b in zip(hs[:-1], hs[1:]):
            st = (b - a) / fs
            if not 0.6 < st < 2.5:
                continue
            t_off = to[(to > a) & (to < b)]
            o_hs = hs_o[(hs_o > a) & (hs_o < b)]
            o_to = to_o[(to_o > a) & (to_o < b)]
            if len(t_off) != 1 or len(o_hs) != 1 or len(o_to) != 1:
                continue
            stance = (t_off[0] - a) / fs
            ds1 = (o_to[0] - a) / fs                # initial double support
            ds2 = (t_off[0] - o_hs[0]) / fs         # terminal double support
            seg = slice(a, b + 1)
            row = dict(side=s, start=int(a), end=int(b), stride_time=st,
                       stance_time=stance, swing_time=st - stance,
                       stance_pct=100 * stance / st,
                       step_time=(o_hs[0] - a) / fs,   # this HS -> contralateral HS
                       double_support=ds1 + ds2)
            for j in ("hip", "knee", "ankle"):
                ang = angles[f"{s}_{j}"][seg]
                row[f"{j}_rom"] = float(np.nanmax(ang) - np.nanmin(ang))
            row["peak_knee_flex"] = float(np.nanmax(angles[f"{s}_knee"][seg]))
            rows.append(row)
    return rows


def trial_summary(strides: list[dict]) -> dict:
    """Aggregate per-stride rows into per-trial labels (heads 2, 4, 5, 6)."""
    import pandas as pd
    if not strides:
        return {"n_strides": 0}
    df = pd.DataFrame(strides)
    out = {"n_strides": len(df)}
    out["cadence"] = 120.0 / df.stride_time.mean()  # steps/min
    for c in ("stride_time", "stance_time", "swing_time", "stance_pct", "double_support",
              "hip_rom", "knee_rom", "ankle_rom", "peak_knee_flex"):
        out[c] = df[c].mean()
    out["stride_time_sd"] = df.stride_time.std()
    out["stride_time_cv"] = 100 * df.stride_time.std() / df.stride_time.mean()
    by = df.groupby("side").mean(numeric_only=True)
    if {"L", "R"} <= set(by.index):
        for c in ("step_time", "stance_time", "swing_time", "knee_rom", "hip_rom", "ankle_rom"):
            out[f"SI_{c}"] = symmetry_index(by.loc["L", c], by.loc["R", c])
    return out
