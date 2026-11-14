"""Classic (no learning) pipeline on the model's input: the baseline the brief requires.

gap-fill hidden keypoints -> 6 Hz low-pass -> Zeni events (+ the same force-plate bias
correction used for the labels) -> stride formulas -> window labels. Speed: during
stance the foot is fixed on the ground, so its backward velocity relative to the pelvis
is the walking speed (leg lengths / s).
"""
from __future__ import annotations

import numpy as np

from canonical import J, SIDES, detect_events, lowpass, sagittal_angles, stride_table
from gaitdata import EVENTS, FS, IN_IDX, TI, aggregate, strides_to_array


def to_sag(x: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """Model input [T,10,2] + visibility -> filtered canonical [T,11,2] (pelvis = 0)."""
    T = len(x)
    t = np.arange(T)
    x = x.copy()
    for j in range(x.shape[1]):
        ok = vis[:, j] > 0
        if ok.sum() < 2:
            x[:, j] = 0
            continue
        for c in range(2):
            x[:, j, c] = np.interp(t, t[ok], x[ok, j, c])
    if T > 15:
        x = lowpass(x.reshape(T, -1), FS).reshape(x.shape)
    sag = np.zeros((T, 11, 2))
    sag[:, IN_IDX] = x
    return sag


def events(sag: np.ndarray, offsets: dict[str, float]) -> dict[str, np.ndarray]:
    ev = detect_events(sag, FS)
    return {k: v - offsets[k[:2]] * FS for k, v in ev.items()}   # fractional 30 Hz frames


def speed_ll(sag: np.ndarray, ev: dict) -> float:
    v = []
    for s in SIDES:
        vx = np.gradient(sag[:, J[f"{s}_ankle"], 0]) * FS
        for hs in ev[f"HS_{s}"]:
            to = ev[f"TO_{s}"][ev[f"TO_{s}"] > hs]
            if len(to) and to[0] - hs < 1.5 * FS:
                a, b = int(round(hs)), int(round(to[0]))
                q = (b - a) // 4
                v.append(-np.median(vx[a + q:b - q + 1]))
    return float(np.median(v)) if v else np.nan


def predict(x: np.ndarray, vis: np.ndarray, offsets: dict[str, float]):
    """-> (events in seconds {name: array}, window labels [n_targets])."""
    sag = to_sag(x, vis)
    ev = events(sag, offsets)
    evi = {k: np.clip(np.round(v).astype(int), 0, len(sag) - 1) for k, v in ev.items()}
    S = strides_to_array(stride_table(evi, sagittal_angles(sag), FS), FS)
    y = aggregate(S, 0, (len(sag) - 1) / FS, speed_ll(sag, ev))
    return {k: ev[k] / FS for k in EVENTS}, y
