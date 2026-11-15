"""Toronto Older Adults gait archive -> our skeleton. Xsens suit (inertial mocap), Z up, 100 Hz, ~60 s walking
back and forth; split into straight passes. No heel point, so a virtual heel is fixed 5 cm behind the ankle.
The archive's own video is filmed from the front (no feet keypoints), so it is not used for testing video.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from helpers.canonical import JOINTS, lowpass

ROOT = Path(__file__).resolve().parents[2] / "data" / "toronto_older_adults"
FS = 100.0
UP = np.array([0.0, 0.0, 1.0])
HEEL_BACK = 0.05
SEG = {"pelvis": "Pelvis", "hip": "ULeg", "knee": "LLeg", "ankle": "Foot", "toe": "Toe"}
POSE_JOINTS = ["LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle"]


def info() -> pd.DataFrame:
    t = pd.read_excel(ROOT / "Participants.xlsx")
    t.columns = [c.strip() for c in t.columns]
    t = t[pd.to_numeric(t["Participant No."], errors="coerce").notna()].copy()
    t["pid"] = t["Participant No."].astype(int)
    t["subject"] = t.pid.map(lambda p: f"toronto_{p:02d}")
    return t.rename(columns={"Age (years)": "age", "Sex": "sex", "Height (cm)": "height_cm",
                             "Weight (lbs)": "weight_lbs", "Number of falls in the last 6 months": "falls_6mo",
                             "POMA-Balance": "poma_balance", "POMA-Gait": "poma_gait",
                             "BBS": "bbs", "TUG (s)": "tug_s"})


def _virtual_heel(ankle, toe, up):
    """Heel rigidly attached to the ankle->toe line, in the vertical plane containing the foot."""
    u = toe - ankle
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    side = np.cross(up, u)
    side /= np.linalg.norm(side, axis=1, keepdims=True)
    n = np.cross(u, side)                       # perpendicular to foot, pointing up
    # coefficients from the standing frame: heel = ankle + a*u + b*n
    target = ankle[0] - HEEL_BACK * (u[0] - (u[0] @ up) * up) / np.linalg.norm(u[0] - (u[0] @ up) * up)
    target = target - (target @ up - toe[0] @ up) * up   # drop to toe (floor) height
    a, b = (target - ankle[0]) @ u[0], (target - ankle[0]) @ n[0]
    return ankle + a * u + b * n


def load_xsens(pid: int) -> np.ndarray | None:
    p = ROOT / "Xsens" / f"OAW{pid:02d}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    get = lambda n: d[[f"{n}X", f"{n}Y", f"{n}Z"]].to_numpy(float)
    pts = {"pelvis": get("Pelvis")}
    for s in "LR":
        for k in ("hip", "knee", "ankle", "toe"):
            pts[f"{s}_{k}"] = get(f"{s}{SEG[k]}")
        pts[f"{s}_heel"] = _virtual_heel(pts[f"{s}_ankle"], pts[f"{s}_toe"], UP)
    return np.stack([pts[j] for j in JOINTS], axis=1)


def split_passes(xyz: np.ndarray, min_speed=0.4, min_len_s=4.0, trim_s=0.5) -> list[tuple[int, int]]:
    """Straight walking passes: sustained horizontal pelvis speed with a stable heading."""
    p = lowpass(xyz[:, JOINTS.index("pelvis"), :2], FS, fc=0.5)
    v = np.gradient(p, axis=0) * FS
    speed = np.linalg.norm(v, axis=1)
    heading = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    turn_rate = np.abs(np.gradient(heading)) * FS            # rad/s
    ok = (speed > min_speed) & (turn_rate < 0.6)
    d = np.diff(np.r_[0, ok.astype(int), 0])
    out = []
    trim = int(trim_s * FS)
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        s, e = s + trim, e - trim
        if (e - s) >= min_len_s * FS:
            out.append((s, e))
    return out


def pose_clips() -> pd.DataFrame:
    rows = []
    for f in sorted((ROOT / "Pose Tracking").rglob("*.csv")):
        m = re.match(r"OAW(\d+)-(\w+)-(top|bottom)-(front|back)-(\d+)\.csv", f.name)
        if m:
            rows.append(dict(path=f, pid=int(m[1]), subject=f"toronto_{int(m[1]):02d}", estimator=m[2],
                             camera=m[3], direction=m[4], clip=int(m[5])))
    return pd.DataFrame(rows)


def load_pose(path: Path) -> dict:
    d = pd.read_csv(path)
    xy = np.stack([d[[f"{j}_x", f"{j}_y"]].to_numpy(float) for j in POSE_JOINTS], axis=1)
    # some exports have no RAnkle_conf column at all
    conf = np.stack([d[f"{j}_conf"].to_numpy(float) if f"{j}_conf" in d else np.full(len(d), np.nan)
                     for j in POSE_JOINTS], axis=1)
    return dict(t=d["time"].to_numpy(float), xy=xy, conf=conf)
