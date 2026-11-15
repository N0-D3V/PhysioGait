"""Video keypoints -> model input -> BiGRU -> gait report (the demo's analysis).

    kp, score, fps, size, boxes = PoseTracker().run("walk.mp4")        # pose_video.py
    report = Analyzer().from_keypoints(kp, score, fps, leg_length_m=0.86)

Steps: keypoints -> our 10 joints at 30 Hz; fix left/right mix-ups; split the clip into walking passes (one per
direction); centre on the hips and scale by leg length; run the model on overlapping 5 s windows; turn the
events and angles into strides and outputs with the gait-lab formulas.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

import augment
from evaluate import RUN, load_model, run_model
from helpers.analysis import ROOT
from helpers.canonical import resample
from helpers.gaitdata import EVENTS, FS, IN_JOINTS, STRIDE_COLS, TARGET_NAMES, TI, aggregate, load_all, strides_from_events
from model import decode_events
from pose_video import H

W = 150                      # 5 s windows at 30 Hz
VIDEO_SIGMA = 0.04           # simulated noise level whose errors are quoted for real video (closest match on Kuopio)
LEG_PER_HEIGHT = 0.49        # hip-to-ankle length / body height (average proportion)
SWAP = augment.JOINT_SWAP
TOE = ["big_toe", "small_toe"]

# outputs shown in the report: (key, label, unit, per-side stride column)
REPORT = [("cadence", "Cadence", "steps/min", None), ("speed_ms", "Walking speed", "m/s", None),
          ("stride_time", "Stride time", "s", "stride_time"), ("step_time", "Step time", "s", "step_time"),
          ("stance_time", "Stance time", "s", "stance_time"), ("swing_time", "Swing time", "s", "swing_time"),
          ("stance_pct", "Stance", "% of stride", "stance_pct"),
          ("double_support", "Double support", "s", "double_support"),
          ("SI_step_time", "Step-time symmetry (SI)", "%", None), ("SI_stance_time", "Stance-time symmetry (SI)", "%", None),
          ("SI_swing_time", "Swing-time symmetry (SI)", "%", None), ("hip_rom", "Hip range of motion", "deg", "hip_rom"),
          ("knee_rom", "Knee range of motion", "deg", "knee_rom"), ("ankle_rom", "Ankle range of motion", "deg", "ankle_rom"),
          ("peak_knee_flex", "Peak knee flexion", "deg", "peak_knee_flex"),
          ("SI_hip_rom", "Hip ROM symmetry (SI)", "%", None), ("SI_knee_rom", "Knee ROM symmetry (SI)", "%", None),
          ("SI_ankle_rom", "Ankle ROM symmetry (SI)", "%", None),
          ("stride_time_cv", "Stride-time variability (CV)", "%", None)]


# ---------------------------------------------------------------- keypoints -> model input
def halpe_to_canonical(kp, score):
    """RTMPose Halpe-26 keypoints [T, 26, 2] -> our 10 joints [T, 10, 2] (y pointing up) and their scores."""
    x = np.zeros((len(kp), len(IN_JOINTS), 2), np.float32)
    s = np.zeros((len(kp), len(IN_JOINTS)), np.float32)
    for j, name in enumerate(IN_JOINTS):
        side, part = name[0], name[2:]
        src = [H[f"{side}_{t}"] for t in (TOE if part == "toe" else [part])]
        x[:, j] = np.nanmean(kp[:, src], 1)
        s[:, j] = score[:, src].min(1)
    x[..., 1] *= -1
    s[~np.isfinite(x).all(-1)] = 0
    return np.nan_to_num(x), s


def resample_30(x, s, fps):
    if abs(fps - FS) < 0.5:
        return x, s
    return resample(x, fps).astype(np.float32), resample(s, fps).astype(np.float32)


def fix_left_right(x: np.ndarray, vis: np.ndarray, penalty: float = 0.5) -> np.ndarray:
    """Per frame: True where the pose estimator mixed up the left and right leg.

    Pose estimators often swap the legs in side views. Each frame is either "as detected" or "swapped"; we pick
    the sequence (dynamic programming) in which every joint moves most smoothly (closest to where its last two
    positions predict it), with a penalty per switch. 0.5 restored all injected swaps in tests on mocap.
    """
    T = len(x)
    if T < 3:
        return np.zeros(T, bool)
    alt = [x, x[:, SWAP]]
    va = [vis, vis[:, SWAP]]
    scale = np.nanmedian(np.linalg.norm(x[:, 0] - x[:, 2], axis=-1)[(vis[:, 0] * vis[:, 2]) > 0]) or 1.0
    INF = 1e18
    # D[a, b] = best cost with state(t-1) = a, state(t) = b
    D = np.zeros((2, 2)) + np.array([[0, 1], [0, 1]]) * 0.05
    back = np.zeros((T, 2, 2), np.int8)
    for t in range(2, T):
        Dn = np.full((2, 2), INF)
        for b in range(2):
            for c in range(2):
                best, arg = INF, 0
                for a in range(2):
                    pred = 2 * alt[b][t - 1] - alt[a][t - 2]
                    ok = (va[c][t] * va[b][t - 1] * va[a][t - 2]) > 0
                    cost = np.linalg.norm(alt[c][t][ok] - pred[ok], axis=-1).sum() / scale if ok.any() else 0.0
                    v = D[a, b] + cost
                    if v < best:
                        best, arg = v, a
                Dn[b, c] = best + 0.05 * c + penalty * (c != b)
                back[t, b, c] = arg
        D = Dn
    b, c = np.unravel_index(np.argmin(D), D.shape)
    states = np.zeros(T, np.int8)
    states[-1], states[-2] = c, b
    for t in range(T - 1, 1, -1):
        a = back[t, states[t - 1], states[t]]
        states[t - 2] = a
    return states.astype(bool)


def prepare(x, s, fps, conf):
    """Keypoints at the video rate -> 30 Hz, visibility flags, left/right mix-ups fixed."""
    x, s = resample_30(x, s, fps)
    vis = (s > conf).astype(np.float32)
    swapped = fix_left_right(x, vis)
    x[swapped], vis[swapped], s[swapped] = x[swapped][:, SWAP], vis[swapped][:, SWAP], s[swapped][:, SWAP]
    return x, s, vis, swapped


def facing(x, vis):
    """Per-frame walking direction from the feet: sum over visible feet of sign(toe x - heel x)."""
    fx = np.zeros(len(x))
    for s in "LR":
        h, t = IN_JOINTS.index(f"{s}_heel"), IN_JOINTS.index(f"{s}_toe")
        fx += vis[:, h] * vis[:, t] * np.sign(x[:, t, 0] - x[:, h, 0])
    return fx


def split_passes(x, vis, min_len=45, max_gap=15):
    """Walking passes of one direction -> [(start, end, +1/-1)]. Turns and frames without hips and ankles end a
    pass, gaps up to `max_gap` frames are bridged, passes shorter than 1.5 s are dropped."""
    hips = vis[:, [IN_JOINTS.index("L_hip"), IN_JOINTS.index("R_hip")]].sum(1) > 0
    ankles = vis[:, [IN_JOINTS.index("L_ankle"), IN_JOINTS.index("R_ankle")]].sum(1) > 0
    valid = hips & ankles
    lab = np.where(valid, np.sign(median_filter(np.where(valid, facing(x, vis), 0.0), size=31, mode="nearest")), 0)
    lab = lab.astype(int)
    edges = np.flatnonzero(np.diff(lab)) + 1
    runs = [(int(a), int(b), lab[a]) for a, b in zip(np.r_[0, edges], np.r_[edges, len(lab)])]
    for i in range(1, len(runs) - 1):
        a, b, v = runs[i]
        if v == 0 and b - a <= max_gap and runs[i - 1][2] == runs[i + 1][2] != 0:
            lab[a:b] = runs[i - 1][2]
    edges = np.flatnonzero(np.diff(lab)) + 1
    return [(int(a), int(b), int(lab[a])) for a, b in zip(np.r_[0, edges], np.r_[edges, len(lab)])
            if lab[a] != 0 and b - a >= min_len]


def normalise_pass(x, vis, direction):
    """Pixels -> hip-centred, divided by the median hip-to-ankle length, walking towards +x."""
    x = x.copy()
    x[..., 0] *= direction
    hips = [IN_JOINTS.index("L_hip"), IN_JOINTS.index("R_hip")]
    pelvis = np.nanmean(np.where(vis[:, hips, None] > 0, x[:, hips], np.nan), 1)
    t = np.arange(len(x))
    for c in range(2):                                   # frames without a visible hip: interpolate
        ok = np.isfinite(pelvis[:, c])
        pelvis[:, c] = np.interp(t, t[ok], pelvis[ok, c]) if ok.any() else 0
    legs = []
    for s in "LR":
        h, a = IN_JOINTS.index(f"{s}_hip"), IN_JOINTS.index(f"{s}_ankle")
        ok = (vis[:, h] * vis[:, a]) > 0
        legs.append(np.linalg.norm(x[ok, h] - x[ok, a], axis=-1))
    L = float(np.median(np.concatenate(legs))) if sum(map(len, legs)) else 1.0
    xn = (x - pelvis[:, None]) / L
    xn[vis == 0] = 0
    return xn.astype(np.float32), vis.astype(np.float32), L


# ---------------------------------------------------------------- reference tables
def reference_ranges():
    """5th-95th percentile of every output over the healthy Fukuchi adults (cached)."""
    p = ROOT / "outputs" / "models" / RUN / "reference_ranges.csv"
    if p.exists():
        return pd.read_csv(p, index_col=0)
    rows = [dict(zip(TARGET_NAMES, aggregate(t.strides, 0, (len(t.pose) - 1) / FS, t.speed_ll)))
            | {"speed_ms": t.speed_ll * t.leg_length} for split in load_all().values() for t in split if t.ds == "fukuchi"]
    df = pd.DataFrame(rows)
    out = pd.DataFrame({"p5": df.quantile(0.05), "p95": df.quantile(0.95)})
    out.to_csv(p)
    return out


def expected_errors():
    """Per-recording average error of the model and of guessing the average, per output and noise level."""
    R = pd.read_csv(ROOT / "outputs" / "eval" / "test_trial" / "regression.csv")
    return R[R.method.isin([f"{RUN}+derived", "mean"])].groupby(["method", "target", "level"]).mae.mean().unstack("method")


# ---------------------------------------------------------------- analysis
@dataclass
class Report:
    summary: pd.DataFrame           # one row per output
    strides: pd.DataFrame           # one row per stride
    events: dict                    # event times (s)
    angles: np.ndarray              # [T, 6] model angle curves (deg)
    passes: list                    # [(start, end, direction)]
    quality: dict
    x: np.ndarray                   # [T, 10, 2] model input
    vis: np.ndarray
    window_speed: np.ndarray        # [n, 2]: window centre (s), speed (leg lengths/s)
    leg_length_m: float | None = None
    px: np.ndarray | None = None    # [T, 10, 2] keypoints in image pixels (y down), after the left/right fix


class Analyzer:
    def __init__(self):
        self.m, self.cfg, self.norm = load_model()
        self.ref = reference_ranges()
        self.err = expected_errors()

    def _run_pass(self, x, vis, offset):
        """One walking pass: 5 s windows overlapping by half, blended with weights that fade at the window edges,
        then events and strides decoded once over the whole pass (no stride lost at a window boundary)."""
        T = len(x)
        wins = [(0, T)] if T <= W else [(s, s + W) for s in range(0, T - W + 1, W // 2)]
        if wins[-1][1] < T:
            wins.append((T - W, T))
        probs, ys, angs = run_model(self.m, self.norm, [x[a:b] for a, b in wins], [vis[a:b] for a, b in wins])
        prob, ang, wsum = np.zeros((T, len(EVENTS))), np.zeros((T, 6)), np.zeros((T, 1))
        for (a, b), p, an in zip(wins, probs, angs):
            w = np.clip(np.sin(np.pi * (np.arange(b - a) + 0.5) / (b - a)), 0.02, 1)[:, None]
            prob[a:b] += w * p
            ang[a:b] += w * an
            wsum[a:b] += w
        prob, ang = prob / wsum, ang / wsum
        ev = decode_events(prob)
        S = strides_from_events(ev, ang)
        S[:, :2] += offset / FS
        ev = {k: v + offset / FS for k, v in ev.items()}
        wspeed = [((a + b) / 2 / FS + offset / FS, float(y[TI["speed_ll"]])) for (a, b), y in zip(wins, ys)]
        return S, ev, ang, float(np.median([v for _, v in wspeed])), wspeed

    def from_canonical(self, x, vis, passes, leg_length_m=None, sigma=VIDEO_SIGMA):
        """x [T, 10, 2] already normalised per pass. sigma: pose noise level used for the typical-error column."""
        T = len(x)
        strides, events, ang = [], {k: [] for k in EVENTS}, np.full((T, 6), np.nan)
        speeds, wspeeds = [], []
        for a, b, _ in passes:
            S, ev, ang[a:b], sp, ws = self._run_pass(x[a:b], vis[a:b], a)
            strides.append(S)
            for e in EVENTS:
                events[e].append(ev[e])
            speeds.append((sp, b - a))
            wspeeds += ws
        S = _dedupe_strides(np.concatenate(strides))
        ev = {e: np.sort(np.concatenate(v)) for e, v in events.items()}
        speed_ll = float(np.average([s for s, _ in speeds], weights=[n for _, n in speeds]))
        y = aggregate(S, -np.inf, np.inf, speed_ll)
        hs = sorted([(t, "L") for t in ev["HS_L"]] + [(t, "R") for t in ev["HS_R"]])
        breaks = sum(a[1] == b[1] and b[0] - a[0] < 2.0 for a, b in zip(hs, hs[1:]))   # same foot twice in a row
        quality = dict(sigma=sigma, hs_alternation_breaks=int(breaks), n_strides_L=int((S[:, 2] == 0).sum()),
                       n_strides_R=int((S[:, 2] == 1).sum()), visible_frac=float((vis > 0).mean()),
                       both_directions=len({d for _, _, d in passes}) > 1, passes=len(passes))
        return Report(self._summary(y, S, speed_ll, leg_length_m, sigma), _stride_df(S), ev, ang, passes, quality,
                      x, vis, np.array(wspeeds).reshape(-1, 2), leg_length_m)

    def from_keypoints(self, kp, score, fps, leg_length_m=None, conf=0.3):
        x, s, vis, swapped = prepare(*halpe_to_canonical(kp, score), fps, conf)
        passes = split_passes(x, vis)
        if not passes:
            raise ValueError("no walking pass of at least 1.5 s with a visible, side-on person was found")
        xn = np.zeros_like(x)
        for a, b, d in passes:
            xn[a:b], vis[a:b], _ = normalise_pass(x[a:b], vis[a:b], d)
        rep = self.from_canonical(xn, vis, passes, leg_length_m)
        rep.px = x * np.array([1, -1], np.float32)
        rep.quality["lr_swaps_fixed"] = int((np.diff(np.r_[0, swapped.astype(int)]) == 1).sum())
        return rep

    def _summary(self, y, S, speed_ll, leg_length_m, sigma):
        levels = np.array(sorted(self.err.index.get_level_values("level").unique()))
        rows = []
        for key, label, unit, col in REPORT:
            val = y[TI[key]] if key != "speed_ms" else speed_ll * (leg_length_m or 1)
            if key == "speed_ms" and not leg_length_m:
                unit = "leg lengths/s"
            left = right = np.nan
            if col:
                left, right = (S[S[:, 2] == i, 3 + STRIDE_COLS.index(col)].mean() if (S[:, 2] == i).any() else np.nan
                               for i in (0, 1))
            mae, rel = np.nan, ""
            if key in self.err.index.get_level_values("target") and unit != "leg lengths/s":
                e = self.err.loc[key]
                mae = float(np.interp(sigma, levels, e[f"{RUN}+derived"].reindex(levels).to_numpy()))
                guess = float(np.interp(sigma, levels, e["mean"].reindex(levels).to_numpy()))
                rel = "good" if mae < 0.5 * guess else "rough" if mae < 0.85 * guess else "unreliable"
            ref = self.ref.loc[key] if key in self.ref.index and unit != "leg lengths/s" else None
            rows.append(dict(key=key, measure=label, value=val, unit=unit, left=left, right=right, expected_error=mae,
                             reliability=rel, healthy_p5=ref.p5 if ref is not None else np.nan,
                             healthy_p95=ref.p95 if ref is not None else np.nan))
        return pd.DataFrame(rows)


def _dedupe_strides(S, tol=0.15):
    """Drop strides found twice (passes can overlap by a few frames)."""
    S = S[np.lexsort((S[:, 0], S[:, 2]))]
    keep = np.r_[True, ~((S[1:, 2] == S[:-1, 2]) & (np.abs(S[1:, 0] - S[:-1, 0]) < tol))] if len(S) else np.zeros(0, bool)
    S = S[keep]
    return S[np.argsort(S[:, 0])]


def _stride_df(S):
    df = pd.DataFrame(S, columns=["start_s", "end_s", "side"] + STRIDE_COLS)
    df["side"] = df.side.map({0.0: "L", 1.0: "R"})
    return df
