"""2D keypoints -> canonical model input -> BiGRU (v4) -> gait report.

    from pose_video import PoseTracker
    from video_pipeline import Analyzer
    kp, score, fps, size, boxes = PoseTracker().run("walk.mp4")
    rep = Analyzer().from_keypoints(kp, score, fps, leg_length_m=0.86)

Steps (all in `from_keypoints`):
 1. Halpe-26 -> the 10 canonical joints (toe = mid big/small toe, as Fukuchi's mid MT1/MT5),
    image y flipped to "up", resampled to 30 Hz; keypoints below `conf` are hidden.
 2. Left/right label swaps (common for side views) fixed by a Viterbi pass over
    "as detected" / "legs swapped" with a constant-velocity motion cost.
 3. Facing direction per frame from the feet (toe ahead of heel); the clip is split into passes
    of constant direction (turns and gaps dropped) and x is flipped so walking is +x.
 4. Per pass: pelvis-centred (mid-hip), scaled by the median hip->ankle length -> model input.
 5. 5 s windows -> events + angle curves -> strides with the label formulas; strides pooled over the
    whole clip. With passes in both directions, each leg is measured from the passes in which it
    is the near leg (higher keypoint confidence).
 6. Pose quality: a jitter statistic calibrated against the evaluation noise levels gives a
    sigma-equivalent; each output gets the validated recording-level error at that sigma.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

import augment
from analysis import ROOT
from evaluate import load_model, run_model
from gaitdata import (ANG_NAMES, EVENTS, FS, IN_JOINTS, STRIDE_COLS, TARGET_NAMES, TI, aggregate, load_all,
                      strides_from_events)
from model import decode_events

RUN = "bigru_v4"
W = 150
EVAL_TRIAL = ROOT / "outputs" / "eval" / "test_trial" / "regression.csv"
CACHE = ROOT / "outputs" / "models" / RUN
VIDEO_SIGMA_FLOOR = 0.02     # evaluation level treated as "realistic video pose"
LEG_PER_HEIGHT = 0.49        # hip-joint-centre -> ankle-joint-centre / standing height (Winter)

# canonical joint <- Halpe-26 keypoint(s)
HALPE_MAP = {"hip": ["hip"], "knee": ["knee"], "ankle": ["ankle"], "heel": ["heel"], "toe": ["big_toe", "small_toe"]}
SIDE_IDX = {s: [i for i, j in enumerate(IN_JOINTS) if j[0] == s] for s in "LR"}
SWAP = augment.JOINT_SWAP
DISTAL = [i for i, j in enumerate(IN_JOINTS) if not j.endswith("hip")]

# outputs shown in the report: (key, label, unit, per-side stride column or None)
REPORT = [
    ("cadence", "Cadence", "steps/min", None),
    ("speed_ms", "Walking speed", "m/s", None),
    ("stride_time", "Stride time", "s", "stride_time"),
    ("step_time", "Step time", "s", "step_time"),
    ("stance_time", "Stance time", "s", "stance_time"),
    ("swing_time", "Swing time", "s", "swing_time"),
    ("stance_pct", "Stance", "% of stride", "stance_pct"),
    ("double_support", "Double support", "s", "double_support"),
    ("SI_step_time", "Step-time symmetry (SI)", "%", None),
    ("SI_stance_time", "Stance-time symmetry (SI)", "%", None),
    ("SI_swing_time", "Swing-time symmetry (SI)", "%", None),
    ("hip_rom", "Hip range of motion", "deg", "hip_rom"),
    ("knee_rom", "Knee range of motion", "deg", "knee_rom"),
    ("ankle_rom", "Ankle range of motion", "deg", "ankle_rom"),
    ("peak_knee_flex", "Peak knee flexion", "deg", "peak_knee_flex"),
    ("SI_hip_rom", "Hip ROM symmetry (SI)", "%", None),
    ("SI_knee_rom", "Knee ROM symmetry (SI)", "%", None),
    ("SI_ankle_rom", "Ankle ROM symmetry (SI)", "%", None),
    ("stride_time_cv", "Stride-time variability (CV)", "%", None),
]


# ---------------------------------------------------------------- keypoints -> canonical
def halpe_to_canonical(kp: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[T, 26, 2] pixels, [T, 26] -> [T, 10, 2] (x right, y UP, pixels), [T, 10] scores."""
    from pose_video import H
    x = np.zeros((len(kp), len(IN_JOINTS), 2), np.float32)
    s = np.zeros((len(kp), len(IN_JOINTS)), np.float32)
    for j, name in enumerate(IN_JOINTS):
        src = [H[f"{name[0]}_{h}"] for h in HALPE_MAP[name[2:]]]
        x[:, j] = np.nanmean(kp[:, src], 1)
        s[:, j] = score[:, src].min(1)
    x[..., 1] *= -1
    s[~np.isfinite(x).all(-1)] = 0
    return np.nan_to_num(x), s


def resample_30(x: np.ndarray, s: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    if abs(fps - FS) < 0.5:
        return x, s
    t = np.arange(len(x)) / fps
    tn = np.arange(0, t[-1] + 1e-9, 1 / FS)
    xi = np.stack([[np.interp(tn, t, x[:, j, c]) for c in range(2)] for j in range(x.shape[1])]).transpose(2, 0, 1)
    si = np.stack([np.interp(tn, t, s[:, j]) for j in range(x.shape[1])], 1)
    return xi.astype(np.float32), si.astype(np.float32)


def fix_left_right(x: np.ndarray, vis: np.ndarray, penalty: float = 0.5) -> np.ndarray:
    """Per-frame bool: True where the detector's left/right legs should be swapped.

    Viterbi over (previous state, state) pairs; cost = distance of each joint from its
    constant-velocity prediction (both legs cross twice per stride, where plain proximity is
    ambiguous but the velocities are opposite). A small cost on 'swapped' breaks the global tie;
    `penalty` per switch (0.5: on held-out mocap with injected 3-20 frame swaps, 100 % of swapped
    frames restored and 0 of 12 000 clean frames flipped; lower values over-correct real video).
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


def _runs(lab: np.ndarray):
    """Contiguous runs of equal values -> [(start, end, value)]."""
    edges = np.flatnonzero(np.diff(lab)) + 1
    starts, ends = np.r_[0, edges], np.r_[edges, len(lab)]
    return [(int(s), int(e), lab[s]) for s, e in zip(starts, ends)]


def split_passes(x: np.ndarray, vis: np.ndarray, min_len: int = 45, max_gap: int = 15):
    """Facing direction per frame from the feet (toe ahead of heel) -> [(start, end, +1/-1)].

    Frames without a visible hip and ankle, and turns (direction changes), end a pass; gaps of
    up to `max_gap` frames inside a pass are bridged. Passes shorter than `min_len` are dropped.
    """
    fx = np.zeros(len(x))
    for s in "LR":
        h, t = IN_JOINTS.index(f"{s}_heel"), IN_JOINTS.index(f"{s}_toe")
        fx += vis[:, h] * vis[:, t] * np.sign(x[:, t, 0] - x[:, h, 0])
    hips = vis[:, [IN_JOINTS.index("L_hip"), IN_JOINTS.index("R_hip")]].sum(1) > 0
    ankles = vis[:, [IN_JOINTS.index("L_ankle"), IN_JOINTS.index("R_ankle")]].sum(1) > 0
    valid = hips & ankles
    lab = np.where(valid, np.sign(median_filter(np.where(valid, fx, 0.0), size=31, mode="nearest")), 0).astype(int)
    runs = _runs(lab)
    for i in range(1, len(runs) - 1):                    # bridge short gaps inside a pass
        s, e, v = runs[i]
        if v == 0 and e - s <= max_gap and runs[i - 1][2] == runs[i + 1][2] != 0:
            lab[s:e] = runs[i - 1][2]
    return [(s, e, int(v)) for s, e, v in _runs(lab) if v != 0 and e - s >= min_len]


def normalise_pass(x: np.ndarray, vis: np.ndarray, direction: int):
    """Pixels (y up) -> pelvis-centred, leg-length-scaled, walking towards +x. Returns (x, vis, leg px)."""
    x = x.copy()
    x[..., 0] *= direction
    hips = [IN_JOINTS.index("L_hip"), IN_JOINTS.index("R_hip")]
    hp = np.where(vis[:, hips, None] > 0, x[:, hips], np.nan)
    pelvis = np.nanmean(hp, 1)
    t = np.arange(len(x))
    for c in range(2):                                   # interpolate frames with no visible hip
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


# ---------------------------------------------------------------- calibration tables
def _jitter(x: np.ndarray, vis: np.ndarray) -> float:
    """Median |second difference| / 2 of the distal joints, leg lengths (needs 3 visible frames)."""
    d = x[1:-1] - 0.5 * (x[:-2] + x[2:])
    ok = (vis[1:-1] * vis[:-2] * vis[2:]) > 0
    v = np.linalg.norm(d, axis=-1)[:, DISTAL][ok[:, DISTAL]]
    return float(np.median(v)) if len(v) else np.nan


def quality_calibration() -> dict:
    """Jitter statistic vs evaluation noise level, measured on held-out test trials (cached)."""
    p = CACHE / "quality_calib.json"
    if p.exists():
        return json.loads(p.read_text())
    rng = np.random.default_rng(0)
    trials = [t for t in load_all()["test"] if len(t.pose) >= W][:120]
    levels = [0, 0.005, 0.01, 0.02, 0.04, 0.08]
    J = []
    for lev in levels:
        J.append(float(np.median([_jitter(*augment.apply(t.pose[:W], rng, augment.Noise.level(lev)))
                                  for t in trials])))
    out = {"levels": levels, "jitter": J}
    p.write_text(json.dumps(out))
    return out


def reference_ranges() -> pd.DataFrame:
    """5th-95th percentile of each output over Fukuchi healthy adults' trials (cached)."""
    p = CACHE / "reference_ranges.csv"
    if p.exists():
        return pd.read_csv(p, index_col=0)
    rows = []
    for split in load_all().values():
        for t in split:
            if t.ds == "fukuchi":
                y = aggregate(t.strides, 0, (len(t.pose) - 1) / FS, t.speed_ll)
                rows.append(dict(zip(TARGET_NAMES, y)) | {"speed_ms": t.speed_ll * t.leg_length})
    df = pd.DataFrame(rows)
    out = pd.DataFrame({"p5": df.quantile(0.05), "p95": df.quantile(0.95)})
    out.to_csv(p)
    return out


def expected_errors() -> pd.DataFrame:
    """Recording-level MAE of the model and of the mean predictor per target and noise level."""
    R = pd.read_csv(EVAL_TRIAL)
    R = R[R.method.isin([f"{RUN}+derived", "mean"])]
    return R.groupby(["method", "target", "level"]).mae.mean().unstack("method")


# ---------------------------------------------------------------- report
@dataclass
class Report:
    summary: pd.DataFrame                 # one row per output
    strides: pd.DataFrame                 # one row per stride
    events: dict[str, np.ndarray]         # seconds, clip time (30 Hz timeline)
    angles: np.ndarray                    # [T, 6] model angle curves (deg), NaN outside passes
    t0_frames: int = 0                    # 30 Hz frame offset of the timeline
    passes: list = field(default_factory=list)
    quality: dict = field(default_factory=dict)
    x: np.ndarray | None = None           # [T, 10, 2] model input (for plotting)
    vis: np.ndarray | None = None
    swapped: np.ndarray | None = None     # [T] L/R swap applied (30 Hz)
    px: np.ndarray | None = None          # [T, 10, 2] image pixels after the L/R fix (30 Hz, y down)
    window_speed: np.ndarray | None = None  # [n, 2]: window centre (s), speed (leg lengths/s)
    leg_length_m: float | None = None


class Analyzer:
    def __init__(self, run: str = RUN):
        self.m, self.cfg, self.norm = load_model(run)
        self.feat = self.cfg.get("features", "v1")
        self.calib = quality_calibration()
        self.ref = reference_ranges()
        self.err = expected_errors()

    # -------------------------------------------------- model on one normalised pass
    def _run_pass(self, x: np.ndarray, vis: np.ndarray, offset: int):
        """50 %-overlapping 5 s windows; per-frame event probabilities and angles are blended with a
        taper (a window's edges, where an event peak may be cut off, count least), then events and
        strides are decoded once over the whole pass, so no stride is lost at a window boundary."""
        T = len(x)
        if T <= W:
            wins = [(0, T)]
        else:
            wins = [(s, s + W) for s in range(0, T - W + 1, W // 2)]
            if wins[-1][1] < T:
                wins.append((T - W, T))
        probs, ys, angs = run_model(self.m, self.norm, [x[a:b] for a, b in wins], [vis[a:b] for a, b in wins],
                                    self.feat)
        prob, ang, wsum = np.zeros((T, len(EVENTS))), np.zeros((T, 6)), np.zeros((T, 1))
        for (a, b), p, an in zip(wins, probs, angs):
            n = b - a
            w = np.clip(np.sin(np.pi * (np.arange(n) + 0.5) / n), 0.02, 1)[:, None]
            prob[a:b] += w * p
            ang[a:b] += w * an
            wsum[a:b] += w
        prob, ang = prob / wsum, ang / wsum
        ev = decode_events(prob)
        S = strides_from_events(ev, ang)
        S[:, :2] += offset / FS
        ev = {k: v + offset / FS for k, v in ev.items()}
        wspeed = [((a + b) / 2 / FS + offset / FS, float(y[TI["speed_ll"]])) for (a, b), y in zip(wins, ys)]
        speed = float(np.median([v for _, v in wspeed]))
        return S, ev, ang, speed, wspeed

    def from_canonical(self, x: np.ndarray, vis: np.ndarray, passes, leg_px: list[float] | None = None,
                       leg_length_m: float | None = None, near: list[str] | None = None,
                       sigma_floor: float = 0.0) -> Report:
        """x [T, 10, 2] already normalised per pass (pelvis-centred, leg lengths, +x forward)."""
        T = len(x)
        allS, allev, ang = [], {k: [] for k in EVENTS}, np.full((T, 6), np.nan)
        speeds, kept, wspeeds = [], [], []
        dirs = {d for _, _, d in passes}
        for k, (a, b, d) in enumerate(passes):
            S, ev, an, sp, ws = self._run_pass(x[a:b], vis[a:b], a)
            wspeeds += ws
            if len(dirs) > 1 and near is not None:     # both directions filmed: near leg only
                S = S[S[:, 2] == "LR".index(near[k])]
            allS.append(S)
            for e in EVENTS:
                allev[e].append(ev[e])
            ang[a:b] = an
            speeds.append((sp, b - a))
            kept.append((a, b, d, near[k] if near else None))
        S = _dedupe_strides(np.concatenate(allS)) if allS else np.zeros((0, 3 + len(STRIDE_COLS)))
        ev = {e: np.sort(np.concatenate(v)) if v else np.zeros(0) for e, v in allev.items()}
        speed_ll = float(np.average([s for s, _ in speeds], weights=[n for _, n in speeds])) if speeds else np.nan
        y = aggregate(S, -np.inf, np.inf, speed_ll)
        jit = _jitter(x, vis)
        sigma_jit = float(np.interp(jit, self.calib["jitter"], self.calib["levels"]))
        # real pose-estimator error is largely temporally correlated (keypoints consistently off the
        # joint centre), which jitter cannot see: for video, assume at least the "realistic video" level
        sigma = max(sigma_jit, sigma_floor)
        hs = sorted([(t, "L") for t in ev["HS_L"]] + [(t, "R") for t in ev["HS_R"]])
        breaks = sum(a[1] == b[1] and b[0] - a[0] < 2.0 for a, b in zip(hs, hs[1:]))
        quality = {"hs_alternation_breaks": int(breaks), "jitter": jit, "sigma_equiv": sigma, "sigma_jitter": sigma_jit, "sigma_floor": sigma_floor, "n_strides_L": int((S[:, 2] == 0).sum()),
                   "n_strides_R": int((S[:, 2] == 1).sum()), "visible_frac": float((vis > 0).mean()),
                   "both_directions": len(dirs) > 1, "speed_ll": speed_ll}
        summary = self._summary(y, S, speed_ll, leg_length_m, sigma)
        rep = Report(summary, _stride_df(S), ev, ang, 0, kept, quality, x, vis)
        rep.window_speed = np.array(wspeeds).reshape(-1, 2)
        rep.leg_length_m = leg_length_m
        return rep

    def from_keypoints(self, kp: np.ndarray, score: np.ndarray, fps: float, leg_length_m: float | None = None,
                       conf: float = 0.3) -> Report:
        x, s = halpe_to_canonical(kp, score)
        x, s = resample_30(x, s, fps)
        vis = (s > conf).astype(np.float32)
        swapped = fix_left_right(x, vis)
        x[swapped], vis[swapped], s[swapped] = x[swapped][:, SWAP], vis[swapped][:, SWAP], s[swapped][:, SWAP]
        passes = split_passes(x, vis)
        if not passes:
            raise ValueError("no walking pass of at least 1.5 s with a visible, side-on person was found")
        xn = np.zeros_like(x)
        near, leg_px = [], []
        for a, b, d in passes:
            xn[a:b], vis[a:b], L = normalise_pass(x[a:b], vis[a:b], d)
            leg_px.append(L)
            m = {sd: s[a:b][:, [j for j in SIDE_IDX[sd] if j in DISTAL]].mean() for sd in "LR"}
            near.append(max(m, key=m.get))
        rep = self.from_canonical(xn, vis, passes, leg_px, leg_length_m, near, sigma_floor=VIDEO_SIGMA_FLOOR)
        rep.swapped = swapped
        rep.px = x * np.array([1, -1], np.float32)
        rep.quality |= {"passes": len(passes), "fps_in": fps, "lr_swaps_fixed": int((np.diff(np.r_[0, swapped.astype(int)]) == 1).sum())}
        return rep

    # -------------------------------------------------- table
    def _summary(self, y, S, speed_ll, leg_length_m, sigma) -> pd.DataFrame:
        rows = []
        lev = np.array(sorted(self.err.index.get_level_values("level").unique()))
        for key, label, unit, col in REPORT:
            if key == "speed_ms":
                val = speed_ll * leg_length_m if leg_length_m else speed_ll
                if not leg_length_m:
                    unit = "leg lengths/s"
            else:
                val = y[TI[key]]
            side = {}
            if col is not None:
                for si, sd in enumerate("LR"):
                    v = S[S[:, 2] == si, 3 + STRIDE_COLS.index(col)]
                    side[sd] = float(v.mean()) if len(v) else np.nan
            e = self.err.loc[key] if key in self.err.index.get_level_values("target") and unit != "leg lengths/s" else None
            if e is not None:
                mae = float(np.interp(sigma, lev, e[f"{RUN}+derived"].reindex(lev).to_numpy()))
                base = float(np.interp(sigma, lev, e["mean"].reindex(lev).to_numpy()))
                rel = "good" if mae < 0.5 * base else "rough" if mae < 0.85 * base else "unreliable"
            else:
                mae, rel = np.nan, ""
            ref = self.ref.loc[key] if key in self.ref.index and unit != "leg lengths/s" else None
            rows.append(dict(key=key, measure=label, value=val, unit=unit, left=side.get("L", np.nan),
                             right=side.get("R", np.nan), expected_error=mae, reliability=rel,
                             healthy_p5=ref.p5 if ref is not None else np.nan,
                             healthy_p95=ref.p95 if ref is not None else np.nan))
        return pd.DataFrame(rows)


def _dedupe(t: np.ndarray, tol: float = 0.15) -> np.ndarray:
    t = np.sort(t)
    keep = np.r_[True, np.diff(t) > tol] if len(t) else np.zeros(0, bool)
    return t[keep]


def _dedupe_strides(S: np.ndarray, tol: float = 0.15) -> np.ndarray:
    S = S[np.lexsort((S[:, 0], S[:, 2]))]
    keep = np.ones(len(S), bool)
    for i in range(1, len(S)):
        keep[i] = not (S[i, 2] == S[i - 1, 2] and abs(S[i, 0] - S[i - 1, 0]) < tol)
    return S[keep][np.argsort(S[keep][:, 0])]


def _stride_df(S: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(S, columns=["start_s", "end_s", "side"] + STRIDE_COLS)
    df["side"] = df.side.map({0.0: "L", 1.0: "R"})
    return df
