"""Model data: subject-level splits, per-trial arrays, X-second windows and their labels.

Input  = canonical 2D pose at 30 Hz (pelvis-centred, leg-length-scaled), pelvis dropped
         because it is the origin. Labels are recomputed *per window* from the gold events
         (force plates where available, else Zeni) and mocap angles, so a 5 s window is
         labelled with exactly the strides it contains.

Speed is predicted in leg lengths per second (`speed_ll`): the input is scale-free, so
metres per second need the patient's leg length (which a physio can measure).
"""
from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analysis import PROC, trials as load_trials
from canonical import JOINTS, SIDES, TARGET_FS, stride_table

FS = TARGET_FS
IN_JOINTS = [j for j in JOINTS if j != "pelvis"]
IN_IDX = [JOINTS.index(j) for j in IN_JOINTS]
EVENTS = ["HS_L", "HS_R", "TO_L", "TO_R"]
ANG_NAMES = [f"{s}_{j}" for s in SIDES for j in ("hip", "knee", "ankle")]

# heads -> targets (brief: core 2, 3, 4, 6; extension 5)
TARGETS = {
    "spatiotemporal": ["cadence", "stride_time", "step_time", "stance_time", "swing_time",
                       "stance_pct", "double_support"],
    "speed": ["speed_ll"],
    "symmetry": ["SI_step_time", "SI_stance_time", "SI_swing_time",
                 "SI_hip_rom", "SI_knee_rom", "SI_ankle_rom"],
    "rom": ["hip_rom", "knee_rom", "ankle_rom", "peak_knee_flex"],
    "variability": ["stride_time_sd", "stride_time_cv"],
}
TARGET_NAMES = [t for v in TARGETS.values() for t in v]
TI = {n: i for i, n in enumerate(TARGET_NAMES)}
STRIDE_COLS = ["stride_time", "step_time", "stance_time", "swing_time", "stance_pct",
               "double_support", "hip_rom", "knee_rom", "ankle_rom", "peak_knee_flex"]
SI_OF = {"SI_step_time": "step_time", "SI_stance_time": "stance_time", "SI_swing_time": "swing_time",
         "SI_hip_rom": "hip_rom", "SI_knee_rom": "knee_rom", "SI_ankle_rom": "ankle_rom"}
SPLITS = PROC / "splits.parquet"


# ---------------------------------------------------------------- splits
def make_splits(seed: int = 0, frac=(0.70, 0.15, 0.15)) -> pd.DataFrame:
    """Subject-level split per dataset (no subject in two splits).

    Toronto (11 subjects, older adults, IMU-derived) is held out entirely as an external
    test set. Fukuchi is stratified by age group so both groups reach every split.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for ds in ("fukuchi", "ferber"):
        t = load_trials(ds, kept_only=True)
        subj = t.groupby("subject").first().reset_index()
        strata = subj.age_group if ds == "fukuchi" else pd.Series("all", index=subj.index)
        for _, g in subj.groupby(strata):
            ids = rng.permutation(g.subject.to_numpy())
            n_tr, n_va = round(frac[0] * len(ids)), round(frac[1] * len(ids))
            for i, s in enumerate(ids):
                rows.append((ds, s, "train" if i < n_tr else "val" if i < n_tr + n_va else "test"))
    for s in load_trials("toronto", kept_only=True).subject.unique():
        rows.append(("toronto", s, "test"))
    df = pd.DataFrame(rows, columns=["dataset", "subject", "split"])
    df.to_parquet(SPLITS, index=False)
    return df


def get_splits() -> pd.DataFrame:
    return pd.read_parquet(SPLITS) if SPLITS.exists() else make_splits()


# ---------------------------------------------------------------- per-trial arrays
def zeni_offsets() -> dict[str, float]:
    """Median Zeni-minus-force-plate error (s) per event type, from Fukuchi treadmill trials.

    Subtracting it makes Zeni-labelled events (Ferber, Toronto, Fukuchi overground)
    consistent with the force-plate-labelled ones.
    """
    t = load_trials("fukuchi", kept_only=True)
    return {e: float(np.nanmedian(t[[f"zeni_err_ms_{e}_{s}" for s in SIDES]].to_numpy())) / 1000
            for e in ("HS", "TO")}


@dataclass
class Trial:
    ds: str
    trial: str
    subject: str
    mode: str
    pose: np.ndarray                 # [T, 10, 2] float32, 30 Hz
    events: dict[str, np.ndarray]    # seconds
    strides: np.ndarray              # [S, 3 + len(STRIDE_COLS)]: start_s, end_s, side(0=L), cols
    speed_ll: float
    leg_length: float
    event_source: str
    ang: np.ndarray = field(default=None, repr=False)      # [T, 6] sagittal angles (deg), ANG_NAMES
    ev_soft: np.ndarray = field(default=None, repr=False)  # [T, 4] Gaussian event targets


def strides_to_array(rows: list[dict], fs: float) -> np.ndarray:
    a = [[r["start"] / fs, r["end"] / fs, SIDES.index(r["side"])] + [r[c] for c in STRIDE_COLS]
         for r in rows]
    return np.array(a, float).reshape(-1, 3 + len(STRIDE_COLS))


def soft_events(events_s: dict, T: int, sigma: float = 1.0) -> np.ndarray:
    t = np.arange(T)[:, None]
    out = np.zeros((T, len(EVENTS)), np.float32)
    for k, name in enumerate(EVENTS):
        e = events_s[name] * FS
        e = e[(e > -3 * sigma) & (e < T + 3 * sigma)]
        if len(e):
            out[:, k] = np.exp(-0.5 * ((t - e[None]) / sigma) ** 2).max(1)
    return out


def _load(args) -> Trial | None:
    ds, row, offsets = args
    with np.load(PROC / ds / "npz" / f"{row['trial']}.npz") as z:
        fs = float(z["fs"])
        gold = "gold_HS_L" in z.files
        ev = {k: z[f"{'gold' if gold else 'ev'}_{k}"].astype(int) for k in EVENTS}
        sag30, ang30 = z["sag30"], z["ang30"]
    if not gold:
        ev = {k: v - int(round(offsets[k[:2]] * fs)) for k, v in ev.items()}
    # angles back on the native grid so stride ROM uses the native-rate event frames
    t30 = np.arange(len(ang30)) / FS
    tn = np.arange(int(t30[-1] * fs) + 1) / fs
    ang = {n: np.interp(tn, t30, ang30[:, i]) for i, n in enumerate(ANG_NAMES)}
    ev = {k: v[(v >= 0) & (v < len(tn))] for k, v in ev.items()}
    S = strides_to_array(stride_table(ev, ang, fs), fs)
    events_s = {k: v / fs for k, v in ev.items()}
    tr = Trial(ds, row["trial"], row["subject"], row["mode"], sag30[:, IN_IDX].astype(np.float32),
               events_s, S, float(row["speed"]) / float(row["leg_length_sag"]),
               float(row["leg_length_sag"]), "grf" if gold else "zeni", ang30.astype(np.float32))
    tr.ev_soft = soft_events(events_s, len(sag30))
    return tr


def load_all(workers: int = 10, cache: bool = True) -> dict[str, list[Trial]]:
    """{'train'|'val'|'test': [Trial]} for all kept trials."""
    path = PROC / "model_trials_v2.pkl"
    if cache and path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    split = get_splits().set_index(["dataset", "subject"]).split
    offsets = zeni_offsets()
    jobs = []
    for ds in ("fukuchi", "ferber", "toronto"):
        t = load_trials(ds, kept_only=True)
        for row in t[["trial", "subject", "mode", "speed", "leg_length_sag"]].to_dict("records"):
            jobs.append((ds, row, offsets))
    with ProcessPoolExecutor(workers) as ex:
        trials = list(ex.map(_load, jobs, chunksize=16))
    out = {"train": [], "val": [], "test": []}
    for tr in trials:
        out[split[(tr.ds, tr.subject)]].append(tr)
    if cache:
        with open(path, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


# ---------------------------------------------------------------- window labels
def aggregate(S: np.ndarray, t0: float, t1: float, speed_ll: float = np.nan) -> np.ndarray:
    """Labels for the strides fully inside [t0, t1] seconds. NaN = not measurable."""
    y = np.full(len(TARGET_NAMES), np.nan)
    y[TI["speed_ll"]] = speed_ll
    if len(S) == 0:
        return y
    S = S[(S[:, 0] >= t0 - 1e-6) & (S[:, 1] <= t1 + 1e-6)]
    if len(S) == 0:
        return y
    col = {n: S[:, 3 + i] for i, n in enumerate(STRIDE_COLS)}
    st = col["stride_time"]
    y[TI["cadence"]] = 120.0 / st.mean()
    for n in STRIDE_COLS:
        if n in TI:
            y[TI[n]] = col[n].mean()
    if len(S) >= 3:
        y[TI["stride_time_sd"]] = st.std(ddof=1)
        y[TI["stride_time_cv"]] = 100 * st.std(ddof=1) / st.mean()
    left, right = S[:, 2] == 0, S[:, 2] == 1
    if left.any() and right.any():
        for si, c in SI_OF.items():
            lv, rv = col[c][left].mean(), col[c][right].mean()
            y[TI[si]] = 100 * (lv - rv) / (0.5 * (lv + rv))
    return y


def strides_from_events(ev_s: dict[str, np.ndarray], ang: np.ndarray | None) -> np.ndarray:
    """Stride table from (predicted) event times in seconds and 30 Hz angle curves [T, 6].

    Same definitions as canonical.stride_table, but with sub-frame event times.
    """
    rows = []
    for si, s in enumerate(SIDES):
        o = SIDES[1 - si]
        hs, to = np.sort(ev_s[f"HS_{s}"]), np.sort(ev_s[f"TO_{s}"])
        hs_o, to_o = np.sort(ev_s[f"HS_{o}"]), np.sort(ev_s[f"TO_{o}"])
        for a, b in zip(hs[:-1], hs[1:]):
            st = b - a
            if not 0.6 < st < 2.5:
                continue
            t_off, o_hs, o_to = (e[(e > a) & (e < b)] for e in (to, hs_o, to_o))
            if len(t_off) != 1 or len(o_hs) != 1 or len(o_to) != 1:
                continue
            stance = t_off[0] - a
            row = [a, b, si, st, o_hs[0] - a, stance, st - stance, 100 * stance / st,
                   (o_to[0] - a) + (t_off[0] - o_hs[0])]
            if ang is not None:
                seg = ang[max(0, int(round(a * FS))):int(round(b * FS)) + 1, 3 * si:3 * si + 3]
                if len(seg) < 2:
                    continue
                row += list(seg.max(0) - seg.min(0)) + [seg[:, 1].max()]
            else:
                row += [np.nan] * 4
            rows.append(row)
    return np.array(rows, float).reshape(-1, 3 + len(STRIDE_COLS))


def derive(ev_s: dict[str, np.ndarray], ang: np.ndarray | None, T: int, speed_ll: float = np.nan) -> np.ndarray:
    """Window labels computed from predicted events + angle curves (same formulas as the gold labels)."""
    return aggregate(strides_from_events(ev_s, ang), 0, (T - 1) / FS, speed_ll)


def window_labels(tr: Trial, i0: int, i1: int) -> np.ndarray:
    return aggregate(tr.strides, i0 / FS, (i1 - 1) / FS, tr.speed_ll)


def tile_windows(tr: Trial, W: int) -> list[tuple[int, int]]:
    """Non-overlapping evaluation windows; a trial shorter than W is one window."""
    T = len(tr.pose)
    if T <= W:
        return [(0, T)]
    return [(s, s + W) for s in range(0, T - W + 1, W)]
