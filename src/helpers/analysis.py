"""Shared paths, plot palette and style, trial tables and gait-cycle curves (used by src/ and the notebook)."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers.canonical import SIDES, lowpass, sagittal_angles, sagittal_project

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
DATASETS = ["fukuchi", "ferber", "toronto"]

# validated reference palette (dataviz skill) — categorical slots 1-3 are all-pairs safe
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SIDE_COL = {"L": SERIES[0], "R": SERIES[1]}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "lines.linewidth": 2, "lines.solid_capstyle": "round",
        "legend.frameon": False, "axes.prop_cycle": mpl.cycler(color=SERIES),
    })
    FIG.mkdir(parents=True, exist_ok=True)


def trials(ds: str, kept_only=False) -> pd.DataFrame:
    t = pd.read_parquet(PROC / ds / "trials.parquet")
    t["dataset"] = ds
    return t[t.keep] if kept_only else t


def all_trials(kept_only=True) -> pd.DataFrame:
    return pd.concat([trials(d, kept_only) for d in DATASETS if (PROC / d).exists()], ignore_index=True)


def load_npz(ds: str, trial: str) -> dict:
    with np.load(PROC / ds / "npz" / f"{trial}.npz") as z:
        return {k: z[k] for k in z.files}


def native_sag(z: dict) -> np.ndarray:
    """Re-derive filtered sagittal coords (m) at native rate from the stored lab-frame xyz."""
    xyz = z["xyz"].astype(float)
    fs = float(z["fs"])
    xyz_f = lowpass(xyz.reshape(len(xyz), -1), fs).reshape(xyz.shape)
    return sagittal_project(xyz_f, z["fwd"], z["up"])


def cycle_curves(z: dict, n=101) -> dict[str, np.ndarray]:
    """Time-normalised (0-100 % gait cycle) sagittal angles: {f'{side}_{joint}': [n_strides, n]}."""
    sag = native_sag(z)
    ang = sagittal_angles(sag)
    pref = "gold" if "gold_HS_L" in z else "ev"
    out = {}
    for s in SIDES:
        hs = z[f"{pref}_HS_{s}"]
        for j in ("hip", "knee", "ankle"):
            a = ang[f"{s}_{j}"]
            cyc = [np.interp(np.linspace(0, 1, n), np.linspace(0, 1, b - a0 + 1), a[a0:b + 1])
                   for a0, b in zip(hs[:-1], hs[1:]) if 0.6 < (b - a0) / z["fs"] < 2.5]
            out[f"{s}_{j}"] = np.array(cyc)
    return out
