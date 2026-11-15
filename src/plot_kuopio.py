"""Figure for the Kuopio real-video validation: outputs/figures/07_kuopio.png

    ./py src/plot_kuopio.py        (after evaluate_kuopio.py)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis import INK2, MUTED, ROOT, SIDE_COL, style  # noqa: E402

K = ROOT / "outputs" / "kuopio"
FIG = ROOT / "outputs" / "figures"
COL = {"real": SIDE_COL["L"], "mocap": INK2, "sim0.02": MUTED, "sim0.04": SIDE_COL["R"]}
LAB = {"real": "real OpenPose", "mocap": "clean mocap", "sim0.02": "mocap + noise σ=0.02", "sim0.04": "mocap + noise σ=0.04"}


def main():
    style()
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    # 1. keypoint error components vs the noise model
    ks = pd.read_csv(K / "keypoint_summary.csv")
    ks = ks.groupby("joint")[["bias", "jitter_robust", "gross_frac"]].mean().reindex(["hip", "knee", "ankle", "heel", "toe"])
    x = np.arange(len(ks))
    a = ax[0, 0]
    a.bar(x - 0.2, ks.jitter_robust, 0.4, color=COL["real"], label="real: per-frame jitter (robust SD)")
    a.bar(x + 0.2, ks.bias, 0.4, color=COL["sim0.04"], label="real: constant offset per clip")
    a.axhline(0.02, color=INK2, ls="--", lw=1, label="noise model σ = 0.02 (jitter)")
    a.axhline(0.01, color=INK2, ls=":", lw=1, label="noise model offset (σ/2)")
    for i, g in enumerate(ks.gross_frac):
        a.text(i, max(ks.bias.iloc[i], ks.jitter_robust.iloc[i]) + 0.004, f"{100 * g:.0f}% gross", ha="center", fontsize=8)
    a.set_xticks(x, ks.index)
    a.set_ylabel("leg lengths")
    a.set_title("Keypoint error: real OpenPose vs mocap")
    a.legend(fontsize=7, loc="upper left")
    # 2. heel-strike timing error vs force plates
    E = pd.read_parquet(K / "events_raw.parquet")
    a = ax[0, 1]
    bins = np.arange(-120, 121, 8)
    for cond in ("mocap", "real"):
        d = E[(E.condition == cond) & (E.method == "model") & (E.event == "HS")]
        errs = np.concatenate([np.array(v, float) for v in d.err_ms])
        a.hist(errs, bins, alpha=0.6, color=COL[cond], label=f"{LAB[cond]}: MAE {np.abs(errs).mean():.0f} ms")
    a.set_xlabel("model heel strike − force plate (ms)")
    a.set_title("Heel-strike timing (model)")
    a.legend(fontsize=8)
    # 3. per-clip knee excursion
    A = pd.read_csv(K / "angles.csv")
    a = ax[0, 2]
    for m, mk in (("model", "o"), ("classic", "x")):
        d = A[(A.condition == "real") & (A.method == m) & (A.joint == "knee")]
        mae = np.abs(d.pred_exc - d.true_exc).mean()
        a.scatter(d.true_exc, d.pred_exc, s=6, marker=mk, alpha=0.35,
                  color=COL["real"] if m == "model" else MUTED, label=f"{m}: MAE {mae:.1f}°")
    lim = [30, 80]
    a.plot(lim, lim, color=INK2, lw=1)
    a.set_xlim(lim), a.set_ylim(20, 110)
    a.set_xlabel("mocap knee excursion (°)"), a.set_ylabel("from real video (°)")
    a.set_title("Knee excursion per ~1 s clip")
    a.legend(fontsize=8)
    # 4. knee ROM SI per subject
    SI = pd.read_csv(K / "rom_si.csv")
    a = ax[1, 0]
    for cond in ("sim0.02", "real"):
        d = SI[(SI.condition == cond) & (SI.method == "model") & (SI.joint == "knee")]
        r = np.corrcoef(d.true_si_all, d.pred_si_all)[0, 1]
        a.scatter(d.true_si_all, d.pred_si_all, s=18, color=COL[cond], label=f"{LAB[cond]}: r {r:.2f}")
    a.plot([-15, 15], [-15, 15], color=INK2, lw=1)
    a.set_xlabel("mocap knee-ROM SI (%)"), a.set_ylabel("predicted (%)")
    a.set_title("Knee-ROM symmetry per subject (42 subjects)")
    a.legend(fontsize=8)
    # 5. speed
    SP = pd.read_csv(K / "speed.csv")
    a = ax[1, 1]
    for m in ("classic", "model"):
        d = SP[(SP.condition == "real") & (SP.method == m)]
        mae = np.abs(d.pred_ms - d.true_ms).mean()
        a.scatter(d.true_ms, d.pred_ms, s=6, alpha=0.4, color=COL["real"] if m == "model" else MUTED,
                  label=f"{m}: MAE {mae:.2f} m/s")
    a.plot([0.9, 2.0], [0.9, 2.0], color=INK2, lw=1)
    a.set_xlim(0.9, 2.0), a.set_ylim(0.5, 2.5)
    a.set_xlabel("mocap pelvis speed (m/s)"), a.set_ylabel("from real video (m/s)")
    a.set_title("Walking speed per clip")
    a.legend(fontsize=8)
    # 6. where real video sits on the simulated-noise curve
    AS = pd.read_csv(K / "angles_summary.csv")
    a = ax[1, 2]
    conds = ["mocap", "sim0.02", "sim0.04", "real"]
    for j, joint in enumerate(("hip", "knee", "ankle")):
        v = [AS[(AS.condition == c) & (AS.method == "model") & (AS.joint == joint)].exc_mae.mean() for c in conds]
        a.plot(range(len(conds)), v, marker="o", label=joint)
    a.set_xticks(range(len(conds)), ["clean", "σ 0.02", "σ 0.04", "real video"])
    a.set_ylabel("excursion MAE (°)")
    a.set_title("Model angle error: simulated noise vs real video")
    a.legend(fontsize=8)
    fig.suptitle("Kuopio: held-out real video keypoints (OpenPose) vs motion capture and force plates", fontsize=12)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "07_kuopio.png", dpi=130)
    print(FIG / "07_kuopio.png")


if __name__ == "__main__":
    main()
