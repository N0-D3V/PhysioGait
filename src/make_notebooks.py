"""Generate the analysis notebooks in notebooks/ (then execute them with nbconvert).

    ./py src/make_notebooks.py && (cd notebooks && nix develop .. -c jupyter nbconvert --execute --inplace 0[1-5]*.ipynb)
"""
from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parents[1] / "notebooks"
DISCLAIMER = "*This project is a prototype and is not medical advice.*"

SETUP = """\
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from analysis import *
from canonical import J, JOINTS, SIDES
style()
pd.set_option("display.width", 200, "display.max_columns", 40)"""


def nb(name, cells):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": "gait", "display_name": "Python (gait)", "language": "python"}
    n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c.strip())
               for c in cells]
    nbf.write(n, NB / name)


# ============================================================================ 01
nb("01_data_overview_and_filtering.ipynb", [
f"""md:# 01 · Data overview & filtering

Three walking datasets, converted to one canonical skeleton
`[L/R hip, knee, ankle, heel, toe, pelvis(mid-hip)]` and filtered by `src/build_dataset.py`.

| Dataset | Source | Capture | Unit of analysis |
|---|---|---|---|
| Fukuchi WBDS | Figshare 5722711 (CC BY) | Optical mocap 150 Hz, treadmill (8 speeds, 30 s) + overground (~2.5 s passes), force plates | trial |
| Ferber RIC | Figshare 24255795 (CC BY) | Optical mocap 120/200 Hz, treadmill, cluster markers only | session |
| Toronto OA | Figshare c.5515953 (CC0) | Xsens IMU mocap 100 Hz, overground hallway, 2D video pose | straight pass |

{DISCLAIMER}""",
SETUP,
"""T = pd.concat([trials(d) for d in DATASETS], ignore_index=True)
T["reject_reason"] = T.reject_reason.fillna("kept")
summary = (T.groupby("dataset")
             .agg(trials=("trial", "size"), kept=("keep", "sum"),
                  subjects=("subject", "nunique"),
                  subjects_kept=("subject", lambda s: s[T.loc[s.index, "keep"]].nunique()),
                  hours_kept=("duration_s", lambda d: d[T.loc[d.index, "keep"]].sum() / 3600))
             .reindex(DATASETS))
summary.round(2)""",
"""md:## Filtering funnel
Each trial is rejected for the first rule it fails (thresholds in `build_dataset.QC`):
missing markers after ≤0.1 s gap filling (long treadmill trials are cropped to their longest clean
≥10 s stretch instead), implausible leg length, poor rigid-cluster fit (Ferber), speed outside
walking range (>2.2 m/s ⇒ likely a jog), too few valid strides, stride-time CoV > 20 % (event failure).

~40 % of Ferber sessions were recorded without any foot landmarks (no toe, no metatarsal markers).
Rather than drop them they get a **virtual foot** (`toe_source == "virtual"`, see notebook 02) —
filter on that column if you want marker-only data.""",
"""reasons = T.groupby(["dataset", "reject_reason"]).size().unstack(fill_value=0).reindex(DATASETS)
order = ["kept"] + [c for c in reasons.sum().sort_values(ascending=False).index if c != "kept"]
reasons = reasons[order]
fig, ax = plt.subplots(figsize=(9, 2.8))
left = np.zeros(len(reasons))
rej_cols = ["#c3c2b7", "#a3a29b", "#8a8984", "#6f6e69", "#52514e", "#3a3936", "#262624"]
for i, c in enumerate(order):
    v = reasons[c].to_numpy()
    ax.barh(range(len(reasons)), v, left=left, height=0.6,
            color=COL_KEEP if (COL_KEEP := "#2a78d6") and c == "kept" else rej_cols[(i - 1) % len(rej_cols)],
            edgecolor=SURFACE, linewidth=2, label=c)
    left += v
ax.set_yticks(range(len(reasons)), [LABEL[d] for d in reasons.index]); ax.invert_yaxis()
ax.set_xlabel("trials"); ax.set_title("Trials kept vs rejected, by first failed rule")
ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.3), fontsize=9)
ax.grid(axis="y", visible=False)
save(fig, "01_filter_funnel"); plt.show()
reasons""",
"""md:## Who is in the data
Population differences are the reason metrics must be reported **per dataset**.""",
"""K = all_trials()
# one row per subject (first kept trial) for demographics
S = K.sort_values("trial").groupby(["dataset", "subject"]).first().reset_index()
S["sex"] = S.sex.astype(str).str.upper().str[0].replace({"N": "?", "U": "?"})
fig, axs = plt.subplots(1, 3, figsize=(12, 3.2))
bins = np.arange(10, 101, 5)
for d in DATASETS:
    a = S.loc[S.dataset == d, "age"].dropna()
    axs[0].hist(a, bins=bins, color=COL[d], alpha=0.85, label=f"{d} (n={len(a)})", edgecolor=SURFACE, linewidth=1, density=True)
axs[0].set_title("Age (one row per subject)"); axs[0].set_ylabel("density"); axs[0].set_xlabel("years"); axs[0].legend(fontsize=8)
for d in DATASETS:
    sp = K.loc[K.dataset == d, "speed"].dropna()
    axs[1].hist(sp, bins=np.arange(0.2, 2.4, 0.05), color=COL[d], alpha=0.85, label=d, density=True,
                edgecolor=SURFACE, linewidth=0.5)
axs[1].set_title("Walking speed (per kept trial)"); axs[1].set_xlabel("m/s"); axs[1].set_ylabel("density")
sx = S.groupby(["dataset", "sex"]).size().unstack(fill_value=0).reindex(DATASETS)
sx = sx.div(sx.sum(1), axis=0) * 100
left = np.zeros(len(sx))
for c, colr in zip(["F", "M", "?"], ["#eb6834", "#2a78d6", "#c3c2b7"]):
    if c in sx:
        axs[2].barh(range(len(sx)), sx[c], left=left, color=colr, height=0.6, edgecolor=SURFACE, linewidth=2, label=c)
        left += sx[c].to_numpy()
axs[2].set_yticks(range(len(sx)), sx.index); axs[2].invert_yaxis(); axs[2].set_xlabel("% of subjects")
axs[2].set_title("Sex"); axs[2].legend(ncol=3, fontsize=8, loc="lower right"); axs[2].grid(axis="y", visible=False)
fig.tight_layout(); save(fig, "01_demographics"); plt.show()
S.groupby("dataset")[["age", "height_cm", "speed"]].describe().round(1).T""",
"""md:## Fukuchi: speed protocol
Treadmill trials span 8 speeds per subject (T01–T08 as fractions of the subject's dimensionless
comfortable speed) — the dataset that gives the model speed coverage.""",
"""F = trials("fukuchi", kept_only=True)
fig, ax = plt.subplots(figsize=(9, 3.2))
for i, (grp, g) in enumerate(F.groupby("age_group")):
    tm = g[g["mode"] == "treadmill"]
    x = tm.speed_label.str[1:].astype(int) + (i - 0.5) * 0.25
    ax.scatter(x, tm.speed, s=18, color=SERIES[i], alpha=0.7, label=f"{grp} – treadmill", edgecolor=SURFACE, linewidth=0.5)
ax.set_xticks(range(1, 9), [f"T0{i}" for i in range(1, 9)]); ax.set_ylabel("belt speed (m/s)")
ax.set_title("Fukuchi treadmill speeds by trial and age group"); ax.legend(fontsize=9)
save(fig, "01_fukuchi_speeds"); plt.show()
F.groupby(["age_group", "mode", "speed_label"]).speed.agg(["count", "mean"]).round(2).unstack(0).head(12)""",
"""md:## Ferber: injury labels (cleaned per the dataset README)
`InjSide` is the cross-check for the symmetry head. Injured = anything other than
*No injury* in `InjDefn`/`InjJoint`/`SpecInjury`; spelling variants merged
(`oa`→`osteoarthritis`, `pfps`→`patellofemoral pain syndrome`, `itbs`→`itb syndrome`, `bi-lateral`→`bilateral`).""",
"""R = trials("ferber", kept_only=True)
fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
side = R.InjSide_clean.fillna("unknown").value_counts()
axs[0].barh(side.index, side.values, color=SERIES[0], height=0.6)
axs[0].invert_yaxis(); axs[0].set_title("Injury side (kept sessions)"); axs[0].set_xlabel("sessions")
for i, v in enumerate(side.values): axs[0].text(v, i, f" {v}", va="center", color=INK2, fontsize=9)
inj = R.SpecInjury_clean.dropna().value_counts().head(12)
axs[1].barh(inj.index, inj.values, color=SERIES[0], height=0.6)
axs[1].invert_yaxis(); axs[1].set_title("Top 12 specific injuries"); axs[1].set_xlabel("sessions")
for a in axs: a.grid(axis="y", visible=False)
fig.tight_layout(); save(fig, "01_ferber_injuries"); plt.show()
pd.crosstab(R.InjJoint.fillna("?").str.title(), R.InjSide_clean.fillna("?")).sort_values("left", ascending=False).head(10)""",
"""md:## Toronto: clinical scores
Only 14 participants (11 with Xsens data), POMA ceiling effects — use for qualitative checks, not
for training a clinical-score head with any confidence.""",
"""tor = trials("toronto")
P = tor.groupby("subject").first()
P["has_xsens"] = tor.groupby("subject").keep.any()
P[["age", "sex", "height_cm", "poma_gait", "poma_balance", "bbs", "tug_s", "falls_6mo", "has_xsens"]]""",
])

# ============================================================================ 02
nb("02_canonical_skeleton_qc.ipynb", [
f"""md:# 02 · Canonical skeleton & sagittal projection QC

Checks that each loader's joint centres are anatomically sane and that the sagittal projection
(global plane: progression axis × gravity) produces comparable stick figures across datasets.

* **Fukuchi** – Harrington hip regression from ASIS/PSIS; lateral epicondyle/malleolus; heel marker; MT1–MT5 mid as toe.
* **Ferber** – static joint centres (dataset's own definitions) carried into walking by per-frame Kabsch fit of each segment cluster; toe marker; heel = posterior foot-cluster point at floor height.
* **Toronto** – Xsens segment origins; virtual heel rigidly attached to the foot.

{DISCLAIMER}""",
SETUP,
"""examples = {"fukuchi": "WBDS10walkT05", "ferber": trials("ferber", True).trial.iloc[5], "toronto": "OAW03_pass1"}
fig, axs = plt.subplots(1, 3, figsize=(14, 2.6), sharey=True, layout="constrained")
bones = [("pelvis", "{s}_hip"), ("{s}_hip", "{s}_knee"), ("{s}_knee", "{s}_ankle"),
         ("{s}_ankle", "{s}_heel"), ("{s}_heel", "{s}_toe"), ("{s}_ankle", "{s}_toe")]
for ax, (ds, tr) in zip(axs, examples.items()):
    z = load_npz(ds, tr); sag = z["sag30"]
    hs = z["ev30_HS_L"]; frames = np.linspace(hs[2], hs[3], 7).astype(int)
    for k, f in enumerate(frames):
        off = k * 0.9
        for s in SIDES:
            for a, b in bones:
                a_, b_ = J[a.format(s=s)], J[b.format(s=s)]
                ax.plot(sag[f, [a_, b_], 0] + off, sag[f, [a_, b_], 1], color=SIDE_COL[s], lw=2,
                        alpha=1 if s == "L" else 0.75)
        ax.scatter(sag[f, :, 0] + off, sag[f, :, 1], s=8, color=INK, zorder=3)
    ax.set_title(f"{ds}: {tr}"); ax.set_aspect("equal"); ax.set_xlabel("forward (leg lengths, offset per frame)")
axs[0].set_ylabel("up (leg lengths)")
axs[0].plot([], [], color=SIDE_COL["L"], label="left"); axs[0].plot([], [], color=SIDE_COL["R"], label="right")
fig.legend(loc="upper right", ncol=2, fontsize=9)
fig.suptitle("One left stride (HS→HS), 7 snapshots — pelvis-centred, leg-length-scaled, 30 Hz", x=0.01, ha="left", fontsize=11)
save(fig, "02_stick_figures"); plt.show()""",
"""md:## Keypoint trajectories in the normalised frame
What the model sees: pelvis-centred, leg-length-scaled 2D coordinates resampled to 30 Hz.""",
"""fig, axs = plt.subplots(2, 3, figsize=(13, 5.5), sharex=True)
for c, (ds, tr) in enumerate(examples.items()):
    z = load_npz(ds, tr); sag = z["sag30"]; t = np.arange(len(sag)) / 30
    m = t < 5
    for s in SIDES:
        axs[0, c].plot(t[m], sag[m, J[f"{s}_ankle"], 0], color=SIDE_COL[s], label=f"{s} ankle")
        axs[1, c].plot(t[m], sag[m, J[f"{s}_toe"], 1], color=SIDE_COL[s], label=f"{s} toe")
        for h in z[f"ev30_HS_{s}"]:
            if h / 30 < 5: axs[0, c].axvline(h / 30, color=SIDE_COL[s], lw=0.8, ls=":")
    axs[0, c].set_title(ds); axs[1, c].set_xlabel("time (s)")
axs[0, 0].set_ylabel("ankle forward (LL)"); axs[1, 0].set_ylabel("toe height (LL)")
axs[0, 0].legend(fontsize=8); axs[1, 0].legend(fontsize=8)
fig.suptitle("First 5 s · dotted lines = heel strikes used for labels", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); save(fig, "02_trajectories"); plt.show()""",
"""md:## Leg length (hip→ankle in sagittal plane) vs measured height
A wrong joint-centre mapping shows up as a broken height relationship or a dataset offset.""",
"""K = all_trials()
S = K.groupby(["dataset", "subject"]).agg(leg=("leg_length_sag", "median"), h=("height_cm", "first")).reset_index()
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
for d in DATASETS:
    g = S[S.dataset == d]
    axs[0].scatter(g.h, g.leg, s=14, color=COL[d], alpha=0.6, label=f"{d} (n={len(g)})", edgecolor=SURFACE, linewidth=0.4)
    r = (g.leg / (g.h / 100)).dropna()
    axs[1].hist(r, bins=np.arange(0.38, 0.56, 0.005), color=COL[d], alpha=0.85, density=True, label=d, edgecolor=SURFACE, linewidth=0.5)
axs[0].set_xlabel("height (cm)"); axs[0].set_ylabel("sagittal hip→ankle (m)"); axs[0].legend(fontsize=8)
axs[0].set_title("Leg length vs height (one dot per subject)")
axs[1].set_xlabel("leg length / height"); axs[1].set_title("Ratio (anthropometric norm ≈ 0.45–0.50)")
fig.tight_layout(); save(fig, "02_leg_length"); plt.show()
S.assign(ratio=S.leg / (S.h / 100)).groupby("dataset").ratio.describe().round(3)""",
"""md:## Ferber: rigid-cluster reconstruction quality
Median per-frame RMS residual of the Kabsch fit (worst segment per session). Sub-millimetre =
clusters stayed rigid; the 5 mm reject threshold catches marker swaps / cluster slippage.""",
"""R = trials("ferber")
fig, axs = plt.subplots(1, 2, figsize=(12, 3.4))
axs[0].hist(R.max_cluster_rms_mm.dropna(), bins=np.logspace(-1, 2, 50), color=SERIES[1], edgecolor=SURFACE, linewidth=0.5)
axs[0].set_xscale("log"); axs[0].axvline(5, color=INK2, ls="--", lw=1); axs[0].text(5.3, axs[0].get_ylim()[1] * 0.9, "reject > 5 mm", color=INK2, fontsize=9)
axs[0].set_xlabel("worst-segment RMS (mm)"); axs[0].set_ylabel("sessions"); axs[0].set_title("Ferber cluster-fit residual")
fs = R.fs.value_counts().sort_index()
axs[1].bar([f"{int(f)} Hz" for f in fs.index], fs.values, color=SERIES[1], width=0.5)
axs[1].set_title("Ferber sampling rates (resampled to 30 Hz downstream)"); axs[1].grid(axis="x", visible=False)
fig.tight_layout(); save(fig, "02_ferber_qc"); plt.show()
R.toe_source.value_counts(dropna=False)""",
"""md:### Virtual foot (Ferber sessions with no foot landmarks)
Heel = most posterior foot-cluster marker at floor height; toe = static ankle centre + 0.122 × height
straight ahead (subjects stood feet-forward on a template). Both ride on the foot cluster.
Validated by stripping the toe/metatarsal markers from 25 sessions that have them and re-running
the pipeline:

| label | MAE virtual vs real | r |
|---|---|---|
| cadence | 0.00 steps/min | 1.000 |
| stance % | 0.13 % | 0.997 |
| step-time SI | 0.04 % | 1.000 |
| knee ROM | 0.00° | 1.000 |
| ankle ROM | 1.2° (−1.0° bias) | 0.980 |

Only the ankle angle (which uses the toe) is materially affected.""",
"""fig, axs = plt.subplots(1, 3, figsize=(13, 3.2))
for a, c in zip(axs, ["ankle_rom", "stance_pct", "knee_rom"]):
    for i, src in enumerate(["marker", "virtual"]):
        v = R.loc[R.keep & (R.toe_source == src), c]
        a.hist(v, bins=40, density=True, color=SERIES[i], alpha=0.7, label=f"{src} (n={len(v)})", edgecolor=SURFACE, linewidth=0.3)
    a.set_title(c); a.legend(fontsize=8)
fig.suptitle("Ferber labels by foot source", x=0.01, ha="left", fontsize=11); fig.tight_layout(); save(fig, "02_ferber_virtual_foot"); plt.show()""",
"""md:## Projection plane check
For treadmill trials the progression axis is the belt axis (snapped from mean heel→toe direction);
for overground it's PCA of the pelvis path. Out-of-plane (medio-lateral) motion is discarded —
the residual ML excursion of the pelvis tells us how much a side camera would miss.""",
"""rows = []
for ds in DATASETS:
    for tr in trials(ds, True).trial.sample(min(150, len(trials(ds, True))), random_state=0):
        z = load_npz(ds, tr)
        up = z["up"]; f = z["fwd"]; m = np.cross(f, up)
        p = z["xyz"][:, J["pelvis"]].astype(float)
        ml = p @ m
        rows.append(dict(dataset=ds, trial=tr, ml_range_cm=100 * (np.percentile(ml, 97.5) - np.percentile(ml, 2.5))))
ML = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(8, 3))
for i, d in enumerate(DATASETS):
    v = ML.loc[ML.dataset == d, "ml_range_cm"]
    ax.boxplot(v, positions=[i], widths=0.5, vert=False, patch_artist=True,
               boxprops=dict(facecolor=COL[d], color=COL[d]), medianprops=dict(color=SURFACE, lw=2),
               whiskerprops=dict(color=COL[d]), capprops=dict(color=COL[d]), flierprops=dict(markeredgecolor=COL[d], markersize=3))
ax.set_yticks(range(3), DATASETS); ax.set_xlabel("pelvis medio-lateral range (cm, 95 % interval)")
ax.set_title("Out-of-plane pelvis motion discarded by the sagittal projection"); ax.grid(axis="y", visible=False)
save(fig, "02_ml_motion"); plt.show()
ML.groupby("dataset").ml_range_cm.describe().round(1)""",
])

# ============================================================================ 03
nb("03_events_and_spatiotemporal.ipynb", [
f"""md:# 03 · Gait events & spatiotemporal labels

* Fukuchi treadmill: **force-plate events** (Fy > 30 N, both belts) — gold standard (subjects 43–51
  have a dead left belt → Zeni fallback).
* Everything else: **Zeni et al. 2008** coordinate algorithm (heel / toe relative to pelvis).

This notebook quantifies Zeni vs force plates, then the label distributions (heads 2–5).

{DISCLAIMER}""",
SETUP,
"""F = trials("fukuchi", True); FT = F[F["mode"] == "treadmill"]
cols = [c for c in FT if c.startswith("zeni_err_ms_")]
E = FT[cols + ["speed", "age_group"]].melt(["speed", "age_group"], var_name="event", value_name="err").dropna()
E["event"] = E.event.str.replace("zeni_err_ms_", "")
fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
for i, ev in enumerate(["HS", "TO"]):
    v = E[E.event.str.startswith(ev)].err
    axs[0].hist(v, bins=np.arange(-130, 131, 5), color=SERIES[i], alpha=0.85, label=f"{ev}: mean {v.mean():.0f} ms, SD {v.std():.0f}",
                edgecolor=SURFACE, linewidth=0.5)
axs[0].axvline(0, color=INK2, lw=1); axs[0].set_xlabel("Zeni − force plate (ms), per-trial mean")
axs[0].set_title("Coordinate-based event timing error (Fukuchi treadmill)"); axs[0].legend(fontsize=9)
for i, ev in enumerate(["HS", "TO"]):
    g = E[E.event.str.startswith(ev)]
    axs[1].scatter(g.speed, g.err, s=10, color=SERIES[i], alpha=0.5, label=ev, edgecolor="none")
axs[1].axhline(0, color=INK2, lw=1); axs[1].set_xlabel("belt speed (m/s)"); axs[1].set_ylabel("error (ms)")
axs[1].set_title("Error vs speed"); axs[1].legend()
fig.tight_layout(); save(fig, "03_zeni_vs_grf"); plt.show()
E.groupby("event").err.describe().round(1)""",
"""md:**Take-away:** Zeni heel strike is systematically early (~35 ms) and toe-off late (~30 ms) on this
marker set — stance time is inflated by ~6 % of the cycle. The bias is consistent, so it can be
corrected with offsets learned on Fukuchi before labelling Ferber/Toronto (or the event head trained
on GRF-only labels and the bias reported).""",
"""md:## Cadence, stance and stride variability vs speed""",
"""K = all_trials()
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
for d in DATASETS:
    g = K[(K.dataset == d) & (K["mode"] != "overground")]
    kw = dict(s=10, color=COL[d], alpha=0.45, edgecolor="none", label=f"{d} (n={len(g)})")
    axs[0].scatter(g.speed, g.cadence, **kw)
    axs[1].scatter(g.speed, g.stance_pct, **kw)
    axs[2].scatter(g.speed, g.stride_time_cv, **kw)
axs[0].set_ylabel("cadence (steps/min)"); axs[1].set_ylabel("stance (% cycle)"); axs[2].set_ylabel("stride-time CoV (%)")
axs[2].set_yscale("log")
for a in axs: a.set_xlabel("speed (m/s)")
axs[0].legend(fontsize=8, markerscale=2)
axs[0].set_title("Cadence"); axs[1].set_title("Stance fraction"); axs[2].set_title("Stride-time variability")
fig.tight_layout(); save(fig, "03_spatiotemporal_vs_speed"); plt.show()
K.groupby("dataset")[["speed", "cadence", "stride_time", "stance_pct", "double_support", "stride_time_cv"]].median().round(2)""",
"""md:Note the Fukuchi stance % split: GRF-labelled trials sit near the textbook ~60 %, Zeni-labelled
(Ferber, Toronto, Fukuchi 43–51) higher — the event bias above, not a population difference.""",
"""fig, ax = plt.subplots(figsize=(8, 3.2))
grp = K[K["mode"] != "overground"].groupby(["dataset", "event_source"]).stance_pct
for i, ((d, src), v) in enumerate(grp):
    ax.boxplot(v.dropna(), positions=[i], widths=0.5, vert=False, patch_artist=True,
               boxprops=dict(facecolor=COL[d], color=COL[d]), medianprops=dict(color=SURFACE, lw=2),
               whiskerprops=dict(color=COL[d]), capprops=dict(color=COL[d]), flierprops=dict(markeredgecolor=COL[d], markersize=3))
ax.set_yticks(range(len(grp)), [f"{d} · {s}" for d, s in grp.groups]); ax.set_xlabel("stance (% cycle)")
ax.set_title("Stance fraction by label source"); ax.grid(axis="y", visible=False)
save(fig, "03_stance_by_source"); plt.show()""",
"""md:## Interlimb symmetry (head 4)
Robinson SI = 100·(L−R)/mean(L,R); 0 = symmetric. Signals of a few percent are exactly what a
physio cannot see by eye.""",
"""si_cols = ["SI_step_time", "SI_stance_time", "SI_swing_time", "SI_knee_rom", "SI_hip_rom", "SI_ankle_rom"]
fig, axs = plt.subplots(1, len(si_cols), figsize=(15, 3.2), sharey=True)
for a, c in zip(axs, si_cols):
    for d in DATASETS:
        v = K.loc[(K.dataset == d) & (K["mode"] != "overground"), c].dropna()
        a.hist(v.clip(-30, 30), bins=np.arange(-30, 31, 1.5), color=COL[d], alpha=0.75, density=True,
               edgecolor=SURFACE, linewidth=0.3, label=d)
    a.axvline(0, color=INK2, lw=1); a.set_title(c.replace("SI_", ""), fontsize=10); a.set_xlabel("SI (%)")
axs[0].set_ylabel("density"); axs[0].legend(fontsize=8)
fig.suptitle("Symmetry-index distributions (clipped to ±30 %)", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); save(fig, "03_symmetry_distributions"); plt.show()
K.groupby("dataset")[si_cols].agg(lambda s: s.abs().median()).round(2).rename(columns=lambda c: "|" + c + "| median")""",
"""md:## Ferber: does symmetry track the injured side?
If SI carries clinical signal, unilateral injuries should push SI in a consistent direction
(L−R > 0 when the right is injured and the patient offloads it, or vice versa).""",
"""R = trials("ferber", True)
R["grp"] = R.InjSide_clean.where(R.InjSide_clean.isin(["left", "right", "bilateral", "none"]), "other")
order = ["none", "left", "right", "bilateral"]
cs = ["SI_stance_time", "SI_step_time", "SI_knee_rom", "SI_hip_rom"]
fig, axs = plt.subplots(1, len(cs), figsize=(14, 3.4), sharey=True)
for a, c in zip(axs, cs):
    data = [R.loc[R.grp == g, c].dropna().clip(-25, 25) for g in order]
    bp = a.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=False)
    for i, b in enumerate(bp["boxes"]): b.set(facecolor=[GRID, SERIES[0], SERIES[1], "#c3c2b7"][i], edgecolor=INK2)
    for m in bp["medians"]: m.set(color=INK, lw=2)
    a.axvline(0, color=INK2, lw=1, ls=":"); a.set_title(c.replace("SI_", "SI "), fontsize=10); a.set_xlabel("SI (%)")
    a.set_yticks(range(1, 5), [f"{g} (n={len(x)})" for g, x in zip(order, data)])
axs[0].set_ylabel("injury side"); fig.suptitle("Ferber symmetry by self-reported injury side", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); save(fig, "03_ferber_si_by_injside"); plt.show()
from scipy.stats import mannwhitneyu
pd.DataFrame({c: {"median L-inj": R.loc[R.grp == "left", c].median(), "median R-inj": R.loc[R.grp == "right", c].median(),
                  "median none": R.loc[R.grp == "none", c].median(),
                  "p (L vs R, MWU)": mannwhitneyu(R.loc[R.grp == "left", c].dropna(), R.loc[R.grp == "right", c].dropna()).pvalue}
              for c in cs}).T.round(3)""",
"""md:## Stride-to-stride view
Per-stride table (`strides.parquet`) — the unit for per-cycle regression heads.""",
"""ST = pd.concat([strides(d) for d in DATASETS], ignore_index=True)
print(ST.groupby("dataset").size())
fig, axs = plt.subplots(1, 3, figsize=(13, 3.2))
for d in DATASETS:
    g = ST[ST.dataset == d]
    for a, c, b in zip(axs, ["stride_time", "double_support", "knee_rom"],
                       [np.arange(0.6, 2.0, 0.02), np.arange(0, 0.8, 0.01), np.arange(20, 95, 1)]):
        a.hist(g[c], bins=b, density=True, color=COL[d], alpha=0.75, label=d, edgecolor=SURFACE, linewidth=0.3)
for a, t in zip(axs, ["stride time (s)", "double support (s)", "knee ROM (°)"]): a.set_xlabel(t); a.set_title(t.split(" (")[0].capitalize())
axs[0].legend(fontsize=8); fig.tight_layout(); save(fig, "03_per_stride"); plt.show()""",
])

# ============================================================================ 04
nb("04_joint_kinematics.ipynb", [
f"""md:# 04 · Joint kinematics (head 6)

2D sagittal angles from the projected canonical joints — what a side camera can measure:
hip = thigh vs vertical (no pelvis tilt available), knee = thigh − shank, ankle = foot vs
shank-perpendicular (offset depends on each dataset's heel/toe definition; ROM does not).

{DISCLAIMER}""",
SETUP,
"""def mean_curves(ds, trials_, max_trials=60, seed=0):
    out = {k: [] for k in [f"{s}_{j}" for s in SIDES for j in ("hip", "knee", "ankle")]}
    for tr in pd.Series(trials_).sample(min(max_trials, len(trials_)), random_state=seed):
        c = cycle_curves(load_npz(ds, tr))
        for k, v in c.items():
            if len(v): out[k].append(v.mean(0))
    return {k: np.array(v) for k, v in out.items()}

sel = {"fukuchi": trials("fukuchi", True).query("mode == 'treadmill' and speed_label in ['T04','T05','T06']").trial,
       "ferber": trials("ferber", True).trial,
       "toronto": trials("toronto", True).trial}
MC = {d: mean_curves(d, t) for d, t in sel.items()}
x = np.linspace(0, 100, 101)
fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
for a, j in zip(axs, ["hip", "knee", "ankle"]):
    for d in DATASETS:
        c = np.concatenate([MC[d][f"L_{j}"], MC[d][f"R_{j}"]])
        m, sd = c.mean(0), c.std(0)
        a.fill_between(x, m - sd, m + sd, color=COL[d], alpha=0.15, lw=0)
        a.plot(x, m, color=COL[d], label=f"{d} (n={len(c)} trial-sides)")
    a.set_title(j.capitalize() + (" flexion" if j != "ankle" else " dorsiflexion")); a.set_xlabel("% gait cycle (HS→HS)")
axs[0].set_ylabel("degrees"); axs[0].legend(fontsize=8)
fig.suptitle("Mean ± SD sagittal angles (Fukuchi: treadmill ~comfortable speeds)", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); save(fig, "04_mean_cycles"); plt.show()""",
"""md:## Validation: our 2D knee/hip vs Fukuchi's 3D Visual3D angles
The dataset ships subject-mean gait-cycle angles (`*ang.txt`, 101 points, Z = flexion axis).
If the canonical mapping + projection is right, our curves should overlay theirs (modulo offset).""",
"""import load_fukuchi as LF
FT = trials("fukuchi", True).query("mode == 'treadmill'")
rows, ex = [], None
for tr in FT.trial.sample(80, random_state=1):
    ref = pd.read_csv(LF.ASCII / f"{tr}ang.txt", sep="\\t")
    ours = cycle_curves(load_npz("fukuchi", tr))
    for s in SIDES:
        for j, col in [("knee", "KneeAngleZ"), ("hip", "HipAngleZ"), ("ankle", "AnkleAngleZ")]:
            r = ref[f"{s}{col}"].to_numpy(); o = ours[f"{s}_{j}"].mean(0)
            rows.append(dict(trial=tr, side=s, joint=j, r=np.corrcoef(r, o)[0, 1],
                             rmse_centered=np.sqrt(np.mean(((o - o.mean()) - (r - r.mean())) ** 2)),
                             rom_ref=np.ptp(r), rom_ours=np.ptp(o)))
            if ex is None and j == "knee": ex = (tr, s)
V = pd.DataFrame(rows)
tr, s = ex
ref = pd.read_csv(LF.ASCII / f"{tr}ang.txt", sep="\\t"); ours = cycle_curves(load_npz("fukuchi", tr))
fig, axs = plt.subplots(1, 4, figsize=(15, 3.4))
for a, (j, col) in zip(axs, [("hip", "HipAngleZ"), ("knee", "KneeAngleZ"), ("ankle", "AnkleAngleZ")]):
    a.plot(x, ref[f"{s}{col}"], color=INK2, ls="--", label="Visual3D 3D (ref)")
    a.plot(x, ours[f"{s}_{j}"].mean(0), color=SERIES[0], label="ours 2D sagittal")
    a.set_title(f"{j} · {tr} {s}"); a.set_xlabel("% cycle")
axs[0].legend(fontsize=8)
for j, mk in zip(["hip", "knee", "ankle"], ["o", "s", "^"]):
    g = V[V.joint == j]
    axs[3].scatter(g.rom_ref, g.rom_ours, s=14, alpha=0.6, marker=mk, label=j, edgecolor="none")
lim = [0, max(V.rom_ref.max(), V.rom_ours.max()) + 5]
axs[3].plot(lim, lim, color=INK2, lw=1, ls=":"); axs[3].set_xlabel("ROM ref (°)"); axs[3].set_ylabel("ROM ours (°)")
axs[3].set_title("ROM agreement (80 trials × 2 sides)"); axs[3].legend(fontsize=8)
fig.tight_layout(); save(fig, "04_validation_vs_visual3d"); plt.show()
V.groupby("joint")[["r", "rmse_centered", "rom_ref", "rom_ours"]].median().round(2)""",
"""md:## Knee angle vs walking speed (Fukuchi, gold events)""",
"""fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
sp = FT.groupby("speed_label").speed.mean()
cm = SEQ_CMAP
for i, lab in enumerate(sorted(FT.speed_label.unique())):
    trs = FT[FT.speed_label == lab].trial
    c = mean_curves("fukuchi", trs, max_trials=25)
    k = np.concatenate([c["L_knee"], c["R_knee"]]).mean(0)
    axs[0].plot(x, k, color=cm(0.2 + 0.8 * i / 7), label=f"{lab} ({sp[lab]:.2f} m/s)")
axs[0].set_xlabel("% gait cycle"); axs[0].set_ylabel("knee flexion (°)"); axs[0].set_title("Knee flexion by treadmill speed")
axs[0].legend(fontsize=7, ncol=2)
axs[1].scatter(FT.speed, FT.knee_rom, s=10, c=FT.speed, cmap=cm, alpha=0.7, edgecolor="none")
axs[1].set_xlabel("speed (m/s)"); axs[1].set_ylabel("knee ROM (°)"); axs[1].set_title("Knee ROM rises with speed")
fig.tight_layout(); save(fig, "04_knee_by_speed"); plt.show()""",
"""md:## ROM distributions per dataset & subgroup check by age and sex""",
"""K = all_trials()
K = K[K["mode"] != "overground"].copy()
K["sex1"] = K.sex.astype(str).str.upper().str[0]
K["age_band"] = pd.cut(K.age, [0, 30, 50, 65, 120], labels=["<30", "30–50", "50–65", "65+"])
fig, axs = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)
for a, c in zip(axs, ["hip_rom", "knee_rom", "ankle_rom"]):
    g = K.groupby(["dataset", "age_band"], observed=True)[c]
    labels, data, colors = [], [], []
    for (d, ab), v in g:
        labels.append(f"{d} {ab}"); data.append(v.dropna()); colors.append(COL[d])
    bp = a.boxplot(data, vert=False, widths=0.6, patch_artist=True, showfliers=False)
    for b, col in zip(bp["boxes"], colors): b.set(facecolor=col, color=col)
    for m in bp["medians"]: m.set(color=SURFACE, lw=2)
    a.set_yticks(range(1, len(labels) + 1), labels); a.set_title(c.replace("_", " ").upper()); a.set_xlabel("degrees")
    a.grid(axis="y", visible=False)
fig.tight_layout(); save(fig, "04_rom_by_age"); plt.show()
K.groupby(["dataset", "sex1"])[["hip_rom", "knee_rom", "ankle_rom", "peak_knee_flex"]].median().round(1)""",
])

# ============================================================================ 05
nb("05_toronto_pose_and_clinical.ipynb", [
f"""md:# 05 · Toronto: video-pose quality & clinical scores

**Important finding for the project plan:** the Toronto videos are filmed walking *towards / away
from* the camera (frontal plane), and the released 2D pose has **no foot keypoints**
(COCO-17 lower body = hip/knee/ankle). It cannot serve directly as the sagittal video→pose bridge;
it is still useful as (a) a frontal-view pose-noise reference, and (b) Xsens-derived sagittal labels
with clinical scores (n = 11).

{DISCLAIMER}""",
SETUP,
"""import load_toronto as LT
PC = pd.read_parquet(PROC / "toronto" / "pose_clips.parquet")
fig, axs = plt.subplots(1, 3, figsize=(14, 3.4))
est = ["OpenPose", "Alphapose", "Detectron"]
for i, e in enumerate(est):
    g = PC[PC.estimator == e]
    axs[0].hist(g.mean_conf, bins=np.linspace(0, 1, 30), color=SERIES[i], alpha=0.8, label=e, edgecolor=SURFACE, linewidth=0.4)
    axs[1].hist(g.duration_s, bins=np.arange(0, 15, 0.5), color=SERIES[i], alpha=0.8, label=e, edgecolor=SURFACE, linewidth=0.4)
axs[0].set_title("Mean lower-body keypoint confidence per clip"); axs[0].legend(fontsize=8)
axs[1].set_title("Clip duration (s)")
ct = PC.groupby(["camera", "direction"]).size()
axs[2].bar([f"{c}-{d}" for c, d in ct.index], ct.values, color=SERIES[0], width=0.5)
axs[2].set_title("Clips by camera / walking direction"); axs[2].grid(axis="x", visible=False)
fig.tight_layout(); save(fig, "05_pose_quality"); plt.show()
PC.groupby("estimator")[["n_frames", "duration_s", "mean_conf", "frac_lowconf"]].mean().round(3)""",
"""md:### One clip, three estimators (frontal view, image coordinates)""",
"""clips = LT.pose_clips()
key = ["pid", "camera", "direction", "clip"]
common = clips.groupby(key).estimator.nunique().loc[lambda n: n == 3].index[0]   # a clip all 3 estimators processed
pick = clips.set_index(key).loc[common].reset_index().sort_values("estimator", key=lambda e: e.map({"OpenPose": 0, "Alphapose": 1, "Detectron": 2}))
print("clip:", dict(zip(key, common)))
fig, axs = plt.subplots(1, 2, figsize=(13, 3.6))
for i, r in enumerate(pick.itertuples()):
    p = LT.load_pose(r.path)
    a = p["xy"][:, LT.POSE_JOINTS.index("LAnkle"), 1]
    axs[0].plot(p["t"], a, color=SERIES[i], label=r.estimator, lw=1.5)
    axs[1].plot(p["t"], np.nanmean(p["conf"], 1), color=SERIES[i], label=r.estimator, lw=1.5)
axs[0].invert_yaxis(); axs[0].set_title("Left ankle vertical pixel position (towards camera)"); axs[0].set_xlabel("s"); axs[0].set_ylabel("px (down)")
axs[1].set_title("Mean keypoint confidence (Detectron uses a different score scale)"); axs[1].set_xlabel("s"); axs[0].legend(fontsize=8)
fig.tight_layout(); save(fig, "05_estimator_compare"); plt.show()""",
"""md:## Xsens-derived gait features vs clinical scores
Per-subject means over straight passes. n = 11 with a narrow, ceiling-limited score range —
read these as sanity checks, not findings (the original paper also found variability did not
track clinical scores).""",
"""tor = trials("toronto", True)
P = tor.groupby("subject").agg(speed=("speed", "mean"), cadence=("cadence", "mean"), stride_cv=("stride_time_cv", "mean"),
                               knee_rom=("knee_rom", "mean"), SI_step=("SI_step_time", lambda s: s.abs().mean()),
                               tug=("tug_s", "first"), bbs=("bbs", "first"), poma_gait=("poma_gait", "first"), age=("age", "first"))
feats = ["speed", "cadence", "stride_cv", "knee_rom", "SI_step"]
scores = ["tug", "bbs", "poma_gait"]
fig, axs = plt.subplots(len(scores), len(feats), figsize=(15, 8))
for i, sc in enumerate(scores):
    for k, f in enumerate(feats):
        a = axs[i, k]
        a.scatter(P[f], P[sc], s=40, color=SERIES[2], edgecolor=SURFACE, linewidth=1)
        r = P[[f, sc]].corr(method="spearman").iloc[0, 1]
        a.text(0.02, 0.92, f"ρ = {r:.2f}", transform=a.transAxes, color=INK2, fontsize=9)
        if i == len(scores) - 1: a.set_xlabel(f)
        if k == 0: a.set_ylabel(sc)
fig.suptitle("Toronto (n = 11): gait features vs clinical scores, Spearman ρ", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); save(fig, "05_clinical_scatter"); plt.show()
P.round(2)""",
"""md:## Straight-pass segmentation
Overground hallway walking: turns are cut out (pelvis speed > 0.4 m/s and heading change < 0.6 rad/s,
0.5 s trimmed at each end, ≥ 4 s kept).""",
"""xyz = LT.load_xsens(3); ps = LT.split_passes(xyz)
p = xyz[:, J["pelvis"]]
fig, axs = plt.subplots(1, 2, figsize=(13, 3.4), gridspec_kw=dict(width_ratios=[2, 1]))
t = np.arange(len(p)) / LT.FS
axs[0].plot(t, p[:, 0], color=MUTED, lw=1.2, label="pelvis X (all)")
for k, (s, e) in enumerate(ps):
    axs[0].plot(t[s:e], p[s:e, 0], color=SERIES[0], lw=2.2, label="kept pass" if k == 0 else None)
axs[0].set_xlabel("s"); axs[0].set_ylabel("m"); axs[0].set_title("OAW03 pelvis along hallway"); axs[0].legend(fontsize=8)
axs[1].plot(p[:, 0], p[:, 1], color=MUTED, lw=1)
for s, e in ps: axs[1].plot(p[s:e, 0], p[s:e, 1], color=SERIES[0], lw=2)
axs[1].set_aspect("equal"); axs[1].set_title("Top view"); axs[1].set_xlabel("X (m)"); axs[1].set_ylabel("Y (m)")
fig.tight_layout(); save(fig, "05_pass_segmentation"); plt.show()""",
])

print("notebooks written to", NB)
