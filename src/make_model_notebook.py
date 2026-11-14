"""Generate notebooks/06_model_results.ipynb (needs outputs/models/* and outputs/eval/test/*).

    ./py src/train.py --name bigru_aug && ./py src/train.py --name bigru_clean --no-aug
    ./py src/evaluate.py bigru_aug bigru_clean
    ./py src/make_model_notebook.py && (cd notebooks && nix develop .. -c jupyter nbconvert --execute --inplace 06_model_results.ipynb)
"""
from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parents[1] / "notebooks"
DISCLAIMER = "*This project is a prototype and is not medical advice.*"

SETUP = """\
import sys, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from analysis import *
style()
pd.set_option("display.width", 200, "display.max_columns", 40)
EV = ROOT / "outputs" / "eval" / "test"
MOD = ROOT / "outputs" / "models"
RUN, ABL = "bigru_aug", "bigru_clean"
METHODS = [RUN, ABL, "classic", "mean"]
MLABEL = {RUN: "BiGRU + pose-noise aug", ABL: "BiGRU, clean training", "classic": "Classic pipeline", "mean": "Mean predictor"}
MCOL = {RUN: SERIES[0], ABL: SERIES[1], "classic": SERIES[2], "mean": MUTED}
MSTY = {RUN: "-", ABL: "-", "classic": "-", "mean": "--"}
reg = pd.read_csv(EV / "regression.csv")
evs = pd.read_csv(EV / "events.csv")
P = pd.read_parquet(EV / "predictions.parquet")
LEVELS = sorted(reg.level.unique())
UNIT = {"cadence": "steps/min", "stance_pct": "% cycle", "speed_ms": "m/s", "SI_step_time": "SI %",
        "SI_stance_time": "SI %", "knee_rom": "deg", "ankle_rom": "deg", "hip_rom": "deg",
        "double_support": "s", "stride_time": "s", "stride_time_cv": "%", "peak_knee_flex": "deg"}"""


def nb(name, cells):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": "gait", "display_name": "Python (gait)", "language": "python"}
    n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c.strip())
               for c in cells]
    nbf.write(n, NB / name)


nb("06_model_results.ipynb", [
f"""md:# 06 · Multi-head BiGRU: results on held-out subjects

**Model** (`src/model.py`): 5 s windows of canonical 2D sagittal pose at 30 Hz (10 joints, pelvis-centred,
leg-length-scaled) + velocities + per-joint visibility → input MLP → 2-layer BiGRU (128 per direction) →
- **events head** (per frame): heel-strike / toe-off, left / right
- pooled heads: **spatiotemporal** (cadence, stride / step / stance / swing time, stance %, double support),
  **speed** (leg lengths/s → m/s with the patient's leg length), **symmetry** (Robinson SI of step / stance /
  swing time and hip / knee / ankle ROM), **ROM** (hip / knee / ankle excursion, peak knee flexion),
  **variability** (stride-time SD / CoV, extension head)

**Labels** are recomputed per window from the strides fully inside it: force-plate events for Fukuchi
treadmill, Zeni events elsewhere — shifted by the Zeni-vs-force-plate bias measured on Fukuchi so all
labels are force-plate-like.

**Split**: subject-level (`data/processed/splits.parquet`), Fukuchi stratified by age group, Ferber random;
**Toronto is held out entirely** as an external test set (different population and capture: older adults, IMU).

**Pose-noise augmentation** (`src/augment.py`): keypoint jitter, per-joint offset, dropout, far-leg
occlusion where the legs cross, leg-length scale error and camera tilt. Evaluation sweeps one noise knob
(σ in leg lengths; 0.01 ≈ 8 mm) and every method sees the identical noisy input.

**Baselines**: the mean of the training labels, and the classic pipeline (gap-fill → 6 Hz low-pass →
Zeni events with the same bias correction → stride formulas; `src/classic.py`).

{DISCLAIMER}""",
SETUP,
"""md:## Training curves
Validation loss on held-out validation subjects with fixed noise (σ = 0.02); the best epoch is kept.""",
"""heads = ["events", "spatiotemporal", "speed", "symmetry", "rom", "variability"]
fig, axs = plt.subplots(1, len(heads), figsize=(16, 2.8), sharex=True)
for r in (RUN, ABL):
    h = pd.read_csv(MOD / r / "history.csv")
    for a, k in zip(axs, heads):
        a.plot(h.epoch, h[f"val_{k}"], color=MCOL[r], label=MLABEL[r])
        a.plot(h.epoch, h[f"train_{k}"], color=MCOL[r], lw=1, ls=":")
for a, k in zip(axs, heads): a.set_title(k); a.set_xlabel("epoch")
axs[0].set_ylabel("loss (solid = val, dotted = train)")
axs[0].legend(fontsize=8)
fig.tight_layout(); save(fig, "06_training_curves"); plt.show()""",
"""md:## Headline table
Mean absolute error per dataset on held-out subjects, clean input (σ = 0) and realistic video-pose noise
(σ = 0.02). Lower is better; the best method per row is bold.""",
"""core = ["cadence", "stride_time", "stance_pct", "double_support", "speed_ms", "SI_step_time", "SI_stance_time",
        "hip_rom", "knee_rom", "ankle_rom", "peak_knee_flex", "stride_time_cv"]
T = (reg[reg.target.isin(core) & reg.level.isin([0, 0.02])]
     .pivot_table(index=["dataset", "target"], columns=["level", "method"], values="mae")
     .reindex(columns=METHODS, level=1))
T = T.reindex(pd.MultiIndex.from_product([DATASETS, core]), axis=0).dropna(how="all")
def bold_min(row):
    out = []
    for lev in row.index.get_level_values(0):
        sub = row[lev]
        out.append("font-weight: bold" if False else "")
    return out
T.style.format("{:.3g}").apply(lambda r: ["font-weight: bold" if v == r[l].min() else "" for (l, m), v in r.items()], axis=1)""",
"""md:## How each method degrades with pose noise
The brief's central claim: on clean mocap the classic pipeline is hard to beat (labels are computed by the
same formulas); the learned model earns its place by holding up as the input looks more like a video
pose estimator. The dashed line is the mean predictor: a method above it is worse than knowing nothing.""",
"""targets = ["cadence", "stance_pct", "speed_ms", "SI_step_time", "knee_rom", "ankle_rom"]
fig, axs = plt.subplots(len(DATASETS), len(targets), figsize=(17, 8.2), sharex=True)
for i, ds in enumerate(DATASETS):
    for k, t in enumerate(targets):
        a = axs[i, k]
        for m in METHODS:
            g = reg[(reg.dataset == ds) & (reg.target == t) & (reg.method == m)].sort_values("level")
            a.plot(g.level, g.mae, color=MCOL[m], ls=MSTY[m], marker="o", ms=4, label=MLABEL[m])
        a.set_ylim(bottom=0)
        if i == 0: a.set_title(t)
        if k == 0: a.set_ylabel(f"{ds}\\nMAE")
        if i == len(DATASETS) - 1: a.set_xlabel("pose noise σ (leg lengths)")
        a.text(0.98, 0.95, UNIT.get(t, ""), transform=a.transAxes, ha="right", va="top", color=MUTED, fontsize=8)
axs[0, 0].legend(fontsize=8, loc="upper left")
fig.tight_layout(); save(fig, "06_degradation"); plt.show()""",
"""md:## Gait events (head 1)
Detection F1 within ±100 ms and timing error of the detected events against the labels (force plates for
Fukuchi treadmill, bias-corrected Zeni elsewhere). Events within 0.2 s of a window edge are excluded.""",
"""E = evs[evs.method != "mean"]
fig, axs = plt.subplots(2, len(DATASETS), figsize=(15, 6), sharex=True)
for k, ds in enumerate(DATASETS):
    for m in [RUN, ABL, "classic"]:
        for ev, ls in (("HS", "-"), ("TO", ":")):
            g = E[(E.dataset == ds) & (E.method == m) & (E.event == ev)].sort_values("level")
            axs[0, k].plot(g.level, g.f1, color=MCOL[m], ls=ls, marker="o", ms=4, label=f"{MLABEL[m]} {ev}")
            axs[1, k].plot(g.level, g.mae_ms, color=MCOL[m], ls=ls, marker="o", ms=4)
    axs[0, k].set_title(ds); axs[1, k].set_xlabel("pose noise σ (leg lengths)")
    axs[1, k].set_ylim(bottom=0)
axs[0, 0].set_ylabel("F1 (±100 ms)"); axs[1, 0].set_ylabel("timing MAE (ms)")
axs[0, 0].legend(fontsize=7, ncol=2)
fig.tight_layout(); save(fig, "06_events"); plt.show()
E[E.level.isin([0, 0.02])].pivot_table(index=["dataset", "event"], columns=["level", "method"],
                                       values=["f1", "mae_ms", "bias_ms"]).round(3)""",
"""md:## Bland–Altman (model, σ = 0.02)
Agreement with the gold-standard labels per window. Lines: bias and 95 % limits of agreement (pooled
across datasets; per-dataset numbers in the table below).""",
"""ba = ["SI_step_time", "SI_stance_time", "cadence", "stance_pct", "speed_ms", "knee_rom"]
Q = P[(P.method == RUN) & (P.level == 0.02)]
fig, axs = plt.subplots(1, len(ba), figsize=(18, 3.6))
rows = []
for a, t in zip(axs, ba):
    q = Q[[f"true_{t}", f"pred_{t}", "dataset"]].dropna()
    mean, diff = (q[f"true_{t}"] + q[f"pred_{t}"]) / 2, q[f"pred_{t}"] - q[f"true_{t}"]
    for ds in DATASETS:
        s = q.dataset == ds
        a.scatter(mean[s], diff[s], s=6, alpha=0.35, color=COL[ds], label=ds, rasterized=True)
        rows.append(dict(target=t, dataset=ds, n=int(s.sum()), bias=diff[s].mean(), loa=1.96 * diff[s].std()))
    b, sd = diff.mean(), diff.std()
    for y, ls in ((b, "-"), (b + 1.96 * sd, "--"), (b - 1.96 * sd, "--")):
        a.axhline(y, color=INK2, ls=ls, lw=1)
    a.set_title(t); a.set_xlabel(f"mean of model & gold ({UNIT.get(t, '')})")
axs[0].set_ylabel("model − gold"); axs[0].legend(fontsize=8, markerscale=3)
fig.tight_layout(); save(fig, "06_bland_altman"); plt.show()
pd.DataFrame(rows).pivot(index="target", columns="dataset", values=["bias", "loa"]).round(3)""",
"""md:## Symmetry: does the model track real asymmetry or just predict ~0?
Most people are near-symmetric, so a model can score a low MAE on SI by always predicting the mean.
Correlation with the gold SI (and slope) shows whether real asymmetry gets through.""",
"""si = ["SI_step_time", "SI_stance_time", "SI_swing_time", "SI_knee_rom", "SI_hip_rom", "SI_ankle_rom"]
rows = []
for lev in (0, 0.02, 0.04):
    for m in [RUN, ABL, "classic"]:
        for ds in DATASETS:
            q = P[(P.method == m) & (P.level == lev) & (P.dataset == ds)]
            for t in si:
                d = q[[f"true_{t}", f"pred_{t}"]].dropna()
                if len(d) > 10:
                    rows.append(dict(level=lev, method=m, dataset=ds, target=t, r=d.corr().iloc[0, 1],
                                     slope=np.polyfit(d.iloc[:, 0], d.iloc[:, 1], 1)[0]))
R = pd.DataFrame(rows)
fig, axs = plt.subplots(1, len(DATASETS), figsize=(15, 3.4), sharey=True)
for a, ds in zip(axs, DATASETS):
    q = R[(R.dataset == ds) & (R.level == 0.02)]
    x = np.arange(len(si))
    for j, m in enumerate([RUN, ABL, "classic"]):
        v = q[q.method == m].set_index("target").reindex(si).r
        a.bar(x + (j - 1) * 0.27, v, width=0.25, color=MCOL[m], label=MLABEL[m])
    a.set_xticks(x, [s.replace("SI_", "") for s in si], rotation=30); a.set_title(f"{ds} (σ = 0.02)")
    a.axhline(0, color=MUTED, lw=1)
axs[0].set_ylabel("Pearson r with gold SI"); axs[0].legend(fontsize=8)
fig.tight_layout(); save(fig, "06_symmetry_r"); plt.show()
R.pivot_table(index=["dataset", "target"], columns=["level", "method"], values="r").round(2)""",
"""md:## Example: one held-out window at σ = 0.04
Noisy input (the heel's forward position relative to the pelvis, with hidden keypoints as gaps), model event
probabilities, and the gold events.""",
"""import torch, augment, gaitdata as G
from evaluate import load_model, run_model
data = G.load_all()["test"]
tr = next(t for t in data if t.ds == "fukuchi" and t.mode == "treadmill")
x, v = augment.apply(tr.pose[300:450], np.random.default_rng(3), augment.Noise.level(0.04))
m, cfg, norm = load_model(RUN)
probs, y = run_model(m, norm, [x], [v])
t = np.arange(150) / 30
fig, axs = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
for s, c in SIDE_COL.items():
    j = G.IN_JOINTS.index(f"{s}_heel")
    hx = np.where(v[:, j] > 0, x[:, j, 0], np.nan)
    axs[0].plot(t, tr.pose[300:450, j, 0], color=c, lw=1, alpha=0.5)
    axs[0].plot(t, hx, color=c, lw=0, marker="o", ms=3, label=f"{s} heel (noisy input)")
for k, name in enumerate(G.EVENTS):
    s = name[-1]
    axs[1].plot(t, probs[0][:, k], color=SIDE_COL[s], ls="-" if name.startswith("HS") else ":", label=name)
    for e in tr.events[name] - 10:
        if 0 <= e <= 5: axs[1].axvline(e, color=SIDE_COL[s], lw=0.8, ls="-" if name.startswith("HS") else ":", alpha=0.6)
axs[0].set_ylabel("forward (leg lengths)"); axs[0].legend(fontsize=8, ncol=2)
axs[0].set_title(f"{tr.trial} (held-out subject), thin line = clean")
axs[1].set_ylabel("event probability"); axs[1].set_xlabel("s"); axs[1].legend(fontsize=8, ncol=4)
fig.tight_layout(); save(fig, "06_example_window"); plt.show()
pd.Series(y[0], index=G.TARGET_NAMES).to_frame("model").assign(gold=G.window_labels(tr, 300, 450)).round(3).T""",
"""md:## Subgroups (age, sex)
Model MAE at σ = 0.02 by subgroup, from the trial metadata. Toronto (all older adults) is the external check.""",
"""meta = all_trials()[["dataset", "trial", "age", "sex"]].copy()
meta["sex"] = meta.sex.astype(str).str.upper().str[0].map({"M": "male", "F": "female"})
meta["age_band"] = pd.cut(meta.age, [0, 40, 60, 120], labels=["<40", "40–60", "≥60"])
Q = P[(P.method == RUN) & (P.level == 0.02)].merge(meta, on=["dataset", "trial"], how="left")
rows = []
for by in ("age_band", "sex"):
    for (ds, grp), g in Q.groupby(["dataset", by], observed=True):
        for t in ["cadence", "stance_pct", "speed_ms", "SI_step_time", "knee_rom"]:
            d = g[[f"true_{t}", f"pred_{t}"]].dropna()
            rows.append(dict(group=by, dataset=ds, level=str(grp), target=t, n=len(d),
                             mae=(d.iloc[:, 1] - d.iloc[:, 0]).abs().mean()))
S = pd.DataFrame(rows)
S.pivot_table(index=["group", "dataset", "level"], columns="target", values="mae").round(3).join(
    S.groupby(["group", "dataset", "level"]).n.max())""",
])

print("wrote", NB / "06_model_results.ipynb")
