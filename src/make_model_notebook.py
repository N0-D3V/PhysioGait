"""Generate notebooks/06_model_results.ipynb (needs outputs/models/* and outputs/eval/test*/).

    ./py src/train.py --name bigru_v4 --features v3 --angle-beta 0.1 --asym-p 0.5
    ./py src/train.py --name bigru_v3 --features v3 --angle-beta 0.1
    ./py src/train.py --name bigru_v2 && ./py src/train.py --name bigru_v2_noswap --swap-p 0
    ./py src/train.py --name bigru_aug --features v1 --no-angle-head --swap-p 0 --epochs 40
    ./py src/train.py --name bigru_clean --features v1 --no-angle-head --swap-p 0 --no-aug --epochs 40
    ./py src/evaluate.py bigru_v4 bigru_v3 bigru_v2 bigru_v2_noswap bigru_aug bigru_clean
    ./py src/evaluate_trials.py bigru_v4 bigru_v3 && ./py src/evaluate_trials.py bigru_v4 bigru_v3 --two-pass
    ./py src/make_model_notebook.py && (cd notebooks && nix develop .. -c jupyter nbconvert --execute --inplace 06_model_results.ipynb)
"""
from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parents[1] / "notebooks"
DISCLAIMER = "*This project is a prototype and is not medical advice.*"
FINDINGS = (Path(__file__).parent / "model_findings.md")

SETUP = """\
import sys, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display
from analysis import *
style()
pd.set_option("display.width", 200, "display.max_columns", 40)
EV = ROOT / "outputs" / "eval" / "test"
MOD = ROOT / "outputs" / "models"
V4, PREV = "bigru_v4", "bigru_aug"
RUN = f"{V4}+derived"                    # the reported model: events + angle curves -> formulas
METHODS = [RUN, PREV, "classic", "mean"]
MLABEL = {RUN: "v4 (final)", PREV: "v1: pooled regression heads",
          "classic": "Classic pipeline", "mean": "Mean predictor", V4: "v4 pooled heads (auxiliary)",
          "bigru_v3+derived": "v3: no synthetic asymmetry", "bigru_v2+derived": "v2: no gated geometry / local branch",
          "bigru_v2_noswap+derived": "v2 without L/R mirror", "bigru_clean": "v1 trained on clean mocap"}
MCOL = {RUN: SERIES[0], PREV: SERIES[1], "classic": SERIES[2], "mean": MUTED, V4: SERIES[0]}
MSTY = {RUN: "-", PREV: "-", "classic": "-", "mean": "--"}
reg = pd.read_csv(EV / "regression.csv")
evs = pd.read_csv(EV / "events.csv")
angs = pd.read_csv(EV / "angles.csv")
sbins = pd.read_csv(EV / "symmetry_bins.csv")
P = pd.read_parquet(EV / "predictions.parquet")
TR1 = pd.read_csv(ROOT / "outputs" / "eval" / "test_trial" / "regression.csv").assign(protocol="one pass")
TR2 = pd.read_csv(ROOT / "outputs" / "eval" / "test_trial_twopass" / "regression.csv").assign(protocol="two passes")
LEVELS = sorted(reg.level.unique())
SI = ["SI_step_time", "SI_stance_time", "SI_swing_time", "SI_hip_rom", "SI_knee_rom", "SI_ankle_rom"]
UNIT = {"cadence": "steps/min", "stance_pct": "% cycle", "speed_ms": "m/s", "SI_step_time": "SI %",
        "SI_stance_time": "SI %", "SI_swing_time": "SI %", "SI_knee_rom": "SI %", "SI_hip_rom": "SI %",
        "SI_ankle_rom": "SI %", "knee_rom": "deg", "ankle_rom": "deg", "hip_rom": "deg",
        "peak_knee_flex": "deg", "double_support": "s", "stride_time": "s", "stride_time_cv": "%"}"""


def nb(name, cells):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": "gait", "display_name": "Python (gait)", "language": "python"}
    n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c.strip())
               for c in cells]
    nbf.write(n, NB / name)


cells = [
f"""md:# 06 · Multi-head BiGRU: results on held-out subjects

**Encoder** (`src/model.py`): 5 s windows of canonical 2D sagittal pose at 30 Hz (10 joints, pelvis-centred,
leg-length-scaled) + velocities + visibility flags + sin/cos of the 6 segment angles + the 6 geometric joint
angles with validity flags → input MLP → 2-layer BiGRU (128 per direction), plus a local conv branch
(0.3 s receptive field) feeding the per-frame heads.

**v1 → v2.** The first model (v1) regressed every output from time-pooled features. That shrank
symmetry towards 0 (the mostly-symmetric training population) and could not express a max − min ROM.
v2 adds **dense per-frame heads** and computes the stride outputs from them with the **same formulas as
the gold labels**:
- **events head** (per frame): heel-strike / toe-off, left / right → sub-frame event times
- **angle head** (per frame): the 6 sagittal joint angles
- **formulas** (`gaitdata.derive`): strides from the events → cadence, stride / step / stance / swing time,
  stance %, double support, stride-time variability; ROM and peak knee flexion per stride from the angle
  curves; Robinson SI of each from the per-side values
- **pooled heads** kept as auxiliary losses, for **speed** (leg lengths/s → m/s with the patient's leg length),
  and as a fallback when no complete stride is decoded
- **left/right mirror augmentation** (swap sides, flip the SI sign)

**v2 → v3.** The v2 angle head re-learned trigonometry and was 2–4° off even on clean pose, so ROM symmetry
(a ~3° left/right difference) was invisible. v3 makes the angle head a **gated skip**:
`gate × exact geometric angle + (1 − gate) × learned angle`, gate forced to 0 where a needed keypoint is
hidden: exact geometry on clean pose, a denoised estimate on noisy pose. A **local conv branch** lets the
event and angle heads use the immediate foot motion next to the BiGRU's global context.

**v3 → v4.** Event errors were small (3–5 ms) but *correlated with asymmetry*: random jitter of the same size
keeps the SI slope at 1.0, the model's own events gave 0.6. It had learned "the legs are half a cycle apart"
from a near-symmetric training population. v4 adds **synthetic asymmetry** (p = 0.5): one leg phase-shifted
by up to ±50 ms and its hip/knee/ankle excursions scaled ×0.8–1.25 by rigid segment rotations, with all
labels recomputed from the modified events and angles.

**Labels** are recomputed per window from the strides fully inside it (force-plate events for Fukuchi
treadmill, bias-corrected Zeni elsewhere). **Split**: subject-level; **Toronto is held out entirely**
(older adults, IMU capture). **Pose-noise augmentation** (`src/augment.py`) and every evaluation level
give all methods the identical noisy input (σ in leg lengths; 0.01 ≈ 8 mm).

{DISCLAIMER}""",
SETUP,
"""md:## Training curves
Validation loss on held-out validation subjects with fixed noise (σ = 0.02); the best epoch is kept. v4's
training loss is higher by design (half its windows carry synthetic asymmetry); v2's angle loss used a
different smooth-L1 β and is not shown.""",
"""heads = ["events", "angles", "spatiotemporal", "speed", "symmetry", "rom"]
fig, axs = plt.subplots(1, len(heads), figsize=(16, 2.8), sharex=True)
for r, c in ((V4, SERIES[0]), ("bigru_v3", SERIES[2]), ("bigru_v2", MUTED), (PREV, SERIES[1])):
    h = pd.read_csv(MOD / r / "history.csv")
    for a, k in zip(axs, heads):
        if f"val_{k}" in h and not (k == "angles" and r == "bigru_v2"):
            a.plot(h.epoch, h[f"val_{k}"], color=c, label=r)
            a.plot(h.epoch, h[f"train_{k}"], color=c, lw=1, ls=":")
for a, k in zip(axs, heads): a.set_title(k); a.set_xlabel("epoch")
axs[0].set_ylabel("loss (solid = val, dotted = train)")
axs[0].legend(fontsize=8)
fig.tight_layout(); save(fig, "06_training_curves"); plt.show()""",
"""md:## Headline table
Mean absolute error per dataset on held-out subjects (per 5 s window), clean input (σ = 0) and realistic
video-pose noise (σ = 0.02). Best per row in bold.""",
"""core = ["cadence", "stride_time", "stance_pct", "double_support", "speed_ms", "SI_step_time", "SI_stance_time",
        "SI_knee_rom", "SI_ankle_rom", "hip_rom", "knee_rom", "ankle_rom", "peak_knee_flex", "stride_time_cv"]
T = (reg[reg.target.isin(core) & reg.level.isin([0, 0.02])]
     .pivot_table(index=["dataset", "target"], columns=["level", "method"], values="mae")
     .reindex(columns=METHODS, level=1))
T = T.reindex(pd.MultiIndex.from_product([DATASETS, core]), axis=0).dropna(how="all")
T.style.format("{:.3g}").apply(lambda r: ["font-weight: bold" if v == r[l].min() else "" for (l, m), v in r.items()], axis=1)""",
"""md:## Symmetry: is real asymmetry reported at its true size?
A low MAE on SI is easy: most people are near-symmetric, so predicting ~0 scores well. What matters for a
patient is whether a *large* asymmetry comes out large. Two checks: the slope of predicted vs gold SI
(1 = correct size, 0 = always the mean) and the error by size of the true asymmetry, with the average
amount by which |SI| is under-reported (negative = shrunk towards 0).""",
"""S = reg[reg.target.isin(SI) & (reg.level == 0.02) & reg.method.isin([RUN, PREV, "classic"])]
fig, axs = plt.subplots(2, len(DATASETS), figsize=(15, 6), sharey="row")
x = np.arange(len(SI))
for k, ds in enumerate(DATASETS):
    for j, m in enumerate([RUN, PREV, "classic"]):
        q = S[(S.dataset == ds) & (S.method == m)].set_index("target").reindex(SI)
        axs[0, k].bar(x + (j - 1) * 0.27, q.slope, width=0.25, color=MCOL[m], label=MLABEL[m])
        axs[1, k].bar(x + (j - 1) * 0.27, q.r, width=0.25, color=MCOL[m])
    for a in axs[:, k]:
        a.axhline(1, color=MUTED, lw=1, ls="--"); a.axhline(0, color=MUTED, lw=1)
        a.set_xticks(x, [s.replace("SI_", "") for s in SI], rotation=30)
    axs[0, k].set_title(f"{ds} (σ = 0.02)")
axs[0, 0].set_ylabel("slope, predicted vs gold SI"); axs[1, 0].set_ylabel("Pearson r")
axs[0, 0].legend(fontsize=8)
fig.tight_layout(); save(fig, "06_symmetry_slope"); plt.show()""",
"""B = sbins[(sbins.level == 0.02) & sbins.method.isin([RUN, PREV, "classic"])]
fig, axs = plt.subplots(1, 4, figsize=(16, 3.4), sharey=True)
for a, t in zip(axs, ["SI_step_time", "SI_stance_time", "SI_knee_rom", "SI_ankle_rom"]):
    for j, m in enumerate([RUN, PREV, "classic"]):
        q = B[(B.target == t) & (B.method == m)].groupby("abs_bin").apply(
            lambda g: np.average(g.mean_abs_shrink, weights=g.n)).reindex(["<3", "3-7", ">=7"])
        a.bar(np.arange(3) + (j - 1) * 0.27, q, width=0.25, color=MCOL[m], label=MLABEL[m])
    a.axhline(0, color=MUTED, lw=1); a.set_xticks(range(3), ["|SI| < 3", "3–7", "≥ 7"]); a.set_title(t)
axs[0].set_ylabel("mean |pred| − |gold| (SI %)"); axs[0].legend(fontsize=8)
fig.tight_layout(); save(fig, "06_symmetry_shrinkage"); plt.show()
(sbins[(sbins.level == 0.02) & sbins.method.isin([RUN, "bigru_v3+derived", PREV, "classic"])]
 .pivot_table(index=["target", "abs_bin"], columns=["method"], values=["mae", "mean_abs_shrink"], aggfunc="mean")
 .round(2))""",
"""md:## How each method degrades with pose noise
On clean mocap the classic pipeline computes the labels with their own formulas, so it is hard to beat; the
learned model has to earn its place as the input starts to look like video pose. Dashed = mean predictor.""",
"""targets = ["cadence", "stance_pct", "speed_ms", "SI_step_time", "SI_knee_rom", "knee_rom", "ankle_rom"]
fig, axs = plt.subplots(len(DATASETS), len(targets), figsize=(19, 8.2), sharex=True)
for i, ds in enumerate(DATASETS):
    for k, t in enumerate(targets):
        a = axs[i, k]
        for m in METHODS:
            g = reg[(reg.dataset == ds) & (reg.target == t) & (reg.method == m)].sort_values("level")
            a.plot(g.level, g.mae, color=MCOL[m], ls=MSTY[m], marker="o", ms=4, label=MLABEL[m])
        a.set_ylim(0, 2.2 * reg[(reg.dataset == ds) & (reg.target == t) & (reg.method == "mean")].mae.max())
        if i == 0: a.set_title(t)
        if k == 0: a.set_ylabel(f"{ds}\\nMAE")
        if i == len(DATASETS) - 1: a.set_xlabel("pose noise σ")
        a.text(0.98, 0.95, UNIT.get(t, ""), transform=a.transAxes, ha="right", va="top", color=MUTED, fontsize=8)
axs[0, 0].legend(fontsize=8, loc="upper left")
fig.tight_layout(); save(fig, "06_degradation"); plt.show()""",
"""md:## Joint-angle curves (angle head)
RMSE of the per-frame angles over the whole window, against the clean mocap angles. The classic curve is
the angle formula on the gap-filled, 6 Hz low-passed noisy input; v2 is the ungated angle head.""",
"""J6 = ["L_hip", "L_knee", "L_ankle"]
fig, axs = plt.subplots(1, len(DATASETS), figsize=(15, 3.4), sharey=True)
for a, ds in zip(axs, DATASETS):
    for m, c in ((V4, SERIES[0]), ("classic", SERIES[2])):
        for j, ls in zip(J6, ("-", "--", ":")):
            g = angs[(angs.dataset == ds) & (angs.method == m)].sort_values("level")
            a.plot(g.level, (g[j] + g["R" + j[1:]]) / 2, color=c, ls=ls, marker="o", ms=3,
                   label=f"{'v4 angle head' if m == V4 else 'classic'} {j[2:]}")
    a.set_title(ds); a.set_xlabel("pose noise σ"); a.set_ylim(bottom=0)
axs[0].set_ylabel("angle RMSE (deg)"); axs[0].legend(fontsize=7, ncol=2)
fig.tight_layout(); save(fig, "06_angle_rmse"); plt.show()
angs[angs.method.isin([V4, "bigru_v2", "classic"])].set_index(["dataset", "level", "method"]).drop(columns="n").round(2).unstack("method")""",
"""md:## Gait events
Detection F1 within ±100 ms and timing error against the labels; events within 0.2 s of a window edge excluded.""",
"""E = evs[evs.method.isin([V4, PREV, "classic"])]
EC = {V4: SERIES[0], PREV: SERIES[1], "classic": SERIES[2]}
fig, axs = plt.subplots(2, len(DATASETS), figsize=(15, 6), sharex=True)
for k, ds in enumerate(DATASETS):
    for m in EC:
        for ev, ls in (("HS", "-"), ("TO", ":")):
            g = E[(E.dataset == ds) & (E.method == m) & (E.event == ev)].sort_values("level")
            axs[0, k].plot(g.level, g.f1, color=EC[m], ls=ls, marker="o", ms=4, label=f"{m} {ev}")
            axs[1, k].plot(g.level, g.mae_ms, color=EC[m], ls=ls, marker="o", ms=4)
    axs[0, k].set_title(ds); axs[1, k].set_xlabel("pose noise σ"); axs[1, k].set_ylim(bottom=0)
axs[0, 0].set_ylabel("F1 (±100 ms)"); axs[1, 0].set_ylabel("timing MAE (ms)")
axs[0, 0].legend(fontsize=7, ncol=2)
fig.tight_layout(); save(fig, "06_events"); plt.show()
E[E.level.isin([0, 0.02])].pivot_table(index=["dataset", "event"], columns=["level", "method"],
                                       values=["f1", "mae_ms", "bias_ms"]).round(3)""",
"""md:## Per recording: one output per video, and the two-pass protocol
A physio films more than 5 s. Here each test trial gets **one** noise draw over its whole length (fixed far leg,
keypoint offsets, scale and tilt, as in one real video), every window's strides are pooled, and the outputs are
computed over all strides of the trial (30 s treadmill trials ≈ 50 strides). Averaging removes random error but
not systematic shrinkage. **Two passes** = walk across and back: each leg is measured from the pass in which it
is nearest the camera, so no left/right comparison pits a near leg against an occluded far leg.""",
"""tg = ["SI_step_time", "SI_stance_time", "SI_hip_rom", "SI_knee_rom", "SI_ankle_rom", "knee_rom", "ankle_rom",
      "cadence", "stance_pct", "speed_ms"]
TR = pd.concat([TR1, TR2])
TR = TR[TR.target.isin(tg) & (TR.level == 0.02) & TR.method.isin([RUN, "bigru_v3+derived", "classic", "mean"])]
TR = TR[~((TR.method == "mean") & (TR.protocol == "two passes"))]
TR.assign(method=TR.method.map(lambda m: MLABEL.get(m, m))).pivot_table(
    index=["target", "dataset"], columns=["protocol", "method"], values=["mae", "slope"]).round(2)""",
"""TRA = pd.concat([TR1, TR2])
fig, axs = plt.subplots(1, 4, figsize=(18, 3.6))
for a, t in zip(axs, ["SI_step_time", "SI_hip_rom", "SI_knee_rom", "SI_ankle_rom"]):
    for m in (RUN, "classic"):
        for prot, ls in (("one pass", "-"), ("two passes", "--")):
            g = TRA[(TRA.target == t) & (TRA.method == m) & (TRA.protocol == prot) & (TRA.dataset == "ferber")]
            g = g.sort_values("level")
            a.plot(g.level, g.mae, color=MCOL[m], ls=ls, marker="o", ms=4, label=f"{MLABEL[m]}, {prot}")
    mn = TR1[(TR1.target == t) & (TR1.method == "mean") & (TR1.dataset == "ferber")].mae.mean()
    a.axhline(mn, color=MUTED, ls=":", lw=1.5, label="mean predictor")
    a.set_title(f"{t} (Ferber, per recording)"); a.set_xlabel("pose noise σ"); a.set_ylim(0, 2.5 * mn)
axs[0].set_ylabel("MAE (SI %)"); axs[0].legend(fontsize=7)
fig.tight_layout(); save(fig, "06_recording_level"); plt.show()""",
"""md:## Ablation (σ = 0.02)
What each change buys, removing one step at a time: v4 → v3 (no synthetic asymmetry) → v2 (no gated geometry,
no local branch) → v2 without the L/R mirror; v4's own pooled heads (same network, no formula step); the first
model (v1) and v1 trained on clean mocap only. MAE first, then the SI slopes (1 = asymmetry at its true size).""",
"""abl = [RUN, "bigru_v3+derived", "bigru_v2+derived", "bigru_v2_noswap+derived", V4, PREV, "bigru_clean", "classic"]
A = (reg[(reg.level == 0.02) & reg.target.isin(core) & reg.method.isin(abl)]
     .pivot_table(index=["dataset", "target"], columns="method", values="mae").reindex(columns=abl))
A = A.reindex(pd.MultiIndex.from_product([DATASETS, core])).dropna(how="all")
A.columns = [MLABEL.get(c, c) for c in A.columns]
display(A.style.format("{:.3g}").highlight_min(axis=1, props="font-weight: bold"))
Sl = (reg[(reg.level == 0.02) & reg.target.isin(SI) & reg.method.isin(abl)]
      .pivot_table(index=["dataset", "target"], columns="method", values="slope").reindex(columns=abl))
Sl.columns = [MLABEL.get(c, c) for c in Sl.columns]
Sl.round(2)""",
"""md:## Bland–Altman (v4, σ = 0.02)
Agreement with the gold labels per window; bias and 95 % limits of agreement pooled across datasets
(per-dataset numbers below).""",
"""ba = ["SI_step_time", "SI_stance_time", "SI_knee_rom", "cadence", "stance_pct", "speed_ms", "knee_rom", "ankle_rom"]
Q = P[(P.method == RUN) & (P.level == 0.02)]
fig, axs = plt.subplots(2, 4, figsize=(17, 7))
rows = []
for a, t in zip(axs.flat, ba):
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
axs[0, 0].set_ylabel("model − gold"); axs[1, 0].set_ylabel("model − gold"); axs[0, 0].legend(fontsize=8, markerscale=3)
fig.tight_layout(); save(fig, "06_bland_altman"); plt.show()
pd.DataFrame(rows).pivot(index="target", columns="dataset", values=["bias", "loa"]).round(3)""",
"""md:## Example: one held-out window at σ = 0.04
Noisy input, event probabilities with the gold events, and the predicted knee and ankle curves against the
clean mocap curves.""",
"""import torch, augment, gaitdata as G
from evaluate import load_model, run_model
from model import decode_events
data = G.load_all()["test"]
tr = next(t for t in data if t.ds == "fukuchi" and t.mode == "treadmill" and t.trial.endswith("T05"))
x, v = augment.apply(tr.pose[300:450], np.random.default_rng(3), augment.Noise.level(0.04))
m, cfg, norm = load_model(V4)
probs, y, ang = run_model(m, norm, [x], [v], cfg["features"])
t = np.arange(150) / 30
fig, axs = plt.subplots(3, 1, figsize=(13, 7.5), sharex=True)
for s, c in SIDE_COL.items():
    j = G.IN_JOINTS.index(f"{s}_heel")
    axs[0].plot(t, tr.pose[300:450, j, 0], color=c, lw=1, alpha=0.5)
    axs[0].plot(t, np.where(v[:, j] > 0, x[:, j, 0], np.nan), color=c, lw=0, marker="o", ms=3, label=f"{s} heel (noisy input)")
for k, name in enumerate(G.EVENTS):
    s, ls = name[-1], "-" if name.startswith("HS") else ":"
    axs[1].plot(t, probs[0][:, k], color=SIDE_COL[s], ls=ls, label=name)
    for e in tr.events[name] - 10:
        if 0 <= e <= 5: axs[1].axvline(e, color=SIDE_COL[s], lw=0.8, ls=ls, alpha=0.6)
for k, (n, ls) in enumerate((("L_knee", "-"), ("L_ankle", "--"))):
    i = G.ANG_NAMES.index(n)
    axs[2].plot(t, tr.ang[300:450, i], color=MUTED, ls=ls, lw=3, label=f"{n} gold")
    axs[2].plot(t, ang[0][:, i], color=SIDE_COL["L"], ls=ls, label=f"{n} predicted")
axs[0].set_ylabel("forward (leg lengths)"); axs[0].legend(fontsize=8, ncol=2)
axs[0].set_title(f"{tr.trial} (held-out subject), thin line = clean")
axs[1].set_ylabel("event probability"); axs[1].legend(fontsize=8, ncol=4)
axs[2].set_ylabel("angle (deg)"); axs[2].set_xlabel("s"); axs[2].legend(fontsize=8, ncol=2)
fig.tight_layout(); save(fig, "06_example_window"); plt.show()
yd = G.derive(decode_events(probs[0]), ang[0], 150, y[0][G.TI["speed_ll"]])
pd.DataFrame({"v4 derived": yd, "v4 pooled": y[0], "gold": G.window_labels(tr, 300, 450)},
             index=G.TARGET_NAMES).round(3).T""",
"""md:## Subgroups (age, sex)
v4 MAE at σ = 0.02 by subgroup, from the trial metadata. Toronto (all older adults) is the external check.""",
"""meta = all_trials()[["dataset", "trial", "age", "sex"]].copy()
meta["sex"] = meta.sex.astype(str).str.upper().str[0].map({"M": "male", "F": "female"})
meta["age_band"] = pd.cut(meta.age, [0, 40, 60, 120], labels=["<40", "40–60", "≥60"])
Q = P[(P.method == RUN) & (P.level == 0.02)].merge(meta, on=["dataset", "trial"], how="left")
rows = []
for by in ("age_band", "sex"):
    for (ds, grp), g in Q.groupby(["dataset", by], observed=True):
        for t in ["cadence", "stance_pct", "speed_ms", "SI_step_time", "knee_rom", "ankle_rom"]:
            d = g[[f"true_{t}", f"pred_{t}"]].dropna()
            rows.append(dict(group=by, dataset=ds, level=str(grp), target=t, n=len(d),
                             mae=(d.iloc[:, 1] - d.iloc[:, 0]).abs().mean()))
S = pd.DataFrame(rows)
S.pivot_table(index=["group", "dataset", "level"], columns="target", values="mae").round(3).join(
    S.groupby(["group", "dataset", "level"]).n.max())""",
]
if FINDINGS.exists():
    cells.append("md:" + FINDINGS.read_text())

nb("06_model_results.ipynb", cells)
print("wrote", NB / "06_model_results.ipynb")
