"""Gradio demo: side-on walking video -> pose -> gait measurements.

    ./py src/app.py            # http://127.0.0.1:7860
    ./py src/app.py --share    # temporary public link (gradio.live)

This project is a prototype and is not medical advice.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import gradio as gr  # noqa: E402

import augment  # noqa: E402
import render  # noqa: E402
from analysis import ROOT  # noqa: E402
from gaitdata import FS, STRIDE_COLS, TARGET_NAMES, TI, aggregate, load_all  # noqa: E402
from pose_video import PoseTracker  # noqa: E402
from video_pipeline import LEG_PER_HEIGHT, Analyzer  # noqa: E402

DISCLAIMER = "This project is a prototype and is not medical advice."
DEMO_DIR = ROOT / "data" / "demo_videos"
OUT = Path(tempfile.gettempdir()) / "gait_demo"
OUT.mkdir(exist_ok=True)
REL = {"good": "✓ good", "rough": "~ rough", "unreliable": "✗ not reliable at this pose quality", "": ""}

_state: dict = {}


def analyzer() -> Analyzer:
    if "an" not in _state:
        _state["an"] = Analyzer()
    return _state["an"]


def tracker() -> PoseTracker:
    if "tr" not in _state:
        _state["tr"] = PoseTracker()
    return _state["tr"]


# ---------------------------------------------------------------- formatting
def _fmt(v, unit):
    if v is None or not np.isfinite(v):
        return "–"
    if unit == "s":
        return f"{v:.3f}"
    if unit in ("deg", "%", "% of stride", "steps/min"):
        return f"{v:.1f}"
    return f"{v:.2f}"


def summary_table(rep, gold: dict | None = None) -> pd.DataFrame:
    rows = []
    for r in rep.summary.itertuples():
        row = {"Measure": r.measure, "Value": _fmt(r.value, r.unit), "Unit": r.unit,
               "Left": _fmt(r.left, r.unit), "Right": _fmt(r.right, r.unit)}
        if gold is not None:
            row["Gold (mocap)"] = _fmt(gold.get(r.key, np.nan), r.unit)
        row["Typical error ±"] = _fmt(r.expected_error, r.unit)
        row["Reliability"] = REL.get(r.reliability, r.reliability)
        row["Healthy adults (5–95th pct)"] = (f"{_fmt(r.healthy_p5, r.unit)} – {_fmt(r.healthy_p95, r.unit)}"
                                             if np.isfinite(r.healthy_p5) else "–")
        rows.append(row)
    return pd.DataFrame(rows)


def quality_md(rep, extra: str = "") -> str:
    q = rep.quality
    lines = [f"**Pose quality:** estimated noise ≈ σ {q['sigma_equiv']:.3f} leg lengths "
             f"(evaluation levels: 0 = clean mocap, 0.02 = realistic video pose). Typical errors and the "
             f"reliability column are the validated per-recording errors at this level.",
             f"**Strides measured:** {q['n_strides_L']} left, {q['n_strides_R']} right. "
             f"Visible keypoints: {100 * q['visible_frac']:.0f} %."]
    if "passes" in q:
        lines.append(f"**Walking passes found:** {q['passes']}"
                     + (" in both directions (each leg measured from the pass where it faces the camera)."
                        if q["both_directions"] else
                        " in one direction: the leg further from the camera is partly hidden, so left/right "
                        "comparisons are less certain. Filming the walk back as well is recommended.")
                     + (f" Left/right label swaps corrected: {q['lr_swaps_fixed']}." if q.get("lr_swaps_fixed") else ""))
    if q["n_strides_L"] + q["n_strides_R"] < 6:
        lines.append("⚠️ **Few strides**: values are noisy; a clip of ≥ 20 s of steady walking is recommended.")
    if extra:
        lines.append(extra)
    lines.append(f"*{DISCLAIMER}*")
    return "\n\n".join(lines)


def strides_table(rep) -> pd.DataFrame:
    S = rep.strides.copy()
    for c in ("start_s", "end_s") + tuple(c for c in STRIDE_COLS if c not in ("stance_pct",)):
        if c in S:
            S[c] = S[c].round(3 if c.endswith(("time", "_s", "support")) else 1)
    S["stance_pct"] = S.stance_pct.round(1)
    return S


def _csv(rep, name: str) -> str:
    p = OUT / f"{name}_report.csv"
    rep.summary.drop(columns=["key"]).to_csv(p, index=False)
    return str(p)


# ---------------------------------------------------------------- tab 1: video
def analyse_video(video, height_cm, leg_cm, conf, max_s, progress=gr.Progress()):
    if not video:
        raise gr.Error("Upload a side-on walking video first (or pick the example below).")
    leg_m = float(leg_cm) / 100 if leg_cm else (LEG_PER_HEIGHT * float(height_cm) / 100 if height_cm else None)
    progress(0, desc="loading models")
    kp, score, fps, size, _ = tracker().run(video, max_seconds=float(max_s), progress=progress)
    progress(0.9, desc="gait model")
    try:
        rep = analyzer().from_keypoints(kp, score, fps, leg_length_m=leg_m, conf=float(conf))
    except ValueError as e:
        raise gr.Error(str(e))
    progress(0.95, desc="rendering")
    stem = Path(video).stem
    vid = render.annotate_video(video, rep, str(OUT / f"{stem}_annotated.mp4"), max_seconds=float(max_s))
    note = "" if leg_m else "Walking speed in m/s needs the patient's height or leg length (left column)."
    return (vid, quality_md(rep, note), summary_table(rep), render.symmetry_plot(rep), render.gait_diagram(rep),
            render.angle_plot(rep), strides_table(rep), _csv(rep, stem))


# ---------------------------------------------------------------- tab 2: held-out mocap trial
def _test_choices():
    if "choices" in _state:
        return _state["choices"]
    test = load_all()["test"]
    pick = []
    for ds, mode, n in (("fukuchi", "treadmill", 4), ("fukuchi", "overground", 2), ("ferber", "treadmill", 5),
                        ("toronto", "overground_pass", 3)):
        cand = [t for t in test if t.ds == ds and t.mode == mode and len(t.strides) >= 4]
        si = [abs(aggregate(t.strides, 0, len(t.pose) / FS)[TI["SI_step_time"]]) for t in cand]
        order = np.argsort(si)[::-1]                    # include the most asymmetric ones
        pick += [cand[i] for i in order[:n]]
    labels = {}
    for t in pick:
        g = aggregate(t.strides, 0, (len(t.pose) - 1) / FS, t.speed_ll)
        labels[f"{t.ds} · {t.mode} · {t.subject} · {t.trial} · {len(t.pose) / FS:.0f} s · "
               f"{t.speed_ll * t.leg_length:.2f} m/s · step SI {g[TI['SI_step_time']]:+.1f} %"] = t
    _state["choices"] = labels
    return labels


def analyse_trial(choice, sigma, progress=gr.Progress()):
    if not choice:
        raise gr.Error("Pick a held-out test trial.")
    tr = _test_choices()[choice]
    progress(0.1, desc="noise + gait model")
    x, vis = augment.apply(tr.pose, np.random.default_rng(0), augment.Noise.level(float(sigma)))
    rep = analyzer().from_canonical(x, vis, [(0, len(x), 1)], leg_length_m=tr.leg_length)
    T = len(tr.pose)
    g = aggregate(tr.strides, 0, (T - 1) / FS, tr.speed_ll)
    gold = {n: g[i] for i, n in enumerate(TARGET_NAMES)} | {"speed_ms": tr.speed_ll * tr.leg_length}
    progress(0.6, desc="rendering")
    vid = render.stick_video(rep, str(OUT / f"{tr.trial}_s{sigma}.mp4"))
    extra = (f"**Input:** motion-capture pose of a held-out subject ({tr.ds}, {tr.subject}), projected to the side "
             f"view at 30 Hz, with simulated video-pose noise σ = {float(sigma):.2f}. Gold labels: "
             f"{'force-plate' if tr.event_source == 'grf' else 'bias-corrected Zeni'} events + mocap angles.")
    return (vid, quality_md(rep, extra), summary_table(rep, gold), render.symmetry_plot(rep),
            render.gait_diagram(rep), render.angle_plot(rep), strides_table(rep))


# ---------------------------------------------------------------- layout
ABOUT = f"""
### What it measures
From a side-on walking clip: **cadence, walking speed, stride / step / stance / swing time, stance %, double
support, hip / knee / ankle range of motion, peak knee flexion, stride-time variability**, and the **left/right
symmetry index** of each (Robinson SI = 100 × (L − R) / mean; 0 = symmetric, > 0 = left larger).

### How it works
1. **Pose** (`pose_video.py`): YOLOX person detector + RTMPose-m (Halpe-26 keypoints, incl. heels and toes), ONNX on CPU.
2. **Canonical input** (`video_pipeline.py`): hips, knees, ankles, heels, toes; left/right label swaps fixed with a
   motion-continuity Viterbi pass; walking passes split by facing direction; pelvis-centred, leg-length-scaled, 30 Hz.
3. **Model** (`model.py`, v4): BiGRU encoder → per-frame gait events (heel strike / toe-off) and joint-angle curves
   (gated: exact geometry where the pose is clean, a learned denoised estimate where it is not).
4. **Formulas** (`gaitdata.derive`): strides from the events, every output computed with the same definitions as the
   gold-standard labels; strides pooled over the clip.
5. **Typical error & reliability**: the clip's keypoint jitter is matched to the evaluation noise levels, and each
   output shows the error measured on held-out subjects at that level (✗ = no better than guessing the average).

### Validated on (held-out subjects)
Fukuchi (healthy young and older adults, force plates), Ferber (clinic, walking), Toronto (older adults, never used
for training) — motion capture with simulated video-pose noise. **Not yet validated on real video against a
reference system.**

### Known limits
- Ankle-ROM symmetry is not measurable at typical video pose quality (flagged ✗).
- Stance/swing-time asymmetry is under-reported by ~40 %; step-time asymmetry by ~10–25 %.
- Side view only: no frontal-plane measures (pelvic drop, knee valgus). Treadmill or straight walkway, one person.
- Walking speed needs the patient's height or leg length; leg length from height uses an average body proportion.

### Recording tips
Camera at hip height, perpendicular to the walking line, whole body including feet in frame, ≥ 20 s of steady
walking; walk across **and back** so each leg is filmed facing the camera once.

Demo video: *Yoon J, Park H-S, Damiano DL (2012) A novel walking speed estimation scheme and its application to
treadmill control for gait rehabilitation. J NeuroEng Rehabil 9:62*, CC BY 2.0, via Wikimedia Commons (re-encoded).

**{DISCLAIMER}**
"""


def build() -> gr.Blocks:
    examples = sorted(DEMO_DIR.glob("*.mp4"))
    with gr.Blocks(title="Gait measurement prototype") as demo:
        gr.Markdown(f"# Gait measurement from a side-on video\nSymmetry, timing and joint range of motion from one "
                    f"cheap clip, for tracking a patient's progress, not for diagnosis.  \n**{DISCLAIMER}**")
        with gr.Tab("Analyse a walking video"):
            with gr.Row():
                with gr.Column(scale=1):
                    v_in = gr.Video(label="Side-on walking video", sources=["upload"])
                    height = gr.Number(label="Patient height (cm), for speed in m/s", value=None, precision=0)
                    leg = gr.Number(label="…or measured leg length, hip joint to ankle (cm)", value=None, precision=0)
                    with gr.Accordion("Advanced", open=False):
                        conf = gr.Slider(0.1, 0.8, value=0.3, step=0.05, label="Keypoint confidence threshold")
                        max_s = gr.Slider(5, 90, value=60, step=5, label="Analyse at most (seconds)")
                    go = gr.Button("Analyse", variant="primary")
                    if examples:
                        gr.Examples([[str(p), 165, None] for p in examples], inputs=[v_in, height, leg],
                                    label="Example (CC BY 2.0, user-driven treadmill 0.5 → 1.5 → 0.5 m/s)")
                with gr.Column(scale=2):
                    v_out = gr.Video(label="Detected pose and gait phase", autoplay=True)
                    q_md = gr.Markdown()
            table = gr.Dataframe(label="Measurements", wrap=True)
            sym = gr.Plot(label="Left vs right")
            diag = gr.Plot(label="Gait diagram")
            ang = gr.Plot(label="Joint angles")
            with gr.Accordion("Per-stride values (the strides behind every number)", open=False):
                strides = gr.Dataframe()
            csv = gr.File(label="Download measurements (CSV)")
            go.click(analyse_video, [v_in, height, leg, conf, max_s], [v_out, q_md, table, sym, diag, ang, strides, csv])

        with gr.Tab("Held-out test trial (mocap)"):
            gr.Markdown("Canned input: a trial from a subject the model never saw, as side-view pose from motion "
                        "capture, with optional simulated video-pose noise; the gold-standard values are shown next "
                        "to the model's.")
            with gr.Row():
                pick = gr.Dropdown(choices=list(_test_choices()), label="Held-out trial", scale=3)
                sig = gr.Slider(0, 0.04, value=0.02, step=0.01, label="Simulated pose noise σ (leg lengths)", scale=1)
                go2 = gr.Button("Run", variant="primary", scale=1)
            with gr.Row():
                t_vid = gr.Video(label="Model input (stick figure) and decoded gait phase", autoplay=True)
                t_md = gr.Markdown()
            t_table = gr.Dataframe(label="Model vs gold standard", wrap=True)
            t_sym = gr.Plot()
            t_diag = gr.Plot()
            t_ang = gr.Plot()
            with gr.Accordion("Per-stride values", open=False):
                t_str = gr.Dataframe()
            go2.click(analyse_trial, [pick, sig], [t_vid, t_md, t_table, t_sym, t_diag, t_ang, t_str])

        with gr.Tab("How it works & limits"):
            gr.Markdown(ABOUT)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build().queue().launch(server_port=a.port, share=a.share, allowed_paths=[str(OUT), str(DEMO_DIR)])
