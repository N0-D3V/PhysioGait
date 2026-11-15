"""Gradio demo: side-on walking video -> pose -> gait measurements.

    ./py src/app.py [--share]      # http://127.0.0.1:7860

This project is a prototype and is not medical advice.
"""
import argparse
import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

import augment
from helpers import render
from helpers.analysis import ROOT
from helpers.gaitdata import FS, TARGET_NAMES, TI, aggregate, load_all
from pose_video import PoseTracker
from video_pipeline import LEG_PER_HEIGHT, Analyzer

DISCLAIMER = "This project is a prototype and is not medical advice."
DEMO_DIR = ROOT / "data" / "demo_videos"
OUT = Path(tempfile.gettempdir()) / "gait_demo"
OUT.mkdir(exist_ok=True)
RELIABILITY = {"good": "✓ good", "rough": "~ rough", "unreliable": "✗ not reliable at this pose quality", "": ""}
COLUMNS = ["Measure", "Value", "Unit", "Left", "Right", "Typical error ±", "Reliability", "Healthy adults (5–95th pct)"]
_cache = {}   # models are loaded once, on first use


def analyzer():
    if "analyzer" not in _cache:
        _cache["analyzer"] = Analyzer()
    return _cache["analyzer"]


def fmt(v, unit):
    if v is None or not np.isfinite(v):
        return "–"
    return f"{v:.3f}" if unit == "s" else f"{v:.1f}" if unit in ("deg", "%", "% of stride", "steps/min") else f"{v:.2f}"


def summary_table(rep, gold=None):
    rows = []
    for r in rep.summary.itertuples():
        row = {"Measure": r.measure, "Value": fmt(r.value, r.unit), "Unit": r.unit, "Left": fmt(r.left, r.unit),
               "Right": fmt(r.right, r.unit)}
        if gold is not None:
            row["Gold (mocap)"] = fmt(gold.get(r.key, np.nan), r.unit)
        row["Typical error ±"] = fmt(r.expected_error, r.unit)
        row["Reliability"] = RELIABILITY[r.reliability]
        row["Healthy adults (5–95th pct)"] = (f"{fmt(r.healthy_p5, r.unit)} – {fmt(r.healthy_p95, r.unit)}"
                                             if np.isfinite(r.healthy_p5) else "–")
        rows.append(row)
    return pd.DataFrame(rows)


def quality_text(rep, extra="", video=False):
    q = rep.quality
    lines = [f"**Typical errors** are those measured on held-out people at pose noise σ = {q['sigma']:.2f} leg lengths "
             f"(0 = perfect motion capture). Real research video matched about 0.04; event timing and hip angles were "
             f"somewhat worse still, so treat them as a lower bound.",
             f"**Strides measured:** {q['n_strides_L']} left, {q['n_strides_R']} right. "
             f"Visible keypoints: {100 * q['visible_frac']:.0f} %."]
    if video:
        lines.append(f"**Walking passes:** {q['passes']}" + (" in both directions." if q["both_directions"] else
                     ". Only one direction was filmed, so the leg further from the camera is partly hidden; "
                     "filming the walk back as well is recommended."))
    if q.get("lr_swaps_fixed"):
        lines.append(f"The pose estimator mixed up the left and right leg in {q['lr_swaps_fixed']} stretch(es); corrected.")
    if q["hs_alternation_breaks"]:
        lines.append(f"⚠️ {q['hs_alternation_breaks']} place(s) where the same foot landed twice in a row: the legs may "
                     "have been confused there.")
    if q["n_strides_L"] + q["n_strides_R"] < 6:
        lines.append("⚠️ **Few strides**: values are noisy; record at least 20 s of steady walking.")
    return "\n\n".join(lines + [extra, f"*{DISCLAIMER}*"])


def analyse_video(video, height_cm, leg_cm, conf, max_s, progress=gr.Progress()):
    if not video:
        raise gr.Error("Upload a side-on walking video first (or pick the example below).")
    leg_m = float(leg_cm) / 100 if leg_cm else (LEG_PER_HEIGHT * float(height_cm) / 100 if height_cm else None)
    if "tracker" not in _cache:
        _cache["tracker"] = PoseTracker(threads=min(8, os.cpu_count() or 4))
    kp, score, fps, _, _ = _cache["tracker"].run(video, max_seconds=float(max_s), progress=progress)
    progress(0.9, desc="gait model")
    try:
        rep = analyzer().from_keypoints(kp, score, fps, leg_length_m=leg_m, conf=float(conf))
    except ValueError as e:
        raise gr.Error(str(e))
    progress(0.95, desc="rendering")
    stem = Path(video).stem
    vid = render.annotate_video(video, rep, str(OUT / f"{stem}_annotated.mp4"), max_seconds=float(max_s))
    csv = OUT / f"{stem}_report.csv"
    rep.summary.drop(columns=["key"]).to_csv(csv, index=False)
    note = "" if leg_m else "Walking speed in m/s needs the patient's height or leg length."
    return (vid, quality_text(rep, note, video=True), summary_table(rep), render.symmetry_plot(rep), render.timeline_plot(rep),
            render.gait_diagram(rep), render.angle_plot(rep), rep.strides.round(3), str(csv))


def test_trials():
    """A handful of held-out test trials for the demo, spread over how asymmetric they are."""
    if "trials" in _cache:
        return _cache["trials"]
    test = load_all()["test"]
    picked = []
    for ds, mode, n in (("fukuchi", "treadmill", 4), ("fukuchi", "overground", 2), ("ferber", "treadmill", 5),
                        ("toronto", "overground_pass", 3)):
        # Fukuchi's two slowest treadmill speeds (T01/T02) have known force-plate errors in their labels
        cand = [t for t in test if t.ds == ds and t.mode == mode and len(t.strides) >= 4
                and not (ds == "fukuchi" and t.trial[-3:] in ("T01", "T02"))]
        si = [abs(aggregate(t.strides, 0, len(t.pose) / FS)[TI["SI_step_time"]]) for t in cand]
        order = np.argsort(si)
        picked += [cand[order[i]] for i in np.unique(np.round(np.linspace(0.3, 0.9, n) * (len(cand) - 1)).astype(int))]
    labels = {}
    for t in picked:
        si = aggregate(t.strides, 0, (len(t.pose) - 1) / FS)[TI["SI_step_time"]]
        labels[f"{t.ds} · {t.mode} · {t.subject} · {t.trial} · {len(t.pose) / FS:.0f} s · "
               f"{t.speed_ll * t.leg_length:.2f} m/s · step SI {si:+.1f} %"] = t
    _cache["trials"] = labels
    return labels


def analyse_trial(choice, sigma, progress=gr.Progress()):
    if not choice:
        raise gr.Error("Pick a held-out test trial.")
    tr = test_trials()[choice]
    x, vis = augment.apply(tr.pose, np.random.default_rng(0), augment.Noise.level(float(sigma)))
    rep = analyzer().from_canonical(x, vis, [(0, len(x), 1)], leg_length_m=tr.leg_length, sigma=float(sigma))
    g = aggregate(tr.strides, 0, (len(tr.pose) - 1) / FS, tr.speed_ll)
    gold = dict(zip(TARGET_NAMES, g)) | {"speed_ms": tr.speed_ll * tr.leg_length}
    progress(0.6, desc="rendering")
    vid = render.stick_video(rep, str(OUT / f"{tr.trial}_s{sigma}.mp4"))
    extra = (f"**Input:** motion-capture pose of a held-out person ({tr.ds}, {tr.subject}) seen from the side, with "
             f"simulated video-pose noise σ = {float(sigma):.2f}. Gold values: "
             f"{'force-plate' if tr.event_source == 'grf' else 'marker-based'} events and mocap angles.")
    return (vid, quality_text(rep, extra), summary_table(rep, gold), render.symmetry_plot(rep), render.gait_diagram(rep),
            render.angle_plot(rep), rep.strides.round(3))


ABOUT = f"""
### What it measures
From a side-on walking clip: cadence, walking speed, stride / step / stance / swing time, double support, hip / knee /
ankle range of motion, peak knee flexion, stride-time variability, and the left/right symmetry index of each
(100 × (L − R) / mean; 0 = symmetric, > 0 = left larger).

### How it works
1. **Pose**: a person detector (YOLOX) and a pose model (RTMPose, with heels and toes) find the joints in every frame.
2. **Clean-up**: left/right mix-ups are fixed, the clip is split into walking passes, and the body is centred and
   scaled to leg length.
3. **Model**: a BiGRU network outputs, for every frame, the chance that a heel lands or a toe lifts, and the hip,
   knee and ankle angles.
4. **Formulas**: every number is computed from those events and angles with the same definitions a gait lab uses.

### Validated on (held-out people)
Gait-lab recordings with simulated video noise (Fukuchi, Ferber, Toronto older adults), and **real video** (Kuopio,
24 people, against motion capture and force plates): heel strike ±16 ms, step time ±17 ms, speed ±0.04 m/s, knee
swing range ±6–7°. The model was also trained on real pose-estimator mistakes from 18 other Kuopio people.

### Known limits
- The error column uses simulated noise σ = 0.04, the level real video matched; timing on real video was a bit worse.
- Joint-motion symmetry is not reliable on real video yet (hip and ankle no better than guessing, knee only slightly).
- This demo's pose model (RTMPose) has not itself been checked against a gait lab; the real-video test used OpenPose.
- Stance/swing-time asymmetry is under-reported by ~40 %; step-time asymmetry by ~10–25 %.
- Side view only (no sideways sway). One person, straight walkway or treadmill.

### Recording tips
Camera at hip height, side-on, whole body including feet in frame, ≥ 20 s of steady walking, across and back.

Demo video: Yoon J, Park H-S, Damiano DL (2012), J NeuroEng Rehabil 9:62, CC BY 2.0, via Wikimedia Commons.

**{DISCLAIMER}**
"""


def build():
    examples = sorted(DEMO_DIR.glob("*.mp4"))
    with gr.Blocks(title="Gait measurement prototype") as demo:
        gr.Markdown(f"# Gait measurement from a side-on video\nSymmetry, timing and joint range of motion from one "
                    f"cheap clip, for tracking a patient's progress, not for diagnosis.  \n**{DISCLAIMER}**")
        with gr.Tab("Analyse a walking video"):
            with gr.Row():
                with gr.Column(scale=1):
                    v_in = gr.Video(label="Side-on walking video", sources=["upload"])
                    height = gr.Number(label="Patient height (cm), for speed in m/s (optional)", precision=0)
                    leg = gr.Number(label="…or leg length, hip joint to ankle (cm) (optional)", precision=0)
                    with gr.Accordion("Advanced", open=False):
                        conf = gr.Slider(0.1, 0.8, value=0.3, step=0.05, label="Keypoint confidence threshold")
                        max_s = gr.Slider(5, 90, value=60, step=5, label="Analyse at most (seconds)")
                    go = gr.Button("Analyse", variant="primary")
                    if examples:
                        gr.Examples([[str(p), None, None] for p in examples], inputs=[v_in, height, leg],
                                    label="Example (CC BY 2.0): treadmill at 0.5 → 1.0 → 1.5 → 1.0 → 0.5 m/s; height "
                                          "unknown, so speed is in leg lengths/s")
                with gr.Column(scale=2):
                    v_out = gr.Video(label="Detected pose and gait phase", autoplay=True)
                    q_md = gr.Markdown()
            table = gr.Dataframe(label="Measurements", wrap=True, value=pd.DataFrame(columns=COLUMNS))
            plots = [gr.Plot(label=t) for t in ("Left vs right", "Over the recording", "Gait diagram", "Joint angles")]
            with gr.Accordion("Per-stride values (the strides behind every number)", open=False):
                strides = gr.Dataframe()
            csv = gr.File(label="Download measurements (CSV)")
            go.click(analyse_video, [v_in, height, leg, conf, max_s], [v_out, q_md, table, *plots, strides, csv])

        with gr.Tab("Held-out test trial (mocap)"):
            gr.Markdown("A trial from a person the model never saw, as side-view pose from motion capture with "
                        "optional simulated video noise; the gold-standard values are shown next to the model's.")
            with gr.Row():
                pick = gr.Dropdown(choices=list(test_trials()), label="Held-out trial", scale=3)
                sig = gr.Slider(0, 0.04, value=0.02, step=0.01, label="Simulated pose noise σ (leg lengths)", scale=1)
                go2 = gr.Button("Run", variant="primary", scale=1)
            with gr.Row():
                t_vid = gr.Video(label="Model input (stick figure) and decoded gait phase", autoplay=True)
                t_md = gr.Markdown()
            t_table = gr.Dataframe(label="Model vs gold standard", wrap=True,
                                   value=pd.DataFrame(columns=COLUMNS[:5] + ["Gold (mocap)"] + COLUMNS[5:]))
            t_plots = [gr.Plot() for _ in range(3)]
            with gr.Accordion("Per-stride values", open=False):
                t_str = gr.Dataframe()
            go2.click(analyse_trial, [pick, sig], [t_vid, t_md, t_table, *t_plots, t_str])

        with gr.Tab("How it works & limits"):
            gr.Markdown(ABOUT)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build().queue().launch(server_port=a.port, share=a.share, allowed_paths=[str(OUT), str(DEMO_DIR)], show_error=True)
