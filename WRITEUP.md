# Gait measurement from a single side-on walking video

**Track:** AI + Health (brief 4.1.4, own idea). **User:** a physiotherapist in a clinic with no gait
lab, tracking a patient's recovery. **Framing:** measurement and progress tracking, not diagnosis.

*This project is a prototype and is not medical advice.*

## 1. Problem

Gait labs (motion capture + force plates) measure what a clinician cannot see: a 7 % left/right
timing asymmetry, a 4° loss of knee range of motion, a change in stride-time variability. Most clinics
don't have one. A phone on a tripod films the same walk, so the question is how much of the
gait-lab measurement can be recovered from one sagittal 2D video, and how well, with honest error bars.

## 2. What we built

From a side-on walking clip → 2D pose → a BiGRU that outputs per-frame gait events and joint-angle
curves → the same stride formulas used to make the gold-standard labels. Outputs:

| Group | Outputs |
|---|---|
| Spatiotemporal | cadence, stride / step / stance / swing time, stance %, double support, walking speed |
| Symmetry (SI = 100·(L−R)/mean) | step-, stance- and swing-time SI; hip-, knee- and ankle-ROM SI |
| Kinematics | hip / knee / ankle sagittal range of motion, peak knee flexion, full angle curves |
| Variability | stride-time SD and CV |

The Gradio demo (`src/app.py`) has three tabs:
- **Analyse a walking video.** Pose, L/R repair, per-leg measurements, an annotated video with the gait phase drawn on, a gait diagram, angle curves, left-vs-right plots with every stride, and a per-output reliability flag.
- **Held-out test trial (mocap).** Runs the model on a test trial and shows the gold labels side by side.
- **How it works & limits.**

## 3. What we added (own ideas)

1. **Pose-noise augmentation and degradation curves.** The model is trained on clean mocap with simulated video pose error: jitter, constant keypoint offsets, dropout, a noisier far leg that disappears behind the near leg, leg-length scale error and camera tilt. It is reported at noise levels σ = 0…0.08 leg lengths.
2. **Measure, don't regress.** Pooled regression heads shrank every asymmetry towards zero. v2 predicts dense per-frame events and angle curves instead, and computes every output with the label formulas, so a model output and a label mean the same thing.
3. **Gated geometric skip (v3).** The angle head blends a learned curve with the angle computed directly from the keypoints, through a learned per-frame gate. On clean pose it reproduces exact geometry (angle RMSE 0.05–0.25°); on noisy pose it falls back to the learned curve.
4. **Synthetic asymmetry (v4).** We diagnosed that the model had learned "the legs are half a cycle apart" from a near-symmetric training set. Fix: per-leg time shifts and rigid per-joint amplitude scaling, with every label recomputed. This raised the step-time SI slope from 0.65 to 0.86 (Ferber, σ = 0.02).
5. **Recording-level evaluation and the across-and-back protocol.** Strides are pooled over a whole recording, and the patient is filmed walking both ways. On mocap + noise, measuring each leg from its near-camera pass helped slightly. On real Kuopio video, pooling all strides was better (section 6).
6. **Video pipeline.** YOLOX + RTMPose (Halpe-26, with heels and toes) on CPU; Viterbi left/right label repair; facing-direction pass splitting; overlapping windows with tapered blending; and a pose-quality estimate mapped to each output's validated error.
7. **External validation on real video keypoints (Kuopio).** Geometry-only video-to-mocap synchronisation, with force plates kept as an independent check (section 6).

## 4. How it works

**Data.** Four public datasets, each in its own directory. All subjects are split at subject level
(no subject in two splits); results are reported per dataset.

| Dataset | Subjects (kept trials) | Ground truth | Role |
|---|---|---|---|
| Fukuchi 2018 WBDS | 51 (1972) | markers 150 Hz, treadmill force plates | train / val / test |
| Ferber Running Injury Clinic, walking | 1670 (2061) | marker clusters 200/120 Hz, Zeni events | train / val / test |
| Toronto Older Adults | 11 (48) | Xsens IMU motion capture | external test (older adults) |
| Kuopio | 42 (817 video clips) | Vicon 100 Hz + 3 force plates + **OpenPose on real video** | external test on real video |

Splits: train 1205 / val 258 / test 269 subjects (Toronto test-only).

**Canonical skeleton → model input.** Each loader maps its marker set to
`[L/R hip, knee, ankle, heel, toe, pelvis]`. Joint centres use Harrington hip regression (Fukuchi),
Kabsch-tracked static joint centres with a virtual foot where markers are missing (Ferber, ~40 % of
sessions), or functional SCoRE centres (Kuopio). The joints are projected onto the camera's sagittal
plane (treadmill belt axis or PCA of pelvis travel), pelvis-centred, divided by leg length,
resampled to 30 Hz and cut into 5 s windows.

**Labels.** Events come from force plates where available, otherwise from Zeni's coordinate
algorithm, corrected by its median bias against force plates (HS −34 ms, TO +31 ms). Sagittal hip,
knee and ankle angles come from the joint coordinates. Per-stride formulas give timing and ROM;
per-window aggregation gives SI, CV and speed.

**Model (v4).** The input is a 74-feature frame vector: coordinates, velocities, segment sin/cos,
geometric angles and validity flags. It goes through a 2-layer BiGRU (hidden 128) plus a local 2-layer
convolution branch. Outputs:
- an event head: per-frame HS/TO probabilities per foot;
- a gated angle head: 6 curves;
- pooled heads, used only for speed and as a fallback.

Decoding works in four steps:
1. Peaks are taken with sub-frame parabolic refinement.
2. Strides are formed from heel strike to heel strike.
3. `gaitdata.derive` computes the outputs.
4. Values are pooled over the recording.

Training data comes from three sources: pose noise (random level per window), the L/R mirror
(p = 0.5) and synthetic asymmetry (p = 0.5). Training ran 50 epochs on CPU (about 2.6 h).

**Baselines** (required by the brief): a mean predictor, and a classic pipeline on the same input (gap
filling, 6 Hz low-pass, Zeni events, the same formulas).

## 5. Results on held-out subjects (mocap + simulated pose noise)

Per 5 s window, σ = 0.02 ("realistic video" level), MAE (lower is better):

| Output | Ferber v4 / classic / mean | Fukuchi v4 / classic / mean | Toronto v4 / classic / mean |
|---|---|---|---|
| Heel-strike timing (ms) | **3.5** / 11.3 / – | **7.5** / 14.6 / – | **5.5** / 14.0 / – |
| Cadence (steps/min) | **0.18** / 0.37 / 6.3 | **0.46** / 0.87 / 19.3 | **0.31** / 0.44 / 7.6 |
| Speed (m/s) | **0.03** / 0.13 / 0.12 | **0.03** / 0.10 / 0.35 | 0.05 / 0.05 / 0.28 |
| Stance (% of stride) | **0.33** / 0.69 / 1.08 | **1.02** / 1.53 / 2.88 | **0.84** / 1.13 / 1.43 |
| Knee ROM (°) | **1.8** / 4.1 / 4.4 | **1.8** / 5.4 / 8.9 | **3.1** / 4.7 / 17.0 |
| Hip ROM (°) | **1.3** / 2.0 / 3.9 | **1.2** / 2.7 / 6.0 | **1.2** / 2.6 / 8.6 |
| Ankle ROM (°) | **2.2** / 5.5 / 3.2 | **2.3** / 10.9 / 3.7 | 10.1 / **8.1** / 11.0 |
| Step-time SI (%) | **1.2** / 3.1 / 2.1 | **3.2** / 5.4 / 3.8 | **2.3** / 4.0 / 3.2 |
| Knee-ROM SI (%) | **3.8** / 6.4 / 4.9 | **3.7** / 8.5 / 4.3 | **5.7** / 7.6 / 10.5 |
| Ankle-ROM SI (%) | 10.2 / 17.9 / 11.2 | 10.3 / 26.5 / 9.6 | 23.3 / 22.4 / 28.0 |

Event errors on Ferber and Toronto are measured against Zeni-derived labels; Fukuchi treadmill and
Kuopio are measured against force plates.

**Per recording** (≈30 s, strides pooled, σ = 0.02), Ferber step-time SI: MAE 0.82 (mean predictor
1.77), slope 0.89. **Symmetry slope** (1 = an asymmetry is reported at its true size), step-time SI at
σ = 0.02: v1 0.51 → v4 0.86 (Ferber), 0.33 → 0.73 (Toronto). The ablation for each step is in
`src/model_findings.md` and `notebooks/06_model_results.ipynb`, together with Bland–Altman plots and
age and sex subgroups.

## 6. Real-video validation (Kuopio)

No real video keypoints were used in training or in the tests above, so Kuopio is the first check of
the whole idea on real pose. Every Kuopio subject is held out, and the setup differs from training in four ways:
- a different lab and camera;
- a different pose estimator (OpenPose BODY_25, not the demo's RTMPose);
- barefoot, overground walking;
- a different event reference (three floor force plates).

**Setup.** There are 817 sagittal clips (about 1.05 s each, roughly one stride) from 42 subjects,
around 20 per subject, half walking each way past a fixed camera. 73 clips were dropped for gaps in
the mocap. The dataset gives neither the video frame rate nor the clip start times, so we recovered
both from geometry alone:
- One camera per subject (scale and offset) is shared by all that subject's clips, plus one start
  time per clip.
- The fit minimises the distance between L/R-labelled OpenPose hip/knee/ankle keypoints and the
  projected mocap joint centres. The labels matter: they resolve the one-step ambiguity (on subject 01:
  median error 14 px for the chosen alignment against 52 px for the one a step off).
- The frame rate came out at 97 fps across 12 subjects.

The force plates were not used, which makes them an independent check. In both walking directions,
the first plate heel strike falls a median 0.254 s after the clip start (IQR 0.248–0.262 s), a spread of
only 14 ms over 817 independently synced clips.

Each clip goes through the deployed pipeline steps (30 Hz, L/R swap repair, facing direction, mid-hip
origin, leg-length scale, v4), and also through the classic pipeline. Both run on the same frames
under four inputs: real OpenPose, clean projected mocap, and mocap plus our training noise at σ = 0.02
and σ = 0.04.

| Real video (OpenPose), per clip | Model | Classic | Mean | Model on clean mocap input |
|---|---|---|---|---|
| Heel strike vs force plate: MAE (bias, SD), recall | **20 ms** (+6, 29), 96 % | 24 ms (+11, 31), 96 % | – | 13 ms (+8, 14) |
| Toe-off vs force plate: MAE (bias, SD), recall | 35 ms (−31, 27), 93 % | 34 ms (−29, 30), 92 % | – | 35 ms (−34, 12) |
| Stance time of the plate foot (ms), r | **33**, 0.92 | 38, 0.90 | 50 | 44 (bias −44), 0.99 |
| Step time (ms), r | **26**, 0.82 | 27, 0.80 | 40 | 9, 0.96 |
| Walking speed (m/s), r | **0.075**, 0.84 | 0.48, 0.15 | 0.134 | 0.063 |
| Hip / knee / ankle excursion per clip (°) | **4.1 / 5.7 / 3.8** | 5.7 / 8.3 / 13.5 | – | 0.6 / 0.5 / 0.4 |
| Knee-ROM SI per subject (%), r | **4.2**, 0.56 | 10.2, 0.20 | 5.1 | 0.7, 0.99 |
| Hip-ROM SI per subject (%), r | 3.7, 0.43 | 9.5, 0.36 | **3.2** | 1.6, 0.94 |
| Ankle-ROM SI per subject (%), r | 10.1, 0.02 | 29.8, 0.03 | **8.7** | 1.7, 0.99 |

![Kuopio validation](outputs/figures/07_kuopio.png)

**What this shows.**
1. **The model transfers to real keypoints from an unseen camera, estimator and lab.** It beats the
   classic pipeline on every measure except toe-off timing, which is a tie. The classic pipeline breaks down on real pose: speed MAE is
   0.48 m/s and ankle-excursion error is 13.5°. This is the brief's central claim, now on real video
   rather than simulated noise: the learned model earns its place by holding up on real 2D pose.
2. **Real pose error is not our "realistic" σ = 0.02.**
   - The per-frame jitter of real OpenPose matches the noise model: robust SD 0.02–0.035 leg lengths.
   - The rest does not. Real error is strongly correlated in time (lag-1 r ≈ 0.6–0.7 at 30 Hz).
   - It has larger constant offsets at the ankle and foot (0.05–0.07 leg lengths, partly because a
     keypoint marks the skin and not the joint centre).
   - 4–7 % of knee and foot frames have gross errors (> 0.2 leg lengths).
   - Angle and ROM-SI errors on real video sit at or beyond the simulated σ = 0.04 level (last panel).
     The demo's reliability flags assumed σ = 0.02, so they are optimistic.
3. **Absolute stance and swing times carry a reference offset.** On clean mocap input, the model already
   puts heel strike 8 ms late and toe-off 34 ms early against Kuopio's force plates, so stance comes out
   44 ms short. That comes from defining events differently (barefoot overground walking and these
   foot markers, against our training labels); video does not cause it. Real video adds spread (SD
   14 → 29 ms for heel strike) but almost no extra bias. A symmetry index cancels an offset shared by
   both legs; absolute stance/swing times need per-setting calibration.
4. **Symmetry on real video.** Knee-ROM SI is informative per subject: about 20 single-stride clips give
   MAE 4.2 against 5.1 for the mean predictor, with r 0.56. Hip-ROM SI is no better than the mean, and
   ankle-ROM SI carries no information. Timing symmetry could not be tested, because each clip holds one
   complete stance and the same foot hits the middle plate in both directions.
5. **Near-leg-only measurement did not help here.** OpenPose errors on Kuopio's camera are about the
   same for the near and far leg, so dropping half the clips only added noise. Walking both ways still
   helps by averaging opposite near/far offsets, but pooling every stride beats near-leg-only.
6. **Frame-rate sensitivity.** If the clips were really 100 fps, not the geometric estimate of 97 fps,
   the sync fits slightly worse (12.2 vs 11.7 px). Heel strike becomes 23 ms, step time 22 ms and
   knee-ROM SI 4.3, much the same; but stance time worsens to 52 ms and speed to 0.105 m/s, because
   both scale with the frame rate. Every conclusion above holds at either rate
   (`outputs/kuopio/fps100/`).

## 7. Where it fails

- **Real pose error is larger and more structured than the training noise model** (Kuopio). Angle
  and ROM-SI accuracy on real OpenPose sits at or beyond the simulated σ = 0.04 level, and knee
  excursion is under-read by about 5°. The demo's reliability floor (σ = 0.02) is optimistic.
- **Hip- and ankle-ROM symmetry on real video** were no better than the mean predictor, from about 20
  one-stride clips per subject. Knee-ROM SI was informative but compressed (slope 0.44).
- **Stance- and swing-time SI** are compressed on mocap + noise (slope ≈ 0.6). Part of this is label
  noise: toe-off labels are bias-corrected Zeni.
- **Ankle-ROM SI** is near the mean predictor at video-like noise for every method. About 6° of per-frame
  foot-angle noise sits against a 3° left/right difference. The demo flags it as unreliable.
- **Far leg.** In a one-direction recording the far leg is measured worse; the across-and-back protocol
  mitigates this.
- **By design:** frontal-plane quantities (pelvic drop, knee adduction) are absent with a single
  sagittal view. Toronto's own video is frontal, so it could not be used for video validation.
- **Population.** Training subjects are healthy adults and runners (Ferber), and testing adds older
  adults (Toronto). Pathological gait (stroke, Parkinson's, amputation) is not covered by any training
  set.

## 8. Next steps

1. Replace the hand-set noise model with real pose-error statistics: temporally correlated error, larger
   foot offsets and gross outliers, fitted on Kuopio's paired data. Fine-tune on a subject-split part of
   Kuopio, keeping the rest for testing. Raise the demo's reliability floor to the measured real level, and
   pool both walking directions instead of using near-leg-only.
2. Stance/swing time-warp augmentation and a sub-frame event-offset head (the timing-SI ideas).
3. Stride-averaged ROM curves, mirror test-time augmentation, and a de-shrink calibration for step-time SI.
4. 60 fps capture and a foot-specialised keypoint model for ankle measurements.
5. A pathological-gait test set before any clinical pilot.

## 9. Project log (what was done, in order)

1. **Data.** Downloaded Fukuchi, Ferber (walking subset), Toronto; built canonical skeletons, sagittal
   projection, labels and QC filters. Notebooks 01–05 cover the data overview and filter funnel, skeleton
   QC, events against force plates (Zeni bias), joint kinematics (validated against the datasets' own
   angles), and Toronto pose and clinical scores.
2. **Environment.** A Nix flake with a pinned nixpkgs and a Jupyter kernel; `./py` wrapper.
3. **Model v1.** A BiGRU with pooled heads and pose-noise augmentation, with baselines and degradation
   curves. It held up under noise but shrank asymmetries toward zero.
4. **Gap analysis.** What blocked a submittable entry: symmetry shrinkage, broken ankle ROM, no
   real-video path.
5. **v2 → v4.** Dense heads with formulas, L/R mirror, gated geometric angles, local conv branch,
   synthetic asymmetry. Ablated at every step; recording-level and two-pass evaluation added
   (`src/model_findings.md`, notebook 06).
6. **Video pipeline and demo.** YOLOX + RTMPose on ONNX/CPU, Viterbi L/R repair, pass splitting,
   overlapping windows, quality calibration, Gradio app. Checked on a real treadmill speed-ramp clip:
   cadence and speed track the belt.
7. **Kuopio real-video validation.** Selective download, a C3D reader, geometry-only sync, and an
   evaluation on 890 real OpenPose clips (section 6).

## 10. Sources and credits

- Fukuchi CA, Fukuchi RK, Duarte M (2018). A public dataset of overground and treadmill walking
  kinematics and kinetics in healthy individuals. PeerJ 6:e4640. Figshare 10.6084/m9.figshare.5722711, CC BY.
- Ferber R et al. Running Injury Clinic kinematic dataset. Figshare 24255795 (Sci Data 2024,
  10.1038/s41597-024-04011-7), CC BY 4.0.
- Toronto Older Adults Gait Archive (Sci Data 2022, 10.1038/s41597-022-01495-z), CC BY.
- Lavikainen J et al. Kuopio gait dataset: motion capture, inertial measurement and video-based
  sagittal-plane keypoint data from walking trials. Zenodo 10.5281/zenodo.10559504, CC BY 4.0.
- Zeni JA et al. (2008) Two simple methods for determining gait events during treadmill and overground
  walking using kinematic data. Gait Posture 27:710–714. Harrington ME et al. (2007) J Biomech 40:595–602.
  Robinson RO et al. (1987) symmetry index.
- Pose models: RTMPose, YOLOX (OpenMMLab MMPose / MMDetection, Apache-2.0); OpenPose (Kuopio's keypoints).
- Libraries: PyTorch, NumPy, SciPy, pandas, matplotlib, OpenCV, ONNX Runtime, Gradio, FFmpeg, Nix.
- Demo video: Yoon J, Park H-S, Damiano DL (2012), J NeuroEng Rehabil 9:62, CC BY 2.0.

*This project is a prototype and is not medical advice.*
