# The complete guide to this codebase

Everything about how this project works, from the raw datasets to the demo, and how to explain it to
different audiences. Read the parts you need; the table of contents is ordered from "big picture" to "every
constant".

*This project is a prototype and is not medical advice.*

## Contents

1. [The project in one minute](#1-the-project-in-one-minute)
2. [Repository map](#2-repository-map)
3. [End-to-end flow](#3-end-to-end-flow)
4. [Conventions: coordinates, units, names](#4-conventions-coordinates-units-names)
5. [The data](#5-the-data)
6. [Labels: how the "right answers" are made](#6-labels-how-the-right-answers-are-made)
7. [File formats](#7-file-formats)
8. [The model](#8-the-model)
9. [Training and data augmentation](#9-training-and-data-augmentation)
10. [Turning model outputs into measurements](#10-turning-model-outputs-into-measurements)
11. [Evaluation](#11-evaluation)
12. [The video pipeline and the demo](#12-the-video-pipeline-and-the-demo)
13. [Real-video validation (Kuopio)](#13-real-video-validation-kuopio)
14. [Environment and commands](#14-environment-and-commands)
15. [Key results cheat sheet](#15-key-results-cheat-sheet)
16. [How to explain it to people](#16-how-to-explain-it-to-people)
17. [Questions you will get, and answers](#17-questions-you-will-get-and-answers)
18. [Cookbook: changing things](#18-cookbook-changing-things)
19. [Glossary](#19-glossary)

---

## 1. The project in one minute

**Problem.** After an injury or operation, a physiotherapist wants to track whether a patient walks better
over time: how symmetric the two legs are, how long each foot stays on the ground, how far the knee bends.
Gait labs measure this with marker-based motion capture and force plates; most clinics don't have one.

**What we built.** A tool that takes an ordinary side-on video of someone walking and outputs gait-lab-style
numbers: cadence, speed, stride/step/stance/swing time, double support, hip/knee/ankle range of motion, peak
knee flexion, stride-time variability, and the left/right symmetry of each.

**How.**

1. A pose model finds hips, knees, ankles, heels and toes in each frame.
2. A small neural network (a BiGRU) reads 5 s of those points. For every frame it outputs the chance that a heel
   lands or a toe lifts, plus a cleaned-up hip, knee and ankle angle.
3. Ordinary gait-lab formulas turn those events and angles into the numbers.

**The trick.** Nobody has thousands of videos with gait-lab answers. So the model was trained on gait-lab
recordings (≈1,700 people) that were deliberately damaged to look like video: shaky points, missing points, a
hidden far leg, a tilted camera. Synthetic limps were added so it learns that legs can differ.

**Validation.** On people the model never saw:

- **Lab data with simulated video noise:** the model beats the textbook "classic" method and guessing the
  average.
- **Real video from a different lab (Kuopio, 42 people, never used for training):** heel strikes are found to
  about 20 ms, speed to about 0.08 m/s, and knee range of motion to about 6°. Real video turned out noisier
  than the simulation, which is the main limitation.

---

## 2. Repository map

```
gaitanalysis/
├── README.md              short entry point
├── GUIDE.md               this file
├── WRITEUP.md             hackathon write-up (problem, results, limits, project log)
├── REPRODUCE.md           every command from download to demo, with expected checks
├── PROJECT.MD             the original brief
├── flake.nix, flake.lock  Nix environment (pinned Python 3.14 + all libraries)
├── py                     wrapper: runs Python inside the Nix shell with src/ on the import path
├── notebooks/
│   └── gait_walkthrough.ipynb   the whole project explained for non-specialists, with figures
├── src/                   the model and everything that runs it
│   ├── model.py           network, input features, event decoding
│   ├── augment.py         video-like noise, left/right mirror, synthetic limps
│   ├── train.py           training loop
│   ├── classic.py         the no-learning baseline method
│   ├── evaluate.py        per-window evaluation (+ shared: load_model, run_model, metrics)
│   ├── evaluate_trials.py per-recording evaluation (feeds the demo's error table)
│   ├── evaluate_kuopio.py real-video test
│   ├── pose_video.py      video -> 2D keypoints (YOLOX + RTMPose, ONNX)
│   ├── video_pipeline.py  keypoints -> model -> report (the demo's brain)
│   ├── app.py             Gradio demo
│   ├── model_findings.md  how the model evolved v1 -> v4, ablation numbers
│   └── helpers/           data loading, processing, plotting
│       ├── download_data.py   fetch the 4 datasets
│       ├── load_fukuchi.py    Fukuchi loader (markers, force plates)
│       ├── load_ferber.py     Ferber loader (marker clusters, Kabsch)
│       ├── load_toronto.py    Toronto loader (Xsens suit)
│       ├── load_kuopio.py     Kuopio loader + video/mocap sync
│       ├── c3d.py             tiny C3D motion-capture file reader
│       ├── canonical.py       our skeleton: projection, filtering, events, angles
│       ├── build_dataset.py   raw -> processed dataset (labels, quality checks)
│       ├── build_kuopio.py    Kuopio clips synced -> kuopio_clips.pkl
│       ├── gaitdata.py        splits, Trial objects, windows, labels, stride formulas
│       ├── analysis.py        paths, colour palette, plot style, gait-cycle curves
│       ├── render.py          demo figures, annotated video, stick-figure video
│       └── plot_kuopio.py     the real-video validation figure
├── data/                  (not in git) raw datasets + data/processed/
└── outputs/               (not in git) models, evaluations, figures, pose models
```

**Rule of thumb when explaining the layout:** `src/` is "the AI and how it's used"; `src/helpers/` is "getting
data in shape and drawing pictures".

---

## 3. End-to-end flow

```
              TRAINING (offline)                                         USE (demo)
download_data ─► load_* + c3d ─► canonical ─► build_dataset              video.mp4
                                                  │                          │
                                                  ▼                          ▼ pose_video (YOLOX + RTMPose)
                                   data/processed/<ds>/*.npz,parquet    keypoints [T, 26, 2]
                                                  │                          │
                                     gaitdata.load_all (Trials, splits)      ▼ video_pipeline
                                                  │                    30 Hz, L/R fix, passes, normalise
                     augment (noise, mirror, limps)│                          │
                                                  ▼                          ▼
                                        train.py ─► model.pt ─────────► model (5 s windows, overlapping)
                                                  │                          │
                       evaluate.py / evaluate_trials.py                      ▼ decode events, formulas
                                                  │                    report table + plots + video
                           outputs/eval/*.csv ────┴──► typical-error column in the demo
```

Real-video validation is a side branch: `load_kuopio` → `build_kuopio` (sync) → `evaluate_kuopio` →
`plot_kuopio`.

---

## 4. Conventions: coordinates, units, names

| Thing | Convention |
|---|---|
| **Joints** | 11 canonical joints `L_hip, L_knee, L_ankle, L_heel, L_toe, R_hip, R_knee, R_ankle, R_heel, R_toe, pelvis` (`canonical.JOINTS`). The model sees the first 10 (`gaitdata.IN_JOINTS`); pelvis is the origin. |
| **Pelvis** | Mid-point of the two hips (mocap: mean of hip joint centres; video: mean of hip keypoints). |
| **Plane** | 2D sagittal (side) view: `x` = forward along the walking direction, `y` = up. Always walking towards +x. |
| **Units of the model input** | Leg lengths: positions are pelvis-centred and divided by the median hip-to-ankle distance. 0.01 ≈ 8 mm on an adult. This makes the input independent of camera distance and body size. |
| **Frame rate** | Everything model-side is 30 Hz (`gaitdata.FS`). Raw data is 100–200 Hz; video is resampled. |
| **Window** | 5 s = 150 frames (`W`). |
| **Events** | `HS_L, HS_R, TO_L, TO_R`: heel strike (foot lands) and toe-off (foot leaves), left/right. Stored in seconds. |
| **Angles** (`canonical.sagittal_angles`, degrees) | hip = thigh vs vertical, forward positive (no pelvis tilt available in 2D); knee = thigh angle − shank angle, flexion positive; ankle = foot (heel→toe) angle − shank angle, dorsiflexion (toes up) positive. Order `L_hip, L_knee, L_ankle, R_hip, R_knee, R_ankle` (`ANG_NAMES`). |
| **Symmetry index** | `SI = 100 × (L − R) / ((L + R) / 2)` in %. 0 = symmetric; positive = left larger. |
| **Speed** | Model predicts leg lengths per second (`speed_ll`); m/s = `speed_ll × leg length (m)`. |
| **Sides in arrays** | `0 = L`, `1 = R` in stride tables. |

---

## 5. The data

### 5.1 The four datasets

| Dataset | Who | Raw format | Ground truth | Role |
|---|---|---|---|---|
| **Fukuchi 2018 (WBDS)** | 51 healthy adults (young and older); treadmill at 8 speeds + short overground walks | ASCII marker files, 150 Hz, mm, Y up | markers + **treadmill force plates** (gold events) | train / val / test |
| **Ferber Running Injury Clinic** | 1,670 clinic patients, walking on a treadmill | JSON per session, marker clusters, 200 or 120 Hz, mm, Y up | markers; events from Zeni's algorithm | train / val / test (most of the data) |
| **Toronto Older Adults** | 11 older adults (mean age ~86) | Xsens inertial suit CSV, 100 Hz, m, Z up | Xsens segment positions; Zeni events | test only (external) |
| **Kuopio** | 51 adults (42 usable) filmed on video during mocap | Vicon C3D, 100 Hz + OpenPose JSON per video frame + 3 force plates | Vicon functional joint centres + force plates | real-video test only |

### 5.2 How each loader builds the skeleton

Every loader returns `xyz [T, 11, 3]` in metres in the lab frame, in `JOINTS` order.

- **Fukuchi** (`load_fukuchi.py`)
  - Hip joint centres: Harrington 2007 regression from the four pelvis markers.
  - Knee and ankle: the lateral epicondyle and malleolus markers (the medial–lateral offset disappears in the
    side view anyway).
  - Heel marker; toe = mean of the 1st and 5th metatarsal markers.
  - Gold events from the treadmill force plates: vertical force > 30 N, flicker under 50 ms merged, stance
    phases > 200 ms. Each belt is matched to a foot by its centre-of-pressure position.
- **Ferber** (`load_ferber.py`)
  - During walking only rigid 4-marker clusters are on the segments; anatomical markers exist only in the
    standing calibration trial.
  - Joint centres are defined in the standing trial (hip = trochanter marker + 25 % towards the other one;
    knee/ankle = mid-points of the epicondyles/malleoli, as in the dataset's own MATLAB code).
  - These are carried into every walking frame by a per-frame rigid fit of each segment's cluster (**Kabsch**
    algorithm).
  - About 40 % of sessions have no toe markers and get a virtual toe (0.122 × body height in front of the
    ankle).
  - Sessions with a high cluster-fit residual are rejected.
- **Toronto** (`load_toronto.py`)
  - Xsens segment origins map to joints: upper leg → hip, lower leg → knee, foot → ankle, toe → toe.
  - There is no heel point, so a virtual heel is rigidly attached 5 cm behind the ankle at floor level.
  - The ~60 s recording is split into straight passes: pelvis speed > 0.4 m/s, turning < 0.6 rad/s, at least
    4 s long, 0.5 s trimmed at each end.
- **Kuopio** (`load_kuopio.py`)
  - Vicon exports functional joint centres as virtual markers (SCoRE hip, knee and ankle).
  - Heel = rear foot marker; toe = mean of the two forefoot markers.
  - Four subjects lack the ankle markers and are skipped.
  - Force plates 1–3: contact when vertical force > 20 N for > 0.2 s, assigned to the lowest ankle over the
    plate.

### 5.3 From 3D lab data to the model's 2D input (`build_dataset.process`)

1. Drop trials with > 1 % missing frames. For long treadmill or Toronto trials, the longest clean stretch is
   kept if it is ≥ 10 s.
2. **Filter**: 4th-order zero-lag Butterworth low-pass at 6 Hz (`canonical.lowpass`).
3. **Walking direction**: treadmill = the mean heel→toe direction snapped to the nearest lab axis;
   overground = main direction of pelvis travel (PCA).
4. **Project** onto the side plane (forward, up): `canonical.sagittal_project`.
5. **Angles** from the projected joints.
6. **Normalise**: subtract the pelvis per frame, divide by the median hip–ankle distance (`canonical.normalise`).
7. **Events**: force plates if available, else Zeni (section 6).
8. **Strides and labels** with `gaitdata.strides_from_events` and `aggregate`.
9. **Resample to 30 Hz** (linear interpolation).
10. **Quality checks** (`QC` in `build_dataset.py`). Rejections are recorded in `reject_reason`:
    - leg length outside 0.60–1.10 m;
    - speed outside 0.3–2.2 m/s (likely jogging);
    - Ferber cluster-fit residual > 5 mm;
    - too few strides (treadmill: at least 2 per side);
    - stride-time CV > 20 % (event detection failed or the walk wasn't steady).

Result: Fukuchi 1,972 / Ferber 2,061 / Toronto 48 kept trials.

### 5.4 Splits (`gaitdata.make_splits`)

- **By person, never by trial or window**, so nobody appears in two splits: 70 / 15 / 15 % train/val/test,
  seed 0.
- Fukuchi is stratified by age group, so young and older people reach every split.
- Toronto is test-only (an external check on older adults).
- Result: 1,205 / 258 / 269 people; 2,802 / 663 / 616 trials.

---

## 6. Labels: how the "right answers" are made

### 6.1 Events

- **Gold:** force plates (Fukuchi treadmill, Kuopio). A foot is on the ground while the plate measures weight.
- **Zeni 2008 coordinate method** (`canonical.detect_events`) for everything else:
  - heel strike = the moment the heel is furthest in front of the pelvis;
  - toe-off = the moment the toe is furthest behind the pelvis;
  - found as peaks with a minimum spacing and a prominence of 30 % of the range.
- **Bias correction** (`gaitdata.zeni_offsets`): on Fukuchi treadmill trials that have both, Zeni is on
  average **−34 ms** early for heel strike and **+31 ms** late for toe-off. That median offset is subtracted
  from Zeni events everywhere, so all labels mean "force-plate time".

### 6.2 One stride, all the formulas (`gaitdata.strides_from_events`)

This one function defines a stride for the labels, the classic baseline and the model:

- A **stride** = heel strike → the next heel strike of the *same* foot, 0.6–2.5 s long, containing exactly one
  own toe-off and one heel strike and one toe-off of the other foot. Anything else is discarded as unreliable.
- `stride_time` = HS → next HS
- `step_time` = this HS → the other foot's HS
- `stance_time` = HS → own TO; `swing_time` = stride − stance; `stance_pct` = 100 × stance / stride
- `double_support` = (HS → other TO) + (other HS → own TO): both feet on the ground
- `hip_rom`, `knee_rom`, `ankle_rom` = max − min of that joint's angle over the stride; `peak_knee_flex` = max knee angle

Output array `[S, 13]`: `start_s, end_s, side, ` + the 10 `STRIDE_COLS`.

### 6.3 Window / recording labels (`gaitdata.aggregate`)

Uses only the strides fully inside the time range:

- `cadence` = 120 / mean stride time (steps per minute);
- means of the stride columns;
- `stride_time_sd` and `stride_time_cv` (needs ≥ 3 strides);
- the six `SI_*` (needs strides on both sides; per-side means);
- `speed_ll` is passed in from the trial: treadmill belt speed or pelvis velocity, divided by leg length.

20 targets in all (`TARGETS` in `gaitdata.py`):

- spatiotemporal (7): cadence, stride, step, stance, swing time, stance %, double support
- speed (1)
- symmetry (6)
- range of motion (4): hip, knee, ankle, peak knee flexion
- variability (2)

**Why per window?** The model sees 5 s, so each training example is labelled with exactly the strides it
contains (`window_labels`). `derive()` applies the same formulas to predicted events, which is what makes
model outputs comparable with labels.

---

## 7. File formats

| File | Content |
|---|---|
| `data/processed/<ds>/trials.parquet` | One row per trial: metadata (subject, mode, speed, age, sex, height…), quality columns (`missing_frac`, `leg_length_sag`, `reject_reason`, `keep`), per-trial labels (`cadence`, `SI_*`, `*_rom`…), and on Fukuchi `zeni_err_ms_*` (Zeni vs force plate). |
| `data/processed/<ds>/strides.parquet` | One row per stride: `side, start, end` (native frames) + the 10 stride columns + `trial, subject`. |
| `data/processed/<ds>/npz/<trial>.npz` | `xyz` [T,11,3] lab frame, native rate · `fwd`, `up` axes · `fs` · `leg_length` · `sag30` [T30,11,2] normalised side view at 30 Hz · `ang30` [T30,6] · `pelvis_fwd30` · `ev_*` Zeni events, `gold_*` force-plate events (native frames), `ev30_*` events used for labels (30 Hz frames). |
| `data/processed/splits.parquet` | `dataset, subject, split`. |
| `data/processed/model_trials_v2.pkl` | Cache of `gaitdata.load_all()`: `{"train"/"val"/"test": [Trial]}`. A `Trial` has `ds, trial, subject, mode, pose [T,10,2], events (s), strides [S,13], speed_ll, leg_length, event_source ("grf"/"zeni"), ang [T,6], ev_soft [T,4]`. Delete it to rebuild from the npz files. |
| `data/processed/kuopio_clips.pkl` | List of synced clips: `subject, trial, direction, near` (leg nearest the camera), `fps, t0, sync_err_px, geo` (camera), `px` [n,11,2] video keypoints (pixels), `conf` [n,11], `truth_sag` [n,11,2] mocap in the side plane (m), `truth_px` (mocap projected into pixels), `events` (force plates, s from clip start), `leg_length_m, speed_ms, age, sex, height_cm`. |
| `outputs/models/bigru_v4/` | `model.pt` (weights), `config.json` (hidden, layers, W, …), `norm.json` (target and angle means/stds used to z-score), `history.csv` (losses per epoch), `reference_ranges.csv` (healthy 5–95th percentiles for the demo, built on first use). |
| `outputs/eval/test/` | Per-window evaluation: `predictions.parquet` (one row per window × method × noise level with `true_*`/`pred_*`), `regression.csv` (metrics), `events.csv` (event detection). |
| `outputs/eval/test_trial/` | Per-recording evaluation; its `regression.csv` gives the demo's "typical error" column. |
| `outputs/kuopio/` | Real-video test: per-clip tables (`timing.csv`, `angles.csv`, `speed.csv`, `keypoint_error.csv`, `clips.csv`, `events.parquet`), summaries (`*_summary.csv`, `rom_si.csv`) and `summary.md`. |
| `outputs/pose_models/` | The two ONNX pose models (downloaded on first use). |

---

## 8. The model

### 8.1 Input features (`model.features`): 74 per frame

| Block | Size | What |
|---|---|---|
| positions | 20 | x, y of the 10 joints (leg lengths, pelvis origin) |
| velocities | 20 | frame-to-frame change × 3 (i.e. per 1/10 s), zeroed if the joint is hidden in either frame |
| visibility | 10 | 1 = keypoint seen, 0 = missing (missing coordinates are set to 0) |
| segment directions | 12 | sin and cos of thigh, shank, foot angles per side, zeroed if hidden |
| geometric angles | 6 | the 6 joint angles computed directly from the points with the label formulas, ÷ 60 |
| angle validity | 6 | 1 if all points that angle needs are visible |

Why the geometric angles? On clean input they *are* the answer; giving them to the network lets it copy them
when the pose is good and correct them when it isn't (see the gate below).

### 8.2 Architecture (`model.GaitNet`): 867,556 parameters

```
features [B, T, 74]
   ├─► Linear 74→64 + GELU ─► BiGRU (2 layers, 128 units each direction, dropout 0.2) ─► h [B, T, 256]
   └─► Conv1d 74→64 (k=5) + GELU ─► Conv1d 64→64 (k=5) + GELU ─► local [B, T, 64]   (sees ±0.13 s)
h_all = concat(h, local) [B, T, 320]
   ├─► event head:  Linear 320→64 + GELU ─► Linear 64→4      = per-frame logits HS_L, HS_R, TO_L, TO_R
   ├─► angle head:  Linear 320→64 + GELU ─► Linear 64→12     = 6 learned angles + 6 gate logits
   └─► (from h) mean + std over time [B, 512] ─► 5 small MLPs (512→128→n) = whole-window heads (20 targets)
```

| Part | Parameters | Purpose |
|---|---|---|
| input layer | 4,800 | mixes the features |
| BiGRU | 445,440 | reads the whole 5 s forwards and backwards: walking rhythm and context |
| local conv branch | 44,288 | short-range detail, so event timing follows the feet, not just the rhythm |
| event head | 20,804 | when do heels land / toes lift |
| angle head | 21,324 | joint angles (with gate) |
| whole-window heads | 330,900 | speed (used) and fallback values for every target |

**The gated angle output.** For each angle and frame:

```
gate  = sigmoid(gate_logit) × validity            (0 if a needed keypoint is hidden)
angle = gate × zscore(geometric angle) + (1 − gate) × learned angle
```

On clean pose the model learns gate ≈ 1, giving exact geometry (angle error 0.05–0.25° on clean mocap). On
noisy pose it leans on the learned, denoised value. `ang_mean`/`ang_std` are buffers saved with the weights
(the z-scoring of angles).

**Why a BiGRU?** Small data (≈2,800 training trials), CPU-only, a sequence task where the context on both sides
of a frame matters (is this the moment the heel lands?). A transformer would be heavier with no clear gain at
this size.

### 8.3 Model history in one line each (details: `src/model_findings.md`)

- **v1** predicted every number directly from pooled features. It held up under noise but shrank every
  asymmetry towards zero.
- **v2** switched to per-frame events and angles plus the formulas, and added the left/right mirror.
- **v3** added the gated geometric angles and the local conv branch, which fixed ankle range of motion.
- **v4** added synthetic limps, so asymmetry is reported at close to its true size. v4 is the final model.

---

## 9. Training and data augmentation

### 9.1 What one training example is

A random 5 s window of a training trial. Each batch of 64 comes from one data group:

- 50 % Ferber
- 25 % Fukuchi treadmill
- 25 % Fukuchi overground (short trials, so the whole batch is cropped to a common length)

Targets per window:

- **events** as soft targets: a Gaussian bump (σ = 1 frame = 33 ms) at each true event (`soft_events`);
- the **6 angle curves**;
- the **20 window labels**: z-scored with training means/stds (`norm.json`), and NaN where not measurable
  (e.g. no full stride in the window).

### 9.2 Augmentation 1: video-like pose noise (`augment.Noise`, `augment.apply`)

| Effect | Parameter | Simulates |
|---|---|---|
| per-frame jitter | `sigma` (Gaussian, per coordinate) | keypoints wobbling frame to frame |
| constant offset per keypoint | `bias` | keypoints sit on skin/shoes, not on joint centres |
| dropped keypoints | `dropout` (probability per keypoint and frame) | detector misses |
| far leg | noise × 1.5, and hidden where it passes within 0.06 leg lengths of the near leg | the leg away from the camera is noisier and occluded |
| leg-length error | scale × exp(N(0, `scale_sd`)) | wrong body-size estimate |
| camera tilt | rotation N(0, `rot_deg_sd`) degrees | camera not level |

The pelvis origin is recomputed from the *noisy* hips, as a pose estimator would.

- **Training distribution** (`Noise.random`): 20 % clean. Otherwise:
  - σ uniform in 0–0.06
  - bias uniform in 0–σ
  - dropout uniform in 0–0.1
  - far-leg effect in half of the cases
  - scale SD 0.05, tilt SD 3°
- **Evaluation presets** (`Noise.level(σ)`): one knob, σ = 0, 0.01, 0.02, 0.04, 0.08. Bias = σ/2,
  dropout = σ, scale SD = σ, tilt = 50σ degrees, far-leg on. σ = 0.02 is called "realistic video".

### 9.3 Augmentation 2: left/right mirror (`augment.swap_sides`, p = 0.5)

Relabels the left leg as the right and vice versa: joints, events and angles are swapped, and every symmetry
label changes sign. In a side view of mocap this is a valid new example, and it stops the model from
preferring one side.

### 9.4 Augmentation 3: synthetic limps (`augment.asymmetric_window`, p = 0.5)

Most people in the data walk almost symmetrically, so a model learns "both legs are the same, half a cycle
apart" and under-reports real asymmetry. For a random leg:

- **time shift:** that leg's trajectory and events are shifted by up to ±50 ms, which makes step times unequal;
- **excursion scaling:** each of its hip, knee and ankle excursions is scaled by 0.8–1.25 (log-uniform). This
  is done by rigidly rotating the segments below each joint (`scale_leg`), so the points stay a consistent
  skeleton;
- all labels are recomputed from the modified events and angles (`derive`).

### 9.5 Loss, optimiser, schedule (`train.py`)

| Item | Value |
|---|---|
| event loss | binary cross-entropy on the soft targets, positive weight 5 (events are rare), loss weight 2 |
| angle loss | smooth L1 (β = 0.1 in z units) on the angles |
| head losses | smooth L1 per head (spatiotemporal, speed, symmetry, ROM, variability) on finite targets |
| optimiser | AdamW, lr 1e-3, weight decay 1e-4, gradient clipping 1.0 |
| schedule | ReduceLROnPlateau ×0.5 after 3 epochs without improvement; early stop after 10 |
| size | 50 epochs × 200 batches × 64 windows; ~190 s per epoch on 6 CPU threads (~2.5 h) |
| validation | all val windows at fixed noise σ = 0.02 (fixed seed); best checkpoint saved |
| final run | best validation loss 0.923 at epoch 47 of 50 |

---

## 10. Turning model outputs into measurements

1. **Events** (`model.decode_events`):
   - per event type, find peaks of the probability above 0.5, at least 0.4 s apart;
   - refine each peak to sub-frame precision with a parabola through the 3 frames around it.
2. **Strides**: `strides_from_events` with the predicted events and angle curves.
3. **Numbers**: `aggregate` over all strides of the recording (`derive` for one window).
4. **Speed** comes from the whole-window speed head (median over windows), because speed isn't a stride
   formula.
5. **Fallback**: if no complete stride was decoded, the whole-window head values are used instead (method
   `bigru_v4`). The main method is `bigru_v4+derived`.

---

## 11. Evaluation

### 11.1 The two baselines (the brief requires them)

- **Classic method** (`classic.py`): what you'd build without machine learning.
  - Fill gaps in hidden keypoints by interpolation, then low-pass at 6 Hz.
  - Zeni events with the same bias correction, and the same formulas.
  - Speed from how fast the standing foot moves backwards relative to the pelvis (median over the middle half
    of each stance).
- **Guess the average** (`mean`): always answer the training-set average. Any useful method must beat it.

### 11.2 Per window (`evaluate.py`)

- Every test trial is cut into non-overlapping 5 s windows (2,779 windows from 616 trials).
- At each noise level, every method sees the *same* noisy input (fixed seed per level).
- **Event detection:** a predicted event within 0.1 s of a true one counts as a hit (one-to-one closest
  matching, `match`). Events within 0.2 s of a window edge are not scored. Reported: recall, precision, MAE,
  bias (ms).
- **Regression metrics** (`regression_metrics`):

| Metric | Meaning |
|---|---|
| MAE | mean absolute error |
| bias | mean signed error |
| r | correlation with the truth |
| slope | fitted predicted-vs-true slope: **1 = a difference is reported at its true size**, < 1 = shrunk towards the average (important for symmetry) |
| coverage | share of measurable windows that got a prediction |
| mean_pred_mae | MAE of guessing the average on the same data |

### 11.3 Per recording (`evaluate_trials.py`)

- One noise draw per whole trial, so the far leg, offsets, scale and tilt are fixed, like one real recording.
- The trial is tiled into windows, and all their strides are pooled before computing the outputs. This is
  what a physio would get from one video.
- Its `regression.csv` is the source of the demo's "typical error" and reliability columns.

---

## 12. The video pipeline and the demo

### 12.1 Video → keypoints (`pose_video.py`)

- **Detector:** YOLOX-m (HumanArt), 640 px letterbox, person class, score > 0.3. The first detection picks the
  largest confident person; later detections pick the box overlapping the tracked one.
- **Pose:** RTMPose-m, Halpe-26 keypoints (includes big toe, small toe, heel), 192×256 crop with 25 % padding,
  SimCC output.
- **Tracking:** the detector runs every 15 frames (it's ~9× slower). In between, the box comes from the
  previous frame's keypoints. It re-detects early if the median keypoint score drops below 0.35.
- Both models run on CPU through ONNX Runtime and are downloaded from OpenMMLab on first use.

### 12.2 Keypoints → model input (`video_pipeline.py`)

1. **Map** Halpe-26 to our 10 joints (toe = mean of big and small toe) and flip y to point up
   (`halpe_to_canonical`).
2. **Resample** to 30 Hz; keypoints with score < 0.3 are marked hidden.
3. **Fix left/right mix-ups** (`fix_left_right`). Pose estimators often swap the legs in side views.
   - Each frame is either "as detected" or "swapped".
   - Dynamic programming (Viterbi) picks the sequence where every joint moves most smoothly: closest to where
     its previous two positions predict.
   - Penalty 0.5 per switch. On mocap with injected swaps it restored all of them and flipped no clean frames.
4. **Split into walking passes** (`split_passes`): the direction per frame comes from the feet (toe ahead of
   heel), median-filtered over 1 s. Turns and long gaps end a pass; passes shorter than 1.5 s are dropped.
5. **Normalise** each pass: walking towards +x, hip-centred, divided by the median hip–ankle length.
6. **Run the model** on 5 s windows overlapping by 50 %:
   - probabilities and angles are blended with a sine weight that fades at window edges, where an event might
     be cut off;
   - events and strides are decoded once over the whole pass, so no stride is lost at a window boundary;
   - duplicate strides from overlapping passes are dropped.

### 12.3 The report

For each of 19 outputs the report shows:

- **value**, and **left / right** means for per-leg measures;
- **typical error ±**: the per-recording MAE measured on held-out people at σ = 0.02, interpolated;
- **reliability**: "good" if that error is < 50 % of the guessing error, "rough" if < 85 %, otherwise "not
  reliable";
- **healthy range**: 5th–95th percentile of healthy Fukuchi adults.

Speed is shown in m/s if the leg length is known; leg length ≈ 0.49 × body height if only height is given.
Otherwise it is in leg lengths/s.

Also reported:

- "same foot landed twice in a row" warnings (a sign of left/right confusion);
- stride counts per side;
- pass count, and a note if only one direction was filmed.

### 12.4 Figures and video (`helpers/render.py`)

- symmetry plot (bars = mean, dots = each stride)
- timeline (speed per window, cadence per stride)
- gait diagram (stance bars per foot, shaded double support)
- angle curves with heel-strike marks
- the annotated video (skeleton, "Left: stance / Right: swing" banner, a moving gait-diagram strip),
  H.264-encoded through ffmpeg so browsers can play it
- a stick-figure video for the mocap tab

### 12.5 The app (`app.py`, Gradio)

- **Analyse a walking video:** upload or use the example clip (treadmill speed ramp, CC BY 2.0); optional
  height or leg length; outputs the annotated video, a quality note, the table, 4 plots, per-stride table and a
  CSV.
- **Held-out test trial (mocap):** pick a test trial, choose simulated noise σ; shows the model next to the
  gold values. Useful for a demo without video.
- **How it works & limits.**

A 45 s clip takes about 2–3 minutes on CPU, mostly pose estimation.

---

## 13. Real-video validation (Kuopio)

**Why it matters:** everything else is gait-lab data with *simulated* video noise. Kuopio has real video
keypoints (OpenPose, a different pose model from the demo's) recorded together with Vicon and force plates.

### 13.1 Getting the data cheaply (`download_data.py`)

The three Zenodo zips are 23 GB, mostly raw camera files. The script reads each zip's table of contents over
HTTP and downloads only the bytes of the files we need: comfortable-speed walking trials (`.c3d`) and their
OpenPose JSON (~1 GB). It does this with HTTP range requests and CRC-checks every file.

### 13.2 Syncing video to mocap (`load_kuopio.sync_subject`, `build_kuopio.py`)

The dataset gives neither the video frame rate nor when each ~1 s clip starts.

- **Camera model:** one fixed side camera per subject, `pixel = scale × (−lab coordinate) + offset`, shared by
  all 20 of that subject's clips.
- **Fit:**
  1. scan the horizontal offset;
  2. alternate four times between best start times (grid search) and the best shared camera (robust least
     squares);
  3. refine each start time at 2 ms resolution.
- **Cost:** median distance between the video's hip, knee and ankle keypoints and the projected mocap joints,
  *with left/right labels*. The labels are what resolve the one-step ambiguity of a periodic motion: the right
  alignment had a 14 px error, one step off had 52 px.
- **Frame rate:** 97 fps (fitted from geometry on 12 subjects).
- **Independent check:** the force plates are never used in the sync. The first plate heel strike lands a
  median 0.254 s into the clip, with an IQR of only 14 ms across 817 clips, in both walking directions.

### 13.3 What is measured (`evaluate_kuopio.py`)

Each clip is run through the same steps as the demo (`prepare`, `facing`, `normalise_pass`, the model), and
through the classic method, with four inputs on the same frames:

- `real`: OpenPose keypoints
- `mocap`: clean projected mocap
- `sim0.02`, `sim0.04`: mocap with simulated noise

The measures:

- **keypoint error** split into a constant offset per clip, jitter (robust SD), gross errors (> 0.2 leg
  lengths) and lag-1 autocorrelation;
- **event timing** vs force plates;
- **stance, step and swing times** inside the clip;
- **joint excursion** per clip (≈ one stride);
- **range-of-motion symmetry** per subject, pooling about 20 clips: all clips, or near-leg clips only;
- **speed**.

---

## 14. Environment and commands

- **Nix flake** (`flake.nix`, pinned by `flake.lock`): Python 3.14 with numpy, scipy, pandas, pyarrow, torch,
  matplotlib, jupyter, opencv, onnxruntime and gradio, plus 7-Zip, unzip, curl and ffmpeg.
  - gradio is marked broken in nixpkgs (a dependency pin), so the flake overrides its checks.
  - The Jupyter kernel is called `gait`.
- **`./py`** runs Python inside the dev shell with `src/` on the import path, from any directory.
- **CPU only**, no GPU needed.

```sh
./py src/helpers/download_data.py              # data (Ferber is 22.8 GB)
./py src/helpers/build_dataset.py              # processed dataset (~1 h)
./py src/train.py                              # model (~2.5 h)
./py src/evaluate.py && ./py src/evaluate_trials.py
./py src/helpers/build_kuopio.py && ./py src/evaluate_kuopio.py && ./py src/helpers/plot_kuopio.py
./py src/app.py                                # demo at http://127.0.0.1:7860
```

Full details, timings and checks: `REPRODUCE.md`.

---

## 15. Key results cheat sheet

**Held-out people, simulated video noise σ = 0.02, per 5 s window (MAE; model / classic / guess the average):**

| Output | Ferber (251 people) | Fukuchi | Toronto (older, never trained on) |
|---|---|---|---|
| heel-strike timing | **3.5 ms** / 11.3 | **7.5** / 14.6 | **5.5** / 14.0 |
| cadence (steps/min) | **0.18** / 0.37 / 6.3 | **0.46** / 0.87 / 19.3 | **0.31** / 0.44 / 7.6 |
| knee ROM (°) | **1.8** / 4.1 / 4.4 | **1.8** / 5.4 / 8.9 | **3.1** / 4.7 / 17.0 |
| step-time SI (% points) | **1.2** / 3.1 / 2.1 | **3.2** / 5.4 / 3.8 | **2.3** / 4.0 / 3.2 |

**Asymmetry at true size:** the step-time SI slope per recording is 0.89 (v1 was ~0.5).

**Subgroups:** errors are similar for women and men, and somewhat larger at age 65+.

**Real video (Kuopio, 817 clips, 42 people):**

| Measure | Model | Classic | Guess |
|---|---|---|---|
| heel strike vs force plate | **20 ms** | 24 ms | – |
| stance / step time | **33 / 26 ms** | 38 / 27 ms | 50 / 40 ms |
| speed | **0.075 m/s** | 0.48 m/s | 0.134 m/s |
| hip / knee / ankle excursion | **4.1 / 5.7 / 3.8°** | 5.7 / 8.3 / 13.5° | – |
| knee ROM symmetry per person | **4.2 %** (r 0.56) | 10.2 % | 5.1 % |

**The honest headline:** real OpenPose error has the same per-frame jitter as the simulation, but it
persists across frames, sits further off at the feet, and includes 4–7 % gross errors. So real-video accuracy
is about that of simulated σ = 0.04, not 0.02.

---

## 16. How to explain it to people

### 16.1 Thirty seconds (anyone)

> "Physios want to know if a patient's walking is improving, but the machines that measure it cost a fortune.
> We made a tool that gets the same kind of numbers, like how evenly someone uses both legs, from a phone
> video. We taught it on gait-lab recordings made to look like shaky video, and tested it on real video of
> people it had never seen."

### 16.2 Two minutes (a judge or manager)

1. **The need:** a 5–10 % difference between legs is invisible to the eye but is exactly what recovery
   tracking needs.
2. **The idea:** find the joints in each frame, let a neural network work out when each foot lands and lifts
   and how the joints bend, then compute the numbers with the standard gait-lab formulas.
3. **The data trick:** no big dataset of videos with lab answers exists, so we damaged 1,700 people's lab
   recordings to look like video and added fake limps, so the model learns that legs can differ.
4. **The proof:**
   - it beats the textbook method and guessing on people it never saw, including older adults it never
     trained on;
   - on real video from another lab it finds footsteps to about 20 ms and speed to about 0.08 m/s.
5. **The limits:** real video is noisier than we simulated, hip and ankle symmetry aren't reliable from video
   yet, and it is a progress-tracking aid, not a diagnosis.

### 16.3 Five minutes (an engineer)

- **Pipeline:**
  - YOLOX + RTMPose (ONNX, CPU) → 10 sagittal keypoints at 30 Hz;
  - Viterbi left/right repair, pass splitting, pelvis-centred and leg-length-normalised input;
  - BiGRU (2×128, bidirectional) plus a short conv branch on 74 features per frame → per-frame event logits
    and gated angle curves;
  - peak decoding with a parabolic sub-frame fit → strides → the same `strides_from_events` / `aggregate`
    code that made the labels.
- **Why measure instead of regress:** pooled regression heads shrank asymmetry towards the population mean.
  Decoding events and computing differences keeps them at true size.
- **Gated skip:** the angle head can copy the geometric angle when the keypoints are good (gate × validity),
  or fall back to a learned denoised estimate.
- **Training:**
  - 2,800 trials, person-level splits, three augmentations (parametric pose noise, L/R mirror with SI sign
    flip, synthetic limps by time-shift and rigid excursion scaling);
  - losses: soft-target BCE for events, smooth L1 for angles and z-scored heads;
  - AdamW, early stopping.
- **Evaluation:**
  - per window and per recording, against a classic Zeni pipeline and the mean predictor, across noise levels;
  - external test on Toronto;
  - real-video test on Kuopio with a geometry-only video/mocap sync, force plates held out as an independent
    check.

### 16.4 Good analogies

- **Noise augmentation** = training a spell-checker on deliberately misspelled text.
- **Event decoding then formulas** = instead of asking someone to guess your average step time, you ask them to
  mark every footstep and compute it yourself.
- **The gate** = a co-pilot: when the instruments are clear, use them; when they're foggy, use experience.
- **Person-level split** = a fair exam: no student sees the exam questions in practice.
- **Symmetry slope** = a bathroom scale that always says "a bit more than average" is useless for tracking
  weight loss; we check that big differences come out big.

### 16.5 Showing the demo

1. Start with the **mocap tab**, which is fast. Pick a trial, show the model next to the gold values, raise the
   noise slider to 0.04 and show the errors grow.
2. Then the **video tab** with the example clip. The treadmill speeds up and slows down (0.5 → 1.5 → 0.5 m/s);
   the "over the recording" plot shows cadence and speed following it.
3. Point at the **reliability column**: the tool says which numbers not to trust.
4. Finish with the notebook's real-video section, or `outputs/figures/07_kuopio.png`.

---

## 17. Questions you will get, and answers

**"Is this a diagnosis tool?"** No. It measures walking and tracks change over time. It has only been tested on
healthy and orthopaedic walkers.

**"How accurate is it on real phone video?"**

- Tested on research video, not phones: heel strikes to about 20 ms (one frame is 33 ms), speed to 0.08 m/s,
  knee range of motion to about 6°.
- Phone video in a clinic will vary; that is the next thing to test.

**"Why not just measure the angles from the keypoints?"** That is what the classic baseline does. It works on
clean data, but on real video it is 1.4–3.6× worse on joint motion and 6× worse on speed.

**"Didn't you just train and test on the same people?"**

- No. Splits are by person.
- Toronto (older adults) and Kuopio (real video) were never used for training or tuning.

**"Why does the model beat the classic method if both use the same formulas?"** Because the formulas are only
as good as the events and angles fed in. The model's events are 2–3× more precise under noise; it uses 5 s of
context and has learned what walking looks like.

**"What's the hardest part?"**

- Ankle measurements: the foot is small in the frame, and foot keypoints are the least precise.
- Stance/swing asymmetry: toe-off is hard to time even in the lab reference.

**"Why a GRU and not a transformer?"** Small data, CPU-only, and a sequence task with strong local structure. A
900k-parameter BiGRU trains in 2.5 h on a laptop CPU.

**"How did you sync the Kuopio video without timestamps?"** From geometry. One fixed camera per person, and
start times chosen so the video joints land on the projected mocap joints. We checked the sync against the
force plates, which it never used.

**"What would you do next?"**

1. Measure real pose-error statistics and train with them.
2. Fine-tune on part of Kuopio.
3. Record at 60 fps.
4. Use a foot-specialised keypoint model.
5. Test on patients (stroke, Parkinson's).

---

## 18. Cookbook: changing things

| I want to… | Do this |
|---|---|
| retrain | `./py src/train.py --name my_model`; to use it, change `RUN` in `evaluate.py` (the demo and evaluations read it). |
| change the window length | `W` in `train.py` (written to `config.json`) and `W` in `video_pipeline.py`. |
| change augmentation strength | `Noise.random` (training distribution) or the 0.5 probabilities in `train.make_batch`. |
| add an output | add the stride column in `gaitdata.strides_from_events` / `STRIDE_COLS` and a target in `TARGETS`, then retrain (the head sizes change). |
| add a dataset | write `helpers/load_<name>.py` returning `xyz [T, 11, 3]` (m) in `JOINTS` order, add it to `build_dataset.py`, `analysis.DATASETS`, `gaitdata.make_splits`, then delete `model_trials_v2.pkl` so the cache rebuilds. |
| use a different pose model | map its keypoints to the 10 joints in `video_pipeline.halpe_to_canonical` (or a new function) and pass scores; everything after that is the same. |
| make the demo's error column stricter | set `VIDEO_SIGMA` in `video_pipeline.py` to 0.04 (the level real video matched). |
| rebuild the model-data cache | delete `data/processed/model_trials_v2.pkl`; the next `load_all()` recreates it. |

---

## 19. Glossary

| Term | Meaning |
|---|---|
| gait cycle / stride | heel strike to the next heel strike of the same foot (~1 s) |
| step | heel strike to the other foot's heel strike (half a stride) |
| heel strike (HS) / toe-off (TO) | foot lands / leaves the ground |
| stance / swing | foot on the ground (~60 % of a stride) / in the air (~40 %) |
| double support | both feet on the ground (~20 % of a stride) |
| cadence | steps per minute |
| range of motion (ROM) | max − min of a joint angle over a stride |
| flexion / dorsiflexion | bending a hip or knee / pulling the toes up |
| sagittal plane | the side view |
| symmetry index (SI) | 100 × (L − R) / mean, in % |
| coefficient of variation (CV) | SD / mean × 100: how irregular stride times are |
| motion capture (mocap) | camera system tracking reflective markers in 3D (the reference) |
| force plate | floor sensor measuring ground force; the gold standard for foot contact |
| Zeni method | estimating heel strike / toe-off from marker positions relative to the pelvis |
| pose estimation | finding body keypoints in images (RTMPose, OpenPose) |
| keypoint | one detected body point (e.g. left knee) in an image |
| BiGRU | bidirectional gated recurrent unit: a neural network that reads a sequence forwards and backwards |
| held-out / test set | data never used for training or tuning |
| ablation | removing one improvement at a time to measure its effect |
| MAE | mean absolute error |
| slope (here) | predicted vs true regression slope; 1 = reported at true size |
| SimCC | RTMPose's way of predicting coordinates as classification over pixel bins |
| Kabsch algorithm | the best rigid rotation + translation between two point sets |
| Viterbi | dynamic programming that finds the best sequence of hidden states |
