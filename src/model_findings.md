## Findings

*The model versions compared here (v1–v3, ablations, two-pass protocol) were trained and evaluated with earlier
versions of the code; the current code keeps v4 and its fine-tuned successor, v5 (last section).*

**Final model: `bigru_v5+derived`** (v4 fine-tuned on replayed real pose error; the tables below up to the Kuopio
section describe v4). Events + gated angle curves → the label formulas, trained with pose noise, the L/R mirror and
synthetic asymmetry. Numbers are per 5 s window on held-out subjects unless stated; "slope" is
predicted vs gold (1 = an asymmetry is reported at its true size).

### What was fixed

| Problem in v1 | v1 | v4 | classic | mean |
|---|---|---|---|---|
| Step-time SI shrunk (slope, σ = 0.02): Ferber / Toronto | 0.51 / 0.33 | **0.86 / 0.73** | 0.92 / 1.04 | 0 |
| ↳ under-reporting of \|SI\| ≥ 7 (SI points, σ = 0.02) | −7.2 | **−3.2** | −2.2 | |
| Ankle-ROM SI, clean: MAE (slope), Ferber | 11.1 (0.03) | **0.77 (0.90)** | 1.04 (1.00) | 11.2 |
| Knee-ROM SI, σ = 0.02: MAE (slope), Toronto | 10.1 (0.11) | **5.7 (0.69)** | 7.6 (0.97) | 10.5 |
| Ankle ROM (°), clean: Ferber / Toronto | 2.9 / 13.8 | **0.19 / 1.8** | 0.27 / 0.71 | 3.2 / 11.0 |
| Knee ROM (°), σ = 0.02: Ferber / Toronto | 2.4 / 4.2 | **1.8 / 3.1** | 4.1 / 4.7 | 4.4 / 17.0 |
| Stance % (Toronto, σ = 0.02) | 1.35 | **0.84** | 1.13 | 1.43 |
| Stride-time CV (Toronto, σ = 0.02) | 0.65 | **0.43** | 1.04 | 0.98 |

Cadence (0.14–0.46 steps/min) and event timing (Ferber heel strike 2.5 ms clean, 3.5 ms at σ = 0.02, vs classic
8–11 ms) stay at or better than v1. A complete stride was decoded in every test window (0 % fallback to the pooled
heads).

### What each step did (ablation, σ = 0.02 unless stated)
1. **Formulas on dense heads (v2)**: removed most structural shrinkage of pooled regression.
2. **Gated geometric skip + local branch (v3)**: angle RMSE on clean pose 2–4° → 0.05–0.25° (better than the
   filtered classic curves); ROM and ROM-SI on clean or mildly noisy pose now match exact geometry.
3. **Diagnosis for v4**: v3's event errors were tiny (3–5 ms) but correlated with asymmetry. Adding random jitter of
   the same size to the gold events keeps the step-time SI slope at 1.0; the model's own events gave 0.6. It had
   learned "the legs are half a cycle apart" from a near-symmetric training set.
4. **Synthetic asymmetry (v4)**: step-time SI slope 0.65 → 0.86 (Ferber), 0.59 → 0.73 (Toronto); hip-/knee-ROM SI
   slopes up by 0.05–0.2, at about the same MAE.

### Per recording and the two-pass protocol (`evaluate_trials.py`)
Pooling all strides of a 30 s recording (one consistent noise draw per video) gives Ferber step-time SI MAE 0.82
with slope 0.89 at σ = 0.02 (mean predictor 1.77). Filming the patient **walking across and back**, and measuring
each leg from the pass where it is nearest the camera, mainly helps the unbiased-but-noisy classic ROM-SI
estimate. For the model it gives knee-ROM SI slope 0.69 → 0.73 (Toronto). Recommended capture protocol for the demo:
≥ 20 s of walking, both directions.

### Still limited (honest)
- **Stance- and swing-time SI** stay compressed (Ferber slope 0.61; |SI| ≥ 7 under-reported by ~10 points). The
  classic pipeline is compressed nearly as much in that bin, so part of it is label noise (toe-off labels are
  bias-corrected Zeni or GRF with known outliers). The phase-shift augmentation does not create stance asymmetry; a
  per-leg stance/swing time-warp augmentation is the next step.
- **Ankle-ROM SI under video-like noise (σ ≥ 0.02)** is near the mean predictor for every method, including classic
  and two passes. Jitter of ~1.6 cm on a ~20 cm heel–toe segment is ~6° per frame against a ~3° left/right
  difference, so the information is not in a single side view at that quality. The demo should show ankle symmetry
  only with a pose-quality check, or not at all.
- **Toronto ankle ROM under noise** (10.1° vs mean 11.0°, classic 8.1°) and **Toronto speed** (0.05 m/s vs v1's
  pooled 0.03) remain the weakest external-set numbers.
- **v4 vs v3 trade-off**: v3 is slightly better on stance-time SI and Toronto stance % (0.60 vs 0.84); v4 is better
  on every step-time and ROM symmetry slope. v4 is kept because reporting asymmetry at its true size was the goal.
- Everything above uses mocap-derived pose with simulated noise. Real pose-estimator output is tested below (Kuopio).

### Real video keypoints: Kuopio
v4 was tested on 817 real OpenPose clips from 42 subjects against Vicon and force plates; results, figure and
caveats are in the project write-up (v4: `outputs/kuopio_v4/summary.md`). In short: the model transfers and the classic
pipeline breaks down on real pose, but real pose error is correlated, offset and outlier-prone, so accuracy sits
at or beyond the simulated σ = 0.04 level, not at σ = 0.02.

### v5: replayed real pose error
`helpers/kuopio_noise.py` records the real error tracks (OpenPose keypoints minus mocap, after the demo's steps) of
Kuopio subjects 01–26 (18 usable, 347 clips). Training replays them on mocap windows (`augment.Noise.real`): 40 % of
windows get real error (×0.7–1.3), 40 % simulated noise, 20 % none. v5 = v4 fine-tuned this way (lr 3e-4, best
validation loss at epoch 13 of 15). Scored on the 24 other Kuopio subjects:

| Real video, 24 held-out subjects | v4 | v5 |
|---|---|---|
| Heel strike MAE / SD (ms), recall | 20 / 28, 96 % | **16 / 19, 99 %** |
| Step time MAE (ms), r | 25, 0.82 | **17, 0.95** |
| Stance time MAE (ms), r | 32, 0.92 | **28, 0.97** |
| Speed MAE (m/s), r | 0.067, 0.82 | **0.044, 0.91** |
| Hip / knee / ankle angle-curve error (°) | 4.8 / 6.8 / 3.5 | **3.4 / 5.6 / 3.1** |
| Knee / hip / ankle ROM SI MAE (mean predictor 4.2 / 2.9 / 10.2) | 3.8 / 3.6 / 11.7 | 3.9 / 4.0 / 10.4 |

On mocap + simulated noise v5 is slightly worse (median error +2–4 %, Fukuchi hip ROM 1.2 → 1.7°, hip-ROM SI slope
on Ferber 0.62 → 0.52, step-time SI slope 0.86 → 0.83); Toronto stance/swing-time SI improved 15–30 %. v5 is kept
because real video is the intended input. Symmetry on real video did not improve.
