# Reproducing the project

Everything runs on CPU, from the repo root, inside the Nix flake. No step runs code from inside a
`data/` directory, and each dataset lives in its own directory.

*This project is a prototype and is not medical advice.*

## 0. What you need

| | |
|---|---|
| OS | Linux or macOS with [Nix](https://nixos.org/download) and flakes enabled (`experimental-features = nix-command flakes`) |
| Disk | ~75 GB for everything (Ferber alone: 22.8 GB zip + 45 GB extracted). Without Ferber: ~5 GB |
| CPU / RAM | any multi-core CPU, 16 GB RAM. No GPU needed |
| Time | ~1 h downloads (Ferber dominates), ~1 h dataset build, ~2.6 h training, ~1 h evaluation |

`./py` runs Python inside the dev shell (`nix develop`), so it works without entering the shell
first. The first call builds the environment (Python 3.14, numpy, scipy, pandas, pyarrow, torch,
matplotlib, jupyter, opencv, onnxruntime, gradio, ffmpeg, 7-Zip). `flake.lock` pins nixpkgs, so
everyone gets the same versions.

```sh
git clone <repo> gaitanalysis && cd gaitanalysis
./py -c "import torch, gradio; print('ok')"       # builds the environment once
```

## 1. Download the data (`src/download_data.py`)

```sh
./py src/download_data.py fukuchi toronto kuopio   # ~1 GB, minutes
./py src/download_data.py ferber                   # 22.8 GB + 7-Zip extraction (Deflate64)
```

| Dataset | Licence | Source | Role |
|---|---|---|---|
| Fukuchi 2018 WBDS | CC BY | Figshare 10.6084/m9.figshare.5722711 | train/val/test, force-plate events |
| Ferber Running Injury Clinic (walking) | CC BY 4.0 | Figshare 24255795 | train/val/test (bulk of the data) |
| Toronto Older Adults Gait Archive | CC BY | Figshare collection 5515953 | external test (older adults) |
| Kuopio gait dataset | CC BY 4.0 | Zenodo 10559504 | external test on **real video** keypoints |

For Kuopio, only the comfortable-speed walking trials (`.c3d`) and their OpenPose keypoints are
taken from the 23 GB of zips, read with HTTP range requests.

## 2. Build the processed dataset

```sh
./py src/build_dataset.py                 # all three training sources -> data/processed/<dataset>/
./py src/make_notebooks.py                # notebooks 01-05 (data, skeleton QC, events, kinematics, Toronto)
```

The steps are: canonical skeleton → sagittal projection → pelvis-centred, leg-length-scaled → 30 Hz →
Zeni events (force plates where available) → labels → QC filtering. Subject-level splits (70/15/15,
seed 0; Toronto is test-only) are written to `data/processed/splits.parquet`. The first model run caches
all trials in `data/processed/model_trials_v2.pkl`.

## 3. Train

```sh
./py src/train.py --name bigru_v4 --features v3 --angle-beta 0.1 --asym-p 0.5 --threads 6   # final model, ~2.6 h
```

The final model takes 50 epochs × 200 steps, about 190 s per epoch on 6 threads; seed 0 is the default.
The ablation runs from `src/model_findings.md` are optional:

```sh
V1="--features v1 --no-angle-head --swap-p 0 --epochs 40"
./py src/train.py --name bigru_clean $V1 --no-aug                    # v1 without pose noise
./py src/train.py --name bigru_aug $V1                               # v1: pooled heads + pose noise
./py src/train.py --name bigru_v2 --features v2                      # dense event/angle heads + L/R mirror
./py src/train.py --name bigru_v2_noswap --features v2 --swap-p 0    # without the L/R mirror
./py src/train.py --name bigru_v3 --features v3 --angle-beta 0.1     # + gated geometric angles
```

Each run writes `outputs/models/<name>/{model.pt, config.json, norm.json}` and logs to `outputs/logs/`.
CPU training is deterministic per seed within a machine. Across machines and torch builds, expect
small numeric differences, not different conclusions.

## 4. Evaluate on held-out subjects (mocap + simulated pose noise)

```sh
./py src/evaluate.py bigru_v4 bigru_v3 bigru_v2 bigru_v2_noswap bigru_aug bigru_clean   # per 5 s window
./py src/evaluate_trials.py bigru_v4 bigru_v3                                          # per recording
./py src/evaluate_trials.py bigru_v4 bigru_v3 --two-pass                               # across-and-back protocol
./py src/make_model_notebook.py                                                        # notebooks/06_model_results.ipynb
```

Outputs go to `outputs/eval/test*/` (regression.csv, events.csv, symmetry_bins.csv, …). The demo
needs `outputs/eval/test_trial/regression.csv` for its per-output expected error.

## 5. External validation on real video (Kuopio)

```sh
./py src/build_kuopio.py --estimate-fps 12   # video frame rate from geometry (writes outputs/kuopio/fps_estimate.json)
./py src/build_kuopio.py                     # sync every clip to mocap -> data/processed/kuopio_clips.pkl
./py src/evaluate_kuopio.py                  # -> outputs/kuopio/summary.md + CSVs (~15 min, 12 workers)
./py src/plot_kuopio.py                      # -> outputs/figures/07_kuopio.png
# frame-rate sensitivity (optional)
./py src/build_kuopio.py --fps 100 --out data/processed/kuopio_clips_fps100.pkl
./py src/evaluate_kuopio.py --clips data/processed/kuopio_clips_fps100.pkl --out outputs/kuopio/fps100
```

Kuopio gives neither the video frame rate nor the clip start times, so `build_kuopio.py` recovers
them from geometry. Each subject gets one fixed side camera, shared by all of that subject's trials,
plus one start time per clip, chosen to minimise the distance between left/right-labelled hip, knee
and ankle keypoints and the projected mocap joints. The force plates are not used for the sync, so the
heel-strike timing against them is an independent check.

## 6. Run the demo

```sh
./py src/app.py               # http://127.0.0.1:7860
./py src/app.py --share       # temporary public Gradio link
```

The first video analysis downloads the two pose models (~150 MB, OpenMMLab, Apache-2.0) into
`outputs/pose_models/`. A 45 s clip takes about 2–3 minutes on CPU. A sample clip with
attribution is in `data/demo_videos/` (gitignored); any side-on walking video works.

The demo needs only `outputs/models/bigru_v4/` and `outputs/eval/test_trial/regression.csv`. The
held-out-trial tab also needs `data/processed/`. To run the demo without retraining, copy those two
paths from a machine that has them (about 3.5 MB together).

## Checks that a reproduction worked

| Check | Expected |
|---|---|
| `data/processed/*/trials.parquet` | Fukuchi 1972 / Ferber 2061 / Toronto 48 kept trials |
| splits | train 1205 / val 258 / test 269 subjects |
| `outputs/eval/test/events.csv`, bigru_v4, σ=0.02, Ferber HS | MAE ≈ 3.5 ms (classic ≈ 11 ms) |
| `outputs/eval/test/regression.csv`, bigru_v4+derived, σ=0.02, Ferber SI_step_time | slope ≈ 0.86 |
| `data/processed/kuopio_clips.pkl` | 890 clips from 45 subjects (07, 09, 10, 24 lack ankle joint centres) |
| `outputs/kuopio/summary.md` | 817 evaluable clips, 42 subjects, median sync error ≈ 12 px, first plate heel strike ≈ 0.254 s into each clip |

## Troubleshooting

- **gradio marked broken in nixpkgs** (its `mcp` pin): `flake.nix` already overrides it
  (`problems.handlers` plus `doInstallCheck = false; dontCheckRuntimeDeps = true`).
- **`unzip` fails on Ferber's `ric_data.zip`**: the zip uses Deflate64; `download_data.py` uses `7zz` from the dev shell.
- **Annotated video won't play in the browser**: it needs `ffmpeg` (H.264), which the dev shell provides.
- **"Git tree is dirty" warnings** from `nix develop` are harmless.
