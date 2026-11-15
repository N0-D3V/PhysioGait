# Gait measurement from a side perspective walking video

Side perspective walking clip → 2D sagittal pose → gait measurements (left/right symmetry, timing,
joint range of motion), validated with MoCap and force plates on subjects excluded from the training dataset, including
real video keypoints. Should be used for measurement and progress tracking, not diagnosis.

*This project is a prototype and is not medical advice.*

```sh
./py src/app.py        # Gradio demo on http://127.0.0.1:7860 (CPU; --share for a public link)
```

`./py` runs Python inside the Nix flake's dev shell. The demo needs `outputs/models/bigru_v4/` and
`outputs/eval/test_trial/regression.csv` (see `REPRODUCE.md`, not in git by default), and downloads the two pose
models (150 MB) on first use.

- **[REPRODUCE.md](REPRODUCE.md)**: every step from download to demo, with expected checks.
- **[notebooks/gait_walkthrough.ipynb](notebooks/gait_walkthrough.ipynb)**: the whole project explained with relevant charts and figures.
- `src/model_findings.md`: model ablation (v1 to v4).

## Code map (`src/`)

The model and everything that runs it are directly in `src/`. Data loading, processing and plotting helpers (which contain ai assisted code)
are in `src/helpers/`.

| Stage | Files |
|---|---|
| Model, pose-noise and asymmetry augmentation, training | `model.py`, `augment.py`, `train.py` |
| Baseline and evaluation (per window, per recording, real video) | `classic.py`, `evaluate.py`, `evaluate_trials.py`, `evaluate_kuopio.py` |
| Video → pose → model → report and demo | `pose_video.py`, `video_pipeline.py`, `app.py` |
| Data: download, per-dataset loaders, C3D reader, skeleton and labels, processed datasets | `helpers/download_data.py`, `helpers/load_{fukuchi,ferber,toronto,kuopio}.py`, `helpers/c3d.py`, `helpers/canonical.py`, `helpers/build_dataset.py`, `helpers/build_kuopio.py` |
| Model data, splits, the stride and label formulas | `helpers/gaitdata.py` |
| Paths and plot style, demo figures and videos, real-video figure | `helpers/analysis.py`, `helpers/render.py`, `helpers/plot_kuopio.py` |

## Recording tips
- Camera at hip height and perpendicular to the walking line
- whole body including the feet in frame
- at least 20 s of steady walking

Data: Fukuchi WBDS, Ferber Running Injury Clinic, Toronto Older Adults, Kuopio (all CC BY; full citations
in project writeup). Pose models: RTMPose and YOLOX (OpenMMLab, Apache-2.0). Demo video: Yoon et al. 2012,
CC BY 2.0.
