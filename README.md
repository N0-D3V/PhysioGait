# Gait measurement from a side-on walking video

From an X-second side-on walking clip → 2D pose → gait measurements a physio cannot eyeball: interlimb symmetry,
spatiotemporal parameters and joint range of motion, validated against gold-standard motion capture on held-out
subjects. Measurement and progress tracking, not diagnosis.

*This project is a prototype and is not medical advice.*

## Run

Everything runs inside the Nix flake (CPU only). `./py` runs Python in the dev shell.

```sh
./py src/app.py            # Gradio demo on http://127.0.0.1:7860   (--share for a public link)
```

The first video analysis downloads the two pose models (~150 MB, OpenMMLab) into `outputs/pose_models/`.
The app also needs the trained model `outputs/models/bigru_v4/` and the evaluation table
`outputs/eval/test_trial/regression.csv` (both produced by the commands below; `outputs/` is not in git).

## Pipeline

| Step | Code |
|---|---|
| Video → keypoints: YOLOX person detector + RTMPose-m Halpe-26 (heels, toes), ONNX on CPU | `src/pose_video.py` |
| Keypoints → model input: L/R swap repair, walking passes, pelvis-centred, leg-length-scaled, 30 Hz | `src/video_pipeline.py` |
| BiGRU (v4): per-frame gait events + gated joint-angle curves | `src/model.py` |
| Events + angles → strides → outputs, with the same formulas as the labels | `src/gaitdata.py` (`derive`) |
| Figures, annotated video | `src/render.py` |
| Demo | `src/app.py` |

Data, training and evaluation:

```sh
./py src/build_dataset.py                                              # datasets -> data/processed
./py src/train.py --name bigru_v4 --features v3 --angle-beta 0.1 --asym-p 0.5
./py src/evaluate.py bigru_v4 bigru_v3 bigru_v2 bigru_v2_noswap bigru_aug bigru_clean
./py src/evaluate_trials.py bigru_v4 bigru_v3 && ./py src/evaluate_trials.py bigru_v4 bigru_v3 --two-pass
./py src/make_model_notebook.py                                        # notebooks/06_model_results.ipynb
```

Results and limitations: `notebooks/06_model_results.ipynb` and `src/model_findings.md`.

## Recording tips

Camera at hip height, perpendicular to the walking line, whole body including the feet in frame, ≥ 20 s of steady
walking. Walk across **and back**, so each leg is measured from the pass where it faces the camera.

## Sources

- Fukuchi CA, Fukuchi RK, Duarte M (2018). Walking biomechanics dataset (WBDS). PeerJ / Figshare, CC BY.
- Ferber R et al. Running Injury Clinic gait database, walking trials (Figshare 24255795; Sci Data 2024).
- Toronto Older Adults Gait Archive (Sci Data 2022, doi:10.1038/s41597-022-01495-z), CC BY.
- Pose models: RTMPose and YOLOX (OpenMMLab MMPose / MMDetection, Apache-2.0).
- Demo video: Yoon J, Park H-S, Damiano DL (2012), J NeuroEng Rehabil 9:62, CC BY 2.0, via Wikimedia Commons
  (`data/demo_videos/ATTRIBUTION.md`).
