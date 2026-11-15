# Reproducing the project

Runs on CPU from the repo root. Needs 52 GB disk (8 GB without Ferber Clinical Dataset), 16 GB RAM, 6 h end to end.

## Setup

With [Nix](https://nixos.org/download) (flakes enabled), `./py` runs Python in the pinned dev shell:

```sh
./py -c "import torch, gradio; print('ok')"
```

Without Nix, install ffmpeg and 7-Zip, then use `python` in place of `./py`:

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e '.[notebooks]'
python -m ipykernel install --user --name gait --display-name "Python (gait)"
```

## Steps

```sh
./py src/helpers/download_data.py fukuchi toronto kuopio ferber   # ~24 GB
./py src/helpers/build_dataset.py                                 # -> data/processed/
./py src/train.py                                                 # -> outputs/models/bigru_v4/ (~2.5 h)
./py src/helpers/build_kuopio.py                                  # -> data/processed/kuopio_clips.pkl
./py src/helpers/kuopio_noise.py                                  # real pose errors, Kuopio subjects 01-26
./py src/train.py --name bigru_v5 --init bigru_v4 --epochs 25 --lr 3e-4 --real-noise   # final model (~2 h)
./py src/evaluate.py                                              # -> outputs/eval/test/
./py src/evaluate_trials.py                                       # -> outputs/eval/test_trial/
./py src/evaluate_kuopio.py                                       # 24 held-out subjects -> outputs/kuopio/
./py src/helpers/plot_kuopio.py                                   # -> outputs/figures/07_kuopio.png
./py -m jupyter nbconvert --to notebook --execute --inplace notebooks/gait_walkthrough.ipynb
./py src/app.py                                                   # demo on http://127.0.0.1:7860
```

The trained v5 model is included in `outputs/models/bigru_v5/`, so the demo runs without retraining
(its held-out-trial tab still needs `data/processed/`). v5 is v4 fine-tuned with real pose errors from Kuopio;
our run was stopped after 15 epochs and keeps the best-validation checkpoint (epoch 13). `GAIT_MODEL=bigru_v4`
runs any script with v4 instead.

## Expected results

| Check | Expected |
|---|---|
| Kept trials | Fukuchi 1972 / Ferber 2061 / Toronto 48 |
| Splits | train 1205 / val 258 / test 269 subjects |
| `data/processed/kuopio_pose_error.npz` | 347 clips from 18 subjects |
| `outputs/eval/test/events.csv`, σ=0.02, Ferber HS | MAE ≈ 3.8 ms (classic ≈ 11 ms) |
| `outputs/kuopio/summary.md` | 470 evaluable clips, 24 held-out subjects; heel strike MAE ≈ 16 ms |
