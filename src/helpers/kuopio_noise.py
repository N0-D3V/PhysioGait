"""Record real pose errors from Kuopio: for each clip, real video keypoints minus the mocap joint positions, after the
same steps as the demo. Training replays these error tracks on mocap (augment.Noise.real), so the model sees
real pose error: correlated in time, offset at the foot, with gross mistakes. Only the "fit" subjects (01-26) are
used; the rest of Kuopio stays held out for testing.

    ./py src/helpers/kuopio_noise.py   -> data/processed/kuopio_pose_error.npz
"""
import pickle

import numpy as np

from helpers.analysis import ROOT
from helpers.canonical import resample

CLIPS = ROOT / "data" / "processed" / "kuopio_clips.pkl"
OUT = ROOT / "data" / "processed" / "kuopio_pose_error.npz"


def is_fit(subject: str) -> bool:
    """Kuopio subjects used to fit the noise model; evaluate_kuopio.py --half test uses the others."""
    return int(subject) <= 26


def main():
    from evaluate_kuopio import mocap_input, video_input
    clips = [c for c in pickle.load(open(CLIPS, "rb")) if is_fit(c["subject"])]
    errs, viss, nears, subjects = [], [], [], []
    for c in clips:
        sag = resample(c["truth_sag"], c["fps"])
        if not np.isfinite(sag).all():
            continue
        xv, vv, _ = video_input(c)
        xt, _ = mocap_input(sag)
        T = min(len(xv), len(xt))
        e = (xv[:T] - xt[:T]).astype(np.float32)
        e[vv[:T] == 0] = 0
        errs.append(e), viss.append(vv[:T]), nears.append(c["near"]), subjects.append(c["subject"])
    lens = np.array([len(e) for e in errs])
    np.savez_compressed(OUT, err=np.concatenate(errs), vis=np.concatenate(viss), lens=lens,
                        near=np.array(nears), subject=np.array(subjects))
    print(f"{len(errs)} clips from {len(set(subjects))} subjects, {lens.sum()} frames -> {OUT}")


if __name__ == "__main__":
    main()
