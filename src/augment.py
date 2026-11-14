"""Pose-noise augmentation: make clean mocap look like video pose-estimator output.

All in the model's units (pelvis-centred, leg-length-scaled; 0.01 ≈ 8 mm on an adult):
  - per-frame Gaussian keypoint jitter (`sigma`)
  - per-joint constant offset: keypoints sit on the skin/shoe, not on joint centres (`bias`)
  - random keypoint dropout (`dropout`): coordinates zeroed, visibility flag 0
  - far-leg occlusion: the leg away from the camera is noisier and disappears where it
    crosses behind the near leg
  - camera effects: leg-length estimate error (scale) and a slightly tilted camera (rotation)
The pelvis origin is re-derived from the *noisy* hips, as it would be from a pose estimator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gaitdata import IN_JOINTS

HIPS = [IN_JOINTS.index("L_hip"), IN_JOINTS.index("R_hip")]
DISTAL = {s: [IN_JOINTS.index(f"{s}_{j}") for j in ("knee", "ankle", "heel", "toe")] for s in "LR"}
SIDE_IDX = {s: [i for i, j in enumerate(IN_JOINTS) if j.startswith(s)] for s in "LR"}


@dataclass
class Noise:
    sigma: float = 0.0
    bias: float = 0.0
    dropout: float = 0.0
    far_leg: bool = False
    far_mult: float = 1.5
    cross_thresh: float = 0.06   # far joint hidden when within this horizontal distance of the near one
    scale_sd: float = 0.0
    rot_deg_sd: float = 0.0

    @staticmethod
    def level(sigma: float) -> "Noise":
        """Evaluation preset: one knob from clean (0) to very noisy (0.08)."""
        if sigma == 0:
            return Noise()
        return Noise(sigma=sigma, bias=sigma / 2, dropout=min(sigma, 0.1), far_leg=True,
                     scale_sd=sigma, rot_deg_sd=50 * sigma)

    @staticmethod
    def random(rng: np.random.Generator) -> "Noise":
        """Training distribution: 20 % clean, otherwise a random mix up to roughly level 0.06."""
        if rng.random() < 0.2:
            return Noise()
        s = rng.uniform(0.0, 0.06)
        return Noise(sigma=s, bias=rng.uniform(0, s), dropout=rng.uniform(0, 0.1),
                     far_leg=rng.random() < 0.5, scale_sd=0.05, rot_deg_sd=3.0)


def apply(x: np.ndarray, rng: np.random.Generator, nz: Noise) -> tuple[np.ndarray, np.ndarray]:
    """x [T, 10, 2] clean -> (noisy x, visibility [T, 10])."""
    T, Jn, _ = x.shape
    x = x.copy()
    vis = np.ones((T, Jn), np.float32)
    if nz == Noise():
        return x, vis
    mult = np.ones(Jn)
    far = rng.choice(["L", "R"])
    near = "R" if far == "L" else "L"
    if nz.far_leg:
        mult[SIDE_IDX[far]] = nz.far_mult
    x += rng.normal(0, nz.bias, (1, Jn, 2)) * mult[None, :, None]
    x += rng.normal(0, nz.sigma, (T, Jn, 2)) * mult[None, :, None]
    # pelvis origin from the noisy hips
    x -= x[:, HIPS].mean(1, keepdims=True)
    if nz.scale_sd:
        x *= np.exp(rng.normal(0, nz.scale_sd))
    if nz.rot_deg_sd:
        a = np.radians(rng.normal(0, nz.rot_deg_sd))
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        x = x @ R.T
    if nz.far_leg:
        for jf, jn in zip(DISTAL[far], DISTAL[near]):
            vis[np.abs(x[:, jf, 0] - x[:, jn, 0]) < nz.cross_thresh, jf] = 0
    if nz.dropout:
        vis[rng.random((T, Jn)) < nz.dropout] = 0
    x[vis == 0] = 0
    return x.astype(np.float32), vis
