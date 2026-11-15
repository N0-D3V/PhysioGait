"""Training augmentation: make clean mocap look like video pose (noise), mirror left/right, and create
synthetic limps (one leg shifted in time or bending more/less). Units: leg lengths (0.01 ≈ 8 mm)."""
from dataclasses import dataclass

import numpy as np

from helpers.gaitdata import ANG_NAMES, EVENTS, FS, IN_JOINTS, derive, soft_events

HIPS = [IN_JOINTS.index("L_hip"), IN_JOINTS.index("R_hip")]
DISTAL = {s: [IN_JOINTS.index(f"{s}_{j}") for j in ("knee", "ankle", "heel", "toe")] for s in "LR"}
SIDE_IDX = {s: [i for i, j in enumerate(IN_JOINTS) if j.startswith(s)] for s in "LR"}


@dataclass
class Noise:
    """Video-like pose error: per-frame jitter, a constant offset per keypoint, dropped keypoints, a noisier far
    leg hidden where it crosses the near one, and camera effects (leg-length scale error, tilt)."""
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


def apply(x: np.ndarray, rng: np.random.Generator, nz: Noise, far: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """x [T, 10, 2] clean -> (noisy x, visibility [T, 10]). far: force the far-from-camera leg ("L"/"R")."""
    T, Jn, _ = x.shape
    x = x.copy()
    vis = np.ones((T, Jn), np.float32)
    if nz == Noise():
        return x, vis
    mult = np.ones(Jn)
    far = rng.choice(["L", "R"]) if far is None else far
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


# ---------------------------------------------------------------- left/right mirror
JOINT_SWAP = [IN_JOINTS.index(("R" if j[0] == "L" else "L") + j[1:]) for j in IN_JOINTS]
EVENT_SWAP = [1, 0, 3, 2]          # HS_L, HS_R, TO_L, TO_R
ANGLE_SWAP = [3, 4, 5, 0, 1, 2]    # L_hip, L_knee, L_ankle, R_hip, R_knee, R_ankle


def swap_sides(x, ev, ang, y, si_idx):
    """Relabel left as right: valid for sagittal mocap (no near/far leg). SI labels change sign."""
    y = y.copy()
    y[si_idx] = -y[si_idx]
    return x[:, JOINT_SWAP], ev[:, EVENT_SWAP], ang[:, ANGLE_SWAP], y


# ---------------------------------------------------------------- synthetic limps
# Most training people walk almost symmetrically, so without this the model learns to report everyone as
# symmetric. Labels are recomputed from the modified events and angles.
LEG = {s: [IN_JOINTS.index(f"{s}_{j}") for j in ("hip", "knee", "ankle", "heel", "toe")] for s in "LR"}


def _rotate(p, centre, deg):
    """Rotate points p [T, k, 2] about centre [T, 2] by deg [T] (CCW = the angles' positive direction)."""
    a = np.radians(deg)[:, None]
    d = p - centre[:, None]
    c, s = np.cos(a), np.sin(a)
    return centre[:, None] + np.stack([c * d[..., 0] - s * d[..., 1], s * d[..., 0] + c * d[..., 1]], -1)


def scale_leg(x, ang, side, k_hip, k_knee, k_ankle):
    """Scale one leg's hip/knee/ankle excursions about their window means by rigid rotations.

    hip: rotate knee..toe about the hip; knee: rotate ankle..toe about the knee (knee = thigh - shank,
    so the shank turns the other way); ankle: rotate heel, toe about the ankle.
    """
    x, ang = x.copy(), ang.copy()
    h, k, a, he, to = LEG[side]
    for name, kk, pivot, moved, sign in (("hip", k_hip, h, [k, a, he, to], 1), ("knee", k_knee, k, [a, he, to], -1),
                                         ("ankle", k_ankle, a, [he, to], 1)):
        i = ANG_NAMES.index(f"{side}_{name}")
        d = (kk - 1) * (ang[:, i] - ang[:, i].mean())
        x[:, moved] = _rotate(x[:, moved], x[:, pivot], sign * d)
        ang[:, i] += d
    return x, ang


def asymmetric_window(tr, i0, i1, rng, max_shift_s=0.05, k_range=(0.8, 1.25)):
    """A training window of trial `tr` with one leg phase-shifted and its excursions scaled.

    -> (pose [n,10,2], soft events [n,4], angles [n,6], labels) with labels recomputed.
    """
    side = "L" if rng.random() < 0.5 else "R"
    n, T = i1 - i0, len(tr.pose)
    shift = rng.uniform(-max_shift_s, max_shift_s)
    t = np.clip(np.arange(i0, i1) - shift * FS, 0, T - 1)       # leg shown `shift` s later
    lo = np.floor(t).astype(int)
    hi, w = np.minimum(lo + 1, T - 1), (t - lo)[:, None]
    x, ang = tr.pose[i0:i1].copy(), tr.ang[i0:i1].copy()
    lj = LEG[side]
    x[:, lj] = tr.pose[lo][:, lj] * (1 - w[..., None]) + tr.pose[hi][:, lj] * w[..., None]
    ai = [ANG_NAMES.index(f"{side}_{j}") for j in ("hip", "knee", "ankle")]
    ang[:, ai] = tr.ang[lo][:, ai] * (1 - w) + tr.ang[hi][:, ai] * w
    ev = {e: tr.events[e] - i0 / FS + (shift if e.endswith(side) else 0) for e in EVENTS}
    ks = np.exp(rng.uniform(*np.log(k_range), 3))
    x, ang = scale_leg(x, ang, side, *ks)
    y = derive(ev, ang, n, tr.speed_ll)
    return x.astype(np.float32), soft_events(ev, n), ang.astype(np.float32), y
