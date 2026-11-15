"""BiGRU gait model: per-frame heel-strike/toe-off probabilities and joint angles, plus whole-window heads."""
import numpy as np
import torch
from scipy.signal import find_peaks
from torch import nn

from helpers.canonical import sagittal_angles
from helpers.gaitdata import ANG_NAMES, EVENTS, FS, IN_JOINTS, TARGETS

N_FEAT = 74
SEGMENTS = [(0, 1), (1, 2), (3, 4), (5, 6), (6, 7), (8, 9)]  # thigh, shank, foot per side (IN_JOINTS indices)
# keypoints each angle needs: hip = thigh, knee = thigh + shank, ankle = shank + foot
NEEDS = {"hip": ("hip", "knee"), "knee": ("hip", "knee", "ankle"), "ankle": ("knee", "ankle", "heel", "toe")}
ANG_NEEDS = [[IN_JOINTS.index(f"{n[0]}_{j}") for j in NEEDS[n[2:]]] for n in ANG_NAMES]


def features(x, vis):
    """x [..., T, 10, 2], vis [..., T, 10] -> [..., T, 74]: positions, velocities, visibility, segment
    directions, and the joint angles measured straight from the points (/60) with their validity."""
    v = np.diff(x, axis=-3, prepend=x[..., :1, :, :]) * FS / 10
    v = v * (vis * np.concatenate([vis[..., :1, :], vis[..., :-1, :]], axis=-2))[..., None]
    sh = x.shape[:-2]
    parts = [x.reshape(*sh, -1), v.reshape(*sh, -1), vis]
    for a, b in SEGMENTS:
        d = x[..., b, :] - x[..., a, :]
        ang, ok = np.arctan2(d[..., 1], d[..., 0]), vis[..., a] * vis[..., b]
        parts += [(np.sin(ang) * ok)[..., None], (np.cos(ang) * ok)[..., None]]
    a = sagittal_angles(x.reshape(-1, len(IN_JOINTS), 2))
    ang = np.stack([a[n] for n in ANG_NAMES], -1).reshape(*sh, 6)
    ok = np.stack([vis[..., ix].prod(-1) for ix in ANG_NEEDS], -1)
    parts += [np.where(ok > 0, ang, 0.0) / 60, ok]
    return np.concatenate(parts, -1).astype(np.float32)


class GaitNet(nn.Module):
    def __init__(self, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        d = 2 * hidden
        self.inp = nn.Sequential(nn.Linear(N_FEAT, 64), nn.GELU())
        self.gru = nn.GRU(64, hidden, layers, batch_first=True, bidirectional=True, dropout=dropout)
        # short-range conv (0.3 s) next to the GRU, so event timing follows the feet and not just the rhythm
        self.local = nn.Sequential(nn.Conv1d(N_FEAT, 64, 5, padding=2), nn.GELU(),
                                   nn.Conv1d(64, 64, 5, padding=2), nn.GELU())
        self.event_head = nn.Sequential(nn.Linear(d + 64, 64), nn.GELU(), nn.Linear(64, len(EVENTS)))
        # per angle: a learned value and a gate logit; the gate mixes in the angle measured from the points
        self.angle_head = nn.Sequential(nn.Linear(d + 64, 64), nn.GELU(), nn.Linear(64, 12))
        self.heads = nn.ModuleDict({k: nn.Sequential(nn.Linear(2 * d, 128), nn.GELU(), nn.Dropout(dropout),
                                                     nn.Linear(128, len(v))) for k, v in TARGETS.items()})
        self.register_buffer("ang_mean", torch.zeros(6))   # angle normalisation, set by train.py
        self.register_buffer("ang_std", torch.ones(6))

    def forward(self, f):
        """f [B, T, 74] -> (event logits [B, T, 4], {head: [B, n], "angles": [B, T, 6] z-scored})."""
        h, _ = self.gru(self.inp(f))
        mean = h.mean(1)
        std = ((h - mean[:, None]) ** 2).mean(1).clamp(min=1e-6).sqrt()
        out = {k: head(torch.cat([mean, std], -1)) for k, head in self.heads.items()}
        h = torch.cat([h, self.local(f.transpose(1, 2)).transpose(1, 2)], -1)
        a = self.angle_head(h)
        geom, ok = f[..., -12:-6] * 60, f[..., -6:]
        gate = torch.sigmoid(a[..., 6:]) * ok
        out["angles"] = gate * (geom - self.ang_mean) / self.ang_std + (1 - gate) * a[..., :6]
        return self.event_head(h), out


def decode_events(prob, thresh=0.5, min_sep_s=0.4):
    """Per-frame probabilities [T, 4] -> event times (s), peaks refined to sub-frame with a parabola."""
    out = {}
    for k, name in enumerate(EVENTS):
        p = prob[:, k]
        pk = find_peaks(np.r_[0, p, 0], height=thresh, distance=int(min_sep_s * FS))[0] - 1
        t = pk.astype(float)
        inner = (pk > 0) & (pk < len(p) - 1)
        a, b, c = p[pk[inner] - 1], p[pk[inner]], p[pk[inner] + 1]
        den = a - 2 * b + c
        t[inner] += np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / np.where(den == 0, 1, den), 0)
        out[name] = t / FS
    return out
