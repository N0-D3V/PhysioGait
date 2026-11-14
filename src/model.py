"""Multi-head BiGRU gait model.

[B, T, F] -> input MLP -> BiGRU (shared encoder)
  -> event head: per-frame logits for HS_L, HS_R, TO_L, TO_R
  -> per-head MLPs on masked mean+std pooled features: spatiotemporal, speed, symmetry,
     ROM, variability (z-scored targets)

  -> angle head (v2, per frame): the 6 sagittal joint angles (denoised), from which ROM is
     measured per stride with the predicted events

Features per frame (v1): 10 joints x (x, y), their velocities, and 10 visibility flags;
v2 adds sin/cos of the 6 segment angles (thigh, shank, foot per side), zeroed when hidden;
v3 adds the 6 geometric joint angles (label definitions, /60) and their validity flags.

v3 model: a local conv branch (receptive field 9 frames = 0.3 s) feeds the event and angle
heads next to the BiGRU, so event timing can follow local foot kinematics rather than the
global rhythm; the angle head is a gated skip: gate * geometric + (1 - gate) * learned, with
the gate forced to 0 where a needed keypoint is hidden. Clean pose -> exact geometry,
noisy pose -> denoised estimate.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.signal import find_peaks
from torch import nn

from canonical import sagittal_angles
from gaitdata import ANG_NAMES, EVENTS, FS, IN_JOINTS, TARGETS

N_FEAT = {"v1": 10 * 2 * 2 + 10, "v2": 10 * 2 * 2 + 10 + 12, "v3": 10 * 2 * 2 + 10 + 12 + 12}
GEOM_SCALE = 60.0
# keypoints each angle needs: hip = thigh, knee = thigh + shank, ankle = shank + foot
_NEEDS = {"hip": ("hip", "knee"), "knee": ("hip", "knee", "ankle"), "ankle": ("knee", "ankle", "heel", "toe")}
ANG_NEEDS = [[IN_JOINTS.index(f"{n[0]}_{j}") for j in _NEEDS[n[2:]]] for n in ANG_NAMES]
# segments (from, to) in IN_JOINTS order: thigh hip->knee, shank knee->ankle, foot heel->toe
SEGMENTS = [(0, 1), (1, 2), (3, 4), (5, 6), (6, 7), (8, 9)]


def features(x: np.ndarray, vis: np.ndarray, version: str = "v2") -> np.ndarray:
    """x [..., T, 10, 2], vis [..., T, 10] -> [..., T, N_FEAT[version]]."""
    v = np.diff(x, axis=-3, prepend=x[..., :1, :, :]) * FS / 10
    both = vis * np.concatenate([vis[..., :1, :], vis[..., :-1, :]], axis=-2)
    v = v * both[..., None]
    sh = x.shape[:-2]
    parts = [x.reshape(*sh, -1), v.reshape(*sh, -1), vis]
    if version in ("v2", "v3"):
        for a, b in SEGMENTS:
            d = x[..., b, :] - x[..., a, :]
            ok = vis[..., a] * vis[..., b]
            ang = np.arctan2(d[..., 1], d[..., 0])
            parts += [(np.sin(ang) * ok)[..., None], (np.cos(ang) * ok)[..., None]]
    if version == "v3":
        g, ok = geom_angles(x, vis)
        parts += [g / GEOM_SCALE, ok]
    return np.concatenate(parts, -1).astype(np.float32)


def geom_angles(x: np.ndarray, vis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Joint angles (deg) computed exactly as the labels are -> (angles [..., T, 6] zeroed where
    a needed keypoint is hidden, validity [..., T, 6])."""
    sh = x.shape[:-2]
    a = sagittal_angles(x.reshape(-1, len(IN_JOINTS), 2))
    ang = np.stack([a[n] for n in ANG_NAMES], -1).reshape(*sh, 6)
    ok = np.stack([vis[..., ix].prod(-1) for ix in ANG_NEEDS], -1)
    return np.where(ok > 0, ang, 0.0), ok


class GaitNet(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 2, dropout: float = 0.2,
                 heads: dict[str, list[str]] = TARGETS, feat: str = "v2", angle_head: bool = True):
        super().__init__()
        self.feat = feat
        self.inp = nn.Sequential(nn.Linear(N_FEAT[feat], 64), nn.GELU())
        self.gru = nn.GRU(64, hidden, layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        d = 2 * hidden
        dl = 0
        if feat == "v3":
            dl = 64
            self.local = nn.Sequential(nn.Conv1d(N_FEAT[feat], dl, 5, padding=2), nn.GELU(),
                                       nn.Conv1d(dl, dl, 5, padding=2), nn.GELU())
            # angle normalisation, so the geometric angles can be mixed in z units (set by train.py)
            self.register_buffer("ang_mean", torch.zeros(6))
            self.register_buffer("ang_std", torch.ones(6))
        self.event_head = nn.Sequential(nn.Linear(d + dl, 64), nn.GELU(), nn.Linear(64, len(EVENTS)))
        self.heads = nn.ModuleDict({
            k: nn.Sequential(nn.Linear(2 * d, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, len(v)))
            for k, v in heads.items()})
        self.head_names = list(heads)
        n_ang = 12 if feat == "v3" else 6     # v3: learned angle + gate logit per angle
        self.angle_head = nn.Sequential(nn.Linear(d + dl, 64), nn.GELU(), nn.Linear(64, n_ang)) if angle_head else None

    def forward(self, f: torch.Tensor, mask: torch.Tensor):
        """f [B,T,F], mask [B,T] (1 = real frame) -> (event logits [B,T,4], {head: [B,n]}).

        With the angle head, the heads dict also holds "angles" [B,T,6] (z-scored).

        Batches are built from equal-length windows (packing is ~5x slower on CPU), so the
        mask only matters if a caller pads; it is honoured in pooling and losses.
        """
        h, _ = self.gru(self.inp(f))
        m = mask[..., None]
        n = m.sum(1).clamp(min=1)
        mean = (h * m).sum(1) / n
        std = (((h - mean[:, None]) ** 2 * m).sum(1) / n).clamp(min=1e-6).sqrt()
        pooled = torch.cat([mean, std], -1)
        out = {k: self.heads[k](pooled) for k in self.head_names}
        if self.feat == "v3":
            h = torch.cat([h, self.local(f.transpose(1, 2)).transpose(1, 2)], -1)
        if self.angle_head is not None:
            a = self.angle_head(h)
            if self.feat == "v3":
                geom = f[..., -12:-6] * GEOM_SCALE
                ok = f[..., -6:]
                gate = torch.sigmoid(a[..., 6:]) * ok
                a = gate * (geom - self.ang_mean) / self.ang_std + (1 - gate) * a[..., :6]
                out["gate"] = gate
            out["angles"] = a
        return self.event_head(h), out


def decode_events(prob: np.ndarray, thresh: float = 0.5, min_sep_s: float = 0.4) -> dict[str, np.ndarray]:
    """Per-frame probabilities [T, 4] -> event times (s), parabolic sub-frame refinement."""
    out = {}
    for k, name in enumerate(EVENTS):
        p = prob[:, k]
        pk, _ = find_peaks(np.r_[0, p, 0], height=thresh, distance=int(min_sep_s * FS))
        pk = pk - 1
        t = pk.astype(float)
        inner = (pk > 0) & (pk < len(p) - 1)
        a, b, c = p[pk[inner] - 1], p[pk[inner]], p[pk[inner] + 1]
        den = a - 2 * b + c
        t[inner] += np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / np.where(den == 0, 1, den), 0)
        out[name] = t / FS
    return out
