"""Multi-head BiGRU gait model.

[B, T, F] -> input MLP -> BiGRU (shared encoder)
  -> event head: per-frame logits for HS_L, HS_R, TO_L, TO_R
  -> per-head MLPs on masked mean+std pooled features: spatiotemporal, speed, symmetry,
     ROM, variability (z-scored targets)

Features per frame: 10 joints x (x, y), their velocities, and 10 visibility flags.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.signal import find_peaks
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from gaitdata import EVENTS, FS, TARGETS

N_FEAT = 10 * 2 * 2 + 10


def features(x: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """x [..., T, 10, 2], vis [..., T, 10] -> [..., T, 50]."""
    v = np.diff(x, axis=-3, prepend=x[..., :1, :, :]) * FS / 10
    both = vis * np.concatenate([vis[..., :1, :], vis[..., :-1, :]], axis=-2)
    v = v * both[..., None]
    sh = x.shape[:-2]
    return np.concatenate([x.reshape(*sh, -1), v.reshape(*sh, -1), vis], -1).astype(np.float32)


class GaitNet(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 2, dropout: float = 0.2,
                 heads: dict[str, list[str]] = TARGETS):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(N_FEAT, 64), nn.GELU())
        self.gru = nn.GRU(64, hidden, layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        d = 2 * hidden
        self.event_head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, len(EVENTS)))
        self.heads = nn.ModuleDict({
            k: nn.Sequential(nn.Linear(2 * d, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, len(v)))
            for k, v in heads.items()})
        self.head_names = list(heads)

    def forward(self, f: torch.Tensor, mask: torch.Tensor):
        """f [B,T,F], mask [B,T] (1 = real frame) -> (event logits [B,T,4], {head: [B,n]})."""
        lengths = mask.sum(1).long().cpu()
        h = self.inp(f)
        packed = pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.gru(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=f.shape[1])
        m = mask[..., None]
        n = m.sum(1).clamp(min=1)
        mean = (h * m).sum(1) / n
        std = (((h - mean[:, None]) ** 2 * m).sum(1) / n).clamp(min=1e-6).sqrt()
        pooled = torch.cat([mean, std], -1)
        return self.event_head(h), {k: self.heads[k](pooled) for k in self.head_names}


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
