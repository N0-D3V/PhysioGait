"""Figures and annotated videos for the demo (matplotlib + OpenCV + ffmpeg)."""
from __future__ import annotations

import shutil
import subprocess

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analysis import INK2, MUTED, SIDE_COL, style  # noqa: E402
from gaitdata import ANG_NAMES, FS, IN_JOINTS, STRIDE_COLS  # noqa: E402

style()
BONES = [("hip", "knee"), ("knee", "ankle"), ("ankle", "heel"), ("heel", "toe"), ("ankle", "toe")]
J = {n: i for i, n in enumerate(IN_JOINTS)}


def _bgr(hex_):
    h = hex_.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))


SIDE_BGR = {s: _bgr(c) for s, c in SIDE_COL.items()}


# ---------------------------------------------------------------- phases from events
def stance_intervals(events: dict, side: str) -> list[tuple[float, float]]:
    """Heel strike -> next toe-off of the same foot."""
    hs, to = np.sort(events[f"HS_{side}"]), np.sort(events[f"TO_{side}"])
    out = []
    for h in hs:
        nxt = to[to > h]
        nh = hs[hs > h]
        if len(nxt) and (not len(nh) or nxt[0] < nh[0]) and nxt[0] - h < 1.5:
            out.append((float(h), float(nxt[0])))
    return out


def phase_at(t: float, stance: list) -> str | None:
    for a, b in stance:
        if a <= t <= b:
            return "stance"
    return "swing" if stance and stance[0][0] - 1.5 < t < stance[-1][1] + 1.5 else None


# ---------------------------------------------------------------- figures
def gait_diagram(rep, t_max: float | None = None):
    """Stance bars per foot over time; shaded = double support."""
    fig, ax = plt.subplots(figsize=(11, 2.3))
    st = {s: stance_intervals(rep.events, s) for s in "LR"}
    for k, s in enumerate("LR"):
        for a, b in st[s]:
            ax.barh(1 - k, b - a, left=a, height=0.55, color=SIDE_COL[s], edgecolor="white", linewidth=1)
    for a, b in st["L"]:
        for c, d in st["R"]:
            lo, hi = max(a, c), min(b, d)
            if hi > lo:
                ax.axvspan(lo, hi, color=MUTED, alpha=0.18, lw=0)
    for a, b, d, near in rep.passes:
        ax.axvline(a / FS, color=INK2, lw=0.8, ls=":")
    ax.set_yticks([1, 0], ["Left foot", "Right foot"])
    ax.set_xlabel("time (s)")
    ax.set_title("Gait diagram: bars = foot on the ground (stance), gaps = swing, grey = both feet down")
    if t_max:
        ax.set_xlim(0, t_max)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def angle_plot(rep):
    fig, axs = plt.subplots(3, 1, figsize=(11, 6.2), sharex=True)
    t = np.arange(len(rep.angles)) / FS
    for a, j in zip(axs, ("hip", "knee", "ankle")):
        for s in "LR":
            a.plot(t, rep.angles[:, ANG_NAMES.index(f"{s}_{j}")], color=SIDE_COL[s], lw=1.6,
                   label=f"{'Left' if s == 'L' else 'Right'}")
            for h in rep.events[f"HS_{s}"]:
                a.axvline(h, color=SIDE_COL[s], lw=0.6, alpha=0.35)
        a.set_ylabel(f"{j} (deg)")
    axs[0].set_title("Joint angles (model output); thin lines = heel strikes")
    axs[0].legend(loc="upper right", ncol=2, fontsize=8)
    axs[-1].set_xlabel("time (s)")
    fig.tight_layout()
    return fig


def symmetry_plot(rep):
    """Left vs right per measure, every stride shown as a dot (the strides behind each mean)."""
    items = [("step_time", "Step time (s)"), ("stance_time", "Stance time (s)"), ("swing_time", "Swing time (s)"),
             ("hip_rom", "Hip ROM (deg)"), ("knee_rom", "Knee ROM (deg)"), ("ankle_rom", "Ankle ROM (deg)")]
    S = rep.strides
    fig, axs = plt.subplots(1, len(items), figsize=(13, 3.2))
    si = rep.summary.set_index("key")
    rng = np.random.default_rng(0)
    for a, (c, lab) in zip(axs, items):
        for k, s in enumerate("LR"):
            v = S[S.side == s][c].to_numpy()
            if len(v):
                a.bar(k, v.mean(), width=0.6, color=SIDE_COL[s], alpha=0.85)
                a.scatter(k + rng.uniform(-0.15, 0.15, len(v)), v, s=10, color=INK2, zorder=3, lw=0)
        lo = np.nanmin(S[c]) if len(S) else 0
        hi = np.nanmax(S[c]) if len(S) else 1
        pad = 0.25 * (hi - lo + 1e-9)
        a.set_ylim(max(0, lo - 3 * pad), hi + pad)
        a.set_xticks([0, 1], ["L", "R"])
        sik = f"SI_{c}"
        sub = f"SI {si.loc[sik, 'value']:+.1f} %" if sik in si.index and np.isfinite(si.loc[sik, "value"]) else ""
        a.set_title(f"{lab}\n{sub}", fontsize=9)
        a.grid(axis="x", visible=False)
    fig.suptitle("Left vs right (bars = mean, dots = individual strides; SI > 0 = left larger)", fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- videos
def _writer(path: str, w: int, h: int, fps: float):
    """ffmpeg H.264 pipe (browser-playable); falls back to OpenCV mp4v."""
    w, h = w - w % 2, h - h % 2
    if shutil.which("ffmpeg"):
        p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                              "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264",
                              "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "24", "-movflags", "+faststart",
                              path], stdin=subprocess.PIPE)
        return (lambda img: p.stdin.write(np.ascontiguousarray(img[:h, :w]).tobytes())), \
               (lambda: (p.stdin.close(), p.wait()))
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    return (lambda img: vw.write(img[:h, :w])), vw.release


def _draw_skeleton(img, pts, vis, thick=3):
    for s in "LR":
        c = SIDE_BGR[s]
        for a, b in BONES:
            i, j = J[f"{s}_{a}"], J[f"{s}_{b}"]
            if vis[i] and vis[j]:
                cv2.line(img, tuple(map(int, pts[i])), tuple(map(int, pts[j])), c, thick, cv2.LINE_AA)
        for i in (J[f"{s}_{n}"] for n in ("hip", "knee", "ankle", "heel", "toe")):
            if vis[i]:
                cv2.circle(img, tuple(map(int, pts[i])), thick + 2, c, -1, cv2.LINE_AA)
                cv2.circle(img, tuple(map(int, pts[i])), thick + 2, (255, 255, 255), 1, cv2.LINE_AA)


def _strip(rep, w: int, t: float, t_max: float, h: int = 46) -> np.ndarray:
    """Gait-diagram strip (stance bars per foot) with a moving cursor."""
    img = np.full((h, w, 3), 250, np.uint8)
    st = {s: stance_intervals(rep.events, s) for s in "LR"}
    px = lambda tt: int(8 + (w - 16) * tt / max(t_max, 1e-6))  # noqa: E731
    for k, s in enumerate("LR"):
        y0 = 6 + k * 19
        cv2.putText(img, s, (px(0) - 6, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.0001, (0, 0, 0), 1)
        for a, b in st[s]:
            cv2.rectangle(img, (px(a), y0), (max(px(a) + 1, px(b)), y0 + 14), SIDE_BGR[s], -1)
    cv2.line(img, (px(t), 0), (px(t), h), (40, 40, 40), 2)
    return img


def _banner(img, t, st):
    y = 30
    for s in "LR":
        ph = phase_at(t, st[s])
        if ph is None:
            continue
        txt = f"{'Left' if s == 'L' else 'Right'}: {ph}"
        cv2.putText(img, txt, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(img, txt, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, SIDE_BGR[s], 2, cv2.LINE_AA)
        y += 30


def annotate_video(src: str, rep, out: str, max_width: int = 960, max_seconds: float = 90.0) -> str:
    """Source video + detected skeleton (after the L/R fix) + phase banner + gait-diagram strip."""
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W0, H0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sc = min(1.0, max_width / W0)
    w, h = int(W0 * sc), int(H0 * sc)
    t_max = len(rep.px) / FS
    write, close = _writer(out, w, h + 46, fps)
    st = {s: stance_intervals(rep.events, s) for s in "LR"}
    vis = rep.vis_px if getattr(rep, "vis_px", None) is not None else rep.vis
    i = 0
    while i < max_seconds * fps:
        ok, img = cap.read()
        if not ok:
            break
        if sc != 1.0:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        t = i / fps
        k = min(int(round(t * FS)), len(rep.px) - 1)
        _draw_skeleton(img, rep.px[k] * sc, vis[k] > 0, thick=max(2, int(3 * w / 960)))
        _banner(img, t, st)
        write(np.vstack([img, _strip(rep, w, t, t_max)]))
        i += 1
    cap.release()
    close()
    return out


def stick_video(rep, out: str, size=(640, 420), scale: float = 170.0) -> str:
    """Animated stick figure of the model input (pelvis fixed), for held-out mocap trials."""
    w, h = size
    write, close = _writer(out, w, h + 46, FS)
    st = {s: stance_intervals(rep.events, s) for s in "LR"}
    origin = np.array([w / 2, h * 0.28])
    t_max = len(rep.x) / FS
    for k in range(len(rep.x)):
        img = np.full((h, w, 3), 252, np.uint8)
        pts = origin + rep.x[k] * np.array([scale, -scale])
        ground = origin[1] + scale * 1.08
        cv2.line(img, (0, int(ground)), (w, int(ground)), (200, 200, 200), 1)
        _draw_skeleton(img, pts, rep.vis[k] > 0)
        _banner(img, k / FS, st)
        write(np.vstack([img, _strip(rep, w, k / FS, t_max)]))
    close()
    return out
