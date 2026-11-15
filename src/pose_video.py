"""Video -> 2D keypoints: YOLOX person detector + RTMPose (Halpe-26, has heels and toes), ONNX on CPU.

Models (downloaded once into outputs/pose_models/, see MODELS):
  yolox-m-humanart.onnx    person boxes; mmdeploy end2end export with NMS inside
  rtmpose-m-halpe26.onnx   26 keypoints incl. big toe, small toe and heel per foot (SimCC head)

    from pose_video import PoseTracker
    kp, score, fps, size = PoseTracker().run("walk.mp4")   # kp [T, 26, 2] pixels, score [T, 26]
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "outputs" / "pose_models"
_OMM = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
MODELS = {
    "yolox-m-humanart.onnx": _OMM + "yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
    "rtmpose-m-halpe26.onnx": _OMM + "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
}
HALPE26 = ["nose", "L_eye", "R_eye", "L_ear", "R_ear", "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
           "L_wrist", "R_wrist", "L_hip", "R_hip", "L_knee", "R_knee", "L_ankle", "R_ankle", "head", "neck",
           "hip", "L_big_toe", "R_big_toe", "L_small_toe", "R_small_toe", "L_heel", "R_heel"]
H = {n: i for i, n in enumerate(HALPE26)}


def ensure_models() -> None:
    """Download and unpack the two ONNX files if they are missing."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        if (MODEL_DIR / name).exists():
            continue
        with urllib.request.urlopen(url, timeout=120) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        member = next(m for m in z.namelist() if m.endswith("end2end.onnx"))
        (MODEL_DIR / name).write_bytes(z.read(member))


def _session(name: str, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(str(MODEL_DIR / name), so, providers=["CPUExecutionProvider"])


class Detector:
    """YOLOX-m (HumanArt): image (BGR) -> person boxes [N, 5] (x1, y1, x2, y2, score)."""

    def __init__(self, threads: int = 4, size: int = 640):
        self.s, self.size = _session("yolox-m-humanart.onnx", threads), size

    def __call__(self, img: np.ndarray, thresh: float = 0.3) -> np.ndarray:
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        pad = np.full((self.size, self.size, 3), 114, np.uint8)
        pad[:int(h * r), :int(w * r)] = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        x = pad.transpose(2, 0, 1)[None].astype(np.float32)          # BGR, no normalisation (pipeline.json)
        dets, labels = self.s.run(None, {self.s.get_inputs()[0].name: x})
        d, lab = dets[0], labels[0]
        keep = (lab == 0) & (d[:, 4] > thresh)
        d = d[keep].copy()
        d[:, :4] /= r
        return d


class PoseEstimator:
    """RTMPose-m Halpe-26, top-down: (image, box) -> keypoints [26, 2] (pixels), scores [26]."""

    W, H_, PAD = 192, 256, 1.25
    MEAN = np.array([123.675, 116.28, 103.53], np.float32)
    STD = np.array([58.395, 57.12, 57.375], np.float32)

    def __init__(self, threads: int = 4):
        self.s = _session("rtmpose-m-halpe26.onnx", threads)

    def _affine(self, box):
        x1, y1, x2, y2 = box[:4]
        c = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        w, h = (x2 - x1) * self.PAD, (y2 - y1) * self.PAD
        if w / h > self.W / self.H_:                     # fix the aspect ratio to the model input
            h = w * self.H_ / self.W
        else:
            w = h * self.W / self.H_
        src = np.float32([c, c + [0, -h / 2], c + [-w / 2, -h / 2]])
        dst = np.float32([[self.W / 2, self.H_ / 2], [self.W / 2, 0], [0, 0]])
        return cv2.getAffineTransform(src, dst)

    def __call__(self, img: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        M = self._affine(box)
        crop = cv2.warpAffine(img, M, (self.W, self.H_), flags=cv2.INTER_LINEAR)
        x = ((crop[..., ::-1].astype(np.float32) - self.MEAN) / self.STD).transpose(2, 0, 1)[None]
        sx, sy = self.s.run(None, {self.s.get_inputs()[0].name: x})
        sx, sy = sx[0], sy[0]                            # [26, 384], [26, 512]: SimCC, split ratio 2
        px, py = sx.argmax(1) / 2.0, sy.argmax(1) / 2.0
        score = np.minimum(sx.max(1), sy.max(1))       # as mmpose get_simcc_maximum
        Mi = cv2.invertAffineTransform(M)
        pts = np.c_[px, py, np.ones(len(px))] @ Mi.T
        return pts.astype(np.float32), score.astype(np.float32)


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def _box_from_keypoints(k: np.ndarray, sc: np.ndarray, size, thresh: float = 0.3, margin: float = 0.12):
    ok = sc > thresh
    if ok.sum() < 8:
        return None
    x1, y1 = k[ok].min(0)
    x2, y2 = k[ok].max(0)
    mw, mh = margin * (x2 - x1), margin * (y2 - y1)
    return np.array([max(0, x1 - mw), max(0, y1 - mh), min(size[0], x2 + mw), min(size[1], y2 + mh)])


class PoseTracker:
    """Single-subject keypoints for a whole video.

    The subject is the largest confident person in the first detection. Between detections
    (every `det_every` frames, or sooner when the keypoints become unsure) the box comes from the
    previous frame's keypoints, the usual top-down tracking scheme; the detector dominates the
    cost (~270 ms vs ~30 ms for the pose model on CPU).
    """

    def __init__(self, threads: int = 4, det_every: int = 15):
        ensure_models()
        self.det, self.pose, self.det_every = Detector(threads), PoseEstimator(threads), det_every

    def _detect(self, img, box):
        d = self.det(img)
        if not len(d):
            return None
        if box is None:
            area = (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])
            return d[np.argmax(area * d[:, 4]), :4]
        ious = np.array([_iou(box, b) for b in d])
        return d[np.argmax(ious), :4] if ious.max() > 0.1 else None

    def run(self, path: str, max_seconds: float = 90.0, progress=None):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"cannot open video {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        n_max = int(max_seconds * fps)
        kps, scores, boxes = [], [], []
        box, size, i, since = None, None, 0, 0
        while i < n_max:
            ok, img = cap.read()
            if not ok:
                break
            size = img.shape[1], img.shape[0]
            if box is None or since >= self.det_every:
                det = self._detect(img, box)
                if det is not None:
                    box, since = det, 0
            if box is None:
                kps.append(np.full((26, 2), np.nan, np.float32))
                scores.append(np.zeros(26, np.float32))
                boxes.append(np.full(4, np.nan))
            else:
                k, s = self.pose(img, box)
                kps.append(k), scores.append(s), boxes.append(box.copy())
                nxt = _box_from_keypoints(k, s, size)
                if nxt is None or float(np.median(s)) < 0.35:
                    since = self.det_every                   # unsure: detect again next frame
                else:
                    box, since = nxt, since + 1
            i += 1
            if progress is not None and i % 15 == 0:
                progress(min(i / max(min(n_total, n_max), 1), 1.0), desc=f"pose: frame {i}")
        cap.release()
        if not kps:
            raise ValueError("no frames could be read from the video")
        return np.stack(kps), np.stack(scores), float(fps), size, np.stack(boxes)
