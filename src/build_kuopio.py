"""Kuopio clips -> synchronised (video keypoints, mocap truth) pairs: data/processed/kuopio_clips.pkl

    ./py src/build_kuopio.py --estimate-fps 12     # frame rate from 12 subjects (geometry only)
    ./py src/build_kuopio.py                       # sync every subject at that frame rate

Never reads force plates for the sync; they are only stored as the gold events.
"""
from __future__ import annotations

import argparse
import json
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import load_kuopio as K
from canonical import JOINTS, fill_gaps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "kuopio_clips.pkl"
FPS_FILE = ROOT / "outputs" / "kuopio" / "fps_estimate.json"
SYNC_IDX = [JOINTS.index(j) for j in K.SYNC_JOINTS]
J = {n: i for i, n in enumerate(JOINTS)}


def subject_clips(sub: str, invalid: set[str]):
    vids, mocs = {}, {}
    for t in K.trials(sub):
        if t in invalid:
            continue
        kp = K.openpose(sub, t)
        if len(kp) < 30 or np.isnan(kp[:, 8, 0]).mean() > 0.5:
            continue
        try:
            m = K.mocap(sub, t)
        except KeyError:
            return {}, {}
        m["xyz"] = fill_gaps(m["xyz"], max_gap=10)
        vids[t], mocs[t] = K.video_canonical(kp), m
    return vids, mocs


def _sync(sub, invalid, fps):
    warnings.filterwarnings("ignore")
    vids, mocs = subject_clips(sub, invalid)
    if len(vids) < 4:
        return sub, None
    clips = {t: (vids[t][0][:, SYNC_IDX], mocs[t]["xyz"][:, SYNC_IDX, 1:3]) for t in vids}
    fps_, geo, res, diag = K.sync_subject(clips, fps=fps)
    return sub, (fps_, geo, res, diag, vids, mocs)


def records(sub, fps, geo, res, vids, mocs, row) -> list[dict]:
    out = []
    for t, (t0, err) in res.items():
        (px, conf), m = vids[t], mocs[t]
        n = len(px)
        idx = t0 * K.FS_MOCAP + np.arange(n) * K.FS_MOCAP / fps
        xyz = K._interp(m["xyz"], idx)                                  # truth at the video frames
        d = m["direction"]
        full = m["xyz"]
        leg = np.nanmedian(np.concatenate([np.linalg.norm(full[:, J[f"{s}_hip"]] - full[:, J[f"{s}_ankle"]], axis=-1)
                                           for s in "LR"]))
        pel = xyz[:, J["pelvis"], 1] * d
        ok = np.isfinite(pel)
        speed = float(np.polyfit(np.arange(n)[ok] / fps, pel[ok], 1)[0]) if ok.sum() > 10 else np.nan
        out.append(dict(
            subject=sub, trial=t, direction=d, near="R" if t[0] == "r" else "L", fps=fps, t0=t0, sync_err_px=err,
            geo=np.asarray(geo), px=px.astype(np.float32), conf=conf.astype(np.float32),
            truth_sag=K.sagittal(xyz, d).astype(np.float32),                 # (forward, up) m
            truth_px=K._project(xyz[..., 1:3], geo).astype(np.float32),
            events={k: [e - t0 for e in v if -0.5 < e - t0 < n / fps + 0.5] for k, v in m["events"].items()},
            leg_length_m=float(leg), speed_ms=speed,
            age=row.get("Age"), sex=row.get("Gender"), height_cm=row.get("Height")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate-fps", type=int, default=0, help="estimate the frame rate on this many subjects")
    ap.add_argument("--fps", type=float, default=0, help="video frame rate (default: the estimate on file)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    info = K.info().set_index("subject")
    subs = [s for s in info.index if (K.RAW / s / "openpose").exists()]
    inv = {s: K.invalid_trials(info.loc[s]) for s in subs}
    if a.estimate_fps:
        subs = subs[:a.estimate_fps]
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(_sync, subs, [inv[s] for s in subs], [None] * len(subs)))
        curves = {s: dict(r[3]["fps_costs"]) for s, r in res if r}
        grid = sorted(next(iter(curves.values())))
        total = [float(np.mean([c[f] / min(c.values()) for c in curves.values()])) for f in grid]
        est = {s: float(r[0]) for s, r in res if r}
        best = grid[int(np.argmin(total))]
        FPS_FILE.parent.mkdir(parents=True, exist_ok=True)
        FPS_FILE.write_text(json.dumps({"per_subject": est, "grid": grid, "mean_rel_cost": total, "fps": best}, indent=1))
        print("per subject", est)
        print("mean relative cost", dict(zip(grid, np.round(total, 3))))
        print("best", best)
        return
    fps = a.fps or json.loads(FPS_FILE.read_text())["fps"]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(_sync, subs, [inv[s] for s in subs], [fps] * len(subs)))
    recs = []
    for s, r in res:
        if r is None:
            print(s, "skipped (too few usable clips)")
            continue
        f, geo, sync, diag, vids, mocs = r
        recs += records(s, f, geo, sync, vids, mocs, info.loc[s].to_dict())
        print(s, len(sync), "clips", f"median sync error {np.median([e for _, e in sync.values()]):.1f} px")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(recs, open(out, "wb"))
    print(len(recs), "clips from", len({r['subject'] for r in recs}), "subjects ->", out)


if __name__ == "__main__":
    main()
