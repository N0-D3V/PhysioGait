"""Minimal reader for C3D motion-capture files (nixpkgs has no C3D package): marker positions, analog channels
(force plates) and the parameter section. Intel byte order only, which is what Vicon writes.
"""
import numpy as np

_DTYPE = {-1: "S1", 1: "i1", 2: "<i2", 4: "<f4"}


def _params(buf: bytes) -> dict:
    groups, pending = {}, []
    i = 4
    while i + 2 <= len(buf):
        n, gid = np.frombuffer(buf, "i1", 2, i)
        n = abs(int(n))
        if n == 0:
            break
        name = buf[i + 2:i + 2 + n].decode("latin-1").upper()
        j = i + 2 + n
        nxt = int(np.frombuffer(buf, "<i2", 1, j)[0])
        j += 2
        if gid < 0:
            groups.setdefault(-gid, {})["__name__"] = name
        else:
            t = int(np.frombuffer(buf, "i1", 1, j)[0])
            nd = int(buf[j + 1])
            dims = list(buf[j + 2:j + 2 + nd])
            j += 2 + nd
            size = abs(t) * int(np.prod(dims)) if nd else abs(t)
            raw = buf[j:j + size]
            if t == -1:
                if nd <= 1:
                    val = raw.decode("latin-1").strip()
                else:
                    w = dims[0]
                    val = [raw[k:k + w].decode("latin-1").strip() for k in range(0, len(raw), w)]
            else:
                val = np.frombuffer(raw, _DTYPE[t]).reshape(dims[::-1]) if nd else np.frombuffer(raw, _DTYPE[t])[0]
            pending.append((gid, name, val))
        if nxt == 0:
            break
        i = i + 2 + n + nxt
    out = {}
    for gid, name, val in pending:
        out.setdefault(groups.get(gid, {}).get("__name__", str(gid)), {})[name] = val
    return out


def read(path) -> dict:
    data = open(path, "rb").read()
    pblock = data[0]
    proc = data[(pblock - 1) * 512 + 3]
    if proc != 84:
        raise ValueError(f"{path}: only Intel-format C3D is supported (processor type {proc})")
    h = np.frombuffer(data[:512], "<i2")
    n_pts, n_an_per_frame = int(h[1]), int(h[2])
    first, last = int(h[3]) & 0xFFFF, int(h[4]) & 0xFFFF
    scale = float(np.frombuffer(data[12:16], "<f4")[0])
    start = int(h[8])
    rate = float(np.frombuffer(data[20:24], "<f4")[0])
    nblk = data[(pblock - 1) * 512 + 2]
    P = _params(data[(pblock - 1) * 512:(pblock - 1 + nblk) * 512])
    pt = P.get("POINT", {})
    if "FRAMES" in pt:
        n_frames = int(pt["FRAMES"]) & 0xFFFF
    else:
        n_frames = last - first + 1
    trial = P.get("TRIAL", {})
    if "ACTUAL_END_FIELD" in trial:                     # > 65535 frames
        a = np.asarray(trial["ACTUAL_END_FIELD"], np.int64).ravel()
        b = np.asarray(trial["ACTUAL_START_FIELD"], np.int64).ravel()
        n_frames = int((a[0] & 0xFFFF) + (a[1] << 16) - ((b[0] & 0xFFFF) + (b[1] << 16)) + 1)
    isfloat = scale < 0
    w = 4 if isfloat else 2
    per = n_pts * 4 + n_an_per_frame
    raw = np.frombuffer(data, "<f4" if isfloat else "<i2", n_frames * per, (start - 1) * 512).reshape(n_frames, per)
    pts = raw[:, :n_pts * 4].reshape(n_frames, n_pts, 4).astype(np.float64)
    resid = pts[..., 3]
    if not isfloat:
        pts[..., :3] *= scale
    xyz = pts[..., :3].copy()
    xyz[resid < 0] = np.nan
    xyz[(xyz == 0).all(-1)] = np.nan
    an = raw[:, n_pts * 4:].astype(np.float64)
    an_p = P.get("ANALOG", {})
    n_ch = int(np.asarray(an_p.get("USED", 0)).ravel()[0]) if "USED" in an_p else 0
    if n_ch:
        an = an.reshape(n_frames * (n_an_per_frame // n_ch), n_ch)
        gen = float(np.asarray(an_p.get("GEN_SCALE", 1.0)).ravel()[0])
        sc = np.asarray(an_p.get("SCALE", np.ones(n_ch)), float).ravel()[:n_ch]
        off = np.asarray(an_p.get("OFFSET", np.zeros(n_ch)), float).ravel()[:n_ch]
        an = (an - off) * sc * gen
    labels = pt.get("LABELS", [])
    labels = [labels] if isinstance(labels, str) else list(labels)
    labels += list(pt.get("LABELS2", [])) if "LABELS2" in pt else []
    an_labels = an_p.get("LABELS", [])
    return dict(points=xyz, labels=labels[:n_pts], rate=rate, first_frame=first, analog=an,
                analog_rate=float(np.asarray(an_p.get("RATE", rate)).ravel()[0]) if an_p else rate,
                analog_labels=[an_labels] if isinstance(an_labels, str) else list(an_labels),
                units=pt.get("UNITS", "mm"), params=P, word_size=w)
