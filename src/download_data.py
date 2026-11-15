"""Download the four datasets, each into its own directory under data/ (run from the repo root).

    ./py src/download_data.py fukuchi toronto ferber kuopio      # or any subset; default: all

| dataset  | directory                                  | download | on disk  |
|----------|--------------------------------------------|----------|----------|
| fukuchi  | data/fukuchi_wbds/                         | ~0.4 GB  | ~2 GB    |
| toronto  | data/toronto_older_adults/                 | ~70 MB   | ~0.3 GB  |
| ferber   | data/Ferber Running Injury Dataset Full/   | 22.8 GB  | ~70 GB   |
| kuopio   | data/kuopio/  (comfortable-speed trials)   | ~0.5 GB  | ~1.2 GB  |

Ferber's ric_data.zip uses Deflate64, which Python's zipfile cannot read: it is extracted with 7-Zip
(`7zz`, in the flake dev shell). Kuopio is 23 GB of zips, mostly raw camera files; only the OpenPose
json and the walking-trial .c3d files are read out of the remote zips with HTTP range requests.
Archives are only listed/extracted by this script, never executed.
"""
from __future__ import annotations

import io
import re
import shutil
import struct
import subprocess
import sys
import urllib.request
import zipfile
import zlib
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
FIG = "https://ndownloader.figshare.com/files/"
ZEN = "https://zenodo.org/records/10559504/files/"

FILES = {
    "fukuchi": ("fukuchi_wbds", {"WBDSinfo.csv": FIG + "68504641", "WBDSascii.zip": FIG + "68194516"}),
    "toronto": ("toronto_older_adults", {"Participants.xlsx": FIG + "35454230", "Xsens.zip": FIG + "35454227",
                                         "PoseTracking.zip": FIG + "28868364"}),
    "ferber": ("Ferber Running Injury Dataset Full", {"README.txt": FIG + "43420746",
                                                      "walk_data_meta.csv": FIG + "42922405",
                                                      "ric_data.zip": FIG + "42651352"}),
    "kuopio": ("kuopio", {"readme.txt": ZEN + "readme.txt?download=1",
                          "info_participants.xlsx": ZEN + "info_participants.xlsx?download=1"}),
}
KUOPIO_ZIPS = ["measurement_data_1_to_17.zip", "measurement_data_18_to_34.zip", "measurement_data_35_to_51.zip"]
KUOPIO_KEEP = re.compile(r"^\d\d/(mocap/([lr]_comf_\d\d|calib_static)\.c3d|openpose/json_[lr]_comf_\d\d/)")


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name}")
        return
    print(f"  get  {dest.name}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "gait-prototype"}), timeout=120) as r, \
            open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.rename(dest)


def extract(zpath: Path, to: Path) -> None:
    print(f"  extract {zpath.name}", flush=True)
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():                      # refuse paths escaping the dataset directory
            if not (to / m.filename).resolve().is_relative_to(to.resolve()):
                raise ValueError(f"unsafe path in {zpath.name}: {m.filename}")
        z.extractall(to)


# ---------------------------------------------------------------- remote zip (HTTP range requests)
class RemoteFile(io.RawIOBase):
    def __init__(self, url: str):
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.url, self.pos = url, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read_range(self, a: int, b: int) -> bytes:
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b - 1}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except OSError:
                if attempt == 4:
                    raise
        raise RuntimeError

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        d = self.read_range(self.pos, min(self.pos + len(b), self.size))
        b[:len(d)] = d
        self.pos += len(d)
        return len(d)


def remote_extract(url: str, to: Path, keep: re.Pattern) -> None:
    rf = RemoteFile(url)
    z = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 16))
    todo = [m for m in z.infolist() if keep.search(m.filename) and not m.is_dir()
            and not ((to / m.filename).exists() and (to / m.filename).stat().st_size == m.file_size)]
    todo.sort(key=lambda m: m.header_offset)
    print(f"  {len(todo)} files, {sum(m.compress_size for m in todo) / 1e6:.0f} MB", flush=True)
    k = 0
    while k < len(todo):                             # neighbouring members in one request
        j, a = k, todo[k].header_offset
        while (j + 1 < len(todo) and todo[j + 1].header_offset - (todo[j].header_offset + todo[j].compress_size) < 1 << 16
               and todo[j + 1].header_offset - a < 64 << 20):
            j += 1
        buf = rf.read_range(a, min(todo[j].header_offset + 30 + 1024 + todo[j].compress_size, rf.size))
        for m in todo[k:j + 1]:
            o = m.header_offset - a
            n1, n2 = struct.unpack("<HH", buf[o + 26:o + 30])
            d = buf[o + 30 + n1 + n2:o + 30 + n1 + n2 + m.compress_size]
            d = zlib.decompress(d, -15) if m.compress_type == zipfile.ZIP_DEFLATED else d
            if len(d) != m.file_size or zlib.crc32(d) != m.CRC:
                raise IOError(f"bad member {m.filename}")
            p = to / m.filename
            if not p.resolve().is_relative_to(to.resolve()):
                raise ValueError(f"unsafe path {m.filename}")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(d)
        k = j + 1


def main(names: list[str]) -> None:
    for name in names or list(FILES):
        sub, files = FILES[name]
        d = DATA / sub
        d.mkdir(parents=True, exist_ok=True)
        print(f"{name} -> {d}")
        for fn, url in files.items():
            fetch(url, d / fn)
        if name == "fukuchi" and not (d / "51subjs").exists():
            extract(d / "WBDSascii.zip", d)
        if name == "toronto":
            if not (d / "Xsens").exists():
                extract(d / "Xsens.zip", d)
            if not (d / "Pose Tracking").exists():
                extract(d / "PoseTracking.zip", d)
        if name == "ferber" and not (d / "extracted" / "reformat_data").exists():
            exe = shutil.which("7zz") or shutil.which("7z")
            if not exe:
                sys.exit("ferber: needs 7-Zip (7zz) for Deflate64; run inside `nix develop`")
            subprocess.run([exe, "x", "-y", f"-o{d / 'extracted'}", str(d / "ric_data.zip")], check=True)
        if name == "kuopio":
            for zname in KUOPIO_ZIPS:
                print(f"  {zname} (remote, selected members)")
                remote_extract(ZEN + zname + "?download=1", d / "raw", KUOPIO_KEEP)


if __name__ == "__main__":
    main(sys.argv[1:])
