"""Turn a user-facing --input value into a source kind + GStreamer snippet."""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def classify(src: str) -> str:
    s = src.strip()
    if s in ("rpi", "libcamera", "picam", "csi"):
        return "rpi"
    if s == "usb" or s.startswith("/dev/video"):
        return "usb"
    if s.startswith(("rtsp://", "rtsps://")):
        return "rtsp"
    if s.startswith(("http://", "https://")):
        return "uri"
    p = Path(s).expanduser()
    if p.suffix.lower() in IMAGE_EXT:
        return "image"
    return "file"


def default_usb_device() -> Optional[str]:
    """First /dev/video* that is a USB camera (Pi 5 also exposes many ISP nodes), or None."""
    for dev in sorted(glob.glob("/dev/video*"), key=lambda d: int(d[10:]) if d[10:].isdigit() else 999):
        name = Path(f"/sys/class/video4linux/{os.path.basename(dev)}/device")
        try:
            if "usb" in os.path.realpath(name):
                return dev
        except OSError:
            continue
    return None


def usb_device(src: str) -> str:
    """Resolve `--input usb` / `/dev/videoN` to a device path, with a clear error if there is none."""
    if src != "usb":
        if not os.path.exists(src):
            raise FileNotFoundError(f"{src} does not exist (list cameras with: v4l2-ctl --list-devices)")
        return src
    dev = default_usb_device()
    if dev is None:
        raise FileNotFoundError(
            "no USB camera found (the /dev/video* nodes on a Pi 5 are ISP/codec nodes, not cameras). "
            "Plug one in, or pass --input /dev/videoN explicitly."
        )
    return dev


def gst_source(src: str, width: int, height: int, fps: int, camera_raw: bool = False) -> str:
    kind = classify(src)
    if kind == "rpi":
        return f"libcamerasrc name=src ! video/x-raw, width={width}, height={height}, framerate={fps}/1 ! "
    if kind == "usb":
        dev = usb_device(src)
        if camera_raw:
            return f"v4l2src name=src device={dev} ! video/x-raw, framerate={fps}/1 ! "
        return (
            f"v4l2src name=src device={dev} ! image/jpeg, width={width}, height={height}, framerate={fps}/1 ! "
            "queue name=src_q leaky=no max-size-buffers=3 ! decodebin ! "
        )
    if kind == "rtsp":
        return f'rtspsrc name=src location="{src}" latency=200 ! queue name=src_q ! decodebin ! '
    if kind == "uri":
        return f'uridecodebin name=src uri="{src}" ! '
    path = str(Path(src).expanduser().resolve())
    if not os.path.isfile(path):
        raise FileNotFoundError(f"input not found: {path}")
    src = f'filesrc name=src location="{path}" ! queue name=src_q max-size-buffers=5 ! decodebin ! '
    if os.environ.get("HCPOSE_PACE_FILE") == "1":
        # research/edge latency runs: release decoded frames at the clip's frame rate, like a
        # live camera, so downstream queues stay empty and the latency from live_pace to the
        # sink is processing time, not the backlog of a decoder running ahead (use --no-sync).
        src += "identity name=live_pace sync=true ! "
    return src
