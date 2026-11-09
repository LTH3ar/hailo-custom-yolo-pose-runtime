"""`hcpose doctor`: check every piece the runtime needs and say how to fix what's missing."""
from __future__ import annotations

import glob
import platform
import shutil
import subprocess

from . import env

OK, BAD, WARN = "\033[32m✓\033[0m", "\033[31m✗\033[0m", "\033[33m!\033[0m"


def _run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return r.returncode, (r.stdout + r.stderr).strip()
    except Exception as e:
        return 1, str(e)


def doctor() -> int:
    problems = 0

    def line(ok, what, detail="", fix=""):
        nonlocal problems
        mark = OK if ok is True else (WARN if ok is None else BAD)
        if ok is False:
            problems += 1
        print(f" {mark} {what}" + (f": {detail}" if detail else ""))
        if fix and ok is not True:
            print(f"     fix: {fix}")

    print(f"hcpose doctor  ({platform.machine()}, Python {platform.python_version()})\n")

    try:
        with open("/proc/device-tree/model") as f:
            line(True, "board", f.read().strip("\x00\n"))
    except OSError:
        line(None, "board", "not a Raspberry Pi (fine for development)")

    if shutil.which("hailortcli"):
        rc, out = _run(["hailortcli", "fw-control", "identify"])
        arch = next((l.split(":", 1)[1].strip() for l in out.splitlines() if "Device Architecture" in l), None)
        fw = next((l.split(":", 1)[1].strip() for l in out.splitlines() if "Firmware Version" in l), None)
        line(rc == 0, "Hailo device", f"{arch}, firmware {fw}" if rc == 0 else out.splitlines()[-1] if out else "",
             "check the AI HAT+ ribbon cable; `lspci | grep Hailo`; reboot after installing hailo-all")
    else:
        line(False, "hailortcli", "not found", "sudo apt install hailo-all && sudo reboot")

    try:
        import hailo_platform as hp

        line(True, "HailoRT python (hailo_platform)", getattr(hp, "__version__", "?"))
    except ImportError as e:
        line(False, "HailoRT python (hailo_platform)", str(e),
             "sudo apt install hailo-all; recreate the venv with --system-site-packages (./install.sh --recreate-venv)")

    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        line(True, "GStreamer", Gst.version_string())
        for el in ("hailonet", "hailofilter", "hailocropper", "hailoaggregator", "hailotracker", "hailooverlay"):
            line(Gst.ElementFactory.find(el) is not None, f"  element {el}", "", "sudo apt install hailo-all (hailo-tappas-core)")
        enc = next((e for e in ("x264enc", "openh264enc", "v4l2h264enc") if Gst.ElementFactory.find(e)), None)
        line(True if enc else None, "  H.264 encoder (for --output)", enc or "none",
             "sudo apt install gstreamer1.0-plugins-ugly")
        line(True if Gst.ElementFactory.find("textoverlay") else None, "  textoverlay (engagement % on screen)",
             "", "sudo apt install gstreamer1.0-x")
        line(True if Gst.ElementFactory.find("libcamerasrc") else None, "  libcamerasrc (Pi camera)",
             "" , "sudo apt install gstreamer1.0-libcamera")
    except Exception as e:
        line(False, "GStreamer python bindings (gi)", str(e), "sudo apt install python3-gi gstreamer1.0-tools")

    try:
        import hailo  # noqa: F401

        line(True, "TAPPAS python bindings (hailo)", "needed for --save-json with the gst engine")
    except ImportError:
        line(None, "TAPPAS python bindings (hailo)", "missing; gst engine can display/record but not --save-json",
             "see README > Troubleshooting, or use --engine python")

    line(env.tappas_version() is not None, "TAPPAS pkg-config", env.tappas_version() or "not found",
         "sudo apt install hailo-all")
    wb = env.find_whole_buffer_cropper()
    line(wb is not None, "TAPPAS libwhole_buffer.so", str(wb or ""), "sudo apt install hailo-all")
    so = env.find_postprocess_so()
    line(so is not None, "hcpose post-process .so", str(so or "not built"), "./install.sh  (or ./install.sh --build-only)")

    try:
        import cv2

        line(True, "OpenCV (python engine)", cv2.__version__)
    except ImportError:
        line(None, "OpenCV (python engine)", "missing", "sudo apt install python3-opencv")
    try:
        import yaml  # noqa: F401

        line(True, "PyYAML", "")
    except ImportError:
        line(False, "PyYAML", "missing", "pip install pyyaml")

    vids = sorted(glob.glob("/dev/video*"))
    from .sources import default_usb_device

    usb = default_usb_device()
    line(True if usb else None, "video devices",
         f"{len(vids)} nodes, USB camera: {usb or 'none found (ISP/codec nodes only)'}" if vids else "none")

    print()
    if problems:
        print(f"{problems} problem(s) found.")
    else:
        print("All required pieces found.")
    return 1 if problems else 0
