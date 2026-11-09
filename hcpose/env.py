"""Locate system pieces (TAPPAS libraries, our post-process .so, GStreamer elements)."""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

PKG_DIR = Path(__file__).resolve().parent
REPO_DIR = PKG_DIR.parent
POSTPROCESS_SO_NAME = "libhcpose_yolov8_pose.so"
TAPPAS_PKGS = ("hailo-tappas-core", "hailo-apps-core", "hailo_tappas")


def _pkg_var(var: str) -> Optional[str]:
    if not shutil.which("pkg-config"):
        return None
    for pkg in TAPPAS_PKGS:
        try:
            out = subprocess.run(
                ["pkg-config", f"--variable={var}", pkg], capture_output=True, text=True, timeout=5
            ).stdout.strip()
        except Exception:
            continue
        if out:
            return out
    return None


def tappas_version() -> Optional[str]:
    if not shutil.which("pkg-config"):
        return None
    for pkg in TAPPAS_PKGS:
        r = subprocess.run(["pkg-config", "--modversion", pkg], capture_output=True, text=True)
        if r.returncode == 0:
            return f"{pkg} {r.stdout.strip()}"
    return None


def find_postprocess_so() -> Optional[Path]:
    candidates = [
        os.environ.get("HCPOSE_POSTPROCESS_SO"),
        PKG_DIR / "lib" / POSTPROCESS_SO_NAME,
        REPO_DIR / "build" / POSTPROCESS_SO_NAME,
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    return None


def find_whole_buffer_cropper() -> Optional[Path]:
    """libwhole_buffer.so is the TAPPAS cropper used to letterbox the frame for hailonet."""
    dirs = [
        os.environ.get("TAPPAS_POST_PROC_DIR"),
        _pkg_var("tappas_postproc_lib_dir"),
        "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes",
        "/usr/lib/x86_64-linux-gnu/hailo/tappas/post_processes",
        "/usr/local/hailo/resources/so",
    ]
    for d in dirs:
        if not d:
            continue
        p = Path(d) / "cropping_algorithms" / "libwhole_buffer.so"
        if p.is_file():
            return p
    hits = glob.glob("/usr/lib/**/libwhole_buffer.so", recursive=True)
    return Path(hits[0]) if hits else None


def gst_has(element: str) -> bool:
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        return Gst.ElementFactory.find(element) is not None
    except Exception:
        return False
