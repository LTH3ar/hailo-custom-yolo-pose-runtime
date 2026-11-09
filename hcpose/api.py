"""Python entry point for building on top of hcpose (e.g. engagement analysis).

    from hcpose.api import run

    def on_frame(result, frame):
        for p in result.persons:
            print(p.track_id, p.keypoints[5:7])   # shoulders, pixel coords + visibility

    run("models/upper_body_n.hef", source="/dev/video0", on_frame=on_frame, display=False)
"""
from __future__ import annotations

from typing import Optional

from .config import load_config
from .hef import read_hef
from .results import FrameCallback


def run(
    hef: str,
    source: str = "usb",
    on_frame: Optional[FrameCallback] = None,
    engine: str = "gst",
    sidecar: Optional[str] = None,
    **kwargs,
) -> int:
    """Run a model and call on_frame(FrameResult, frame_or_None) for every frame.

    engine="gst" gives frame=None (pixels stay in native code); engine="python"
    passes the BGR frame. Extra kwargs go to the runner (width, height, fps,
    display, output, tracker, overlay, show_fps, engagement=EngagementAnalyzer(...),
    and score_threshold etc. which override the sidecar).
    """
    cfg_keys = {"score_threshold", "iou_threshold", "keypoint_threshold", "max_detections",
                "score_activation", "keypoint_activation"}
    overrides = {k: kwargs.pop(k) for k in list(kwargs) if k in cfg_keys}
    cfg = load_config(hef, sidecar, overrides)
    info = read_hef(cfg.hef)
    layout = info.layout(cfg.num_keypoints, cfg.num_classes, cfg.reg_max)
    cfg.fill_from_layout(layout.num_keypoints, layout.num_classes)
    callbacks = [on_frame] if on_frame else []
    if engine == "gst":
        from .gst_engine import GstRunner

        return GstRunner(cfg, layout, source=source, callbacks=callbacks, **kwargs).run()
    from .py_engine import PyRunner

    return PyRunner(cfg, info, layout, source=source, callbacks=callbacks, **kwargs).run()
