"""Pure-Python engine: OpenCV I/O + HailoRT + NumPy decoder.

Slower than the GStreamer engine but has no native build step, handles single
images, and is the reference path for debugging and evaluation.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .config import ModelConfig
from .decode import Pose, PoseDecoder
from .hef import HefInfo, import_hailo_platform
from .layout import Layout
from .results import FpsCounter, FrameCallback, FrameResult, Person
from .sources import classify, usb_device
from .tiling import make_tiles, merge, offset, parse_region, parse_spec
from .tracker import IouTracker

log = logging.getLogger("hcpose")


def letterbox(img: np.ndarray, w: int, h: int, pad: int = 114) -> Tuple[np.ndarray, float, int, int]:
    import cv2

    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = int(round(iw * s)), int(round(ih * s))
    out = np.full((h, w, 3), pad, dtype=np.uint8)
    px, py = (w - nw) // 2, (h - nh) // 2
    out[py : py + nh, px : px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out, s, px, py


def poses_to_persons(
    poses: List[Pose], layout: Layout, scale: float, px: int, py: int, fw: int, fh: int, labels: List[str]
) -> List[Person]:
    out = []
    for p in poses:
        b = p.box.copy()
        b[[0, 2]] = (b[[0, 2]] * layout.input_width - px) / scale
        b[[1, 3]] = (b[[1, 3]] * layout.input_height - py) / scale
        b[[0, 2]] = np.clip(b[[0, 2]], 0, fw)
        b[[1, 3]] = np.clip(b[[1, 3]], 0, fh)
        k = p.keypoints.copy()
        k[:, 0] = np.clip((k[:, 0] * layout.input_width - px) / scale, 0, fw)
        k[:, 1] = np.clip((k[:, 1] * layout.input_height - py) / scale, 0, fh)
        label = labels[p.class_id] if p.class_id < len(labels) else f"class_{p.class_id}"
        out.append(Person(None, p.score, p.class_id, label, b.tolist(), k))
    return out


STATE_COLORS = {"focused": (0, 200, 0), "distracted": (0, 0, 230), "unknown": (200, 200, 200)}  # BGR


def draw(frame: np.ndarray, persons: List[Person], skeleton, kpt_thr: float,
         display: str = "none", class_text: Optional[str] = None) -> None:
    import cv2

    for p in persons:
        x1, y1, x2, y2 = (int(v) for v in p.box)
        state = (p.engagement or {}).get("state") if display != "none" else None
        if display == "class":
            color, tag = STATE_COLORS["unknown"], ""  # no per-student state on screen
        elif state:
            color, tag = STATE_COLORS[state], (state if state != "unknown" else "?")
        else:
            color = (0, 200, 255)
            tag = f"{p.label} {p.score:.2f}" + (f" #{p.track_id}" if p.track_id is not None else "")
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if tag:
            cv2.putText(frame, tag, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        k = p.keypoints
        for a, b in skeleton:
            if k[a, 2] >= kpt_thr and k[b, 2] >= kpt_thr:
                cv2.line(frame, (int(k[a, 0]), int(k[a, 1])), (int(k[b, 0]), int(k[b, 1])), (255, 120, 0), 2, cv2.LINE_AA)
        for x, y, v in k:
            if v >= kpt_thr:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1, cv2.LINE_AA)
    if class_text and display in ("class", "both"):
        (tw, th), _ = cv2.getTextSize(class_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (8, 8), (16 + tw, 18 + th), (0, 0, 0), -1)
        cv2.putText(frame, class_text, (12, 12 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def frames(source: str, width: int, height: int, fps: int) -> Iterator[Tuple[np.ndarray, Optional[float]]]:
    """Yields (BGR frame, media time in seconds or None for live sources)."""
    import cv2

    kind = classify(source)
    if kind == "image":
        img = cv2.imread(os.path.expanduser(source))
        if img is None:
            raise FileNotFoundError(source)
        yield img, 0.0
        return
    if kind == "rpi":
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise RuntimeError("python engine needs picamera2 for the Pi camera (sudo apt install python3-picamera2)") from e
        if not Picamera2.global_camera_info():
            raise RuntimeError("no Pi camera detected (check the ribbon cable; `rpicam-hello --list-cameras`)")
        cam = Picamera2()
        cam.configure(cam.create_video_configuration(main={"size": (width, height), "format": "RGB888"}))
        cam.start()
        try:
            while True:
                yield cam.capture_array(), None  # RGB888 in picamera2 is BGR byte order
        finally:
            cam.stop()
        return
    if kind == "usb":
        dev = usb_device(source)
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
    else:
        cap = cv2.VideoCapture(os.path.expanduser(source))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open input {source}")
    file_fps = cap.get(cv2.CAP_PROP_FPS) if kind == "file" else 0
    n = 0
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield f, (n / file_fps if file_fps and file_fps > 0 else None)
            n += 1
    finally:
        cap.release()


class PyRunner:
    def __init__(
        self,
        cfg: ModelConfig,
        info: HefInfo,
        layout: Layout,
        source: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        display: bool = True,
        output: Optional[str] = None,
        tracker: bool = True,
        overlay: bool = True,
        callbacks: Optional[List[FrameCallback]] = None,
        show_fps: bool = True,
        engagement=None,
        tiles: Optional[str] = None,
        tile_region: str = "0:0.6",
    ):
        self.cfg, self.info, self.layout = cfg, info, layout
        self.source, self.width, self.height, self.fps = source, width, height, fps
        self.display = display and bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if display and not self.display:
            print("[hcpose] no display available; running headless")
        self.output = output
        self.interrupted = False
        self.tracker = IouTracker() if tracker else None
        self.overlay = overlay
        self.callbacks = list(callbacks or [])
        self.engagement = engagement
        self.tile_grid = parse_spec(tiles)
        self.tile_region = parse_region(tile_region)
        self._tiles_cache = {}
        self.display_mode = engagement.s.display if engagement else "none"
        if engagement:
            self.callbacks.insert(0, engagement)
        self.fps_counter = FpsCounter() if show_fps else None
        self.decoder = PoseDecoder(
            layout,
            score_threshold=cfg.score_threshold,
            iou_threshold=cfg.iou_threshold,
            max_detections=cfg.max_detections,
            class_agnostic_nms=cfg.class_agnostic_nms,
            score_activation=cfg.score_activation,
            keypoint_activation=cfg.keypoint_activation,
            keypoint_threshold=cfg.keypoint_threshold,
            pose_dedup=cfg.pose_dedup,
        )

    def run(self) -> int:
        import cv2

        hp = import_hailo_platform()
        hef = hp.HEF(self.cfg.hef)
        writer = None
        is_image = classify(self.source) == "image"
        t0 = time.monotonic()
        with hp.VDevice() as vdev:
            cp = hp.ConfigureParams.create_from_hef(hef, interface=hp.HailoStreamInterface.PCIe)
            ng = vdev.configure(hef, cp)[0]
            ivp = hp.InputVStreamParams.make(ng, format_type=hp.FormatType.UINT8)
            ovp = hp.OutputVStreamParams.make(ng, format_type=hp.FormatType.FLOAT32)
            in_h, in_w = self.layout.input_height, self.layout.input_width
            with ng.activate(ng.create_params()), hp.InferVStreams(ng, ivp, ovp) as pipe:
                try:
                    for idx, (frame, media_t) in enumerate(frames(self.source, self.width, self.height, self.fps)):
                        fh, fw = frame.shape[:2]
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        net_in, s, px, py = letterbox(rgb, in_w, in_h)
                        tiles = self._tiles(fw, fh)
                        crops = [letterbox(rgb[t.y0:t.y1, t.x0:t.x1], in_w, in_h) for t in tiles]
                        batch = np.stack([net_in] + [c[0] for c in crops])
                        out = pipe.infer({self.info.input_name: batch})
                        if idx == 0:
                            self._report_activation(out)
                        poses = self.decoder({k: v[0] for k, v in out.items()})
                        persons = poses_to_persons(poses, self.layout, s, px, py, fw, fh, self.cfg.labels)
                        if tiles:
                            tiled = []
                            for j, (t, (_, ts, tpx, tpy)) in enumerate(zip(tiles, crops), start=1):
                                tp = self.decoder({k: v[j] for k, v in out.items()})
                                tiled += [(offset(q, t), t) for q in
                                          poses_to_persons(tp, self.layout, ts, tpx, tpy, t.w, t.h, self.cfg.labels)]
                            persons = merge(persons, tiled, fw, fh)
                        if self.tracker:
                            for p, tid in zip(persons, self.tracker.update([p.box for p in persons])):
                                p.track_id = tid
                        t = media_t if media_t is not None else time.monotonic() - t0
                        res = FrameResult(idx, t, fw, fh, persons)
                        for cb in self.callbacks:
                            cb(res, frame)
                        if self.overlay and (self.display or self.output):
                            draw(frame, persons, self.cfg.skeleton, self.cfg.keypoint_threshold, self.display_mode,
                                 self.engagement.class_text() if self.engagement else None)
                        if self.output:
                            if is_image:
                                cv2.imwrite(self.output, frame)
                            else:
                                if writer is None:
                                    writer = cv2.VideoWriter(self.output, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (fw, fh))
                                writer.write(frame)
                        if is_image:
                            for p in persons:
                                state = f" {p.engagement['state']}" if p.engagement else ""
                                print(f"  {p.label} {p.score:.2f} box={[round(v) for v in p.box]}{state}")
                        if self.display:
                            cv2.imshow("hcpose", frame)
                            if self._wait_key(is_image) in (27, ord("q")):
                                break
                        if self.fps_counter:
                            self.fps_counter.tick(self.engagement.class_text() if self.engagement
                                                  else f"persons: {len(persons)}")
                except KeyboardInterrupt:
                    print("\n[hcpose] stopped")
                    self.interrupted = True
        if writer is not None:
            writer.release()
        if self.display:
            cv2.destroyAllWindows()
        if self.fps_counter and self.fps_counter.total > 1:
            print(f"[hcpose] {self.fps_counter.summary()}")
        if self.output:
            print(f"[hcpose] saved {os.path.abspath(self.output)}")
        return 130 if self.interrupted else 0

    @staticmethod
    def _wait_key(still: bool) -> int:
        """Key press in the window. For a still image, wait until a key is pressed or the
        window is closed, in short slices: one long cv2.waitKey(0) blocks inside the GUI
        toolkit, where Python can't see Ctrl+C."""
        import cv2

        if not still:
            return cv2.waitKey(1) & 0xFF
        while True:
            k = cv2.waitKey(100) & 0xFF
            if k != 0xFF:
                return k
            try:
                if cv2.getWindowProperty("hcpose", cv2.WND_PROP_VISIBLE) < 1:
                    return 0  # window closed: continue
            except cv2.error:
                return 0

    def _tiles(self, w: int, h: int):
        if not self.tile_grid:
            return []
        if (w, h) not in self._tiles_cache:
            self._tiles_cache[(w, h)] = make_tiles(w, h, self.tile_grid, self.tile_region)
            ts = self._tiles_cache[(w, h)]
            print(f"[hcpose] tiling: full frame + {len(ts)} tile(s) over rows {self.tile_region[0]:.0%}-"
                  f"{self.tile_region[1]:.0%} ({ts[0].w}x{ts[0].h} px each)")
        return self._tiles_cache[(w, h)]

    def _report_activation(self, out) -> None:
        self.decoder({k: v[0] for k, v in out.items()})  # resolves auto activations
        print(
            f"[hcpose] class scores: {'logits -> sigmoid applied here' if self.decoder.score_sigmoid else 'already sigmoid in HEF'}; "
            f"keypoint visibility: {'logits -> sigmoid applied here' if self.decoder.kpt_sigmoid else 'already sigmoid in HEF'}"
        )
