"""GStreamer engine: hailonet + the generic C++ hailofilter + hailotracker + hailooverlay.

This is the fast path (decode, scaling, inference and post-process all run
in native code). Python only sees per-frame metadata when a callback or the
JSONL writer is enabled.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import tempfile
from collections import deque
import time
from typing import List, Optional

from . import env
from .config import ModelConfig
from .layout import Layout
from .results import FpsCounter, FrameCallback, FrameResult, Person
from .sources import classify, gst_source

log = logging.getLogger("hcpose")


def _q(name: str, size: int = 3, leaky: str = "no") -> str:
    return f"queue name={name} leaky={leaky} max-size-buffers={size} max-size-bytes=0 max-size-time=0"


def pick_encoder(Gst) -> Optional[str]:
    for factory, props in (
        ("x264enc", "tune=zerolatency speed-preset=ultrafast bitrate=6000 key-int-max=60"),
        ("openh264enc", "bitrate=6000000"),
        ("v4l2h264enc", ""),
        ("avenc_h264_omx", ""),
    ):
        if Gst.ElementFactory.find(factory):
            return f"{factory} {props}".strip()
    return None


class GstRunner:
    def __init__(
        self,
        cfg: ModelConfig,
        layout: Layout,
        source: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        display: bool = True,
        output: Optional[str] = None,
        sync: Optional[bool] = None,
        batch_size: int = 1,
        tracker: bool = True,
        overlay: bool = True,
        camera_raw: bool = False,
        callbacks: Optional[List[FrameCallback]] = None,
        show_fps: bool = True,
        engagement=None,
    ):
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import GLib, Gst

        Gst.init(None)
        self.Gst, self.GLib = Gst, GLib
        self.cfg, self.layout = cfg, layout
        self.source = source
        self.kind = classify(source)
        if self.kind == "image":
            raise ValueError("single images are handled by the python engine (use --engine python)")
        self.width, self.height, self.fps = width, height, fps
        self.display, self.output = display, output
        # Files play at their own frame rate when displayed; otherwise run as fast as possible.
        self.sync = sync if sync is not None else (self.kind == "file" and display and not output)
        self.batch_size = batch_size
        self.tracker, self.overlay, self.camera_raw = tracker, overlay, camera_raw
        self.callbacks = list(callbacks or [])
        # engagement analysis runs first so the JSONL writer / user callbacks see its labels
        self.engagement = engagement
        self.display_mode = engagement.s.display if engagement else "none"
        if engagement:
            self.callbacks.insert(0, engagement)
        self.fps_counter = FpsCounter() if show_fps else None
        self._text_overlay = None
        self._text_last = None
        self.frame_index = 0
        self._raw_dets: deque = deque(maxlen=64)  # (pts, persons) snapshots taken before hailotracker
        self.t_start = None
        self.exit_code = 0
        self._stopping = False

        self.so_path = env.find_postprocess_so()
        if not self.so_path:
            raise RuntimeError(
                f"{env.POSTPROCESS_SO_NAME} not found. Run ./install.sh (it builds the C++ post-process), "
                "or use --engine python."
            )
        self.cropper = env.find_whole_buffer_cropper()
        if not self.cropper:
            raise RuntimeError("TAPPAS libwhole_buffer.so not found (part of hailo-tappas-core). Run `hcpose doctor`.")
        missing = [e for e in ("hailonet", "hailofilter", "hailocropper", "hailoaggregator") if not Gst.ElementFactory.find(e)]
        if tracker:
            missing += [e for e in ("hailotracker",) if not Gst.ElementFactory.find(e)]
        if overlay:
            missing += [e for e in ("hailooverlay",) if not Gst.ElementFactory.find(e)]
        if self.kind == "rpi" and not Gst.ElementFactory.find("libcamerasrc"):
            raise RuntimeError(
                "--input rpi needs the libcamerasrc GStreamer element: sudo apt install gstreamer1.0-libcamera "
                "(or use --engine python, which uses picamera2)"
            )
        if missing:
            raise RuntimeError(f"missing GStreamer elements: {missing}. Is hailo-all installed? Run `hcpose doctor`.")

        self.has_textoverlay = Gst.ElementFactory.find("textoverlay") is not None
        if self.display_mode in ("class", "both") and overlay and (display or output) and not self.has_textoverlay:
            log.warning("textoverlay element missing (gstreamer1.0-plugins-base pango); class percentage shown in the console only")

        if self.callbacks:
            try:
                import hailo  # noqa: F401  (TAPPAS python bindings for buffer metadata)

                self._hailo = hailo
            except ImportError as e:
                raise RuntimeError(
                    "the `hailo` python module (TAPPAS bindings) is needed for --save-json / callbacks "
                    "but is not importable. Run `hcpose doctor`."
                ) from e

        fd, self.pp_config = tempfile.mkstemp(prefix="hcpose_pp_", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(cfg.postprocess_json(layout.input_width, layout.input_height), f)

    # ------------------------------------------------------------------ pipeline
    def pipeline_string(self) -> str:
        sync = "true" if self.sync else "false"
        src = gst_source(self.source, self.width, self.height, self.fps, self.camera_raw)
        pre = (
            f"{src}{_q('pre_scale_q')} ! videoscale name=pre_scale n-threads=2 ! "
            f"{_q('pre_convert_q')} ! videoconvert name=pre_convert n-threads=3 qos=false ! "
            f"video/x-raw, format=RGB, pixel-aspect-ratio=1/1, width={self.width}, height={self.height} ! "
        )
        inner = (
            f"{_q('inf_scale_q')} ! videoscale name=inf_scale n-threads=2 qos=false ! "
            f"{_q('inf_convert_q')} ! video/x-raw, pixel-aspect-ratio=1/1 ! videoconvert name=inf_convert n-threads=2 ! "
            f"{_q('inf_net_q')} ! hailonet name=net hef-path=\"{self.cfg.hef}\" batch-size={self.batch_size} force-writable=true ! "
            # inf_filter_q is larger than everything the branch can hold in flight
            # (bypass_q=20): hailonet then never blocks on push, which is when it
            # discards in-flight frames at EOS (they came out with no detections).
            f"{_q('inf_filter_q', 30)} ! hailofilter name=pose so-path=\"{self.so_path}\" config-path=\"{self.pp_config}\" "
            f"function-name=filter_letterbox qos=false ! {_q('inf_out_q')} "
        )
        infer = (
            f"{_q('wrap_in_q')} ! hailocropper name=crop so-path=\"{self.cropper}\" function-name=create_crops "
            f"use-letterbox=true resize-method=inter-area internal-offset=true "
            f"hailoaggregator name=agg "
            f"crop. ! {_q('bypass_q', 20)} ! agg.sink_0 "
            f"crop. ! {inner} ! agg.sink_1 "
            f"agg. ! {_q('wrap_out_q')} ! "
        )
        post = ""
        if self.tracker:
            post += (
                # keep-tracked-frames=0: a track that misses a frame is moved to 'lost'
                # (no longer emitted) instead of being re-emitted with its stale box
                # for up to N frames, which created phantom/duplicate persons.
                # keep-lost-frames=15: lost tracks can still be re-matched for 0.5 s,
                # so ids survive short occlusions.
                "hailotracker name=tracker class-id=-1 kalman-dist-thr=0.8 iou-thr=0.9 init-iou-thr=0.7 "
                "keep-new-frames=2 keep-tracked-frames=0 keep-lost-frames=15 keep-past-metadata=false qos=false ! "
                f"{_q('tracker_q')} ! "
            )
        post += f"identity name=meta_tap ! {_q('meta_q')} ! "
        if self.overlay and (self.display or self.output):
            show_conf = "false" if self.display_mode != "none" else "true"
            # thicker lines/text so the focused/distracted labels are readable at 720p
            thick = "line-thickness=2 font-thickness=2 " if self.display_mode != "none" else ""
            post += f"hailooverlay name=overlay show-confidence={show_conf} {thick}qos=false ! {_q('overlay_q')} ! "
            if self.display_mode in ("class", "both") and self.has_textoverlay:
                pt = max(10, round(self.height / 52))  # ~14 pt at 720p, ~21 pt at 1080p
                post += (
                    'textoverlay name=class_text text="Engagement: starting" valignment=top halignment=left '
                    f'line-alignment=left wrap-mode=none xpad=12 ypad=8 font-desc="Sans Bold {pt}" '
                    'shaded-background=true ! '
                )

        sinks = []
        if self.display:
            sinks.append(f"videoconvert name=disp_convert n-threads=2 qos=false ! autovideosink name=display sync={sync}")
        if self.output:
            enc = pick_encoder(self.Gst)
            if not enc:
                raise RuntimeError("no H.264 encoder found; install gstreamer1.0-plugins-ugly (x264enc)")
            mux = "matroskamux" if self.output.lower().endswith(".mkv") else "mp4mux"
            out = os.path.abspath(self.output)
            sinks.append(
                f"videoconvert name=enc_convert n-threads=2 qos=false ! video/x-raw, format=I420 ! {enc} ! "
                f"h264parse ! {mux} ! filesink location=\"{out}\" sync={sync}"
            )
        if not sinks:
            sinks.append(f"fakesink name=sink sync={sync}")

        if len(sinks) == 1:
            tail = sinks[0]
        else:
            tail = "tee name=t " + " ".join(f"t. ! {_q(f'tee_q{i}', 5)} ! {s}" for i, s in enumerate(sinks))
        return pre + infer + post + tail

    # ------------------------------------------------------------------ callbacks
    def _on_buffer(self, pad, info):
        buf = info.get_buffer()
        if buf is None:
            return self.Gst.PadProbeReturn.OK
        now = time.monotonic()
        if self.t_start is None:
            self.t_start = now
        idx = self.frame_index
        self.frame_index += 1
        n = 0
        if self.callbacks:
            # media time (PTS) so durations like "distracted for 2.5 s" mean video seconds,
            # also when a file is processed faster than real time
            t = buf.pts / self.Gst.SECOND if buf.pts != self.Gst.CLOCK_TIME_NONE else now - self.t_start
            raw = self._pop_raw(buf.pts) if self.tracker else None
            result = self._extract(buf, idx, t, raw)
            n = len(result.persons)
            for cb in self.callbacks:
                try:
                    cb(result, None)
                except Exception as e:  # never kill the pipeline from user code
                    log.exception("frame callback failed: %s", e)
            if self.engagement and self.display_mode != "none" and self.overlay and (self.display or self.output):
                try:
                    self._annotate(buf, result)
                except Exception as e:
                    log.exception("engagement overlay failed: %s", e)
        if self.fps_counter:
            extra = f"persons: {n}" if self.callbacks else ""
            if self.engagement:
                extra = self.engagement.class_text()
            self.fps_counter.tick(extra)
        return self.Gst.PadProbeReturn.OK

    # colors come from hailooverlay's class-id palette: 1 green, 0 red, 18 white
    _STATE_CLASS = {"focused": 1, "distracted": 0, "unknown": 18}

    def _annotate(self, buf, result) -> None:
        """Re-label the detections hailooverlay is about to draw with the engagement state."""
        hailo = self._hailo
        if self.display_mode in ("class", "both") and self._text_overlay is not None:
            txt = self.engagement.class_text()
            if txt != self._text_last:
                self._text_overlay.set_property("text", txt)
                self._text_last = txt
        roi = hailo.get_roi_from_buffer(buf)
        W, H = self.width, self.height
        by_id = {p.track_id: p for p in result.persons if p.track_id is not None}
        for det in roi.get_objects_typed(hailo.HAILO_DETECTION):
            b = det.get_bbox()
            ids = det.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            person = by_id.get(int(ids[0].get_id())) if ids else None
            if person is None:
                box = [b.xmin() * W, b.ymin() * H, (b.xmin() + b.width()) * W, (b.ymin() + b.height()) * H]
                best = max(result.persons, key=lambda p: self._iou(p.box, box), default=None)
                if best is not None and self._iou(best.box, box) >= 0.5:
                    person = best
            eng = person.engagement if person is not None and person.engagement else None
            state = eng["state"] if eng else "unknown"
            if self.display_mode == "class":
                cls, label = 18, ""  # class-only mode: no per-student state on screen
            else:
                cls = self._STATE_CLASS[state]
                label = state if state != "unknown" else "?"
            new = hailo.HailoDetection(hailo.HailoBBox(b.xmin(), b.ymin(), b.width(), b.height()),
                                       cls, label, float(det.get_confidence()))
            for obj in det.get_objects():
                new.add_object(obj)
            roi.remove_object(det)
            roi.add_object(new)

    def _on_pre_tracker(self, pad, info):
        """Snapshot the decoder's detections before hailotracker replaces their boxes
        with Kalman-smoothed ones (which also moves the box-relative keypoints) and
        holds back unconfirmed new tracks."""
        buf = info.get_buffer()
        if buf is not None:
            self._raw_dets.append((buf.pts, self._read_persons(buf)))
        return self.Gst.PadProbeReturn.OK

    def _pop_raw(self, pts) -> Optional[List[Person]]:
        # Buffers reach meta_tap in the order they entered the tracker; match on
        # PTS when the source sets one, otherwise take the oldest snapshot.
        none = self.Gst.CLOCK_TIME_NONE
        while self._raw_dets:
            p, persons = self._raw_dets.popleft()
            if pts == none or p == none or p == pts:
                return persons
        return None

    @staticmethod
    def _iou(a, b) -> float:
        iw = min(a[2], b[2]) - max(a[0], b[0])
        ih = min(a[3], b[3]) - max(a[1], b[1])
        if iw <= 0 or ih <= 0:
            return 0.0
        inter = iw * ih
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)

    def _assign_ids(self, raw: List[Person], tracked: List[Person]) -> List[Person]:
        """Give each raw detection the id of the tracker output it overlaps most (greedy, IoU>=0.5).
        Detections the tracker has not confirmed yet keep track_id=None."""
        pairs = sorted(
            ((self._iou(r.box, t.box), i, j) for i, r in enumerate(raw) for j, t in enumerate(tracked)),
            reverse=True,
        )
        used_r, used_t = set(), set()
        for v, i, j in pairs:
            if v < 0.5:
                break
            if i in used_r or j in used_t:
                continue
            used_r.add(i)
            used_t.add(j)
            raw[i].track_id = tracked[j].track_id
        return raw

    def _extract(self, buf, idx: int, t: float, raw: Optional[List[Person]] = None) -> FrameResult:
        res = FrameResult(index=idx, timestamp=t, width=self.width, height=self.height)
        persons = self._read_persons(buf)
        res.persons = self._assign_ids(raw, persons) if raw is not None else persons
        return res

    def _read_persons(self, buf) -> List[Person]:
        import numpy as np

        hailo = self._hailo
        W, H = self.width, self.height
        persons: List[Person] = []
        roi = hailo.get_roi_from_buffer(buf)
        for det in roi.get_objects_typed(hailo.HAILO_DETECTION):
            b = det.get_bbox()
            x1, y1, bw, bh = b.xmin(), b.ymin(), b.width(), b.height()
            track_id = None
            ids = det.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if ids:
                track_id = int(ids[0].get_id())
            kpts = []
            lms = det.get_objects_typed(hailo.HAILO_LANDMARKS)
            if lms:
                for p in lms[0].get_points():
                    kpts.append(((p.x() * bw + x1) * W, (p.y() * bh + y1) * H, p.confidence()))
            persons.append(
                Person(
                    track_id=track_id,
                    score=float(det.get_confidence()),
                    class_id=int(det.get_class_id()),
                    label=det.get_label(),
                    box=[x1 * W, y1 * H, (x1 + bw) * W, (y1 + bh) * H],
                    keypoints=np.array(kpts, dtype=np.float32).reshape(-1, 3),
                )
            )
        return persons

    # ------------------------------------------------------------------ run loop
    def _on_message(self, bus, msg):
        Gst = self.Gst
        t = msg.type
        if t == Gst.MessageType.EOS:
            log.info("end of stream")
            self.loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            src = msg.src.get_name() if msg.src else "?"
            print(f"\n[hcpose] GStreamer error from {src}: {err.message}")
            if dbg:
                print(f"  debug: {dbg.splitlines()[0] if dbg else ''}")
            if "not-negotiated" in (dbg or "") + err.message and self.kind in ("usb", "rpi"):
                print(
                    f"  The camera could not deliver {self.width}x{self.height}@{self.fps}. Try --width/--height/--fps "
                    "values your camera supports (v4l2-ctl --list-formats-ext), or --camera-raw."
                )
            self.exit_code = 1
            self.loop.quit()
        elif t == Gst.MessageType.WARNING:
            w, _ = msg.parse_warning()
            log.warning("GStreamer warning: %s", w.message)
        return True

    def stop(self, *_):
        """Ctrl+C: send EOS so the recorder finalizes the MP4, then force-quit after a timeout."""
        if self._stopping:
            self.loop.quit()
            return False
        self._stopping = True
        print("\n[hcpose] stopping (finishing output)...")
        # Inject EOS right after the source: demuxers run filesrc in pull mode
        # and ignore an EOS sent to the pipeline, so files would play to the end.
        q = self.pipeline.get_by_name("pre_scale_q")
        if q is None or not q.get_static_pad("sink").send_event(self.Gst.Event.new_eos()):
            self.pipeline.send_event(self.Gst.Event.new_eos())
        self.GLib.timeout_add_seconds(4, lambda: (self.loop.quit(), False)[1])
        return True

    def run(self) -> int:
        Gst, GLib = self.Gst, self.GLib
        desc = self.pipeline_string()
        log.debug("pipeline: %s", desc)
        try:
            self.pipeline = Gst.parse_launch(desc)
        except GLib.Error as e:
            print(f"[hcpose] failed to build pipeline: {e.message}\n  pipeline: {desc}")
            return 1
        self._text_overlay = self.pipeline.get_by_name("class_text")
        tap = self.pipeline.get_by_name("meta_tap")
        tap.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, self._on_buffer)
        if self.tracker and self.callbacks:
            trk = self.pipeline.get_by_name("tracker")
            trk.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, self._on_pre_tracker)

        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_message)
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, self.stop)
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGTERM, self.stop)

        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            print("[hcpose] pipeline failed to start (run with -v for the pipeline string)")
            self.pipeline.set_state(Gst.State.NULL)
            return 1
        try:
            self.loop.run()
        finally:
            self.pipeline.set_state(Gst.State.NULL)
            try:
                os.unlink(self.pp_config)
            except OSError:
                pass
        if self.fps_counter and self.fps_counter.total:
            print(f"[hcpose] {self.fps_counter.summary()}")
        if self.output and self.exit_code == 0:
            print(f"[hcpose] saved {os.path.abspath(self.output)}")
        return self.exit_code
