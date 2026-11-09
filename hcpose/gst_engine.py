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
        self.callbacks = callbacks or []
        self.fps_counter = FpsCounter() if show_fps else None
        self.frame_index = 0
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
        if missing:
            raise RuntimeError(f"missing GStreamer elements: {missing}. Is hailo-all installed? Run `hcpose doctor`.")

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
            f"{_q('inf_filter_q')} ! hailofilter name=pose so-path=\"{self.so_path}\" config-path=\"{self.pp_config}\" "
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
                "hailotracker name=tracker class-id=-1 kalman-dist-thr=0.8 iou-thr=0.9 init-iou-thr=0.7 "
                "keep-new-frames=2 keep-tracked-frames=15 keep-lost-frames=2 keep-past-metadata=false qos=false ! "
                f"{_q('tracker_q')} ! "
            )
        post += f"identity name=meta_tap ! {_q('meta_q')} ! "
        if self.overlay and (self.display or self.output):
            post += f"hailooverlay name=overlay qos=false ! {_q('overlay_q')} ! "

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
                f"h264parse ! {mux} ! filesink location=\"{out}\" sync=false"
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
            result = self._extract(buf, idx, now - self.t_start)
            n = len(result.persons)
            for cb in self.callbacks:
                try:
                    cb(result, None)
                except Exception as e:  # never kill the pipeline from user code
                    log.exception("frame callback failed: %s", e)
        if self.fps_counter:
            self.fps_counter.tick(f"persons: {n}" if self.callbacks else "")
        return self.Gst.PadProbeReturn.OK

    def _extract(self, buf, idx: int, t: float) -> FrameResult:
        hailo = self._hailo
        W, H = self.width, self.height
        res = FrameResult(index=idx, timestamp=t, width=W, height=H)
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
            import numpy as np

            res.persons.append(
                Person(
                    track_id=track_id,
                    score=float(det.get_confidence()),
                    class_id=int(det.get_class_id()),
                    label=det.get_label(),
                    box=[x1 * W, y1 * H, (x1 + bw) * W, (y1 + bh) * H],
                    keypoints=np.array(kpts, dtype=np.float32).reshape(-1, 3),
                )
            )
        return res

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
        tap = self.pipeline.get_by_name("meta_tap")
        tap.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, self._on_buffer)

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
