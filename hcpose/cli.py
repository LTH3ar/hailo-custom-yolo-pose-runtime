"""Command line: hcpose run | inspect | doctor"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from . import __version__, env
from .config import load_config
from .hef import read_hef
from .layout import LayoutError
from .presets import preset_for
from .results import FrameResult, JsonlWriter


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("hef", help="path to the compiled .hef")
    p.add_argument("--config", "-c", help="model sidecar YAML/JSON (default: <hef stem>.yaml next to the HEF)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hcpose", description="Run custom YOLOv8-pose HEFs on Hailo.")
    ap.add_argument("--version", action="version", version=f"hcpose {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a model on a camera, video file, image or stream")
    _add_model_args(r)
    r.add_argument("--input", "-i", default="usb",
                   help="usb | /dev/videoN | rpi | video file | image file | rtsp://... (default: usb)")
    r.add_argument("--engine", choices=["auto", "gst", "python"], default="auto",
                   help="gst = native GStreamer pipeline (fast); python = HailoRT + NumPy (no build needed)")
    r.add_argument("--output", "-o", help="save annotated video (.mp4/.mkv) or image")
    r.add_argument("--save-json", help="save per-frame skeletons as JSON lines (privacy-first output, no pixels)")
    r.add_argument("--print", dest="print_dets", action="store_true", help="print detections every frame")
    r.add_argument("--headless", action="store_true", help="no display window")
    r.add_argument("--width", type=int, default=1280, help="processing/output frame width (default 1280)")
    r.add_argument("--height", type=int, default=720, help="processing/output frame height (default 720)")
    r.add_argument("--fps", type=int, default=30, help="camera frame rate (default 30)")
    sync = r.add_mutually_exclusive_group()
    sync.add_argument("--sync", dest="sync", action="store_true", default=None,
                      help="play files in real time (default when displaying a file)")
    sync.add_argument("--no-sync", dest="sync", action="store_false", help="run as fast as possible (benchmarking)")
    r.add_argument("--batch-size", type=int, default=1)
    r.add_argument("--no-tracker", action="store_true")
    r.add_argument("--no-overlay", action="store_true")
    r.add_argument("--camera-raw", action="store_true", help="USB camera: request raw frames instead of MJPEG")
    r.add_argument("--quiet-fps", action="store_true", help="don't print FPS")
    th = r.add_argument_group("detection settings (override the sidecar)")
    th.add_argument("--score-threshold", type=float)
    th.add_argument("--iou-threshold", type=float)
    th.add_argument("--keypoint-threshold", type=float)
    th.add_argument("--max-detections", type=int)
    th.add_argument("--score-activation", choices=["auto", "sigmoid", "none"],
                    help="force whether class scores need a sigmoid (default: auto-detect)")
    eg = r.add_argument_group("engagement analysis (focused / distracted)")
    eg.add_argument("--engagement", nargs="?", const="", metavar="SETTINGS_YAML",
                    help="enable engagement analysis, optionally with a settings file (see configs/engagement.yaml)")
    eg.add_argument("--display", choices=["students", "class", "both", "none"],
                    help="what to show on screen: per-student labels, the class percentage, both, or none")
    eg.add_argument("--camera-position", choices=["auto", "center", "left", "right"],
                    help="where the camera is mounted at the front (left/right as seen from the students)")
    eg.add_argument("--calibration", metavar="JSON",
                    help="per-classroom calibration file: loaded if it exists, saved when the run ends")
    eg.add_argument("--recalibrate", action="store_true", default=None,
                    help="ignore the existing calibration file and learn it again (camera moved)")
    eg.add_argument("--classifier", metavar="auto|none|JSON",
                    help="per-frame decision: bundled learned model (auto, default), rules (none), or a model file")
    r.add_argument("-v", "--verbose", action="store_true")

    i = sub.add_parser("inspect", help="show what the runtime detects in a HEF")
    _add_model_args(i)
    i.add_argument("--no-device", action="store_true", help="don't run a test inference")

    a = sub.add_parser("analyze", help="run engagement analysis offline on a --save-json skeleton file")
    a.add_argument("jsonl", help="skeletons saved with `hcpose run --save-json`")
    a.add_argument("--engagement", metavar="SETTINGS_YAML", help="engagement settings file")
    a.add_argument("--camera-position", choices=["auto", "center", "left", "right"])
    a.add_argument("--calibration", metavar="JSON", help="calibration file to load (if it exists) and save")
    a.add_argument("--recalibrate", action="store_true", default=None)
    a.add_argument("--classifier", metavar="auto|none|JSON")
    a.add_argument("--output", "-o", help="write the skeletons with engagement labels to this JSONL")
    a.add_argument("--every", type=float, default=10.0, help="timeline step in seconds (default 10)")
    a.add_argument("--fps", type=float, help="use frame index / FPS as time (for files saved before timestamps "
                   "were video time, e.g. runs with --no-sync)")

    sub.add_parser("doctor", help="check the installation")
    return ap


def _model(args, overrides=None):
    cfg = load_config(args.hef, args.config, overrides)
    info = read_hef(cfg.hef)
    layout = info.layout(cfg.num_keypoints, cfg.num_classes, cfg.reg_max)
    cfg.fill_from_layout(layout.num_keypoints, layout.num_classes)
    return cfg, info, layout


def cmd_inspect(args) -> int:
    cfg = load_config(args.hef, args.config)
    info = read_hef(cfg.hef)
    print(f"HEF:     {cfg.hef}")
    print(f"sidecar: {cfg.sidecar or '(none, using auto-detection + presets)'}")
    h, w, c = info.input_shape
    print(f"input:   {info.input_name}  {h}x{w}x{c}")
    print("outputs:")
    for o in info.outputs:
        print(f"  {o.name:<40} {o.shape[0]:>3}x{o.shape[1]:<3}x{o.shape[2]:<4} {o.dtype:<8} scale={o.qp_scale:.6g} zp={o.qp_zp:g}")
    try:
        layout = info.layout(cfg.num_keypoints, cfg.num_classes, cfg.reg_max)
    except LayoutError as e:
        print(f"\nlayout:  ERROR {e}")
        return 1
    cfg.fill_from_layout(layout.num_keypoints, layout.num_classes)
    print(f"\nlayout:  {layout.num_keypoints} keypoints, {layout.num_classes} class(es), reg_max {layout.reg_max}")
    for s in layout.scales:
        print(f"  stride {s.stride_x:g}: boxes={s.box}  scores={s.score}  kpts={s.kpt}")
    print(f"preset:  {preset_for(layout.num_keypoints)['name']}  keypoints={cfg.keypoint_names}")
    print(f"labels:  {cfg.labels}")
    if args.no_device:
        return 0
    try:
        import numpy as np

        from .decode import needs_sigmoid
        from .hef import import_hailo_platform

        hp = import_hailo_platform()
        hef = hp.HEF(cfg.hef)
        with hp.VDevice() as vdev:
            ng = vdev.configure(hef, hp.ConfigureParams.create_from_hef(hef, interface=hp.HailoStreamInterface.PCIe))[0]
            ivp = hp.InputVStreamParams.make(ng, format_type=hp.FormatType.UINT8)
            ovp = hp.OutputVStreamParams.make(ng, format_type=hp.FormatType.FLOAT32)
            with ng.activate(ng.create_params()), hp.InferVStreams(ng, ivp, ovp) as pipe:
                out = pipe.infer({info.input_name: np.full((1, h, w, c), 114, np.uint8)})
        scores = [out[s.score] for s in layout.scales]
        vis = [out[s.kpt][..., 2::3] for s in layout.scales]
        smin = min(float(a.min()) for a in scores)
        smax = max(float(a.max()) for a in scores)
        print(f"\ntest inference (gray frame): class score range [{smin:.3f}, {smax:.3f}] -> "
              + ("logits, sigmoid applied in post-process" if needs_sigmoid(scores) else "already sigmoid inside the HEF"))
        print("  keypoint visibility: " + ("logits, sigmoid applied in post-process" if needs_sigmoid(vis) else "already sigmoid inside the HEF"))
    except Exception as e:
        print(f"\ntest inference skipped: {e}")
    return 0


def _printer(result: FrameResult, frame=None) -> None:
    print(f"frame {result.index}: {len(result.persons)} person(s)")
    for p in result.persons:
        tid = f"#{p.track_id}" if p.track_id is not None else ""
        print(f"  {p.label}{tid} {p.score:.2f} box={[round(v) for v in p.box]}")


def _engagement(args, keypoint_names):
    wanted = any(getattr(args, k, None) is not None
                 for k in ("engagement", "display", "camera_position", "calibration", "recalibrate", "classifier"))
    if not wanted:
        return None
    from .engagement import EngagementAnalyzer, load_settings

    settings = load_settings(args.engagement or None, {
        "display": getattr(args, "display", None),
        "camera_position": args.camera_position,
        "calibration": args.calibration,
        "recalibrate": args.recalibrate,
        "classifier": args.classifier,
    })
    an = EngagementAnalyzer(settings, keypoint_names)
    loaded = sum(1 for s in an.seats if s.calibrated)
    decision = f"learned classifier ({an.classifier.meta.get('name', 'custom')})" if an.classifier else "rules"
    print(f"[hcpose] engagement: {decision}, display={settings.display} camera={settings.camera_position} "
          + (f"calibration {settings.calibration} ({loaded} seats loaded)" if settings.calibration else
             "no calibration file (pass --calibration room.json to keep what it learns)"))
    return an


def _finish_engagement(an) -> None:
    if an is None:
        return
    s = an.summary()
    if s["distracted_pct_overall"] is not None:
        print(f"[hcpose] engagement: {s['distracted_pct_overall']}% of counted student-time distracted "
              f"({s['student_seconds_counted']} student-seconds, {len(s['seats'])} seats)")
    if an.s.calibration:
        n = an.save_calibration(an.s.calibration)
        print(f"[hcpose] calibration saved: {n} seats -> {an.s.calibration}")


def cmd_analyze(args) -> int:
    import json

    from .engagement import EngagementAnalyzer, load_settings
    from .results import FrameResult, Person
    import numpy as np

    settings = load_settings(args.engagement, {
        "camera_position": args.camera_position, "calibration": args.calibration, "recalibrate": args.recalibrate,
        "classifier": args.classifier,
    })
    an = None
    out = open(args.output, "w") if args.output else None
    timeline, next_t, bucket = [], None, []
    with open(args.jsonl) as f:
        for line in f:
            d = json.loads(line)
            if d.get("type") == "header":
                an = EngagementAnalyzer(settings, d["keypoint_names"])
                if out:
                    out.write(json.dumps({**d, "engagement_settings": settings.to_dict()}) + "\n")
                continue
            if an is None:
                raise ValueError(f"{args.jsonl}: no header line (was it written by hcpose run --save-json?)")
            persons = [Person(p.get("track_id"), p["score"], p["class_id"], p["label"], p["box"],
                              np.asarray(p["keypoints"], dtype=np.float32).reshape(-1, 3)) for p in d["persons"]]
            t = d["i"] / args.fps if args.fps else d["t"]
            if args.fps:
                d["t"] = round(t, 4)
            res = FrameResult(d["i"], t, d["w"], d["h"], persons)
            c = an.update(res)
            if out:
                out.write(json.dumps({**d, "persons": [p.to_dict() for p in persons], **res.extra}) + "\n")
            next_t = res.timestamp if next_t is None else next_t
            if c["pct_distracted"] is not None:
                bucket.append((c["pct_distracted"], c["counted"]))
            if res.timestamp >= next_t + args.every:
                timeline.append((next_t, bucket))
                next_t, bucket = res.timestamp, []
    if bucket:
        timeline.append((next_t, bucket))
    if out:
        out.close()
    print(f"{'time':>8}  {'distracted':>10}  {'students':>8}")
    for t0, b in timeline:
        if b:
            print(f"{t0:8.1f}  {np.mean([x for x, _ in b]):9.1f}%  {np.mean([n for _, n in b]):8.1f}")
        else:
            print(f"{t0:8.1f}  {'-':>10}  {'0':>8}")
    _finish_engagement(an)
    if args.output:
        print(f"[hcpose] labelled skeletons written to {args.output}")
    return 0


def cmd_run(args) -> int:
    overrides = {
        "score_threshold": args.score_threshold,
        "iou_threshold": args.iou_threshold,
        "keypoint_threshold": args.keypoint_threshold,
        "max_detections": args.max_detections,
        "score_activation": args.score_activation,
    }
    cfg, info, layout = _model(args, overrides)
    print(f"[hcpose] {cfg.hef}: {layout.num_keypoints} keypoints, {layout.num_classes} class(es), "
          f"input {layout.input_width}x{layout.input_height}" + (f", sidecar {cfg.sidecar}" if cfg.sidecar else ""))

    engagement = _engagement(args, cfg.keypoint_names)
    callbacks = []
    writer = None
    if args.save_json:
        writer = JsonlWriter(args.save_json, {
            "hef": cfg.hef,
            "source": args.input,
            "keypoint_names": cfg.keypoint_names,
            "skeleton": cfg.skeleton,
            "labels": cfg.labels,
            "score_threshold": cfg.score_threshold,
            **({"engagement_settings": engagement.s.to_dict()} if engagement else {}),
        })
        callbacks.append(writer)
    if args.print_dets:
        callbacks.append(_printer)

    engine = args.engine
    from .sources import classify

    if engine == "auto":
        if classify(args.input) == "image":
            engine = "python"
        elif env.find_postprocess_so() is None:
            print("[hcpose] C++ post-process not built; using the python engine (run ./install.sh for the faster one)")
            engine = "python"
        else:
            engine = "gst"

    common = dict(
        source=args.input, width=args.width, height=args.height, fps=args.fps,
        display=not args.headless, output=args.output, tracker=not args.no_tracker,
        overlay=not args.no_overlay, callbacks=callbacks, show_fps=not args.quiet_fps,
        engagement=engagement,
    )
    try:
        if engine == "gst":
            from .gst_engine import GstRunner

            runner = GstRunner(cfg, layout, sync=args.sync, batch_size=args.batch_size, camera_raw=args.camera_raw, **common)
            if args.verbose:
                print("[hcpose] pipeline:\n  " + runner.pipeline_string().replace(" ! ", " !\n    "))
        else:
            from .py_engine import PyRunner

            runner = PyRunner(cfg, info, layout, **common)
        return runner.run()
    finally:
        if writer:
            writer.close()
            print(f"[hcpose] skeletons saved to {args.save_json}")
        _finish_engagement(engagement)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
                        format="[hcpose] %(message)s")
    try:
        if args.cmd == "doctor":
            from .doctor import doctor

            return doctor()
        if args.cmd == "inspect":
            return cmd_inspect(args)
        if args.cmd == "analyze":
            return cmd_analyze(args)
        return cmd_run(args)
    except (FileNotFoundError, ValueError, RuntimeError, LayoutError) as e:
        print(f"[hcpose] error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
