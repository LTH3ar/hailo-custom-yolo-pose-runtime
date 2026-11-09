# hailo-custom-yolo-pose-runtime (`hcpose`)

Run **any** custom YOLOv8-pose model compiled to HEF on a Raspberry Pi 5 with a Hailo-8 / Hailo-8L AI HAT+. Clone, install, point it at a `.hef`, run.

Part of the *Privacy-First Classroom Engagement Analysis System via Edge Computing and Lightweight Deep Learning Models* project:

| Repo | Role |
|---|---|
| [hailo-custom-yolo-pose](https://github.com/LTH3ar/hailo-custom-yolo-pose) | train (Ultralytics) and compile (Hailo DFC) the model |
| **hailo-custom-yolo-pose-runtime** (this) | run the HEF on the Pi: camera/video → skeletons |
| [hailo-custom-yolo-pose-report](https://github.com/LTH3ar/hailo-custom-yolo-pose-report) | thesis report |

Nothing about the model is hard-coded. The keypoint count, class count, strides, tensor order, uint8/uint16 outputs, and whether the HEF already applies sigmoid to the scores are all read from the HEF at startup. A 13-keypoint upper-body model, the stock 17-keypoint model, or a model with other keypoints all run with the same command.

## Install

Requirements: Raspberry Pi 5, Hailo AI HAT+ (Hailo-8 or 8L), Raspberry Pi OS 64-bit (Bookworm or newer) with the Hailo stack installed:

```bash
sudo apt update && sudo apt install hailo-all && sudo reboot
```

Then:

```bash
git clone https://github.com/LTH3ar/hailo-custom-yolo-pose-runtime.git
cd hailo-custom-yolo-pose-runtime
./install.sh
```

`install.sh` checks the Hailo stack, installs the few build packages it needs (meson, ninja, rapidjson…), builds the C++ post-process into `hcpose/lib/`, creates `.venv` (with system site packages, so it sees the apt-installed HailoRT/GStreamer bindings), and runs `hcpose doctor`. Nothing is written outside the repo except apt packages.

## Run

```bash
# what does the runtime see in this HEF?
./hcpose.sh inspect models/upper_body_n.hef

# USB webcam, live window
./hcpose.sh run models/upper_body_n.hef --input usb

# Pi camera module
./hcpose.sh run models/upper_body_n.hef --input rpi

# classroom recording -> annotated video + skeletons, no window
./hcpose.sh run models/upper_body_n.hef -i classroom.mp4 -o annotated.mp4 --save-json skeletons.jsonl --headless

# single image (uses the python engine automatically)
./hcpose.sh run models/upper_body_n.hef -i photo.jpg -o photo_pose.jpg

# throughput benchmark: no display, no sync
./hcpose.sh run models/upper_body_n.hef -i classroom.mp4 --headless --no-sync
```

`--input` accepts `usb`, `/dev/videoN`, `rpi`, a video file, an image, `rtsp://…`, or `http(s)://…`. Ctrl+C stops cleanly and still finalizes the output MP4.

Useful flags: `--score-threshold`, `--keypoint-threshold`, `--iou-threshold`, `--width/--height/--fps`, `--no-tracker`, `--no-overlay`, `--print`, `-v` (prints the GStreamer pipeline). See `./hcpose.sh run -h`.

### Engines

| `--engine` | How | When |
|---|---|---|
| `gst` (default) | `hailonet` → generic C++ `hailofilter` → `hailotracker` → `hailooverlay`, all native | live use, benchmarks |
| `python` | OpenCV + HailoRT `InferVStreams` + NumPy decoder | images, debugging, evaluation, or if the C++ build is unavailable |

Both decoders produce identical detections (see Tests).

## Your models

Any HEF exported the way `hailo-custom-yolo-pose/scripts/04-pytorch-to-hailo.py` does it works: the **9 raw head convs** as end nodes (box DFL, class score, keypoints at strides 8/16/32), no on-chip NMS.

Put the HEF anywhere (`models/` is git-ignored for `.hef`). Optionally add a **sidecar** with the same name (`models/upper_body_n.yaml`) to set thresholds, keypoint names, skeleton, or labels; see [`models/example.yaml`](models/example.yaml). Without one, 17 keypoints get the COCO skeleton, 13 get the upper-body skeleton (COCO 0–12), anything else gets generic names and no skeleton lines.

## Skeleton output (privacy-first)

`--save-json out.jsonl` writes one JSON object per frame with only coordinates and no pixels:

```json
{"type": "header", "hef": "...", "keypoint_names": ["nose", ...], "skeleton": [[0,1], ...], "labels": ["person"]}
{"type": "frame", "i": 0, "t": 0.033, "w": 1280, "h": 720,
 "persons": [{"track_id": 3, "score": 0.87, "class_id": 0, "label": "person",
              "box": [x1, y1, x2, y2], "keypoints": [[x, y, visibility], ...]}]}
```

Coordinates are pixels in the processed frame (`w`×`h`). Boxes and keypoints are always the decoder's own output; the tracker only supplies `track_id` (`hailotracker` in gst, a simple IoU tracker in python). In the gst engine a person seen for the first time has `track_id: null` until `hailotracker` confirms the track (about 1–2 frames). `hailotracker` replaces boxes with Kalman-smoothed ones, which also moves the box-relative keypoints, so hcpose reads the detections before the tracker and only takes the ids from after it.

## Building on it (part 2: engagement analysis)

```python
from hcpose.api import run

def on_frame(result, frame):          # frame is None with the gst engine
    for p in result.persons:
        nose, l_sh, r_sh = p.keypoints[0], p.keypoints[5], p.keypoints[6]
        ...

run("models/upper_body_n.hef", source="/dev/video0", on_frame=on_frame, display=False)
```

## What changed vs. the old `custom_pose_estimation` patch in the hailo-apps fork

- **Double sigmoid fixed.** The `.alls` applies sigmoid to the class-score outputs inside the HEF, and the old C++ post-process applied it again. That compresses confidences into 0.5–0.73, so a 0.6 threshold behaved like ~0.4 and displayed scores were wrong. The Python standalone path didn't do this, so the two paths disagreed. Now detected per HEF (`hcpose inspect` shows which case your HEF is).
- **NMS now sorts by score first.** The inherited Hailo code assumed candidates were sorted but never sorted them, so overlapping boxes were resolved by anchor position instead of confidence.
- No hard-coded keypoint count (it was 13 in four places), class count, strides, tensor order or output dtype.
- Box clamping clamps corners rather than width/height independently (which could shift boxes at the frame edge).
- Recording finalizes the MP4 on end-of-file and on Ctrl+C.
- No longer depends on a specific hailo-apps release or its resource downloader (the old config registered the custom models as Model Zoo downloads, which can't exist). Uses only the apt-installed HailoRT/TAPPAS.

## Tests

```bash
.venv/bin/pip install pytest
# C++ test harness (needs the TAPPAS/HailoRT headers, present after hailo-all):
g++ -std=c++17 -O2 $(pkg-config --cflags hailo-tappas-core) \
    -I"$(pkg-config --variable=rapidjson_includedir hailo-tappas-core)" tests/cpp_harness.cpp cpp/yolov8_pose_postprocess.cpp -o build/cpp_harness
.venv/bin/python -m pytest -q tests
```

The fixture is the raw head output of the official `yolov8n-pose` on Ultralytics' `bus.jpg`. The NumPy decoder matches Ultralytics' own decode+NMS exactly. The C++ post-process matches the NumPy decoder for float32, uint8/uint16-quantized, sigmoid-in-HEF, shuffled-order and 13-keypoint variants, and for the letterbox mapping. `tests/check_gst_pipeline.py` runs the generated GStreamer pipeline end-to-end on a machine without Hailo using stand-in elements.

## Troubleshooting

Run `./hcpose.sh doctor` first; it checks every dependency and prints the fix.

- **`expected 3 outputs per scale`**: the HEF was compiled with NMS or different end nodes. Recompile with the 9 head convs as end nodes.
- **Camera `not-negotiated`**: the camera can't do the requested mode. Try `--width 640 --height 480`, `--fps 15`, or `--camera-raw`; list modes with `v4l2-ctl -d /dev/video0 --list-formats-ext`.
- **`--input rpi` says `libcamerasrc` is missing**: `sudo apt install gstreamer1.0-libcamera`, or use `--engine python` (picamera2).
- **`--input usb` says no USB camera found**: on a Pi 5 the `/dev/video*` nodes are ISP/codec nodes; plug in a camera or pass `--input /dev/videoN`.
- **`--save-json` with the gst engine says the `hailo` module is missing**: the TAPPAS Python bindings aren't installed. Use `--engine python` meanwhile.
- **Different results between engines**: they letterbox slightly differently (`hailocropper` vs OpenCV resize), which can move borderline detections; the decoders themselves are identical.

## License

AGPL-3.0, same as the training repo (Ultralytics models are AGPL-3.0).
