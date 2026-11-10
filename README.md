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

Requirements: Raspberry Pi 5, Hailo AI HAT+ (Hailo-8 or 8L), Raspberry Pi OS 64-bit (tested on Debian 13 "trixie" with `hailo-all` 5.1.1, HailoRT 4.23, TAPPAS 5.1; Bookworm should also work) with the Hailo stack installed:

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

## Engagement analysis (focused / distracted)

Turns each student's skeleton into **focused**, **distracted** or **unknown**, and the class into a live estimate with a range ("43% distracted (20-65%) | 14 of ~30 students"). It uses keypoints only, so no images are stored.

```bash
# live, both labels and percentage on screen, camera at the front-left of the room
./hcpose.sh run models/upper_body_s.hef --input usb --engagement --camera-position left \
    --calibration ~/room_A.json

# teacher-facing screen without per-student labels
./hcpose.sh run models/upper_body_s.hef --input usb --engagement --display class

# all settings in a file
./hcpose.sh run models/upper_body_s.hef --input usb --engagement configs/engagement.yaml

# far rows too small to detect: also run the model on zoomed-in tiles of the top 60% of the frame
# (full frame + 3 tiles = 4 inferences per frame, in either engine)
./hcpose.sh run models/upper_body_m.hef --input usb --engagement --tiles auto --score-threshold 0.3

# choose the per-student classifier: trees (default), logistic, mlp, or none (rules)
./hcpose.sh run models/upper_body_m.hef --input usb --engagement --classifier mlp

# offline: replay a skeleton file recorded with --save-json (no Hailo needed)
./hcpose.sh analyze skeletons.jsonl -o labelled.jsonl --camera-position left --classifier logistic
```

**How it decides** (`hcpose/engagement/`):

1. **Seats instead of track ids.** Seated students barely move, so each detection is matched to a seat by position. This is robust to tracker id switches and to students disappearing behind others for a moment.
2. **Cues from the pose**, normalized by the student's own size:
   - head turned away from that seat's attentive direction
   - face hidden while the shoulders are visible (turned around)
   - head down
   - torso rotated
   - hand at the face

   A raised hand always counts as focused. These cues name the *reason* shown for a distracted student.
   **The per-frame decision** comes from a classifier trained on SCB classroom images ([`research/scb/`](research/scb/README.md)), switchable with `--classifier`:

   | Value | Model |
   |---|---|
   | `trees` / `auto` (default) | gradient-boosted trees |
   | `logistic` | logistic regression |
   | `mlp` | small neural net |
   | `none` | the hand-written cue weights |

   All run in NumPy on the Pi. They are trained on the m HEF's poses of SCB students, with and without tiles. On SCB front-view students they reach a within-image AUC of about 0.74–0.84, against 0.54 for the rules. They depend on good poses: on a real exam-hall clip, trees gave about 9–13% distracted with the m model, but 36% with n without tiles (16% with tiles), while the rules stayed at 5–11% with either model. With the n model, prefer `--tiles auto` or `--classifier none`.
3. **Attentive baseline per seat.** Each seat learns its own "looking at the board" pose from its first `calibration_seconds` of observation (median, so glances don't matter). Until then the seat's running history and the room's typical direction are used. That works with a camera at the front center, front-left or front-right, without telling it where. `--camera-position` adds a check against a class that is mostly distracted from the very start. `--calibration room.json` saves what was learned for that classroom and loads it next time; use `--recalibrate` after moving the camera.
4. **Decisions over time.** The per-frame score is smoothed and switches state with hysteresis: distracted after about 2.5 s, focused again after about 1.5 s. A glance doesn't flip a student. Students without usable keypoints for 2 s are **unknown** and left out of the percentage.
5. **Class estimate.** The camera misses students, mostly in far rows, and the misses aren't random, so a plain percentage of *detected* students is biased. The estimate counts every recently seen seat with its last state (seat memory, `presence_seconds`). It weights near, middle and far rows by their number of seats, not by how many students happen to be detected there. It shows a 95% range around the number on screen. The range covers students that weren't counted, not classifier mistakes. `--class-size N` gives it the real head count; only then does counting most of the class narrow the range, because a size inferred from the seats seen so far may be too small.

Settings, with explanations: [`configs/engagement.yaml`](configs/engagement.yaml). The `--display` switch is `students | class | both | none`; `class` deliberately shows no per-student state.

With `--save-json`, each person gets `"engagement": {"seat", "state", "score", "reason", "calibrated"}` and each frame gets a `"class"` block. In the gst engine the on-screen box color comes from `hailooverlay`'s palette: green is focused, red is distracted, white is unknown.

**Assumptions.** These are worth stating in the thesis:
- During the first minute or two, most students mostly look toward the board, unless `--camera-position` is set.
- "Head down" can't separate writing from a phone; that's why it has a lower weight.
- The classifiers learn SCB's *action* labels (reading, turning, discussing …) as a proxy for attention. SCB is mostly Chinese K-12 classrooms.

## Building on it

```python
from hcpose.api import run
from hcpose.engagement import EngagementAnalyzer, load_settings
from hcpose.presets import UPPER_BODY13_NAMES

def on_frame(result, frame):          # frame is None with the gst engine
    print(result.extra.get("class"))  # class stats when engagement is enabled
    for p in result.persons:
        print(p.engagement)           # seat, state, score, reason

an = EngagementAnalyzer(load_settings("configs/engagement.yaml"), UPPER_BODY13_NAMES)
run("models/upper_body_n.hef", source="/dev/video0", on_frame=on_frame, engagement=an, display=False)
```

## What changed vs. the old `custom_pose_estimation` patch in the hailo-apps fork

- **Double sigmoid fixed.** The `.alls` applies sigmoid to the class-score outputs inside the HEF, and the old C++ post-process applied it again. That compresses confidences into 0.5–0.73, so a 0.6 threshold behaved like ~0.4 and displayed scores were wrong. The Python standalone path didn't do this, so the two paths disagreed. Now detected per HEF (`hcpose inspect` shows which case your HEF is).
- **NMS now sorts by score first.** The inherited Hailo code assumed candidates were sorted but never sorted them, so overlapping boxes were resolved by anchor position instead of confidence.
- **Same-skeleton duplicates removed.** YOLO-pose often outputs a tight box and a loose box for one student with the same skeleton. Their box IoU is low, so NMS keeps both, and the engagement analyzer then counts one student twice. After NMS, a box lying inside a stronger one with a matching skeleton is dropped (`pose_dedup`, default 0.1; 0 turns it off). On 1,500 SCB images this removed 6.5% of detections, about one per image.
- No hard-coded keypoint count (it was 13 in four places), class count, strides, tensor order or output dtype.
- Box clamping clamps corners rather than width/height independently (which could shift boxes at the frame edge).
- Recording finalizes the MP4 on end-of-file and on Ctrl+C.
- No longer depends on a specific hailo-apps release or its resource downloader (the old config registered the custom models as Model Zoo downloads, which can't exist). Uses only the apt-installed HailoRT/TAPPAS.

## Tests

```bash
.venv/bin/pip install pytest
# C++ test harnesses (need the TAPPAS/HailoRT headers, present after hailo-all):
mkdir -p build
G="-std=c++17 -O2 -Wno-psabi $(pkg-config --cflags hailo-tappas-core) -I$(pkg-config --variable=rapidjson_includedir hailo-tappas-core)"
g++ $G tests/cpp_harness.cpp cpp/yolov8_pose_postprocess.cpp cpp/tiling.cpp -o build/cpp_harness
g++ $G tests/tile_harness.cpp cpp/yolov8_pose_postprocess.cpp cpp/tiling.cpp -o build/tile_harness
.venv/bin/python -m pytest -q tests
```

The fixture is the raw head output of the official `yolov8n-pose` on Ultralytics' `bus.jpg`. The NumPy decoder matches Ultralytics' own decode+NMS exactly. The C++ post-process matches the NumPy decoder for float32, uint8/uint16-quantized, sigmoid-in-HEF, shuffled-order and 13-keypoint variants, and for the letterbox mapping, including on a tile. `tests/test_tiling.py` checks the tile geometry and the merge, and that the C++ merge in the GStreamer pipeline gives the same result as the Python one on random scenes. `tests/check_gst_pipeline.py` runs the generated GStreamer pipeline end-to-end on a machine without Hailo using stand-in elements. `tests/check_gst_engagement.py` does the same with a fake `hailo` bindings module that feeds a synthetic class through the engagement path: re-labelling for the overlay, the class text and the JSONL. `tests/test_engagement.py` covers the rules on synthetic classrooms:
- turning away, glances, each cue
- corner cameras
- most of the class distracted
- calibration save/load and recovering from a bad calibration
- occlusion
- the `analyze` command

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
