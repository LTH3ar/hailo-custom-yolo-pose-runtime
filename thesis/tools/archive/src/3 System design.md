# 3 System design and implementation

## 3.1 Architecture

The system has two stages on one device. The detection runtime turns camera frames into skeletons on the Hailo-8, and the engagement analysis turns skeletons into a class-level share. Frames exist only in memory, and per-student states are never output by default.

&#91;embedded content: hcpose architecture · default settings\]

The skeleton file (`--save-json`) and an annotated video (`--output`) can be written at the hand-off between the two stages for development and evaluation. Both are off by default. The data-flow view used for the privacy analysis is in chapter 4.

**\[Figure to add, manual\] The device.** Photograph the Pi 5 with the AI HAT+, the active cooler and the camera, ideally mounted high at the front of a room. Suggested caption: *The device: Raspberry Pi 5 with the Hailo-8 AI HAT+ (26 TOPS), active cooler and camera.*

## 3.2 Pose models and compilation

The pose models are YOLOv8-pose networks in three sizes, n, s and m, that predict one class (person) and 13 upper-body keypoints: COCO keypoints 0-12 \[13\], from the nose to the hips. Classroom desks hide the legs, so the lower-body points carry no information. The models were trained on COCO with Ultralytics \[14\], in the companion repository `hailo-custom-yolo-pose`, and kept fixed for this thesis.

**Training.** All three sizes start from Ultralytics' COCO-pretrained YOLOv8-pose weights. They were fine-tuned on COCO 2017 person keypoints: train2017 for training, val2017 for validation. The labels were converted with Ultralytics' `convert_coco` and cut from 17 to 13 keypoints. The settings were the same for every size except the batch:

- 100 epochs at 640×640; batch 16 for n and m, 32 for s; seed 0, deterministic mode, mixed precision.
- `optimizer=auto`: Ultralytics chose the optimizer and learning rate itself, so the script's `lr0=0.01` and `momentum=0.937` were not used. The logged rates show a 3-epoch warm-up, then a linear fall to 1% of the peak by epoch 100.
- Loss weights and augmentation at Ultralytics defaults: HSV jitter, scale 0.5, translate 0.1, and horizontal flip at 0.5 with left and right keypoints swapped. Mosaic was on except in the last 10 epochs.

Training ran on the project's workstation (one NVIDIA RTX 3060) with Ultralytics' June 2026 release.

| Model | Best epoch | Training time (h) | Box mAP50-95 (training log) |
| --- | --- | --- | --- |
| n | 100 | 43.1 | 0.527 |
| s | 99 | 42.4 | 0.584 |
| m | 96 | 89.9 | 0.619 |

**Keypoint accuracy.** On COCO val2017 the trained models reach 0.470, 0.568 and 0.625 keypoint AP on their 13 points, 4-5 points below the stock 17-keypoint models on the same points. On the Hailo-8, INT8 costs n 5.3 points and m 6.8. The s HEF loses 30.8 points, so it is left out of every accuracy result.

| Model | Stock 17-point model, on the 13 points | Trained 13-point model (.pt) | INT8 HEF on the Hailo-8 | INT8 cost (AP points) |
| --- | --- | --- | --- | --- |
| n | 0.523 | 0.470 | 0.417 | −5.3 |
| s | 0.613 | 0.568 | 0.260 | −30.8 |
| m | 0.665 | 0.625 | 0.557 | −6.8 |

Keypoint AP (OKS 0.50:0.95) on the 5,000 val2017 images, scored with pycocotools \[15\] and COCO's per-keypoint tolerances. Every model gets the same 640×640 letterbox. As a check on the scoring, the stock models score 0.499, 0.594 and 0.645 on all 17 points, within 0.6 points of Ultralytics' published 50.4, 60.0 and 65.0.

- **Size.** m gains 0.14 AP over n on the Hailo-8 (0.557 against 0.417), for 4.7× the latency.
- **Training on 13 points.** Most of the 4-5 point loss is in placing the face points precisely. With one loose tolerance for every point, the gap shrinks to 1.1-1.5 points.
- **The s HEF.** It still finds people (AP50 0.683) but places keypoints poorly (AP75 0.127). On matched detections its median keypoint error is 4-9% of the box size, against 2-3% for the n and m HEFs, and every point is shifted the same way. The n model was compiled with the same script and is unaffected, so the fault is in this compile, not in the settings. The s model appears only in the speed results.
- **Training-log scores.** The log's keypoint mAP50-95 (0.329, 0.376, 0.401) is far below the same models scored by pycocotools with the same uniform tolerance (0.579, 0.660, 0.706). The cause is people with no labelled keypoint among the 13. They make up 4,503 of the 10,777 people in the validation labels (41.8%). pycocotools ignores them, but Ultralytics' validator counts them as targets it can never match. With those people removed, Ultralytics' own validator gives 0.555, 0.632 and 0.669, close to the pycocotools figures. The log therefore understates keypoint accuracy, and its keypoint scores are not used as results.

For the Hailo-8 each model is quantised to INT8 and compiled with the Hailo Dataflow Compiler into a HEF:

- **No on-chip NMS.** The nine raw head outputs are exported as end nodes: box distribution (DFL), class score and keypoints at strides 8, 16 and 32. Decoding and NMS run on the host, which keeps the HEF independent of the model's keypoint count.
- **Sigmoid inside the HEF.** The class scores are passed through sigmoid inside the HEF. The runtime detects this per model, which fixed a double sigmoid in the code it replaced.

| Model | Hailo-8 FPS | Hardware latency (ms) |
| --- | --- | --- |
| n | 131.5 | 5.7 |
| s | 69.5 | 11.9 |
| m | 31.5 | 26.7 |

**Quantisation.** The compiler (Hailo Dataflow Compiler 3.33.1, Model Zoo 2.18) calibrated the INT8 ranges on 1,024 random train2017 images resized to 640×640. The n and s models share one model script and m has its own. Both scripts keep three of the nine output layers at 16 bits; by their order these are the keypoint outputs. Both run Hailo's post-quantisation fine-tuning at learning rate 0.00015, and the n/s script also turns off pre-quantisation equalisation.

## 3.3 Runtime: hcpose

hcpose runs any YOLOv8-pose HEF on a Raspberry Pi 5. It reads the keypoint count, strides, tensor order, output data type and score activation from the HEF at start-up, so the same command runs the 13-keypoint models, the stock 17-keypoint model or others.

**Two engines, one result.**

- **GStreamer engine** (default, for live use): the camera or file source feeds `hailonet`. A C++ `hailofilter` decodes and post-processes the outputs, `hailotracker` assigns track ids, and `hailooverlay` draws.
- **Python engine** (for images, debugging and evaluation): OpenCV and HailoRT with a NumPy decoder.
- **Equivalence:** the two decoders produce identical detections, checked by the tests on recorded model outputs.

**Post-processing.**

- **Decoding:** DFL boxes and keypoints are decoded and mapped back through the letterbox.
- **NMS:** candidates are sorted by score first. The inherited code skipped this, so overlaps were resolved by anchor order.
- **Duplicate-skeleton removal:** after NMS, a box is dropped when it lies inside a stronger box and its skeleton matches: at least three common keypoints, closer on average than 0.1 × √(smaller box area). YOLO-pose often outputs a tight and a loose box for one student, and box NMS keeps both.

**Tiling for far rows.** With `--tiles`, the model also runs on a row of overlapping, roughly square tiles over the top 60% of the frame (three tiles at 1280×720).

- **In the GStreamer engine:** a custom crop function hands `hailocropper` the full frame plus the tiles. A second `hailofilter` after the aggregator merges the results.
- **The merge:** tile detections are mapped to frame coordinates, those cut by an inner tile edge are dropped, and duplicates across passes are merged by IoU, or by containment plus agreeing keypoints.
- **Checks:** the C++ merge equals the Python merge on 25 random scenes.
- **Cost:** tiling multiplies inference by four.

**\[Figure to add, manual\] The tiles.** Use one frame of the exam-hall clip \[17\], whose students are adults. On a 1280×720 frame, `--tiles auto` uses three tiles over the top 60%, each overlapping its neighbour by 20%: x 0-493, 394-887 and 788-1280, all with y 0-432. Draw the three rectangles on the frame. Ideally show two panels, detections without tiles and with tiles, made from two runs with `-o`:

```bash
./hcpose.sh run ~/models/upper_body_n.hef -i ~/classroom/clip.mp4 --headless --score-threshold 0.3 -o ~/fig/no_tiles.mp4
./hcpose.sh run ~/models/upper_body_n.hef -i ~/classroom/clip.mp4 --headless --score-threshold 0.3 --tiles auto -o ~/fig/tiles.mp4
ffmpeg -ss 30 -i ~/fig/no_tiles.mp4 -frames:v 1 ~/fig/no_tiles.png && ffmpeg -ss 30 -i ~/fig/tiles.mp4 -frames:v 1 ~/fig/tiles.png
```

Suggested caption: *Detections on the exam-hall clip without (left) and with (right) the three tiles over the far rows.*

**Outputs.**

- The annotated window, or headless operation.
- Optionally an annotated video (`--output`), for development only.
- Optionally a JSON-lines skeleton file (`--save-json`) with boxes and keypoints, no pixels.

**\[Figure to add, manual\] The two displays.** Two screenshots of the Pi's screen on clip\_4: per-student colours (`--display both`) and the teacher's view (`--display class`). Run each command from the Pi's desktop and capture once the seats are calibrated, after about 2 minutes (`grim ~/fig/screen.png` on Raspberry Pi OS with Wayland):

```bash
./hcpose.sh run ~/models/upper_body_n.hef -i ~/classroom/clip_4_720p.mp4 --engagement --display both --tiles auto --score-threshold 0.3
./hcpose.sh run ~/models/upper_body_n.hef -i ~/classroom/clip_4_720p.mp4 --engagement --display class --tiles auto --score-threshold 0.3
```

Suggested caption: *Left: development view with per-student states (green focused, red distracted, white calibrating). Right: the teacher's view, with the class share and its 95% range only.*

## 3.4 Engagement analysis

The analyzer turns each frame's skeletons into a focused, distracted or unknown state per student and a class-level share with a range. It works on keypoints alone and can replay a saved skeleton file (`hcpose analyze`) with any settings.

**Seats.** Students are tracked as anonymous seats, matched frame to frame by the top centre of their box (the head). The head stays put when a student leans or turns around, while the box centre jumps. A seat counts toward the class only while it is current and consistently seen.

**Features**, normalised by shoulder width:

- **Two head-turn measures.** Ear-based yaw: the nose between the ears. Head offset: the nose relative to the shoulder centre, which keeps working in profile.
- **Head height** above the shoulder line.
- **Torso ratio**, which shrinks when the body rotates.
- **Face hidden** while the shoulders are visible.
- **Hand raised**, and **hand at face**.

**Per-seat calibration.**

- Each seat learns its own attentive pose as the median of its first 120 s of usable observation. Before that, partial medians and the room's median are used.
- The seat's yaw and head-offset baselines are kept within 0.3 and 0.15 of the room's, so a student who looks away during calibration does not learn "turned" as normal.
- `--camera-position` (left, centre, right) sets a prior for the room's attentive direction.

**Decision.** Each cue is a ramp from "normal" to "clearly off" relative to the seat's baseline. They combine with a noisy-OR, so weak cues such as head down (writing looks the same) only count together with another cue:

```latex
s = 1 - \prod_{c} \left(1 - w_c \, r_c\right)
```

Here r\_c is cue c's ramp value and w\_c its weight. A raised hand overrides the score to 0. The default weights are: head turned 1.0, face hidden 0.9, head down 0.7, body turned 0.5, hand at face 0.4. The learned classifiers (trees, logistic regression, MLP) can replace the noisy-OR, but are off by default (chapter 4).

**Smoothing.** The score is smoothed with an exponential moving average (τ = 1 s) and switched with hysteresis:

- **to distracted:** above 0.6 for 2.5 s
- **back to focused:** below 0.4 for 1.5 s
- **unknown:** no usable keypoints for 2 s

This keeps brief glances from flipping a student.

**Class estimate.**

- Seats keep their last state for 15 s after they were last seen.
- They are grouped into near, middle and far zones by image height, and each zone is weighted by its number of seats.
- The class share comes with a 95% Wilson interval \[16\] around the shown (smoothed) value. It is narrowed by the finite-population correction only when the real head count is given (`--class-size`).
- On screen: "43% distracted (20-65%) | 14 of \~30 students".

## 3.5 Configuration, testing and reproducibility

**Configuration.** Every threshold, weight and option is a setting in `configs/engagement.yaml` or a command-line flag:

- camera position
- display mode (per-student, class, both or none)
- class size
- calibration file
- classifier
- tiling

A per-model sidecar YAML sets detection thresholds and keypoint names.

**Tests.** 44 automated tests cover:

- both decoders against recorded HEF outputs and Ultralytics' own decoding
- the C++ letterbox mapping and tile merge against Python
- the duplicate removal
- the engagement logic on synthetic classes: calibration, glances, missing frames, seat counting, the interval, the head-offset measure
- the staged-session scorer on a synthetic session with a known answer

**Reproducibility.** Results and the tools that made them are in the repository:

- skeleton files from the Pi
- edge logs
- the human labels
- staged-session scripts
- threshold sweeps

Every analysis can be rerun from the saved skeletons without the camera footage. The evaluated version is tagged `v1.0-thesis`.
