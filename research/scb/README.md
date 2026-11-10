# Engagement evaluation on the SCB classroom dataset

[SCB-Dataset](https://github.com/Whiffe/SCB-dataset) ([paper](https://arxiv.org/abs/2304.02488)) has classroom images with behavior boxes. We use three subsets:

| Label | Treated as |
|---|---|
| read, write, hand-raising | focused |
| TurnHead, BowHead, discuss | distracted |

These are *action* labels, not attention ratings, so treat the results as a proxy. The dataset states no license, so use it for research with a citation and don't redistribute it.

## Run on the Pi (real INT8 HEF on the Hailo-8)

```bash
# 1. data (~1.2 GB) from the Hugging Face mirror
mkdir -p ~/scb && cd ~/scb
B=https://huggingface.co/datasets/wintonYF/SCB-Dataset/resolve/main
for f in SCB_BowTurnHead_20250509/SCB_BowTurnHead_20250509.zip \
         SCB5-Handrise-Read-write-2024-9-17/SCB5-Handrise-Read-write-2024-9-17.zip \
         SCB5-Discuss-2024-9-17/SCB5-Discuss-2024-9-17.zip; do
  wget -q --show-progress "$B/$f" && unzip -q -o "$(basename $f)" -d "$(basename $f .zip)"
done

# 2. pose on every image with the deployed model
cd ~/hailo-custom-yolo-pose-runtime
.venv/bin/pip install scikit-learn
for m in n s m; do
  .venv/bin/python research/scb/extract_hef.py ~/models/upper_body_$m.hef ~/scb/poses_hef_$m.jsonl \
    ~/scb/SCB_BowTurnHead_20250509/SCB_BowTurnHead_20250509/SCB5-Turn-Bow-Head-2024-9-17:BowHead,TurnHead \
    ~/scb/SCB5-Discuss-2024-9-17/SCB5-Discuss-2024-9-17:discuss \
    ~/scb/SCB5-Handrise-Read-write-2024-9-17/SCB5-Handrise-Read-write-2024-9-17:hand-raising,read,write
done

# 3. evaluate the rules and a learned classifier
for m in n s m; do
  echo "== $m"; .venv/bin/python research/scb/evaluate.py ~/scb/poses_hef_$m.jsonl --out ~/scb/report_$m.json
  .venv/bin/python research/scb/learn.py ~/scb/poses_hef_$m.jsonl
done
```

`extract_pt.py` does the same with a PyTorch `.pt` (Ultralytics) on any machine. The first results came from it, using the stock yolov8n-pose as a stand-in.

## Train the learned classifier

The classifier takes the keypoints and pose cues computed by `hcpose.engagement.classifier.feature_vector()`, the same code the live analyzer runs. Image-context features (position in the frame, crowd size) are deliberately excluded.

`train_classifier.py`:
- Compares logistic regression, gradient-boosted trees and a small MLP with 5-fold cross-validation by lesson.
- Fits gradient-boosted trees on all the data.
- Exports them as JSON, which the Pi evaluates in NumPy (no scikit-learn needed at runtime).

```bash
# on the Pi: pose caches from the deployed m HEF, at the runtime detection threshold
.venv/bin/python research/scb/train_classifier.py ~/scb/poses_hef_m.jsonl --det-thr 0.3 \
    --name scb-m-int8 --export-dir hcpose/engagement/models
```

This writes three models to `hcpose/engagement/models/`, all trained on the same students and features:

| File | Model | Select with |
|---|---|---|
| `scb_trees.json` | gradient-boosted trees | `--classifier trees`, or `auto` (the default) |
| `scb_logistic.json` | logistic regression | `--classifier logistic` |
| `scb_mlp.json` | MLP 64-32 | `--classifier mlp` |

`--classifier none` uses the rules. Each export is checked against scikit-learn (max difference below 1e-5). If a model file already exists, the script also reports how far the new predictions are from the old file's; for the trees it should say "identical", because training is deterministic.

All three output a *balanced* probability: 0.5 means equally likely under equal class priors. The trees and logistic regression get this from balanced class weights. The MLP can't use class weights, so its logit is shifted by the training prior (`logit_offset` in its JSON). The analyzer's thresholds therefore mean the same thing for all three. Cross-validated balanced accuracy for the MLP is now measured at 0.5 on that scale, rather than at a prevalence-matched threshold as before.

**For the report:**
- `--save-cv cv_m.npz` keeps every family's out-of-fold predictions, the labels, the lesson groups and the feature matrix, plus a `.json` with the metrics. Every table, ROC curve or confusion matrix can be redrawn from them without retraining.
- `--save-models DIR` also fits and pickles the logistic regression and the MLP.

Everything uses fixed seeds, so rerunning the script reproduces the same numbers.

Stand-in yolov8n-pose, FP32, threshold 0.3, 22,674 students:

| Model | AUC | Within-image AUC | Balanced accuracy |
|---|---|---|---|
| Logistic regression | 0.768 | 0.684 | 70.7% |
| Gradient-boosted trees | 0.831 | 0.750 | 74.7% |
| MLP 64-32 | 0.832 | 0.739 | 72.2% |

Training takes about a minute on 2 CPU cores. The exported 200-tree model is 0.6 MB and scores 30 students in about 0.7 ms.

## What the scripts measure

Both scripts first run `scbdata.load()`, which fixes two dataset issues the first Pi run found:
- **Duplicate images.** About 1,900 images are byte-identical copies across subsets, and each subset labels only its own behaviors. Copies are merged into one image with the union of their labels. A student labelled both focused and distracted across copies is dropped as ambiguous.
- **Lesson groups.** Cross-validation groups are whole lessons, with duplicate images joined into the same group, so no lesson or image appears on both sides of a fold.

`--no-dedup` restores the old behavior.

`evaluate.py`:
- Sorts each image into front, back or mixed camera view.
- Reports detection coverage: how many labelled students got a pose detection at `--det-thr` (default 0.5).
- Scores the rules in their uncalibrated single-frame mode, per behavior.

`learn.py`:
- Trains a gradient-boosted classifier on front-view keypoints, with 5-fold cross-validation grouped by lesson.
- Reports which feature groups matter.
- Runs two **shortcut checks**. Most SCB images contain only focused or only distracted labels, so a model can partly "recognize the image" instead of the pose. A context-only model (position, box size, crowd size, no pose) shows how big that shortcut is. The **within-image AUC**, computed only on images with both classes, is the number to quote: it compares students in the same room and frame, so the shortcut can't help.

Class order per subset comes from the dataset's own YAML files on the Hugging Face mirror:

| Subset | Classes |
|---|---|
| BowTurnHead | BowHead, TurnHead |
| Discuss | discuss |
| Handrise | hand-raising, read, write |

Single images can't test the temporal smoothing or the per-seat calibration, so these numbers are a lower bound for the live system.

## Results so far

The deployed INT8 HEFs on the Hailo-8, before the fixes above, are in [`results/hef_results.md`](results/hef_results.md). In that run the learned classifier beat the rules for all three models, and m was best (AUC 0.74–0.81 depending on grouping, against 0.53–0.55 for the rules).

**Stand-in yolov8n-pose (PyTorch FP32), with the duplicate merging and lesson groups:**

**Data after merging:**
- 8,116 unique images, from 10,138.
- 2,891 students had both a focused and a distracted label and were dropped:
  - 2,078 (72%) were "BowHead" in one copy and "read" or "write" in another, so even the SCB annotators couldn't separate a bowed head from reading or writing.
  - 674 were "TurnHead" and "hand-raising" at the same time.

**Classifier vs rules (front view):**

| Measure | Learned | Rules |
|---|---|---|
| Overall AUC | 0.854 | 0.559 |
| Context-only AUC (no pose) | 0.646 | – |
| **Within-image AUC** | **0.763** | **0.542** |

Part of the overall 0.854 is the image shortcut. The within-image figure is the honest one, and it still shows clear pose signal, while the rules sit near chance.

**Detection coverage** of labelled front-view students, by detection threshold:

| `--det-thr` | Coverage |
|---|---|
| 0.5 | ~40% |
| 0.35 | ~48% |
| 0.25 | ~57% |

For classrooms, a threshold around 0.3–0.35 (`score_threshold` in the model sidecar) is worth testing.

## Tiling: detecting the far rows

`extract_pt.py --tiles auto` runs the model on the full frame plus one row of overlapping, roughly square tiles over the top 60% of the image, and merges the detections the same way `hcpose run --tiles auto` does. `compare_coverage.py` compares two caches on the same images.

The test used 1,500 front-view SCB images (500 per subset) with the stand-in yolov8n-pose, and counted labelled students matched to a detection.

| Students | No tiling @0.3 | Tiles @0.3 | No tiling @0.5 | Tiles @0.5 |
|---|---|---|---|---|
| **All** | 44.3% | **73.7%** | 29.6% | 57.1% |
| Small (far third) | 11.0% | **70.3%** | 4.7% | 48.0% |
| Medium | 50.5% | 77.2% | 28.1% | 61.3% |
| Large (near third) | 71.4% | 73.6% | 56.0% | 62.0% |
| BowHead | 26.7% | 59.4% | 16.3% | 40.7% |
| TurnHead | 50.3% | 86.7% | 31.2% | 68.5% |
| discuss | 37.2% | 58.7% | 24.8% | 46.2% |
| read | 51.8% | 82.9% | 36.4% | 64.7% |

Detections per image went from 12.9 to 26.0 at threshold 0.3. Visual checks show the extra detections are mostly real back-row students with usable skeletons. Tiling also narrows the gap between behaviors, which reduces the bias toward attentive poses noted earlier.

**Same test with the project's own upper_body_n** (`best.pt`, 13 keypoints, PyTorch FP32, same 1,500 images, 11,903 labelled students):

| Students | No tiling @0.3 | Tiles @0.3 | No tiling @0.5 | Tiles @0.5 |
|---|---|---|---|---|
| **All** | 48.6% | **75.5%** | 30.7% | 58.9% |
| Small (far third) | 17.0% | **74.4%** | 5.8% | 52.9% |
| Medium | 57.2% | 78.3% | 31.0% | 62.0% |
| Large (near third) | 71.6% | 73.9% | 55.3% | 61.9% |
| BowHead | 28.4% | 62.2% | 17.7% | 44.9% |
| TurnHead | 56.1% | 89.2% | 32.3% | 70.5% |
| discuss | 40.3% | 60.8% | 25.4% | 48.4% |
| hand-raising | 49.9% | 70.6% | 33.7% | 53.6% |
| read | 57.1% | 84.4% | 38.3% | 66.3% |
| write | 52.6% | 74.3% | 35.8% | 57.0% |

Detections per image went from 14.7 to 27.6 at threshold 0.3. The upper-body model starts a little higher than the stock one, and with tiles the far third is detected about as often as the near third (74% vs 74%), so coverage no longer depends much on distance. No model weights were changed; this is only how the frame is fed to the model.

```bash
python research/scb/extract_pt.py best.pt ub_tiles_none.jsonl --conf 0.25 <SCB dirs...>
python research/scb/extract_pt.py best.pt ub_tiles_auto.jsonl --conf 0.25 --tiles auto <SCB dirs...>
python research/scb/compare_coverage.py ub_tiles_none.jsonl ub_tiles_auto.jsonl --names none,tiles --det-thr 0.3
```

**Duplicate boxes.** About one box per image is a second box for a student who already has one: a looser box with the same skeleton. Box NMS keeps it because the two boxes overlap little. The decoders now drop it (`pose_dedup`, see the main README). On the upper_body_n caches this removed 6.5% of detections without tiles and 6.3% with them. Measured coverage fell 0.6 and 0.8 points, because the SCB matcher had been crediting a second student with the duplicate box. The tables above are from before this change.

### Do the classifiers work on tiled poses?

`eval_tiling.py` scores the bundled classifiers on the same students with and without tiling, and on the students only tiling finds. Results for upper_body_n `best.pt`, threshold 0.3; "pred" is the share the classifier calls distracted:

| Students | n | True distracted | Trees AUC | Trees pred | MLP AUC | MLP pred | Logistic pred | Rules AUC |
|---|---|---|---|---|---|---|---|---|
| Found by both, untiled pose | 5,160 | 55% | 0.770 | 55% | 0.768 | 50% | 61% | 0.571 |
| Found by both, tiled pose | 5,160 | 55% | 0.792 | 47% | 0.799 | 45% | 52% | 0.593 |
| Found only with tiles | 3,348 | 65% | 0.705 | 47% | 0.698 | 44% | 49% | 0.582 |

The tiled pose of a student ranks *better* (AUC up), but the classifiers call it distracted less often. They were trained on full-frame poses only, where a far student is a few pixels wide. The far students that only tiling finds are under-called the most: 47% predicted against 65% true. This explains most of the drop in the class distracted share that the Pi check saw with tiles on. The SCB images are subset-balanced, so the true shares here aren't classroom rates. The bundled models were trained on these images (with the m HEF's poses), so their absolute AUCs here are optimistic; the point is the difference between the rows.

The fix is in the classifier, not the detector. Training on untiled and tiled poses together, with lesson-grouped out-of-fold predictions on these caches (trees), gave:

| Students | True | Pred, trained untiled | Pred, trained untiled + tiled | AUC untiled → pooled |
|---|---|---|---|---|
| Both, untiled pose | 55% | 52% | 54% | 0.720 → 0.719 |
| Both, tiled pose | 55% | 44% | 52% | 0.728 → 0.748 |
| Only with tiles | 65% | 38% | 53% | 0.646 → 0.689 |

The bundled classifiers (`scb-m-int8-tiles`) are now trained this way, on pooled caches from the deployed m HEF. On the Pi, the pooled lesson-grouped CV gave AUC / within-image AUC / balanced accuracy of 0.855 / 0.840 / 77.0% for trees, 0.863 / 0.838 / 78.0% for the MLP and 0.772 / 0.735 / 71.3% for logistic regression. See [`results/pi_followup.md`](results/pi_followup.md), which also has the runtime recheck. Commands:

```bash
# on the Pi: INT8 m caches with and without tiles (the decoder now applies pose_dedup)
.venv/bin/python research/scb/extract_hef.py ~/models/upper_body_m.hef ~/scb/poses_hef_m.jsonl <SCB dirs...>
.venv/bin/python research/scb/extract_hef.py ~/models/upper_body_m.hef ~/scb/poses_hef_m_tiles.jsonl <SCB dirs...> --tiles auto
.venv/bin/python research/scb/train_classifier.py ~/scb/poses_hef_m.jsonl ~/scb/poses_hef_m_tiles.jsonl \
    --det-thr 0.3 --name scb-m-int8-tiles --export-dir hcpose/engagement/models --save-cv ~/scb/cv_m_pooled.npz
.venv/bin/python research/scb/eval_tiling.py ~/scb/poses_hef_m.jsonl ~/scb/poses_hef_m_tiles.jsonl
```

The cost is one extra inference per tile: full frame plus 3 tiles is four passes per frame, which is fine at the 5–10 FPS that engagement needs. Both engines support it; the gst engine does the crops and the merge in C++ (`cpp/tiling.cpp`), with the same rules as the Python engine.

## Check on a real classroom clip

A 72 s exam-hall CCTV clip (1280x720, about 30 students writing, camera high at the front corner; not in the repo). The true distracted share should be close to 0. Mean class distracted %, gst engine on the Pi, pooled classifiers:

| Model | Tiles | Trees | Rules |
|---|---|---|---|
| m | none | 12.8% | – |
| m | auto | 8.7% | – |
| n | none | 35.8% | – |
| n | auto | 15.6% | – |

The same clip through `best.pt` (n, FP32) and `hcpose analyze` on a PC, with `pose_dedup`, gave:
- Trees: 33% without tiles, 12% with them. Logistic: 43% / 26%. MLP: 43% / 14%.
- Rules: 8% without tiles, 5% with them.
- An earlier Pi run with m untiled gave 10.5% for trees and 11.0% for the rules.

So the learned classifiers transfer to this room only when the poses are good: the m model, or n with tiles. With n's full-frame poses they over-call heavily. Retraining on the n model's poses with head-height cues removed didn't fix it in a quick test, so the cause is pose quality, not the SCB label definitions. The rules are less accurate on SCB but much more stable across models. This is single-clip evidence, so it calls for a small labelled set from the target classrooms before choosing the classifier for deployment.

## Check on a dense self-study room with a low camera

A 6 min study-hall clip (about 50 students, all reading or writing, camera at desk height on the teacher's desk; not in the repo), with `best.pt` (n, FP32) at 10 FPS, `pose_dedup`, `--camera-position center`:

| | No tiles | Tiles |
|---|---|---|
| Students per frame | 9.7 | 23.7 |
| Class size inferred from seats, before the seat fix | 28 | 78 |
| Class size inferred from seats, after the seat fix | 19 | 52 |
| Class % after 2:30, trees | 24% | 37% |
| Class % after 2:30, rules | 37% | 44% |
| Class % after 2:30, rules, `face_hidden` 0.4 | – | 23% |
| Class % after 2:30, rules, `face_hidden` 0 | – | 16% |

**What this shows:**
- **Tiling** more than doubles the students found in a dense room.
- **Seat fix.** Before it, every box that persisted about 1.5 s became a permanent seat, so the inferred class size grew to 78. Now a seat counts only if seen in the last 60 s and, before calibration, in at least 15% of frames. That gives 52, close to the real count.
- **Low camera.** Most "distracted" labels here (two thirds, rules) come from `face_hidden`. A head bowed over the desk toward a low camera hides the face, so it looks like turning away. Shoulder left/right order doesn't separate the two: the model guesses a back view for most bowed heads. With a camera high on the wall (the ALTIMA clip) faces stay visible while writing, and the weight barely matters there (rules 5.4% → 4.2% at 0.4).
- **Default kept.** It stays at 0.9 for the intended high mounting, and the setting is documented for low cameras.
- **Calibration.** Only 20 of 52 seats had finished calibrating after 6 min. Far students are detected intermittently and need 120 s of usable observation, so a session needs to be longer than a few minutes before per-seat baselines apply to most of the room.

## Check on an attentive lecture with a high front camera

A 3 min 18 s clip (about 35 middle-school students, camera high at the front like the ALTIMA clip, faces visible; nearly all students sit upright facing the front or read the book on the desk; not in the repo). `best.pt` (n, FP32) at 12.5 FPS, `pose_dedup`, `--camera-position center`. Class distracted %, mean after 2:30:

| | No tiles | Tiles |
|---|---|---|
| Students per frame | 14.5 | 19.3 |
| Class size inferred from seats | 19 | 28 |
| Trees | 60% | 62% |
| Logistic | 89% | 89% |
| MLP | 61% | 66% |
| Rules | 0% | 1% |

The learned classifiers call students who sit upright and look forward *distracted*, and students reading with the head down *focused* (median head height above the shoulders 0.45 vs 0.32 at 2:00). This is a gap in SCB's labels, not pose quality:
- SCB's focused classes are read, write and hand-raising, all head-down or hand-up poses.
- Its distracted classes include TurnHead and discuss, both head-up.
- SCB has no "listening / looking at the board" label, which is the most common attentive state in a lecture.

So a model trained on SCB learns "head up = distracted". The rules compare each student with their own calibrated pose and with the room, and they get this clip right. They also gave 5–8% on the exam-hall clip.

**Conclusion for deployment:**
- The SCB-trained classifiers rank SCB actions well (within-image AUC 0.84) but don't measure attention in a lecture.
- The rules are the better default for live use. The learned models remain useful as an SCB benchmark and as a comparison in the thesis.
- Making them usable in lectures would need training labels that include attentive, head-up students from the target classrooms.
