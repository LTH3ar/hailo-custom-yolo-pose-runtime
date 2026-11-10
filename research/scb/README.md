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

The cost is one extra inference per tile: full frame plus 3 tiles is four passes per frame, which is fine at the 5–10 FPS that engagement needs.
