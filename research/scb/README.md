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
- 2,891 students had both a focused and a distracted label and were dropped. Most of them were "BowHead" in one copy and "read/write" in another, so even the SCB annotators couldn't separate a bowed head from reading or writing.

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
