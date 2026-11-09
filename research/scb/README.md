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

`evaluate.py`:
- Sorts each image into front, back or mixed camera view by how many students show a face.
- Matches SCB boxes to pose detections and reports how many labelled students were detected at all.
- Scores the current rules in their uncalibrated single-frame mode, per behavior.

`learn.py` trains a gradient-boosted classifier on front-view keypoints with 5-fold cross-validation split by source lesson, so no lesson appears in both training and test data. It compares the classifier with the rules on the same students and reports which feature groups matter.

Single images can't test the temporal smoothing or the per-seat calibration, so these numbers are a lower bound for the live system.

## First results (stand-in yolov8n-pose, PyTorch FP32)

**Images by view:** 5,794 front, 2,398 back, 1,172 mixed, 774 with too few people.

**Detection coverage:** only 16–38% of labelled students matched a detection at confidence 0.5. The misses are mostly small, back-row students.

**Front view, 16,896 students from 264 lessons:**

| Method | AUC | Balanced accuracy |
|---|---|---|
| Rules | 0.56–0.63 | 54% |
| Learned | **0.74** | **68.5%** |

**Share flagged "distracted" by the learned classifier:**
- TurnHead: 70% (the rules flag 16%)
- discuss: 64%
- read, write, hand-raising: 22–24% (false alarms)
- BowHead: 16%. A bowed head can't be told apart from reading or writing.
