## Handoff 05 report (2026-11-22)

### PR #8 and tests
- PR #8 is merged; training repo main is at `13e6eee`.
- pi5, `classifiers/tests` plus `baselines/crop_cls/tests`: 15 passed, 1 failed.
  - The failure is `test_clips_runs`, with `No module named 'cv2'`. pi5's `.venv_cls` has no OpenCV, and the stand-in for Ultralytics doesn't cover the test video. I didn't install OpenCV there.
- ubuntu3060, all 16 tests: passed.
  - The venv needed the runtime package (`pip install -e hailo-custom-yolo-pose-runtime`), as the requirements file says for export_split.

### A. Split
- 67,845 labelled students, of which 43,940 are evaluated (matched to a pose box). They form 230 groups in 5 folds.
- ClaBehavior uses the 0.40 grouping from Handoff 04: 42 groups. I checked that `groups.json` equals `h04/corr0.40/groups.json`.
- Students per fold:

  | Dataset | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
  |---|---|---|---|---|---|
  | SCB | 8,224 | 14,133 | 12,304 | 11,543 | 13,558 |
  | ClaBehavior | 4,593 | 352 | 512 | 2,495 | 131 |

  ClaBehavior is very uneven across folds because its two big camera groups (155 and 122 frames) are each in a single fold.
- **Warning:** `[warn] -340 classifier students not found among the labelled boxes`.
  - The negative count means 340 boxes are listed twice in the labels. 168 are raise_hand + read on the same box, 144 are the same action twice, and the rest are other pairs; 16 are ClaBehavior pairs with lookup.
  - The classifiers count each box once; the crop split counts it twice, for training and for testing. That's 0.8% of the evaluated set.

#### Pose classifiers on the split (AUC / within-image AUC / balanced accuracy)
- **Thesis mapping (SCB):**
  - Logistic 0.765 / 0.738 / 70.5
  - Trees 0.853 / 0.838 / 77.2
  - MLP 0.857 / 0.837 / 77.6

| Mapping | Model | Pooled | On SCB | On ClaBehavior |
|---|---|---|---|---|
| Strict | Logistic | 0.773 / 0.730 / 71.1 | 0.781 / 0.756 / 71.9 | 0.695 / 0.675 / 65.0 |
| Strict | Trees | 0.876 / 0.846 / 79.1 | 0.881 / 0.853 / 79.5 | 0.840 / 0.830 / 75.9 |
| Strict | MLP | 0.893 / 0.863 / 80.9 | 0.898 / 0.874 / 81.5 | 0.852 / 0.840 / 76.0 |
| Lenient | Logistic | 0.759 / 0.723 / 70.2 | 0.760 / 0.730 / 70.3 | 0.725 / 0.703 / 67.4 |
| Lenient | Trees | 0.851 / 0.833 / 77.3 | 0.852 / 0.835 / 77.3 | 0.841 / 0.830 / 75.4 |
| Lenient | MLP | 0.865 / 0.844 / 78.6 | 0.865 / 0.842 / 78.5 | 0.852 / 0.847 / 76.5 |

- Lenient is identical to Handoff 04's 0.40 run, because the split's folds are built from the lenient students.
- The cross-dataset lines are identical to Handoff 02 and 04. For example, SCB → ClaBehavior with strict labels gives trees 0.790 and MLP 0.794.

### B. Data and environment
- Copied only the 8,507 images that `split.json` uses, pi5 → ubuntu3060 directly, with `rsync -aL` so the symlinks were resolved: 1.3 GB in `~/hcpose/pi_data/`.
- Both machines' home folder is `/home/raven`, so the path map is `/home/raven=/home/raven/hcpose/pi_data`.
- Environment `~/hcpose/venvs/cropcls`: Python 3.11.18, torch 2.15.0+cu132 (CUDA 13.2), ultralytics 8.4.216, RTX 3060 12 GB, driver 595.99.
- ubuntu3060 cannot ssh to pi5, so every copy back to pi5 was a pull from pi5.

### C. Crops and training
- 67,842 crops from the labelled boxes (3 boxes too small), and 43,940 pose-box crops for testing. No missing images.
- Batch 128 fit in GPU memory: 2.7 GB in use, about 10 it/s.

| Run | Training crops | Minutes | Last top-1 (val overlaps training) |
|---|---|---|---|
| fold0 | 55,025 | 35 | 0.987 |
| fold1 | 53,359 | 34 | 0.983 |
| fold2 | 55,026 | 36 | 0.984 |
| fold3 | 53,805 | 35 | 0.990 |
| fold4 | 54,153 | 35 | 0.990 |
| all | 67,842 | 44 | 0.982 |
| scb_only | 59,759 | 38 | 0.986 |
| cla_only | 8,083 | 4 | 0.993 |

### D. Crop baseline (YOLOv8s-cls, FP32; AUC / within-image AUC / balanced accuracy)
- **Thesis mapping (SCB):** 0.897 / 0.849 / 76.6, on 36,687 crops.

| Mapping | Pooled | On SCB | On ClaBehavior |
|---|---|---|---|
| Strict | 0.912 / 0.860 / 79.2 (41,289) | 0.929 / 0.869 / 80.5 | 0.815 / 0.844 / 71.9 |
| Lenient | 0.883 / 0.846 / 75.6 (43,940) | 0.897 / 0.849 / 76.5 | 0.815 / 0.838 / 71.4 |

Cross-dataset:

| Mapping | Train → test | AUC | Within-image AUC | Balanced acc. | Pred / true distracted |
|---|---|---|---|---|---|
| Strict | ClaBehavior → SCB | 0.668 | 0.639 | 61.0% | 25% / 21% |
| Strict | SCB → ClaBehavior | 0.720 | 0.770 | 65.1% | 31% / 13% |
| Lenient | ClaBehavior → SCB | 0.647 | 0.610 | 60.2% | 34% / 26% |
| Lenient | SCB → ClaBehavior | 0.713 | 0.762 | 63.9% | 31% / 14% |

- **Speed:** 87,880 crop predictions (5 folds, plus scb_only and cla_only on the other dataset) took 127 s. That's about 690 crops per second including JPEG reading and 7 model loads (FP32, RTX 3060).
- **Against the pose classifiers on the same folds:**
  - The crop model is ahead on SCB AUC: strict 0.929 vs MLP 0.898.
  - Its within-image AUC is about equal: 0.869 vs 0.874. So the AUC lead comes mostly from telling images apart, not students within an image.
  - It is behind on ClaBehavior (0.815 vs 0.852) and cross-dataset (SCB → ClaBehavior 0.720 vs 0.794).

### E. Clips (boxes and seats from Handoff 02's rules runs, tiles on; model `runs/all`)
Class % is the plain share of the frame's students marked distracted.

**v1 labels** (262 focused, 13 distracted; lecture + exam):

| Model | Mapping | Output | Not matched | False alarms / 262 | Specificity (95% CI) | Caught / 13 | Re-weighted (lecture / exam) |
|---|---|---|---|---|---|---|---|
| n | strict | raw | 32 | 26 | 90% (86–93) | 3 | 13.7 / 0.4% |
| n | strict | smooth | 32 | 15 | 94% (91–97) | 1 | 7.9 / 0.1% |
| n | lenient | raw | 32 | 29 | 89% (85–92) | 3 | 13.7 / 4.2% |
| n | lenient | smooth | 32 | 15 | 94% (91–97) | 1 | 7.9 / 0.1% |
| m | strict | raw | 13 | 31 | 88% (84–92) | 3 | 16.9 / 2.1% |
| m | strict | smooth | 13 | 23 | 91% (87–94) | 1 | 13.0 / 0.1% |
| m | lenient | raw | 13 | 33 | 87% (83–91) | 3 | 17.5 / 2.3% |
| m | lenient | smooth | 13 | 23 | 91% (87–94) | 1 | 13.0 / 0.1% |

- For comparison, Handoff 02 with m tiles:
  - Rules: 24 false alarms, 2 caught.
  - Thesis trees: 88, 4 caught.
  - Strict trees: 78.
  - Lenient MLP: 72.

**clip_5 labels** (121 focused, 28 distracted):

| Model | Mapping | Output | Not matched | False alarms / 121 | Specificity (95% CI) | Caught / 28 | Re-weighted |
|---|---|---|---|---|---|---|---|
| n | strict or lenient | raw | 4 | 2 | 98% (94–100) | 2 | 2.2% |
| n | strict or lenient | smooth | 4 | 1 | 99% (95–100) | 1 | 0.1% |
| m | strict or lenient | raw | 2 | 3 | 98% (93–99) | 4 | 4.3% |
| m | strict or lenient | smooth | 2 | 1 | 99% (95–100) | 3 | 0.4% |

- For comparison, Handoff 02 m tiles rules: 22 false alarms, 16 of 28 caught.

**Class % (lecture after 2:30 / whole exam clip):**

| Model | Mapping | Raw | Smooth |
|---|---|---|---|
| n | strict | 20.1 / 4.8% | 18.3 / 2.0% |
| n | lenient | 20.2 / 6.6% | 18.3 / 2.1% |
| m | strict | 27.2 / 5.6% | 23.9 / 3.2% |
| m | lenient | 27.4 / 7.6% | 23.9 / 3.3% |

- For comparison, Handoff 02 m tiles on the lecture: rules 1.4%, thesis trees 50.5%, logistic 74.4%.
- clip_5 whole-clip class %: 0.4–2.2%.

**clip_4** (script offset 0.1 s, seats (0.44, 0.37) and (0.61, 0.40), as in Handoff 02):

| Model | Mapping | Output | Seconds caught | Precision | False alarms attentive / writing | Events missed |
|---|---|---|---|---|---|---|
| n | strict | raw | 52% | 45% | 88% / 50% | 1 of 7 |
| n | strict | smooth | 58% | 47% | 89% / 50% | 2 of 7 |
| n | lenient | raw | 55% | 46% | 90% / 50% | 1 of 7 |
| n | lenient | smooth | 61% | 49% | 90% / 50% | 1 of 7 |
| m | strict | raw | 32% | 34% | 84% / 50% | 1 of 7 |
| m | strict | smooth | 34% | 36% | 84% / 50% | 2 of 7 |
| m | lenient | raw | 33% | 35% | 84% / 53% | 1 of 7 |
| m | lenient | smooth | 34% | 35% | 84% / 50% | 2 of 7 |

- Per behaviour, the crop model catches the phone under the table (100%) and, with n, the phone on the table (100%). It misses looking to the side (0–8%) and turning around (0–28%).
- The pose rules are the opposite: 56% caught, 100% precision, 0 / 0 false alarms, but both phones missed.

### Anything odd
1. **The comparison is not quite paired at the student level.** The classifiers count a student once per pose cache, so students found both with and without tiles count twice (e.g. 57,972 SCB students for thesis). The crop baseline tests one pose box per unique student (36,687). Same folds and same labels, but different weights.
2. **clip_4: the crop model flags 84–90% of the attentive seconds.** It calls the two adults distracted most of the time, and its class-% correlation is negative (r −0.46 to +0.02). This is probably a domain gap: adults on a different camera, while training was all school classrooms.
3. **`score.py` can't find the clip_4 start on the crop outputs.** The automatic start detection uses the raised-hands keypoints, which crop outputs don't have. I passed `--offset 0.1`, the value it finds on the Handoff 02 rules files of the same video.
4. **The two clip_4 encodes differ.** `clip_4_720p.mp4` on ubuntu3060 is a different encode from pi5's (different md5, same 8,563 frames at 1280×720). I copied pi5's encode to `ubuntu3060:~/hcpose/crop_cls/videos/` so the crops match the frames hcpose analysed. clip_4 is the adults' clip.
5. **"Not matched" for m on v1 is 13 here vs 18 in Handoff 02.** The crop model gives every box a state, whereas the rules leave some students unknown while calibrating, and unknown counts as not matched.
6. **Strict and lenient give near-identical clip results.** The crop model's probabilities are dominated by the actions both mappings share.
7. **Uncommitted tools:** `tools/d3_score_crop.py` (a copy of Handoff 02's `d3_score.py` with the crop decisions) and `run_e.sh`.

### Files
- `pi5:~/hcpose_paper/h05/`:
  - `split.json`, `split.txt`
  - `cls_*.{txt,json}`
  - `preds.jsonl`, `crop_cls.json`, `score.txt`
  - `predict.txt`, `make_crops.txt`, `train_times.txt`
  - `clips/` (32 `*_crop_*.jsonl`, logs, `scores/`)
  - `tools/`
- `ubuntu3060:~/hcpose/crop_cls/`: the same small files, plus `runs/`, `crops/`, `ds/` and `videos/`. Crops and frames stay there only.
