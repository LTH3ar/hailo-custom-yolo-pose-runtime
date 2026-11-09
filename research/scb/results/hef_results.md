# SCB evaluation with the deployed INT8 HEFs (Hailo-8)

Run on 2026-11-09 on a Raspberry Pi 5 + Hailo-8 (HailoRT 4.23.0, TAPPAS 5.1.0, Python 3.13.5, scikit-learn 1.9.1), following `research/scb/README.md` exactly: `extract_hef.py` on all 10,138 images of the three SCB subsets (BowTurnHead 2,410, Discuss 864, Handrise-Read-write 6,864) for each of `upper_body_{n,s,m}.hef`, then `evaluate.py` and `learn.py` with default arguments. The pose caches (`poses_hef_*.jsonl`) and `report_*.json` are not committed.

## View split (images)

| Model | front | back | mixed | too few people |
|---|---|---|---|---|
| n | 5,555 | 2,582 | 1,098 | 903 |
| s | 5,457 | 1,846 | 1,059 | 1,776 |
| m | 5,908 | 2,509 | 1,146 | 575 |

## Detection coverage (front view, labelled students matched to a pose detection at confidence 0.5)

| Behavior | n | s | m |
|---|---|---|---|
| read | 41% (8,111/19,941) | 36% (7,080/19,457) | 41% (8,641/21,283) |
| write | 37% (2,778/7,530) | 32% (2,366/7,461) | 38% (3,455/9,029) |
| hand-raising | 39% (2,390/6,173) | 33% (2,011/6,005) | 42% (2,812/6,732) |
| TurnHead | 35% (2,750/7,929) | 30% (2,435/8,033) | 39% (3,295/8,421) |
| discuss | 26% (813/3,124) | 19% (579/2,974) | 22% (784/3,491) |
| BowHead | 17% (696/4,023) | 13% (467/3,661) | 20% (851/4,178) |

## Rules vs learned (front view, `learn.py`)

| Model | Students / groups | Rules AUC | Rules bal. acc | Learned AUC | Learned bal. acc |
|---|---|---|---|---|---|
| n | 17,095 / 265 | 0.538 | 53.0% | 0.671 | 62.0% |
| s | 14,339 / 265 | 0.527 | 53.2% | 0.715 | 66.0% |
| m | 19,262 / 265 | 0.545 | 53.9% | 0.737 | 66.7% |

## Full console output

### upper_body_n.hef

`evaluate.py ~/scb/poses_hef_n.jsonl --out ~/scb/report_n.json`

```
images by camera view: {'back': 2582, 'few': 903, 'front': 5555, 'mixed': 1098}

label -> matched to a pose detection (front view):
  BowHead          696/4023   (17%)
  TurnHead        2750/7929   (35%)
  discuss          813/3124   (26%)
  hand-raising    2390/6173   (39%)
  read            8111/19941  (41%)
  write           2778/7530   (37%)

=== front view: 17538 labelled students ===
  behavior           n  pred distracted  top cues
  read            7907            9.1%   face_hidden:296, head_turned:180, body_turned:119
  write           2717           11.9%   face_hidden:99, head_down:74, body_turned:57
  hand-raising    2332           12.3%   head_turned:131, face_hidden:109, body_turned:41
  TurnHead        2659           10.7%   face_hidden:157, head_turned:93, body_turned:25
  BowHead          686           13.7%   head_down:26, face_hidden:26, body_turned:18
  discuss          794           37.5%   face_hidden:231, body_turned:30, head_turned:28
  focused vs distracted @ on_threshold=0.6: recall(distracted) 16.3%, specificity(focused) 89.7%, precision 33.7%, F1 0.22, balanced acc 53.0%, AUC 0.621
  confusion [rows true focused/distracted, cols pred focused/distracted]: [[11627, 1329], [3463, 676]]

=== mixed view: 2013 labelled students ===
  behavior           n  pred distracted  top cues
  read             536           46.6%   face_hidden:223, body_turned:10, head_turned:10
  write            133           44.4%   face_hidden:44, body_turned:7, head_turned:6
  hand-raising     578           63.8%   face_hidden:352, head_turned:8, body_turned:8
  TurnHead         293           57.3%   face_hidden:154, body_turned:7, head_turned:7
  BowHead           37           78.4%   face_hidden:26, body_turned:3
  discuss          313           69.6%   face_hidden:202, body_turned:13, head_turned:2
  focused vs distracted @ on_threshold=0.6: recall(distracted) 64.5%, specificity(focused) 45.6%, precision 38.0%, F1 0.48, balanced acc 55.1%, AUC 0.582
  confusion [rows true focused/distracted, cols pred focused/distracted]: [[569, 678], [228, 415]]

=== back view: 4117 labelled students ===
  behavior           n  pred distracted  top cues
  read             618           87.2%   face_hidden:538, head_turned:1
  write            131           84.0%   face_hidden:106, body_turned:4
  hand-raising    1968           83.7%   face_hidden:1632, body_turned:13, head_turned:3
  TurnHead         773           94.0%   face_hidden:723, head_turned:3, body_turned:1
  BowHead          217           98.6%   face_hidden:214
  discuss           69           91.3%   face_hidden:63
  focused vs distracted @ on_threshold=0.6: recall(distracted) 94.8%, specificity(focused) 15.5%, precision 30.4%, F1 0.46, balanced acc 55.1%, AUC 0.682
  confusion [rows true focused/distracted, cols pred focused/distracted]: [[420, 2297], [55, 1004]]

1018 matched students had no usable keypoints (unknown)

front view, feature medians by behavior:
  read          yaw_dev  0.14  head_drop  0.02  face_hidden   4%  hand_raised   0%
  write         yaw_dev  0.13  head_drop  0.05  face_hidden   4%  hand_raised   0%
  hand-raising  yaw_dev  0.16  head_drop -0.01  face_hidden   5%  hand_raised   6%
  TurnHead      yaw_dev  0.15  head_drop  0.00  face_hidden   6%  hand_raised   0%
  BowHead       yaw_dev  0.16  head_drop  0.08  face_hidden   4%  hand_raised   0%
  discuss       yaw_dev  0.16  head_drop  0.00  face_hidden  29%  hand_raised   0%
```

`learn.py ~/scb/poses_hef_n.jsonl`

```
17095 students from 265 videos; distracted share 24.2%
rules   : AUC 0.538  balanced acc 53.0%
learned : AUC 0.671  balanced acc 62.0%  (5-fold, split by video)

per behavior, share predicted distracted (learned @0.5 | rules):
  read          n= 7907  29.6% |  9.1%
  write         n= 2717  27.1% | 11.9%
  hand-raising  n= 2332  22.7% | 12.3%
  TurnHead      n= 2659  58.5% | 10.7%
  BowHead       n=  686  13.3% | 13.7%
  discuss       n=  794  62.8% | 37.5%

AUC when one block is removed (5-fold):
  without keypoint geometry                        AUC 0.650
  without visibilities                             AUC 0.670
  without position in room                         AUC 0.649
  without rule features (yaw, head, face, hands)   AUC 0.677
```

### upper_body_s.hef

`evaluate.py ~/scb/poses_hef_s.jsonl --out ~/scb/report_s.json`

```
images by camera view: {'back': 1846, 'few': 1776, 'front': 5457, 'mixed': 1059}

label -> matched to a pose detection (front view):
  BowHead          467/3661   (13%)
  TurnHead        2435/8033   (30%)
  discuss          579/2974   (19%)
  hand-raising    2011/6005   (33%)
  read            7080/19457  (36%)
  write           2366/7461   (32%)

=== front view: 14938 labelled students ===
  behavior           n  pred distracted  top cues
  read            6792           10.1%   face_hidden:338, head_turned:145, body_turned:107
  write           2291           12.2%   face_hidden:118, body_turned:54, head_turned:43
  hand-raising    1913           10.9%   face_hidden:88, head_turned:76, body_turned:31
  TurnHead        2319           12.5%   face_hidden:176, head_turned:75, body_turned:27
  BowHead          456           17.8%   face_hidden:51, body_turned:10, head_turned:9
  discuss          568           35.0%   face_hidden:158, body_turned:23, head_turned:13
  focused vs distracted @ on_threshold=0.6: recall(distracted) 17.1%, specificity(focused) 89.4%, precision 32.7%, F1 0.22, balanced acc 53.2%, AUC 0.626
  confusion [rows true focused/distracted, cols pred focused/distracted]: [[9825, 1171], [2773, 570]]

=== mixed view: 1605 labelled students ===
  behavior           n  pred distracted  top cues
  read             383           47.8%   face_hidden:171, head_turned:6, body_turned:6
  write            119           44.5%   face_hidden:47, body_turned:2, hand_at_face:2
  hand-raising     492           61.8%   face_hidden:285, body_turned:14, head_turned:5
  TurnHead         275           57.8%   face_hidden:146, head_turned:11, body_turned:2
  BowHead           32           84.4%   face_hidden:26, head_turned:1
  discuss          235           77.4%   face_hidden:176, body_turned:3, head_turned:2
  focused vs distracted @ on_threshold=0.6: recall(distracted) 67.9%, specificity(focused) 45.7%, precision 40.5%, F1 0.51, balanced acc 56.8%, AUC 0.610
  confusion [rows true focused/distracted, cols pred focused/distracted]: [[454, 540], [174, 368]]

=== back view: 2557 labelled students ===
  behavior           n  pred distracted  top cues
  read             400           88.8%   face_hidden:353, head_turned:1, body_turned:1
  write             71           80.3%   face_hidden:56, body_turned:1
  hand-raising    1182           83.2%   face_hidden:957, body_turned:26
  TurnHead         536           93.7%   face_hidden:500, body_turned:2
  BowHead          113           97.3%   face_hidden:109, body_turned:1
  discuss           50           90.0%   face_hidden:45
  focused vs distracted @ on_threshold=0.6: recall(distracted) 94.0%, specificity(focused) 15.6%, precision 32.0%, F1 0.48, balanced acc 54.8%, AUC 0.691
  confusion [rows true focused/distracted, cols pred focused/distracted]: [[258, 1395], [42, 657]]

1019 matched students had no usable keypoints (unknown)

front view, feature medians by behavior:
  read          yaw_dev  0.13  head_drop  0.02  face_hidden   5%  hand_raised   0%
  write         yaw_dev  0.13  head_drop  0.04  face_hidden   5%  hand_raised   0%
  hand-raising  yaw_dev  0.14  head_drop -0.00  face_hidden   5%  hand_raised   5%
  TurnHead      yaw_dev  0.14  head_drop  0.00  face_hidden   8%  hand_raised   0%
  BowHead       yaw_dev  0.13  head_drop  0.07  face_hidden  11%  hand_raised   0%
  discuss       yaw_dev  0.14  head_drop  0.00  face_hidden  28%  hand_raised   0%
```

`learn.py ~/scb/poses_hef_s.jsonl`

```
14339 students from 265 videos; distracted share 23.3%
rules   : AUC 0.527  balanced acc 53.2%
learned : AUC 0.715  balanced acc 66.0%  (5-fold, split by video)

per behavior, share predicted distracted (learned @0.5 | rules):
  read          n= 6792  25.3% | 10.1%
  write         n= 2291  21.9% | 12.2%
  hand-raising  n= 1913  27.4% | 10.9%
  TurnHead      n= 2319  64.1% | 12.5%
  BowHead       n=  456  11.0% | 17.8%
  discuss       n=  568  64.6% | 35.0%

AUC when one block is removed (5-fold):
  without keypoint geometry                        AUC 0.678
  without visibilities                             AUC 0.709
  without position in room                         AUC 0.694
  without rule features (yaw, head, face, hands)   AUC 0.725
```

### upper_body_m.hef

`evaluate.py ~/scb/poses_hef_m.jsonl --out ~/scb/report_m.json`

```
images by camera view: {'back': 2509, 'few': 575, 'front': 5908, 'mixed': 1146}

label -> matched to a pose detection (front view):
  BowHead          851/4178   (20%)
  TurnHead        3295/8421   (39%)
  discuss          784/3491   (22%)
  hand-raising    2812/6732   (42%)
  read            8641/21283  (41%)
  write           3455/9029   (38%)

=== front view: 19838 labelled students ===
  behavior           n  pred distracted  top cues
  read            8401           11.5%   face_hidden:457, head_turned:275, body_turned:99
  write           3397           15.7%   face_hidden:236, body_turned:98, head_turned:94
  hand-raising    2671           14.8%   head_turned:178, face_hidden:157, body_turned:46
  TurnHead        3186           16.4%   face_hidden:240, head_turned:217, body_turned:44
  BowHead          834           18.6%   face_hidden:88, body_turned:21, head_down:17
  discuss          773           42.4%   face_hidden:251, body_turned:39, head_turned:29
  focused vs distracted @ on_threshold=0.6: recall(distracted) 21.0%, specificity(focused) 86.9%, precision 34.7%, F1 0.26, balanced acc 53.9%, AUC 0.641
  confusion [rows true focused/distracted, cols pred focused/distracted]: [[12578, 1891], [3788, 1005]]

=== mixed view: 2218 labelled students ===
  behavior           n  pred distracted  top cues
  read             540           46.1%   face_hidden:221, head_turned:16, body_turned:7
  write            182           44.5%   face_hidden:66, head_turned:7, body_turned:4
  hand-raising     668           66.6%   face_hidden:422, head_turned:12, body_turned:8
  TurnHead         322           50.3%   face_hidden:143, head_turned:12, body_turned:7
  BowHead           55           72.7%   face_hidden:29, head_turned:4, body_turned:4
  discuss          286           83.2%   face_hidden:220, body_turned:12, head_turned:5
  focused vs distracted @ on_threshold=0.6: recall(distracted) 66.4%, specificity(focused) 44.2%, precision 36.2%, F1 0.47, balanced acc 55.3%, AUC 0.575
  confusion [rows true focused/distracted, cols pred focused/distracted]: [[615, 775], [223, 440]]

=== back view: 4322 labelled students ===
  behavior           n  pred distracted  top cues
  read             595           87.4%   face_hidden:513, body_turned:4, hand_at_face:2
  write            130           85.4%   face_hidden:108, body_turned:3
  hand-raising    2227           85.8%   face_hidden:1894, body_turned:13, head_turned:3
  TurnHead         709           85.3%   face_hidden:598, head_turned:6, body_turned:1
  BowHead          175           98.9%   face_hidden:173
  discuss           52           86.5%   face_hidden:43, body_turned:2
  focused vs distracted @ on_threshold=0.6: recall(distracted) 87.9%, specificity(focused) 13.9%, precision 24.5%, F1 0.38, balanced acc 50.9%, AUC 0.652
  confusion [rows true focused/distracted, cols pred focused/distracted]: [[411, 2541], [113, 823]]

1307 matched students had no usable keypoints (unknown)

front view, feature medians by behavior:
  read          yaw_dev  0.14  head_drop  0.03  face_hidden   5%  hand_raised   0%
  write         yaw_dev  0.13  head_drop  0.05  face_hidden   7%  hand_raised   0%
  hand-raising  yaw_dev  0.17  head_drop -0.02  face_hidden   6%  hand_raised  12%
  TurnHead      yaw_dev  0.18  head_drop -0.01  face_hidden   8%  hand_raised   0%
  BowHead       yaw_dev  0.15  head_drop  0.10  face_hidden  11%  hand_raised   0%
  discuss       yaw_dev  0.19  head_drop  0.00  face_hidden  32%  hand_raised   0%
```

`learn.py ~/scb/poses_hef_m.jsonl`

```
19262 students from 265 videos; distracted share 24.9%
rules   : AUC 0.545  balanced acc 53.9%
learned : AUC 0.737  balanced acc 66.7%  (5-fold, split by video)

per behavior, share predicted distracted (learned @0.5 | rules):
  read          n= 8401  19.9% | 11.5%
  write         n= 3397  26.8% | 15.7%
  hand-raising  n= 2671  24.4% | 14.8%
  TurnHead      n= 3186  65.6% | 16.4%
  BowHead       n=  834  11.8% | 18.6%
  discuss       n=  773  62.9% | 42.4%

AUC when one block is removed (5-fold):
  without keypoint geometry                        AUC 0.704
  without visibilities                             AUC 0.737
  without position in room                         AUC 0.736
  without rule features (yaw, head, face, hands)   AUC 0.741
```

## Caveats

These come from checks outside the repo scripts, run on the same data.

1. **Duplicate images with contradictory labels.** 1,891 images are byte-identical copies in two or three subsets (BowTurnHead & Handrise 1,179, Discuss & Handrise 424, BowTurnHead & Discuss 158, all three 130). Each subset only annotates its own behaviors, so the same student gets opposite labels in different copies: across subsets, 4,540 same-student box pairs (IoU > 0.7) are focused in one copy and distracted in the other, and only 10 agree. These contaminate both training and scoring.
2. **The group split is not by lesson.** `video_id()` puts the subset name in the group key, so one lesson that appears in two subsets counts as two groups (265 groups; 177–179 actual lessons). Grouping by lesson ID alone (the same `learn.py` with `video_id` returning `p:<N>` / `c:<first 4 chars>` and no subset prefix) gave, with everything else unchanged:

| Model | Groups | Rules AUC | Learned AUC | Learned bal. acc |
|---|---|---|---|---|
| n | 177 | 0.538 | 0.756 | 68.0% |
| s | 177 | 0.527 | 0.775 | 69.5% |
| m | 179 | 0.545 | 0.814 | 73.7% |

   The learned AUC moves by up to about 0.08 depending only on how the folds are split, and the stricter split scores *higher*. That fits contradictory duplicates no longer ending up on both sides of a fold, but this isn't confirmed. Until the duplicates are deduplicated and their labels merged or dropped, trust the ordering (m > s > n, learned well above rules) more than the absolute values.
3. **Unconfirmed class order.** The SCB label files don't name their classes, and the downloaded archives contain no class-name file. The order passed to `extract_hef.py` (BowHead,TurnHead / discuss / hand-raising,read,write) is assumed. If it's wrong for BowTurnHead, the BowHead and TurnHead rows are swapped.

---

# Rerun with duplicate merging and lesson groups (`6d0e515`)

Same pose caches as above. After pulling `scb-eval` (`scbdata.py`: identical images merged, conflicting students dropped, union-find lesson groups; class order now taken from the dataset's YAML on the HF mirror), I ran `learn.py ~/scb/poses_hef_<m>.jsonl` and `evaluate.py ~/scb/poses_hef_<m>.jsonl --det-thr 0.3`. To separate the threshold effect from the dedup effect, I also ran `evaluate.py` at the default 0.5 for the coverage comparison.

## `[scbdata]` line (identical for n, s and m, and for both scripts)

```
[scbdata] 10138 records -> 8116 unique images (1892 duplicated across subsets); 2891 students with both focused and distracted labels dropped; 189 lesson groups
```

## Rules vs learned (front view, split by lesson)

| Model | Students / lesson groups | Distracted share | Rules AUC | Rules bal. acc | Learned AUC | Learned bal. acc |
|---|---|---|---|---|---|---|
| n | 15,303 / 177 | 20.2% | 0.537 | 53.5% | 0.813 | 71.8% |
| s | 13,026 / 177 | 19.8% | 0.518 | 53.4% | 0.854 | 75.3% |
| m | 17,048 / 179 | 20.8% | 0.538 | 54.8% | 0.885 | 78.9% |

## Shortcut checks

| Model | Students in images with both classes | Context-only AUC (no pose) | Within-image AUC, learned | Within-image AUC, rules |
|---|---|---|---|---|
| n | 3,616 (24%) | 0.644 | 0.705 | 0.511 |
| s | 2,585 (20%) | 0.643 | 0.757 | 0.480 |
| m | 4,044 (24%) | 0.661 | 0.827 | 0.515 |

Verbatim:

```
[n] shortcut checks: 3616 students (24%) are in images that contain both classes
[n]   context only (where in the image, box size, crowd size; no pose): AUC 0.644
[n]   within-image AUC (only images with both classes): learned 0.705, rules 0.511
[s] shortcut checks: 2585 students (20%) are in images that contain both classes
[s]   context only (where in the image, box size, crowd size; no pose): AUC 0.643
[s]   within-image AUC (only images with both classes): learned 0.757, rules 0.480
[m] shortcut checks: 4044 students (24%) are in images that contain both classes
[m]   context only (where in the image, box size, crowd size; no pose): AUC 0.661
[m]   within-image AUC (only images with both classes): learned 0.827, rules 0.515
```

## Detection coverage, front view, `--det-thr 0.3` (with dedup)

| Behavior | n | s | m |
|---|---|---|---|
| read | 60% (11,537/19,098) | 55% (10,272/18,671) | 60% (11,509/19,175) |
| write | 51% (3,706/7,247) | 51% (3,371/6,630) | 58% (4,163/7,218) |
| hand-raising | 57% (3,571/6,267) | 51% (3,197/6,228) | 59% (3,688/6,219) |
| TurnHead | 59% (4,222/7,195) | 52% (3,708/7,155) | 60% (4,400/7,345) |
| discuss | 34% (980/2,845) | 28% (699/2,498) | 34% (971/2,884) |
| BowHead | 19% (263/1,411) | 16% (208/1,331) | 21% (301/1,411) |

For comparison: the same with dedup at the default `--det-thr 0.5`, then 0.3:

| Behavior | n @0.5 → 0.3 | s @0.5 → 0.3 | m @0.5 → 0.3 |
|---|---|---|---|
| read | 43% → 60% | 39% → 55% | 43% → 60% |
| write | 42% → 51% | 36% → 51% | 42% → 58% |
| hand-raising | 40% → 57% | 35% → 51% | 43% → 59% |
| TurnHead | 35% → 59% | 31% → 52% | 39% → 60% |
| discuss | 21% → 34% | 16% → 28% | 19% → 34% |
| BowHead | 10% → 19% | 7% → 16% | 11% → 21% |

View split with dedup (images; the split itself depends on `--det-thr`):

| Model | thr | front | back | mixed | too few |
|---|---|---|---|---|---|
| n | 0.5 | 4,130 | 2,337 | 843 | 806 |
| n | 0.3 | 4,338 | 2,558 | 902 | 318 |
| s | 0.5 | 4,053 | 1,665 | 856 | 1,542 |
| s | 0.3 | 4,315 | 2,211 | 977 | 613 |
| m | 0.5 | 4,387 | 2,293 | 914 | 522 |
| m | 0.3 | 4,351 | 2,524 | 946 | 295 |

The denominators differ between thresholds and from the first run, for two reasons. Dedup merges the duplicate copies and drops the 2,891 conflicting students (BowHead falls from about 4,000 to 1,400 front-view labels). And the front/back/mixed classification is computed from the detections that pass `--det-thr`.

## Notes

- **Dropping conflicts removed most BowHead students.** `learn.py` now has 128 / 83 / 146 BowHead students (n / s / m), against 686 / 456 / 834 in the first run, because a large share of the dropped conflicting students carry a BowHead label (front-view BowHead labels fall from about 4,000 to 1,400; the conflict pairs weren't broken down by label here). BowHead was the class the classifier couldn't separate, so part of the jump in learned AUC (0.67–0.74 → 0.81–0.89) is the hardest class dropping out, not only cleaner labels. The "distracted" class is now mostly TurnHead and discuss.
- **The image is still a shortcut.** Without any pose information, the context-only model reaches AUC 0.64–0.66. Only 20–24% of students are in images that contain both classes. Within those images the learned model still ranks distracted above focused (0.71 / 0.76 / 0.83), while the rules are at chance (0.48–0.52). The within-image AUC is the better measure of what pose alone contributes.
- **Ordering is unchanged:** m > s > n on every learned metric.
- **Class order:** `scbdata.py` says the per-subset class order comes from the dataset's YAML files on the HF mirror. That resolves caveat 3 above if correct, but I haven't checked it on the Pi (the downloaded zips contain no YAML).
- **Coverage:** going from 0.5 to 0.3 raises coverage by about 15–20 points for read, write, hand-raising and TurnHead, and by about 8–15 for BowHead and discuss. BowHead stays lowest (16–21%).

## Full console output

### upper_body_n.hef

`learn.py ~/scb/poses_hef_n.jsonl`

```
[scbdata] 10138 records -> 8116 unique images (1892 duplicated across subsets); 2891 students with both focused and distracted labels dropped; 189 lesson groups
15303 students from 177 lesson groups; distracted share 20.2%
rules   : AUC 0.537  balanced acc 53.5%
learned : AUC 0.813  balanced acc 71.8%  (5-fold, split by lesson)

per behavior, share predicted distracted (learned @0.5 | rules):
  read          n= 7558  20.7% |  9.1%
  write         n= 2442   9.5% | 11.5%
  hand-raising  n= 2218  18.6% | 12.3%
  TurnHead      n= 2365  62.9% | 11.0%
  BowHead       n=  128  32.0% | 19.5%
  discuss       n=  592  62.8% | 41.2%

AUC when one block is removed (5-fold):
  without keypoint geometry                        AUC 0.759
  without visibilities                             AUC 0.816
  without position in room                         AUC 0.777
  without rule features (yaw, head, face, hands)   AUC 0.805

shortcut checks: 3616 students (24%) are in images that contain both classes
  context only (where in the image, box size, crowd size; no pose): AUC 0.644
  within-image AUC (only images with both classes): learned 0.705, rules 0.511
```

`evaluate.py ~/scb/poses_hef_n.jsonl --det-thr 0.3 --out ~/scb/report_n_thr03.json`

```
[scbdata] 10138 records -> 8116 unique images (1892 duplicated across subsets); 2891 students with both focused and distracted labels dropped; 189 lesson groups
images by camera view: {'back': 2558, 'front': 4338, 'mixed': 902, 'few': 318}

label -> matched to a pose detection (front view):
  BowHead          263/1411   (19%)
  TurnHead        4222/7195   (59%)
  discuss          980/2845   (34%)
  hand-raising    3571/6267   (57%)
  read           11537/19098  (60%)
  write           3706/7247   (51%)

=== front view: 24279 labelled students ===
  behavior           n  pred distracted  top cues
  read           10469           10.3%   face_hidden:514, head_turned:263, body_turned:163
  write           3433           14.9%   face_hidden:209, head_down:93, body_turned:78
  hand-raising    3224           13.3%   head_turned:183, face_hidden:171, body_turned:68
  TurnHead        3634           12.7%   face_hidden:282, head_turned:130, body_turned:41
  BowHead          217           24.9%   face_hidden:27, head_down:8, head_turned:7
  discuss          892           37.9%   face_hidden:271, body_turned:34, head_turned:24
  focused vs distracted @ on_threshold=0.6: recall(distracted) 18.0%, specificity(focused) 88.2%, precision 29.7%, F1 0.22, balanced acc 53.1%, AUC 0.545
  confusion [rows true focused/distracted, cols pred focused/distracted]: [[15107, 2019], [3890, 853]]

=== mixed view: 3129 labelled students ===
  behavior           n  pred distracted  top cues
  read             830           45.3%   face_hidden:332, body_turned:21, head_turned:17
  write            182           34.1%   face_hidden:44, body_turned:8, head_turned:6
  hand-raising     729           62.1%   face_hidden:415, head_turned:28, body_turned:9
  TurnHead         434           54.4%   face_hidden:206, head_turned:19, body_turned:10
  BowHead           32           75.0%   face_hidden:23, body_turned:1
  discuss          488           77.7%   face_hidden:341, body_turned:28, head_turned:6
  focused vs distracted @ on_threshold=0.6: recall(distracted) 67.0%, specificity(focused) 48.8%, precision 41.8%, F1 0.51, balanced acc 57.9%, AUC 0.617
  confusion [rows true focused/distracted, cols pred focused/distracted]: [[850, 891], [315, 639]]

=== back view: 6122 labelled students ===
  behavior           n  pred distracted  top cues
  read             842           84.4%   face_hidden:708, head_turned:3
  write            161           83.9%   face_hidden:128, body_turned:5, head_turned:2
  hand-raising    2700           84.6%   face_hidden:2263, body_turned:14, head_turned:5
  TurnHead        1073           93.4%   face_hidden:998, head_turned:2, body_turned:2
  BowHead          289           99.0%   face_hidden:286
  discuss           91           90.1%   face_hidden:82
  focused vs distracted @ on_threshold=0.6: recall(distracted) 94.3%, specificity(focused) 15.5%, precision 30.5%, F1 0.46, balanced acc 54.9%, AUC 0.711
  confusion [rows true focused/distracted, cols pred focused/distracted]: [[574, 3129], [83, 1370]]

3902 matched students had no usable keypoints (unknown)

front view, feature medians by behavior:
  read          yaw_dev  0.14  head_drop  0.02  face_hidden   5%  hand_raised   0%
  write         yaw_dev  0.13  head_drop  0.05  face_hidden   6%  hand_raised   0%
  hand-raising  yaw_dev  0.17  head_drop -0.01  face_hidden   5%  hand_raised   6%
  TurnHead      yaw_dev  0.16  head_drop  0.00  face_hidden   8%  hand_raised   0%
  BowHead       yaw_dev  0.16  head_drop  0.05  face_hidden  12%  hand_raised   0%
  discuss       yaw_dev  0.17  head_drop  0.03  face_hidden  30%  hand_raised   0%
```

`evaluate.py ~/scb/poses_hef_n.jsonl` (default `--det-thr 0.5`, first 10 lines: view split and coverage)

```
[scbdata] 10138 records -> 8116 unique images (1892 duplicated across subsets); 2891 students with both focused and distracted labels dropped; 189 lesson groups
images by camera view: {'back': 2337, 'few': 806, 'front': 4130, 'mixed': 843}

label -> matched to a pose detection (front view):
  BowHead          134/1407   (10%)
  TurnHead        2452/7005   (35%)
  discuss          607/2910   (21%)
  hand-raising    2274/5667   (40%)
  read            7757/17993  (43%)
  write           2502/5941   (42%)
```

### upper_body_s.hef

`learn.py ~/scb/poses_hef_s.jsonl`

```
[scbdata] 10138 records -> 8116 unique images (1892 duplicated across subsets); 2891 students with both focused and distracted labels dropped; 189 lesson groups
13026 students from 177 lesson groups; distracted share 19.8%
rules   : AUC 0.518  balanced acc 53.4%
learned : AUC 0.854  balanced acc 75.3%  (5-fold, split by lesson)

per behavior, share predicted distracted (learned @0.5 | rules):
  read          n= 6522  15.1% |  9.9%
  write         n= 2126   8.7% | 11.6%
  hand-raising  n= 1800  18.3% | 10.8%
  TurnHead      n= 2060  67.5% | 12.8%
  BowHead       n=   83  18.1% | 19.3%
  discuss       n=  435  62.3% | 37.0%

AUC when one block is removed (5-fold):
  without keypoint geometry                        AUC 0.783
  without visibilities                             AUC 0.855
  without position in room                         AUC 0.824
  without rule features (yaw, head, face, hands)   AUC 0.845

shortcut checks: 2585 students (20%) are in images that contain both classes
  context only (where in the image, box size, crowd size; no pose): AUC 0.643
  within-image AUC (only images with both classes): learned 0.757, rules 0.480
```

`evaluate.py ~/scb/poses_hef_s.jsonl --det-thr 0.3 --out ~/scb/report_s_thr03.json`

```
[scbdata] 10138 records -> 8116 unique images (1892 duplicated across subsets); 2891 students with both focused and distracted labels dropped; 189 lesson groups
images by camera view: {'back': 2211, 'front': 4315, 'few': 613, 'mixed': 977}

label -> matched to a pose detection (front view):
  BowHead          208/1331   (16%)
  TurnHead        3708/7155   (52%)
  discuss          699/2498   (28%)
  hand-raising    3197/6228   (51%)
  read           10272/18671  (55%)
  write           3371/6630   (51%)

=== front view: 21455 labelled students ===
  behavior           n  pred distracted  top cues
  read            9140           11.8%   face_hidden:584, head_turned:199, body_turned:177
  write           3054           14.8%   face_hidden:215, body_turned:82, head_turned:74
  hand-raising    2792           12.3%   face_hidden:141, head_turned:137, body_turned:51
  TurnHead        3207           15.3%   face_hidden:317, head_turned:131, body_turned:33
  BowHead          164           23.2%   face_hidden:27, head_turned:4, body_turned:4
  discuss          648           36.3%   face_hidden:189, body_turned:27, head_turned:13
  focused vs distracted @ on_threshold=0.6: recall(distracted) 19.0%, specificity(focused) 87.5%, precision 29.0%, F1 0.23, balanced acc 53.3%, AUC 0.529
  confusion [rows true focused/distracted, cols pred focused/distracted]: [[13109, 1877], [3254, 765]]

=== mixed view: 3060 labelled students ===
  behavior           n  pred distracted  top cues
  read             778           46.5%   face_hidden:320, head_turned:18, body_turned:18
  write            321           38.9%   face_hidden:105, head_turned:12, body_turned:5
  hand-raising     644           59.6%   face_hidden:344, body_turned:17, head_turned:16
  TurnHead         453           51.7%   face_hidden:206, head_turned:20, body_turned:8
  BowHead           29           75.9%   face_hidden:20, head_down:1, head_turned:1
  discuss          481           81.1%   face_hidden:379, body_turned:6, head_turned:5
  focused vs distracted @ on_threshold=0.6: recall(distracted) 67.1%, specificity(focused) 50.0%, precision 42.6%, F1 0.52, balanced acc 58.6%, AUC 0.625
  confusion [rows true focused/distracted, cols pred focused/distracted]: [[872, 871], [317, 646]]

=== back view: 4826 labelled students ===
  behavior           n  pred distracted  top cues
  read             695           87.6%   face_hidden:606, head_turned:2, body_turned:1
  write            134           83.6%   face_hidden:111, body_turned:1
  hand-raising    2011           85.3%   face_hidden:1678, body_turned:34, head_turned:3
  TurnHead         902           90.8%   face_hidden:811, head_turned:4, body_turned:3
  BowHead          197           98.5%   face_hidden:194
  discuss          100           87.0%   face_hidden:87
  focused vs distracted @ on_threshold=0.6: recall(distracted) 91.7%, specificity(focused) 14.2%, precision 31.1%, F1 0.46, balanced acc 53.0%, AUC 0.694
  confusion [rows true focused/distracted, cols pred focused/distracted]: [[403, 2437], [99, 1100]]

3686 matched students had no usable keypoints (unknown)

front view, feature medians by behavior:
  read          yaw_dev  0.14  head_drop  0.03  face_hidden   6%  hand_raised   0%
  write         yaw_dev  0.14  head_drop  0.04  face_hidden   7%  hand_raised   0%
  hand-raising  yaw_dev  0.15  head_drop -0.01  face_hidden   5%  hand_raised   5%
  TurnHead      yaw_dev  0.16  head_drop  0.00  face_hidden  10%  hand_raised   0%
  BowHead       yaw_dev  0.17  head_drop  0.01  face_hidden  16%  hand_raised   0%
  discuss       yaw_dev  0.15  head_drop  0.00  face_hidden  29%  hand_raised   0%
```

`evaluate.py ~/scb/poses_hef_s.jsonl` (default `--det-thr 0.5`, first 10 lines: view split and coverage)

```
[scbdata] 10138 records -> 8116 unique images (1892 duplicated across subsets); 2891 students with both focused and distracted labels dropped; 189 lesson groups
images by camera view: {'back': 1665, 'few': 1542, 'front': 4053, 'mixed': 856}

label -> matched to a pose detection (front view):
  BowHead           87/1323   (7%)
  TurnHead        2167/7078   (31%)
  discuss          443/2789   (16%)
  hand-raising    1891/5458   (35%)
  read            6805/17588  (39%)
  write           2196/6164   (36%)
```

### upper_body_m.hef

`learn.py ~/scb/poses_hef_m.jsonl`

```
[scbdata] 10138 records -> 8116 unique images (1892 duplicated across subsets); 2891 students with both focused and distracted labels dropped; 189 lesson groups
17048 students from 179 lesson groups; distracted share 20.8%
rules   : AUC 0.538  balanced acc 54.8%
learned : AUC 0.885  balanced acc 78.9%  (5-fold, split by lesson)

per behavior, share predicted distracted (learned @0.5 | rules):
  read          n= 8003  12.4% | 11.3%
  write         n= 3015   5.4% | 15.3%
  hand-raising  n= 2484  16.3% | 14.1%
  TurnHead      n= 2789  72.8% | 16.5%
  BowHead       n=  146  26.0% | 35.6%
  discuss       n=  611  63.3% | 45.7%

AUC when one block is removed (5-fold):
  without keypoint geometry                        AUC 0.826
  without visibilities                             AUC 0.883
  without position in room                         AUC 0.869
  without rule features (yaw, head, face, hands)   AUC 0.882

shortcut checks: 4044 students (24%) are in images that contain both classes
  context only (where in the image, box size, crowd size; no pose): AUC 0.661
  within-image AUC (only images with both classes): learned 0.827, rules 0.515
```

`evaluate.py ~/scb/poses_hef_m.jsonl --det-thr 0.3 --out ~/scb/report_m_thr03.json`

```
[scbdata] 10138 records -> 8116 unique images (1892 duplicated across subsets); 2891 students with both focused and distracted labels dropped; 189 lesson groups
images by camera view: {'back': 2524, 'front': 4351, 'few': 295, 'mixed': 946}

label -> matched to a pose detection (front view):
  BowHead          301/1411   (21%)
  TurnHead        4400/7345   (60%)
  discuss          971/2884   (34%)
  hand-raising    3688/6219   (59%)
  read           11509/19175  (60%)
  write           4163/7218   (58%)

=== front view: 25032 labelled students ===
  behavior           n  pred distracted  top cues
  read           10340           13.1%   face_hidden:653, head_turned:351, body_turned:191
  write           3810           19.5%   face_hidden:319, body_turned:161, head_turned:116
  hand-raising    3198           14.1%   head_turned:191, face_hidden:180, body_turned:59
  TurnHead        3887           17.6%   face_hidden:354, head_turned:257, body_turned:60
  BowHead          241           34.0%   face_hidden:50, body_turned:13, head_turned:9
  discuss          885           44.2%   face_hidden:308, body_turned:47, head_turned:30
  focused vs distracted @ on_threshold=0.6: recall(distracted) 23.1%, specificity(focused) 85.3%, precision 31.3%, F1 0.27, balanced acc 54.2%, AUC 0.576
  confusion [rows true focused/distracted, cols pred focused/distracted]: [[14803, 2545], [3855, 1158]]

=== mixed view: 3234 labelled students ===
  behavior           n  pred distracted  top cues
  read             832           45.9%   face_hidden:334, head_turned:30, body_turned:14
  write            293           39.2%   face_hidden:93, body_turned:10, head_turned:9
  hand-raising     749           65.3%   face_hidden:453, head_turned:21, body_turned:11
  TurnHead         431           50.8%   face_hidden:181, head_turned:31, body_turned:7
  BowHead           30           86.7%   face_hidden:20, head_down:3, body_turned:1
  discuss          435           80.5%   face_hidden:322, body_turned:20, head_turned:5
  focused vs distracted @ on_threshold=0.6: recall(distracted) 66.4%, specificity(focused) 47.4%, precision 37.6%, F1 0.48, balanced acc 56.9%, AUC 0.608
  confusion [rows true focused/distracted, cols pred focused/distracted]: [[888, 986], [301, 595]]

=== back view: 6061 labelled students ===
  behavior           n  pred distracted  top cues
  read             809           87.5%   face_hidden:698, body_turned:5, head_turned:3
  write            152           86.8%   face_hidden:127, body_turned:3, head_turned:2
  hand-raising    2778           85.8%   face_hidden:2357, body_turned:21, head_turned:5
  TurnHead        1007           84.4%   face_hidden:835, head_turned:15
  BowHead          283           99.6%   face_hidden:282
  discuss           66           84.8%   face_hidden:53, body_turned:3
  focused vs distracted @ on_threshold=0.6: recall(distracted) 87.6%, specificity(focused) 13.8%, precision 26.9%, F1 0.41, balanced acc 50.7%, AUC 0.637
  confusion [rows true focused/distracted, cols pred focused/distracted]: [[515, 3224], [168, 1188]]

4237 matched students had no usable keypoints (unknown)

front view, feature medians by behavior:
  read          yaw_dev  0.14  head_drop  0.03  face_hidden   6%  hand_raised   0%
  write         yaw_dev  0.14  head_drop  0.05  face_hidden   8%  hand_raised   0%
  hand-raising  yaw_dev  0.17  head_drop -0.01  face_hidden   6%  hand_raised  14%
  TurnHead      yaw_dev  0.19  head_drop -0.00  face_hidden   9%  hand_raised   0%
  BowHead       yaw_dev  0.17  head_drop  0.08  face_hidden  21%  hand_raised   0%
  discuss       yaw_dev  0.21  head_drop  0.01  face_hidden  35%  hand_raised   0%
```

`evaluate.py ~/scb/poses_hef_m.jsonl` (default `--det-thr 0.5`, first 10 lines: view split and coverage)

```
[scbdata] 10138 records -> 8116 unique images (1892 duplicated across subsets); 2891 students with both focused and distracted labels dropped; 189 lesson groups
images by camera view: {'back': 2293, 'few': 522, 'front': 4387, 'mixed': 914}

label -> matched to a pose detection (front view):
  BowHead          152/1423   (11%)
  TurnHead        2885/7387   (39%)
  discuss          621/3279   (19%)
  hand-raising    2614/6102   (43%)
  read            8235/19248  (43%)
  write           3068/7333   (42%)
```

