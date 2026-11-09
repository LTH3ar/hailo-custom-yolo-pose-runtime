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
