# Human labels vs the INT8 models

`research/labels/student_labels_v1.json`: 301 blind labels from clip_3 and clip; 275 focused/distracted, 13 of them distracted. Scored against the Pi INT8 analysis files from the 2026-11-10 run (`~/clips_results/<clip>_<model>_<tiles>_<classifier>.jsonl`, camera position center for clip_3, auto for clip).

## Pooled over both clips

From `research/labels/pooled_summary.py`. "Not matched" counts items with no INT8 detection at IoU ≥ 0.3 near the labelled time, or matched to state `unknown`; they count as not flagged.

| Model | Tiles | Decision | Not matched (of 275) | False alarms on focused | Specificity (95% CI) | Distracted caught | False alarms, matched focused only |
|---|---|---|---|---|---|---|---|
| n | off | Rules | 91 | 6 / 262 | 98% (95-99%) | 1 / 13 | 6 / 173 (3%) |
| n | off | Trees | 91 | 103 / 262 | 61% (55-66%) | 2 / 13 | 103 / 173 (60%) |
| n | off | MLP | 91 | 100 / 262 | 62% (56-68%) | 4 / 13 | 100 / 173 (58%) |
| n | off | Logistic | 91 | 128 / 262 | 51% (45-57%) | 4 / 13 | 128 / 173 (74%) |
| n | on | Rules | 48 | 10 / 262 | 96% (93-98%) | 0 / 13 | 10 / 215 (5%) |
| n | on | Trees | 48 | 98 / 262 | 63% (57-68%) | 0 / 13 | 98 / 215 (46%) |
| n | on | MLP | 48 | 99 / 262 | 62% (56-68%) | 3 / 13 | 99 / 215 (46%) |
| n | on | Logistic | 48 | 146 / 262 | 44% (38-50%) | 3 / 13 | 146 / 215 (68%) |
| m | off | Rules | 62 | 12 / 262 | 95% (92-97%) | 0 / 13 | 12 / 201 (6%) |
| m | off | Trees | 62 | 99 / 262 | 62% (56-68%) | 3 / 13 | 99 / 201 (49%) |
| m | off | MLP | 62 | 85 / 262 | 68% (62-73%) | 4 / 13 | 85 / 201 (42%) |
| m | off | Logistic | 62 | 146 / 262 | 44% (38-50%) | 3 / 13 | 146 / 201 (73%) |
| m | on | Rules | 17 | 16 / 262 | 94% (90-96%) | 0 / 13 | 16 / 245 (7%) |
| m | on | Trees | 17 | 89 / 262 | 66% (60-71%) | 4 / 13 | 89 / 245 (36%) |
| m | on | MLP | 17 | 77 / 262 | 71% (65-76%) | 3 / 13 | 77 / 245 (31%) |
| m | on | Logistic | 17 | 152 / 262 | 42% (36-48%) | 3 / 13 | 152 / 245 (62%) |

Earlier FP32 reference (best.pt, n, tiles; research/labels/README.md):

| Decision | False alarms on focused | Distracted caught |
|---|---|---|
| Rules | 27 / 262 | 5 / 13 |
| Trees | 114 / 262 | 3 / 13 |
| MLP | 120 / 262 | 2 / 13 |
| Logistic | 199 / 262 | 5 / 13 |

## Per clip (score_labels.py)

### model n, tiles off

301 labelled, 275 focused/distracted, 26 can't tell; distracted share as labelled 5%, re-weighted 4.2%

| File | Items | Not matched | Precision | Recall | Specificity | Balanced acc. | Flagged share | Precision (re-weighted) | Recall (re-weighted) | Flagged share (re-weighted) |
|---|---|---|---|---|---|---|---|---|---|---|
| clip_3_n_none_none.jsonl | 182 | 58 | 0% | 0% | 99% | 50% | 1% | 0% | 0% | 0% |
| clip_n_none_none.jsonl | 93 | 33 | 17% | 11% | 94% | 53% | 6% | 10% | 4% | 3% |
| clip_3_n_none_trees.jsonl | 182 | 58 | 0% | 0% | 54% | 27% | 45% | 0% | 0% | 48% |
| clip_n_none_trees.jsonl | 93 | 33 | 8% | 22% | 74% | 48% | 26% | 9% | 32% | 24% |
| clip_3_n_none_logistic.jsonl | 182 | 58 | 2% | 50% | 41% | 46% | 59% | 2% | 50% | 61% |
| clip_n_none_logistic.jsonl | 93 | 33 | 8% | 22% | 73% | 47% | 27% | 8% | 32% | 26% |
| clip_3_n_none_mlp.jsonl | 182 | 58 | 1% | 25% | 57% | 41% | 43% | 1% | 28% | 47% |
| clip_n_none_mlp.jsonl | 93 | 33 | 12% | 33% | 73% | 53% | 28% | 9% | 33% | 22% |

### model n, tiles on

301 labelled, 275 focused/distracted, 26 can't tell; distracted share as labelled 5%, re-weighted 4.2%

| File | Items | Not matched | Precision | Recall | Specificity | Balanced acc. | Flagged share | Precision (re-weighted) | Recall (re-weighted) | Flagged share (re-weighted) |
|---|---|---|---|---|---|---|---|---|---|---|
| clip_3_n_auto_none.jsonl | 182 | 36 | 0% | 0% | 99% | 49% | 1% | 0% | 0% | 0% |
| clip_n_auto_none.jsonl | 93 | 12 | 0% | 0% | 90% | 45% | 9% | 0% | 0% | 1% |
| clip_3_n_auto_trees.jsonl | 182 | 36 | 0% | 0% | 51% | 26% | 48% | 0% | 0% | 52% |
| clip_n_auto_trees.jsonl | 93 | 12 | 0% | 0% | 87% | 43% | 12% | 0% | 0% | 2% |
| clip_3_n_auto_logistic.jsonl | 182 | 36 | 2% | 50% | 31% | 40% | 69% | 2% | 50% | 71% |
| clip_n_auto_logistic.jsonl | 93 | 12 | 4% | 11% | 73% | 42% | 26% | 1% | 2% | 17% |
| clip_3_n_auto_mlp.jsonl | 182 | 36 | 1% | 25% | 53% | 39% | 46% | 1% | 28% | 51% |
| clip_n_auto_mlp.jsonl | 93 | 12 | 11% | 22% | 81% | 52% | 19% | 4% | 5% | 8% |

### model m, tiles off

301 labelled, 275 focused/distracted, 26 can't tell; distracted share as labelled 5%, re-weighted 4.2%

| File | Items | Not matched | Precision | Recall | Specificity | Balanced acc. | Flagged share | Precision (re-weighted) | Recall (re-weighted) | Flagged share (re-weighted) |
|---|---|---|---|---|---|---|---|---|---|---|
| clip_3_m_none_none.jsonl | 182 | 25 | – | 0% | 100% | 50% | 0% | – | 0% | 0% |
| clip_m_none_none.jsonl | 93 | 37 | 0% | 0% | 86% | 43% | 13% | 0% | 0% | 10% |
| clip_3_m_none_trees.jsonl | 182 | 25 | 2% | 50% | 51% | 50% | 49% | 2% | 56% | 52% |
| clip_m_none_trees.jsonl | 93 | 37 | 8% | 11% | 87% | 49% | 13% | 2% | 2% | 7% |
| clip_3_m_none_logistic.jsonl | 182 | 25 | 2% | 75% | 25% | 50% | 75% | 2% | 78% | 77% |
| clip_m_none_logistic.jsonl | 93 | 37 | 0% | 0% | 85% | 42% | 14% | 0% | 0% | 9% |
| clip_3_m_none_mlp.jsonl | 182 | 25 | 2% | 50% | 56% | 53% | 45% | 3% | 56% | 47% |
| clip_m_none_mlp.jsonl | 93 | 37 | 25% | 22% | 93% | 58% | 9% | 4% | 5% | 8% |

### model m, tiles on

301 labelled, 275 focused/distracted, 26 can't tell; distracted share as labelled 5%, re-weighted 4.2%

| File | Items | Not matched | Precision | Recall | Specificity | Balanced acc. | Flagged share | Precision (re-weighted) | Recall (re-weighted) | Flagged share (re-weighted) |
|---|---|---|---|---|---|---|---|---|---|---|
| clip_3_m_auto_none.jsonl | 182 | 8 | 0% | 0% | 99% | 50% | 1% | 0% | 0% | 0% |
| clip_m_auto_none.jsonl | 93 | 9 | 0% | 0% | 82% | 41% | 16% | 0% | 0% | 10% |
| clip_3_m_auto_trees.jsonl | 182 | 8 | 2% | 50% | 53% | 52% | 47% | 3% | 56% | 50% |
| clip_m_auto_trees.jsonl | 93 | 9 | 25% | 22% | 93% | 58% | 9% | 26% | 6% | 2% |
| clip_3_m_auto_logistic.jsonl | 182 | 8 | 2% | 75% | 23% | 49% | 77% | 2% | 78% | 80% |
| clip_m_auto_logistic.jsonl | 93 | 9 | 0% | 0% | 82% | 41% | 16% | 0% | 0% | 11% |
| clip_3_m_auto_mlp.jsonl | 182 | 8 | 3% | 50% | 60% | 55% | 41% | 3% | 56% | 42% |
| clip_m_auto_mlp.jsonl | 93 | 9 | 17% | 11% | 94% | 53% | 6% | 6% | 2% | 3% |

