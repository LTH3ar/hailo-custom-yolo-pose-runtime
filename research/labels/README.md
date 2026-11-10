# Human labels for student-moments

A small labelled set from real classroom clips, to measure precision and recall of the per-student decision on real classes. The SCB benchmark measures action labels; the staged session measures scripted behaviour.

1. `make_page.py` builds an offline page from a clip, its skeleton file and two `hcpose analyze` outputs (rules and a learned model).
   - Moments are sampled by stratum (rules vs learned decision) so both can be scored. Each moment shows three crops over 2 s plus the whole room.
   - No model output is shown.
   - The page holds video frames, so keep it local and don't publish it.
2. Label in the browser (keys 1/2/3), then export `student_labels.json`.
3. `score_labels.py student_labels.json CLIP=analysis.jsonl ...` reports per file:
   - precision, recall, specificity, balanced accuracy and flagged share
   - the same figures re-weighted from the strata to the whole clip

   It works with any later model run (e.g. the INT8 HEFs on the Pi), because moments are matched by clip, time and box.

## First labelled set (`student_labels_v1.json`)

301 student-moments, labelled blind: 182 from the lecture clip (clip_3) and 93 from the exam-hall clip, plus 26 "can't tell". 13 of the 275 judgeable moments (4.7%) are distracted, or 4.2% re-weighted to the whole clips.

Scored against `best.pt` (n, FP32, tiles), pooled over both clips, as sampled:

| Decision | False alarms on focused moments | Specificity (95% CI) | Distracted moments caught |
|---|---|---|---|
| Rules | 27 / 262 | 90% (85-93%) | 5 / 13 |
| Trees | 114 / 262 | 56% (50-62%) | 3 / 13 |
| MLP | 120 / 262 | 54% (48-60%) | 2 / 13 |
| Logistic | 199 / 262 | 24% (19-30%) | 5 / 13 |

Re-weighted to the whole clips, the rules flag 1% of lecture moments and 3% of exam moments. The learned models flag 56-94% of lecture moments. 13 distracted moments are too few to estimate recall; the staged sessions (`research/staged/`) measure that.
