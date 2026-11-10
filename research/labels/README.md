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
