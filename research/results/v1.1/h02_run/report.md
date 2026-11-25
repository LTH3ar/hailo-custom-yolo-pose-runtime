# Handoff 02 report (2026-11-22, Pi 5 + Hailo-8, results in `pi5:~/hcpose_paper/`)

## P and setup
- Backup md5s match the Handoff 01 record: `~/models/thesis_v1.0.3/` holds n 85d8aef2…, s e9929691…, m 632b64d9…
- The new `upper_body_{n,s,m}.hef` md5s match the seeded files:
  - n d2e8eb66… = `upper_body_n_dfc3331_seed2.hef`
  - s 9e3bb4dc… = `upper_body_s_dfc3331_seed2.hef`
  - m b8b021ad… = `upper_body_m_dfc3331_seed1.hef`
  - Nothing else in `~/models` changed. `~/scb` and `~/datasets` were not touched.
- Repo commits: one, `d2af104` on branch `models-v1.1` (C6; bundled classifiers retrained on the new m poses). The working tree is otherwise clean, apart from the untracked `gstshark_*` folders left by `bench.sh`.
- pytest:
  - Runtime: 45 passed. The thesis says 44; one test was added since.
  - Training repo `classifiers`: 7 passed.
- Deviations:
  - `PYTHONPATH` / `ROOT` point at `~/hailo-custom-yolo-pose-runtime`.
  - Every script I wrote or adapted is uncommitted and lives in `~/hcpose_paper/tools/`. Among them is `sweep_rules_h02.py`, a copy of `research/tune/sweep_rules.py` with my paths.
  - **Bug in the committed `sweep_rules.py`.** Its "offset off" rows do not write `head_offset_cue: false`. Since ac18363 the offset is on by default, so those baseline rows really have the offset on. The copy fixes this; the committed script still needs the same fix, and its table header still says "head-offset measure off" for the defaults.
  - **Sweep restarted.** The first attempt died with ENOSPC, because `/tmp` on the Pi is a 4 GB tmpfs. The rerun's work folder is `~/hcpose_paper/sweep_work` (12 GB, safe to delete).
  - **E3 repeat redone.** The first repeat ran during the sweep, and its monitor was never stopped. It is kept in `edge_repeat_invalid/` and not used. The valid repeat is in `edge_repeat/`.
  - Handoff 03's docstring error and the README date correction were reported in Handoff 03.

## T: checklist
- There are 66 rows, every one filled in `~/hcpose_paper/checklist.md` (thesis value → new value, source, step).
- Rows not covered by B–E:
  - `c5_learn` `learn.py` ablations, head-offset AUC, `eval_tiling`, and C7 `evaluate.py`: re-run on the new m poses (`t1/c5_*`, `t1/c7_*`).
  - E6 yaw table and E7 JSON record: own scripts.
  - "Removing head height did not fix it" cannot be re-tested with the existing scripts and is marked as such.

## A: ClaBehavior layout
- 400 images (train + val), none missing.
- Boxes per class: lookup 4480, read 1021, turn_head 1011, raise_hand 684, write 579, discuss 242, stand 66.
- 107 camera groups. Only 14 have more than one frame (150, 121, 10, 3×4, 2×7); 93 are single frames. After pose matching, 106 groups remain in the CV.
- Extractor argument: `~/datasets/clabehavior_yolo:write,read,lookup,turn_head,raise_hand,stand,discuss`

## B: poses
| cache | images | students detected (score ≥ 0.25 / ≥ 0.3) | time |
|---|---|---|---|
| scb_m | 10,138 | 144,430 / 129,378 | 426 s |
| scb_m_tiles | 10,138 | 212,378 / 200,944 | 1549 s |
| scb_n | 10,138 | 151,051 / 135,298 | 260 s |
| scb_n_tiles | 10,138 | 211,459 / 197,259 | 667 s |
| cla_m | 400 | 6,586 / 5,980 | 33 s |
| cla_m_tiles | 400 | 11,428 / 11,134 | 80 s |
| scb_n_nodedup / _tiles_nodedup (E4) | 10,138 | n/a | 262 / 605 s |

## C: tier 1 (5-fold grouped CV; AUC / within-image AUC / balanced acc)
| run | model | pooled | on SCB | on ClaBehavior |
|---|---|---|---|---|
| C1 thesis mapping, SCB (57,972 students, 24.8% distracted) | logistic | 0.768 / 0.730 / 71.0% | = pooled | – |
| | trees | 0.860 / 0.834 / 77.8% | | – |
| | MLP | 0.858 / 0.815 / 77.9% | | – |
| C2 strict, SCB (54,437) | logistic | 0.792 / 0.762 / 72.7% | = pooled | – |
| | trees | 0.890 / 0.861 / 80.3% | | – |
| | MLP | 0.903 / 0.872 / 82.3% | | – |
| C3 strict, both (64,917; Cla 10,480, 13.2% distr.) | logistic | 0.772 / 0.730 / 70.9% | 0.779 / 0.751 / 71.6% | 0.703 / 0.685 / 65.1% |
| | trees | 0.873 / 0.843 / 78.7% | 0.878 / 0.848 / 79.0% | 0.841 / 0.832 / 75.9% |
| | MLP | 0.883 / 0.857 / 79.7% | 0.888 / 0.867 / 80.2% | 0.848 / 0.836 / 75.3% |
| C4 lenient, both (68,575; Cla 10,603, 14.2%) | logistic | 0.762 / 0.726 / 70.5% | 0.762 / 0.729 / 70.4% | 0.742 / 0.718 / 68.7% |
| | trees | 0.855 / 0.836 / 77.8% | 0.855 / 0.833 / 77.7% | 0.848 / 0.842 / 76.4% |
| | MLP | 0.865 / 0.842 / 78.5% | 0.864 / 0.838 / 78.4% | 0.855 / 0.851 / 76.7% |

| cross | train → test | model | AUC | within-img AUC | bal. acc | pred / true distracted |
|---|---|---|---|---|---|---|
| strict | Cla → SCB | logistic / trees / MLP | 0.667 / 0.696 / 0.723 | 0.665 / 0.708 / 0.716 | 63.5 / 63.9 / 66.9% | 41 / 28 / 35% vs 20% |
| strict | SCB → Cla | logistic / trees / MLP | 0.655 / 0.790 / 0.794 | 0.647 / 0.793 / 0.785 | 60.6 / 71.8 / 71.0% | 47 / 34 / 18% vs 13% |
| lenient | Cla → SCB | logistic / trees / MLP | 0.688 / 0.695 / 0.721 | 0.665 / 0.695 / 0.701 | 64.2 / 63.8 / 67.0% | 49 / 34 / 43% vs 25% |
| lenient | SCB → Cla | logistic / trees / MLP | 0.694 / 0.789 / 0.810 | 0.677 / 0.785 / 0.810 | 63.5 / 72.7 / 73.6% | 43 / 30 / 31% vs 14% |

- **C1 vs thesis** (trees 0.855 / 0.840 / 77.0%):
  - New trees: 0.860 / 0.834 / 77.8%. Within ±0.01 on every measure.
  - Exported models match the training outputs within 1e-6.
  - The MLP gives RuntimeWarnings in its logit offset; harmless.
- **Same-picture pairs:** 6,175 joined, reducing the lesson groups from 189 to 188.
  - This is not comparable to Handoff 01's 279. That count used a stricter pixel test, and here the dHash ≤ 4 plus 64×64 mean-diff < 6 test joins more near-duplicates.
  - The joins only matter for the grouping, and they merged one lesson.
- **C5 numbers the thesis quotes** (new m, untiled):
  - `learn.py`, detection threshold 0.5:
    - Rules: AUC 0.544, balanced acc 54.9%, within-image AUC 0.521.
    - Learned model: 0.880 / 78.7%, within-image 0.832.
    - Without each feature block: keypoint geometry 0.824, visibilities 0.876, position 0.864, rule features 0.872. Context only: 0.639.
  - Detection threshold 0.3: rules 0.537 / 53.9%; learned 0.876 / 78.3%.
  - Head-offset AUC (thesis 0.53): 0.559 for TurnHead vs read+write; 0.590 for all distracted vs focused.
  - `eval_tiling` (v1.1 bundled models; optimistic, since those models saw this data):
    - Both found: rules 0.551 / 56%, trees 0.937 untiled / 0.950 tiled, MLP 0.930, logistic 0.803.
    - Found only with tiles: 13,446 students, 32% distracted, trees 0.893.
    - Matched counts (m): 23,073 → 36,294 (+57%); thesis 23,186 → 36,226.
  - C7 `evaluate.py` front view (it does not use the head offset): detection threshold 0.5 AUC 0.549 / 54.9%; 0.3 AUC 0.574 / 53.9%.
- **C6:** https://github.com/LTH3ar/hailo-custom-yolo-pose-runtime/pull/15 ("Bundled classifiers retrained on the new HEF"). Not merged.

## D: tier 2
v1 labels: 275 judgeable, 262 focused, 13 distracted; lecture + exam; tiles on.

| label set | model+tiles | decision | false alarms / focused | specificity (95% CI) | caught / distracted | flagged share re-weighted (lecture / exam) |
|---|---|---|---|---|---|---|
| v1 | n tiles (32 not matched) | rules | 12 / 262 | 95% (92–97) | 2 / 13 | 0.7% / 8.3% |
| v1 | n tiles | thesis trees / logistic / MLP | 104 / 159 / 102 | 60 / 39 / 61% | 2 / 3 / 2 | 55.2 / 81.2 / 54.2% · 2.2 / 11.4 / 5.5% |
| v1 | n tiles | strict trees / MLP | 96 / 101 | 63 / 61% | 1 / 2 | 51.2 / 55.6% |
| v1 | n tiles | lenient trees / MLP | 93 / 94 | 65 / 64% | 1 / 2 | 48.6 / 48.8% |
| v1 | m tiles (18 not matched) | rules | 24 / 262 | 91% (87–94) | 2 / 13 | 0.5% / 20.5% |
| v1 | m tiles | thesis trees / logistic / MLP | 88 / 153 / 84 | 66 / 42 / 68% | 4 / 3 / 4 | 49.8 / 79.7 / 50.4% |
| v1 | m tiles | strict trees / MLP | 78 / 80 | 70 / 69% | 5 / 3 | 46.0 / 47.3% |
| v1 | m tiles | lenient trees / MLP | 84 / 72 | 68 / 73% | 4 / 3 | 46.4 / 42.7% |
| v1 | n / m untiled | rules | 6 / 262; 20 / 262 | 98% / 92% | 1 / 13; – | |
| clip_5 (121 focused, 28 distracted) | n tiles | rules | 13 / 121 | 89% (82–94) | 7 / 28 | 4.9% |
| clip_5 | n tiles | thesis trees / logistic / MLP | 31 / 39 / 36 | 74 / 68 / 70% | 2 / 9 / 3 | 22.3 / 36.7 / 25.2% |
| clip_5 | n tiles | strict MLP; lenient trees | 23; 25 | 81; 79% | 7; 2 | 14.7; 14.4% |
| clip_5 | m tiles | rules | 22 / 121 | 82% (74–88) | 16 / 28 | 9.5% |
| clip_5 | m tiles | thesis trees / logistic / MLP | 28 / 55 / 33 | 77 / 55 / 73% | 6 / 9 / 11 | 11.1 / 35.4 / 16.1% |
| clip_5 | m tiles | strict MLP; lenient MLP | 17; 19 | 86; 84% | 9; 8 | 9.4; 8.4% |

Full tables (all 9 classifiers, tiles on/off): `labels/v1_pooled.md`, `labels/clip5.md`.

Per clip. Class % is the rules / thesis trees / logistic / MLP share flagged; range width is for the rules.

| clip | model | tiles | students/frame | duplicate pairs/frame | class % (range width) | class size | seats calibrated |
|---|---|---|---|---|---|---|---|
| study hall clip_2 | n | off / on | 10.3 / 23.6 | 0.00 / 0.01 | 64.2 / 31.0 / 40.9 / 32.1 (40) · 55.4 / 38.6 / 39.0 / 40.8 (28) | 25 / 59 | 8 of 25 / 16 of 59 |
| study hall clip_2 | m | off / on | 10.6 / 20.2 | 0.00 / 0.00 | 43.7 / 11.9 / 18.6 / 12.0 (44) · 42.3 / 16.7 / 18.1 / 14.9 (34) | 19 / 39 | 10 of 19 / – |
| lecture clip_3 | n | off / on | 16.4 / 18.7 | 0.01 / – | 2.5 / 68.3 / 84.6 / 60.2 (20) · 2.1 / 59.1 / 82.3 / 57.7 (17) | 22 / 26 | |
| lecture clip_3 | m | off / on | 19.8 / 22.0 | | 1.9 / 56.8 / 78.2 / 56.4 (17) · 1.4 / 50.5 / 74.4 / 53.8 (15) | 24 / 26 | |
| clip_5 | n | off / on | 10.2 / 10.5 | 0.03 / – | 17.8 / 36.8 / 59.6 / 37.8 (39) · 10.6 / 16.5 / 37.9 / 18.7 (32) | 16 / 16 | |
| clip_5 | m | off / on | 10.2 / 10.2 | | 23.6 / 13.7 / 24.8 / 17.6 (42) · 13.6 / 7.1 / 28.9 / 13.3 (37) | 14 / 13 | |
| exam clip | n | off / on | 13.5 / 18.8 | 0.00–0.01 | 8.4 / 26.1 / 34.9 / 27.9 (24) · 14.5 / 10.3 / 17.9 / 10.7 (26) | 25 / 29 | |
| exam clip | m | off / on | 14.4 / 22.2 | 0.00 | 25.4 / 11.2 / 15.7 / 7.8 (37) · 21.6 / 9.1 / 17.8 / 5.4 (31) | 20 / 29 | |

- Duplicates: 0.00–0.03 pairs per frame on all 16 runs.
- The shown class % fell inside its range on every frame of all 208 analyses.
- Lecture, m with tiles, the other label sets (trees / logistic / MLP): strict 43.8 / 79.8 / 51.2%; lenient 45.0 / 74.3 / 48.2%.
- D6: head height above the shoulders, distracted vs focused, at 2:00 on the lecture: n tiles 0.45 vs 0.33; m tiles 0.41 vs 0.29.
- D7, front-row student in clip_5 (m tiles, rules):
  - 2 of 9 focused moments flagged, both by `hand_at_face`.
  - 4 of 6 distracted moments flagged.
  - The thesis had 6 of 43 false alarms on this student.
- **D5, study hall with `face_hidden` 0.3:** rules class % drops from 55.4 to 30.3% (n tiles), 64.2 to 41.4 (n untiled) and 43.7 to 26.6 (m untiled). m tiles was not run.
- **Reading:**
  - ClaBehavior's `lookup` class cut the classifiers' false alarms on the lecture only modestly:
    - v1 labels, m tiles: thesis trees / MLP 88 / 84 → strict 78 / 80 → lenient 84 / 72.
    - Lecture class %, m tiles: thesis 50.5% trees → 43.8–46% with the ClaBehavior sets.
    - Every learned model still flags 43–82% of a lecture where the labels say about 5% are distracted. Logistic is the worst at 74–82%.
  - Strict vs lenient: no consistent winner. Strict is better for trees (78 vs 84 false alarms, 5 vs 4 caught); lenient is better for MLP (72 vs 80). Both are far above the rules (24).
  - On clip_5 the two sets differ by at most 4 false alarms. Strict MLP is the best classifier there: 17 false alarms, 9 of 28 caught.
  - The rules stay much better at not flagging focused students (91–98% specificity on v1, 82–89% on clip_5) but catch little on v1 (2 of 13).

## E
| clip_4 | model | decision | distracted seconds caught | precision | false alarms attentive / writing | events missed |
|---|---|---|---|---|---|---|
| | n | rules, original (offset off; E2) | 11% | 100% | 0 / 0 | n/a (sweep does not report it) |
| | n | rules + offset (default) | 56% | 100% | 0 / 0 | 2 of 7 |
| | n | trees / MLP / logistic | 35 / 37 / 9% | 100 / 100 / 52% | 0 / 0; 0 / 0; 18% / 0 | 3 / 4 / 5 of 7 |
| | m | rules, original (offset off; E2) | 11% | 100% | 0 / 0 | n/a |
| | m | rules + offset (default) | 56% | 100% | 0 / 0 | 2 of 7 |
| | m | trees / MLP / logistic | 50 / 49 / 43% | 100 / 100 / 69% | 0 / 0; 0 / 0; 44% / 0 | 2 / 3 / 3 of 7 |
| | m | strict / lenient trees, MLP | 44% / 49% | | | |

- Tiles on and tiles off give identical results on clip_4.
- Phones (on and under the table) are missed by every decision.
- Seats were found at (0.44, 0.37) and (0.61, 0.40).

**E2 sweep.** The sweep uses the tiled n and m runs. The table 4.9 rows are yaw 0.35–0.85, on 0.6, offset off → 0.2–0.4:

| model | clip_4 caught | labels FA | lecture % | exam % |
|---|---|---|---|---|
| n | 11 → 56% | 7 → 12 / 230 (3 → 5%) | 2.1 → 2.1 | 7.1 → 14.5 |
| m | 11 → 56% | 14 → 24 / 244 (6 → 10%) | 1.4 → 1.4 | 11.2 → 21.6 |

- Thesis values: n 11 → 45, 4 → 5%, 4.7 → 4.7, 3.6 → 8.4; m 11 → 56, 6 → 7%, 0.0 → 0.0, 11.7 → 14.6.
- **0.2–0.4 is still best.** Every offset ramp (0.1–0.3, 0.15–0.35, 0.2–0.4) reaches 56% at 100% precision with on 0.6. Among them 0.2–0.4 has the fewest label false alarms: n 12 vs 17 / 24; m 24 vs 24 / 29.
- **Yaw only** (thesis 25%):
  - 25% (n) and 11–25% (m) at on 0.6.
  - The best at 100% precision is 32% (m, yaw 0.25–0.65 or 0.2–0.5, on 0.4).
  - The loosest setting reaches 48% (n, 0.1–0.3, on 0.4), but at 68% precision and 27% label false alarms.
  - Loosening the yaw thresholds alone never reaches 56%.

| edge | model | tiles | FPS | latency median / p95 ms | CPU | memory | board W | max °C | throttled |
|---|---|---|---|---|---|---|---|---|---|
| | n | off | 108.0 | 21 / 24 | 73% | 342 MB | 5.06 | 59.0 | no |
| | n | on | 32.8 | 47 / 56 | 47% | 300 MB | 4.75 | 60.1 | no |
| | s | off | 69.3 | | 47% | 343 MB | 4.53 | 60.1 | no |
| | s | on | 17.4 | | 26% | 303 MB | 3.68 | 58.4 | no |
| | m | off | 31.5 | 42 / 47 | 23% | 346 MB | 3.13 | 57.9 | no |
| | m | on | 7.9 | 2508 / 2514 | 14% | 319 MB | 2.60 | 55.6 | no |

- Hailo-only FPS / hardware latency: n 131.2 / 5.72 ms, s 69.4 / 11.87, m 31.5 / 26.73. The thesis had 131.5 / 69.5 / 31.5.
- 15-minute soaks:
  - n tiles: 30.0 FPS; 55.6 / 56.8 / 59.0 °C at 5 / 10 / 15 min; max 60.1 °C; 4.09 W.
  - m tiles: 7.9 FPS; 51.8 / 51.8 / 52.9 °C; max 53.5 °C; 2.57 W.
  - No throttling.
- Idle: 2.08 W at 48 °C.
- hcpose uses about 3.0 cores at 108 FPS.
- Deployment power: 4.1 + 2.5 = 6.6 W (the thesis had 6.8 W).
- **n untiled, 108.0 vs the thesis's 119.1.** The repeat on an idle Pi (`edge_repeat/`, clip_3, two runs each) gave 111.3 / 110.9 FPS for the new n HEF and 116.3 / 116.4 for the old n HEF.
  - The new compile costs about 5 FPS (4.5%) end to end. Its Hailo-only FPS is unchanged (131.2), so the cost is in post-processing: it finds 20–21 students per frame vs 18.
  - The rest (108 vs 111; 116 vs 119.1) is run-to-run spread.

**E4 coverage.** Share of SCB's annotated students the pose model finds, without → with tiles:

| model | all | far third |
|---|---|---|
| n | 54.6 → 72.0% | 24.6 → 65.5% |
| m | 55.4 → 75.1% | 25.0 → 69.4% |

- m matched counts: 23,073 → 36,294.
- Dedup removes 6.6% (untiled) / 5.9% (tiled) of n detections at score ≥ 0.25, or 4.4% / 3.9% at ≥ 0.3. It changes coverage by −0.5 points.

**E6 / E7.**
- E6, ear-based yaw vs nose offset (m):
  - Attentive: −0.16 / +0.07 vs −0.08 / +0.05.
  - Talking, grey student: −0.01 vs −0.36.
  - Side-look, blue student: −0.27 (−0.10 from baseline) vs −0.52.
- E7, JSON record at 3:50: `class_pct` 50.0, ci [9.5, 90.5], `class_size` 2.

**E5 figures.** All under `figures/`:
- `clip4_overview.png` plus six panels (clip_4, adults only).
- `scb_coverage_by_third.csv`.
- `lecture_exam_class_pct.csv`. Numbers only, no frames from the minors' clips.

**Overall:**
- The INT8 rules are more conservative than the thesis's on the v1 labels: higher specificity, but they catch less (2 of 13).
- With the offset, the rules find 56% of the scripted distraction on clip_4 for both n and m. The thesis had 45–56%.
- The learned models keep their tier-1 AUC (trees 0.86) but still over-flag the lecture.
