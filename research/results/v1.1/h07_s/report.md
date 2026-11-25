## Handoff 07 report

- **md5 of upper_body_s.hef = seed2?** Yes. `~/models/upper_body_s.hef` = `upper_body_s_dfc3331_seed2.hef` = 9e3bb4dcc40ed1e7fc0696f5d227579b.
- **Repos at:**
  - Runtime `ce0563b` (main). Its `hcpose/` and `research/` are unchanged since Handoff 02's `d2af104`; only a README line differs.
  - Training `98ec31c` (main).
  - Bundled classifiers v1.1: `scb-m-int8-tiles-v1.1`, trained on the m poses and applied to s unchanged.
  - Thesis models: `~/hcpose_paper/models/thesis/`.

### A coverage, s
SCB, 10,138 images, 68,882 labelled students, detection threshold 0.3 (`compare_coverage.py`, `coverage_s.txt`).

| | none | tiles | gain |
|---|---|---|---|
| all | 52.0% | 72.5% | +20.5 pts |
| far third | 20.6% | 66.6% | +46.0 |
| middle third | 60.3% | 74.8% | +14.5 |
| near third | 75.1% | 76.2% | +1.1 |
| detections / image | 12.1 | 19.3 | |

For comparison, Handoff 02 had n 54.6 → 72.0 (far 24.6 → 65.5) and m 55.4 → 75.1 (far 25.0 → 69.4).

- **Matched counts** (front-view students, `eval_tiling`, as in Handoff 02):
  - 22,394 → 35,326 (+58%): both 22,191; only with tiles 13,135; only without tiles 203.
  - Handoff 02 had m 23,073 → 36,294; 13,446; 225.
- **Only-with-tiles students:** 34% distracted (both-found 22%).
  - Bundled trees AUC 0.816 on them vs 0.875 / 0.883 on both-found (untiled / tiled poses).
  - MLP 0.823 vs 0.883 / 0.897. Rules 0.548 vs 0.547 / 0.553.
- **Dedup share** (`pose_dedup` 0 vs default):
  - Removes 3.6% (untiled) / 3.8% (tiled) of s detections at score ≥ 0.25, and 2.4% / 2.4% at ≥ 0.3.
  - Coverage without dedup: 52.3 → 72.9%, so dedup costs 0.3–0.4 points.
  - Handoff 02 n: 6.6 / 5.9% at ≥ 0.25 and 4.4 / 3.9% at ≥ 0.3.
- **CSV:** `scb_coverage_by_third.csv` is Handoff 02's file plus 4 s rows.

### B labels (s tiles; s none for reference)
Matching: nearest frame within 0.25 s, then IoU ≥ 0.3; a student who is not matched counts as not flagged. Full tables: `scores/labels_v1.md`, `scores/labels_clip5.md`.

**v1 (lecture + exam): 262 focused, 13 distracted.** Not matched: tiles 15, none 57.

| decision | FA / focused, tiles | specificity (95% CI), tiles | caught, tiles | flagged share lecture / exam, tiles | FA / spec / caught, none |
|---|---|---|---|---|---|
| rules | 19 | 93% (89–95) | 1 / 13 | 0.6 / 19.3% | 14 · 95% (91–97) · 2 |
| thesis trees | 94 | 64% (58–70) | 3 | 54.0 / 0.9% | 96 · 63% · 3 |
| thesis logistic | 133 | 49% (43–55) | 3 | 71.1 / 13.1% | 121 · 54% · 3 |
| thesis MLP | 97 | 63% (57–69) | 4 | 55.5 / 4.0% | 86 · 67% · 4 |
| strict trees | 75 | 71% (66–77) | 3 | 47.8 / 0.0% | 77 · 71% · 3 |
| strict logistic | 138 | 47% (41–53) | 3 | 75.0 / 11.4% | 124 · 53% · 3 |
| strict MLP | 94 | 64% (58–70) | 2 | 55.4 / 0.0% | 89 · 66% · 2 |
| lenient trees | 80 | 69% (64–75) | 3 | 47.4 / 0.8% | 83 · 68% · 3 |
| lenient logistic | 132 | 50% (44–56) | 3 | 70.7 / 13.2% | 121 · 54% · 3 |
| lenient MLP | 76 | 71% (65–76) | 5 | 43.1 / 4.1% | 76 · 71% · 5 |

**clip_5: 121 focused, 28 distracted.** Not matched: tiles 4, none 53.

| decision | FA / focused, tiles | specificity (95% CI), tiles | caught, tiles | flagged share, tiles | FA / spec / caught, none |
|---|---|---|---|---|---|
| rules | 15 | 88% (81–92) | 14 / 28 | 6.3% | 15 · 88% · 8 |
| thesis trees | 25 | 79% (71–86) | 4 | 14.6% | 18 · 85% · 8 |
| thesis logistic | 44 | 64% (55–72) | 6 | 28.7% | 34 · 72% · 8 |
| thesis MLP | 34 | 72% (63–79) | 13 | 27.3% | 27 · 78% · 11 |
| strict trees | 26 | 79% (70–85) | 6 | 13.9% | 14 · 88% · 8 |
| strict logistic | 44 | 64% (55–72) | 4 | 28.6% | 31 · 74% · 9 |
| strict MLP | 19 | 84% (77–90) | 11 | 10.5% | 21 · 83% · 10 |
| lenient trees | 22 | 82% (74–88) | 7 | 14.6% | 15 · 88% · 8 |
| lenient logistic | 38 | 69% (60–76) | 6 | 23.7% | 32 · 74% · 9 |
| lenient MLP | 23 | 81% (73–87) | 6 | 11.9% | 20 · 83% · 8 |

- Handoff 02, tiles, rules: v1 n 12 / 262 with 2 caught, m 24 / 262 with 2 caught; clip_5 n 13 / 121 with 7 caught, m 22 / 121 with 16 caught.
- The s rules sit between n and m on false alarms. On clip_5 they catch 14 of 28, between n's 7 and m's 16.
- Classifier names: the plain `trees` / `logistic` / `mlp` (bundled v1.1) gave the same outputs as `thesis_*` on every clip.

### B per clip (s none / tiles)
Class % is the rules, then thesis trees / logistic / MLP; range width is for the rules. Lecture, study hall and clip_5 are averaged after 2:30; the exam over the whole clip.

| clip (true count) | students/frame | duplicate pairs/frame | class % (rules range width), none · tiles | class size | seats calibrated (of seats seen) |
|---|---|---|---|---|---|
| study hall clip_2 (~50) | 9.6 / 22.0 | 0.00 / 0.00 | 43.8 / 11.6 / 16.0 / 11.2 (46) · 40.7 / 23.0 / 22.1 / 21.3 (31) | 18 / 48 | 8 of 37 / 19 of 98 |
| lecture clip_3 (28) | 18.9 / 20.7 | 0.00 / 0.00 | 3.5 / 58.1 / 77.3 / 57.5 (18) · 0.5 / 51.8 / 78.4 / 60.6 (13) | 27 / 28 | 18 of 50 / 21 of 54 |
| clip_5 (10) | 10.5 / 10.3 | 0.04 / 0.03 | 23.6 / 14.4 / 41.2 / 33.7 (43) · 14.9 / 9.1 / 26.6 / 15.9 (36) | 15 / 13 | 13 of 33 / 15 of 28 |
| exam clip (23 + teacher) | 15.1 / 22.1 | 0.00 / 0.00 | 14.8 / 12.8 / 13.8 / 10.5 (29) · 18.3 / 5.8 / 10.8 / 3.8 (29) | 23 / 31 | 0 of 33 / 0 of 37 |

- **Range check:** the shown class % was inside its range on every frame of all 115 s analyses:
  - 104 nine-decision plus rules replays;
  - 2 `face_hidden` 0.3 runs;
  - 9 replays of the 15 FPS runs.
- **Class size vs true count:**
  - Tiles get the lecture right (28) and come close on the study hall (48 of ~50; 18 untiled).
  - Tiles overcount the exam (31 vs 24) and clip_5 (13 vs 10).
- **Exam:** 0 seats calibrate with camera position `auto`.
- **Throughput:** s untiled 68.9–69.6 FPS; with tiles 17.4 FPS on every clip.

### B lecture / exam class %
Order: rules, then thesis / strict / lenient × trees / logistic / MLP. File: `lecture_exam_class_pct.csv`, which is Handoff 02's 80 rows plus 40 s rows.

| | rules | thesis T / L / M | strict T / L / M | lenient T / L / M |
|---|---|---|---|---|
| lecture, s tiles | 0.5 | 51.8 / 78.4 / 60.6 | 44.7 / 76.4 / 62.1 | 47.8 / 74.5 / 53.5 |
| lecture, s none | 3.5 | 58.1 / 77.3 / 57.5 | 47.0 / 75.6 / 61.4 | 51.3 / 74.9 / 52.1 |
| exam, s tiles | 18.3 | 5.8 / 10.8 / 3.8 | 2.2 / 11.0 / 1.0 | 6.1 / 12.8 / 4.7 |
| exam, s none | 14.8 | 12.8 / 13.8 / 10.5 | 6.1 / 15.0 / 9.5 | 12.2 / 18.0 / 14.5 |

**Study hall with `face_hidden` 0.3** (rules, after 2:30):
- s tiles: 40.7 → 20.7%, range width 26.
- s untiled: 43.8 → 26.5%, range width 42.
- Handoff 02: n tiles 55.4 → 30.3%, n untiled 64.2 → 41.4%, m untiled 43.7 → 26.6%.

### C clip_4 (s none / tiles)
Tiles on and off give identical results (as in Handoff 02). Settings: `--rows 2 --calib 58 --offset auto`, with the start detected at 0.1 s. Seats were found at (0.44, 0.38) and (0.61, 0.40).

| decision | seconds caught | precision | FA attentive / writing | events missed |
|---|---|---|---|---|
| rules, offset off (sweep) | 18% | 100% | 0 / 0 | n/a |
| rules + offset (default 0.2–0.4) | 43% | 100% | 0 / 0 | 3 of 7 |
| thesis trees / logistic / MLP | 35 / 35 / 49% | 100 / 64 / 100% | 0 / 0; 44% / 0; 0 / 0 | 4 / 4 / 3 of 7 |
| strict trees / logistic / MLP | 24 / 28 / 49% | 100 / 55 / 100% | 0 / 0; 51% / 0; 0 / 0 | 4 / 4 / 3 of 7 |
| lenient trees / logistic / MLP | 28 / 36 / 49% | 100 / 65 / 100% | 0 / 0; 44% / 0; 0 / 0 | 4 / 3 / 3 of 7 |

- Every decision misses both phone events (on and under the table).
- Rules: look to the side 100%, talk 50%, turn around 100%.
- Handoff 02 rules + offset: 56% for both n and m.

### D sweep, s
Head-offset sweep on yaw 0.35–0.85, on 0.6, run on the untiled clip_4 and the tiled lecture, exam and study hall.
- Label FA counts only matched focused students, so the denominator is 247.
- Full grid: `sweep/sweep_offset_s.md`.

| head offset | clip_4 caught | label FA | lecture % | exam % |
|---|---|---|---|---|
| yaw only (off) | 18% | 11 / 247 (4%), 0 / 13 caught | 0.5 | 6.4 |
| 0.1–0.3 | 56% | 27 / 247 (11%), 6 caught | 1.0 | 27.3 |
| 0.15–0.35 | 56% | 23 / 247 (9%), 4 caught | 0.5 | 22.5 |
| **0.2–0.4 (default)** | 43% | 19 / 247 (8%), 1 caught | 0.5 | 18.3 |

- Every row keeps 100% precision and 0 false alarms on clip_4.
- At on 0.5, 0.2–0.4 also reaches 56% (21 / 247 FA, exam 22.1%).
- Yaw only, the best is 32% (yaw 0.2–0.5).
- Handoff 02, yaw only: n 25%, m 11–25%.

### E 15 FPS (s tiles)
**FPS kept.** Each run processed every frame of its 15 FPS copy, with `--sync`, paced like a camera:
- exam: 1,075 of 1,075 frames, 14.9 FPS average;
- lecture: 2,971 of 2,971, 15.0 FPS;
- clip_5: 12,613 of 12,613, 15.0 FPS.

During those runs the Pi used 22–23% CPU (hcpose 77–83% of one core), peaked at 59.0 °C and drew 3.1–3.3 W, with no throttling.

**Labels and class % at 15 FPS vs full rate** (replay of the real-time skeletons):

| | rules | thesis trees | thesis MLP |
|---|---|---|---|
| v1 FA / focused, 15 FPS (full) | 16 (19) | 86 (94) | 93 (97) |
| v1 specificity, 15 FPS (full) | 94% (93%) | 67% (64%) | 65% (63%) |
| v1 caught / 13, 15 FPS (full) | 2 (1) | 3 (3) | 3 (4) |
| v1 not matched, 15 FPS (full) | 18 (15) | | |
| clip_5 FA / 121, 15 FPS (full) | 20 (15) | 24 (25) | 35 (34) |
| clip_5 caught / 28, 15 FPS (full) | 14 (14) | 5 (4) | 14 (13) |
| lecture class %, 15 FPS (full) | 1.1 (0.5) | 52.9 (51.8) | 57.2 (60.6) |
| exam class %, 15 FPS (full) | 15.6 (18.3) | 5.6 (5.8) | 3.2 (3.8) |
| clip_5 class %, 15 FPS (full) | 15.5 (14.9) | 8.6 (9.1) | 15.0 (15.9) |

**Latency:**
- Lecture at 15 FPS, GstShark interlatency from `live_pace` to the sink, over 1,046 frames: median **72 ms**, p95 **85 ms**, max 148 ms.
- First half 69 ms, second half 74 ms: frames do not queue.
- For reference, Handoff 02 n tiles on the full-rate lecture: 47 / 56 ms.

**Soak:** 15 min on the 15 FPS copy of the soak source (`soak_input.mp4`, the study hall looped).

| measure | result |
|---|---|
| FPS | 15.0 (13,544 frames in 903 s) |
| CPU | 23% of the Pi (hcpose 84% of one core) |
| memory | 502 MB max, flat from 1.5 min on (499 → 502 MB) |
| board | 3.14 W |
| SoC temperature at 5 / 10 / 15 min | 56.8 / 57.9 / 56.2 °C |
| max temperature | 60.6 °C |
| throttling | none |

The CPU clock averaged 2170 MHz, with a minimum of 1500 MHz from the governor, not from throttling.

### Settings taken from Handoff 02
- **Camera positions** (`hailo_queue.sh`): exam `auto`; study hall, lecture and clip_5 `center`.
- **clip_4:** `--rows 2 --calib 58`, with the offset left at auto (start detected at 0.1 s, as in Handoff 02). Score threshold 0.3.
- **Labels:** `student_labels_v1.json` and `student_labels_clip5.json`. SCB subsets are the same as Handoff 02.
- **15 FPS copies:** `ffmpeg -vf fps=15 -c:v libx264 -crf 18 -an`.
- **Latency and soak runs** use `bench.sh`'s settings:
  - Latency: `HCPOSE_PACE_FILE=1`, GstShark interlatency, `--no-sync`, 70 s. The `rc=124` in the log is the planned stop by `timeout`.
  - Soak: `--sync`, `timeout 900`.

### Anything odd
1. **Memory:** about 500 MB in every s-tiles real-time run. Handoff 02's bench showed 303 MB for s tiles (max-rate, `--no-sync`). It is flat during the soak, so it is not a leak. I did not check the cause; it may be from `--sync` pacing or a different process set matched by the monitor.
2. **clip_4 with s rules:** 43% at the default offset ramp vs 56% for n and m. 0.15–0.35 or on 0.5 recovers 56% at no clip_4 cost, but raises label false alarms (+4 / +2) and the exam share (+4 pts).
3. **Exam: no seat ever calibrates** (`auto` camera position), with or without tiles. The rules there run uncalibrated.
4. **Tiles overcount** clip_5 (13 vs 10) and the exam (31 vs 24).
5. **Bundled classifiers on s:** they were trained on m poses. On SCB they lose about 0.06 AUC vs Handoff 02's m numbers (trees both-found 0.875 vs 0.937).
6. **Helper scripts:** new and uncommitted, in `h07/tools/`:
   - `sweep_rules_s.py` (a copy of `sweep_rules.py`, whose `SKELETONS` list only n and m);
   - `d2_replay_s.sh` and `d3_score_s.py` (s-only copies of Handoff 02's);
   - `seats_cal.py`, `dedup_share.py`, `classpct.py`, `mk_csv.py`;
   - `edge15.py`, with `report.py`'s `stats`/`latency_ms` copied unchanged, because `report.py` only loops over n and m.
7. **Deleted:** `video15/` (including the soak copy) and the sweep scratch `sweep_work/`. No frames, crops or video were saved from the private clips.
