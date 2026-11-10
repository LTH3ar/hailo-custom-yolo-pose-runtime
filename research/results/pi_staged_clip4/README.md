# Staged clip_4 session and human-label scores on the INT8 models (Pi 5 + Hailo-8, 2026-11-10)

INT8 HEFs `upper_body_{n,m}.hef`, tiles off and on, all four decisions (rules = `none`, trees, logistic, MLP). Branch `edge-eval` at 18f9919; the hcpose code is unchanged since the edge run in `../pi_20261110/`.

## Files

- `clip4_summary.md`: `research/staged/score.py` output (`~/classroom/results/summary.md`).
- `clip_4_720p_<model>_<tiles>.skeleton.jsonl.xz`: skeletons (xz -9; unpack with `xz -dk`). Regenerate the analysis files with `hcpose analyze SKELETON --classifier C --camera-position center`.
- `run_all.log`: console output of `run_all.sh`.
- `labels_int8.md`: the INT8 analyses scored against `research/labels/student_labels_v1.json`:
  - a pooled table, from the new `research/labels/pooled_summary.py`
  - the per-clip `score_labels.py` tables

## clip_4: source and conversion

ffprobe of `clip_4.mp4` (630 MB, phone recording):

| | Source | After conversion (`clip_4_720p.mp4`, 27 MB) |
|---|---|---|
| Video | H.264 High, yuv420p, BT.709, 1920×1080 landscape | H.264 (libx264, veryfast, CRF 20), 1280×720 |
| Rotation | no rotation or display-matrix metadata | no rotation or display-matrix metadata |
| Frame rate | 30000/1001 nominal, variable: 8553 frames in 285.44 s = 29.96 FPS average | 30 FPS constant, 8563 frames, 285.43 s |
| Bitrate | 17.5 Mb/s | |
| Audio | AAC | dropped (`-an`) |
| Recorder | Android 16 | |

Not portrait and not rotated, so I converted it with the command as given.

## Start signal and seats

- **Start signal:** found automatically at **0.1 s** into the video, inside the 00:00–00:03 hands-up window. No `--offset 1.5` rerun was needed.
- **Seats:** found as a share of the frame (x, y):
  - S1 (0.44, 0.64)
  - S2 (0.61, 0.69)
- **Left and right:** S1 has the smaller x, so it is the left seat (the man in blue) and S2 the right seat (the woman in grey), as the script expects. I didn't need to look at a frame.
- **Calibration:** hcpose calibrated two seat ids (1 and 3) in every run, i.e. one per person.

## Reading the staged results

Seven scripted distracted events. Scoring starts at 58 s (`--calib 58`) with a 5 s margin around each change. hcpose calibrates a seat over its first 120 s of observation, so S2's "look to the side" (1:47) and S1's "phone under the table" (2:15) fall at the end of or right after calibration.

- **Recall per scripted behaviour** (share of the event's seconds flagged):

  | Behaviour | m, trees / MLP | m, logistic | n, any learned | n, rules | m, rules |
  |---|---|---|---|---|---|
  | Look to the side (S2 1:47, S1 3:38) | ~50% | 100% | 0% | 0% | 0% |
  | Talk to each other (2:45–3:17) | 100% | 50% | 23–50% | 0% | 0% |
  | Turn around (S1 4:00) | 22% | 22% | 44–94% | 94% | 22% |
  | Phone under or on the table | 0% | 0% | 0% | 0% | 0% |

  - Look to the side: n with a learned classifier is 0%, and the rules don't catch it on either model.
  - Turn around: n trees reaches 94%, the same as n rules. On m every decision catches it 22% of the time.
  - Phone: nothing catches it, with any model, tiling or classifier. Head down over a phone looks like head down over notes from the pose, so the decisions that don't false-alarm on writing also don't fire on the phone. This is the main blind spot.
- **Overall recall:** m trees and m MLP reach 40% recall with 100% precision (2–3 of 7 events missed, latency 3 s). The rules reach 2–10% (6 of 7 events missed).
- **False alarms during writing** (00:58–01:47, "hard focused"): **0% for all 16 combinations**.
- **False alarms on the remaining attentive seconds:** 0% for rules, trees and MLP. Logistic false-alarms on 44% of attentive seconds with m and 12% with n, so its 100% on "look to the side" isn't selective.
- **n vs m:**
  - m is clearly better with the learned classifiers: recall 34–40% against 11–19%, latency 0.5–3 s against 9–11.5 s, and class error 26 points against 42–45.
  - n rules catch "turn around" (94%) where m rules don't (22%).
  - Coverage is 100% for n and 96% for m.
- **Tiles:** **no effect on this clip**. Tiled and untiled skeletons are the same in 8562 of 8563 frames (n) and 8554 of 8563 (m), so the scores are identical.
  - Why: both people are large and close to the camera (boxes span about 37–90% of the frame height). The auto tiles cover only the top 60% of rows, so the tile copies are cut at the tile edge and dropped, and the full-frame detections remain. Tiles are for small, distant students.
  - Tiling did run: 32.9 FPS against 131.0 for n, 7.9 against 31.6 for m.
- **Class-level range:** contains the true share 55–77% of the time. The best is 77% (m trees and MLP).

## Reading the human-label scores (see `labels_int8.md`)

275 judgeable moments, 13 of them distracted.

- **Rules:**
  - INT8: 6–16 false alarms on the 262 focused moments (specificity 94–98%), against 27 for FP32.
  - But INT8 rules catch **0–1 of 13** distracted moments, against 5 for FP32.
  - The misses aren't detection failures: of the 13, the INT8 runs detect and score 11 (n untiled), 12 (n tiled, m untiled) or all 13 (m tiled), and the rules call them focused. So with INT8 poses the rules have become more conservative, not blind.
- **Learned classifiers:** 77–152 false alarms out of 262, against 114–199 for FP32, catching 0–4 of 13.
  - Best: m + tiles MLP, 77 false alarms (71% specificity) and 3 of 13 caught.
  - Worst: logistic, 128–152 false alarms.
  - On the focused lecture (clip_3) every learned decision still flags 41–77% of moments.
- **Not matched:** far more than "a few":

  | Model | Tiles | Not matched (of 275) |
  |---|---|---|
  | n | off | 91 |
  | n | on | 48 |
  | m | off | 62 |
  | m | on | 17 |

  - Most of these are students the INT8 model didn't detect at that moment (best IoU < 0.3, median about 0–0.17). The rest are matched to state `unknown`. On clip_3, `unknown` covers 15 items for n untiled, 12 for n tiled and 5 for each m run.
  - The labels came from FP32 n + tiles detections, so lower-coverage settings lose items.
  - The scorer counts unmatched items as "not flagged", which inflates specificity for the low-coverage settings. The last column of the pooled table gives false alarms among matched focused items only: rules 3–7%.

## Anything that failed

Nothing failed:
- Tests: 42 passed.
- The conversion, all four hcpose runs, the 16 analyses and the scoring completed.

Two things to know:
1. `run_all.sh` filters the hcpose output to FPS lines, so the tiled runs' `tiling:` line isn't in `run_all.log`; the 4× lower FPS shows the tiles ran.
2. The label scoring reused the 2026-11-10 analysis files instead of regenerating them; `hcpose/` hasn't changed since they were made.
