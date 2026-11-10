# Pi check: tiling in the gst engine + switchable classifiers (PR #3, `class-coverage`)

Raspberry Pi 5 + Hailo-8 (fw 4.23.0), Raspberry Pi OS trixie, hailo-all 5.1.1 (TAPPAS core 5.1.0), GStreamer 1.26.2.
Branch `class-coverage` at 934b430 + 2f4523f (the classifier commit from this check), 2026-11-10.
Test video: `~/classroom/clip.mp4` (1280x720, 30 fps, 2150 frames, 71.7 s) and the Hailo `example.mp4` (1280x720, 336 frames).

**Verdict:** the build, tests, tiling pipeline, classifier switch and Ctrl+C all work on the device.
Tiling adds detections without seam artefacts or drawn tiles. The issues below are worth fixing before or soon after merge.
The most important ones: the whole-class CI is inconsistent with its point estimate (§7.1), and with m + tiles the display run cannot keep real time (§7.2).

## 1. Update and build — OK

- `git checkout class-coverage && git pull`: clean, at 934b430.
- `./install.sh`: rebuilt `build/libhcpose_yolov8_pose.so`; the doctor reports all required pieces (only the camera items are optional and missing).
  The copy in `hcpose/lib/` is byte-identical to `build/`.
- `nm -D`: all 6 symbols present: `init`, `free_resources`, `filter`, `filter_letterbox`, `create_tile_crops`, `merge_tiles`.

## 2. Tests — OK

- `build/cpp_harness` and `build/tile_harness` compile. The only warnings are `-Wpessimizing-move` in the TAPPAS header `hailo_common.hpp:256`, none in our code.
- `pytest -q -rs tests`: **37 passed, 0 skipped** (30 s).

## 3. Classifier export (INT8 m cache, det-thr 0.3) — OK

```
[scbdata] 10138 records -> 8116 unique images (1892 duplicated across subsets); 2891 students with both focused and distracted labels dropped; 189 lesson groups
22361 students (1 pose cache(s)), 179 lesson groups, distracted 22.4%

model                       AUC  within-img AUC  bal. acc  (5-fold, grouped by lesson)
logistic regression       0.792           0.721    72.8%
gradient-boosted trees    0.861           0.811    77.0%
   per behavior, share predicted distracted: read 18%, write 9%, hand-raising 23%, TurnHead 73%, BowHead 25%, discuss 74%
MLP (64-32)               0.867           0.814    79.1%

exported scb_logistic.json (0.00 MB); NumPy runtime vs sklearn max |diff| = 3.9e-07
exported scb_trees.json    (0.63 MB); NumPy runtime vs sklearn max |diff| = 6.7e-16; vs previous scb_trees.json: max |diff| 0.0e+00 (identical)
exported scb_mlp.json      (0.11 MB); NumPy runtime vs sklearn max |diff| = 8.1e-07
```

- All runtime-vs-sklearn diffs are < 1e-5. The trees output is identical.
- The re-exported `scb_trees.json` differed in bytes only in `meta`: a new `created` timestamp, `cv_gradient_boosted_trees` renamed to `cv`, a new `kind`, and a new top-level `logit_offset: 0.0`, which equals the runtime default.
  The weights are identical, so I kept the committed file (`git checkout`).
  The committed trees file therefore still has the old meta layout (`cv_gradient_boosted_trees`, no `kind`). Harmless, but re-export it when convenient so all three files match.
- `scb_mlp.json` carries `logit_offset = 1.24` and `scb_logistic.json` 0.0.
- Committed as 2f4523f "engagement: bundled logistic and MLP classifiers (scb-m-int8)" (sklearn 1.9.1, numpy 2.2.4) and pushed.

## 4. Tiling in the gst engine

`--tiles auto` prints: `[hcpose] tiling: full frame + 3 tile(s) over rows 0%-60% (493x432 px each); 4 inferences per frame`, as expected.
No errors, tracebacks or GStreamer warnings in any headless run.

### FPS (headless, `--no-sync`, recording to mp4 + JSONL, `--engagement --score-threshold 0.3`)

| model | tiles none | tiles auto | ratio |
|---|---|---|---|
| upper_body_n | 68.6 FPS (2150 fr / 31.4 s) | 32.8 FPS (2150 fr / 65.6 s) | 0.48x |
| upper_body_m | 31.4 FPS (2150 fr / 68.4 s) | **7.9 FPS** (963 fr / 122 s, stopped by the 120 s timeout, clean shutdown) | 0.25x |
| upper_body_m, example.mp4 | 30.6 FPS | 7.8 FPS | 0.25x |

m is device-bound (≈31 inferences/s), so 4 inferences per frame cost exactly 4x.
n is host-bound without tiles, so it only halves.

### Persons per frame

| run | frames | persons/frame |
|---|---|---|
| n none | 2150 | 14.1 |
| n auto | 2150 | 20.1 (+42%) |
| m none | 2150 (first 963: 14.9) | 15.1 |
| m auto | 963 | 23.7 (+59% vs same 963 frames) |
| m none, example.mp4 | 336 | 6.05 |
| m auto, example.mp4 | 336 | 6.50 |

### Visual check (frame at 5 s, m none vs m auto; also n)

- **Back row: yes.** Auto adds boxes for back-row students and the teacher at the back (e.g. track IDs 27, 28, 21, 20, 12, 19 in m/auto), none of which are detected without tiles.
- **No tile rectangles** drawn, in either video.
- **No boxes cut at the seams.** Box edges within 6 px of the inner tile edges (x ≈ 394, 493, 787, 886, rows < 432):
  n 13.4% → 11.2%, m 7.6% → 8.7% (none → auto). The non-tiled baseline is already above uniform (3.8%) because some students sit at those x positions. Tiling adds nothing at the seams, and no box bottom sits at y = 432.
- **Duplicates: some, mostly not caused by tiling with n, but more of them with m.**
  Pairs of boxes in one frame with near-identical skeletons (mean keypoint distance < 0.1 box height), per frame over the first 963 frames:
  n none 1.63 → n auto 1.49; **m none 0.87 → m auto 1.54**; example m 0.13 → 0.13.
  Visually, m/auto at 5 s has two overlapping skeleton boxes on one student in the right-hand back cluster (IDs 14 and 5) and a crowded middle cluster (IDs 12/5/9).
  The n model already produces such pairs without tiling (IDs 8 and 6 on one student in n/none), because NMS at IoU 0.7 lets them through.
  `merge_tiles` only removes duplicates across sources (`k->src != c.src`), with IoU ≥ 0.5 or containment ≥ 0.7 plus keypoint agreement.
  The m/auto extras are probably full-frame vs tile detections of the same partly-occluded student whose keypoints disagree by more than 0.2·√area. Worth a look, but not a blocker.
- example.mp4: looks fine. One tiny far-away detection near the right edge (~1230, 340) appears with tiles.

### Display run (m, `--tiles auto`, screen attached, Wayland desktop)

- The window opens and shows the annotated video with the banner (screenshot at 25 s looked right).
- **Ctrl+C (SIGINT to the process group): clean stop in 2.1 s**, prints `stopping (finishing output)...`, the frame summary and the engagement summary.
- **Problem:** this run (file input + display, so `sync=true`) printed `GStreamer warning: A lot of buffers are being dropped.` 51 times in 60 s.
  At 7.8 FPS of processing, the synced display sink cannot keep the 30 fps file clock and drops late buffers. The run processed only 471 frames in 60 s.
  Without tiles, m runs at ~31 FPS, so this only happens with tiles.

## 5. Classifier switch — OK

Each run names the chosen classifier at startup with no error (60 s timeout, ~1915 frames each, 31.4 FPS):

| `--classifier` | startup line | mean class_pct_distracted | mean visible pct_distracted | end-of-run summary |
|---|---|---|---|---|
| trees | `learned classifier trees (scb-m-int8)` | 10.5% | 11.4% | 11.3% of student-time |
| logistic | `learned classifier logistic (scb-m-int8)` | 18.3% | 16.7% | 16.6% |
| mlp | `learned classifier mlp (scb-m-int8)` | **4.0%** | 3.7% | 3.5% |
| none | `rule-based cues` | 11.0% | 12.0% | 12.1% |

(Default score threshold here, as in the brief, so ~15 students per frame are counted. The class_pct mean is over the ~1825 frames where it is present.)
The MLP has the best CV numbers but labels far fewer students distracted on this clip than the other three. Its `logit_offset = 1.24` calibration is a likely suspect.

## 6. Push

- 2f4523f (classifiers) pushed to `origin/class-coverage`.
- This report is committed as `research/scb/results/pi_tiling_classifiers.md` and pushed.

## 7. Things that look wrong

1. **Whole-class CI vs its point estimate** (`hcpose/engagement/analyzer.py:322-324`, `_class_estimate` ~l.390-398).
   - `class_pct_distracted` is the 5 s smoothed estimate (`est_smooth`), but `class_ci` is the current frame's interval. The printed number falls outside its own interval in **29–95% of frames**, depending on the run (trees: 87%, m/auto: 95%).
     Example, trees run, last frame: `class_pct_distracted 14.1, class_ci [16.7, 25.6]`.
   - Without `--class-size`, N = the number of known seats. Whenever every known seat is counted (N == n), the finite-population correction sets `half = 0`, giving a **zero-width CI**.
     This happened in 1295/1825 frames of the classifier runs and in **all** m/auto frames, so the banner drops the range (`3% distracted | 31 of ~31 students`).
     This is technically a census, but it ignores classifier error, and "~31" means the class size is itself an estimate.
     Suggestion: smooth the CI the same way or show the per-frame estimate, and keep a non-zero floor when `size_from == "seats"`.
2. **m + `--tiles auto` cannot keep real time** (7.9 FPS). With a display and file input (`sync=true`), GStreamer floods "A lot of buffers are being dropped".
   Options: `--tiles auto` could warn, or fall back for m; the display sink could use `qos`/`max-lateness` and a leaky queue; or quiet that specific warning when tiling. n + tiles (33 FPS) is the realistic real-time combination.
3. **Tiling moves the distracted share a lot.** n: class estimate 35.2% → 13.0% (visible 26.6% → 6.8%). m, same 963 frames: 6.2% → 2.5%.
   The newly detected back-row students are mostly classified focused, and the n model's low-res full-frame poses may have been reading as distracted.
   There's no ground truth here, so I can't tell which number is right, but it's worth checking against the SCB tiling results before treating tiled and untiled numbers as comparable.
4. **Duplicate boxes with m + tiles** (§4): ~0.7 extra same-skeleton pairs per frame. Probably full-frame vs tile detections of one student whose keypoints disagree by more than the 0.2·√area test allows.
5. Minor: the re-exported `scb_trees.json` meta differs from the committed one (§3). The committed file has the old `cv_gradient_boosted_trees` key and no `kind`. Weights are identical.
6. Minor: `[hcpose] calibrating` stays on for the whole 72 s clip (calibration_seconds is 120), so every `class` block in these runs is from the calibrating phase.
