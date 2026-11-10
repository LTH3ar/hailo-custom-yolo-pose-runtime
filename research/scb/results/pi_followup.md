# Pi follow-up check: a648f1f fixes + classifiers retrained on tiled poses

Raspberry Pi 5 + Hailo-8 (fw 4.23.0), Raspberry Pi OS trixie, hailo-all 5.1.1, GStreamer 1.26.2, sklearn 1.9.1. 2026-11-10.
Branch `class-coverage`: a648f1f, plus 26185e7 (classifiers, this check) and this report.
The earlier check is `pi_tiling_classifiers.md`.

**Verdict:** all four fixes work on the device.
- The range always contains the shown estimate and is never zero-width.
- Same-skeleton duplicate boxes are gone.
- `--tiles` + display no longer drops buffers.
- The retrained classifiers no longer under-call tiled students. With m, turning tiles on no longer lowers the class distracted share.

Still open: with n, tiles still cut the share by more than half, and the ranges are now very wide (§6).

## 1. Update, build, test — OK

- Fast-forwarded to a648f1f. `./install.sh` rebuilt the post-process `.so`, and the doctor reports all required pieces found.
- Both harnesses compile. The only warning is the known `-Wpessimizing-move` in the TAPPAS header.
- `pytest -q -rs tests`: **40 passed, 0 skipped**.

## 2. SCB extraction — OK

- The paths in the brief match the `img` paths in the old cache. The old cache is kept as `~/scb/poses_hef_m_nodedup.jsonl`.
- `extract_hef.py` opens its output with `"w"`, so the untiled cache was fully rebuilt with `pose_dedup` 0.1 (the PoseDecoder default).
- Untiled pass: `done 10138` at 09:45. Tiled pass (`--tiles auto`, ~7 img/s): `done 10138` at 10:10. No errors.
- Each cache has 10,138 lines. The untiled one is 59.9 MB (was 61.4 MB): 146,077 detections, down from 150,354 (−2.8%) with `pose_dedup`; the tiled one is 84.6 MB.

## 3. Classifiers

### Pooled CV (untiled + tiled caches, det-thr 0.3)

```
58185 students (2 pose cache(s)), 182 lesson groups, distracted 24.7%

model                       AUC  within-img AUC  bal. acc  (5-fold, grouped by lesson)
logistic regression       0.772           0.735    71.3%
gradient-boosted trees    0.855           0.840    77.0%
   per behavior, share predicted distracted: read 19%, write 10%, hand-raising 29%, TurnHead 77%, BowHead 30%, discuss 74%
MLP (64-32)               0.863           0.838    78.0%

exported scb_logistic.json; NumPy runtime vs sklearn max |diff| = 4.0e-07; vs previous: 3.0e-01
exported scb_trees.json;    NumPy runtime vs sklearn max |diff| = 5.8e-16; vs previous: 6.7e-01
exported scb_mlp.json;      NumPy runtime vs sklearn max |diff| = 1.5e-06; vs previous: 9.9e-01
```

All three runtime-vs-sklearn diffs are < 1e-5. The trees differ from the previous file, as expected.
Out-of-fold predictions are saved in `~/scb/cv_m_pooled.npz` / `.json`.
Committed as 26185e7 "engagement: classifiers trained on untiled + tiled INT8 m poses". The models report `scb-m-int8-tiles` at startup.

### Single-cache CV on the new (dedup) untiled cache, for comparison

```
22272 students (1 pose cache(s)), 180 lesson groups, distracted 22.3%

model                       AUC  within-img AUC  bal. acc  (5-fold, grouped by lesson)
logistic regression       0.786           0.717    72.0%
gradient-boosted trees    0.859           0.814    76.4%
   per behavior, share predicted distracted: read 18%, write 9%, hand-raising 24%, TurnHead 72%, BowHead 25%, discuss 74%
MLP (64-32)               0.861           0.815    77.7%
```

(On the old no-dedup cache it was logistic 0.792, trees 0.861, MLP 0.867 AUC. Dedup removed 89 students and changed the AUCs by less than 0.01.)
Pooling raises the within-image AUC (trees 0.814 → 0.840, MLP 0.815 → 0.838) and keeps the overall AUC about the same.

### eval_tiling (bundled = new pooled models; **in-sample**, so only pred vs true matters)

```
front-view students, det-thr 0.3. 'pred' = share labelled distracted at score 0.5

group                       n   true |    rules AUC  bal  pred |    trees AUC  bal  pred | logistic AUC  bal  pred |      mlp AUC  bal  pred
both: untiled pose      22957    23% |         0.551  56%   24% |         0.938  86%   32% |         0.807  74%   40% |         0.953  88%   29%
both: tiled pose        22957    23% |         0.545  56%   26% |         0.951  88%   30% |         0.813  75%   38% |         0.964  90%   28%
only with tiles         13269    32% |         0.564  56%   18% |         0.896  82%   41% |         0.745  69%   45% |         0.918  84%   38%
only without tiles        229    34% |         0.671  64%   39% |         0.840  74%   47% |         0.682  61%   62% |         0.889  78%   38%
```

pred − true, compared with the README table (models trained on untiled poses only; n `best.pt` caches):

| group | README trees | README MLP | now trees | now MLP |
|---|---|---|---|---|
| both: tiled pose | 47 − 55 = **−8** | 45 − 55 = **−10** | 30 − 23 = **+7** | 28 − 23 = **+5** |
| only with tiles | 47 − 65 = **−18** | 44 − 65 = **−21** | 41 − 32 = **+9** | 38 − 32 = **+6** |

- The under-calling of tiled and tiles-only students is gone: the gap is smaller in size and now on the other side.
- The tiled and untiled poses of the same students now get nearly the same pred (trees 30% vs 32%, MLP 28% vs 29%). The README had 47% vs 55%.
- The pooled models now over-call slightly in every row, including untiled poses (trees +9, logistic +17). The true share here is lower (23%) than in the README table (55%): front-view students with this model and threshold. So the two tables aren't the same population, only the direction is comparable.

## 4. Runtime recheck (classroom clip, gst, headless, `--no-sync`, `--score-threshold 0.3`, new models)

| model | tiles | FPS | frames | persons/frame | same-skeleton pairs/frame | estimate outside range | zero-width ranges | mean class_pct | mean range width |
|---|---|---|---|---|---|---|---|---|---|
| n | none | 101.2 | 2150 | 12.6 | **0.01** (was 1.63) | **0** / 2060 | **0** | 35.8% | 36.7 pts |
| n | auto | 32.8 | 2150 | 18.6 | 0.01 (was 1.49) | 0 / 2060 | 0 | 15.6% | 25.2 pts |
| m | none | 31.4 | 2150 | 14.2 | 0.00 (was 0.87) | 0 / 2060 | 0 | 12.8% (first 963 frames: 7.8%) | 28.7 pts |
| m | auto | 7.9 (120 s timeout, 963 frames) | 963 | 22.2 | **0.00** (was 1.54) | 0 / 873 | 0 | 8.7% | 24.1 pts |

- No errors or warnings in any run. Tiling and classifier lines are as expected.
- n none runs at 101 FPS (was 68.6) because this step writes no mp4 (no `--output`). Not a like-for-like comparison with the earlier table.
- Persons/frame is slightly lower than before (n none 14.1 → 12.6, m auto 23.7 → 22.2), as expected from `pose_dedup` removing duplicates.

Effect of tiles on the class distracted share (same frames):

| model | before (old models) | now (pooled models) |
|---|---|---|
| m (first 963 frames) | 6.2% → 2.5% | **7.8% → 8.7%** |
| n (all frames) | 35.2% → 13.0% | 35.8% → 15.6% |

- With m, tiles no longer lower the share.
- With n, the drop is only slightly smaller (§6.1).

## 5. Display run (m, `--tiles auto`, Wayland desktop)

- Prints `[hcpose] tiling: playing the file as fast as it is processed (--sync to force real time)`.
- **0 "A lot of buffers are being dropped" warnings** (was 51 in 60 s). Runs at 7.9 FPS, 441 frames in 56 s.
- The banner shows a range, e.g. `10% distracted (3-29%) | 23 of ~23 students | calibrating`.
  The window shows boxes on the back row and no duplicate skeletons. Labels mostly "focused", 3 "distracted".
- Ctrl+C (SIGINT to the process group): clean stop in 2.6 s, with `stopping (finishing output)...`, the frame summary and the engagement summary.

## 6. Things that look wrong or worth a look

1. **n + tiles still more than halves the distracted share** (35.8% → 15.6%), while m doesn't change much.
   The visible-only share drops too, so it's the per-student labels changing, not the whole-class weighting.
   My guess, not verified: the classifiers are trained on m poses only (untiled + tiled). n's noisy full-frame poses of small far students look "distracted" to them, and tiling replaces those poses with cleaner ones.
   If so, the untiled n number is the inflated one. Training on n caches too, or checking `eval_tiling.py` on the n caches, would show which.
2. **The ranges are now very wide**: 24–37 points on average (e.g. "(23-59%)" for n none, "(3-29%)" for m auto). That is honest for ~20–30 students without a known class size, but on screen it is a weak signal. `--class-size` narrows it.
3. **The pooled models over-call slightly in-sample** (§3: trees +7 to +9 pts, logistic +15 to +17 pts at score 0.5). Not a problem for ranking (AUC), but the absolute share leans high. Logistic leans highest.
4. Minor, test harness only: a `pgrep` on the hcpose command line can also match the short-lived `setsid` wrapper. My first SIGINT went to the wrong group. The app wasn't at fault; the second SIGINT to its real group stopped it cleanly.
