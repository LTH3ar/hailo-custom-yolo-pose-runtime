# Pi 5 + Hailo-8 results behind thesis v1.1 onward

These are the results behind every number in thesis v1.1 and later (thesis release v1.7.x, chapter 4). They replace the 10 November runs on the old HEFs (`../pi_20261110`, `../pi_clip5`, `../pi_staged_clip4`), which are kept unchanged.

- **HEFs:** INT8, on the Pi's Hailo-8.

  | model | md5 |
  |---|---|
  | `upper_body_n.hef` | `d2e8eb664c7c0761c55c16b515e9dc78` |
  | `upper_body_s.hef` (= `upper_body_s_dfc3331_seed2.hef`) | `9e3bb4dcc40ed1e7fc0696f5d227579b` |
  | `upper_body_m.hef` | `b8b021ad7144675d66ec8cc22baaa624` |

- **Code:** runtime tag `v1.1.0` (`da19ad3`). The `hcpose/` and `research/` code is unchanged since then; only a line in `research/clips/README.md` differs.
- **Hardware:** Raspberry Pi 5 with a Hailo-8 AI HAT and official active cooling on the fan header. The SCB benchmarks and the crop baseline ran on a PC (RTX 3060); see `h05_crop/`.
- **Runs:** each folder's work happened on pi5 as part of one handoff.

  | handoff | date | folders |
  |---|---|---|
  | 02 | 22 Nov 2026 | `h02_*`, `models/` |
  | 04 | 22 Nov 2026 | `h02_scb/h04/` |
  | 05 | 22 Nov 2026 | `h05_crop/` |
  | 06 | 22 Nov 2026 | `h06_paired/` |
  | 07 | 25 Nov 2026 | `h07_s/`, the s rows of `figures/` |

  The originals stay in `pi5:~/hcpose_paper/`.

## Folders

| folder | contents |
|---|---|
| `h02_clips/` | Skeletons (`*.skeleton.jsonl.xz`) and run logs for exam (`clip`), study hall (`clip_2`), lecture (`clip_3`) and `clip_5_720p` × n/m × tiles off (`none`) / on (`auto`); `clips_report.md`. `d5/`: logs and settings of the study hall with `face_hidden` 0.3 |
| `h02_labels/` | Label scores for all decisions: `v1_pooled.md/csv` (lecture + exam, v1 labels), `clip5.md/csv`, `score_labels_*.md`, `pooled_summary.md`, `d4_range_check.txt`, `d6_d7.md` |
| `h02_staged/` | clip_4 skeletons (n/m × tiles), `summary.md` (rules / bundled models), `summary_nine.md` (all nine classifiers), `e6_yaw_offset.md`, `staged.out` |
| `h02_sweep/` | Rule-threshold sweeps: `sweep_current`, `sweep_yaw`, `sweep_offset` (`.md` tables + logs) |
| `h02_edge/` | `edge_report.md` (the `report.py` output), `system.txt`, `hailo_benchmark_{n,s,m}.txt`, monitor CSVs, throughput / latency / soak logs, `latency_*.trace.xz` (GstShark interlatency), `idle.csv`. `edge_repeat/`: the idle-Pi repeat of n untiled, new vs old HEF. `edge_repeat_invalid/`: a first repeat that ran during the sweep and was discarded |
| `h02_scb/` | `coverage/` (`compare_coverage.py` for n, m, n without dedup), `dedup_n.txt`; `t1/`: SCB lesson-grouped CV (`scb_thesis`, `scb_strict`), SCB + ClaBehavior (`both_strict`, `both_lenient`), `c5_learn_*`, `c5_head_offset_auc_*`, `c5_eval_tiling_m.txt`, `c7_rules_*` (`.json.xz`); `h04/`: ClaBehavior view regrouping (`scan.txt`, `corr0.85/`, `corr0.40/`, `groups.json` per grouping, `groups_h02.json`); `prepare_clabehavior.txt` |
| `h02_run/` | Handoff 02's `report.md`, `checklist.md`, `hailo_queue.sh` (the exact commands), `logs/`, `tools/` (the helper scripts used) |
| `models/` | The nine classifiers: `{thesis,strict,lenient}/scb_{trees,logistic,mlp}.json` |
| `figures/` | Chart data only: `scb_coverage_by_third.csv` (n, s, m) and `lecture_exam_class_pct.csv` (n, s, m × tiles × rules + nine classifiers). These are Handoff 07's files: Handoff 02's rows unchanged, plus s rows |
| `h05_crop/` | Crop baseline (YOLOv8s-cls): `report.md`, `crop_cls.json`, `score.txt`, `cls_{thesis,strict,lenient}.{json,txt}`, `split.json.xz`, `split.txt`, `preds.jsonl.xz`, `files.txt` (image paths only), training / prediction logs, `tools/`; `clips/`: per-student crop decisions on the m and n tiled runs (`*_crop_{raw,smooth}.jsonl.xz`), logs and `scores/` |
| `h06_paired/` | Paired crop vs pose comparison: `paired.json/txt`, `cv_*.json`, `cls_*.txt`, `cla_check.py/txt` (within-view check on ClaBehavior), `run.sh`, `run.log` |
| `h07_s/` | The s model: `report.md`, `queue.sh`; `clips/` (s skeletons + logs, `clips_report*.md`); `staged/` (clip_4); `fps15/` (skeletons and logs of s + tiles in real time on 15 FPS copies); `edge15/` (latency trace, monitor CSVs, soak, `edge15_report.txt`); `scores/`; `sweep/`; coverage / dedup / `eval_tiling_s.txt`; `logs/`; `tools/` |
| `scb_poses/` | m pose caches (`extract_hef.py` output; boxes, keypoints and labels only, no pixels), the poses the learned classifiers were trained on: `scb_m.jsonl.xz`, `scb_m_tiles.jsonl.xz` (SCB) and `cla_m.jsonl.xz`, `cla_m_tiles.jsonl.xz` (ClaBehavior) |

Left out:
- The per-classifier replay outputs (`<skeleton>_<decision>.jsonl`). `hcpose analyze` regenerates them in seconds.
- `soak_input.mp4`, all video, and every image. That includes the clip_4 figure PNGs, even though clip_4's two people are adults who agreed.
- The `.npz` CV arrays (`t1/`, `h06/`; per-student predictions and features), left out to stay well clear of pixel data.
- The image hashes (`scb.dhash.json`, `*.md5.json`).
- The GstShark CTF directories.
- The sweep scratch folders.
- The SCB pose caches for n and s, and the four with `pose_dedup` off (`scb_{n,s}[_tiles][_nodedup]`, 10-16 MB each, 104 MB together), to keep this folder under 300 MB. Their results are in `h02_scb/` and `h07_s/`, and `research/scb/extract_hef.py` (or `h02_run/tools/extract_hef_nodedup.py` for dedup off) regenerates them.

## Thesis table / figure → files
Chapter 4 of `hcpose-thesis` (`en/chapters/ch4.tex`).

| thesis table / figure | files |
|---|---|
| Fig. SCB coverage by size (n, plus s and m in the text) | `figures/scb_coverage_by_third.csv`; `h02_scb/coverage/scb_{n,m}_cov.txt`; `h07_s/coverage_s.txt` |
| Matched counts with / without tiles (text) | `h02_scb/t1/c5_eval_tiling_m.txt` (m); `h07_s/eval_tiling_s.txt` (s) |
| Tab. students detected per frame | `h02_clips/clips_report.md` (n, m); `h07_s/clips/clips_report.md` (s) |
| Tab. effect of same-skeleton duplicate removal | "After" column: `h02_scb/dedup_n.txt`, `h02_scb/coverage/scb_n_nodedup_cov.txt` (n); `h07_s/dedup_s.txt`, `h07_s/coverage_s_nodedup.txt` (s); duplicate pairs per frame in the `clips_report.md` files. **Not mapped:** the "Before" column (1.63 / 1.54 pairs per frame), which predates these runs and is not in this folder |
| Class size and seats (text: 59 / 26 / 29, s 48 / 28 / 31; 16 of 59 seats calibrated) | `clips_report.md` (class size); `h02_run/report.md` (seats, D); `h07_s/scores/seats.txt` |
| Tab. per-student decisions on SCB (5-fold CV) | `h02_scb/t1/scb_thesis.txt`, `c5_learn_scb_m.txt`, `c5_learn_scb_m_thr03.txt`; head-offset AUC `c5_head_offset_auc_scb_m.txt`; rules front view `c7_rules_*` |
| Fig. mean class distracted share, lecture and exam (n, s, m with tiles) | `figures/lecture_exam_class_pct.csv` |
| Tab. per-student decisions against the v1 human labels (lecture + exam) | `h02_labels/v1_pooled.md` (n, m); `h07_s/scores/labels_v1.md` (s) |
| Tab. per-student decisions against the clip_5 human labels | `h02_labels/clip5.md` (n, m); `h07_s/scores/labels_clip5.md` (s); seat counting on clip_5: `h02_labels/d6_d7.md`, clip logs |
| Tab. staged two-person session (clip_4), m; s in the text | `h02_staged/summary_nine.md`, `summary.md`; `h07_s/staged/summary_nine.md` |
| Tab. ear-based yaw and nose offset on clip_4 | `h02_staged/e6_yaw_offset.md` |
| Tab. effect of the head-offset measure across all evidence | `h02_sweep/sweep_offset.md`, `sweep_yaw.md`, `sweep_current.md` (n, m); `h07_s/sweep/sweep_offset_s.md` (s) |
| Fig. what the pose model sees on clip_4 | **Not included:** the PNGs are left out (no images in this folder); they are drawn from `h02_staged/clip_4_720p_m_auto.skeleton.jsonl.xz` |
| Strict / lenient mapping on SCB alone (text) | `h02_scb/t1/scb_strict.txt` |
| Tab. cross-dataset (ClaBehavior ↔ SCB) | `h02_scb/t1/both_strict.txt`, `both_lenient.txt`; within-ClaBehavior CV and grouping: `h02_scb/h04/` (41 groups = `corr0.40/`) |
| Tab. SCB alone vs SCB + ClaBehavior on real classes | `h02_labels/v1_pooled.md` (thesis / strict / lenient rows); lecture %: `figures/lecture_exam_class_pct.csv` |
| Tab. crop benchmark (crop vs pose on the same students) | `h05_crop/score.txt`, `crop_cls.json`, `cls_strict.txt`; intervals `h06_paired/paired.txt`; within-view check `h06_paired/cla_check.txt` |
| Tab. crop classifier on real classes | `h05_crop/clips/scores/` (`v1_pooled.md`, `clip5.md`, `clip4_summary.md`, `class_pct.txt`) |
| Tab. Hailo-8 throughput and latency | `h02_edge/hailo_benchmark_{n,s,m}.txt` |
| Tab. end-to-end performance | `h02_edge/edge_report.md`, `throughput_*.csv/log`; the idle-Pi repeat (111 vs 116 FPS): `h02_edge/edge_repeat/` |
| Latency (text: 21 / 47 / 42 ms; m tiles 2.5 s) | `h02_edge/latency_*.log`, `latency_*.trace.xz`, `edge_report.md` |
| s with tiles on a 15 FPS camera (text) | `h07_s/fps15/*.log`, `h07_s/edge15/edge15_report.txt`, `h07_s/scores/labels_*_15fps.md`, `h07_s/scores/classpct_15fps.txt` |
| Tab. thermal behaviour over 15-minute runs | `h02_edge/soak_n_auto_realtime.*`, `soak_m_auto_max.*`, `edge_report.md`; `h07_s/edge15/soak_s_auto_15fps.*`, `edge15_report.txt`; idle `h02_edge/idle.csv` |
| Tab. vendor power figures | Not a measurement: vendor specification, no file |
| Tab. sources of evidence; data flow; legal provisions | Descriptive, no result files |
| Study hall with `face_hidden` 0.3 (limitations) | `h02_run/report.md` (D5); `h02_clips/d5/`; `h07_s/scores/classpct_d5.txt` |

## Rerun
Settings used everywhere:
- score threshold 0.3;
- camera position: exam (`clip`) `auto`; study hall (`clip_2`), lecture (`clip_3`) and `clip_5_720p` `center`;
- clip_4: `--rows 2 --calib 58`, start offset auto (detected 0.1 s).

From the repository root:

```bash
R=research/results/v1.1
xz -dk $R/h02_clips/clip_3_m_auto.skeleton.jsonl.xz

# rules (default) or a learned classifier
./hcpose.sh analyze $R/h02_clips/clip_3_m_auto.skeleton.jsonl --classifier none --camera-position center -o clip_3_m_auto_none.jsonl
./hcpose.sh analyze $R/h02_clips/clip_3_m_auto.skeleton.jsonl --classifier $R/models/thesis/scb_trees.json --camera-position center -o clip_3_m_auto_thesis_trees.jsonl

# human labels (nearest frame within 0.25 s, then IoU >= 0.3)
python research/labels/score_labels.py research/labels/student_labels_v1.json clip_3_m_auto_*.jsonl clip_m_auto_*.jsonl
python research/labels/score_labels.py research/labels/student_labels_clip5.json clip_5_720p_m_auto_*.jsonl

# per-clip summary (students per frame, duplicates, class %, class size) over a folder of analyses
python research/clips/summarize.py <folder>

# clip_4
xz -dk $R/h02_staged/clip_4_720p_m_auto.skeleton.jsonl.xz
./hcpose.sh analyze $R/h02_staged/clip_4_720p_m_auto.skeleton.jsonl --classifier none --camera-position center -o clip_4_720p_m_auto_none.jsonl
python research/staged/score.py clip_4_720p_m_auto_*.jsonl --script research/staged/script_clip4.csv --rows 2 --offset auto --calib 58

# SCB coverage from the pose caches
xz -dk $R/scb_poses/scb_m.jsonl.xz $R/scb_poses/scb_m_tiles.jsonl.xz
python research/scb/compare_coverage.py $R/scb_poses/scb_m.jsonl $R/scb_poses/scb_m_tiles.jsonl --names none,tiles --det-thr 0.3
```

- **Exact commands:** `h02_run/hailo_queue.sh` and `h07_s/queue.sh`.
- **Helper scripts:**
  - `h02_run/tools/`: `d2_replay.sh` runs the nine classifiers; `d3_score.py` gives the pooled label tables; `sweep_rules_h02.py` is the sweep.
  - `h07_s/tools/`: the s versions, plus `seats_cal.py`, `dedup_share.py`, `classpct.py` and `edge15.py`.
- The ClaBehavior and crop experiments run from the training repository (`hailo-custom-yolo-pose`, `classifiers/experiments/`).

## Privacy
- This folder holds only skeletons (boxes, keypoints, per-seat states), labels, logs, numbers and classifier models.
- It has no frames, crops, images or video of any clip or dataset. That covers SCB, ClaBehavior, the lecture, the study hall, clip_5, the exam hall and clip_4.
- `h05_crop/files.txt` and `split.json.xz` name dataset image files by path only.
