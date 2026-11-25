#!/bin/bash
# Handoff 07: CPU-only work after the Hailo queue (replays, scores, sweep). Log: ~/hcpose_paper/h07/logs/cpu.log
set -uo pipefail
R=~/hailo-custom-yolo-pose-runtime; O=~/hcpose_paper; H=$O/h07; T=$H/tools; L=$H/logs/cpu.log; PY=$R/.venv/bin/python
cd $R
step() { local name=$1; shift; local s=$(date +%s); echo "[$(date -Is)] start $name" >> $L
  "$@"; local rc=$?; echo "[$(date -Is)] end $name rc=$rc $(( $(date +%s)-s ))s" >> $L; return $rc; }
mkdir -p $H/d5 $H/scores $H/sweep

# A: coverage and dedup share
step coverage bash -c "$PY research/scb/compare_coverage.py $H/poses/scb_s.jsonl $H/poses/scb_s_tiles.jsonl --names none,tiles --det-thr 0.3 > $H/coverage_s.txt 2>&1"
step dedup bash -c "python3 $T/dedup_share.py $H/poses/scb_s.jsonl $H/poses/scb_s_nodedup.jsonl $H/poses/scb_s_tiles.jsonl $H/poses/scb_s_tiles_nodedup.jsonl > $H/dedup_s.txt 2>&1"
step coverage_nodedup bash -c "$PY research/scb/compare_coverage.py $H/poses/scb_s_nodedup.jsonl $H/poses/scb_s_tiles_nodedup.jsonl --names none,tiles --det-thr 0.3 > $H/coverage_s_nodedup.txt 2>&1"

# B1 / C: nine classifiers on every s skeleton
step d2_clips $T/d2_replay_s.sh $H/clips clip:auto clip_2:center clip_3:center clip_5_720p:center
step d2_staged env STAGED=1 $T/d2_replay_s.sh $H/staged clip_4_720p:center

# B5: study hall with face_hidden 0.3
for t in none auto; do
  step d5_$t ./hcpose.sh analyze $H/clips/clip_2_s_$t.skeleton.jsonl --engagement $T/face_hidden_0.3.yaml --classifier none \
    --camera-position center -o $H/d5/clip_2_s_${t}_none.jsonl
done

# E2: replays of the 15 FPS real-time skeletons (rules, thesis trees, thesis MLP)
for spec in clip:auto clip_3:center clip_5_720p:center; do n=${spec%:*}; pos=${spec##*:}
  SK=$H/fps15/${n}_s_auto_15fps.skeleton.jsonl
  ./hcpose.sh analyze $SK --classifier none --camera-position $pos -o $H/fps15/${n}_s_auto_none.jsonl > /dev/null
  for k in trees mlp; do
    ./hcpose.sh analyze $SK --classifier $O/models/thesis/scb_$k.json --camera-position $pos -o $H/fps15/${n}_s_auto_thesis_$k.jsonl > /dev/null
  done
done
echo "[$(date -Is)] fps15 replays done" >> $L

# B2 / E2: label scores
for lab in v1 clip5; do LF=$R/research/labels/student_labels_$lab.json
  $PY $T/d3_score_s.py $LF $H/clips --csv $H/scores/labels_${lab}.csv > $H/scores/labels_${lab}.md 2>&1
  $PY $T/d3_score_s.py $LF $H/fps15 --csv $H/scores/labels_${lab}_15fps.csv > $H/scores/labels_${lab}_15fps.md 2>&1
done
$PY research/labels/score_labels.py $R/research/labels/student_labels_v1.json $(ls $H/clips/clip_3_s_*.jsonl $H/clips/clip_s_*.jsonl | grep -v skeleton) \
  > $H/scores/score_labels_v1.md 2>&1
$PY research/labels/score_labels.py $R/research/labels/student_labels_clip5.json $(ls $H/clips/clip_5_720p_s_*.jsonl | grep -v skeleton) \
  > $H/scores/score_labels_clip5.md 2>&1
echo "[$(date -Is)] label scores done" >> $L

# B3 / B4: per clip
$PY research/clips/summarize.py $H/clips > $H/clips/clips_report_nine.md 2>&1
$PY $T/classpct.py $(ls $H/clips/*_s_*.jsonl | grep -v skeleton) > $H/scores/classpct.txt 2>&1
$PY $T/classpct.py $H/d5/*.jsonl > $H/scores/classpct_d5.txt 2>&1
$PY $T/classpct.py $(ls $H/fps15/*_s_auto_*.jsonl | grep -v skeleton) > $H/scores/classpct_15fps.txt 2>&1
python3 $T/seats_cal.py $H/clips/*_s_*_none.jsonl > $H/scores/seats.txt 2>&1

# C: clip_4 with all decisions
$PY research/staged/score.py $H/staged/clip_4_720p_s_*_*.jsonl --script research/staged/script_clip4.csv --rows 2 \
  --offset auto --calib 58 --out $H/staged/summary_nine.md > $H/staged/score_nine.txt 2>&1
echo "[$(date -Is)] per-clip and clip_4 done" >> $L

# D: head-offset sweep on the s files
step sweep nice $PY $T/sweep_rules_s.py --grid offset --jobs 3 --work $H/sweep_work --results-dir $H/clips --staged-dir $H/staged \
  --out $H/sweep/sweep_offset_s.md > $H/sweep/sweep_offset_s.log 2>&1
echo "[$(date -Is)] CPU_DONE" >> $L
