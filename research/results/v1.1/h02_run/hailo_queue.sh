#!/bin/bash
# Handoff 02: every Hailo job, one after another (the Hailo serves one program at a time).
# Steps: E3 edge bench (Pi otherwise idle), B SCB + ClaBehavior poses, D1 clips, E1 clip_4.
# Skips a step whose output exists, so it can be restarted. Log: ~/hcpose_paper/logs/hailo_queue.log
set -uo pipefail
R=~/hailo-custom-yolo-pose-runtime; O=~/hcpose_paper; L=$O/logs/hailo_queue.log; C=~/classroom
mkdir -p $O/logs $O/poses $O/clips $O/staged $O/edge
cd $R
step() { local name=$1; shift; local s=$(date +%s); echo "[$(date -Is)] start $name" >> $L
  "$@"; local rc=$?; echo "[$(date -Is)] end $name rc=$rc $(( $(date +%s)-s ))s" >> $L; return $rc; }
SCB="$HOME/scb/SCB_BowTurnHead_20250509/SCB_BowTurnHead_20250509/SCB5-Turn-Bow-Head-2024-9-17:BowHead,TurnHead $HOME/scb/SCB5-Discuss-2024-9-17/SCB5-Discuss-2024-9-17:discuss $HOME/scb/SCB5-Handrise-Read-write-2024-9-17/SCB5-Handrise-Read-write-2024-9-17:hand-raising,read,write"
CLA="$HOME/datasets/clabehavior_yolo:write,read,lookup,turn_head,raise_hand,stand,discuss"
X=research/scb/extract_hef.py

# E3: edge (first, while nothing else runs)
[ -s $O/edge/edge_report.md ] || step edge research/edge/bench.sh $C/clip_3.mp4 $C/clip_2.mp4 $O/edge > $O/logs/edge_bench.out 2>&1

# B: poses
for m in m n; do
  [ -e $O/poses/scb_$m.done ] || { step scb_$m .venv/bin/python $X ~/models/upper_body_$m.hef $O/poses/scb_$m.jsonl $SCB > $O/logs/scb_$m.out 2>&1 && touch $O/poses/scb_$m.done; }
  [ -e $O/poses/scb_${m}_tiles.done ] || { step scb_${m}_tiles .venv/bin/python $X ~/models/upper_body_$m.hef $O/poses/scb_${m}_tiles.jsonl $SCB --tiles auto > $O/logs/scb_${m}_tiles.out 2>&1 && touch $O/poses/scb_${m}_tiles.done; }
done
[ -e $O/poses/cla_m.done ] || { step cla_m .venv/bin/python $X ~/models/upper_body_m.hef $O/poses/cla_m.jsonl $CLA > $O/logs/cla_m.out 2>&1 && touch $O/poses/cla_m.done; }
[ -e $O/poses/cla_m_tiles.done ] || { step cla_m_tiles .venv/bin/python $X ~/models/upper_body_m.hef $O/poses/cla_m_tiles.jsonl $CLA --tiles auto > $O/logs/cla_m_tiles.out 2>&1 && touch $O/poses/cla_m_tiles.done; }

# D1: classroom clips (n and m, tiles off and on; replays with the old bundled classifiers for reference)
step clips research/clips/run_clips.sh $O/clips $C/clip.mp4:auto $C/clip_2.mp4:center $C/clip_3.mp4:center $C/clip_5_720p.mp4:center > $O/logs/clips.out 2>&1

# E1: staged clip_4
step staged env OUT=$O/staged research/staged/run_all.sh $C/clip_4_720p.mp4 research/staged/script_clip4.csv --rows 2 --calib 58 > $O/logs/staged.out 2>&1
echo "[$(date -Is)] QUEUE_DONE" >> $L

# E4: n caches with pose_dedup off, to measure the share of detections dedup removes (uncommitted copy of extract_hef.py)
XN=$O/tools/extract_hef_nodedup.py
[ -e $O/poses/scb_n_nodedup.done ] || { step scb_n_nodedup .venv/bin/python $XN ~/models/upper_body_n.hef $O/poses/scb_n_nodedup.jsonl $SCB > $O/logs/scb_n_nodedup.out 2>&1 && touch $O/poses/scb_n_nodedup.done; }
[ -e $O/poses/scb_n_tiles_nodedup.done ] || { step scb_n_tiles_nodedup .venv/bin/python $XN ~/models/upper_body_n.hef $O/poses/scb_n_tiles_nodedup.jsonl $SCB --tiles auto > $O/logs/scb_n_tiles_nodedup.out 2>&1 && touch $O/poses/scb_n_tiles_nodedup.done; }
echo "[$(date -Is)] QUEUE2_DONE" >> $L
# E3 check: n without tiles ran at 108 FPS against the thesis 119.1; repeat it with the new and the backed-up old n HEF
mkdir -p $O/edge_repeat
for i in 1 2; do for h in new:$HOME/models/upper_body_n.hef old:$HOME/models/thesis_v1.0.3/upper_body_n.hef; do
  tag=${h%%:*}; f=${h#*:}; sleep 20
  step edge_repeat_${tag}_$i bash -c "cd $R && $R/.venv/bin/python research/edge/monitor.py $O/edge_repeat/n_none_${tag}_$i.csv & M=\$!; cd $R && ./hcpose.sh run $f --input $C/clip_3.mp4 --no-sync --engine gst --headless --engagement --score-threshold 0.3 --camera-position center > $O/edge_repeat/n_none_${tag}_$i.log 2>&1; kill -INT \$M; wait \$M"
done; done
echo "[$(date -Is)] QUEUE3_DONE" >> $L
