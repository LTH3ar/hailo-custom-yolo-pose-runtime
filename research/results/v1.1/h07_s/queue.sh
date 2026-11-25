#!/bin/bash
# Handoff 07: every Hailo job for the s model, one after another (the Hailo serves one program at a time).
# Same settings as Handoff 02 (~/hcpose_paper/hailo_queue.sh). Skips steps whose output exists, so it can be
# restarted. Log: ~/hcpose_paper/h07/logs/queue.log
set -uo pipefail
R=~/hailo-custom-yolo-pose-runtime; O=~/hcpose_paper; H=$O/h07; L=$H/logs/queue.log; C=~/classroom
S=~/models/upper_body_s.hef
mkdir -p $H/logs $H/poses $H/clips $H/staged $H/fps15 $H/edge15 $H/video15
cd $R
step() { local name=$1; shift; local s=$(date +%s); echo "[$(date -Is)] start $name" >> $L
  "$@"; local rc=$?; echo "[$(date -Is)] end $name rc=$rc $(( $(date +%s)-s ))s" >> $L; return $rc; }
mon() { .venv/bin/python research/edge/monitor.py "$H/edge15/$1.csv" & MON=$!; }
unmon() { kill -INT $MON 2>/dev/null; wait $MON 2>/dev/null; }
SCB="$HOME/scb/SCB_BowTurnHead_20250509/SCB_BowTurnHead_20250509/SCB5-Turn-Bow-Head-2024-9-17:BowHead,TurnHead $HOME/scb/SCB5-Discuss-2024-9-17/SCB5-Discuss-2024-9-17:discuss $HOME/scb/SCB5-Handrise-Read-write-2024-9-17/SCB5-Handrise-Read-write-2024-9-17:hand-raising,read,write"
X=research/scb/extract_hef.py; XN=$O/tools/extract_hef_nodedup.py

# E: 15 FPS copies (CPU only; kept on pi5 in video15/, deleted at the end)
for c in $C/clip.mp4 $C/clip_3.mp4 $C/clip_5_720p.mp4; do
  f=$H/video15/$(basename "${c%.*}")_15fps.mp4
  [ -s $f ] || step ffmpeg_$(basename "${c%.*}") ffmpeg -v error -y -i "$c" -vf fps=15 -c:v libx264 -crf 18 -an "$f"
done
[ -s $H/video15/soak_15fps.mp4 ] || step ffmpeg_soak ffmpeg -v error -y -i $O/edge/soak_input.mp4 -vf fps=15 -c:v libx264 -crf 18 -an $H/video15/soak_15fps.mp4

# A: SCB poses with s (and with pose_dedup off, to measure its share as in Handoff 02 E4)
[ -e $H/poses/scb_s.done ] || { step scb_s .venv/bin/python $X $S $H/poses/scb_s.jsonl $SCB > $H/logs/scb_s.out 2>&1 && touch $H/poses/scb_s.done; }
[ -e $H/poses/scb_s_tiles.done ] || { step scb_s_tiles .venv/bin/python $X $S $H/poses/scb_s_tiles.jsonl $SCB --tiles auto > $H/logs/scb_s_tiles.out 2>&1 && touch $H/poses/scb_s_tiles.done; }
[ -e $H/poses/scb_s_nodedup.done ] || { step scb_s_nodedup .venv/bin/python $XN $S $H/poses/scb_s_nodedup.jsonl $SCB > $H/logs/scb_s_nodedup.out 2>&1 && touch $H/poses/scb_s_nodedup.done; }
[ -e $H/poses/scb_s_tiles_nodedup.done ] || { step scb_s_tiles_nodedup .venv/bin/python $XN $S $H/poses/scb_s_tiles_nodedup.jsonl $SCB --tiles auto > $H/logs/scb_s_tiles_nodedup.out 2>&1 && touch $H/poses/scb_s_tiles_nodedup.done; }

# B: the clips with s, camera positions as in Handoff 02
[ -s $H/clips/clips_report.md ] || step clips env MODELS=s research/clips/run_clips.sh $H/clips $C/clip.mp4:auto $C/clip_2.mp4:center $C/clip_3.mp4:center $C/clip_5_720p.mp4:center > $H/logs/clips.out 2>&1

# C: clip_4 with s (rows, offset and calib as in Handoff 02)
[ -s $H/staged/summary.md ] || step staged env OUT=$H/staged research/staged/run_all.sh $C/clip_4_720p.mp4 research/staged/script_clip4.csv --rows 2 --calib 58 --models s > $H/logs/staged.out 2>&1
echo "[$(date -Is)] OFFLINE_DONE" >> $L

# E1: s with tiles in real time on the 15 FPS copies (the Pi otherwise idle)
for spec in clip:auto clip_3:center clip_5_720p:center; do n=${spec%:*}; pos=${spec##*:}
  SK=$H/fps15/${n}_s_auto_15fps.skeleton.jsonl
  [ -s $SK ] && continue
  sleep 30; mon realtime_${n}_s_auto_15fps
  step realtime_$n ./hcpose.sh run $S --input $H/video15/${n}_15fps.mp4 --engine gst --headless --sync --tiles auto \
    --score-threshold 0.3 --engagement --camera-position $pos --save-json $SK > $H/fps15/${n}_s_auto_15fps.log 2>&1
  unmon
done

# E3: latency, s with tiles on the 15 FPS lecture copy, as research/edge/bench.sh measures it
if [ ! -s $H/edge15/latency_s_auto.trace ]; then
  sleep 30
  step latency env HCPOSE_PACE_FILE=1 GST_SHARK_LOCATION=$H/edge15/gstshark_ctf_s_auto GST_TRACERS=interlatency GST_DEBUG=GST_TRACER:7 \
    GST_DEBUG_FILE=$H/edge15/latency_s_auto.trace timeout -k 10 -s INT 70 ./hcpose.sh run $S --input $H/video15/clip_3_15fps.mp4 \
    --engine gst --headless --engagement --score-threshold 0.3 --camera-position center --no-sync --tiles auto > $H/edge15/latency_s_auto.log 2>&1
fi

# E4: 15-minute soak, s with tiles in real time on the 15 FPS soak source
if [ ! -s $H/edge15/soak_s_auto_15fps.csv ]; then
  sleep 60; mon soak_s_auto_15fps
  step soak timeout -k 30 -s INT 900 ./hcpose.sh run $S --input $H/video15/soak_15fps.mp4 --engine gst --headless --engagement \
    --score-threshold 0.3 --camera-position center --sync --tiles auto > $H/edge15/soak_s_auto_15fps.log 2>&1
  unmon
fi
echo "[$(date -Is)] QUEUE_DONE" >> $L
