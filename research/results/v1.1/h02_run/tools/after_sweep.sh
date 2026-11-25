#!/usr/bin/env bash
# Handoff 02: E2 sweep (work folder on the SSD; /tmp is a 4 GB tmpfs), then the E3 n-untiled repeat
# with the Pi otherwise idle: new vs backed-up old n HEF, twice each, same command as research/edge/bench.sh.
O=~/hcpose_paper; R=~/hailo-custom-yolo-pose-runtime; L=$O/logs/after_sweep.log
cd $R
for g in current offset yaw; do
  echo "[$(date -Is)] sweep $g" >> $L
  nice .venv/bin/python $O/tools/sweep_rules_h02.py --grid $g --jobs 3 --work $O/sweep_work --out $O/sweep/sweep_$g.md > $O/sweep/sweep_$g.log 2>&1
  echo "[$(date -Is)] sweep $g rc=$?" >> $L
done
echo "[$(date -Is)] SWEEP_DONE" >> $L
mkdir -p $O/edge_repeat
sleep 60
for i in 1 2; do
  for h in new:$HOME/models/upper_body_n.hef old:$HOME/models/thesis_v1.0.3/upper_body_n.hef; do
    tag=${h%%:*}; f=${h#*:}
    .venv/bin/python research/edge/monitor.py $O/edge_repeat/n_none_${tag}_$i.csv &
    M=$!
    ./hcpose.sh run "$f" --engine gst --headless --engagement --score-threshold 0.3 --camera-position center \
      --input ~/classroom/clip_3.mp4 --no-sync > $O/edge_repeat/n_none_${tag}_$i.log 2>&1
    kill -INT $M; wait $M
    echo "[$(date -Is)] edge_repeat ${tag}_$i: $(grep -o '[0-9.]* FPS avg' $O/edge_repeat/n_none_${tag}_$i.log)" >> $L
    sleep 20
  done
done
echo "[$(date -Is)] ALL_DONE" >> $L
