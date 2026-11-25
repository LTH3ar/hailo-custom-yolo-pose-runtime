#!/usr/bin/env bash
cd ~/hailo-custom-yolo-pose; P=.venv_cls/bin/python; O=~/hcpose_paper; H=$O/h04
S="$O/poses/scb_m.jsonl $O/poses/scb_m_tiles.jsonl"; C="$O/poses/cla_m.jsonl $O/poses/cla_m_tiles.jsonl"
for t in 0.85 0.40; do
  D=$H/corr$t; mkdir -p $D
  $P -m classifiers.prepare_clabehavior ~/datasets/clabehavior ~/datasets/clabehavior_yolo --min-corr $t | tee $D/prepare.txt
  cp ~/datasets/clabehavior_yolo/groups.json $D/groups.json
  for m in strict lenient; do
    $P -m classifiers.train_all $S --clabehavior $C --mapping $m --cross --det-thr 0.3 \
       --report $D/both_$m.json --name both-$m-corr$t | tee $D/both_$m.txt
  done
done
echo H04_DONE
