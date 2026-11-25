#!/usr/bin/env bash
cd ~/hailo-custom-yolo-pose; O=~/hcpose_paper; H=$O/h05
P=.venv_cls/bin/python; S="$O/poses/scb_m.jsonl $O/poses/scb_m_tiles.jsonl"; C="$O/poses/cla_m.jsonl $O/poses/cla_m_tiles.jsonl"
$P -m classifiers.train_all $S --det-thr 0.3 --folds $H/split.json --report $H/cls_thesis.json > $H/cls_thesis.txt 2>&1
for m in strict lenient; do
  $P -m classifiers.train_all $S --clabehavior $C --mapping $m --cross --det-thr 0.3 --folds $H/split.json \
     --report $H/cls_$m.json > $H/cls_$m.txt 2>&1
done
echo H05A_DONE
