#!/usr/bin/env bash
cd ~/hailo-custom-yolo-pose
P=.venv_cls/bin/python; O=~/hcpose_paper; H=$O/h06; mkdir -p $H; SP=$O/h05/split.json
S="$O/poses/scb_m.jsonl $O/poses/scb_m_tiles.jsonl"; C="$O/poses/cla_m.jsonl $O/poses/cla_m_tiles.jsonl"
$P -m classifiers.train_all $S --det-thr 0.3 --folds $SP --save-cv $H/cv_thesis.npz > $H/cls_thesis.txt 2>&1
for m in strict lenient; do
  $P -m classifiers.train_all $S --clabehavior $C --mapping $m --det-thr 0.3 --folds $SP --save-cv $H/cv_$m.npz \
     > $H/cls_$m.txt 2>&1
done
$P -m baselines.crop_cls.paired $SP $O/h05/preds.jsonl \
   --cls thesis=$H/cv_thesis.npz strict=$H/cv_strict.npz lenient=$H/cv_lenient.npz --report $H/paired.json \
   > $H/paired.txt 2>&1
echo "H06_DONE rc=$?"
