#!/usr/bin/env bash
# Handoff 05 Part E: crop baseline on the clips (boxes and seats of hcpose's rules runs, tiles on)
cd ~/hcpose/hailo-custom-yolo-pose
R=~/hcpose/crop_cls; P=~/hcpose/venvs/cropcls/bin/python; V=~/hcpose/classroom; W=$R/runs/all/weights/last.pt
mkdir -p $R/clips_out
for spec in clip_3:$V/clip_3.mp4:clip_3 clip:$V/clip.mp4:clip clip_5:$V/clip_5_720p.mp4:clip_5_720p clip_4:$R/videos/clip_4_720p.mp4:clip_4_720p; do
  IFS=: read name vid stem <<< "$spec"
  for m in n m; do
    for map in strict lenient; do
      out=$R/clips_out/${name}_${m}_tiles_${map}
      [ -f ${out}_crop_smooth.jsonl ] && continue
      s=$(date +%s)
      $P -m baselines.crop_cls.clips $W $vid $R/clips_in/${stem}_${m}_auto_none.jsonl -o $out --mapping $map > ${out}.log 2>&1
      echo "$(date -Is) ${name}_${m}_${map} rc=$? $(( $(date +%s) - s )) s" >> $R/clips_out/times.txt
    done
  done
done
echo H05E_DONE >> $R/clips_out/times.txt
