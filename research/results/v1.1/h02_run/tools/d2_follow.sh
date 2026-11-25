#!/usr/bin/env bash
# replay each clip skeleton with the nine classifiers as soon as run_clips.sh finishes it
D=~/hcpose_paper/clips; T=~/hcpose_paper/tools
while :; do
  nice -n 10 $T/d2_replay.sh $D clip:auto clip_2:center clip_3:center clip_5_720p:center 2>&1 | grep -v "^\[hcpose\]"
  [ -s $D/clips_report.md ] && { nice -n 10 $T/d2_replay.sh $D clip:auto clip_2:center clip_3:center clip_5_720p:center 2>&1 | grep -v "^\[hcpose\]"; echo D2_FOLLOW_DONE; break; }
  sleep 120
done
