#!/usr/bin/env bash
# Handoff 07 (uncommitted copy of Handoff 02 d2_replay.sh; models from $MODELS, default s): replay skeletons with the nine retrained classifiers.
# usage: d2_replay.sh DIR NAME:POS [NAME:POS ...]   (skeletons DIR/NAME_<m>_<t>.skeleton.jsonl)
set -uo pipefail
cd ~/hailo-custom-yolo-pose-runtime
D=$1; shift
for spec in "$@"; do NAME=${spec%:*}; POS=${spec##*:}
  for m in ${MODELS:-s}; do for t in none auto; do
    SK=$D/${NAME}_${m}_${t}.skeleton.jsonl
    [ -s "$SK" ] || { echo "missing $SK"; continue; }
    # a skeleton is complete once its run has finished: run_clips.sh writes the mlp analysis last, run_all.sh the log
    [ -s "${SK%.skeleton.jsonl}_mlp.jsonl" ] || [ -n "${STAGED:-}" ] || { echo "not finished $SK"; continue; }
    for set in thesis strict lenient; do for k in trees logistic mlp; do
      OUTF=$D/${NAME}_${m}_${t}_${set}_${k}.jsonl
      [ -s "$OUTF" ] && continue
      ./hcpose.sh analyze "$SK" --classifier ~/hcpose_paper/models/$set/scb_$k.json --camera-position "$POS" -o "$OUTF" > /dev/null \
        || echo "FAILED $OUTF"
    done; done
    echo "done $NAME $m $t"
  done; done
done
