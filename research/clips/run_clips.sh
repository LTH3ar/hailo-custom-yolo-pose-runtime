#!/usr/bin/env bash
# Run hcpose on classroom clips with n and m, tiles off and on, then replay every skeleton
# file with each classifier and summarise (class distracted %, its range, class size, ...).
#
# usage: research/clips/run_clips.sh OUT_DIR CLIP.mp4:CAMERA_POSITION [CLIP.mp4:CAMERA_POSITION ...]
#   CAMERA_POSITION = auto | center | left | right
# env: MODELS_DIR (default ~/models), MODELS (default "n m")
set -uo pipefail
cd "$(dirname "$0")/../.."
[ $# -ge 2 ] || { sed -n 2,8p "$0"; exit 2; }
OUT=$1; shift
MODELS_DIR=${MODELS_DIR:-$HOME/models}; MODELS=${MODELS:-"n m"}
mkdir -p "$OUT"
for spec in "$@"; do
  CLIP=${spec%:*}; POS=${spec##*:}; NAME=$(basename "$CLIP" .mp4)
  for m in $MODELS; do for t in none auto; do
    T=(); [ "$t" = auto ] && T=(--tiles auto)
    SK="$OUT/${NAME}_${m}_${t}.skeleton.jsonl"
    if [ ! -s "$SK" ]; then
      echo "== $NAME: model $m, tiles $t"
      ./hcpose.sh run "$MODELS_DIR/upper_body_$m.hef" --input "$CLIP" --engine gst --headless --no-sync \
        --score-threshold 0.3 "${T[@]}" --save-json "$SK" > "$OUT/${NAME}_${m}_${t}.log" 2>&1
      tail -n 1 "$OUT/${NAME}_${m}_${t}.log"
    fi
    for c in none trees logistic mlp; do
      ./hcpose.sh analyze "$SK" --classifier "$c" --camera-position "$POS" -o "$OUT/${NAME}_${m}_${t}_${c}.jsonl" > /dev/null
    done
  done; done
done
.venv/bin/python research/clips/summarize.py "$OUT" | tee "$OUT/clips_report.md"
