#!/usr/bin/env bash
# Run hcpose on a staged-session video with every model / tiling / classifier combination
# and score the results against the session script.
#
# usage: research/staged/run_all.sh VIDEO SCRIPT_CSV --rows 3,4,3 [--offset SECONDS] [--models "n m"]
# env:   MODELS_DIR (default ~/models: upper_body_<m>.hef), OUT (default <video dir>/results)
set -euo pipefail
cd "$(dirname "$0")/../.."
[ $# -ge 2 ] || { sed -n 2,6p "$0"; exit 2; }
VIDEO=$1; SCRIPT=$2; shift 2
ROWS=""; OFFSET="auto"; MODELS="n m"
while [ $# -gt 0 ]; do
  case $1 in
    --rows) ROWS=$2; shift 2;;
    --offset) OFFSET=$2; shift 2;;
    --models) MODELS=$2; shift 2;;
    *) echo "unknown option $1"; exit 2;;
  esac
done
[ -n "$ROWS" ] || { echo "--rows is required (seats per row, front to back, e.g. 3,4,3)"; exit 2; }
MODELS_DIR=${MODELS_DIR:-$HOME/models}
OUT=${OUT:-$(dirname "$VIDEO")/results}
mkdir -p "$OUT"
NAME=$(basename "$VIDEO" .mp4)
PY=.venv/bin/python

for m in $MODELS; do
  for t in none auto; do
    TILES=(); [ "$t" = auto ] && TILES=(--tiles auto)
    SK="$OUT/${NAME}_${m}_${t}.skeleton.jsonl"
    if [ ! -s "$SK" ]; then
      echo "== hcpose: model $m, tiles $t"
      ./hcpose.sh run "$MODELS_DIR/upper_body_$m.hef" --input "$VIDEO" --engine gst --headless --no-sync \
        --score-threshold 0.3 "${TILES[@]}" --save-json "$SK" 2>&1 | grep -E "fps|FPS|frames|error|Error" || true
    fi
    for c in trees logistic mlp none; do
      ./hcpose.sh analyze "$SK" --classifier "$c" --camera-position center \
        -o "$OUT/${NAME}_${m}_${t}_${c}.jsonl" > /dev/null
    done
  done
done

$PY research/staged/score.py "$OUT"/"${NAME}"_*_*_*.jsonl --script "$SCRIPT" --rows "$ROWS" \
  --offset "$OFFSET" --out "$OUT/summary.md"
echo "summary written to $OUT/summary.md"
