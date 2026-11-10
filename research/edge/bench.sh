#!/usr/bin/env bash
# Edge measurements on the Pi 5 + Hailo-8: idle baseline, Hailo-only benchmark, end-to-end
# throughput with load/temperature/board power, pipeline latency, and two 15-minute soaks.
#
# usage: research/edge/bench.sh CLIP.mp4 SOAK_SOURCE.mp4 [OUT_DIR]
#   CLIP.mp4         a classroom clip of a few minutes (e.g. ~/classroom/clip_3.mp4)
#   SOAK_SOURCE.mp4  a clip that is looped to 18 min for the soaks (e.g. ~/classroom/clip_2.mp4)
# env: MODELS_DIR (default ~/models, upper_body_{n,s,m}.hef), MODELS (default "n s m"),
#      FROM=latency to resume at the latency runs
set -uo pipefail
cd "$(dirname "$0")/../.."
[ $# -ge 2 ] || { sed -n 2,9p "$0"; exit 2; }
CLIP=$1; SOAK_SRC=$2; OUT=${3:-$HOME/edge_results}
MODELS_DIR=${MODELS_DIR:-$HOME/models}; MODELS=${MODELS:-"n s m"}
PY=.venv/bin/python
mkdir -p "$OUT"
$PY -c "import psutil" 2>/dev/null || .venv/bin/pip install -q psutil

mon() { $PY research/edge/monitor.py "$OUT/$1.csv" & MON=$!; }
unmon() { kill -INT $MON 2>/dev/null; wait $MON 2>/dev/null; }
run() { ./hcpose.sh run "$@" --engine gst --headless --engagement --score-threshold 0.3 --camera-position center; }

FROM=${FROM:-start}  # FROM=latency resumes at the latency runs
if [ "$FROM" = start ]; then
echo "== system"
{ cat /proc/device-tree/model; echo; uname -a; vcgencmd version; hailortcli fw-control identify;
  echo "cooling state: $(cat /sys/class/thermal/cooling_device0/cur_state 2>/dev/null) of $(cat /sys/class/thermal/cooling_device0/max_state 2>/dev/null)";
  free -m; } > "$OUT/system.txt" 2>&1

echo "== idle, 60 s"
$PY research/edge/monitor.py "$OUT/idle.csv" --duration 60

for m in $MODELS; do
  echo "== Hailo-only benchmark: $m"
  hailortcli benchmark "$MODELS_DIR/upper_body_$m.hef" --time-to-run 15 > "$OUT/hailo_benchmark_$m.txt" 2>&1 \
    || hailortcli run "$MODELS_DIR/upper_body_$m.hef" --time-to-run 15 --measure-latency --measure-overall-latency \
       > "$OUT/hailo_benchmark_$m.txt" 2>&1
done

for m in $MODELS; do for t in none auto; do
  T=(); [ "$t" = auto ] && T=(--tiles auto)
  echo "== throughput: $m, tiles $t"
  mon "throughput_${m}_$t"
  run "$MODELS_DIR/upper_body_$m.hef" --input "$CLIP" --no-sync "${T[@]}" > "$OUT/throughput_${m}_$t.log" 2>&1
  unmon
  sleep 20  # let the temperature settle between runs
done; done
fi

for m in n m; do for t in none auto; do
  T=(); [ "$t" = auto ] && T=(--tiles auto)
  echo "== latency (clip paced at its frame rate like a camera, 60 s): $m, tiles $t"
  # GStreamer's core "latency" tracer deadlocks this pipeline (GStreamer 1.26.2 + Hailo elements),
  # so GstShark's interlatency is used. HCPOSE_PACE_FILE=1 releases decoded frames at the clip's
  # frame rate (identity live_pace sync=true); report.py measures live_pace -> sink.
  HCPOSE_PACE_FILE=1 GST_SHARK_LOCATION="$OUT/gstshark_ctf_${m}_$t" GST_TRACERS="interlatency" GST_DEBUG="GST_TRACER:7" GST_DEBUG_FILE="$OUT/latency_${m}_$t.trace" \
    timeout -k 10 -s INT 70 ./hcpose.sh run "$MODELS_DIR/upper_body_$m.hef" --input "$CLIP" --engine gst --headless \
    --engagement --score-threshold 0.3 --camera-position center --no-sync "${T[@]}" > "$OUT/latency_${m}_$t.log" 2>&1
  sleep 10
done; done

echo "== soak source: $SOAK_SRC looped to 18 min"
SOAK="$OUT/soak_input.mp4"
if [ ! -s "$SOAK" ]; then
  printf "file '%s'\n" "$(realpath "$SOAK_SRC")" "$(realpath "$SOAK_SRC")" "$(realpath "$SOAK_SRC")" > "$OUT/soak_list.txt"
  ffmpeg -v error -y -f concat -safe 0 -i "$OUT/soak_list.txt" -c copy "$SOAK"
fi
echo "== soak 1, 15 min: n + tiles in real time (the deployment setting)"
sleep 60; mon soak_n_auto_realtime
timeout -k 30 -s INT 900 ./hcpose.sh run "$MODELS_DIR/upper_body_n.hef" --input "$SOAK" --engine gst --headless --engagement \
  --score-threshold 0.3 --camera-position center --sync --tiles auto > "$OUT/soak_n_auto_realtime.log" 2>&1
unmon
echo "== soak 2, 15 min: m + tiles as fast as possible (worst case)"
sleep 120; mon soak_m_auto_max
timeout -k 30 -s INT 900 ./hcpose.sh run "$MODELS_DIR/upper_body_m.hef" --input "$SOAK" --engine gst --headless --engagement \
  --score-threshold 0.3 --camera-position center --no-sync --tiles auto > "$OUT/soak_m_auto_max.log" 2>&1
unmon

$PY research/edge/report.py "$OUT" | tee "$OUT/edge_report.md"
