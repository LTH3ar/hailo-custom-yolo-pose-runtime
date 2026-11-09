#!/usr/bin/env bash
# hcpose installer for Raspberry Pi 5 + Hailo-8/8L (Raspberry Pi OS Bookworm or newer).
#
#   ./install.sh                 full install (apt build deps, C++ post-process, venv)
#   ./install.sh --build-only    rebuild only the C++ post-process
#   ./install.sh --recreate-venv delete and recreate .venv
#   ./install.sh --no-apt        never call apt (print what is missing instead)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
ROOT="$(pwd)"

BUILD_ONLY=0; RECREATE=0; NO_APT=0
for a in "$@"; do
  case "$a" in
    --build-only) BUILD_ONLY=1 ;;
    --recreate-venv) RECREATE=1 ;;
    --no-apt) NO_APT=1 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$*"; }

# --------------------------------------------------------------------------
say "Checking Hailo runtime"
missing_hailo=0
if command -v hailortcli >/dev/null; then
  ok "hailortcli $(hailortcli --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
else
  fail "hailortcli not found"; missing_hailo=1
fi
for el in hailonet hailofilter hailocropper hailoaggregator hailotracker hailooverlay; do
  if gst-inspect-1.0 "$el" >/dev/null 2>&1; then :; else fail "GStreamer element $el missing"; missing_hailo=1; fi
done
TAPPAS_PKG=""
for p in hailo-tappas-core hailo-apps-core hailo_tappas; do
  if pkg-config --exists "$p" 2>/dev/null; then TAPPAS_PKG="$p"; break; fi
done
if [ -n "$TAPPAS_PKG" ]; then ok "$TAPPAS_PKG $(pkg-config --modversion "$TAPPAS_PKG")"; else fail "TAPPAS core (pkg-config) not found"; missing_hailo=1; fi
if [ "$missing_hailo" = 1 ]; then
  echo
  echo "The Hailo stack is not (fully) installed. On Raspberry Pi OS:"
  echo "    sudo apt update && sudo apt install hailo-all && sudo reboot"
  echo "then re-run ./install.sh"
  exit 1
fi

# --------------------------------------------------------------------------
say "Checking build/runtime packages"
need=()
for pkg in meson ninja-build g++ pkg-config python3-venv python3-gi python3-opencv gstreamer1.0-x; do
  dpkg -s "$pkg" >/dev/null 2>&1 || need+=("$pkg")
done
RAPIDJSON_DIR="$(pkg-config --variable=rapidjson_includedir "$TAPPAS_PKG" 2>/dev/null || true)"
if [ -z "$RAPIDJSON_DIR" ] || [ ! -f "$RAPIDJSON_DIR/rapidjson/document.h" ]; then
  if [ ! -f /usr/include/rapidjson/document.h ]; then need+=("rapidjson-dev"); fi
  RAPIDJSON_DIR=""
fi
if [ ${#need[@]} -gt 0 ]; then
  if [ "$NO_APT" = 1 ]; then
    fail "missing: ${need[*]}  (sudo apt install ${need[*]})"; exit 1
  fi
  say "Installing: ${need[*]}"
  sudo apt-get install -y "${need[@]}"
else
  ok "all present"
fi

# --------------------------------------------------------------------------
say "Building C++ post-process"
MESON_ARGS=()
[ -n "$RAPIDJSON_DIR" ] && MESON_ARGS+=("-Drapidjson_include=$RAPIDJSON_DIR")
if [ -d build ] && [ -f build/build.ninja ]; then
  meson setup --reconfigure build cpp "${MESON_ARGS[@]}" >/dev/null
else
  rm -rf build
  meson setup build cpp "${MESON_ARGS[@]}" >/dev/null
fi
ninja -C build
mkdir -p hcpose/lib
cp build/libhcpose_yolov8_pose.so hcpose/lib/
ok "hcpose/lib/libhcpose_yolov8_pose.so"
[ "$BUILD_ONLY" = 1 ] && exit 0

# --------------------------------------------------------------------------
say "Python environment (.venv, with system packages for HailoRT/GStreamer)"
if [ "$RECREATE" = 1 ]; then rm -rf .venv; fi
if [ -d .venv ] && ! grep -q "include-system-site-packages = true" .venv/pyvenv.cfg 2>/dev/null; then
  fail ".venv exists without --system-site-packages; recreating"; rm -rf .venv
fi
[ -d .venv ] || python3 -m venv --system-site-packages .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -e .
ok "hcpose $(.venv/bin/python -c 'import hcpose; print(hcpose.__version__)') installed"

# --------------------------------------------------------------------------
say "Checking installation"
.venv/bin/hcpose doctor || true
cat <<EOF

Done. Run without activating the venv:
    ./hcpose.sh inspect models/your_model.hef
    ./hcpose.sh run models/your_model.hef --input usb
    ./hcpose.sh run models/your_model.hef --input classroom.mp4 --output annotated.mp4 --save-json skeletons.jsonl
EOF
