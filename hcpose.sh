#!/usr/bin/env bash
# Run hcpose from the repo without activating the venv.
DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
if [ ! -x "$DIR/.venv/bin/hcpose" ]; then
  echo "hcpose is not installed yet; run $DIR/install.sh first" >&2
  exit 1
fi
exec "$DIR/.venv/bin/hcpose" "$@"
