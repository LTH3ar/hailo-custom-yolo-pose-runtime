#!/usr/bin/env bash
# Re-render the chapters from src/*.md (Claude doc exports) and compile the PDF.
set -euo pipefail
cd "$(dirname "$0")"
python3 tools/doc2tex.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null
echo "built $(pwd)/main.pdf"
