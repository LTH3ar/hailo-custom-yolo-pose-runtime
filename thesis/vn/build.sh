#!/usr/bin/env bash
# Compile the thesis PDF (pdfLaTeX). The .tex files in chapters/ are the master copy.
set -euo pipefail
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null
echo "built $(pwd)/main.pdf"
