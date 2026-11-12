#!/usr/bin/env bash
# Compile the thesis from the committed generated/ chapters (no pandoc needed) and package:
#   <out>/thesis.pdf             the compiled thesis
#   <out>/thesis_overleaf.zip    everything Overleaf needs (upload as a new project, pdfLaTeX)
set -euo pipefail
OUT=$(mkdir -p "${1:-dist}" && cd "${1:-dist}" && pwd)
cd "$(dirname "$0")/.."
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null
cp main.pdf "$OUT/thesis.pdf"
rm -f "$OUT/thesis_overleaf.zip"
zip -q -r "$OUT/thesis_overleaf.zip" main.tex hust-thesis.sty info.tex frontmatter.tex acknowledgements.tex \
    generated figures README.md -x "*.DS_Store"
echo "wrote $OUT/thesis.pdf and $OUT/thesis_overleaf.zip"
