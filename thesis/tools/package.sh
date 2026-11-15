#!/usr/bin/env bash
# Compile the thesis in each language and package it:
#   <out>/thesis_<lang>.pdf            the compiled thesis (en = English, vn = Vietnamese)
#   <out>/thesis_<lang>_overleaf.zip   everything Overleaf needs (upload as a new project, pdfLaTeX)
# Usage: tools/package.sh [out_dir] [lang ...]     (default: dist, en vn)
set -euo pipefail
OUT=$(mkdir -p "${1:-dist}" && cd "${1:-dist}" && pwd)
shift || true
LANGS=("${@:-en vn}")
ROOT=$(cd "$(dirname "$0")/.." && pwd)
for lang in ${LANGS[@]}; do
    cd "$ROOT/$lang"
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null
    cp main.pdf "$OUT/thesis_$lang.pdf"
    rm -f "$OUT/thesis_${lang}_overleaf.zip"
    zip -q -r "$OUT/thesis_${lang}_overleaf.zip" main.tex hust-thesis.sty info.tex frontmatter.tex \
        acknowledgements.tex chapters figures -x "*.DS_Store"
    (cd "$ROOT" && zip -q "$OUT/thesis_${lang}_overleaf.zip" README.md)
    echo "wrote $OUT/thesis_$lang.pdf and $OUT/thesis_${lang}_overleaf.zip"
done
