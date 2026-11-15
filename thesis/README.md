# Thesis, LaTeX version (HUST template ĐT.QT12.BM01)

The thesis in LaTeX, in the layout of HUST's master's thesis template (ĐT.QT12.BM01, Lần ban hành 02,
30/6/2025), in two languages:

- `en/`: the English thesis (the original)
- `vn/`: the Vietnamese thesis (translated from `en/`, with the template's Vietnamese labels)

Each folder is a complete, self-contained LaTeX project with its own chapters and figures. The `.tex`
files are the master copy of the text: edit `en/chapters/` and `vn/chapters/` directly. A change to one
language is not copied to the other automatically. (The text was
first drafted in a Claude doc and converted once; the converter and that draft are kept in
`tools/archive/` for reference only and are no longer used.)

## Build

```bash
en/build.sh                     # compile en/main.pdf
vn/build.sh                     # compile vn/main.pdf
tools/package.sh dist           # both PDFs and both Overleaf zips into dist/
```

It compiles with **pdfLaTeX**, Overleaf's default, on the same base as HUST's own LaTeX template:
`inputenc` utf8, the T5 font encoding (vntex) for Vietnamese letters and `mathptmx` for Times.
XeLaTeX and LuaLaTeX work too (they switch to TeX Gyre Termes).

On Overleaf: *New project → Upload project*, upload `thesis_en_overleaf.zip` or `thesis_vn_overleaf.zip` (or a zip of `en/` or `vn/`),
and leave the compiler on pdfLaTeX. The main document is `main.tex`.

## Releases

`.github/workflows/release.yml` runs on a version tag (`git tag -a v1.0-thesis -m ... && git push origin v1.0-thesis`).
It runs the tests and builds the runtime as a Python sdist and wheel. It also compiles this thesis with
`tools/package.sh`, then attaches `thesis_en.pdf`, `thesis_vn.pdf` and their two Overleaf zips to the GitHub release. On every push and
pull request, `ci.yml` runs the same tests and thesis build and keeps the PDFs as a build artifact.

## Files

Inside `en/` and `vn/`:

| File | What |
|---|---|
| `chapters/ch1.tex` … `ch5.tex` | the chapters |
| `chapters/abstract.tex`, `chapters/references.tex` | the abstract and the IEEE reference list |
| `info.tex`, `acknowledgements.tex` | personal details and acknowledgements |
| `frontmatter.tex`, `main.tex`, `hust-thesis.sty` | cover, front pages, layout |
| `figures/` | figures; `figures/manual/` holds the ones made by hand |

In `thesis/`: `tools/package.sh` builds both PDFs and Overleaf zips (used by CI and the release
workflow); `tools/archive/` holds the original doc export and converter, no longer used.

## Language

`en/info.tex` sets `\vietnamesefalse` (CHAPTER, Figure, Table, CONTENTS); `vn/info.tex` sets
`\vietnamesetrue`, which switches every label, the cover and the forms to the template's Vietnamese
(CHƯƠNG, Hình, Bảng, MỤC LỤC, equations as PT n.m).

The Vietnamese version keeps decimal points (0.45, 4.3 W) as in the English text, the figures and the
code, and keeps the references in English as published. The diagrams and charts in `vn/figures/` are
still the English ones; replace them under the same file names if Vietnamese labels are wanted.

## Template rules applied

- A4; margins top 2 cm, bottom 2 cm, left 3.5 cm, right 2.5 cm; 13 pt Times; justified; line spacing 1.1
- chapters "CHAPTER n. TITLE" (CHƯƠNG) in capitals on a new page; at most three heading levels
- captions: figures below (Figure/Hình n.m), tables above (Table/Bảng n.m); equations (n.m), or PT n.m in Vietnamese
- order: cover → (Bản xác nhận chỉnh sửa) → Đề tài → Lời cảm ơn + Tóm tắt (one signed page) → Mục lục →
  Danh mục hình vẽ → Danh mục bảng biểu → chapters → (Phụ lục) → Tài liệu tham khảo
- IEEE references numbered in order of citation
- the form code ĐT.QT12.BM01 in the footer of the front pages, page numbers bottom right in the chapters
- short tables kept on one page; the one long table (legal provisions) may break, with a "(continued)" header

## To do by hand

**Pictures** (drop the file in `en/figures/manual/` and `vn/figures/manual/` under the name given; until then the PDF shows a red placeholder box):

| File | What | How (full instructions in the doc) |
|---|---|---|
| `device.png` | photo of the Pi 5, AI HAT+, cooler and camera | take a photo |
| `tiles.png` | exam-hall frame without and with tiles, the three tile rectangles drawn on | two runs with `-o`, take one frame each, draw x 0–493 / 394–887 / 788–1280, y 0–432 |
| `displays.png` | per-student view and teacher's view side by side | screenshots on the Pi (`grim`), clip_4 |
| `dedup.png` | one student with two boxes, then one | runs with `--pose-dedup 0` and default, crop the same region |
| `topic_form.pdf` | Đề tài luận văn signed by the supervisor | scan |

**Optional redrawing:** the architecture and data-flow diagrams and the two charts are screen renders
(1344 px wide, about 280 dpi at print size). They print fine. Redraw them only if your
supervisor wants vector figures, e.g. in draw.io exported to PDF with the same file names.

**Text and forms:**
- `info.tex` (in each folder): personal details, major, supervisor, school, year
- `acknowledgements.tex`
- the summary page must fit on one side with the acknowledgements (both versions fit now)
- after the defence: set `\showrevisionformtrue` in `info.tex` and fill the revision form
- signatures on the cover (supervisor) and on the summary page (you)
- the spine text for binding: Khóa – Ngành – Họ và tên – Mã số học viên (template section 1.9)
- the legal article numbers in chapter 4: check them against the official Vietnamese text

**Privacy:** no frame or crop of clip_5 or of any minor goes in the thesis; figures come from
clip_4 (adults who agreed) and the exam-hall clip (adults).
