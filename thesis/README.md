# Thesis, LaTeX version (HUST template ĐT.QT12.BM01)

This is a LaTeX rendering of the thesis doc in Claude in the layout of HUST's master's thesis
template (ĐT.QT12.BM01, Lần ban hành 02, 30/6/2025). The doc is the source of truth: the chapters
in `generated/` are produced from it and are overwritten on every update.

## Build

```bash
./build.sh                      # re-render generated/ from src/*.md, then compile main.pdf
latexmk -xelatex main.tex       # compile only
```

On Overleaf: upload this folder, set *Menu → Compiler → XeLaTeX*, main document `main.tex`.
Times New Roman is used if installed, otherwise its metric clone TeX Gyre Termes (on Overleaf by default).

## What is generated and what is yours

| File | Who edits it |
|---|---|
| `src/*.md` | exported from the Claude doc, one file per tab |
| `generated/*.tex` | made by `tools/doc2tex.py` from `src/`; don't edit, changes are lost |
| `figures/*.png` | renders of the doc's diagrams and charts, and the clip_4 figure |
| `tools/captions.json` | table and figure captions that the doc doesn't hold |
| `info.tex` | **you**: name, email, student ID, major, supervisor, school, year |
| `acknowledgements.tex` | **you**: Lời cảm ơn, 100–150 words |
| `figures/manual/` | **you**: the pictures listed below |

How updates work: when the doc changes, Claude exports the changed tabs into `src/`, runs
`./build.sh`, checks the PDF and commits. It's a step Claude does with each edit, not an automatic sync.
If you edit the doc yourself, ask Claude to refresh the LaTeX version.

## Language

Labels are in English by default (CHAPTER, Figure, Table, CONTENTS, the cover and forms). Set
`\vietnamesetrue` in `info.tex` to switch every label to the template's Vietnamese (CHƯƠNG, Hình, Bảng,
MỤC LỤC, equations as PT n.m). The body text comes from the doc, in whatever language it is written.

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

**Pictures** (drop the file in `figures/manual/` under the name given; until then the PDF shows a red placeholder box):

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
- `info.tex`: personal details, major, supervisor, school, year
- `acknowledgements.tex`
- the summary page must fit on one side with the acknowledgements. A Vietnamese summary may be required; it
  goes in the doc's Abstract tab
- after the defence: set `\showrevisionformtrue` in `info.tex` and fill the revision form
- signatures on the cover (supervisor) and on the summary page (you)
- the spine text for binding: Khóa – Ngành – Họ và tên – Mã số học viên (template section 1.9)
- the legal article numbers in chapter 4: check them against the official Vietnamese text

**Privacy:** no frame or crop of clip_5 or of any minor goes in the thesis; figures come from
clip_4 (adults who agreed) and the exam-hall clip (adults).
