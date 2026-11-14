"""Convert the thesis doc (one Markdown export per tab, in thesis/src/) into LaTeX in thesis/generated/.

The Claude doc is the source of truth; this script only re-renders it in the HUST thesis layout.
It handles what a plain pandoc run would get wrong for the template:
  - chapter and section headings without the doc's typed numbers (LaTeX numbers them)
  - tables as captioned, page-breakable xltabular ("Bảng"), captions from tools/captions.json
  - embedded widgets and pictures as captioned figures ("Hình") from thesis/figures/
  - "[Figure to add, manual]" notes as figure placeholders: the instructions are dropped, the
    suggested caption is kept, and figures/manual/<name>.(png|jpg|pdf) is used once it exists
  - IEEE citations [n], [a]-[b] as \\cite, and the References tab as \\bibitem entries
  - ```latex blocks as numbered equations
Usage: python3 tools/doc2tex.py            (run from thesis/)
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "src", ROOT / "generated"
CFG = json.loads((ROOT / "tools" / "captions.json").read_text(encoding="utf8"))
WARN = []


FIXUPS = {"p\u0303": "\\ensuremath{\\tilde{p}}"}  # combining characters pdfLaTeX can't set


def pandoc(md: str) -> str:
    for a, b in FIXUPS.items():
        md = md.replace(a, b)
    r = subprocess.run(["pandoc", "-f", "markdown+raw_tex-auto_identifiers+smart", "-t", "latex", "--wrap=preserve"],
                       input=md, capture_output=True, text=True, check=True)
    return r.stdout


def cites(text: str) -> str:
    def rng(m):
        a, b = int(m.group(1)), int(m.group(2))
        return "\\cite{" + ",".join(f"b{i}" for i in range(a, b + 1)) + "}"
    text = re.sub(r"\\\[(\d+)\\\]\s*[–-]\s*\\\[(\d+)\\\]", rng, text)
    text = re.sub(r"\\\[(\d+)\\\]", lambda m: f"\\cite{{b{m.group(1)}}}", text)
    # merge "\cite{a}, \cite{b}" into one citation
    while True:
        new = re.sub(r"\\cite\{([^}]*)\},\s*\\cite\{([^}]*)\}", r"\\cite{\1,\2}", text)
        if new == text:
            return text
        text = new


def unescape(s: str) -> str:
    return re.sub(r"\\(.)", r"\1", s.replace("&#91;", "["))


def tex_escape(s: str) -> str:
    return re.sub(r"([_%&#$])", r"\\\1", s)


def raw(tex: str) -> str:
    return "\n```{=latex}\n" + tex.strip("\n") + "\n```\n"


def latex_label(prefix: str, text: str) -> str:
    return prefix + ":" + re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]


def table_tex(lines):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines]
    head, body = rows[0], rows[2:]
    key = " | ".join(unescape(c) for c in head)
    cap = None
    for k, v in CFG["tables"].items():
        if key.startswith(k):
            cap = v
            break
    if cap is None:
        cap = "TODO caption: " + key[:60]
        WARN.append(f"table without caption: {key}")
    cap = tex_escape(cap)
    n = len(head)
    cells = [c if c else "EMPTYCELL" for r in [head] + body for c in (r + [""] * (n - len(r)))[:n]]
    conv = pandoc("\n\nCELLSEP\n\n".join(cites(c) for c in cells))
    parts = [p.strip().replace("EMPTYCELL", "") for p in conv.split("CELLSEP")]
    if len(parts) != len(cells):
        raise SystemExit(f"cell split failed for table {key}")
    hc, bc = parts[:n], parts[n:]
    size = "\\footnotesize" if n >= 6 else "\\small"
    spec = "*{%d}{>{\\raggedright\\arraybackslash}X}" % n
    hrow = " & ".join("\\textbf{%s}" % c for c in hc) + " \\\\"
    label = latex_label('tab', cap)
    if sum(len(c) for c in bc) < 1200 and len(bc) // n <= 14:
        # short tables stay on one page, as the template asks
        out = [f"\\begin{{table}}[htbp]{size}", f"\\caption{{{cap}}}\\label{{{label}}}",
               f"\\begin{{tabularx}}{{\\textwidth}}{{{spec}}}", "\\toprule", hrow, "\\midrule"]
        for i in range(0, len(bc), n):
            out.append(" & ".join(bc[i:i + n]) + " \\\\")
        out += ["\\bottomrule", "\\end{tabularx}", "\\end{table}"]
        return "\n".join(out)
    out = [f"\\begingroup{size}", f"\\begin{{xltabular}}{{\\textwidth}}{{{spec}}}",
           f"\\caption{{{cap}}}\\label{{{latex_label('tab', cap)}}}\\\\", "\\toprule", hrow, "\\midrule", "\\endfirsthead",
           f"\\multicolumn{{{n}}}{{l}}{{\\emph{{(continued)}}}}\\\\", "\\toprule", hrow, "\\midrule", "\\endhead",
           "\\bottomrule", "\\endlastfoot"]
    for i in range(0, len(bc), n):
        out.append(" & ".join(bc[i:i + n]) + " \\\\")
    out += ["\\end{xltabular}", "\\endgroup"]
    return "\n".join(out)


def figure_tex(file, caption, width):
    return ("\\begin{figure}[htbp]\n\\centering\n"
            f"\\includegraphics[width={width}\\textwidth]{{{file}}}\n"
            f"\\caption{{{caption}}}\\label{{{latex_label('fig', caption)}}}\n\\end{{figure}}")


def manual_figure_tex(title, caption):
    name = CFG["manual_figures"].get(title)
    if not name:
        WARN.append(f"manual figure without a file name: {title}")
        name = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    cap = pandoc(cites(caption)).strip() if caption else title
    return (f"\\manualfigure{{figures/manual/{name}}}{{{title}}}{{{cap}}}{{{latex_label('fig', name)}}}")


def blocks(md: str):
    """Yield ('code', lang, text) | ('table', lines) | ('para', text) blocks."""
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("```"):
            lang = l[3:].strip()
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            yield ("code", lang, "\n".join(lines[i + 1:j]))
            i = j + 1
        elif l.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            yield ("table", lines[i:j])
            i = j
        elif not l.strip():
            i += 1
        else:
            j = i
            while j < len(lines) and lines[j].strip() and not lines[j].startswith("```") and not lines[j].startswith("|"):
                j += 1
            yield ("para", "\n".join(lines[i:j]))
            i = j


def convert_chapter(md: str) -> str:
    out, pending_caption, skipping = [], None, False
    for b in blocks(md):
        if skipping:
            if b[0] == "para" and "Suggested caption:" in b[1]:
                skipping = False
            continue
        if b[0] == "code":
            lang, text = b[1], b[2]
            if lang == "latex":
                out.append(raw("\\begin{equation}\n" + text + "\n\\end{equation}"))
            else:
                out.append(raw("\\begin{codeblock}\n" + text + "\n\\end{codeblock}"))
            continue
        if b[0] == "table":
            out.append(raw(table_tex(b[1])))
            continue
        p = b[1]
        m = re.match(r"^(#+)\s+(.*)$", p)
        if m:
            level, title = len(m.group(1)), re.sub(r"^\d+(\.\d+)*\s+", "", m.group(2).strip())
            cmd = {1: "chapter", 2: "section", 3: "subsection"}.get(level, "subsubsection")
            if level == 1:
                title = title.upper()  # the template sets chapter titles in capitals, in the contents too
            out.append(raw(f"\\{cmd}{{{pandoc(cites(title)).strip()}}}"))
            continue
        m = re.match(r"^\*\*\\\[Figure to add, manual\\\]\s*(.*?)\.\*\*", p)
        if m:
            title = unescape(m.group(1)).strip()
            cm = re.search(r"Suggested caption:\s*\*(.*?)\*\s*$", p, re.S)
            if cm:
                out.append(raw(manual_figure_tex(title, cm.group(1))))
            else:
                skipping = True
                out.append(("MANUAL", title))
            continue
        if p.startswith("Figure:"):
            pending_caption = p[len("Figure:"):].strip()
            continue
        m = re.match(r"^&#91;(embedded content|image):\s*(.*?)\\\]\s*$", p)
        if m:
            cap_text = unescape(m.group(2))
            hit = next((v for k, v in CFG["embeds"].items() if cap_text.startswith(k)), None)
            if not hit:
                WARN.append(f"embed without a figure file: {cap_text}")
                continue
            cap = tex_escape(hit["caption"]) if hit["caption"] else None
            cap = cap or (pandoc(cites(pending_caption)).strip().rstrip(".") if pending_caption else cap_text)
            pending_caption = None
            out.append(raw(figure_tex(hit["file"], cap, hit["width"])))
            continue
        out.append(cites(p) + "\n")
    return out


def render(parts, md_src: str) -> str:
    # find captions for placeholders whose "Suggested caption" sat in a later paragraph
    caps = {}
    for m in re.finditer(r"\*\*\\\[Figure to add, manual\\\]\s*(.*?)\.\*\*(.*?)Suggested caption:\s*\*(.*?)\*", md_src, re.S):
        caps[unescape(m.group(1)).strip()] = m.group(3)
    md = []
    for item in parts:
        if isinstance(item, tuple):
            md.append(raw(manual_figure_tex(item[1], caps.get(item[1], ""))))
        else:
            md.append(item)
    return pandoc("\n".join(md))


def references(md: str) -> str:
    items = re.findall(r"^\d+\.\s+(.*)$", md, re.M)
    # print the full address, as IEEE asks, instead of the doc's short link text
    items = [re.sub(r"\[([^\]]*)\]\((https?://[^)\s]+)\)", lambda m: m.group(2) if "doi.org" not in m.group(2) else m.group(1), it) for it in items]
    body = pandoc("\n\nREFSEP\n\n".join(items)).split("REFSEP")
    out = ["\\begin{thebibliography}{99}"]
    for i, b in enumerate(body, 1):
        out.append(f"\\bibitem{{b{i}}} " + b.strip())
    out.append("\\end{thebibliography}")
    return "\n".join(out) + "\n"


def main():
    OUT.mkdir(exist_ok=True)
    chapters = sorted(p for p in SRC.glob("*.md") if re.match(r"^\d+ ", p.name))
    names = []
    for p in chapters:
        md = p.read_text(encoding="utf8")
        tex = render(convert_chapter(md), md)
        name = "ch%s" % p.name.split()[0]
        (OUT / f"{name}.tex").write_text(f"% generated from src/{p.name} by tools/doc2tex.py; do not edit\n" + tex, encoding="utf8")
        names.append(name)
    ab = (SRC / "Abstract.md").read_text(encoding="utf8")
    ab = re.sub(r"^# .*\n", "", ab)
    (OUT / "abstract.tex").write_text("% generated from src/Abstract.md\n" + pandoc(cites(ab)), encoding="utf8")
    (OUT / "references.tex").write_text("% generated from src/References.md\n" + references((SRC / "References.md").read_text(encoding="utf8")), encoding="utf8")
    (OUT / "chapters.tex").write_text("".join(f"\\input{{generated/{n}}}\n" for n in names), encoding="utf8")
    used = set(re.findall(r"\\cite\{([^}]*)\}", "".join((OUT / f"{n}.tex").read_text() for n in names)))
    keys = {k for u in used for k in u.split(",")}
    nref = len(re.findall(r"^\d+\.\s", (SRC / "References.md").read_text(encoding="utf8"), re.M))
    missing = [f"b{i}" for i in range(1, nref + 1) if f"b{i}" not in keys]
    if missing:
        WARN.append("references never cited: " + ", ".join(missing))
    for w in WARN:
        print("WARNING:", w, file=sys.stderr)
    print(f"wrote {len(names)} chapters, abstract, {nref} references to {OUT}")


if __name__ == "__main__":
    main()
