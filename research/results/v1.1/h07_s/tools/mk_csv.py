"""Handoff 07: Handoff 02's two figure CSVs plus s rows (H02 rows copied unchanged)."""
import json, re
from pathlib import Path
import numpy as np
O = Path.home() / "hcpose_paper"; H = O / "h07"

# SCB coverage by third: s rows parsed from coverage_s.txt (compare_coverage.py output)
rows = (O / "figures/scb_coverage_by_third.csv").read_text().rstrip("\n").split("\n")
txt = (H / "coverage_s.txt").read_text()
for g in ("all", "small (far)", "medium", "large (near)"):
    m = re.search(rf"^{re.escape(g)}\s+([\d.]+)%\s+([\d.]+)%\s+([+-][\d.]+) pts\s+\(n=(\d+)\)", txt, re.M)
    a, b, gain, n = m.groups()
    rows.append(f"s,{g},{a},{b},{float(gain):.1f},{n}")
(H / "scb_coverage_by_third.csv").write_text("\n".join(rows) + "\n")

# Lecture / exam class %: same computation as classpct.py (mean class_pct_distracted; after 2:30 if clip >= 3 min)
def pct(p):
    C = [(d["t"], d["class"]) for d in map(json.loads, open(p))
         if d.get("type") == "frame" and d.get("class") and d["class"].get("class_pct_distracted") is not None]
    t0 = 150 if C[-1][0] >= 180 else 0
    return float(np.mean([c["class_pct_distracted"] for t, c in C if t >= t0])), ("after 2:30" if t0 else "whole")

rows = (O / "figures/lecture_exam_class_pct.csv").read_text().rstrip("\n").split("\n")
dec = ["lenient_logistic", "lenient_mlp", "lenient_trees", "rules", "strict_logistic", "strict_mlp", "strict_trees",
       "thesis_logistic", "thesis_mlp", "thesis_trees"]
for clip, stem in (("lecture", "clip_3"), ("exam", "clip")):
    for t, tl in (("auto", "on"), ("none", "off")):
        for d in dec:
            v, w = pct(H / "clips" / f"{stem}_s_{t}_{'none' if d == 'rules' else d}.jsonl")
            rows.append(f"{clip},s,{tl},{d},{v:.1f},{w}")
(H / "lecture_exam_class_pct.csv").write_text("\n".join(rows) + "\n")
print("ok", len(rows) - 1, "class pct rows")
