"""Mean class distracted % (after 2:30 if the clip is >= 3 min, as summarize.py), mean 95% range width,
final class size, and the share of frames where the shown value is outside its range.
usage: classpct.py ANALYSIS.jsonl [...]"""
import json
import sys

import numpy as np

for p in sys.argv[1:]:
    C = [(d["t"], d["class"]) for d in map(json.loads, open(p))
         if d.get("type") == "frame" and d.get("class") and d["class"].get("class_pct_distracted") is not None]
    t0 = 150 if C[-1][0] >= 180 else 0
    late = [c for t, c in C if t >= t0]
    w = [c["class_ci"][1] - c["class_ci"][0] for c in late if c.get("class_ci")]
    out = sum(1 for _, c in C if c.get("class_ci")
              and not (c["class_ci"][0] - 1e-6 <= c["class_pct_distracted"] <= c["class_ci"][1] + 1e-6))
    print(f"{p.split('/')[-1]:48} {np.mean([c['class_pct_distracted'] for c in late]):5.1f}%  "
          f"width {np.mean(w):4.1f}  size {C[-1][1].get('class_size')}  outside-range frames {out}/{len(C)}")
