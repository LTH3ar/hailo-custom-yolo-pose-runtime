"""Latency over the run (same pairing as research/edge/report.py latency_ms)."""
import re, sys
from pathlib import Path
import numpy as np
for p in map(Path, sys.argv[1:]):
    txt = re.sub(r"\x1b\[[0-9;]*m", "", p.read_text(errors="replace"))
    def wall(pad):
        return [(int(h) * 3600 + int(mi) * 60 + float(se)) * 1000 for h, mi, se in re.findall(
            rf"^(\d+):(\d+):([\d.]+) .*?interlatency, from_pad=\(string\)\S+ to_pad=\(string\){pad},", txt, re.M)]
    a, b = wall("live_pace_src"), wall("sink_sink"); k = min(len(a), len(b))
    ms = np.array([b[i] - a[i] for i in range(k)])
    q = np.array_split(ms, 4)
    print(f"{p.name}: n={k} live_pace records {len(a)}, sink {len(b)}; first frame {ms[0]:.0f} ms; "
          f"median by quarter {' / '.join(f'{np.median(x):.0f}' for x in q)}; max {ms.max():.0f}")
