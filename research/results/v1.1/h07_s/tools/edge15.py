"""Handoff 07 E: the same numbers research/edge/report.py prints, for the s + tiles 15 FPS files
(report.py only loops over n and m, so its functions are copied here unchanged)."""
import csv, re, sys
from pathlib import Path
import numpy as np
D = Path(sys.argv[1])

def load(name):
    p = D / f"{name}.csv"
    return list(csv.DictReader(open(p))) if p.is_file() else None

def col(rows, k):
    v = [float(r[k]) for r in rows if r.get(k) not in ("", None)]
    return np.array(v) if v else None

def stats(rows):
    out = {}
    for k in ("cpu_total", "cpu_hcpose", "pmic_w", "freq_mhz"):
        v = col(rows, k); out[k] = None if v is None else float(v.mean())
    for k in ("rss_mb", "temp_c"):
        v = col(rows, k); out[k + "_max"] = None if v is None else float(v.max())
    thr = {r["throttled"] for r in rows if r.get("throttled")}
    out["throttled"] = ", ".join(sorted(t for t in thr if t not in ("0x0", ""))) or "none"
    return out

def latency_ms(p):
    txt = re.sub(r"\x1b\[[0-9;]*m", "", p.read_text(errors="replace"))
    def wall(pad):
        return [(int(h) * 3600 + int(mi) * 60 + float(se)) * 1000 for h, mi, se in re.findall(
            rf"^(\d+):(\d+):([\d.]+) .*?interlatency, from_pad=\(string\)\S+ to_pad=\(string\){pad},", txt, re.M)]
    a, b = wall("live_pace_src"), wall("sink_sink")
    k = min(len(a), len(b))
    return np.array([b[i] - a[i] for i in range(k)])

ms = latency_ms(D / 'latency_s_auto.trace')
print(f"latency s tiles, 15 FPS lecture: {len(ms)} frames, median {np.median(ms):.0f} ms, p95 {np.percentile(ms, 95):.0f} ms, max {ms.max():.0f} ms")
print(f"  first half median {np.median(ms[:len(ms)//2]):.0f}, second half median {np.median(ms[len(ms)//2:]):.0f} ms")
for name in ["realtime_clip_s_auto_15fps", "realtime_clip_3_s_auto_15fps", "realtime_clip_5_720p_s_auto_15fps", "soak_s_auto_15fps"]:
    rows = load(name); s = stats(rows); t = col(rows, "t"); temp = col(rows, "temp_c"); freq = col(rows, "freq_mhz")
    at = lambda sec: float(temp[np.argmin(np.abs(t - sec))])
    print(f"{name}: {len(rows)} samples over {t[-1]:.0f} s; CPU {s['cpu_total']:.0f}% (hcpose {s['cpu_hcpose']:.0f}% of a core), "
          f"memory max {s['rss_mb_max']:.0f} MB, board {s['pmic_w']:.2f} W, max {s['temp_c_max']:.1f} C, "
          f"clock mean {s['freq_mhz']:.0f} min {freq.min():.0f} MHz, throttled {s['throttled']}")
    if name.startswith("soak"):
        print(f"  temp 10 s {at(10):.1f}, 5 min {at(300):.1f}, 10 min {at(600):.1f}, 15 min {at(890):.1f} C")
