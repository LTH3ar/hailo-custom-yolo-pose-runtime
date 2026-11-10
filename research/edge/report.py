"""Turn the files written by research/edge/bench.sh into markdown tables.

usage: python research/edge/report.py ~/edge_results
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1])


def load(name):
    p = D / f"{name}.csv"
    if not p.is_file():
        return None
    rows = list(csv.DictReader(open(p)))
    return rows or None


def col(rows, k):
    v = [float(r[k]) for r in rows if r.get(k) not in ("", None)]
    return np.array(v) if v else None


def fmt(x, f="{:.1f}"):
    return "–" if x is None else f.format(x)


def stats(rows):
    if not rows:
        return {}
    out = {}
    for k in ("cpu_total", "cpu_hcpose", "pmic_w", "freq_mhz"):
        v = col(rows, k)
        out[k] = None if v is None else float(v.mean())
    for k in ("rss_mb", "temp_c"):
        v = col(rows, k)
        out[k + "_max"] = None if v is None else float(v.max())
    thr = {r["throttled"] for r in rows if r.get("throttled")}
    out["throttled"] = ", ".join(sorted(t for t in thr if t not in ("0x0", ""))) or "none"
    return out


def fps(log):
    p = D / log
    m = re.findall(r"(\d+) frames in ([\d.]+)s \(([\d.]+) FPS avg\)", p.read_text(errors="replace")) if p.is_file() else []
    return float(m[-1][2]) if m else None


lines = ["# Edge measurements", ""]
sysf = D / "system.txt"
if sysf.is_file():
    lines += ["```", *sysf.read_text(errors="replace").strip().splitlines()[:12], "```", ""]

idle = stats(load("idle"))
if idle:
    lines += ["## Idle (60 s, nothing running)", "",
              f"CPU {fmt(idle['cpu_total'])}%, SoC {fmt(idle['temp_c_max'])} °C max, "
              f"Pi board rails (PMIC) {fmt(idle['pmic_w'], '{:.2f}')} W, throttling: {idle['throttled']}.", ""]

bench = sorted(D.glob("hailo_benchmark_*.txt"))
if bench:
    lines += ["## Hailo-8 only (hailortcli)", ""]
    for b in bench:
        txt = b.read_text(errors="replace")
        # hailortcli prints progress lines, then a Summary block; keep the summary (or, for
        # `hailortcli run` and errors, the last lines that mention fps/latency/power/error)
        if "Summary" in txt:
            keep = [ln.rstrip() for ln in txt.split("Summary", 1)[1].splitlines()
                    if ln.strip() and not set(ln.strip()) <= {"="}]
        else:
            keep = [ln.strip() for ln in re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "\n", txt).splitlines()
                    if re.search(r"fps|latency|power|error", ln, re.I)][-6:]
        lines += [f"**{b.stem.split('_')[-1]}**", "", "```", *keep[:12], "```", ""]

lines += ["## End to end (clip, as fast as possible)", "",
          "| Model | Tiles | FPS | CPU, whole Pi | CPU, hcpose (% of a core) | Memory (MB) | Max SoC temp (°C) | Pi board rails (W) | Throttling |",
          "|---|---|---|---|---|---|---|---|---|"]
for m in ("n", "s", "m"):
    for t in ("none", "auto"):
        s = stats(load(f"throughput_{m}_{t}"))
        if not s:
            continue
        lines.append(f"| {m} | {'on' if t == 'auto' else 'off'} | {fmt(fps(f'throughput_{m}_{t}.log'))} | "
                     f"{fmt(s['cpu_total'], '{:.0f}')}% | {fmt(s['cpu_hcpose'], '{:.0f}')}% | {fmt(s['rss_mb_max'], '{:.0f}')} | "
                     f"{fmt(s['temp_c_max'])} | {fmt(s['pmic_w'], '{:.2f}')} | {s['throttled']} |")

def latency_ms(p):
    """Per-frame live_pace -> sink latency (ms) from a GstShark interlatency trace.

    Uses each record's log timestamp (wall clock), not interlatency's own "time since the
    source" value: that one is attached to compressed buffers and does not follow frames
    reliably through the decoder (B-frame reordering gave negative differences). Frames pass
    live_pace and the sink in order, so the n-th record of each pad is the same frame."""
    txt = re.sub(r"\x1b\[[0-9;]*m", "", p.read_text(errors="replace"))

    def wall(pad):
        return [(int(h) * 3600 + int(mi) * 60 + float(se)) * 1000 for h, mi, se in re.findall(
            rf"^(\d+):(\d+):([\d.]+) .*?interlatency, from_pad=\(string\)\S+ to_pad=\(string\){pad},", txt, re.M)]
    a, b = wall("live_pace_src"), wall("sink_sink")
    k = min(len(a), len(b))
    return np.array([b[i] - a[i] for i in range(k)])


lat_rows = []
for m in ("n", "m"):
    for t in ("none", "auto"):
        p = D / f"latency_{m}_{t}.trace"
        if not p.is_file():
            continue
        ms = latency_ms(p)
        if len(ms):
            lat_rows.append(f"| {m} | {'on' if t == 'auto' else 'off'} | {len(ms)} | {np.median(ms):.0f} | "
                            f"{np.percentile(ms, 95):.0f} | {fps(f'latency_{m}_{t}.log') or '–'} |")
if lat_rows:
    lines += ["", "## Pipeline latency (clip paced at its frame rate like a camera; GstShark interlatency, live_pace to sink)", "",
              "| Model | Tiles | Frames | Median (ms) | 95th percentile (ms) | FPS |", "|---|---|---|---|---|---|", *lat_rows,
              "", "When a setting runs below the clip's frame rate (m with tiles), frames queue up and the latency "
              "includes that wait."]

for name, label in (("soak_n_auto_realtime", "n + tiles, real time (deployment setting)"),
                    ("soak_m_auto_max", "m + tiles, as fast as possible (worst case)")):
    rows = load(name)
    if not rows:
        continue
    t = col(rows, "t")
    temp = col(rows, "temp_c")
    freq = col(rows, "freq_mhz")
    s = stats(rows)
    at = lambda sec: fmt(float(temp[np.argmin(np.abs(t - sec))])) if temp is not None and len(t) else "–"  # noqa: E731
    lines += ["", f"## 15-minute soak: {label}", "",
              f"- FPS: {fmt(fps(name + '.log'))}",
              f"- SoC temperature: {at(10)} °C after 10 s, {at(300)} after 5 min, {at(600)} after 10 min, "
              f"{at(890)} after 15 min (max {fmt(s['temp_c_max'])})",
              f"- CPU clock: mean {fmt(s['freq_mhz'], '{:.0f}')} MHz, minimum {fmt(None if freq is None else float(freq.min()), '{:.0f}')} MHz",
              f"- Throttling flags seen: {s['throttled']}",
              f"- CPU: {fmt(s['cpu_total'], '{:.0f}')}% of the whole Pi; Pi board rails {fmt(s['pmic_w'], '{:.2f}')} W"]

lines += ["", "Pi board rails = sum of V x I over the Pi 5 PMIC rails (`vcgencmd pmic_read_adc`): the board's own "
          "supplies, without the AI HAT and USB devices. Hailo-8 power comes from the vendor's specification."]
print("\n".join(lines))
