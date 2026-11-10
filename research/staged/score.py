"""Score hcpose engagement output against a scripted (staged) classroom session.

Inputs:
  * one or more skeleton files written by `hcpose analyze -o` (persons carry an
    "engagement" block, frames a "class" block)
  * the session script (CSV: start,end,seats,behaviour,label; mm:ss relative to the
    "hands up" start signal; label = distracted | focused | absent; everything outside
    an event is "attentive")
  * the seats per row, front to back (--rows 3,4,3)

Seats S1..SN are found from where people sit during the calibration minutes: the
front row is the lowest in the image, numbered left to right as the camera sees it.
Every frame's detections are then assigned to the nearest seat, so hcpose's own seat
ids don't need to match.

usage:
  python research/staged/score.py take1_n_auto_*.jsonl --script research/staged/script_v1.csv --rows 3,4,3
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

LABELS = ("distracted", "focused", "absent")


# ---------------------------------------------------------------- script
def mmss(s: str) -> float:
    m, sec = s.strip().split(":")
    return 60 * int(m) + float(sec)


def load_script(path: str, seats: List[str]) -> List[dict]:
    events = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            who = row["seats"].split()
            who = list(seats) if who == ["all"] else [s for s in who if s in seats]
            label = row["label"].strip()
            if label not in LABELS:
                raise ValueError(f"{path}: unknown label {label!r} (use {', '.join(LABELS)})")
            if who:
                events.append({"start": mmss(row["start"]), "end": mmss(row["end"]), "seats": who,
                               "behaviour": row["behaviour"].strip(), "label": label})
    return events


def truth(events: List[dict], seat: str, t: float) -> Tuple[str, Optional[dict]]:
    """Scripted label of a seat at script time t (attentive outside events). A seat-specific
    event wins over an 'all' event (e.g. writing notes for everyone)."""
    hit = None
    for e in events:
        if seat in e["seats"] and e["start"] <= t < e["end"]:
            if hit is None or len(e["seats"]) < len(hit["seats"]):
                hit = e
    return (hit["label"], hit) if hit else ("attentive", None)


def near_change(events: List[dict], seat: Optional[str], t: float, margin: float) -> bool:
    for e in events:
        if seat is None or seat in e["seats"]:
            if abs(t - e["start"]) < margin or abs(t - e["end"]) < margin:
                return True
    return False


# ---------------------------------------------------------------- skeleton files
def load_frames(path: str):
    header, frames = None, []
    for line in open(path):
        d = json.loads(line)
        if d.get("type") == "header":
            header = d
        elif d.get("type") == "frame":
            frames.append(d)
    if header is None or not frames:
        raise ValueError(f"{path}: no header or no frames (is it an `hcpose analyze -o` output?)")
    return header, frames


def find_start(header: dict, frames: List[dict], search: float = 120.0) -> Optional[float]:
    """Video time of the 'hands up' start signal: the first moment most people have both
    wrists above their nose, held for at least a second."""
    names = header.get("keypoint_names") or []
    try:
        nose, lw, rw = names.index("nose"), names.index("left_wrist"), names.index("right_wrist")
    except ValueError:
        return None
    hits = []
    for fr in frames:
        if fr["t"] > search:
            break
        P = fr["persons"]
        up = 0
        for p in P:
            k = p["keypoints"]
            if all(k[i][2] >= 0.3 for i in (nose, lw, rw)) and k[lw][1] < k[nose][1] and k[rw][1] < k[nose][1]:
                up += 1
        hits.append((fr["t"], len(P) >= 3 and up >= 0.6 * len(P)))
    for i, (t, ok) in enumerate(hits):
        if ok:
            run = [h for tt, h in hits[i:] if tt < t + 1.0]
            if len(run) >= 3 and np.mean(run) >= 0.7:
                return t
    return None


def center(p: dict, W: float, H: float) -> Tuple[float, float, float]:
    x1, y1, x2, y2 = p["box"]
    return (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, max(1e-3, (x2 - x1) / W)


def find_seats(frames: List[dict], offset: float, rows: List[int], t0: float, t1: float) -> Dict[str, Tuple[float, float, float]]:
    """Cluster where people sit between script times t0..t1 into sum(rows) seats."""
    pts = []
    for fr in frames:
        if t0 <= fr["t"] - offset <= t1:
            for p in fr["persons"]:
                pts.append(center(p, fr["w"], fr["h"]))
    n = sum(rows)
    if len(pts) < n:
        raise ValueError("too few detections in the calibration minutes to find the seats")
    P = np.array(pts)
    # greedy clustering: densest point first, radius = 0.5 box widths
    seats = []
    used = np.zeros(len(P), bool)
    while len(seats) < n and not used.all():
        best, members = None, None
        cand = np.flatnonzero(~used)
        sample = cand if len(cand) <= 4000 else cand[np.linspace(0, len(cand) - 1, 4000).astype(int)]
        for i in sample:
            d = np.hypot(P[cand, 0] - P[i, 0], P[cand, 1] - P[i, 1]) / P[i, 2]
            m = cand[d < 0.5]
            if best is None or len(m) > len(members):
                best, members = i, m
        used[members] = True
        seats.append((len(members), *np.median(P[members], axis=0)))
    if len(seats) < n:
        raise ValueError(f"found only {len(seats)} seats, expected {n}")
    seats.sort(key=lambda s: -s[2])  # front row = lowest in the image = largest y
    out, k = {}, 0
    for size in rows:
        row = sorted(seats[k:k + size], key=lambda s: s[1])
        for s in row:
            out[f"S{len(out) + 1}"] = (s[1], s[2], s[3])
        k += size
    return out


def assign(persons: List[dict], seats: Dict[str, Tuple[float, float, float]], W: float, H: float) -> Dict[str, dict]:
    pairs = []
    for pi, p in enumerate(persons):
        cx, cy, w = center(p, W, H)
        for name, (sx, sy, sw) in seats.items():
            d = math.hypot(cx - sx, (cy - sy) * H / W) / max(w, sw)
            if d < 0.6:
                pairs.append((d, pi, name))
    pairs.sort()
    out, used = {}, set()
    for d, pi, name in pairs:
        if name not in out and pi not in used:
            out[name] = persons[pi]
            used.add(pi)
    return out


# ---------------------------------------------------------------- scoring
def score(path: str, events: List[dict], rows: List[int], offset_arg: str, calib: float, end: float, margin: float) -> dict:
    header, frames = load_frames(path)
    if offset_arg == "auto":
        offset = find_start(header, frames)
        if offset is None:
            raise ValueError(f"{path}: no 'hands up' start found in the first 2 minutes; pass --offset SECONDS")
    else:
        offset = float(offset_arg)
    seats = find_seats(frames, offset, rows, 10.0, calib - 10.0)
    names = list(seats)

    # per seat-second: majority state; per second: mean class estimate + last range
    states = defaultdict(Counter)
    cls = defaultdict(list)
    for fr in frames:
        t = fr["t"] - offset
        if t < 0 or t >= end:
            continue
        sec = int(t)
        for name, p in assign(fr["persons"], seats, fr["w"], fr["h"]).items():
            st = (p.get("engagement") or {}).get("state", "unknown")
            states[(name, sec)][st] += 1
        c = fr.get("class") or {}
        if c.get("class_pct_distracted") is not None:
            cls[sec].append((c["class_pct_distracted"], c.get("class_ci")))

    def pred(name, sec):
        c = states.get((name, sec))
        if not c:
            return "missing"
        st, _ = c.most_common(1)[0]
        return st if st in ("focused", "distracted") else "unknown"

    secs = range(int(calib), int(end))
    tally = Counter()
    for sec in secs:
        t = sec + 0.5
        for name in names:
            gt, _ = truth(events, name, t)
            if gt == "absent" or near_change(events, name, t, margin):
                continue
            tally[(gt, pred(name, sec))] += 1

    def rate(num, den):
        return num / den if den else float("nan")

    dis = sum(v for (g, _), v in tally.items() if g == "distracted")
    tp = tally[("distracted", "distracted")]
    pred_dis = sum(v for (g, p), v in tally.items() if p == "distracted")
    judged = lambda g: tally[(g, "focused")] + tally[(g, "distracted")]  # noqa: E731
    total = sum(tally.values())
    covered = sum(v for (g, p), v in tally.items() if p in ("focused", "distracted"))

    # latency and per-behaviour recall, per (event, seat)
    lat, missed, per_beh = [], 0, defaultdict(list)
    for e in events:
        if e["label"] != "distracted" or e["start"] < calib:
            continue
        for name in e["seats"]:
            hit = next((s - e["start"] for s in range(int(e["start"]), int(e["end"])) if pred(name, s) == "distracted"), None)
            if hit is None:
                missed += 1
            else:
                lat.append(max(0.0, hit))
            inside = [pred(name, s) == "distracted" for s in range(int(e["start"] + margin), int(e["end"] - margin))]
            if inside:
                per_beh[e["behaviour"]].append(float(np.mean(inside)))

    # class level
    err, hit_ci, gts, pcs = [], [], [], []
    for sec in secs:
        t = sec + 0.5
        if sec not in cls or near_change(events, None, t, margin):
            continue
        labels = [truth(events, n, t)[0] for n in names]
        present = [lab for lab in labels if lab != "absent"]
        g = 100 * sum(lab == "distracted" for lab in present) / len(present)
        pc = float(np.mean([v for v, _ in cls[sec]]))
        ci = cls[sec][-1][1]
        err.append(abs(pc - g))
        gts.append(g)
        pcs.append(pc)
        if ci:
            hit_ci.append(ci[0] - 0.5 <= g <= ci[1] + 0.5)
    r = float(np.corrcoef(gts, pcs)[0, 1]) if len(gts) > 2 and np.std(gts) > 0 and np.std(pcs) > 0 else float("nan")

    return {
        "file": Path(path).name, "offset": offset, "seats": seats,
        "coverage": rate(covered, total),
        "recall": rate(tp, dis),
        "precision": rate(tp, pred_dis),
        "fa_attentive": rate(tally[("attentive", "distracted")], judged("attentive")),
        "fa_hard": rate(tally[("focused", "distracted")], judged("focused")),
        "latency": float(np.median(lat)) if lat else float("nan"),
        "events": len(lat) + missed, "missed": missed,
        "class_mae": float(np.mean(err)) if err else float("nan"),
        "class_r": r,
        "range_hit": float(np.mean(hit_ci)) if hit_ci else float("nan"),
        "per_behaviour": {k: float(np.mean(v)) for k, v in per_beh.items()},
    }


def pct(x):
    return "–" if x != x else f"{100 * x:.0f}%"


def num(x, fmt):
    return "–" if x != x else format(x, fmt)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+", help="`hcpose analyze -o` outputs")
    ap.add_argument("--script", required=True, help="session script CSV")
    ap.add_argument("--rows", required=True, help="seats per row, front to back, e.g. 3,4,3")
    ap.add_argument("--offset", default="auto", help="video time of script 00:00 in seconds, or auto (hands-up signal)")
    ap.add_argument("--calib", type=float, default=180.0, help="calibration minutes at the start, not scored (s)")
    ap.add_argument("--end", type=float, default=900.0, help="script end (s)")
    ap.add_argument("--margin", type=float, default=5.0, help="seconds ignored on each side of every scripted change")
    ap.add_argument("--out", help="also write the summary as markdown here")
    args = ap.parse_args(argv)

    rows = [int(x) for x in args.rows.split(",")]
    names = [f"S{i + 1}" for i in range(sum(rows))]
    events = load_script(args.script, names)
    res = []
    for f in args.files:
        try:
            res.append(score(f, events, rows, args.offset, args.calib, args.end, args.margin))
        except ValueError as e:
            print(f"[score] {e}", file=sys.stderr)
    if not res:
        return 1

    lines = ["| File | Coverage | Recall | Precision | False alarms, attentive | False alarms, hard focused | "
             "Latency (s) | Events missed | Class error (pts) | Class r | Range hit |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        lines.append(f"| {r['file']} | {pct(r['coverage'])} | {pct(r['recall'])} | {pct(r['precision'])} | "
                     f"{pct(r['fa_attentive'])} | {pct(r['fa_hard'])} | "
                     f"{num(r['latency'], '.1f')} | {r['missed']} of {r['events']} | "
                     f"{num(r['class_mae'], '.1f')} | {num(r['class_r'], '.2f')} | {pct(r['range_hit'])} |")
    behs = sorted({b for r in res for b in r["per_behaviour"]})
    lines += ["", "Recall per scripted behaviour (share of the event's seconds labelled distracted):", "",
              "| File | " + " | ".join(behs) + " |", "|---|" + "---|" * len(behs)]
    for r in res:
        lines.append(f"| {r['file']} | " + " | ".join(pct(r["per_behaviour"].get(b, float("nan"))) for b in behs) + " |")
    first = res[0]
    lines += ["", f"Start signal at {first['offset']:.1f} s into the video (first file). Seats found "
              "(x, y as a share of the frame): " + ", ".join(f"{k} ({v[0]:.2f}, {v[1]:.2f})" for k, v in first["seats"].items())]
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
