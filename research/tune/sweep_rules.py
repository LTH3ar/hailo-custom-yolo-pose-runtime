"""Sweep the rule thresholds on recorded skeletons and score every setting on all the evidence.

The rules were set by hand. This replays saved INT8 skeleton files (no model is run)
with a grid of
  * head_turn ramp (yaw change where the head_turned cue starts / is full)
  * on_threshold (smoothed score needed to switch to distracted; off = on - 0.2)
and reports, per setting:
  * staged clip_4: recall, precision, false alarms (attentive, and while writing), latency
  * human labels (student_labels_v1.json, lecture + exam clips): false alarms among matched
    focused moments, and distracted moments caught
  * class distracted % on the lecture, exam and study-hall clips (truth: low in all three)

usage: python research/tune/sweep_rules.py [--jobs 8] [--out research/tune/sweep.md]
Clip_4 is used to choose the setting; keep the 9-person staged session as the held-out test.
"""
import argparse
import itertools
import json
import lzma
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "staged"))
import score as staged  # noqa: E402

R1 = ROOT / "research/results/pi_20261110"
R4 = ROOT / "research/results/pi_staged_clip4"
SKELETONS = {  # name: (xz file, camera position)
    "clip4_n": (R4 / "clip_4_720p_n_none.skeleton.jsonl.xz", "center"),
    "clip4_m": (R4 / "clip_4_720p_m_none.skeleton.jsonl.xz", "center"),
    "lecture_n": (R1 / "clip_3_n_auto.skeleton.jsonl.xz", "center"),
    "lecture_m": (R1 / "clip_3_m_auto.skeleton.jsonl.xz", "center"),
    "exam_n": (R1 / "clip_n_auto.skeleton.jsonl.xz", "auto"),
    "exam_m": (R1 / "clip_m_auto.skeleton.jsonl.xz", "auto"),
    "study_n": (R1 / "clip_2_n_auto.skeleton.jsonl.xz", "center"),
    "study_m": (R1 / "clip_2_m_auto.skeleton.jsonl.xz", "center"),
}
LABEL_CLIP = {"lecture": "clip_3", "exam": "clip"}

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
ap.add_argument("--work", default=str(Path(tempfile.gettempdir()) / "hcpose_sweep"))
ap.add_argument("--out", default=str(ROOT / "research/tune/sweep.md"))
ap.add_argument("--only", help="comma-separated skeleton names (default: all)")
ap.add_argument("--grid", default="full", choices=["full", "current"])
args = ap.parse_args()

TURNS = [(0.35, 0.85), (0.25, 0.65), (0.2, 0.5), (0.15, 0.4), (0.1, 0.3)]
ONS = [0.6, 0.5, 0.4]
GRID = [((0.35, 0.85), 0.6)] if args.grid == "current" else list(itertools.product(TURNS, ONS))
names = args.only.split(",") if args.only else list(SKELETONS)
work = Path(args.work)
work.mkdir(parents=True, exist_ok=True)


def unpack(name):
    src, _ = SKELETONS[name]
    dst = work / f"{name}.skeleton.jsonl"
    if not dst.is_file():
        with lzma.open(src) as f, open(dst, "wb") as o:
            o.write(f.read())
    return dst


def run(job):
    (turn, on), name = job
    tag = f"t{turn[0]}-{turn[1]}_on{on}"
    cfg = work / f"{tag}.yaml"
    cfg.write_text(f"head_turn_start: {turn[0]}\nhead_turn_full: {turn[1]}\non_threshold: {on}\noff_threshold: {round(on - 0.2, 2)}\n")
    out = work / f"{name}__{tag}.jsonl"
    if not out.is_file():
        subprocess.run([sys.executable, "-m", "hcpose.cli", "analyze", str(work / f"{name}.skeleton.jsonl"),
                        "--engagement", str(cfg), "--classifier", "none", "--camera-position", SKELETONS[name][1],
                        "-o", str(out)], cwd=ROOT, check=True, capture_output=True)
    return (turn, on), name, out


def frames(path):
    return [d for d in map(json.loads, open(path)) if d.get("type") == "frame"]


def class_pct(path):
    C = [(d["t"], d["class"]["class_pct_distracted"]) for d in frames(path)
         if d.get("class") and d["class"].get("class_pct_distracted") is not None]
    if not C:
        return float("nan")
    t0 = 150 if C[-1][0] >= 180 else 0
    return float(np.mean([v for t, v in C if t >= t0]))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9)


LABELS = [x for x in json.load(open(ROOT / "research/labels/student_labels_v1.json"))["items"]
          if x["label"] in ("focused", "distracted")]


def label_score(paths):
    """paths: {label clip name: analysis file}. Returns (false alarms, matched focused, caught, matched distracted)."""
    fa = nf = tp = nd = 0
    for clip, path in paths.items():
        F = frames(path)
        ts = np.array([f["t"] for f in F])
        for x in (x for x in LABELS if x["clip"] == clip):
            k = int(np.argmin(np.abs(ts - x["t"])))
            if abs(ts[k] - x["t"]) > 0.25:
                continue
            c = [(iou(x["box"], q["box"]), q) for q in F[k]["persons"]]
            c = [z for z in c if z[0] >= 0.3]
            if not c:
                continue
            st = (max(c, key=lambda z: z[0])[1].get("engagement") or {}).get("state")
            if st not in ("focused", "distracted"):
                continue
            if x["label"] == "focused":
                nf += 1
                fa += st == "distracted"
            else:
                nd += 1
                tp += st == "distracted"
    return fa, nf, tp, nd


for n in names:
    unpack(n)
jobs = [(g, n) for g in GRID for n in names]
print(f"{len(jobs)} replays ({len(GRID)} settings x {len(names)} skeleton files) on {args.jobs} processes", flush=True)
done = {}
with ProcessPoolExecutor(args.jobs) as ex:
    for g, n, out in ex.map(run, jobs):
        done[(g, n)] = out
        print(f"  {n} {g}", flush=True)

events = staged.load_script(str(ROOT / "research/staged/script_clip4.csv"), ["S1", "S2"])
pc = lambda v: "–" if v != v else f"{100 * v:.0f}%"  # noqa: E731
lines = ["| Head turn ramp | On threshold | Model | clip_4 recall | clip_4 precision | clip_4 false alarms, attentive / writing | "
         "clip_4 latency (s) | Labels: false alarms on focused | Labels: distracted caught | Lecture % | Exam % | Study hall % |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for g in GRID:
    for m in ("n", "m"):
        if f"clip4_{m}" not in names:
            continue
        r = staged.score(str(done[(g, f"clip4_{m}")]), events, [2], "auto", 58.0, 900.0, 5.0)
        lab = {LABEL_CLIP[c]: done[(g, f"{c}_{m}")] for c in ("lecture", "exam") if f"{c}_{m}" in names}
        fa, nf, tp, nd = label_score(lab) if lab else (0, 0, 0, 0)
        cp = {c: (class_pct(done[(g, f"{c}_{m}")]) if f"{c}_{m}" in names else float("nan")) for c in ("lecture", "exam", "study")}
        lines.append(f"| {g[0][0]}-{g[0][1]} | {g[1]} | {m} | {pc(r['recall'])} | {pc(r['precision'])} | "
                     f"{pc(r['fa_attentive'])} / {pc(r['fa_hard'])} | {staged.num(r['latency'], '.1f')} | "
                     f"{fa}/{nf} ({pc(fa / nf if nf else float('nan'))}) | {tp}/{nd} | "
                     + " | ".join("–" if v != v else f"{v:.1f}%" for v in cp.values()) + " |")
text = "\n".join(lines)
print(text)
Path(args.out).write_text("# Rule threshold sweep\n\nCurrent defaults: head turn ramp 0.35-0.85, on threshold 0.6.\n\n" + text + "\n")
print(f"written to {args.out}")
