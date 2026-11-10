"""Score `hcpose analyze` outputs against human labels exported from the labelling page.

Each labelled moment (clip, time, box) is matched to the nearest frame (within 0.25 s) and
the detection with the highest IoU (at least 0.3) in each analysis file of the same clip.
Items were sampled per stratum (rules vs learned decision at labelling time), so besides the
plain counts the scorer reports estimates re-weighted to the whole clip
(weight = stratum size / items labelled in that stratum).

usage: python research/labels/score_labels.py student_labels.json RESULTS/*_none.jsonl RESULTS/*_trees.jsonl ...
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("labels")
ap.add_argument("files", nargs="+", help="`hcpose analyze -o` outputs, optionally as CLIP=path; else the clip is "
                                          "the stem of the header's source, or the file name up to its _<model>_ part")
args = ap.parse_args()

L = json.load(open(args.labels))["items"]
items = [x for x in L if x["label"] in ("focused", "distracted")]
n_by = defaultdict(int)
for x in items:
    n_by[(x["clip"], x["stratum"])] += 1
for x in items:
    x["w"] = x["stratum_size"] / n_by[(x["clip"], x["stratum"])]
print(f"{len(L)} labelled, {len(items)} focused/distracted, {len(L) - len(items)} can't tell; "
      f"distracted share as labelled {np.mean([x['label'] == 'distracted' for x in items]):.0%}, "
      f"re-weighted {np.average([x['label'] == 'distracted' for x in items], weights=[x['w'] for x in items]):.1%}\n")


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)


def clip_of(path, header):
    src = (header or {}).get("source")
    if src and src not in ("usb", "rpi"):
        return Path(src).stem
    stem = Path(path).stem
    for m in ("_n_", "_s_", "_m_"):
        if m in stem:
            return stem.split(m)[0]
    return stem


rows = []
for f in args.files:
    forced = None
    if "=" in f:
        forced, f = f.split("=", 1)
    header, frames = None, []
    for line in open(f):
        d = json.loads(line)
        if d.get("type") == "header":
            header = d
        elif d.get("type") == "frame":
            frames.append(d)
    clip = forced or clip_of(f, header)
    mine = [x for x in items if x["clip"] == clip]
    if not mine or not frames:
        continue
    ts = np.array([fr["t"] for fr in frames])
    y, p, w = [], [], []
    missing = 0
    for x in mine:
        k = int(np.argmin(np.abs(ts - x["t"])))
        best = None
        if abs(ts[k] - x["t"]) <= 0.25:
            cands = [(iou(x["box"], q["box"]), q) for q in frames[k]["persons"]]
            cands = [c for c in cands if c[0] >= 0.3]
            if cands:
                best = max(cands, key=lambda c: c[0])[1]
        st = ((best or {}).get("engagement") or {}).get("state")
        if st not in ("focused", "distracted"):
            missing += 1
            st = "focused"  # not flagged: counts as a miss for a distracted student
        y.append(x["label"] == "distracted")
        p.append(st == "distracted")
        w.append(x["w"])
    y, p, w = np.array(y), np.array(p), np.array(w)

    def m(weights):
        tp = (weights * (y & p)).sum()
        fp = (weights * (~y & p)).sum()
        fn = (weights * (y & ~p)).sum()
        tn = (weights * (~y & ~p)).sum()
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        spec = tn / (tn + fp) if tn + fp else float("nan")
        return prec, rec, spec, (rec + spec) / 2, (weights * p).sum() / weights.sum()

    raw, rw = m(np.ones(len(y))), m(w)
    rows.append((Path(f).name, len(y), missing, raw, rw))

pc = lambda v: "–" if v != v else f"{100 * v:.0f}%"  # noqa: E731
print("| File | Items | Not matched | Precision | Recall | Specificity | Balanced acc. | Flagged share | "
      "Precision (re-weighted) | Recall (re-weighted) | Flagged share (re-weighted) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for name, n, miss, a, b in rows:
    print(f"| {name} | {n} | {miss} | {pc(a[0])} | {pc(a[1])} | {pc(a[2])} | {pc(a[3])} | {pc(a[4])} | "
          f"{pc(b[0])} | {pc(b[1])} | {pc(b[4])} |")
