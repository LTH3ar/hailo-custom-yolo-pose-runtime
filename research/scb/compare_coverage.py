"""Compare how many labelled SCB students two pose caches detect (e.g. without vs with tiling).

usage: compare_coverage.py A.jsonl B.jsonl [--det-thr 0.3] [--names none,tiles]
Same images in both files are required. Coverage = labelled student matched to a detection
(box overlap as in evaluate.py). Broken down by behavior and by the student's size in the
image (small = far rows), since tiling is meant for the small ones.
"""
import argparse
import json
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--det-thr", type=float, default=0.3)
ap.add_argument("--names", default="A,B")
args = ap.parse_args()
names = args.names.split(",")


def load(path):
    return {d["img"]: d for d in map(json.loads, open(path))}


def matched(gt, dets):
    pairs = []
    for gi, g in enumerate(gt):
        for di, x in enumerate(dets):
            b = x["box"]
            ix = max(0, min(g[3], b[2]) - max(g[1], b[0]))
            iy = max(0, min(g[4], b[3]) - max(g[2], b[1]))
            inter = ix * iy
            ga = (g[3] - g[1]) * (g[4] - g[2])
            da = (b[2] - b[0]) * (b[3] - b[1])
            cov, iou = inter / max(ga, 1e-9), inter / max(ga + da - inter, 1e-9)
            if cov >= 0.5 or iou >= 0.3:
                pairs.append((iou + cov, gi, di))
    pairs.sort(reverse=True)
    ug, ud = set(), set()
    for _, gi, di in pairs:
        if gi not in ug and di not in ud:
            ug.add(gi)
            ud.add(di)
    return ug


A, B = load(args.a), load(args.b)
common = sorted(set(A) & set(B))
rows = []  # (behavior, rel_height, hitA, hitB)
nd = {names[0]: [], names[1]: []}
for img in common:
    for nm, src in zip(names, (A, B)):
        nd[nm].append(sum(x["score"] >= args.det_thr for x in src[img]["dets"]))
    d = A[img]
    hit = [matched(src[img]["gt"], [x for x in src[img]["dets"] if x["score"] >= args.det_thr]) for src in (A, B)]
    for gi, g in enumerate(d["gt"]):
        rows.append((g[0], (g[4] - g[2]) / d["h"], gi in hit[0], gi in hit[1]))

print(f"{len(common)} images, {len(rows)} labelled students, detection threshold {args.det_thr}")
print(f"detections per image: {names[0]} {np.mean(nd[names[0]]):.1f}, {names[1]} {np.mean(nd[names[1]]):.1f}\n")
h = np.array([r[1] for r in rows])
cuts = np.quantile(h, [1 / 3, 2 / 3])
size = np.where(h < cuts[0], "small (far)", np.where(h < cuts[1], "medium", "large (near)"))
groups = defaultdict(list)
for r, sz in zip(rows, size):
    groups[("size", sz)].append(r)
    groups[("behavior", r[0])].append(r)
    groups[("all", "all")].append(r)
print(f"{'':28} {names[0]:>8} {names[1]:>8}   gain")
for key in [("all", "all"), ("size", "small (far)"), ("size", "medium"), ("size", "large (near)")] + \
           sorted(k for k in groups if k[0] == "behavior"):
    rs = groups[key]
    a = np.mean([r[2] for r in rs])
    b = np.mean([r[3] for r in rs])
    print(f"{key[1]:28} {a:8.1%} {b:8.1%}  {100 * (b - a):+5.1f} pts   (n={len(rs)})")
