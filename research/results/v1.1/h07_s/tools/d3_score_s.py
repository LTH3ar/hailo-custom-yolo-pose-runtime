"""Handoff 07 (uncommitted copy of Handoff 02 d3_score.py, s model): pooled label scores for every model/tiles/decision, with the same matching
as research/labels/score_labels.py and pooled_summary.py (nearest frame within 0.25 s, best IoU >= 0.3;
no match or state unknown = not flagged). Re-weighted flagged share per clip as in score_labels.py.
usage: d3_score.py LABELS.json CLIPS_DIR [--csv OUT.csv]"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("labels")
ap.add_argument("dir")
ap.add_argument("--csv")
a = ap.parse_args()

L = [x for x in json.load(open(a.labels))["items"] if x["label"] in ("focused", "distracted")]
nby = defaultdict(int)
for x in L:
    nby[(x["clip"], x["stratum"])] += 1
for x in L:
    x["w"] = x["stratum_size"] / nby[(x["clip"], x["stratum"])]
clips = sorted({x["clip"] for x in L}, key=lambda c: -sum(x["clip"] == c for x in L))


def iou(p, q):
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0]))
    iy = max(0, min(p[3], q[3]) - max(p[1], q[1]))
    i = ix * iy
    return i / max((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i, 1e-9)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** .5 / d
    return c - h, c + h


DEC = [("rules", "none")] + [(f"{s} {k}", f"{s}_{k}") for s in ("thesis", "strict", "lenient")
                             for k in ("trees", "logistic", "mlp")]
ndis = sum(x["label"] == "distracted" for x in L)
print(f"labels {os.path.basename(a.labels)}: {len(L)} judgeable, {ndis} distracted; clips {clips}\n")
print("| Model | Tiles | Decision | Not matched | False alarms / focused | Specificity (95% CI) | Caught / distracted"
      " | FA, matched focused only | Flagged share re-weighted (" + " / ".join(clips) + ") |")
print("|---|---|---|---|---|---|---|---|---|")
out = []
for m in ("s",):
    for t in ("none", "auto"):
        for name, suf in DEC:
            st, rw, ok = {}, {}, True
            for clip in clips:
                f = f"{a.dir}/{clip}_{m}_{t}_{suf}.jsonl"
                if not os.path.exists(f):
                    ok = False
                    break
                fr = [d for d in map(json.loads, open(f)) if d.get("type") == "frame"]
                ts = np.array([d["t"] for d in fr])
                for i, x in enumerate(L):
                    if x["clip"] != clip:
                        continue
                    k = int(np.argmin(np.abs(ts - x["t"])))
                    s = None
                    if abs(ts[k] - x["t"]) <= .25:
                        c = [(iou(x["box"], q["box"]), q) for q in fr[k]["persons"]]
                        c = [z for z in c if z[0] >= .3]
                        if c:
                            s = (max(c, key=lambda z: z[0])[1].get("engagement") or {}).get("state")
                    st[i] = s if s in ("focused", "distracted") else None
                idx = [i for i, x in enumerate(L) if x["clip"] == clip]
                w = np.array([L[i]["w"] for i in idx])
                p = np.array([st[i] == "distracted" for i in idx])
                rw[clip] = (w * p).sum() / w.sum()
            if not ok:
                continue
            foc = [i for i, x in enumerate(L) if x["label"] == "focused"]
            dis = [i for i, x in enumerate(L) if x["label"] == "distracted"]
            nm = sum(st[i] is None for i in st)
            fa = sum(st[i] == "distracted" for i in foc)
            ca = sum(st[i] == "distracted" for i in dis)
            mf = [i for i in foc if st[i] is not None]
            lo, hi = wilson(len(foc) - fa, len(foc))
            sp = (len(foc) - fa) / len(foc)
            tiles = "on" if t == "auto" else "off"
            r = dict(model=m, tiles=tiles, decision=name, not_matched=nm, fa=fa, focused=len(foc), spec=sp,
                     lo=lo, hi=hi, caught=ca, distracted=len(dis), matched_focused=len(mf),
                     **{f"rw_{c}": rw[c] for c in clips})
            out.append(r)
            print(f"| {m} | {tiles} | {name} | {nm} | {fa} / {len(foc)} | {100*sp:.0f}% ({100*lo:.0f}-{100*hi:.0f}%) | "
                  f"{ca} / {len(dis)} | {fa} / {len(mf)} ({100*fa/max(len(mf), 1):.0f}%) | "
                  + " / ".join(f"{100*rw[c]:.1f}%" for c in clips) + " |")
if a.csv and out:
    with open(a.csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
