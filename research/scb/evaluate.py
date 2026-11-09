"""Evaluate hcpose's rule-based engagement cues on SCB classroom images.

Single images, so this measures the per-frame rules in their *uncalibrated* mode
(baseline = median of the other students in the same image), without the temporal
smoothing and per-seat calibration the live system adds.

usage: evaluate.py POSES.jsonl [--out report.json]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from hcpose.engagement.features import Baseline, KeypointIndex, compute, score  # noqa: E402
from hcpose.engagement.settings import load_settings  # noqa: E402
from hcpose.presets import UPPER_BODY13_NAMES  # noqa: E402

FOCUSED = {"read", "write", "hand-raising"}
DISTRACTED = {"TurnHead", "BowHead", "discuss"}

ap = argparse.ArgumentParser()
ap.add_argument("poses")
ap.add_argument("--out")
ap.add_argument("--det-thr", type=float, default=0.5)
args = ap.parse_args()

S = load_settings()
IDX = KeypointIndex(UPPER_BODY13_NAMES)
THR = S.keypoint_threshold


def face_visible(k):
    return k[0, 2] >= THR and (k[1, 2] >= THR or k[2, 2] >= THR)


def overlap(gt, det):
    ix = max(0.0, min(gt[2], det[2]) - max(gt[0], det[0]))
    iy = max(0.0, min(gt[3], det[3]) - max(gt[1], det[1]))
    inter = ix * iy
    ga = (gt[2] - gt[0]) * (gt[3] - gt[1])
    da = (det[2] - det[0]) * (det[3] - det[1])
    return inter / max(ga, 1e-9), inter / max(ga + da - inter, 1e-9)


rows = []
views = Counter()
gt_total = Counter()
gt_matched = Counter()
for line in open(args.poses):
    d = json.loads(line)
    dets = [x for x in d["dets"] if x["score"] >= args.det_thr]
    K = [np.array(x["kpts"], dtype=np.float32) for x in dets]
    shoulders = [k for k in K if k[5, 2] >= THR and k[6, 2] >= THR]
    if len(shoulders) < 3:
        view = "few"
    else:
        frac = np.mean([face_visible(k) for k in shoulders])
        view = "front" if frac >= 0.6 else ("back" if frac <= 0.3 else "mixed")
    views[view] += 1
    feats = [compute(k, x["box"], IDX, THR) for k, x in zip(K, dets)]
    base = Baseline.from_samples([f for f in feats if f.valid], min_n=3)
    # greedy match SCB behavior boxes to pose detections
    pairs = []
    for gi, g in enumerate(d["gt"]):
        for di, x in enumerate(dets):
            cov, iou = overlap(g[1:], x["box"])
            if cov >= 0.5 or iou >= 0.3:
                pairs.append((iou + cov, gi, di))
    pairs.sort(reverse=True)
    used_g, used_d = set(), set()
    for _, gi, di in pairs:
        if gi in used_g or di in used_d:
            continue
        used_g.add(gi)
        used_d.add(di)
        f = feats[di]
        sc, reason = score(f, base, S.weights, S.hand_raised_is_focused) if f.valid else (None, None)
        cls = d["gt"][gi][0]
        rows.append({"view": view, "cls": cls, "score": sc, "reason": reason, "valid": f.valid,
                     "yaw_dev": None if f.yaw is None or base.yaw is None else abs(f.yaw - base.yaw),
                     "head_drop": None if f.head_height is None or base.head_height is None else base.head_height - f.head_height,
                     "face_hidden": f.face_hidden, "hand_raised": f.hand_raised, "n_students": len(dets),
                     "subset": d["subset"]})
    for g in d["gt"]:
        gt_total[(view, g[0])] += 1
    for gi in used_g:
        gt_matched[(view, d["gt"][gi][0])] += 1

print(f"images by camera view: {dict(views)}")
print("\nlabel -> matched to a pose detection (front view):")
for c in sorted({k[1] for k in gt_total}):
    t, m = gt_total[("front", c)], gt_matched[("front", c)]
    print(f"  {c:13} {m:6}/{t:<6} ({m / t:.0%})" if t else f"  {c:13} -")


def report(sel, title):
    print(f"\n=== {title}: {len(sel)} labelled students ===")
    print(f"  {'behavior':13} {'n':>6} {'pred distracted':>16}  top cues")
    for c in ["read", "write", "hand-raising", "TurnHead", "BowHead", "discuss"]:
        rs = [r for r in sel if r["cls"] == c and r["score"] is not None]
        if not rs:
            continue
        pd = np.mean([r["score"] >= S.on_threshold for r in rs])
        cues = Counter(r["reason"] for r in rs if r["score"] >= S.on_threshold).most_common(3)
        print(f"  {c:13} {len(rs):6} {pd:15.1%}   {', '.join(f'{k}:{v}' for k, v in cues)}")
    y = np.array([r["cls"] in DISTRACTED for r in sel if r["score"] is not None])
    s = np.array([r["score"] for r in sel if r["score"] is not None])
    if len(set(y)) < 2:
        return {}
    pred = s >= S.on_threshold
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    # AUC (rank-based)
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    auc = (ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
    print(f"  focused vs distracted @ on_threshold={S.on_threshold}: recall(distracted) {rec:.1%}, "
          f"specificity(focused) {spec:.1%}, precision {prec:.1%}, F1 {f1:.2f}, balanced acc {(rec + spec) / 2:.1%}, AUC {auc:.3f}")
    print(f"  confusion [rows true focused/distracted, cols pred focused/distracted]: [[{tn}, {fp}], [{fn}, {tp}]]")
    return {"recall": rec, "specificity": spec, "precision": prec, "f1": f1, "auc": float(auc), "n": int(len(s))}


res = {}
for v in ("front", "mixed", "back"):
    res[v] = report([r for r in rows if r["view"] == v], f"{v} view")
inv = sum(1 for r in rows if not r["valid"])
print(f"\n{inv} matched students had no usable keypoints (unknown)")

# Which single features separate the classes best (front view)?
fr = [r for r in rows if r["view"] == "front" and r["valid"]]
print("\nfront view, feature medians by behavior:")
for c in ["read", "write", "hand-raising", "TurnHead", "BowHead", "discuss"]:
    rs = [r for r in fr if r["cls"] == c]
    if not rs:
        continue
    yd = [r["yaw_dev"] for r in rs if r["yaw_dev"] is not None]
    hd = [r["head_drop"] for r in rs if r["head_drop"] is not None]
    print(f"  {c:13} yaw_dev {np.median(yd) if yd else float('nan'):5.2f}  head_drop {np.median(hd) if hd else float('nan'):5.2f}  "
          f"face_hidden {np.mean([r['face_hidden'] for r in rs]):4.0%}  hand_raised {np.mean([r['hand_raised'] for r in rs]):4.0%}")
if args.out:
    json.dump({"views": dict(views), "metrics": res, "rows": rows}, open(args.out, "w"))
