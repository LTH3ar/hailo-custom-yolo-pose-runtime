"""Do the per-student classifiers behave the same on tiled and untiled poses?

Tiling finds many far-row students the full frame misses. The live Pi check saw the
class's distracted share drop a lot with tiles on, so this measures, on SCB front-view
students, for each classifier (rules, trees, logistic, MLP):
  * accuracy (AUC, balanced accuracy) on the students found in both runs, with their
    untiled vs tiled pose (does tiling change the pose the classifier sees?)
  * accuracy on the students only tiling finds
  * predicted vs true distracted share in each group (is any group biased?)

usage:
  python research/scb/eval_tiling.py none.jsonl tiles.jsonl [--det-thr 0.3]
Both caches must hold the same images (extract_pt.py / extract_hef.py with and without --tiles).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import scbdata  # noqa: E402
from hcpose.engagement.classifier import feature_vector, resolve  # noqa: E402
from hcpose.engagement.features import Baseline, KeypointIndex, compute, score  # noqa: E402
from hcpose.engagement.settings import load_settings  # noqa: E402
from hcpose.presets import UPPER_BODY13_NAMES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("none")
ap.add_argument("tiles")
ap.add_argument("--det-thr", type=float, default=0.3)
ap.add_argument("--kpt-thr", type=float, default=0.5)
args = ap.parse_args()
IDX, T = KeypointIndex(UPPER_BODY13_NAMES), args.kpt_thr
S = load_settings(overrides={"classifier": "none"})
MODELS = {k: resolve(k) for k in ("trees", "logistic", "mlp")}


def is_front(K):
    sh = [k for k in K if k[5, 2] >= T and k[6, 2] >= T]
    return len(sh) >= 3 and np.mean([k[0, 2] >= T and (k[1, 2] >= T or k[2, 2] >= T) for k in sh]) >= 0.6


def match(gt, dets):
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
    ug, ud, out = set(), set(), {}
    for _, gi, di in pairs:
        if gi not in ug and di not in ud:
            ug.add(gi)
            ud.add(di)
            out[gi] = di
    return out


def students(d):
    """gt index -> (feature vector, rule score) for the labelled, matched, valid students."""
    dets = [x for x in d["dets"] if x["score"] >= args.det_thr]
    K = [np.asarray(x["kpts"], np.float32) for x in dets]
    feats = [compute(k, x["box"], IDX, T) for k, x in zip(K, dets)]
    base = Baseline.from_samples([f for f in feats if f.valid], min_n=3)
    out = {}
    for gi, di in match(d["gt"], dets).items():
        if d["gt"][gi][0] in scbdata.FOCUSED | scbdata.DISTRACTED and feats[di].valid:
            out[gi] = (feature_vector(K[di], dets[di]["box"], feats[di], base, IDX, T),
                       score(feats[di], base, S.weights, S.hand_raised_is_focused)[0])
    return out, is_front(K)


A = {d["img"]: d for d in scbdata.load(args.none, verbose=False)}
B = {d["img"]: d for d in scbdata.load(args.tiles, verbose=False)}
groups = {"both: untiled pose": [], "both: tiled pose": [], "only with tiles": [], "only without tiles": []}
for img in sorted(set(A) & set(B)):
    sa, fa = students(A[img])
    sb, fb = students(B[img])
    if not (fa or fb):
        continue
    for gi in set(sa) | set(sb):
        y = A[img]["gt"][gi][0] in scbdata.DISTRACTED
        if gi in sa and gi in sb:
            groups["both: untiled pose"].append((*sa[gi], y, A[img]["group"]))
            groups["both: tiled pose"].append((*sb[gi], y, A[img]["group"]))
        elif gi in sb:
            groups["only with tiles"].append((*sb[gi], y, A[img]["group"]))
        else:
            groups["only without tiles"].append((*sa[gi], y, A[img]["group"]))

print(f"front-view students, det-thr {args.det_thr}. 'pred' = share labelled distracted at score 0.5\n")
print(f"{'group':22} {'n':>6} {'true':>6} | " + " | ".join(f"{m:>8} AUC  bal  pred" for m in ["rules", *MODELS]))
for g, rows in groups.items():
    if len(rows) < 20:
        continue
    X = np.vstack([r[0] for r in rows])
    rule = np.array([r[1] for r in rows], float)
    y = np.array([r[2] for r in rows])
    cells = []
    for name, s, thr in [("rules", rule, 0.5), *[(k, m.predict_proba(X), 0.5) for k, m in MODELS.items()]]:
        auc = roc_auc_score(y, s) if 0 < y.mean() < 1 else float("nan")
        cells.append(f"{auc:13.3f} {balanced_accuracy_score(y, s >= thr):4.0%} {np.mean(s >= thr):5.0%}")
    print(f"{g:22} {len(y):6d} {y.mean():6.0%} | " + " | ".join(cells))
