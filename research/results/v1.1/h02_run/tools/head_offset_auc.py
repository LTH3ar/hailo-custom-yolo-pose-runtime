"""Handoff 02 C5 (ad hoc, uncommitted): how well the nose-vs-shoulders head offset separates SCB classes
on single images, measured as |offset - image median offset| (evaluate.py baseline, front view, same matching).
usage: head_offset_auc.py POSES.jsonl [--det-thr 0.3]"""
import argparse, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
R = Path.home() / "hailo-custom-yolo-pose-runtime"
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "research/scb"))
from hcpose.engagement.features import Baseline, KeypointIndex, compute
from hcpose.engagement.settings import load_settings
from hcpose.presets import UPPER_BODY13_NAMES
import scbdata
ap = argparse.ArgumentParser(); ap.add_argument("poses"); ap.add_argument("--det-thr", type=float, default=0.3)
a = ap.parse_args()
S = load_settings(); IDX = KeypointIndex(UPPER_BODY13_NAMES); THR = S.keypoint_threshold
def fv(k): return k[0, 2] >= THR and (k[1, 2] >= THR or k[2, 2] >= THR)
def overlap(g, d):
    ix = max(0.0, min(g[2], d[2]) - max(g[0], d[0])); iy = max(0.0, min(g[3], d[3]) - max(g[1], d[1])); i = ix * iy
    ga = (g[2] - g[0]) * (g[3] - g[1]); da = (d[2] - d[0]) * (d[3] - d[1])
    return i / max(ga, 1e-9), i / max(ga + da - i, 1e-9)
rows = []
for d in scbdata.load(a.poses, dedup=True):
    dets = [x for x in d["dets"] if x["score"] >= a.det_thr]
    K = [np.array(x["kpts"], dtype=np.float32) for x in dets]
    sh = [k for k in K if k[5, 2] >= THR and k[6, 2] >= THR]
    if len(sh) < 3 or np.mean([fv(k) for k in sh]) < 0.6: continue  # front view only
    feats = [compute(k, x["box"], IDX, THR) for k, x in zip(K, dets)]
    base = Baseline.from_samples([f for f in feats if f.valid], min_n=3)
    pairs = sorted(((sum(overlap(g[1:], x["box"])), gi, di) for gi, g in enumerate(d["gt"]) for di, x in enumerate(dets)
                    if overlap(g[1:], x["box"])[0] >= 0.5 or overlap(g[1:], x["box"])[1] >= 0.3), reverse=True)
    ug, ud = set(), set()
    for _, gi, di in pairs:
        if gi in ug or di in ud: continue
        ug.add(gi); ud.add(di); f = feats[di]
        if f.head_offset is None or base.head_offset is None: continue
        y = None if f.yaw is None or base.yaw is None else abs(f.yaw - base.yaw)
        rows.append((d["gt"][gi][0], abs(f.head_offset - base.head_offset), y))
cls = np.array([r[0] for r in rows]); off = np.array([r[1] for r in rows])
def auc(pos, neg):
    m = np.isin(cls, pos) | np.isin(cls, neg)
    return roc_auc_score(np.isin(cls[m], pos), off[m]), int(np.isin(cls, pos).sum()), int(np.isin(cls, neg).sum())
print(f"{len(rows)} front-view students with a head offset ({a.poses})")
for name, pos, neg in [("TurnHead vs read", ["TurnHead"], ["read"]), ("TurnHead vs read+write", ["TurnHead"], ["read", "write"]),
                       ("TurnHead vs all focused", ["TurnHead"], ["read", "write", "hand-raising"]),
                       ("all distracted vs all focused", ["TurnHead", "BowHead", "discuss"], ["read", "write", "hand-raising"])]:
    v, p, n = auc(pos, neg); print(f"  {name:32} AUC {v:.3f}  (n {p} vs {n})")
