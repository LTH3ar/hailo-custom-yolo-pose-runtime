"""Is focused-vs-distracted learnable from upper-body keypoints? (SCB, front-view images)

Builds one feature vector per matched student and trains a gradient-boosted
classifier with GroupKFold over source videos (no frames of the same lesson in
train and test). Compares against the hand-written rules on the same students.

usage: learn.py POSES.jsonl
"""
import json
import re
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from hcpose.engagement.features import Baseline, KeypointIndex, compute, score  # noqa: E402
from hcpose.engagement.settings import load_settings  # noqa: E402
from hcpose.presets import UPPER_BODY13_NAMES  # noqa: E402

S = load_settings()
IDX = KeypointIndex(UPPER_BODY13_NAMES)
FOC = {"read", "write", "hand-raising"}
DIS = {"TurnHead", "BowHead", "discuss"}


def video_id(path, subset):
    name = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    m = re.match(r"(\d+)_", name)
    return f"{subset}:{m.group(1) if m else name[:4]}"


def kp_vector(k, box):
    """Keypoints relative to the shoulder midpoint, scaled by the feature scale; hidden -> 0 + flag."""
    f = compute(k, box, IDX, 0.5)
    if not f.valid:
        return None, f
    if k[5, 2] >= 0.5 and k[6, 2] >= 0.5:
        c = (k[5, :2] + k[6, :2]) / 2
    else:
        c = np.array([(box[0] + box[2]) / 2, box[1] + 0.4 * (box[3] - box[1])])
    rel = (k[:, :2] - c) / f.scale
    vis = k[:, 2]
    rel[vis < 0.5] = 0.0
    return np.concatenate([rel.ravel(), vis]), f


X, y, groups, rule_scores, cls_names = [], [], [], [], []
for line in open(sys.argv[1]):
    d = json.loads(line)
    dets = [x for x in d["dets"] if x["score"] >= 0.5]
    K = [np.array(x["kpts"], dtype=np.float32) for x in dets]
    sh = [k for k in K if k[5, 2] >= 0.5 and k[6, 2] >= 0.5]
    if len(sh) < 3 or np.mean([k[0, 2] >= 0.5 and (k[1, 2] >= 0.5 or k[2, 2] >= 0.5) for k in sh]) < 0.6:
        continue  # front-view images only
    vecs = [kp_vector(k, x["box"]) for k, x in zip(K, dets)]
    feats = [f for _, f in vecs]
    base = Baseline.from_samples([f for f in feats if f.valid], min_n=3)
    pairs = []
    for gi, g in enumerate(d["gt"]):
        if g[0] not in FOC | DIS:
            continue
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
        if gi in ug or di in ud:
            continue
        ug.add(gi)
        ud.add(di)
        v, f = vecs[di]
        if v is None:
            continue
        b = dets[di]["box"]
        ctx = [
            (b[0] + b[2]) / 2 / d["w"], (b[1] + b[3]) / 2 / d["h"], (b[2] - b[0]) / d["w"],  # where in the room
            np.nan if f.yaw is None else f.yaw,
            np.nan if f.yaw is None or base.yaw is None else f.yaw - base.yaw,
            np.nan if f.head_height is None or base.head_height is None else f.head_height - base.head_height,
            float(f.face_hidden), float(f.hand_raised), float(f.hand_at_face), len(dets),
        ]
        X.append(np.concatenate([v, ctx]))
        y.append(d["gt"][gi][0] in DIS)
        cls_names.append(d["gt"][gi][0])
        groups.append(video_id(d["img"], d["subset"]))
        rule_scores.append(score(f, base, S.weights, S.hand_raised_is_focused)[0])

X, y, groups = np.array(X, dtype=np.float32), np.array(y), np.array(groups)
rule_scores, cls_names = np.array(rule_scores), np.array(cls_names)
print(f"{len(y)} students from {len(set(groups))} videos; distracted share {y.mean():.1%}")

oof = np.zeros(len(y))
for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         class_weight="balanced", random_state=0)
    clf.fit(X[tr], y[tr])
    oof[te] = clf.predict_proba(X[te])[:, 1]

print(f"rules   : AUC {roc_auc_score(y, rule_scores):.3f}  balanced acc {balanced_accuracy_score(y, rule_scores >= S.on_threshold):.1%}")
print(f"learned : AUC {roc_auc_score(y, oof):.3f}  balanced acc {balanced_accuracy_score(y, oof >= 0.5):.1%}  (5-fold, split by video)")
print("\nper behavior, share predicted distracted (learned @0.5 | rules):")
for c in ["read", "write", "hand-raising", "TurnHead", "BowHead", "discuss"]:
    m = cls_names == c
    if m.any():
        print(f"  {c:13} n={m.sum():5}  {np.mean(oof[m] >= 0.5):5.1%} | {np.mean(rule_scores[m] >= S.on_threshold):5.1%}")
# what does the model rely on? group importance by dropping feature blocks
blocks = {"keypoint geometry": range(0, 26), "visibilities": range(26, 39), "position in room": range(39, 42),
          "rule features (yaw, head, face, hands)": range(42, 49)}
print("\nAUC when one block is removed (5-fold):")
for name, idx in blocks.items():
    keep = [i for i in range(X.shape[1]) if i not in idx]
    o = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)
        clf.fit(X[tr][:, keep], y[tr])
        o[te] = clf.predict_proba(X[te][:, keep])[:, 1]
    print(f"  without {name:40} AUC {roc_auc_score(y, o):.3f}")
