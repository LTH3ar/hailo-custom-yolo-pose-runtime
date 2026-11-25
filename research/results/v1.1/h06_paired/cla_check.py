"""Handoff 06 check (uncommitted): ClaBehavior strict, pose MLP/trees minus crop AUC, by camera group;
and the bootstrap distribution's shape. Uses baselines.crop_cls.paired's own loaders."""
import json, sys
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score
from classifiers.common import mapping
from baselines.crop_cls.paired import crop_scores, cls_scores

H = sys.argv[1]; SP = sys.argv[2]; PR = sys.argv[3]
S = json.load(open(SP)); students = {s["id"]: s for s in S["students"]}
preds = [json.loads(l) for l in open(PR)]
f, d = mapping.get("strict")
crop = crop_scores(students, preds, f, d)
fams = cls_scores(f"{H}/cv_strict.npz")
for fam in ("trees", "MLP"):
    pose = fams[fam]
    ks = sorted(k for k in crop.keys() & pose.keys() if crop[k][3] == "clabehavior" and crop[k][0] == pose[k][0])
    y = np.array([crop[k][0] for k in ks]); a = np.array([crop[k][1] for k in ks]); b = np.array([pose[k][1] for k in ks])
    g = np.array([crop[k][2] for k in ks])
    cnt = Counter(g.tolist()); big = [x for x, _ in cnt.most_common(2)]
    print(f"{fam}: n {len(ks)}, groups {len(cnt)}; two largest groups hold {cnt[big[0]]} + {cnt[big[1]]} students "
          f"({(cnt[big[0]] + cnt[big[1]]) / len(ks):.0%})")
    def diff(m):
        return roc_auc_score(y[m], b[m]) - roc_auc_score(y[m], a[m]), roc_auc_score(y[m], a[m]), roc_auc_score(y[m], b[m])
    for name, m in (("all", np.ones(len(y), bool)), ("without group 1", g != big[0]), ("without group 2", g != big[1]),
                    ("without both", (g != big[0]) & (g != big[1])), ("group 1 only", g == big[0]), ("group 2 only", g == big[1])):
        dd, ca, pb = diff(m)
        print(f"  {name:16} n {m.sum():5d}  crop {ca:.3f}  pose {pb:.3f}  diff {dd:+.3f}")
    rep = json.load(open(f"{H}/paired.json"))[f"strict/clabehavior/{fam}"]
    print("  paired.json:", {k: rep[k] for k in ("n", "groups", "ci_pose_minus_crop")})
