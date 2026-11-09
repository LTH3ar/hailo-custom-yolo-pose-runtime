"""Train hcpose's learned focused/distracted classifier on SCB front-view students.

Inputs are pose caches from extract_hef.py (deployed HEF) or extract_pt.py.
Features come from hcpose.engagement.classifier.feature_vector(), the same code
the live analyzer runs. Context features that let a model recognise the *image*
(position in the frame, crowd size) are deliberately left out.

Compares three model families with 5-fold cross-validation grouped by lesson,
reports overall and within-image AUC, then fits the chosen family (gradient-
boosted trees) on all data and exports it as JSON for the Pi.

usage:
  python research/scb/train_classifier.py ~/scb/poses_hef_m.jsonl --out hcpose/engagement/models/scb_classifier.json
"""
import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import scbdata  # noqa: E402
from hcpose.engagement.classifier import FEATURE_NAMES, TreeEnsemble, export_hgb, feature_vector  # noqa: E402
from hcpose.engagement.features import Baseline, KeypointIndex, compute  # noqa: E402
from hcpose.presets import UPPER_BODY13_NAMES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("poses", nargs="+", help="pose caches; several are pooled (e.g. n, s and m)")
ap.add_argument("--out", help="export the final model here (JSON)")
ap.add_argument("--det-thr", type=float, default=0.3, help="detection confidence for students (default 0.3)")
ap.add_argument("--kpt-thr", type=float, default=0.5)
ap.add_argument("--name", default="scb", help="model name stored in the JSON")
args = ap.parse_args()

IDX = KeypointIndex(UPPER_BODY13_NAMES)
T = args.kpt_thr


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
    ug, ud = set(), set()
    for _, gi, di in pairs:
        if gi not in ug and di not in ud:
            ug.add(gi)
            ud.add(di)
            yield gi, di


X, y, groups, imgs, labels = [], [], [], [], []
for src_i, path in enumerate(args.poses):
    for d in scbdata.load(path, verbose=src_i == 0):
        dets = [x for x in d["dets"] if x["score"] >= args.det_thr]
        K = [np.asarray(x["kpts"], np.float32) for x in dets]
        if not is_front(K):
            continue
        feats = [compute(k, x["box"], IDX, T) for k, x in zip(K, dets)]
        base = Baseline.from_samples([f for f in feats if f.valid], min_n=3)  # class median, as before calibration
        for gi, di in match(d["gt"], dets):
            cls = d["gt"][gi][0]
            if cls not in scbdata.FOCUSED | scbdata.DISTRACTED or not feats[di].valid:
                continue
            X.append(feature_vector(K[di], dets[di]["box"], feats[di], base, IDX, T))
            y.append(cls in scbdata.DISTRACTED)
            groups.append(d["group"])  # pooled caches share images, so groups stay consistent
            imgs.append(f"{d['img']}")
            labels.append(cls)

X, y, groups, imgs, labels = np.vstack(X), np.array(y), np.array(groups), np.array(imgs), np.array(labels)
print(f"{len(y)} students ({len(args.poses)} pose cache(s)), {len(set(groups))} lesson groups, distracted {y.mean():.1%}")
mixed = np.zeros(len(y), bool)
for im in set(imgs):
    m = imgs == im
    mixed[m] = y[m].any() and (~y[m]).any()


def within_image_auc(s):
    wins = total = 0.0
    for im in set(imgs[mixed]):
        m = imgs == im
        diff = s[m & y][:, None] - s[m & ~y][None, :]
        wins += (diff > 0).sum() + 0.5 * (diff == 0).sum()
        total += diff.size
    return wins / total


def hgb():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, max_depth=8,
                                          l2_regularization=1.0, class_weight="balanced", random_state=0)


families = {
    "logistic regression": lambda: make_pipeline(SimpleImputer(), StandardScaler(),
                                                 LogisticRegression(max_iter=2000, class_weight="balanced")),
    "gradient-boosted trees": hgb,
    "MLP (64-32)": lambda: make_pipeline(SimpleImputer(), StandardScaler(),
                                         MLPClassifier((64, 32), max_iter=300, early_stopping=True, random_state=0)),
}
print(f"\n{'model':24} {'AUC':>6} {'within-img AUC':>15} {'bal. acc':>9}  (5-fold, grouped by lesson)")
cv = {}
for name, make in families.items():
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = make()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    # MLP has no class weights: compare at the prevalence-matched threshold for balanced accuracy
    thr = 0.5 if "MLP" not in name else np.quantile(oof, 1 - y.mean())
    cv[name] = {"auc": roc_auc_score(y, oof), "within_image_auc": within_image_auc(oof),
                "balanced_acc": balanced_accuracy_score(y, oof >= thr)}
    print(f"{name:24} {cv[name]['auc']:6.3f} {cv[name]['within_image_auc']:15.3f} {cv[name]['balanced_acc']:8.1%}")
    if name == "gradient-boosted trees":
        print("   per behavior, share predicted distracted:",
              ", ".join(f"{c} {np.mean(oof[labels == c] >= 0.5):.0%}" for c in
                        ["read", "write", "hand-raising", "TurnHead", "BowHead", "discuss"] if (labels == c).any()))

if args.out:
    t0 = time.time()
    final = hgb().fit(X, y)
    meta = {
        "name": args.name,
        "trained_on": ", ".join(Path(p).name for p in args.poses),
        "students": int(len(y)),
        "lesson_groups": int(len(set(groups))),
        "det_thr": args.det_thr,
        "cv_gradient_boosted_trees": {k: round(float(v), 4) for k, v in cv["gradient-boosted trees"].items()},
        "sklearn": __import__("sklearn").__version__,
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "host": platform.node(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    export_hgb(final, args.out, meta)
    rt = TreeEnsemble.load(args.out)
    err = np.abs(rt.predict_proba(X) - final.predict_proba(X)[:, 1]).max()
    print(f"\nexported {args.out} ({Path(args.out).stat().st_size / 1e6:.2f} MB, {len(final._predictors)} trees, "
          f"{time.time() - t0:.1f}s); NumPy runtime vs sklearn max |diff| = {err:.2e}")
    assert err < 1e-6, "exported model does not reproduce sklearn"
    assert len(FEATURE_NAMES) == X.shape[1]
