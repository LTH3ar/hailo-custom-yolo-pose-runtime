"""Train hcpose's learned focused/distracted classifier on SCB front-view students.

Inputs are pose caches from extract_hef.py (deployed HEF) or extract_pt.py.
Features come from hcpose.engagement.classifier.feature_vector(), the same code
the live analyzer runs. Context features that let a model recognise the *image*
(position in the frame, crowd size) are deliberately left out.

Compares three model families with 5-fold cross-validation grouped by lesson,
reports overall and within-image AUC, then fits the chosen family (gradient-
boosted trees) on all data and exports it as JSON for the Pi.

usage:
  python research/scb/train_classifier.py ~/scb/poses_hef_m.jsonl --export-dir hcpose/engagement/models
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
from hcpose.engagement.classifier import FEATURE_NAMES, export_model, feature_vector, load_model  # noqa: E402
from hcpose.engagement.features import Baseline, KeypointIndex, compute  # noqa: E402
from hcpose.presets import UPPER_BODY13_NAMES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("poses", nargs="+", help="pose caches; several are pooled (e.g. n, s and m)")
ap.add_argument("--out", help="export the final gradient-boosted trees here (JSON)")
ap.add_argument("--export-dir", metavar="DIR",
                help="fit all three models on all data and export scb_trees.json, scb_logistic.json and "
                     "scb_mlp.json here (e.g. hcpose/engagement/models)")
ap.add_argument("--det-thr", type=float, default=0.3, help="detection confidence for students (default 0.3)")
ap.add_argument("--kpt-thr", type=float, default=0.5)
ap.add_argument("--name", default="scb", help="model name stored in the JSON")
ap.add_argument("--save-cv", metavar="NPZ",
                help="save every family's out-of-fold predictions + labels + metrics (for ROC curves, tables in a report)")
ap.add_argument("--save-models", metavar="DIR",
                help="also fit logistic regression and the MLP on all data and pickle them here (for the record)")
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


def imputer():
    # keep_empty_features: an all-NaN column must stay in place, or the exported model's
    # feature indices would shift (it doesn't occur with these features, but be safe)
    return SimpleImputer(strategy="mean", keep_empty_features=True)


families = {
    "logistic regression": lambda: make_pipeline(imputer(), StandardScaler(),
                                                 LogisticRegression(max_iter=2000, class_weight="balanced")),
    "gradient-boosted trees": hgb,
    "MLP (64-32)": lambda: make_pipeline(imputer(), StandardScaler(),
                                         MLPClassifier((64, 32), max_iter=300, early_stopping=True, random_state=0)),
}
KIND = {"logistic regression": "logistic", "gradient-boosted trees": "trees", "MLP (64-32)": "mlp"}


def prior_offset(name, y_train):
    """The MLP can't use class weights, so its probabilities follow the training prior.
    Shifting its logit by log((1-p)/p) gives the balanced probability the other two produce,
    so one set of analyzer thresholds works for all models."""
    if KIND[name] != "mlp":
        return 0.0
    p = float(np.mean(y_train))
    return float(np.log((1 - p) / p))


def shift(prob, offset):
    if offset == 0.0:
        return prob
    q = np.clip(prob, 1e-12, 1 - 1e-12)
    return 1.0 / (1.0 + np.exp(-(np.log(q / (1 - q)) + offset)))
print(f"\n{'model':24} {'AUC':>6} {'within-img AUC':>15} {'bal. acc':>9}  (5-fold, grouped by lesson)")
cv, oofs = {}, {}
for name, make in families.items():
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = make()
        m.fit(X[tr], y[tr])
        oof[te] = shift(m.predict_proba(X[te])[:, 1], prior_offset(name, y[tr]))
    thr = 0.5  # all three are on the balanced-probability scale (see prior_offset)
    oofs[name] = oof
    cv[name] = {"auc": roc_auc_score(y, oof), "within_image_auc": within_image_auc(oof),
                "balanced_acc": balanced_accuracy_score(y, oof >= thr)}
    print(f"{name:24} {cv[name]['auc']:6.3f} {cv[name]['within_image_auc']:15.3f} {cv[name]['balanced_acc']:8.1%}")
    if name == "gradient-boosted trees":
        print("   per behavior, share predicted distracted:",
              ", ".join(f"{c} {np.mean(oof[labels == c] >= 0.5):.0%}" for c in
                        ["read", "write", "hand-raising", "TurnHead", "BowHead", "discuss"] if (labels == c).any()))

if args.save_cv:
    import json

    np.savez_compressed(
        args.save_cv, y=y, labels=labels, groups=groups, images=imgs, mixed_image=mixed,
        feature_names=np.array(FEATURE_NAMES), X=X,
        **{f"oof_{k.split(' ')[0].lower().replace('-', '_')}": v for k, v in oofs.items()},
    )
    Path(args.save_cv).with_suffix(".json").write_text(json.dumps(
        {"poses": args.poses, "det_thr": args.det_thr, "students": int(len(y)),
         "lesson_groups": int(len(set(groups))), "metrics": cv}, indent=1))
    print(f"\nsaved out-of-fold predictions to {args.save_cv} and metrics to {Path(args.save_cv).with_suffix('.json')}")

if args.save_models:
    import pickle

    d = Path(args.save_models)
    d.mkdir(parents=True, exist_ok=True)
    for name, make in families.items():
        m = make().fit(X, y)
        fn = d / (name.split(" ")[0].lower().replace("-", "_") + ".pkl")
        pickle.dump({"model": m, "feature_names": FEATURE_NAMES, "sklearn": __import__("sklearn").__version__}, open(fn, "wb"))
        print(f"saved {fn}")

def export_final(name, path):
    t0 = time.time()
    final = families[name]().fit(X, y)
    off = prior_offset(name, y)
    meta = {
        "name": args.name,
        "kind": KIND[name],
        "trained_on": ", ".join(Path(p).name for p in args.poses),
        "students": int(len(y)),
        "lesson_groups": int(len(set(groups))),
        "det_thr": args.det_thr,
        "cv": {k: round(float(v), 4) for k, v in cv[name].items()},
        "sklearn": __import__("sklearn").__version__,
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "host": platform.node(),
    }
    path = Path(path)
    previous = None
    if path.is_file():
        try:
            previous = load_model(str(path)).predict_proba(X)
        except (ValueError, KeyError) as e:
            print(f"  (existing {path.name} not comparable: {e})")
    path.parent.mkdir(parents=True, exist_ok=True)
    export_model(final, str(path), meta, logit_offset=off)
    rt = load_model(str(path))
    want = shift(final.predict_proba(X)[:, 1], off)
    err = np.abs(rt.predict_proba(X) - want).max()
    note = ""
    if previous is not None:
        d = np.abs(previous - rt.predict_proba(X)).max()
        note = f"; vs previous {path.name}: max |diff| {d:.1e}" + (" (identical)" if d < 1e-9 else "")
    print(f"exported {path} ({path.stat().st_size / 1e6:.2f} MB, {time.time() - t0:.1f}s); "
          f"NumPy runtime vs sklearn max |diff| = {err:.1e}{note}")
    assert err < 1e-5, f"exported {KIND[name]} model does not reproduce sklearn"
    assert len(FEATURE_NAMES) == X.shape[1]


if args.export_dir or args.out:
    print()
if args.export_dir:
    for name in families:
        export_final(name, Path(args.export_dir) / f"scb_{KIND[name]}.json")
elif args.out:
    export_final("gradient-boosted trees", args.out)
