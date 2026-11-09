"""Learned focused/distracted classifier: features + a dependency-free tree-ensemble runtime.

Training happens offline with scikit-learn (research/scb/train_classifier.py);
the fitted HistGradientBoostingClassifier is exported to JSON and evaluated here
with NumPy only. No scikit-learn on the Pi, and no pickle/version coupling.
All trees are traversed together, one vectorized step per tree level, so a
frame with 30 students costs well under a millisecond.

feature_vector() is the single definition of the classifier input, used by the
training script and by the live analyzer, so they can't drift apart.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from .features import NAMES, Baseline, Features, KeypointIndex

FEATURE_NAMES = (
    [f"{n}_{a}" for n in NAMES for a in ("x", "y")]  # keypoints relative to shoulder center / scale
    + [f"{n}_vis" for n in NAMES]
    + ["yaw_dev", "head_drop", "torso_drop", "face_hidden", "hand_raised", "hand_at_face"]
)
FEATURE_VERSION = 1


def feature_vector(kp: np.ndarray, box, f: Features, base: Baseline, index: KeypointIndex, thr: float) -> np.ndarray:
    """kp (K,3) pixels; f from features.compute(); base = attentive baseline for this student."""
    I = index.idx
    k = np.zeros((len(NAMES), 3), np.float32)
    for j, n in enumerate(NAMES):
        i = I[n]
        if i is not None and i < len(kp):
            k[j] = kp[i]
    ls, rs = k[5], k[6]
    if ls[2] >= thr and rs[2] >= thr:
        c = (ls[:2] + rs[:2]) / 2
    else:
        c = np.array([(box[0] + box[2]) / 2, box[1] + 0.4 * (box[3] - box[1])], np.float32)
    rel = (k[:, :2] - c) / max(f.scale, 1e-6)
    rel[k[:, 2] < thr] = 0.0
    nan = np.nan
    extra = [
        nan if f.yaw is None or base.yaw is None else f.yaw - base.yaw,
        nan if f.head_height is None or base.head_height is None else base.head_height - f.head_height,
        nan if f.torso_ratio is None or not base.torso_ratio else 1.0 - f.torso_ratio / base.torso_ratio,
        float(f.face_hidden), float(f.hand_raised), float(f.hand_at_face),
    ]
    return np.concatenate([rel.ravel(), k[:, 2], np.asarray(extra, np.float32)]).astype(np.float32)


class TreeEnsemble:
    """Binary gradient-boosted trees exported by export_hgb()."""

    def __init__(self, data: dict):
        if data.get("format") != "hcpose-hgb-1":
            raise ValueError("not an hcpose classifier file")
        if data.get("feature_version") != FEATURE_VERSION:
            raise ValueError(f"classifier was trained on feature version {data.get('feature_version')}, "
                             f"runtime has {FEATURE_VERSION}; retrain it")
        self.meta = data.get("meta", {})
        self.baseline = float(data["baseline"])
        trees = data["trees"]
        offs = np.cumsum([0] + [len(t["feature"]) for t in trees[:-1]])
        cat = lambda key, dt: np.concatenate([np.asarray(t[key], dt) for t in trees])  # noqa: E731
        self.feature = cat("feature", np.int32)
        self.threshold = cat("threshold", np.float64)
        self.left = np.concatenate([np.asarray(t["left"], np.int64) + o for t, o in zip(trees, offs)])
        self.right = np.concatenate([np.asarray(t["right"], np.int64) + o for t, o in zip(trees, offs)])
        self.is_leaf = cat("is_leaf", bool)
        self.value = cat("value", np.float64)
        self.missing_left = cat("missing_left", bool)
        self.roots = offs.astype(np.int64)
        self.depth = int(data.get("max_depth", 64))
        self.n_features = int(data["n_features"])

    @classmethod
    def load(cls, path: str) -> "TreeEnsemble":
        return cls(json.loads(Path(path).read_text()))

    def decision(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, np.float64))
        if X.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {X.shape[1]}")
        n = X.shape[0]
        node = np.broadcast_to(self.roots, (n, len(self.roots))).copy()
        rows = np.arange(n)[:, None]
        for _ in range(self.depth + 1):
            leaf = self.is_leaf[node]
            if leaf.all():
                break
            x = X[rows, self.feature[node]]
            go_left = np.where(np.isnan(x), self.missing_left[node], x <= self.threshold[node])
            node = np.where(leaf, node, np.where(go_left, self.left[node], self.right[node]))
        return self.baseline + self.value[node].sum(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.decision(X)))


def export_hgb(est, path: str, meta: Optional[dict] = None) -> None:
    """Write a fitted binary sklearn HistGradientBoostingClassifier to hcpose's JSON format."""
    if len(est.classes_) != 2:
        raise ValueError("only binary classifiers are supported")
    if getattr(est, "is_categorical_", None) is not None and np.any(est.is_categorical_):
        raise ValueError("categorical features are not supported")
    trees, max_depth = [], 0
    for it in est._predictors:
        nodes = it[0].nodes
        max_depth = max(max_depth, int(nodes["depth"].max()))
        trees.append({
            "feature": nodes["feature_idx"].tolist(),
            "threshold": [float(v) for v in nodes["num_threshold"]],
            "left": nodes["left"].tolist(),
            "right": nodes["right"].tolist(),
            "is_leaf": nodes["is_leaf"].astype(bool).tolist(),
            "value": [float(v) for v in nodes["value"]],
            "missing_left": nodes["missing_go_to_left"].astype(bool).tolist(),
        })
    data = {
        "format": "hcpose-hgb-1",
        "feature_version": FEATURE_VERSION,
        "feature_names": FEATURE_NAMES,
        "n_features": int(est.n_features_in_),
        "baseline": float(np.ravel(est._baseline_prediction)[0]),
        "max_depth": max_depth,
        "trees": trees,
        "meta": meta or {},
    }
    Path(path).write_text(json.dumps(data, separators=(",", ":")))


BUNDLED = Path(__file__).resolve().parent / "models" / "scb_classifier.json"


def resolve(setting: Optional[str]) -> Optional[TreeEnsemble]:
    """settings.classifier: 'auto' (bundled model if present, else rules), 'none', or a path."""
    if setting in (None, "none", "rules"):
        return None
    if setting == "auto":
        return TreeEnsemble.load(str(BUNDLED)) if BUNDLED.is_file() else None
    return TreeEnsemble.load(str(Path(setting).expanduser()))


def stack(vectors: List[np.ndarray]) -> np.ndarray:
    return np.vstack(vectors) if vectors else np.zeros((0, len(FEATURE_NAMES)), np.float32)
