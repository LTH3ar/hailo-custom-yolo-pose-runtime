"""Learned focused/distracted classifiers: features + dependency-free runtimes.

Training happens offline with scikit-learn (research/scb/train_classifier.py).
Three model kinds are exported to JSON and evaluated here with NumPy only, so
there is no scikit-learn on the Pi and no pickle/version coupling:

  trees     HistGradientBoostingClassifier (default). All trees are traversed
            together, one vectorized step per tree level.
  logistic  SimpleImputer -> StandardScaler -> LogisticRegression
  mlp       SimpleImputer -> StandardScaler -> MLPClassifier

Every model returns a "balanced" probability: 0.5 means equally likely under
equal class priors. Trees and logistic regression are trained with balanced
class weights; the MLP can't be, so its logit is shifted by the training prior
(logit_offset). That keeps the analyzer's on/off thresholds meaningful for all
three. A frame with 30 students costs well under a millisecond with any of them.

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


class _Model:
    kind = "?"

    def _common(self, data: dict, fmt: str) -> None:
        if data.get("format") != fmt:
            raise ValueError(f"not an hcpose {self.kind} classifier file (format {data.get('format')!r})")
        if data.get("feature_version") != FEATURE_VERSION:
            raise ValueError(f"classifier was trained on feature version {data.get('feature_version')}, "
                             f"runtime has {FEATURE_VERSION}; retrain it")
        self.meta = data.get("meta", {})
        self.n_features = int(data["n_features"])
        self.logit_offset = float(data.get("logit_offset", 0.0))

    def _check(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, np.float64))
        if X.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {X.shape[1]}")
        return X

    def decision(self, X: np.ndarray) -> np.ndarray:  # raw logit, before logit_offset
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.decision(X) + self.logit_offset)))

    @classmethod
    def load(cls, path: str):
        return cls(json.loads(Path(path).read_text()))


class TreeEnsemble(_Model):
    """Binary gradient-boosted trees exported by export_hgb()."""

    kind = "trees"

    def __init__(self, data: dict):
        self._common(data, "hcpose-hgb-1")
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

    def decision(self, X: np.ndarray) -> np.ndarray:
        X = self._check(X)
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


class _Scaled(_Model):
    """Shared preprocessing of the logistic and MLP pipelines: mean-impute, then standardize."""

    def _prep(self, data: dict) -> None:
        self.impute = np.asarray(data["impute"], np.float64)
        self.mean = np.asarray(data["mean"], np.float64)
        self.scale = np.asarray(data["scale"], np.float64)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = self._check(X).copy()
        nan = np.isnan(X)
        if nan.any():
            X[nan] = np.broadcast_to(self.impute, X.shape)[nan]
        return (X - self.mean) / self.scale


class LogisticModel(_Scaled):
    kind = "logistic"

    def __init__(self, data: dict):
        self._common(data, "hcpose-linear-1")
        self._prep(data)
        self.coef = np.asarray(data["coef"], np.float64)
        self.intercept = float(data["intercept"])

    def decision(self, X: np.ndarray) -> np.ndarray:
        return self._transform(X) @ self.coef + self.intercept


_ACT = {
    "relu": lambda z: np.maximum(z, 0.0),
    "tanh": np.tanh,
    "logistic": lambda z: 1.0 / (1.0 + np.exp(-z)),
    "identity": lambda z: z,
}


class MLPModel(_Scaled):
    kind = "mlp"

    def __init__(self, data: dict):
        self._common(data, "hcpose-mlp-1")
        self._prep(data)
        self.layers = [(np.asarray(L["W"], np.float64), np.asarray(L["b"], np.float64)) for L in data["layers"]]
        if data["activation"] not in _ACT:
            raise ValueError(f"unsupported activation {data['activation']}")
        self.act = _ACT[data["activation"]]

    def decision(self, X: np.ndarray) -> np.ndarray:
        h = self._transform(X)
        for W, b in self.layers[:-1]:
            h = self.act(h @ W + b)
        W, b = self.layers[-1]
        return (h @ W + b).ravel()  # output unit is logistic: this is its logit


_FORMATS = {"hcpose-hgb-1": TreeEnsemble, "hcpose-linear-1": LogisticModel, "hcpose-mlp-1": MLPModel}


def load_model(path: str) -> _Model:
    data = json.loads(Path(path).read_text())
    cls = _FORMATS.get(data.get("format"))
    if cls is None:
        raise ValueError(f"{path}: not an hcpose classifier file (format {data.get('format')!r})")
    return cls(data)


# ---------------------------------------------------------------------------- export
def _header(fmt: str, n_features: int, meta: Optional[dict], logit_offset: float) -> dict:
    return {
        "format": fmt,
        "feature_version": FEATURE_VERSION,
        "feature_names": FEATURE_NAMES,
        "n_features": int(n_features),
        "logit_offset": float(logit_offset),
        "meta": meta or {},
    }


def export_hgb(est, path: str, meta: Optional[dict] = None, logit_offset: float = 0.0) -> None:
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
    data = _header("hcpose-hgb-1", est.n_features_in_, meta, logit_offset)
    data.update(baseline=float(np.ravel(est._baseline_prediction)[0]), max_depth=max_depth, trees=trees)
    Path(path).write_text(json.dumps(data, separators=(",", ":")))


def _pipeline_parts(pipe):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    steps = [s for _, s in pipe.steps]
    if len(steps) != 3 or not isinstance(steps[0], SimpleImputer) or not isinstance(steps[1], StandardScaler):
        raise ValueError("expected a pipeline SimpleImputer -> StandardScaler -> model")
    imp, sc, model = steps
    if imp.strategy != "mean" or not getattr(imp, "keep_empty_features", False):
        # without keep_empty_features an all-NaN training column would be silently dropped
        raise ValueError("SimpleImputer must use strategy='mean' and keep_empty_features=True")
    if len(imp.statistics_) != len(sc.mean_):
        raise ValueError("imputer and scaler disagree on the number of features")
    scale = np.asarray(sc.scale_ if sc.with_std else np.ones_like(sc.mean_), np.float64)
    mean = np.asarray(sc.mean_ if sc.with_mean else np.zeros_like(sc.mean_), np.float64)
    return imp, mean, scale, model


def export_pipeline(pipe, path: str, meta: Optional[dict] = None, logit_offset: float = 0.0) -> None:
    """Write a fitted SimpleImputer -> StandardScaler -> LogisticRegression | MLPClassifier pipeline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    imp, mean, scale, model = _pipeline_parts(pipe)
    if len(model.classes_) != 2:
        raise ValueError("only binary classifiers are supported")
    prep = {"impute": [float(v) for v in imp.statistics_], "mean": mean.tolist(), "scale": scale.tolist()}
    if isinstance(model, LogisticRegression):
        data = _header("hcpose-linear-1", len(mean), meta, logit_offset)
        data.update(prep, coef=[float(v) for v in np.ravel(model.coef_)], intercept=float(np.ravel(model.intercept_)[0]))
    elif isinstance(model, MLPClassifier):
        if model.out_activation_ != "logistic":
            raise ValueError(f"unexpected MLP output activation {model.out_activation_}")
        data = _header("hcpose-mlp-1", len(mean), meta, logit_offset)
        data.update(prep, activation=model.activation,
                    layers=[{"W": np.asarray(W, np.float64).tolist(), "b": np.asarray(b, np.float64).tolist()}
                            for W, b in zip(model.coefs_, model.intercepts_)])
    else:
        raise ValueError(f"unsupported model {type(model).__name__}")
    Path(path).write_text(json.dumps(data, separators=(",", ":")))


def export_model(est, path: str, meta: Optional[dict] = None, logit_offset: float = 0.0) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier

    if isinstance(est, HistGradientBoostingClassifier):
        export_hgb(est, path, meta, logit_offset)
    else:
        export_pipeline(est, path, meta, logit_offset)


# ---------------------------------------------------------------------------- bundled models
MODELS_DIR = Path(__file__).resolve().parent / "models"
BUNDLED = {k: MODELS_DIR / f"scb_{k}.json" for k in ("trees", "logistic", "mlp")}
LEGACY_TREES = MODELS_DIR / "scb_classifier.json"
CHOICES = ("auto", "trees", "logistic", "mlp", "none")


def resolve(setting: Optional[str]) -> Optional[_Model]:
    """settings.classifier:
    auto      the rules (the default). The SCB-trained models rank SCB actions well but call
              head-up listening "distracted" in real lectures (research/scb/README.md), so they
              are opt-in.
    trees | logistic | mlp   the bundled model of that kind (error if not exported yet)
    none      the rules
    a path    any hcpose classifier JSON
    """
    if setting in (None, "none", "rules", "auto"):
        return None
    if setting in BUNDLED:
        p = BUNDLED[setting]
        if not p.is_file():
            raise FileNotFoundError(
                f"no bundled {setting} classifier ({p}). Export it with research/scb/train_classifier.py "
                "--export-dir hcpose/engagement/models")
        return load_model(str(p))
    return load_model(str(Path(setting).expanduser()))


def describe(model: Optional[_Model]) -> str:
    if model is None:
        return "rules"
    return f"{model.kind} ({model.meta.get('name', 'custom')})"


def stack(vectors: List[np.ndarray]) -> np.ndarray:
    return np.vstack(vectors) if vectors else np.zeros((0, len(FEATURE_NAMES)), np.float32)
