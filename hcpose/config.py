"""Model configuration: defaults <- sidecar file <- command-line overrides.

A sidecar is an optional YAML/JSON file next to the HEF with the same stem
(e.g. models/upper_body_n.hef + models/upper_body_n.yaml). Every key is
optional; anything missing is detected from the HEF or taken from presets.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .presets import preset_for

SIDECAR_SUFFIXES = (".yaml", ".yml", ".json")


@dataclass
class ModelConfig:
    hef: str
    sidecar: Optional[str] = None
    # detection
    score_threshold: float = 0.5
    iou_threshold: float = 0.7
    keypoint_threshold: float = 0.5
    max_detections: int = 100
    class_agnostic_nms: bool = True
    score_activation: str = "auto"  # auto | sigmoid | none
    keypoint_activation: str = "auto"
    # structure overrides (normally auto-detected from the HEF)
    num_keypoints: Optional[int] = None
    num_classes: Optional[int] = None
    reg_max: Optional[int] = None
    # presentation
    labels: List[str] = field(default_factory=list)
    keypoint_names: List[str] = field(default_factory=list)
    skeleton: List[List[int]] = field(default_factory=list)

    def fill_from_layout(self, num_keypoints: int, num_classes: int) -> None:
        preset = preset_for(num_keypoints)
        if not self.keypoint_names:
            self.keypoint_names = list(preset["keypoint_names"])
        if not self.skeleton:
            self.skeleton = [list(p) for p in preset["skeleton"]]
        if not self.labels:
            self.labels = ["person"] if num_classes == 1 else [f"class_{i}" for i in range(num_classes)]
        if len(self.keypoint_names) != num_keypoints:
            raise ValueError(
                f"sidecar lists {len(self.keypoint_names)} keypoint_names but the HEF has {num_keypoints} keypoints"
            )

    def postprocess_json(self, input_width: int, input_height: int) -> Dict[str, Any]:
        """Settings consumed by the C++ hailofilter post-process (init())."""
        d = {
            "score_threshold": self.score_threshold,
            "iou_threshold": self.iou_threshold,
            "keypoint_threshold": self.keypoint_threshold,
            "max_detections": self.max_detections,
            "class_agnostic_nms": self.class_agnostic_nms,
            "score_activation": self.score_activation,
            "keypoint_activation": self.keypoint_activation,
            "input_width": input_width,
            "input_height": input_height,
            "labels": self.labels,
            "skeleton": self.skeleton,
            "verbose": True,
        }
        for k in ("num_keypoints", "num_classes", "reg_max"):
            if getattr(self, k):
                d[k] = getattr(self, k)
        return d

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def find_sidecar(hef: Path) -> Optional[Path]:
    for suf in SIDECAR_SUFFIXES:
        p = hef.with_suffix(suf)
        if p.is_file():
            return p
    return None


def _read(path: Path) -> Dict[str, Any]:
    text = path.read_text()
    if path.suffix == ".json":
        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at top level")
    return data


def load_config(hef: str, sidecar: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> ModelConfig:
    hef_path = Path(hef).expanduser().resolve()
    if not hef_path.is_file():
        raise FileNotFoundError(f"HEF not found: {hef_path}")
    cfg = ModelConfig(hef=str(hef_path))
    side = Path(sidecar).expanduser().resolve() if sidecar else find_sidecar(hef_path)
    if sidecar and not side.is_file():
        raise FileNotFoundError(f"sidecar not found: {side}")
    known = set(ModelConfig.__dataclass_fields__) - {"hef", "sidecar"}
    if side:
        data = _read(side)
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"{side}: unknown keys {sorted(unknown)}; allowed: {sorted(known)}")
        for k, v in data.items():
            setattr(cfg, k, v)
        cfg.sidecar = str(side)
    for k, v in (overrides or {}).items():
        if v is not None:
            if k not in known:
                raise ValueError(f"unknown override {k}")
            setattr(cfg, k, v)
    return cfg
