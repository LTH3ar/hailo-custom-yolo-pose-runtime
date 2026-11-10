"""Engagement-analysis settings: defaults <- engagement YAML <- command-line flags."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

DISPLAY_MODES = ("students", "class", "both", "none")
CAMERA_POSITIONS = ("auto", "center", "left", "right")

# Expected head yaw of an attentive student for each mount position, in the
# feature's units (nose offset between the ears, -1..1, negative = nose toward
# image-left). "left"/"right" are as seen from the students' seats facing the
# board: a camera on the students' front-left sees attentive students turned
# to their right, i.e. nose toward image-left.
CAMERA_YAW_PRIOR = {"auto": 0.0, "center": 0.0, "left": -0.5, "right": 0.5}


@dataclass
class EngagementSettings:
    display: str = "both"  # students | class | both | none
    camera_position: str = "auto"  # auto | center | left | right
    calibration: Optional[str] = None  # JSON file with learned per-seat baselines
    recalibrate: bool = False  # ignore an existing calibration file and learn again
    calibration_seconds: float = 120.0  # how long each seat is observed to learn its baseline

    keypoint_threshold: float = 0.5  # keypoints below this visibility are ignored

    # decision over time
    ema_tau: float = 1.0  # seconds; smoothing of the per-frame distraction score
    on_threshold: float = 0.6  # smoothed score above this ...
    on_seconds: float = 2.5  # ... for this long -> distracted
    off_threshold: float = 0.4  # smoothed score below this ...
    off_seconds: float = 1.5  # ... for this long -> focused
    unknown_after: float = 2.0  # seconds without usable keypoints -> unknown (not counted)
    window_seconds: float = 5.0  # rolling window for the class percentage on screen

    # whole-class estimate (see analyzer._class_estimate)
    seat_forget_seconds: float = 60.0  # a seat not seen for this long no longer counts toward the class size
    min_presence: float = 0.15  # an uncalibrated seat must be detected in this share of frames to count
    presence_seconds: float = 15.0  # a seat counts with its last decision for this long after it was seen
    zones: int = 3  # near / middle / far rows, each weighted by its number of seats (1 = no weighting)
    class_size: Optional[int] = None  # real number of students, if known; else the number of seats found

    # rule weights (0 disables a cue). Cues combine as a noisy-OR, so a cue whose
    # weight is below on_threshold can't make a student "distracted" on its own,
    # only together with another cue.
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "head_turned": 1.0,  # head yaw away from the seat's attentive direction
            "face_hidden": 0.9,  # shoulders visible but no face: turned around / away
            "head_down": 0.7,  # lower weight: writing notes also looks like this
            "body_turned": 0.5,  # shoulders narrower than baseline: torso rotated
            "hand_at_face": 0.4,  # head resting on a hand, phone at face
        }
    )
    hand_raised_is_focused: bool = True  # a raised hand overrides the other cues

    # Per-frame decision: "auto" (= the bundled trees, or the rules if none are bundled),
    # "trees" | "logistic" | "mlp" (bundled learned models), "none" (rules only),
    # or a path to a classifier JSON.
    classifier: Optional[str] = "auto"

    # seat-anchored identity
    seat_match_radius: float = 0.6  # max center distance, in box widths, to reuse a seat
    seat_yaw_max_offset: float = 0.3  # how far a seat's attentive yaw may differ from the room's

    def validate(self) -> "EngagementSettings":
        if self.display not in DISPLAY_MODES:
            raise ValueError(f"display must be one of {DISPLAY_MODES}, got {self.display!r}")
        if self.camera_position not in CAMERA_POSITIONS:
            raise ValueError(f"camera_position must be one of {CAMERA_POSITIONS}, got {self.camera_position!r}")
        if not 0 <= self.off_threshold <= self.on_threshold <= 1:
            raise ValueError("need 0 <= off_threshold <= on_threshold <= 1")
        unknown = set(self.weights) - set(EngagementSettings().weights)
        if unknown:
            raise ValueError(f"unknown weights {sorted(unknown)}")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_settings(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> EngagementSettings:
    s = EngagementSettings()
    known = {f.name for f in fields(EngagementSettings)}
    if path:
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"engagement settings not found: {p}")
        import yaml

        data = yaml.safe_load(p.read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{p}: expected a mapping")
        bad = set(data) - known
        if bad:
            raise ValueError(f"{p}: unknown keys {sorted(bad)}; allowed: {sorted(known)}")
        for k, v in data.items():
            if k == "weights":
                s.weights.update(v or {})
            else:
                setattr(s, k, v)
    for k, v in (overrides or {}).items():
        if v is not None:
            setattr(s, k, v)
    return s.validate()
