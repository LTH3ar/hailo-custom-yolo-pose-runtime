"""Pose features for engagement, computed from one person's keypoints.

All distances are divided by the person's own scale (shoulder width, or a
head-size fallback) so features don't depend on how far the student sits from
the camera. Image coordinates: x to the right, y down.

Features are deliberately simple and explainable:
  yaw         nose offset between the ears (or eyes), -1.5..1.5. 0 = facing the
              camera, negative = nose toward image-left.
  head_height shoulder line minus nose height (positive = head above shoulders);
              drops when the head goes down.
  torso_ratio shoulder width / box height; drops when the torso rotates.
  face_hidden shoulders visible but neither nose nor eyes: turned away.
  hand_raised a wrist above the head.
  hand_at_face a wrist close to the nose.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
)


# (nose - eye midpoint) / eye distance is about 0.42x the ear-based yaw for a
# typical head geometry; this brings the eye-based fallback to ear units.
# Worth re-fitting on real data (fit ear-yaw vs eye-yaw where both are visible).
EYE_TO_EAR_UNITS = 2.4
# Half the ear-to-ear distance relative to the person's scale (shoulder width),
# used when only one ear is visible.
HALF_HEAD_WIDTH = 0.35


class KeypointIndex:
    """Maps the keypoint names the analysis needs to indices of the model's keypoints."""

    def __init__(self, keypoint_names: Sequence[str]):
        lookup = {n: i for i, n in enumerate(keypoint_names)}
        self.idx: Dict[str, Optional[int]] = {n: lookup.get(n) for n in NAMES}
        required = ("nose", "left_shoulder", "right_shoulder")
        missing = [n for n in required if self.idx[n] is None]
        if missing:
            raise ValueError(
                f"engagement analysis needs keypoints {missing}; the model provides {list(keypoint_names)}. "
                "Set keypoint_names in the model sidecar."
            )


@dataclass
class Features:
    valid: bool
    scale: float = 0.0
    yaw: Optional[float] = None
    head_height: Optional[float] = None
    torso_ratio: Optional[float] = None
    # nose offset from the shoulder centre / scale: a second head-turn measure that keeps
    # working in profile, when the far ear (and with it the ear-based yaw) is lost
    head_offset: Optional[float] = None
    face_hidden: bool = False
    hand_raised: bool = False
    hand_at_face: bool = False
    face_visibility: float = 0.0


def _pt(kp: np.ndarray, i: Optional[int], thr: float):
    if i is None or i >= len(kp) or kp[i, 2] < thr:
        return None
    return kp[i, :2]


def compute(kp: np.ndarray, box: Sequence[float], index: KeypointIndex, thr: float) -> Features:
    """kp: (K, 3) pixel x, y, visibility. box: x1, y1, x2, y2 pixels."""
    I = index.idx
    g = lambda n: _pt(kp, I[n], thr)  # noqa: E731
    nose, le, re = g("nose"), g("left_eye"), g("right_eye")
    lear, rear = g("left_ear"), g("right_ear")
    ls, rs = g("left_shoulder"), g("right_shoulder")
    lw, rw = g("left_wrist"), g("right_wrist")
    box_w = max(1.0, float(box[2] - box[0]))
    box_h = max(1.0, float(box[3] - box[1]))

    # scale: shoulder width, else head size, else box width
    sw = float(np.linalg.norm(ls - rs)) if ls is not None and rs is not None else None
    if sw and sw > 0.05 * box_w:
        # shoulder width shrinks when the torso rotates; the box width floor keeps
        # the scale (and every distance divided by it) from blowing up then
        scale = max(sw, 0.5 * box_w)
    elif lear is not None and rear is not None:
        scale = 2.2 * float(np.linalg.norm(lear - rear))
    elif le is not None and re is not None:
        scale = 3.5 * float(np.linalg.norm(le - re))
    else:
        scale = 0.45 * box_w
    if scale <= 1.0:
        return Features(valid=False)

    have_face = nose is not None or (le is not None and re is not None)
    have_shoulders = ls is not None and rs is not None
    if not have_face and not have_shoulders:
        return Features(valid=False)

    f = Features(valid=True, scale=scale)
    vis = [kp[I[n], 2] for n in ("nose", "left_eye", "right_eye") if I[n] is not None]
    f.face_visibility = float(np.mean(vis)) if vis else 0.0

    # yaw, in "ear units": nose offset from the ear midpoint / half the ear distance.
    # The eye-based estimate is rescaled to roughly the same units so the value
    # doesn't jump when an ear flickers in and out of visibility.
    eye_yaw = None
    if nose is not None and le is not None and re is not None and abs(le[0] - re[0]) > 1e-3:
        eye_yaw = EYE_TO_EAR_UNITS * (nose[0] - (le[0] + re[0]) / 2) / abs(le[0] - re[0])
    if nose is not None and lear is not None and rear is not None and abs(lear[0] - rear[0]) > 1e-3:
        mid = (lear[0] + rear[0]) / 2
        f.yaw = float(np.clip((nose[0] - mid) / (abs(lear[0] - rear[0]) / 2), -1.5, 1.5))
    elif nose is not None and (lear is None) != (rear is None):
        # One ear below threshold. On real model output this is the common case, not
        # only for turned heads (one ear is often low-confidence on a frontal face), so
        # estimate the ear midpoint from the visible ear and the expected half head
        # width instead of assuming a full turn. The student's left ear sits on the
        # image-right of the head center.
        half = HALF_HEAD_WIDTH * scale
        center = lear[0] - half if lear is not None else rear[0] + half
        f.yaw = float(np.clip((nose[0] - center) / half, -1.5, 1.5))
    elif eye_yaw is not None:
        f.yaw = float(np.clip(eye_yaw, -1.5, 1.5))

    # head height above the shoulder line
    if have_shoulders:
        sh_y = (ls[1] + rs[1]) / 2
        ref = nose if nose is not None else ((le + re) / 2 if le is not None and re is not None else None)
        if ref is not None:
            f.head_height = float((sh_y - ref[1]) / scale)
        f.torso_ratio = float(sw / box_h) if sw else None
        f.face_hidden = not have_face
        if nose is not None:
            f.head_offset = float((nose[0] - (ls[0] + rs[0]) / 2) / scale)

    # hands
    head_y = min(p[1] for p in (nose, le, re) if p is not None) if have_face else None
    for w in (lw, rw):
        if w is None:
            continue
        if head_y is not None and w[1] < head_y - 0.1 * scale:
            f.hand_raised = True
        if nose is not None and np.linalg.norm(w - nose) < 0.6 * scale:
            f.hand_at_face = True
    return f


def ramp(x: float, lo: float, hi: float) -> float:
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    return (x - lo) / (hi - lo)


@dataclass
class Baseline:
    yaw: Optional[float] = None
    head_height: Optional[float] = None
    torso_ratio: Optional[float] = None
    head_offset: Optional[float] = None

    def to_dict(self) -> dict:
        return {"yaw": self.yaw, "head_height": self.head_height, "torso_ratio": self.torso_ratio,
                "head_offset": self.head_offset}

    @classmethod
    def from_samples(cls, samples: List[Features], min_n: int = 5) -> "Baseline":
        def med(attr):
            vals = [getattr(s, attr) for s in samples if getattr(s, attr) is not None]
            return float(np.median(vals)) if len(vals) >= min_n else None

        return cls(yaw=med("yaw"), head_height=med("head_height"), torso_ratio=med("torso_ratio"),
                   head_offset=med("head_offset"))


def score(f: Features, base: Baseline, weights: Dict[str, float], hand_raised_is_focused: bool,
          head_turn: tuple = (0.35, 0.85), head_offset: Optional[tuple] = None):
    """Per-frame distraction score 0..1 and the cue that produced it.

    head_offset: (start, full) ramp for the nose-vs-shoulders head-turn measure, or None to
    use the ear-based yaw only. When given, head_turned is the larger of the two measures."""
    cues = {}
    if f.yaw is not None and base.yaw is not None:
        cues["head_turned"] = ramp(abs(f.yaw - base.yaw), head_turn[0], head_turn[1])
    if head_offset is not None and f.head_offset is not None and base.head_offset is not None:
        v = ramp(abs(f.head_offset - base.head_offset), head_offset[0], head_offset[1])
        cues["head_turned"] = max(cues.get("head_turned", 0.0), v)
    if f.face_hidden:
        cues["face_hidden"] = 1.0
    if f.head_height is not None and base.head_height is not None:
        cues["head_down"] = ramp(base.head_height - f.head_height, 0.25, 0.6)
    if f.torso_ratio is not None and base.torso_ratio:
        cues["body_turned"] = ramp(1.0 - f.torso_ratio / base.torso_ratio, 0.25, 0.5)
    if f.hand_at_face:
        cues["hand_at_face"] = 1.0
    if hand_raised_is_focused and f.hand_raised:
        return 0.0, "hand_raised"
    # noisy-OR: strong cues decide alone, weak cues (weight below the on-threshold)
    # only tip the balance together, e.g. body turned + hand at face.
    keep, best, reason = 1.0, 0.0, None
    for k, v in cues.items():
        v *= weights.get(k, 0.0)
        keep *= 1.0 - v
        if v > best:
            best, reason = v, k
    total = 1.0 - keep
    return total, (reason if total > 0 else None)
