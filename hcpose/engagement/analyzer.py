"""Engagement analyzer: per-student focused / distracted / unknown, plus class percentage.

Pipeline per frame:
  persons -> seats (seat-anchored identity, robust to tracker id switches)
          -> features (features.py)
          -> per-frame score against the seat's attentive baseline
          -> smoothed score + hysteresis -> state
          -> class statistics over a rolling window

Baselines: each seat learns its own "attentive" head direction / height /
torso width from its first `calibration_seconds` of observation (median, so
brief glances don't matter). Until then the class-wide median of the
currently visible students is used, combined with the camera-position prior.
Learned seats can be saved and reloaded for a fixed camera.
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..results import FrameResult, Person
from .features import Baseline, Features, KeypointIndex, compute, score
from .settings import CAMERA_YAW_PRIOR, EngagementSettings

log = logging.getLogger("hcpose")

FOCUSED, DISTRACTED, UNKNOWN = "focused", "distracted", "unknown"
CALIBRATION_VERSION = 1
MAX_CALIB_SAMPLES = 3000
BASELINE_KEYS = ("yaw", "head_height", "torso_ratio")
PARTIAL_MIN_SECONDS = 3.0  # a seat's own history is used once it has this much


@dataclass
class Seat:
    id: int
    cx: float  # normalized box center / size (resolution independent)
    cy: float
    w: float
    h: float
    first_seen: float
    last_seen: float
    hits: int = 0
    observed: float = 0.0  # seconds this seat had usable features
    samples: Dict[str, List[float]] = field(default_factory=lambda: {k: [] for k in BASELINE_KEYS})
    partial: Optional[Baseline] = None  # running median while calibrating (refreshed ~1/s)
    partial_t: float = -1e9
    baseline: Optional[Baseline] = None  # set once calibrated (or loaded)
    loaded: bool = False
    # decision state
    ema: Optional[float] = None
    state: str = UNKNOWN
    since: float = 0.0  # when the pending condition started
    last_valid: float = -1e9
    last_t: Optional[float] = None
    reason: Optional[str] = None
    distracted_time: float = 0.0
    counted_time: float = 0.0

    @property
    def calibrated(self) -> bool:
        return self.baseline is not None


class EngagementAnalyzer:
    def __init__(self, settings: EngagementSettings, keypoint_names: Sequence[str]):
        self.s = settings
        self.index = KeypointIndex(keypoint_names)
        self.seats: List[Seat] = []
        self._next_id = 1
        self._pct_hist: Deque[Tuple[float, float]] = deque()
        self._prior_warned = False
        self._room_yaw: Optional[float] = None
        self.last_class: dict = {}
        self._t_last: Optional[float] = None
        self._wall0 = time.time()
        if settings.calibration and not settings.recalibrate and Path(settings.calibration).is_file():
            self.load_calibration(settings.calibration)

    # ------------------------------------------------------------------ seats
    def _match_seats(self, persons: List[Person], W: int, H: int, t: float) -> List[Seat]:
        dets = []
        for p in persons:
            x1, y1, x2, y2 = p.box
            dets.append(((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, max(1e-3, (x2 - x1) / W), max(1e-3, (y2 - y1) / H)))
        pairs = []
        for di, (cx, cy, w, h) in enumerate(dets):
            for si, s in enumerate(self.seats):
                d = math.hypot(cx - s.cx, (cy - s.cy) * H / W) / max(w, s.w)
                if d < self.s.seat_match_radius:
                    pairs.append((d, di, si))
        pairs.sort()
        assigned: Dict[int, Seat] = {}
        used = set()
        for d, di, si in pairs:
            if di in assigned or si in used:
                continue
            assigned[di] = self.seats[si]
            used.add(si)
        out = []
        for di, (cx, cy, w, h) in enumerate(dets):
            seat = assigned.get(di)
            if seat is None:
                seat = Seat(self._next_id, cx, cy, w, h, first_seen=t, last_seen=t)
                self._next_id += 1
                self.seats.append(seat)
            else:
                a = 0.05  # seated students drift slowly; standing up spawns a temporary seat
                seat.cx += a * (cx - seat.cx)
                seat.cy += a * (cy - seat.cy)
                seat.w += a * (w - seat.w)
                seat.h += a * (h - seat.h)
            seat.last_seen = t
            seat.hits += 1
            out.append(seat)
        # forget transient seats (a passer-by, someone standing briefly)
        self.seats = [s for s in self.seats if s.calibrated or s.hits >= 15 or t - s.last_seen < 5.0]
        return out

    # ------------------------------------------------------------------ baselines
    def _class_now(self, feats: List[Features]) -> Baseline:
        """Median of the students visible in this frame (used only until seats have history)."""
        return Baseline.from_samples([f for f in feats if f.valid], min_n=3)

    def _compute_room_yaw(self) -> Optional[float]:
        """The room's attentive head direction: median over seats of each seat's own
        long-run median. Robust to many students looking away at the same moment,
        because each seat's median is over its history, not this frame."""
        ys = []
        for seat in self.seats:
            b = seat.baseline if seat.calibrated else seat.partial
            if b is not None and b.yaw is not None:
                ys.append(b.yaw)
        if len(ys) < 3:
            return None
        med = float(np.median(ys))
        if self.s.camera_position != "auto":
            prior = CAMERA_YAW_PRIOR[self.s.camera_position]
            if abs(med - prior) > 0.6:
                if not self._prior_warned:
                    log.warning(
                        "learned attentive head direction (yaw %.2f) disagrees with camera_position=%s "
                        "(expected about %.2f); using the setting. Recalibrate if the camera moved.",
                        med, self.s.camera_position, prior,
                    )
                    self._prior_warned = True
                return prior
        return med

    def _baseline_for(self, seat: Seat, class_now: Baseline) -> Baseline:
        prior = CAMERA_YAW_PRIOR[self.s.camera_position]
        own = seat.baseline if seat.calibrated else seat.partial
        room = self._room_yaw
        if own is not None and own.yaw is not None:
            yaw = own.yaw
            if room is not None:
                # perspective makes each seat's attentive direction a bit different, but a
                # seat that learned while looking away mustn't drift far from the room's
                m = self.s.seat_yaw_max_offset
                yaw = room + float(np.clip(yaw - room, -m, m))
        elif room is not None:
            yaw = room
        elif class_now.yaw is not None:
            yaw = class_now.yaw
            if self.s.camera_position != "auto":
                yaw = float(np.clip(yaw, prior - 0.6, prior + 0.6))
        else:
            yaw = prior
        pick = lambda a: getattr(own, a) if own is not None and getattr(own, a) is not None else getattr(class_now, a)  # noqa: E731
        return Baseline(yaw=yaw, head_height=pick("head_height"), torso_ratio=pick("torso_ratio"))

    def _learn(self, seat: Seat, f: Features, t: float, dt: float) -> None:
        if seat.calibrated or not f.valid:
            return
        seat.observed += dt
        if len(seat.samples["yaw"]) + len(seat.samples["head_height"]) < 2 * MAX_CALIB_SAMPLES:
            for k in BASELINE_KEYS:
                v = getattr(f, k)
                if v is not None:
                    seat.samples[k].append(v)
        if seat.observed >= PARTIAL_MIN_SECONDS and t - seat.partial_t >= 1.0:
            seat.partial = self._median(seat.samples)
            seat.partial_t = t
        if seat.observed >= self.s.calibration_seconds and len(seat.samples["head_height"]) + len(seat.samples["yaw"]) >= 30:
            seat.baseline = self._median(seat.samples)
            seat.samples = {k: [] for k in BASELINE_KEYS}
            seat.partial = None
            log.info("seat %d calibrated: %s", seat.id, seat.baseline.to_dict())

    @staticmethod
    def _median(samples: Dict[str, List[float]]) -> Baseline:
        return Baseline(**{k: (float(np.median(v)) if len(v) >= 5 else None) for k, v in samples.items()})

    # ------------------------------------------------------------------ decisions
    def _decide(self, seat: Seat, valid: bool, frame_score: float, reason, t: float, dt: float) -> None:
        s = self.s
        if valid:
            a = 1.0 - math.exp(-dt / s.ema_tau) if dt > 0 else 1.0
            seat.ema = frame_score if seat.ema is None else seat.ema + a * (frame_score - seat.ema)
            seat.last_valid = t
            if frame_score > 0:
                seat.reason = reason
        elif t - seat.last_valid > s.unknown_after:
            seat.state, seat.ema = UNKNOWN, None
            return
        if seat.ema is None:
            return
        if seat.state == UNKNOWN:
            seat.state = DISTRACTED if seat.ema >= 0.5 else FOCUSED
            seat.since = t
            return
        want = None
        if seat.state == FOCUSED and seat.ema >= s.on_threshold:
            want, hold = DISTRACTED, s.on_seconds
        elif seat.state == DISTRACTED and seat.ema <= s.off_threshold:
            want, hold = FOCUSED, s.off_seconds
        if want is None:
            seat.since = t  # condition not met: restart the hold timer
        elif t - seat.since >= hold:
            seat.state = want
            seat.since = t

    # ------------------------------------------------------------------ main entry
    def __call__(self, result: FrameResult, frame=None) -> None:
        self.update(result)

    def update(self, result: FrameResult) -> dict:
        t = result.timestamp
        dt = 0.0 if self._t_last is None else max(0.0, min(1.0, t - self._t_last))
        self._t_last = t
        W, H = result.width, result.height
        persons = result.persons
        feats = [compute(p.keypoints, p.box, self.index, self.s.keypoint_threshold) for p in persons]
        seats = self._match_seats(persons, W, H, t)
        class_now = self._class_now(feats)
        for f, seat in zip(feats, seats):
            self._learn(seat, f, t, dt)
        self._room_yaw = self._compute_room_yaw()

        for p, f, seat in zip(persons, feats, seats):
            base = self._baseline_for(seat, class_now)
            sc, reason = score(f, base, self.s.weights, self.s.hand_raised_is_focused) if f.valid else (0.0, None)
            self._decide(seat, f.valid, sc, reason, t, dt)
            p.engagement = {
                "seat": seat.id,
                "state": seat.state,
                "score": None if seat.ema is None else round(seat.ema, 3),
                "reason": seat.reason if seat.state == DISTRACTED else None,
                "calibrated": seat.calibrated,
            }
        # seats not seen this frame time out to unknown
        seen = {id(s) for s in seats}
        for seat in self.seats:
            if id(seat) not in seen and t - seat.last_seen > self.s.unknown_after:
                seat.state, seat.ema = UNKNOWN, None

        focused = sum(1 for s in seats if s.state == FOCUSED)
        distracted = sum(1 for s in seats if s.state == DISTRACTED)
        counted = focused + distracted
        for s in seats:
            if s.state in (FOCUSED, DISTRACTED):
                s.counted_time += dt
                if s.state == DISTRACTED:
                    s.distracted_time += dt
        pct = 100.0 * distracted / counted if counted else None
        if pct is not None:
            self._pct_hist.append((t, pct))
        while self._pct_hist and t - self._pct_hist[0][0] > self.s.window_seconds:
            self._pct_hist.popleft()
        smooth = float(np.mean([v for _, v in self._pct_hist])) if self._pct_hist else None
        calibrated = sum(1 for s in self.seats if s.calibrated)
        self.last_class = {
            "visible": len(persons),
            "counted": counted,
            "focused": focused,
            "distracted": distracted,
            "unknown": len(persons) - counted,
            "pct_distracted": None if pct is None else round(pct, 1),
            "pct_distracted_smoothed": None if smooth is None else round(smooth, 1),
            "seats_calibrated": calibrated,
        }
        result.extra["class"] = self.last_class
        return self.last_class

    # ------------------------------------------------------------------ text for display
    def class_text(self) -> str:
        c = self.last_class
        if not c or c.get("pct_distracted_smoothed") is None:
            return "Engagement: waiting for students..."
        txt = f"{c['pct_distracted_smoothed']:.0f}% distracted  ({c['counted']} of {c['visible']} students counted)"
        total = len(self.seats)
        if total and c["seats_calibrated"] < total:
            txt += f"  [calibrating {c['seats_calibrated']}/{total}]"
        return txt

    # ------------------------------------------------------------------ calibration file
    def save_calibration(self, path: str) -> int:
        seats = [s for s in self.seats if s.calibrated]
        data = {
            "version": CALIBRATION_VERSION,
            "camera_position": self.s.camera_position,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seats": [
                {"id": s.id, "cx": s.cx, "cy": s.cy, "w": s.w, "h": s.h, "baseline": s.baseline.to_dict()}
                for s in seats
            ],
        }
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(p)
        return len(seats)

    def load_calibration(self, path: str) -> None:
        data = json.loads(Path(path).expanduser().read_text())
        if data.get("version") != CALIBRATION_VERSION:
            raise ValueError(f"{path}: unsupported calibration version {data.get('version')}")
        if data.get("camera_position") not in (None, self.s.camera_position):
            log.warning("calibration %s was made with camera_position=%s, now %s",
                        path, data.get("camera_position"), self.s.camera_position)
        for d in data["seats"]:
            seat = Seat(d["id"], d["cx"], d["cy"], d["w"], d["h"], first_seen=0.0, last_seen=-1e9,
                        baseline=Baseline(**d["baseline"]), loaded=True, hits=1000)
            self.seats.append(seat)
        self._next_id = max([s.id for s in self.seats], default=0) + 1
        log.info("loaded calibration for %d seats from %s", len(data["seats"]), path)

    # ------------------------------------------------------------------ session summary
    def summary(self) -> dict:
        per_seat = []
        for s in sorted(self.seats, key=lambda s: s.id):
            if s.counted_time > 0:
                per_seat.append({
                    "seat": s.id,
                    "counted_seconds": round(s.counted_time, 1),
                    "distracted_pct": round(100 * s.distracted_time / s.counted_time, 1),
                    "calibrated": s.calibrated,
                })
        tot = sum(s.counted_time for s in self.seats)
        dis = sum(s.distracted_time for s in self.seats)
        return {
            "student_seconds_counted": round(tot, 1),
            "distracted_pct_overall": round(100 * dis / tot, 1) if tot else None,
            "seats": per_seat,
        }
