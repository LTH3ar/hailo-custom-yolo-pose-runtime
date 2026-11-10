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
from .classifier import describe as describe_classifier, feature_vector, resolve as resolve_classifier
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
    last_state: Optional[str] = None  # last focused/distracted decision (seat memory)
    last_state_t: float = -1e9

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
        self._est_hist: Deque[Tuple[float, float]] = deque()
        self.classifier = resolve_classifier(settings.classifier)
        if self.classifier is not None:
            log.info("engagement: learned classifier %s, trained on %s", describe_classifier(self.classifier),
                     self.classifier.meta.get("trained_on", "?"))
        else:
            log.info("engagement: rule-based cues")
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

        bases = [self._baseline_for(seat, class_now) for seat in seats]
        rule = [score(f, b, self.s.weights, self.s.hand_raised_is_focused) if f.valid else (0.0, None)
                for f, b in zip(feats, bases)]
        if self.classifier is not None and persons:
            # learned per-frame probability; the rule cues still name the reason
            valid = [i for i, f in enumerate(feats) if f.valid]
            probs = {}
            if valid:
                X = np.vstack([feature_vector(persons[i].keypoints, persons[i].box, feats[i], bases[i],
                                              self.index, self.s.keypoint_threshold) for i in valid])
                probs = dict(zip(valid, self.classifier.predict_proba(X)))
            if self.s.hand_raised_is_focused:
                probs = {i: (0.0 if feats[i].hand_raised else p) for i, p in probs.items()}
            rule = [(float(probs.get(i, 0.0)), r[1]) for i, r in enumerate(rule)]

        for p, f, seat, (sc, reason) in zip(persons, feats, seats, rule):
            self._decide(seat, f.valid, sc, reason, t, dt)
            p.engagement = {
                "seat": seat.id,
                "state": seat.state,
                "score": None if seat.ema is None else round(seat.ema, 3),
                "reason": seat.reason if seat.state == DISTRACTED else None,
                "calibrated": seat.calibrated,
            }
        for seat in seats:
            if seat.state in (FOCUSED, DISTRACTED):
                seat.last_state, seat.last_state_t = seat.state, t
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
        cls = self._class_estimate(t)
        if cls["pct"] is not None:
            self._est_hist.append((t, cls["pct"]))
        while self._est_hist and t - self._est_hist[0][0] > self.s.window_seconds:
            self._est_hist.popleft()
        est_smooth = float(np.mean([v for _, v in self._est_hist])) if self._est_hist else None
        self.last_class = {
            "visible": len(persons),
            "counted": counted,
            "focused": focused,
            "distracted": distracted,
            "unknown": len(persons) - counted,
            "pct_distracted": None if pct is None else round(pct, 1),
            "pct_distracted_smoothed": None if smooth is None else round(smooth, 1),
            "seats_calibrated": calibrated,
            "calibrating": any(not seat.calibrated for seat in seats),
            # whole-class estimate: seat memory + zone weighting + uncertainty
            "class_pct_distracted": None if est_smooth is None else round(est_smooth, 1),
            "class_ci": cls["ci"],
            "class_counted": cls["counted"],
            "class_size": cls["size"],
            "class_size_from": cls["size_from"],
            "zones": cls["zones"],
        }
        result.extra["class"] = self.last_class
        return self.last_class

    # ------------------------------------------------------------------ text for display
    def class_text(self) -> str:
        """Short one-line banner for the screen (fits a 640 px wide frame)."""
        c = self.last_class
        if not c or c.get("class_pct_distracted") is None:
            return "Engagement: waiting for students"
        lo, hi = c["class_ci"] or (None, None)
        rng = f" ({lo:.0f}-{hi:.0f}%)" if lo is not None and hi - lo >= 1 else ""
        approx = "" if c["class_size_from"] == "setting" else "~"
        txt = f"{c['class_pct_distracted']:.0f}% distracted{rng} | {c['class_counted']} of {approx}{c['class_size']} students"
        if c.get("calibrating"):
            txt += " | calibrating"
        return txt

    # ------------------------------------------------------------------ whole-class estimate
    def _established(self, seat: Seat) -> bool:
        """A real seat, not a passer-by: seen repeatedly over a few seconds."""
        return seat.calibrated or (seat.hits >= 10 and seat.last_seen - seat.first_seen >= 3.0)

    def _class_estimate(self, t: float) -> dict:
        """Distracted share of the *whole class*, correcting for who the camera misses.

        1. Seat memory: every established seat whose student was decided within the last
           presence_seconds counts with that last decision, not only students detected in
           this exact frame.
        2. Zones: seats are split into near / middle / far by apparent size. Each zone's
           distracted share is weighted by how many seats the zone has, not by how many
           students happen to be detected there, because far rows are detected less often.
           With class_size set, students never found at all are added to the far zone.
        3. Uncertainty: Wilson 95% interval on the counted students, narrowed by the
           finite-population correction when they are a large part of the class.
        """
        S = self.s
        seats = [x for x in self.seats if self._established(x)]
        recent = [x for x in seats if x.last_state is not None and t - x.last_state_t <= S.presence_seconds]
        size_from = "setting" if S.class_size else "seats"
        N = max(S.class_size or 0, len(seats)) if S.class_size else len(seats)
        out = {"pct": None, "ci": None, "counted": len(recent), "size": N, "size_from": size_from, "zones": []}
        if not recent:
            return out
        k = max(1, min(S.zones, len(seats) // 3))
        heights = np.array([x.h for x in seats])
        edges = np.quantile(heights, np.linspace(0, 1, k + 1)[1:-1]) if k > 1 else np.array([])
        zone_of = lambda seat: int(np.searchsorted(edges, seat.h, side="right"))  # noqa: E731  0 = far
        names = {1: ["all"], 2: ["far", "near"], 3: ["far", "middle", "near"]}.get(k, [f"z{i}" for i in range(k)])
        num = den = 0.0
        # students the camera has never found are most likely in the far rows
        unseen = max(0, N - len(seats))
        for z in range(k):
            Nz = sum(1 for x in seats if zone_of(x) == z) + (unseen if z == 0 else 0)
            rz = [x for x in recent if zone_of(x) == z]
            dz = sum(1 for x in rz if x.last_state == DISTRACTED)
            out["zones"].append({"zone": names[z], "seats": Nz, "counted": len(rz), "distracted": dz})
            if rz:
                num += Nz * dz / len(rz)
                den += Nz
        p = num / den if den else 0.0
        n = len(recent)
        z95 = 1.96
        center = (p + z95 * z95 / (2 * n)) / (1 + z95 * z95 / n)
        half = z95 * math.sqrt(p * (1 - p) / n + z95 * z95 / (4 * n * n)) / (1 + z95 * z95 / n)
        if N > n:
            half *= math.sqrt((N - n) / max(N - 1, 1))
        elif N:
            half = 0.0
        lo, hi = max(0.0, center - half), min(1.0, center + half)
        if half == 0.0:
            lo = hi = p
        out.update(pct=100 * p, ci=[round(100 * lo, 1), round(100 * hi, 1)])
        return out

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
