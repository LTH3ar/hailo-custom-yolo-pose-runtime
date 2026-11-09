"""Engine-independent per-frame results and the JSONL writer.

Both engines hand a FrameResult to the user callback, so analysis code
(e.g. the engagement analysis in part 2) does not care which engine ran.
Saving skeletons instead of video is the privacy-first output path.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np


@dataclass
class Person:
    track_id: Optional[int]
    score: float
    class_id: int
    label: str
    box: List[float]  # x1, y1, x2, y2 in pixels of the original frame
    keypoints: np.ndarray  # (K, 3): x px, y px, visibility 0..1

    def to_dict(self, keypoint_names=None) -> dict:
        d = {
            "track_id": self.track_id,
            "score": round(self.score, 4),
            "class_id": self.class_id,
            "label": self.label,
            "box": [round(float(v), 1) for v in self.box],
            "keypoints": [[round(float(x), 1), round(float(y), 1), round(float(v), 3)] for x, y, v in self.keypoints],
        }
        return d


@dataclass
class FrameResult:
    index: int
    timestamp: float
    width: int
    height: int
    persons: List[Person] = field(default_factory=list)


FrameCallback = Callable[[FrameResult, Optional[np.ndarray]], None]


class JsonlWriter:
    """One JSON object per frame. First line is a header with model metadata."""

    def __init__(self, path: str, header: dict):
        self.f = open(path, "w", buffering=1)
        self.f.write(json.dumps({"type": "header", "created": time.time(), **header}) + "\n")

    def __call__(self, result: FrameResult, frame=None) -> None:
        self.f.write(
            json.dumps(
                {
                    "type": "frame",
                    "i": result.index,
                    "t": round(result.timestamp, 4),
                    "w": result.width,
                    "h": result.height,
                    "persons": [p.to_dict() for p in result.persons],
                }
            )
            + "\n"
        )

    def close(self) -> None:
        self.f.close()


class FpsCounter:
    def __init__(self, every: float = 2.0, printer=print):
        self.every = every
        self.printer = printer
        self.t0 = time.monotonic()
        self.n = 0
        self.total = 0
        self.start = self.t0

    def tick(self, extra: str = "") -> None:
        self.n += 1
        self.total += 1
        now = time.monotonic()
        if now - self.t0 >= self.every:
            self.printer(f"FPS: {self.n / (now - self.t0):6.1f}  {extra}")
            self.t0 = now
            self.n = 0

    def summary(self) -> str:
        dt = time.monotonic() - self.start
        return f"{self.total} frames in {dt:.1f}s ({self.total / dt if dt else 0:.1f} FPS avg)"
