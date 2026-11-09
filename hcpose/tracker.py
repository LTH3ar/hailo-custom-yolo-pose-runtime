"""Minimal greedy IoU tracker for the Python engine (the GStreamer engine uses hailotracker)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4), b: (M,4) -> (N,M)"""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


class IouTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 15):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks: Dict[int, dict] = {}
        self.next_id = 1

    def update(self, boxes: List[List[float]]) -> List[int]:
        ids = [0] * len(boxes)
        track_ids = list(self.tracks)
        if boxes and track_ids:
            iou = _iou(np.asarray(boxes, dtype=np.float32), np.asarray([self.tracks[t]["box"] for t in track_ids]))
            used_d, used_t = set(), set()
            for flat in np.argsort(-iou, axis=None):
                d, t = divmod(int(flat), len(track_ids))
                if iou[d, t] < self.iou_threshold:
                    break
                if d in used_d or t in used_t:
                    continue
                used_d.add(d)
                used_t.add(t)
                ids[d] = track_ids[t]
        for d, box in enumerate(boxes):
            if not ids[d]:
                ids[d] = self.next_id
                self.next_id += 1
            self.tracks[ids[d]] = {"box": box, "missed": 0}
        seen = set(ids)
        for t in list(self.tracks):
            if t not in seen:
                self.tracks[t]["missed"] += 1
                if self.tracks[t]["missed"] > self.max_missed:
                    del self.tracks[t]
        return ids
