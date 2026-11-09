"""Tiling: extra zoomed-in passes over the far part of the room.

The network sees a 640x640 letterboxed copy of the whole frame, so a back-row
student in a 1080p classroom shot is only a few dozen pixels tall by the time
the model looks at them. Running the same model on overlapping crops of the
back region (in addition to the full frame) gives those students 2-3x more
pixels, without changing the model.

Detections are merged: tile detections cut by an inner tile edge are dropped
(the neighbouring tile or the full frame sees that person whole), then a
greedy suppression keeps the highest-scoring copy of each person.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .results import Person


@dataclass(frozen=True)
class Tile:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0


def parse_spec(spec: Optional[str]) -> Optional[Tuple[int, int]]:
    """'auto' -> (0, 0) meaning choose; 'RxC' -> (rows, cols); None/'none'/'0' -> None."""
    if spec in (None, "", "none", "0", "off"):
        return None
    if spec == "auto":
        return (0, 0)
    r, c = spec.lower().split("x")
    return int(r), int(c)


def parse_region(region: str) -> Tuple[float, float]:
    a, b = (float(v) for v in region.split(":"))
    if not 0 <= a < b <= 1:
        raise ValueError("--tile-region must be y0:y1 with 0 <= y0 < y1 <= 1 (fractions of the frame height)")
    return a, b


def make_tiles(W: int, H: int, grid: Tuple[int, int], region=(0.0, 0.6), overlap: float = 0.2) -> List[Tile]:
    """Overlapping tiles covering rows region[0]..region[1] of the frame.
    grid (0, 0) picks one row of roughly square tiles, which suits the square network input."""
    y0, y1 = int(region[0] * H), int(region[1] * H)
    rh = y1 - y0
    rows, cols = grid
    if rows == 0:
        rows = 1
        cols = max(1, round((W / rh - overlap) / (1 - overlap)))
    tw = W / (cols - (cols - 1) * overlap)
    th = rh / (rows - (rows - 1) * overlap)
    tiles = []
    for r in range(rows):
        for c in range(cols):
            x0 = int(round(c * tw * (1 - overlap)))
            ty0 = y0 + int(round(r * th * (1 - overlap)))
            tiles.append(Tile(x0, ty0, min(W, int(math.ceil(x0 + tw))), min(y1, int(math.ceil(ty0 + th)))))
    return tiles


def offset(p: Person, t: Tile) -> Person:
    k = p.keypoints.copy()
    k[:, 0] += t.x0
    k[:, 1] += t.y0
    b = p.box
    return Person(p.track_id, p.score, p.class_id, p.label, [b[0] + t.x0, b[1] + t.y0, b[2] + t.x0, b[3] + t.y0], k)


def _cut_by_inner_edge(box: Sequence[float], t: Tile, W: int, H: int, margin: float) -> bool:
    mx, my = margin * t.w, margin * t.h
    return ((t.x0 > 0 and box[0] <= t.x0 + mx) or (t.x1 < W and box[2] >= t.x1 - mx)
            or (t.y0 > 0 and box[1] <= t.y0 + my) or (t.y1 < H and box[3] >= t.y1 - my))


def _overlap(a, b) -> Tuple[float, float]:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-9), inter / max(min(aa, ab), 1e-9)


def _same_person(a: Person, b: Person, iou: float, ios: float, thr: float = 0.5) -> bool:
    """Duplicate detection of one student? Plain IoU, or one box mostly inside the other *and*
    their visible keypoints agreeing. Keypoints are checked because in a classroom a back-row
    student's box often sits inside the box of the student in front of them."""
    o_iou, o_ios = _overlap(a.box, b.box)
    if o_iou >= iou:
        return True
    if o_ios < ios:
        return False
    va, vb = a.keypoints[:, 2] >= thr, b.keypoints[:, 2] >= thr
    common = va & vb
    if common.sum() < 3:
        return False
    d = ((a.keypoints[common, :2] - b.keypoints[common, :2]) ** 2).sum(axis=1) ** 0.5
    sa = min((a.box[2] - a.box[0]) * (a.box[3] - a.box[1]), (b.box[2] - b.box[0]) * (b.box[3] - b.box[1])) ** 0.5
    return float(d.mean()) < 0.2 * sa


def merge(full: List[Person], tiled: List[Tuple[Person, Tile]], W: int, H: int,
          iou: float = 0.5, ios: float = 0.7, edge_margin: float = 0.02) -> List[Person]:
    """full: detections on the whole frame; tiled: (detection in frame coordinates, its tile)."""
    # (person, source): -1 = full frame, else tile index. Detections from the same pass were
    # already de-duplicated by the decoder's NMS, so only different passes are compared.
    cands = [(p, -1) for p in full] + [(p, t) for p, t in tiled if not _cut_by_inner_edge(p.box, t, W, H, edge_margin)]
    cands.sort(key=lambda c: -c[0].score)
    kept: List[Tuple[Person, object]] = []
    for p, src in cands:
        if not any(src != s and _same_person(p, q, iou, ios) for q, s in kept):
            kept.append((p, src))
    return [p for p, _ in kept]
