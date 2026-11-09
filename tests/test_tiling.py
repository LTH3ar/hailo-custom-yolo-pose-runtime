"""Tiling geometry and merging (hcpose/tiling.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hcpose.tiling import Tile, make_tiles, merge, offset, parse_region, parse_spec  # noqa: E402
from test_engagement import synth_person  # noqa: E402


def test_auto_tiles_cover_far_rows():
    tiles = make_tiles(1280, 720, parse_spec("auto"), parse_region("0:0.6"))
    assert len(tiles) == 3
    assert tiles[0].x0 == 0 and tiles[-1].x1 == 1280
    assert all(t.y0 == 0 and t.y1 == 432 for t in tiles)
    assert all(abs(t.w / t.h - 1) < 0.25 for t in tiles)  # roughly square, like the network input
    for a, b in zip(tiles, tiles[1:]):
        assert a.x1 - b.x0 >= 0.15 * a.w  # overlap so nobody falls only in a seam
    g = make_tiles(1920, 1080, parse_spec("2x3"), (0.0, 0.7))
    assert len(g) == 6 and g[-1].y1 == 756 and g[-1].x1 == 1920
    assert parse_spec(None) is None and parse_spec("none") is None


def _p(cx, cy, sw, score, **kw):
    p = synth_person(cx, cy, sw, **kw)
    return type(p)(None, score, 0, "person", p.box, p.keypoints)


def test_merge_keeps_best_copy_and_drops_cut_tile_detections():
    W, H = 1280, 720
    t = Tile(0, 0, 500, 432)
    far_full = _p(300, 200, 40, 0.35)  # small back-row student, weak in the full frame
    far_tile = _p(300, 200, 40, 0.80)  # same student seen bigger in the tile
    cut = _p(500, 200, 40, 0.90)  # straddles the tile's inner right edge
    front = _p(640, 600, 120, 0.95)
    kept = merge([far_full, front], [(far_tile, t), (cut, t)], W, H)
    scores = sorted(q.score for q in kept)
    assert scores == [0.80, 0.95], scores  # duplicate resolved to the tile copy; cut one dropped


def test_merge_keeps_student_hidden_inside_front_students_box():
    W, H = 1280, 720
    t = Tile(0, 0, 700, 432)
    front = _p(400, 420, 150, 0.9)  # big box in front
    behind = _p(400, 250, 45, 0.6)  # back-row student whose box lies inside the front box
    assert behind.box[0] > front.box[0] and behind.box[2] < front.box[2]
    kept = merge([front], [(behind, t)], W, H)
    assert len(kept) == 2


def test_offset_moves_box_and_keypoints():
    p = _p(100, 100, 40, 0.5)
    q = offset(p, Tile(200, 50, 600, 450))
    assert q.box[0] == p.box[0] + 200 and q.box[1] == p.box[1] + 50
    assert np.allclose(q.keypoints[:, 0], p.keypoints[:, 0] + 200)
    assert np.allclose(q.keypoints[:, 2], p.keypoints[:, 2])
