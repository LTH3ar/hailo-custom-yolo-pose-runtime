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


# ---------------------------------------------------------------- C++ (GStreamer engine) == Python
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402

TILE_HARNESS = os.environ.get("HCPOSE_TILE_HARNESS",
                              str(Path(__file__).resolve().parents[1] / "build" / "tile_harness"))


def _scenario(rng, W=1280, H=720):
    """A classroom-ish frame: full-frame detections of near students plus tile detections of
    far students, with duplicates across passes, people cut by tile edges and students hidden
    behind others."""
    tiles = make_tiles(W, H, parse_spec("auto"), (0.0, 0.6))
    truth = []
    for _ in range(rng.integers(8, 16)):
        far = rng.random() < 0.6
        sw = rng.uniform(25, 45) if far else rng.uniform(70, 130)
        cy = rng.uniform(120, 380) if far else rng.uniform(420, 640)
        truth.append((rng.uniform(60, W - 60), cy, sw, far))
    full, tiled = [], [[] for _ in tiles]
    for cx, cy, sw, far in truth:
        p = synth_person(cx, cy, sw, jitter=0.01, rng=rng)
        if not far or rng.random() < 0.4:  # far students are seen in the full frame only sometimes
            full.append((p, float(rng.uniform(0.3, 0.6) if far else rng.uniform(0.6, 0.95))))
        for ti, t in enumerate(tiles):
            b = p.box
            if b[2] > t.x0 and b[0] < t.x1 and b[3] > t.y0 and b[1] < t.y1:
                k = p.keypoints.copy()
                k[:, :2] += rng.normal(0, 0.5, (len(k), 2))
                q = type(p)(None, float(rng.uniform(0.5, 0.95)), 0, "person",
                            [max(b[0], t.x0) - t.x0, max(b[1], t.y0) - t.y0, min(b[2], t.x1) - t.x0, min(b[3], t.y1) - t.y0],
                            k - np.array([t.x0, t.y0, 0], np.float32))
                tiled[ti].append(q)
    return W, H, tiles, full, tiled


def _det(p, score=None):
    return {"box": [float(v) for v in p.box], "score": float(p.score if score is None else score),
            "kpts": [[float(x), float(y), float(v)] for x, y, v in p.keypoints]}


@pytest.mark.skipif(not os.path.isfile(TILE_HARNESS), reason="C++ tile harness not built")
def test_cpp_tile_merge_matches_python():
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "cfg.json"
        for trial in range(25):
            W, H, tiles, full, tiled = _scenario(rng)
            cfg.write_text(json.dumps({"tile_merge": {"frame_width": W, "frame_height": H}}))
            spec = ";".join(f"{t.x0 / W},{t.y0 / H},{t.w / W},{t.h / H}" for t in tiles)
            sc = {"W": W, "H": H, "tiles": spec,
                  "full": [_det(p, s) for p, s in full],
                  "tiled": [[_det(q) for q in ts] for ts in tiled]}
            sp = Path(tmp) / "sc.json"
            sp.write_text(json.dumps(sc))
            out = subprocess.run([TILE_HARNESS, str(sp), str(cfg)], capture_output=True, text=True, check=True)
            cpp = json.loads(out.stdout)

            full_p = [type(p)(None, s, 0, "person", p.box, p.keypoints) for p, s in full]
            # the C++ side sees tiles as normalized floats; rebuild the same pixel tiles
            tiled_p = [(offset(q, t), t) for t, ts in zip(tiles, tiled) for q in ts]
            py = merge(full_p, tiled_p, W, H)
            assert len(cpp) == len(py), (trial, len(cpp), len(py))
            for c, q in zip(sorted(cpp, key=lambda d: -d["score"]), py):
                assert abs(c["score"] - q.score) < 1e-5, trial
                np.testing.assert_allclose(c["box"], q.box, atol=0.05)
                np.testing.assert_allclose(np.array(c["kpts"])[:, :2], q.keypoints[:, :2], atol=0.05)
