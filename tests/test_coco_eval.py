"""COCO keypoint evaluation helper: 17 -> 13 keypoint ground truth and pycocotools scoring."""
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pycocotools")
SCRIPT = Path(__file__).resolve().parents[1] / "research" / "coco_eval" / "coco_kpt_eval.py"


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], check=True,
                          capture_output=True, text=True).stdout


def _fixture(tmp_path):
    rnd = random.Random(0)
    cats = [{"id": 1, "name": "person", "keypoints": [f"k{i}" for i in range(17)],
             "skeleton": [[16, 14], [6, 7], [1, 2]]}]
    imgs, anns, preds = [], [], []
    for i in range(1, 11):
        imgs.append({"id": i, "width": 640, "height": 480})
        x0, y0 = rnd.uniform(0, 400), rnd.uniform(0, 200)
        k = []
        for _ in range(17):
            k += [x0 + rnd.uniform(0, 100), y0 + rnd.uniform(0, 200), 2]
        anns.append({"id": i, "image_id": i, "category_id": 1, "iscrowd": 0, "bbox": [x0, y0, 100, 200],
                     "area": 20000, "keypoints": k, "num_keypoints": 17})
        preds.append({"image_id": i, "category_id": 1, "score": 0.9, "bbox": [x0, y0, 100, 200],
                      "keypoints": k})
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps({"images": imgs, "annotations": anns, "categories": cats}))
    p = tmp_path / "p.json"
    p.write_text(json.dumps(preds))
    return gt, p


def test_gt13_cuts_keypoints_and_perfect_predictions_score_one(tmp_path):
    gt, p = _fixture(tmp_path)
    gt13 = tmp_path / "gt13.json"
    _run("gt13", gt, gt13)
    g = json.loads(gt13.read_text())
    assert len(g["categories"][0]["keypoints"]) == 13
    assert all(len(a["keypoints"]) == 39 and a["num_keypoints"] == 13 for a in g["annotations"])
    assert all(max(e) <= 13 for e in g["categories"][0]["skeleton"])
    md = tmp_path / "t.md"
    out = _run("score", gt13, f"exact={p}", "--md", md)  # 17-keypoint predictions are cut to 13
    assert "AP=1.000" in out and "| exact | 1.000" in md.read_text()
