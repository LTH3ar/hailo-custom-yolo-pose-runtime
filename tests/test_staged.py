"""research/staged/score.py on a synthetic session with a known answer: a perfect
labeller that switches 2 s after each scripted change."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "staged"))
sys.path.insert(0, str(ROOT))
import score  # noqa: E402
from hcpose.presets import UPPER_BODY13_NAMES  # noqa: E402

SCRIPT = ROOT / "research" / "staged" / "script_v1.csv"
# x, y centre (px), box w, h, by seat S1..S10 (front row lowest in the image)
SEATS = {"S1": (520, 560, 120, 160), "S2": (640, 560, 120, 160), "S3": (760, 560, 120, 160),
         "S4": (460, 400, 90, 120), "S5": (580, 400, 90, 120), "S6": (700, 400, 90, 120), "S7": (820, 400, 90, 120),
         "S8": (520, 280, 70, 95), "S9": (640, 280, 70, 95), "S10": (760, 280, 70, 95)}


def person(x, y, w, h, hands_up, state):
    x1, y1 = x - w / 2, y - h / 2
    k = np.zeros((13, 3))
    k[:, 2] = 0.9
    k[0] = (x, y1 + 0.2 * h, 0.9)                      # nose
    k[1:5, 0] = x + np.array([-0.05, 0.05, -0.12, 0.12]) * w
    k[1:5, 1] = y1 + 0.18 * h
    k[5:7, 0] = x + np.array([0.25, -0.25]) * w        # shoulders
    k[5:7, 1] = y1 + 0.45 * h
    k[7:9, 0] = x + np.array([0.3, -0.3]) * w
    k[7:9, 1] = y1 + 0.7 * h
    wy = y1 - 0.1 * h if hands_up else y1 + 0.9 * h     # wrists
    k[9:11, 0] = x + np.array([0.2, -0.2]) * w
    k[9:11, 1] = wy
    return {"box": [x1, y1, x1 + w, y1 + h], "score": 0.9, "keypoints": k.round(1).tolist(),
            "engagement": {"state": state}}


def test_scorer_on_a_perfect_delayed_labeller(tmp_path):
    names = list(SEATS)
    events = score.load_script(str(SCRIPT), names)
    offset, fps, delay = 7.0, 5, 2.0
    path = tmp_path / "synthetic.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"type": "header", "keypoint_names": UPPER_BODY13_NAMES}) + "\n")
        for i in range(int((offset + 905) * fps)):
            t = i / fps
            st = t - offset
            persons, labels = [], []
            for n, (x, y, w, h) in SEATS.items():
                gt_now, _ = score.truth(events, n, st)
                if gt_now == "absent":
                    continue
                labels.append(gt_now)
                lag, _ = score.truth(events, n, st - delay)
                persons.append(person(x, y, w, h, 0 <= st < 3, "distracted" if lag == "distracted" else "focused"))
            g = 100 * sum(lab == "distracted" for lab in labels) / len(labels)
            f.write(json.dumps({"type": "frame", "i": i, "t": t, "w": 1280, "h": 720, "persons": persons,
                                "class": {"class_pct_distracted": g, "class_ci": [max(0, g - 10), min(100, g + 10)]}}) + "\n")
    r = score.score(str(path), events, [3, 4, 3], "auto", 180.0, 900.0, 5.0)
    assert abs(r["offset"] - offset) < 0.3
    for n, (x, y, w, h) in SEATS.items():  # seats numbered as the camera sees them
        assert abs(r["seats"][n][0] - x / 1280) < 0.01 and abs(r["seats"][n][1] - (y - h / 2) / 720) < 0.01, n  # head = box top
    assert r["recall"] > 0.98 and r["precision"] > 0.98, r
    assert r["fa_attentive"] < 0.01 and r["fa_hard"] < 0.01, r
    assert r["missed"] == 0 and abs(r["latency"] - delay) <= 1.0, r
    assert r["class_r"] > 0.95 and r["range_hit"] > 0.9, r
    assert r["coverage"] > 0.98

    # and the command line prints the table
    out = tmp_path / "summary.md"
    assert score.main([str(path), "--script", str(SCRIPT), "--rows", "3,4,3", "--out", str(out)]) == 0
    assert "| synthetic.jsonl |" in out.read_text()
