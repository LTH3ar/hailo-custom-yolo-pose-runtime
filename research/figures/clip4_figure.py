"""Thesis figure: what the pose model sees on the staged clip_4, and why the ear-based yaw
misses head turns in profile while the nose-vs-shoulders offset does not.

Draws the saved INT8 skeletons on frames of the 720p video (no model is run): confident
keypoints as filled dots, keypoints below 0.5 confidence as red rings with their value, and
for each person the ear-based yaw and the nose offset from the shoulder centre.

usage: python research/figures/clip4_figure.py clip_4_720p.mp4 clip_4_720p_m_none.skeleton.jsonl OUT_DIR
       (--no-video draws on grey, for testing)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hcpose.engagement.features import KeypointIndex, compute  # noqa: E402
from hcpose.presets import UPPER_BODY13_NAMES as NAMES  # noqa: E402

PANELS = [  # (seconds into the video, title), in time order
    (30, "Attentive (calibration)"), (80, "Both writing"), (150, "Blue: phone under the table"),
    (180, "Talking to each other"), (230, "Blue looks right, grey on phone"), (250, "Blue turns around"),
]
EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12)]

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("skeleton")
ap.add_argument("out")
ap.add_argument("--no-video", action="store_true")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
idx = KeypointIndex(NAMES)
frames = [d for d in map(json.loads, open(args.skeleton)) if d.get("type") == "frame"]
ts = np.array([f["t"] for f in frames])
cap = None if args.no_video else cv2.VideoCapture(args.video)


def grab(t):
    fr = frames[int(np.argmin(np.abs(ts - t)))]
    if cap is None:
        return np.full((fr["h"], fr["w"], 3), 90, np.uint8), fr
    cap.set(cv2.CAP_PROP_POS_MSEC, fr["t"] * 1000)
    ok, img = cap.read()
    if not ok:
        raise SystemExit(f"could not read the video at {fr['t']:.2f} s")
    return img, fr


def draw(img, p):
    k = np.array(p["keypoints"], np.float32)
    for a, b in EDGES:
        if k[a, 2] >= 0.5 and k[b, 2] >= 0.5:
            cv2.line(img, tuple(int(v) for v in k[a, :2]), tuple(int(v) for v in k[b, :2]), (255, 200, 60), 2, cv2.LINE_AA)
    for i, (x, y, c) in enumerate(k):
        if i >= 5 and c < 0.5:
            continue
        if c >= 0.5:
            cv2.circle(img, (int(x), int(y)), 4, (60, 220, 60), -1, cv2.LINE_AA)
        else:  # a low-confidence face point: where the model guesses it, and how unsure it is
            cv2.circle(img, (int(x), int(y)), 7, (40, 40, 230), 2, cv2.LINE_AA)
            cv2.putText(img, f"{c:.2f}", (int(x) + 8, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 230), 2, cv2.LINE_AA)
    f = compute(k, p["box"], idx, 0.5)
    who = "blue" if (p["box"][0] + p["box"][2]) / 2 < 0.52 * img.shape[1] else "grey"
    lines = [who, f"ear yaw {f.yaw:+.2f}" if f.yaw is not None else "ear yaw: n/a",
             f"nose offset {f.head_offset:+.2f}" if f.head_offset is not None else "nose offset: n/a"]
    x, y = int(p["box"][0]), int(p["box"][1]) - 58
    for j, s in enumerate(lines):
        cv2.putText(img, s, (x, y + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, s, (x, y + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


tiles = []
for t, title in PANELS:
    img, fr = grab(t)
    for p in fr["persons"]:
        draw(img, p)
    boxes = np.array([p["box"] for p in fr["persons"]])
    x1, y1 = max(0, int(boxes[:, 0].min()) - 40), max(0, int(boxes[:, 1].min()) - 80)
    x2, y2 = min(img.shape[1], int(boxes[:, 2].max()) + 40), min(img.shape[0], int(boxes[:, 3].max()) + 10)
    crop = cv2.resize(img[y1:y2, x1:x2], (640, int(640 * (y2 - y1) / (x2 - x1))))
    head = np.full((36, 640, 3), 255, np.uint8)
    cv2.putText(head, f"{int(t // 60)}:{int(t % 60):02d}  {title}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
    tile = np.vstack([head, crop])
    cv2.imwrite(str(out / f"clip4_{int(t)}s.png"), tile)
    tiles.append(tile)
h = max(t.shape[0] for t in tiles)
tiles = [np.vstack([t, np.full((h - t.shape[0], 640, 3), 255, np.uint8)]) for t in tiles]
grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
cv2.imwrite(str(out / "clip4_overview.png"), grid)
print(f"wrote {len(tiles)} panels and {out / 'clip4_overview.png'}")
