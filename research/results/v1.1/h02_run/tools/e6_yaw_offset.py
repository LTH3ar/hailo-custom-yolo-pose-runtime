"""Handoff 02 E6 (uncommitted): ear-based yaw vs nose offset on clip_4 (thesis ch4 table).
Per scripted window (5 s margin each side), the median yaw and head offset of the person at each seat.
Seats are found as in research/staged (S1 left, S2 right): the person whose box centre is nearest
x = 0.44 / 0.61 of the frame width. Also reports the yaw change from the seat's median over 0:05-0:58.
usage: e6_yaw_offset.py SKELETON.jsonl"""
import json
import sys
from pathlib import Path

import numpy as np

R = Path.home() / "hailo-custom-yolo-pose-runtime"
sys.path.insert(0, str(R))
from hcpose.engagement.features import KeypointIndex, compute  # noqa: E402
from hcpose.engagement.settings import load_settings  # noqa: E402
from hcpose.presets import UPPER_BODY13_NAMES  # noqa: E402

S = load_settings()
IDX = KeypointIndex(UPPER_BODY13_NAMES)
SEAT_X = {"S1 (blue)": 0.44, "S2 (grey)": 0.61}
WIN = [("attentive, before the notes", 0, 58, ["S1 (blue)", "S2 (grey)"]),
       ("writing notes", 58, 107, ["S1 (blue)", "S2 (grey)"]),
       ("talking to each other", 165, 197, ["S2 (grey)", "S1 (blue)"]),
       ("looking to the side", 218, 240, ["S1 (blue)"]),
       ("looking to the side", 107, 135, ["S2 (grey)"])]

head, frames = None, []
for line in open(sys.argv[1]):
    d = json.loads(line)
    if d.get("type") == "header":
        head = d
    elif d.get("type") == "frame":
        frames.append(d)


def at(seat, t0, t1):
    ys, os_ = [], []
    for f in frames:
        if not (t0 + 5 <= f["t"] <= t1 - 5) or not f["persons"]:
            continue
        W = f.get("w") or 1280
        p = min(f["persons"], key=lambda q: abs((q["box"][0] + q["box"][2]) / 2 / W - SEAT_X[seat]))
        if abs((p["box"][0] + p["box"][2]) / 2 / W - SEAT_X[seat]) > 0.08:
            continue
        ft = compute(np.array(p["keypoints"], dtype=np.float32), p["box"], IDX, S.keypoint_threshold)
        if ft.yaw is not None:
            ys.append(ft.yaw)
        if ft.head_offset is not None:
            os_.append(ft.head_offset)
    return (np.median(ys) if ys else np.nan), (np.median(os_) if os_ else np.nan), len(ys), len(os_)


base = {s: at(s, 0, 58) for s in SEAT_X}
print(f"{sys.argv[1]}\n| Moment | Person | Ear-based yaw (change from 0:05-0:53) | Nose offset | frames yaw / offset |")
print("|---|---|---|---|---|")
for name, t0, t1, seats in WIN:
    for s in seats:
        y, o, ny, no = at(s, t0, t1)
        print(f"| {name} ({t0 // 60}:{t0 % 60:02d}-{t1 // 60}:{t1 % 60:02d}) | {s} | {y:+.2f} ({y - base[s][0]:+.2f}) | {o:+.2f} | {ny} / {no} |")
