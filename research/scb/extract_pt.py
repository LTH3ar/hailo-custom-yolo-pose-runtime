"""Run a YOLOv8-pose .pt (Ultralytics) over SCB images and cache detections + SCB behavior boxes.

usage: extract_pt.py MODEL.pt OUT.jsonl SUBSET_DIR:name0,name1,... [...] [--tiles auto --tile-region 0:0.6]
Each output line: {"img", "subset", "w", "h", "gt": [[cls_name, x1,y1,x2,y2]], "dets": [{"box","score","kpts"}]}
Keypoints are cut to the first 13 (COCO 0..12) so a stock 17-kpt model stands in for the upper-body one.
With --tiles, the model also runs on overlapping crops of the far part of the room and the
detections are merged exactly like hcpose does at runtime (hcpose/tiling.py).
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hcpose.results import Person  # noqa: E402
from hcpose.tiling import make_tiles, merge, offset, parse_region, parse_spec  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("subsets", nargs="+", help="DIR:class0,class1,...")
ap.add_argument("--tiles", default=None, help="auto | RxC (default: no tiling)")
ap.add_argument("--tile-region", default="0:0.6")
ap.add_argument("--conf", type=float, default=0.25)
args = ap.parse_args()

model = YOLO(args.model)
grid = parse_spec(args.tiles)
region = parse_region(args.tile_region)


def to_persons(r, dx=0, dy=0):
    out = []
    if r.boxes is None or not len(r.boxes):
        return out
    kp = r.keypoints.data.cpu().numpy()[:, :13, :].copy()
    for b, s, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), kp):
        out.append(Person(None, float(s), 0, "person", [float(v) for v in b], k))
    return out


n_done = 0
with open(args.out, "w") as out:
    for spec in args.subsets:
        d, names = spec.split(":")
        names = names.split(",")
        imgs = sorted(glob.glob(f"{d}/images/*/*.jpg") + glob.glob(f"{d}/images/*/*.png"))
        print(f"{d}: {len(imgs)} images", flush=True)
        for i, p in enumerate(imgs):
            img = cv2.imread(p)
            if img is None:
                continue
            h, w = img.shape[:2]
            tiles = make_tiles(w, h, grid, region) if grid else []
            batch = [img] + [img[t.y0:t.y1, t.x0:t.x1] for t in tiles]
            res = model.predict(batch, imgsz=640, conf=args.conf, iou=0.7, verbose=False)
            persons = to_persons(res[0])
            if tiles:
                tiled = [(offset(q, t), t) for t, r in zip(tiles, res[1:]) for q in to_persons(r)]
                persons = merge(persons, tiled, w, h)
            gt = []
            try:
                for line in open(p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"):
                    c, x, y, bw, bh = map(float, line.split()[:5])
                    gt.append([names[int(c)], (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
            except FileNotFoundError:
                pass
            dets = [{"box": [round(float(v), 1) for v in q.box], "score": round(q.score, 3),
                     "kpts": [[round(float(a), 1), round(float(b), 1), round(float(c), 3)] for a, b, c in q.keypoints]}
                    for q in persons]
            out.write(json.dumps({"img": p, "subset": Path(d).name, "w": w, "h": h, "gt": gt, "dets": dets,
                                  "tiles": len(tiles)}) + "\n")
            n_done += 1
            if i % 500 == 0:
                print(f"  {i}/{len(imgs)}", flush=True)
print("done", n_done)
