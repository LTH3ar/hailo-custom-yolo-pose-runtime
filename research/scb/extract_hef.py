"""Run a compiled HEF over SCB images on the Hailo device and cache detections + SCB labels.

Same output format as extract_pt.py, so evaluate.py / learn.py work on either.
Uses the hcpose python engine pieces (letterbox, HailoRT, NumPy decoder), i.e. the
exact INT8 model that runs live.

usage (on the Pi, from the repo root):
  .venv/bin/python research/scb/extract_hef.py ~/models/upper_body_n.hef poses_hef_n.jsonl \
      ~/scb/SCB_BowTurnHead_20250509/SCB_BowTurnHead_20250509/SCB5-Turn-Bow-Head-2024-9-17:BowHead,TurnHead \
      ~/scb/SCB5-Discuss-2024-9-17/SCB5-Discuss-2024-9-17:discuss \
      ~/scb/SCB5-Handrise-Read-write-2024-9-17/SCB5-Handrise-Read-write-2024-9-17:hand-raising,read,write

Add --tiles auto (and optionally --tile-region 0:0.6) to also run the zoomed-in far-row
tiles and merge them exactly like `hcpose run --tiles auto`.
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hcpose.decode import PoseDecoder  # noqa: E402
from hcpose.hef import import_hailo_platform, read_hef  # noqa: E402
from hcpose.py_engine import letterbox, poses_to_persons  # noqa: E402
from hcpose.tiling import make_tiles, merge, offset, parse_region, parse_spec  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("hef")
ap.add_argument("out")
ap.add_argument("subsets", nargs="+", help="DIR:class0,class1,...")
ap.add_argument("--tiles", default=None, help="auto | RxC (default: no tiling)")
ap.add_argument("--tile-region", default="0:0.6")
args = ap.parse_args()
hef_path, out_path, subsets = args.hef, args.out, args.subsets
grid, region = parse_spec(args.tiles), parse_region(args.tile_region)
info = read_hef(hef_path)
layout = info.layout()
dec = PoseDecoder(layout, score_threshold=0.25, iou_threshold=0.7)
hp = import_hailo_platform()
hef = hp.HEF(hef_path)
n, t0 = 0, time.time()
with hp.VDevice() as vdev, open(out_path, "w") as out:
    ng = vdev.configure(hef, hp.ConfigureParams.create_from_hef(hef, interface=hp.HailoStreamInterface.PCIe))[0]
    ivp = hp.InputVStreamParams.make(ng, format_type=hp.FormatType.UINT8)
    ovp = hp.OutputVStreamParams.make(ng, format_type=hp.FormatType.FLOAT32)
    with ng.activate(ng.create_params()), hp.InferVStreams(ng, ivp, ovp) as pipe:
        for spec in subsets:
            d, names = spec.split(":")
            d = str(Path(d).expanduser())
            names = names.split(",")
            imgs = sorted(glob.glob(f"{d}/images/*/*.jpg") + glob.glob(f"{d}/images/*/*.png"))
            print(f"{d}: {len(imgs)} images", flush=True)
            for p in imgs:
                bgr = cv2.imread(p)
                if bgr is None:
                    continue
                h, w = bgr.shape[:2]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                net_in, s, px, py = letterbox(rgb, layout.input_width, layout.input_height)
                tiles = make_tiles(w, h, grid, region) if grid else []
                crops = [letterbox(rgb[t.y0:t.y1, t.x0:t.x1], layout.input_width, layout.input_height) for t in tiles]
                res = pipe.infer({info.input_name: np.stack([net_in] + [c[0] for c in crops])})
                persons = poses_to_persons(dec({k: v[0] for k, v in res.items()}), layout, s, px, py, w, h, ["person"])
                if tiles:
                    tiled = []
                    for j, (t, (_, ts, tpx, tpy)) in enumerate(zip(tiles, crops), start=1):
                        tp = dec({k: v[j] for k, v in res.items()})
                        tiled += [(offset(q, t), t) for q in poses_to_persons(tp, layout, ts, tpx, tpy, t.w, t.h, ["person"])]
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
                n += 1
                if n % 500 == 0:
                    print(f"  {n} images, {n / (time.time() - t0):.0f} img/s", flush=True)
print(f"done {n}")
