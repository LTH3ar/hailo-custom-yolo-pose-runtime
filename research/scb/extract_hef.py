"""Run a compiled HEF over SCB images on the Hailo device and cache detections + SCB labels.

Same output format as extract_pt.py, so evaluate.py / learn.py work on either.
Uses the hcpose python engine pieces (letterbox, HailoRT, NumPy decoder), i.e. the
exact INT8 model that runs live.

usage (on the Pi, from the repo root):
  .venv/bin/python research/scb/extract_hef.py ~/models/upper_body_n.hef poses_hef_n.jsonl \
      ~/scb/SCB_BowTurnHead_20250509/SCB_BowTurnHead_20250509/SCB5-Turn-Bow-Head-2024-9-17:BowHead,TurnHead \
      ~/scb/SCB5-Discuss-2024-9-17/SCB5-Discuss-2024-9-17:discuss \
      ~/scb/SCB5-Handrise-Read-write-2024-9-17/SCB5-Handrise-Read-write-2024-9-17:hand-raising,read,write
"""
import glob
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hcpose.decode import PoseDecoder  # noqa: E402
from hcpose.hef import import_hailo_platform, read_hef  # noqa: E402
from hcpose.py_engine import letterbox, poses_to_persons  # noqa: E402

hef_path, out_path, *subsets = sys.argv[1:]
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
                net_in, s, px, py = letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), layout.input_width, layout.input_height)
                res = pipe.infer({info.input_name: net_in[None]})
                persons = poses_to_persons(dec({k: v[0] for k, v in res.items()}), layout, s, px, py, w, h, ["person"])
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
                out.write(json.dumps({"img": p, "subset": Path(d).name, "w": w, "h": h, "gt": gt, "dets": dets}) + "\n")
                n += 1
                if n % 500 == 0:
                    print(f"  {n} images, {n / (time.time() - t0):.0f} img/s", flush=True)
print(f"done {n}")
