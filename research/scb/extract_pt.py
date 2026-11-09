"""Run a YOLOv8-pose model over SCB images and cache detections + SCB behavior boxes.

usage: extract_pose.py MODEL.pt OUT.jsonl SUBSET_DIR:name0,name1,... [...]
Each output line: {"img", "subset", "w", "h", "gt": [[cls_name, x1,y1,x2,y2]], "dets": [{"box","score","kpts"}]}
Keypoints are cut to the first 13 (COCO 0..12) so a stock 17-kpt model stands in for the upper-body one.
"""
import glob
import json
import sys

from ultralytics import YOLO

model_path, out_path, *subsets = sys.argv[1:]
model = YOLO(model_path)
n_done = 0
with open(out_path, "w") as out:
    for spec in subsets:
        d, names = spec.split(":")
        names = names.split(",")
        imgs = sorted(glob.glob(f"{d}/images/*/*.jpg") + glob.glob(f"{d}/images/*/*.png"))
        print(f"{d}: {len(imgs)} images", flush=True)
        for i in range(0, len(imgs), 16):
            batch = imgs[i:i + 16]
            for p, r in zip(batch, model.predict(batch, imgsz=640, conf=0.25, iou=0.7, verbose=False)):
                h, w = r.orig_shape
                gt = []
                lab = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
                try:
                    for line in open(lab):
                        c, x, y, bw, bh = map(float, line.split()[:5])
                        gt.append([names[int(c)], (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
                except FileNotFoundError:
                    pass
                dets = []
                if r.boxes is not None and len(r.boxes):
                    kp = r.keypoints.data.cpu().numpy()[:, :13, :]
                    for b, s, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), kp):
                        dets.append({"box": [round(float(v), 1) for v in b], "score": round(float(s), 3),
                                     "kpts": [[round(float(x), 1), round(float(y), 1), round(float(c), 3)] for x, y, c in k]})
                out.write(json.dumps({"img": p, "subset": d.split("/")[-1], "w": w, "h": h, "gt": gt, "dets": dets}) + "\n")
                n_done += 1
            if (i // 16) % 50 == 0:
                print(f"  {i + len(batch)}/{len(imgs)}", flush=True)
print("done", n_done)
