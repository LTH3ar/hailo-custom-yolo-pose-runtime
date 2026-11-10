"""COCO val2017 keypoint accuracy of the pose models, with the official COCO protocol (pycocotools).

Why: Ultralytics scores a 13-keypoint model with one tolerance for every keypoint (1/13), so the
numbers in the training log can't be compared with published COCO figures. This scores the
13 upper-body keypoints with COCO's own per-keypoint sigmas, the same way for every model:

  - the trained 13-keypoint .pt models (PyTorch, FP32, on a GPU)
  - the stock 17-keypoint YOLOv8-pose .pt models, cut to the same 13 points (the reference)
  - the INT8 HEFs on the Hailo-8 (the deployed models)

All three paths use the same square 640x640 letterbox (hcpose's), so the .pt vs HEF difference is
the cost of INT8 quantisation, not of a different pre-processing.

usage:
  coco_kpt_eval.py gt13  person_keypoints_val2017.json gt13.json
  coco_kpt_eval.py pt    MODEL.pt  VAL2017_DIR OUT.json [--keep 13|17] [--limit N]   (GPU machine)
  coco_kpt_eval.py hef   MODEL.hef VAL2017_DIR OUT.json [--limit N]                  (Pi + Hailo)
  coco_kpt_eval.py score GT.json NAME=PREDS.json [...] [--sigmas coco|uniform] [--md OUT.md]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

COCO_SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
                        .87, .87, .89, .89]) / 10.0
CONF = 0.001   # keep low-score detections: AP integrates over the whole score range
IOU = 0.7      # NMS IoU, as in Ultralytics validation
MAX_DET = 100  # COCO keypoint AP counts the top 20 per image


def images(d: str, limit: int | None):
    files = sorted(Path(d).glob("*.jpg"))
    return files[:limit] if limit else files


def to_json(img_id, box, score, kpts):
    k = np.asarray(kpts, dtype=float)
    return {"image_id": img_id, "category_id": 1, "score": round(float(score), 5),
            "bbox": [round(float(box[0]), 2), round(float(box[1]), 2),
                     round(float(box[2] - box[0]), 2), round(float(box[3] - box[1]), 2)],
            "keypoints": [round(float(v), 3) for v in k.reshape(-1)]}


def cmd_gt13(a):
    gt = json.load(open(a.ann))
    for ann in gt["annotations"]:
        k = ann["keypoints"][: 13 * 3]
        ann["keypoints"] = k
        ann["num_keypoints"] = int(sum(1 for v in k[2::3] if v > 0))
    for c in gt["categories"]:
        if c.get("name") == "person":
            c["keypoints"] = c["keypoints"][:13]
            c["skeleton"] = [p for p in c["skeleton"] if max(p) <= 13]
    json.dump(gt, open(a.out, "w"))
    print(f"wrote {a.out}: {len(gt['images'])} images, {len(gt['annotations'])} person annotations, 13 keypoints")


def cmd_pt(a):
    import cv2
    import torch
    from ultralytics import YOLO

    from hcpose.py_engine import letterbox

    model = YOLO(a.model)
    out, t0 = [], time.time()
    files = images(a.images, a.limit)
    for i, f in enumerate(files):
        bgr = cv2.imread(str(f))
        h, w = bgr.shape[:2]
        net, s, px, py = letterbox(bgr, 640, 640)
        r = model.predict(net, imgsz=640, conf=CONF, iou=IOU, max_det=MAX_DET, verbose=False,
                          half=False, device=0 if torch.cuda.is_available() else "cpu")[0]
        if r.boxes is not None and len(r.boxes):
            kp = r.keypoints.data.cpu().numpy()[:, : a.keep, :].copy()
            kp[..., 0] = np.clip((kp[..., 0] - px) / s, 0, w)
            kp[..., 1] = np.clip((kp[..., 1] - py) / s, 0, h)
            for b, sc, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), kp):
                b = b.copy()
                b[[0, 2]] = np.clip((b[[0, 2]] - px) / s, 0, w)
                b[[1, 3]] = np.clip((b[[1, 3]] - py) / s, 0, h)
                out.append(to_json(int(f.stem), b, sc, k))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(files)} images, {time.time() - t0:.0f} s", flush=True)
    json.dump(out, open(a.out, "w"))
    print(f"wrote {a.out}: {len(out)} detections on {len(files)} images ({time.time() - t0:.0f} s)")


def cmd_hef(a):
    import cv2

    from hcpose.decode import PoseDecoder
    from hcpose.hef import import_hailo_platform, read_hef
    from hcpose.py_engine import letterbox

    hp = import_hailo_platform()
    info = read_hef(a.hef)
    layout = info.layout()
    dec = PoseDecoder(layout, score_threshold=CONF, iou_threshold=IOU, max_detections=MAX_DET,
                      pose_dedup=0.0)
    in_h, in_w = layout.input_height, layout.input_width
    out, t0 = [], time.time()
    files = images(a.images, a.limit)
    hef = hp.HEF(a.hef)
    with hp.VDevice() as vdev:
        ng = vdev.configure(hef, hp.ConfigureParams.create_from_hef(hef, interface=hp.HailoStreamInterface.PCIe))[0]
        ivp = hp.InputVStreamParams.make(ng, format_type=hp.FormatType.UINT8)
        ovp = hp.OutputVStreamParams.make(ng, format_type=hp.FormatType.FLOAT32)
        with ng.activate(ng.create_params()), hp.InferVStreams(ng, ivp, ovp) as pipe:
            for i, f in enumerate(files):
                bgr = cv2.imread(str(f))
                h, w = bgr.shape[:2]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                net, s, px, py = letterbox(rgb, in_w, in_h)
                res = pipe.infer({info.input_name: net[None]})
                for p in dec({k: v[0] for k, v in res.items()}):
                    b = p.box.copy()
                    b[[0, 2]] = np.clip((b[[0, 2]] * in_w - px) / s, 0, w)
                    b[[1, 3]] = np.clip((b[[1, 3]] * in_h - py) / s, 0, h)
                    k = p.keypoints.copy()
                    k[:, 0] = np.clip((k[:, 0] * in_w - px) / s, 0, w)
                    k[:, 1] = np.clip((k[:, 1] * in_h - py) / s, 0, h)
                    out.append(to_json(int(f.stem), b, p.score, k))
                if (i + 1) % 500 == 0:
                    print(f"  {i + 1}/{len(files)} images, {time.time() - t0:.0f} s", flush=True)
    print(f"class scores: {'sigmoid applied on the host' if dec.score_sigmoid else 'sigmoid inside the HEF'}")
    json.dump(out, open(a.out, "w"))
    print(f"wrote {a.out}: {len(out)} detections on {len(files)} images ({time.time() - t0:.0f} s)")


STATS = ["AP", "AP50", "AP75", "AP medium", "AP large", "AR", "AR50", "AR75", "AR medium", "AR large"]


def cmd_score(a):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(a.gt)
    k = len(gt.loadCats(gt.getCatIds(catNms=["person"]))[0]["keypoints"])
    sig = COCO_SIGMAS[:k] if a.sigmas == "coco" else np.ones(k) / k
    rows = []
    for spec in a.preds:
        name, path = spec.split("=", 1)
        preds = json.load(open(path))
        img_ids = sorted({p["image_id"] for p in preds}) if a.only_predicted else sorted(gt.getImgIds())
        for p in preds:
            p["keypoints"] = p["keypoints"][: k * 3]
        with contextlib.redirect_stdout(io.StringIO()):
            dt = gt.loadRes(preds)
            ev = COCOeval(gt, dt, "keypoints")
            ev.params.kpt_oks_sigmas = sig
            ev.params.imgIds = img_ids
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        rows.append((name, ev.stats))
        print(f"{name:28s} " + "  ".join(f"{s}={v:.3f}" for s, v in zip(STATS[:3], ev.stats[:3])), flush=True)
    if a.md:
        hdr = f"COCO val2017 keypoints: {k} keypoints, {a.sigmas} OKS sigmas, pycocotools ({len(rows)} models)\n\n"
        lines = [hdr, "| Model | " + " | ".join(STATS) + " |", "| --- " * (len(STATS) + 1) + "|"]
        lines += [f"| {n} | " + " | ".join(f"{v:.3f}" for v in s) + " |" for n, s in rows]
        Path(a.md).write_text("\n".join(lines) + "\n")
        print(f"wrote {a.md}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gt13")
    g.add_argument("ann")
    g.add_argument("out")
    p = sp.add_parser("pt")
    p.add_argument("model")
    p.add_argument("images")
    p.add_argument("out")
    p.add_argument("--keep", type=int, default=13, help="keypoints to keep (13; 17 for the stock reference)")
    p.add_argument("--limit", type=int)
    h = sp.add_parser("hef")
    h.add_argument("hef")
    h.add_argument("images")
    h.add_argument("out")
    h.add_argument("--limit", type=int)
    s = sp.add_parser("score")
    s.add_argument("gt")
    s.add_argument("preds", nargs="+", help="NAME=PREDS.json")
    s.add_argument("--sigmas", choices=["coco", "uniform"], default="coco")
    s.add_argument("--only-predicted", action="store_true", help="score only images that have predictions (smoke tests with --limit)")
    s.add_argument("--md", help="also write a markdown table")
    a = ap.parse_args()
    {"gt13": cmd_gt13, "pt": cmd_pt, "hef": cmd_hef, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
