"""Build a self-contained, offline HTML page for blind labelling of student-moments.

Each item shows one student for about 2 s (three crops at t-1 s, t, t+1 s) plus a small
view of the whole room with the student marked, and asks: focused, distracted or can't
tell. No model output is shown. Answers stay in the browser and are exported as JSON.

Moments are sampled from an `hcpose analyze` run with the rules and one with a learned
model, stratified by the two decisions (both focused / both distracted / only the rules /
only the learned model), so that both can be scored; each item records its stratum and the
stratum's size, so estimates can be re-weighted to the whole clip.

usage:
  python research/labels/make_page.py out.html \
      --clip VIDEO.mp4 SKELETON.jsonl RULES_OUT.jsonl LEARNED_OUT.jsonl N_ITEMS [--clip ...]
The page contains frames of the video: keep it on your own computer, don't publish it.
"""
import argparse
import base64
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--clip", nargs=5, action="append", required=True,
                metavar=("VIDEO", "SKELETON", "RULES_OUT", "LEARNED_OUT", "N"))
ap.add_argument("--gap", type=float, default=10.0, help="min seconds between two items of the same seat")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = random.Random(args.seed)


def frames(path):
    return [d for d in map(json.loads, open(path)) if d.get("type") == "frame"]


def jpg(img, q=72):
    ok, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(b.tobytes()).decode()


items = []
for video, skel, rules_out, learned_out, n in args.clip:
    n = int(n)
    name = Path(video).stem
    R, L = frames(rules_out), frames(learned_out)
    assert len(R) == len(L), "the two analysis files must come from the same skeleton file"
    dur = R[-1]["t"]
    strata = defaultdict(list)
    for fr_r, fr_l in zip(R, L):
        if not 1.5 <= fr_r["t"] <= dur - 1.5:
            continue
        for p_r, p_l in zip(fr_r["persons"], fr_l["persons"]):
            a = (p_r.get("engagement") or {}).get("state")
            b = (p_l.get("engagement") or {}).get("state")
            if a not in ("focused", "distracted") or b not in ("focused", "distracted"):
                continue
            seat = (p_r.get("engagement") or {}).get("seat")
            strata[f"rules {a[0].upper()} / learned {b[0].upper()}"].append((fr_r["t"], seat, p_r["box"]))
    # equal shares per stratum; slots a small stratum can't fill go to the others, round-robin
    chosen = []
    pools = {k: iter(rng.sample(v, len(v))) for k, v in strata.items()}

    def take(key):
        for t, seat, box in pools[key]:
            if any(c["seat"] == seat and abs(c["t"] - t) < args.gap for c in chosen):
                continue
            chosen.append({"clip": name, "t": round(t, 3), "seat": seat, "box": [round(v, 1) for v in box],
                           "stratum": key, "stratum_size": len(strata[key])})
            return True
        return False

    live = list(pools)
    while len(chosen) < n and live:
        live = [k for k in live if take(k)]
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    for c in chosen:
        x1, y1, x2, y2 = c["box"]
        w, h = x2 - x1, y2 - y1
        cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(w, h) * 0.8
        crops = []
        for dt in (-1.0, 0.0, 1.0):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, round((c["t"] + dt) * fps)))
            ok, img = cap.read()
            if not ok:
                break
            H, W = img.shape[:2]
            raw = img.copy()
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 255), 2)
            a, b = int(max(0, cx - s)), int(max(0, cy - s))
            crop = img[b:int(min(H, cy + s)), a:int(min(W, cx + s))]
            crops.append(jpg(cv2.resize(crop, (220, int(220 * crop.shape[0] / max(1, crop.shape[1]))))))
            if dt == 0.0:
                ctx = raw
                cv2.rectangle(ctx, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 255), 4)
                c["context"] = jpg(cv2.resize(ctx, (400, int(400 * H / W))), 65)
        if len(crops) == 3:
            c["crops"] = crops
            items.append(c)

rng.shuffle(items)
for i, it in enumerate(items):
    it["id"] = i
meta = [{k: v for k, v in it.items() if k not in ("crops", "context")} for it in items]
page = Path(__file__).with_name("page_template.html").read_text()
page = page.replace("/*ITEMS*/[]", json.dumps(items)).replace("/*META*/[]", json.dumps(meta))
Path(args.out).write_text(page)
print(f"{len(items)} items -> {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")
for k in sorted({m['clip'] + ' | ' + m['stratum'] for m in meta}):
    print("  ", k, sum(1 for m in meta if m["clip"] + " | " + m["stratum"] == k))
