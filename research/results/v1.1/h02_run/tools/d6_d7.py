"""Handoff 02 D6 + D7 (uncommitted).
D6: lecture (clip_3), median head height above the shoulders of students the classifiers call distracted vs
    focused, at 2:00 (frames 119.5-120.5 s) and over 2:00-3:18, per analysis file.
D7: clip_5 front-row middle student: the labelled moments whose labelling-time seat was 15 or 8 (the
    thesis's m-with-tiles run), scored on the new m-with-tiles rules: flagged per label, with the cue.
usage: d6_d7.py CLIPS_DIR LABELS_CLIP5.json"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

R = Path.home() / "hailo-custom-yolo-pose-runtime"
sys.path.insert(0, str(R))
from hcpose.engagement.features import KeypointIndex, compute  # noqa: E402
from hcpose.engagement.settings import load_settings  # noqa: E402
from hcpose.presets import UPPER_BODY13_NAMES  # noqa: E402

S = load_settings()
IDX = KeypointIndex(UPPER_BODY13_NAMES)
D = Path(sys.argv[1])


def frames(p):
    return [d for d in map(json.loads, open(p)) if d.get("type") == "frame"]


print("## D6: lecture head height above the shoulders, by the decision's state\n")
print("| Analysis | at 2:00 distracted / focused (n) | 2:00-3:18 distracted / focused (person-frames) |")
print("|---|---|---|")
for m in ("n", "m"):
    for t in ("none", "auto"):
        for c in ("none", "thesis_trees", "thesis_mlp", "thesis_logistic", "lenient_trees", "lenient_mlp"):
            p = D / f"clip_3_{m}_{t}_{c}.jsonl"
            if not p.is_file():
                continue
            res = {}
            for name, (t0, t1, step) in {"at": (119.5, 120.5, 1), "span": (120, 198, 10)}.items():
                hh = {"distracted": [], "focused": []}
                F = [f for f in frames(p) if t0 <= f["t"] <= t1]
                for f in F[::step]:
                    for q in f["persons"]:
                        st = (q.get("engagement") or {}).get("state")
                        if st not in hh:
                            continue
                        ft = compute(np.array(q["keypoints"], dtype=np.float32), q["box"], IDX, S.keypoint_threshold)
                        if ft.head_height is not None:
                            hh[st].append(ft.head_height)
                res[name] = hh
            f_ = lambda h: f"{np.median(h['distracted']) if h['distracted'] else float('nan'):.2f} / {np.median(h['focused']) if h['focused'] else float('nan'):.2f} ({len(h['distracted'])} / {len(h['focused'])})"  # noqa: E731,E501
            print(f"| {p.name} | {f_(res['at'])} | {f_(res['span'])} |")


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9)


L = [x for x in json.load(open(sys.argv[2]))["items"] if x["label"] in ("focused", "distracted")]
front = [x for x in L if x.get("seat") in (15, 8)]
print(f"\n## D7: clip_5 front-row middle student ({len(front)} of {len(L)} judgeable moments; labelling-time seat 15 or 8)\n")
for m, t in (("m", "auto"), ("m", "none"), ("n", "auto")):
    F = frames(D / f"clip_5_720p_{m}_{t}_none.jsonl")
    ts = np.array([f["t"] for f in F])

    def state(x):
        k = int(np.argmin(np.abs(ts - x["t"])))
        if abs(ts[k] - x["t"]) > .25:
            return None, None, None
        c = [(iou(x["box"], q["box"]), q) for q in F[k]["persons"]]
        c = [z for z in c if z[0] >= .3]
        if not c:
            return None, None, None
        q = max(c, key=lambda z: z[0])[1]
        e = q.get("engagement") or {}
        return e.get("state"), e.get("reason") or e.get("cue"), q["box"][3] - q["box"][1]
    fa_all = sum(state(x)[0] == "distracted" for x in L if x["label"] == "focused")
    ca_all = sum(state(x)[0] == "distracted" for x in L if x["label"] == "distracted")
    rows = [(x["label"],) + state(x) for x in front]
    ffa = [r for r in rows if r[0] == "focused" and r[1] == "distracted"]
    fca = [r for r in rows if r[0] == "distracted" and r[1] == "distracted"]
    nfoc = sum(r[0] == "focused" for r in rows)
    ndis = len(rows) - nfoc
    tall = sum(1 for r in rows if r[3] and r[3] > 220)
    print(f"- {m} tiles {'on' if t == 'auto' else 'off'}, rules: all FA {fa_all}/121, caught {ca_all}/28. "
          f"Front student: {len(ffa)} of {nfoc} focused flagged (cues {dict(Counter(r[2] for r in ffa))}), "
          f"{len(fca)} of {ndis} distracted flagged (cues {dict(Counter(r[2] for r in fca))}); "
          f"tall box (> 220 px) on {tall} of {len(rows)}; without this student FA {fa_all - len(ffa)}/{121 - nfoc}, "
          f"caught {ca_all - len(fca)}/{28 - ndis}")
