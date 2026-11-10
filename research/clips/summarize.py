"""Summarise the files written by research/clips/run_clips.sh.

Per clip, model, tiles and classifier: students per frame, same-skeleton duplicate pairs
per frame, inferred class size at the end, class distracted % (mean after 2:30, or over the
whole clip if it is shorter than 3 minutes), mean width of its 95% range, and FPS.

usage: python research/clips/summarize.py OUT_DIR
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

D = Path(sys.argv[1])


def pairs(P):
    c = 0
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            A, B = np.array(P[i]["keypoints"]), np.array(P[j]["keypoints"])
            m = (A[:, 2] >= .5) & (B[:, 2] >= .5)
            if m.sum() < 3:
                continue
            bi, bj = P[i]["box"], P[j]["box"]
            h = min(bi[3] - bi[1], bj[3] - bj[1])
            if np.hypot(*(A[m, :2] - B[m, :2]).T).mean() < 0.1 * h:
                c += 1
    return c


rows = defaultdict(list)
for sk in sorted(D.glob("*.skeleton.jsonl")):
    base = sk.name[: -len(".skeleton.jsonl")]
    clip, m, t = base.rsplit("_", 2)
    F = [d for d in map(json.loads, open(sk)) if d.get("type") == "frame"]
    if not F:
        continue
    per = np.mean([len(f["persons"]) for f in F])
    dup = np.mean([pairs(f["persons"]) for f in F[::5]])
    log = (D / f"{base}.log").read_text(errors="replace") if (D / f"{base}.log").is_file() else ""
    fm = re.findall(r"\(([\d.]+) FPS avg\)", log)
    for c in ("none", "trees", "logistic", "mlp"):
        p = D / f"{base}_{c}.jsonl"
        if not p.is_file():
            continue
        C = [(d["t"], d["class"]) for d in map(json.loads, open(p))
             if d.get("type") == "frame" and d.get("class") and d["class"].get("class_pct_distracted") is not None]
        if not C:
            continue
        dur = C[-1][0]
        late = [c_ for tt, c_ in C if tt >= (150 if dur >= 180 else 0)]
        rows[clip].append({
            "model": m, "tiles": "on" if t == "auto" else "off", "clf": "rules" if c == "none" else c,
            "fps": float(fm[-1]) if fm else None, "per": per, "dup": dup,
            "size": C[-1][1].get("class_size"),
            "pct": float(np.mean([c_["class_pct_distracted"] for c_ in late])),
            "width": float(np.mean([c_["class_ci"][1] - c_["class_ci"][0] for c_ in late if c_.get("class_ci")] or [np.nan])),
            "window": "after 2:30" if dur >= 180 else "whole clip",
        })

out = ["# Classroom clips on the deployed models", ""]
for clip, R in rows.items():
    out += [f"## {clip} (class % {R[0]['window']})", "",
            "| Model | Tiles | FPS | Students / frame | Duplicates / frame | Class size | Rules | Trees | Logistic | MLP | Range width, rules (pts) |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in ("n", "s", "m"):
        for t in ("off", "on"):
            g = {r["clf"]: r for r in R if r["model"] == m and r["tiles"] == t}
            if not g:
                continue
            a = next(iter(g.values()))
            pc = lambda k: f"{g[k]['pct']:.1f}%" if k in g else "–"  # noqa: E731
            out.append(f"| {m} | {t} | {a['fps'] or '–'} | {a['per']:.1f} | {a['dup']:.2f} | {a['size']} | "
                       f"{pc('rules')} | {pc('trees')} | {pc('logistic')} | {pc('mlp')} | "
                       f"{g['rules']['width']:.0f} |" if "rules" in g else "")
    out.append("")
print("\n".join(out))
