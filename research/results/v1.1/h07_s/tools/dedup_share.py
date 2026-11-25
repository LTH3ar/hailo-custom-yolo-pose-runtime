"""Handoff 07 (uncommitted): share of detections pose_dedup removes, as in Handoff 02 E4:
1 - detections(dedup cache) / detections(no-dedup cache), at score >= 0.25 (all decoder output) and >= 0.3.
usage: dedup_share.py DEDUP.jsonl NODEDUP.jsonl [...pairs]"""
import json, sys
def count(p):
    a = b = n = 0
    for d in map(json.loads, open(p)):
        n += 1
        for x in d["dets"]:
            a += x["score"] >= 0.25
            b += x["score"] >= 0.3
    return n, a, b
args = sys.argv[1:]
for dd, nd in zip(args[::2], args[1::2]):
    (n1, a1, b1), (n2, a2, b2) = count(dd), count(nd)
    print(f"{dd.split('/')[-1]} vs {nd.split('/')[-1]}: images {n1}/{n2}; >=0.25 {a1} vs {a2} -> removed {1 - a1 / a2:.1%}; "
          f">=0.3 {b1} vs {b2} -> removed {1 - b1 / b2:.1%}")
