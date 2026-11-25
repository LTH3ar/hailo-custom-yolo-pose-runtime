"""Handoff 07 (uncommitted): seats calibrated per analysis file: distinct seats ever marked calibrated,
seats seen, and the final inferred class size.  usage: seats_cal.py ANALYSIS.jsonl [...]"""
import json, sys
for p in sys.argv[1:]:
    cal, seen, size = set(), set(), None
    for d in map(json.loads, open(p)):
        if d.get("type") != "frame":
            continue
        size = (d.get("class") or {}).get("class_size", size)
        for q in d["persons"]:
            e = q.get("engagement") or {}
            if e.get("seat") is None:
                continue
            seen.add(e["seat"])
            if e.get("calibrated"):
                cal.add(e["seat"])
    print(f"{p.split('/')[-1]:44} seats calibrated {len(cal)}, seats seen {len(seen)}, class size {size}")
