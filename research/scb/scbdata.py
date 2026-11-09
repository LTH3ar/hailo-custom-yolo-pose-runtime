"""Loading SCB pose caches with duplicate handling and lesson-level groups.

Two problems found when running on the Pi (research/scb/results/hef_results.md):

1. ~1,900 images are byte-identical copies in two or three subsets. Each subset
   annotates only its own behaviors, so a copy in BowTurnHead and the same image
   in Handrise carry different, partly contradictory boxes. Here identical images
   are merged into one record with the union of their labels; a student whose
   merged labels include both a focused and a distracted behavior is dropped as
   ambiguous (counted and reported).
2. Cross-validation groups must be whole lessons. Lesson id = the number before
   "_" ("49_001002" -> 49) or the first 4 digits ("0006001" -> 6). Images that
   are duplicates are additionally merged into one group (union-find), so no
   copy of an image can sit on both sides of a fold.

Class order per subset comes from the dataset's own YAML files on the Hugging
Face mirror: BowTurnHead ['BowHead', 'TurnHead'], Discuss ['discuss'],
Handrise-Read-write ['hand-raising', 'read', 'write'].
"""
import hashlib
import json
import os
import re

FOCUSED = {"read", "write", "hand-raising"}
DISTRACTED = {"TurnHead", "BowHead", "discuss"}


def lesson_key(path):
    name = os.path.basename(path).rsplit(".", 1)[0]
    m = re.match(r"(\d+)_", name)
    return int(m.group(1)) if m else int(name[:4])


def _md5(path, cache):
    if path not in cache:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        cache[path] = h.hexdigest()
    return cache[path]


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def load(poses_path, dedup=True, verbose=True):
    """Returns records like the pose cache lines, plus "group" (int) and with merged "gt"."""
    recs = [json.loads(line) for line in open(poses_path)]
    if not dedup:
        for r in recs:
            r["group"] = f"{r['subset']}:{lesson_key(r['img'])}"
        return recs

    cache_path = poses_path + ".md5.json"
    cache = json.load(open(cache_path)) if os.path.isfile(cache_path) else {}
    for r in recs:
        r["md5"] = _md5(r["img"], cache)
    json.dump(cache, open(cache_path, "w"))

    by_hash = {}
    for r in recs:
        by_hash.setdefault(r["md5"], []).append(r)

    uf = _UF()
    merged, n_dup_images, n_conflicts, n_kept = [], 0, 0, 0
    for h, copies in by_hash.items():
        lessons = {f"L{lesson_key(c['img'])}:{'u' if '_' in os.path.basename(c['img']) else 'c'}" for c in copies}
        first = sorted(lessons)[0]
        for lesson in lessons:
            uf.union(lesson, first)
        base = dict(copies[0])
        base["lesson"] = first
        if len(copies) > 1:
            n_dup_images += 1
            # union of labels; cluster boxes of the same student across copies
            clusters = []  # [box, set(labels)]
            for c in copies:
                for g in c["gt"]:
                    for cl in clusters:
                        if _iou(cl[0], g[1:]) > 0.7:
                            cl[1].add(g[0])
                            break
                    else:
                        clusters.append([g[1:], {g[0]}])
            gt = []
            for box, labels in clusters:
                if labels & FOCUSED and labels & DISTRACTED:
                    n_conflicts += 1
                    continue
                n_kept += 1
                gt.append([sorted(labels)[0]] + list(box))
            base["gt"] = gt
            base["subset"] = "+".join(sorted({c["subset"] for c in copies}))
        merged.append(base)

    roots = {}
    for r in merged:
        r["group"] = roots.setdefault(uf.find(r["lesson"]), len(roots))
    if verbose:
        print(f"[scbdata] {len(recs)} records -> {len(merged)} unique images "
              f"({n_dup_images} duplicated across subsets); {n_conflicts} students with both focused and "
              f"distracted labels dropped; {len(roots)} lesson groups")
    return merged
