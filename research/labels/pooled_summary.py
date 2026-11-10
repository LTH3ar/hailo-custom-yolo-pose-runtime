"""Pooled (clip_3 + clip) summary of the INT8 analyses in ~/clips_results vs student_labels_v1.json.

Same matching as score_labels.py (nearest frame within 0.25 s, best box IoU >= 0.3). An item
with no match, or matched to state "unknown", counts as not flagged. Run from the repo root:
    .venv/bin/python research/labels/pooled_summary.py
"""
import json, os, numpy as np
L = [x for x in json.load(open('research/labels/student_labels_v1.json'))['items'] if x['label'] in ('focused', 'distracted')]
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9)
def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** .5 / d
    return c - h, c + h
R = os.path.expanduser('~/clips_results')
print('| Model | Tiles | Decision | Not matched (of 275) | False alarms on focused | Specificity (95% CI) | Distracted caught | False alarms, matched focused only |')
print('|---|---|---|---|---|---|---|---|')
for m in 'nm':
    for t in ('none', 'auto'):
        for c in ('none', 'trees', 'mlp', 'logistic'):
            st = {}
            for clip in ('clip_3', 'clip'):
                fr = [d for d in map(json.loads, open(f'{R}/{clip}_{m}_{t}_{c}.jsonl')) if d.get('type') == 'frame']
                ts = np.array([d['t'] for d in fr])
                for i, x in enumerate(L):
                    if x['clip'] != clip: continue
                    k = int(np.argmin(np.abs(ts - x['t']))); s = None
                    if abs(ts[k] - x['t']) <= .25:
                        cand = [(iou(x['box'], q['box']), q) for q in fr[k]['persons']]
                        cand = [z for z in cand if z[0] >= .3]
                        if cand: s = (max(cand, key=lambda z: z[0])[1].get('engagement') or {}).get('state')
                    st[i] = s if s in ('focused', 'distracted') else None
            foc = [i for i, x in enumerate(L) if x['label'] == 'focused']; dis = [i for i, x in enumerate(L) if x['label'] == 'distracted']
            nm = sum(st[i] is None for i in st)
            fa = sum(st[i] == 'distracted' for i in foc); caught = sum(st[i] == 'distracted' for i in dis)
            mfoc = [i for i in foc if st[i] is not None]
            lo, hi = wilson(len(foc) - fa, len(foc))
            name = {'none': 'Rules', 'trees': 'Trees', 'mlp': 'MLP', 'logistic': 'Logistic'}[c]
            print(f"| {m} | {'on' if t == 'auto' else 'off'} | {name} | {nm} | {fa} / {len(foc)} | {100*(len(foc)-fa)/len(foc):.0f}% ({100*lo:.0f}-{100*hi:.0f}%) | "
                  f"{caught} / {len(dis)} | {fa} / {len(mfoc)} ({100*fa/max(len(mfoc),1):.0f}%) |")
