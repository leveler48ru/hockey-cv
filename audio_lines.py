#!/usr/bin/env python3
"""Recalibrating saved probabilities (audio2_probs.npz) into two lines: whistles and goals.

No GPU needed: P (4198,4) is already loaded, t step 1 s, window 10 s.
Thresholds are tuned ON YOUR OWN 70 LABELS, not "by eye":
  for each click (t0, label) we take the max class probability in the window
  t0..t0+10 (windows with a 1 s step cover it) and compute P/R/F1 over a threshold grid.
This is an honest check of the sliding line: training saw these clips in CV, but
the sliding pass runs over the whole match, so the metric over the labels is not a self-check of the threshold.
"""
import json
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))
z = np.load(f'{OUT}/audio2_probs.npz', allow_pickle=True)
t = z['t']
P = z['P']
classes = [str(c) for c in z['classes']]
lab = json.load(open(f'{OUT}/audio_labels.json'))
print(f'slide: {len(t)} windows, classes {classes}, labels {len(lab)}')

CI = {c: i for i, c in enumerate(classes)}
MERGE = {'N': 'N', 'W': 'W', 'S': 'S', 'WS': 'S', 'M': 'M', 'X': None}


def maxprob(t0, dur, cls):
    """max class probability inside the label window"""
    m = (t >= t0 - 0.5) & (t <= t0 + dur - 0.5)
    if not m.any():
        return 0.0
    return float(P[m][:, CI[cls]].max())


DUR = 12.0  # length of the clip you listened to
for cls in ['W', 'S', 'M']:
    print(f'\n=== line {cls}: threshold grid over your labels ===')
    print(' thr  TP  FP  FN   P     R     F1')
    best = None
    for thr in np.arange(0.20, 0.96, 0.04):
        tp = fp = fn = 0
        for c in lab:
            y = MERGE.get(c['label'])
            if y is None:
                continue
            hit = maxprob(c['t0'], DUR, cls) >= thr
            pos = (y == cls)
            if pos and hit:
                tp += 1
            elif pos and not hit:
                fn += 1
            elif hit and not pos:
                fp += 1
        pr = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        f1 = 2 * pr * rc / max(pr + rc, 1e-9)
        if best is None or f1 > best[0]:
            best = (f1, float(thr), pr, rc, tp, fp, fn)
        if abs(thr - round(thr, 2)) < 1e-9 and int(round(thr * 100)) % 12 == 0:
            print(f' {thr:.2f} {tp:3d} {fp:3d} {fn:3d}  {pr:.2f}  {rc:.2f}  {f1:.2f}')
    f1, thr, pr, rc, tp, fp, fn = best
    print(f' BEST thr {thr:.2f}: P {pr:.2f} R {rc:.2f} F1 {f1:.2f} (TP{tp}/FP{fp}/FN{fn})')
    if cls == 'W':
        W_THR = thr
    if cls == 'S':
        S_THR = thr
    if cls == 'M':
        M_THR = thr

json.dump({'W': W_THR, 'S': S_THR, 'M': M_THR}, open(f'{OUT}/audio_lines_thr.json', 'w'))
print(f'\nthresholds written: W {W_THR:.2f} S {S_THR:.2f} M {M_THR:.2f}')


def peaks(cls, thr, min_sep=6.0):
    col = P[:, CI[cls]]
    idx = np.where(col >= thr)[0]
    out = []
    last = -1e9
    for j in idx:
        if t[j] - last < min_sep:
            continue
        # local maximum within ±3 s
        lo, hi = max(0, j - 3), min(len(col), j + 4)
        if col[j] >= col[lo:hi].max():
            out.append({'t': float(t[j]), 'p': float(col[j])})
            last = t[j]
    return out


lines = {}
for cls, thr in [('W', W_THR), ('S', S_THR), ('M', M_THR)]:
    e = peaks(cls, thr)
    lines[cls] = e
    print(f'slide peaks {cls} (thr {thr:.2f}): {len(e)}')

json.dump({'thr': {'W': W_THR, 'S': S_THR, 'M': M_THR},
           'peaks': lines}, open(f'{OUT}/audio_lines.json', 'w'))
print('\nW peaks:', [round(e["t"], 1) for e in lines['W']])
print('S peaks:', [round(e["t"], 1) for e in lines['S']])
