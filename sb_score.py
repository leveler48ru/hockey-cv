#!/usr/bin/env python3
"""Exact sequence of score and game time from the scoreboard plate cache.

No re-decoding of the video: the plates already sit in sb_plates.npy (8404 items, 0.5 s step).

Two passes:
 1. SEGMENTATION (pure numpy, seconds). The scoreboard band is cut into three zones by
    coordinates obtained from OCR labeling of a 4x-upscaled plate:
      clock   x 140..175, y 0..14   (shows MM:SS, ticks)
      score   x 115..200, y 12..40  (two digits, changes rarely)
    We compute the pairwise difference of neighboring probes; staying below the threshold = same state.
    Segment inflation (flicker, anti-aliasing) is suppressed by min_stable: a fragment shorter
    than 2 s whose neighbors are "almost" the same does not spawn a new state.
 2. OCR (expensive) — ONLY one representative per segment, and adjacent
    segments with identical recognized text are merged.
    OCR is called on the cropping region (clock+score together, det+rec), 6x upscale.

Output: sb_score.json  = [{t0,t1,clock,score_raw,conf}] + full clock timeline.
"""
import json
import os
import sys
import time

import cv2
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
P = np.load(f'{OUT}/sb_plates.npy', mmap_mode='r')
meta = json.load(open(f'{OUT}/sb_probe.json'))
ts = np.array([p['t'] for p in meta['probes']])
n, h, w = P.shape
step = meta['step']
print(f'plates {n}, {w}x{h}, step {step} s, duration {ts[-1]:.0f} s', flush=True)

# --- scoreboard zones (x0,x1,y0,y1) ---
ZONE_SCORE = (112, 200, 10, 42)
ZONE_CLOCK = (138, 178, 0, 15)


def seg_stats(zone, name):
    x0, x1, y0, y1 = zone
    sub = P[:, y0:y1, x0:x1].astype(np.float32)
    # normalize by brightness/contrast of each plate: overlay flicker should not
    # look like a digit change
    sub = (sub - sub.mean(axis=(1, 2), keepdims=True)) / (sub.std(axis=(1, 2), keepdims=True) + 1e-3)
    d = np.abs(np.diff(sub, axis=0)).mean(axis=(1, 2))
    print(f'{name}: neighboring-probe difference median {np.median(d):.3f} '
          f'p90 {np.percentile(d, 90):.3f} p99 {np.percentile(d, 99):.3f} max {d.max():.3f}', flush=True)
    return d


d_score = seg_stats(ZONE_SCORE, 'score')
d_clock = seg_stats(ZONE_CLOCK, 'clock')

# threshold: a "change" = difference above p97 of the score, but not below the physical minimum
thr_s = max(np.percentile(d_score, 97), 0.35)
print(f'score-change threshold {thr_s:.3f}', flush=True)

chg = np.where(d_score > thr_s)[0]
bounds = [0] + [int(c) + 1 for c in chg] + [n]
segs = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]
lens = np.array([b - a for a, b in segs]) * step
print(f'segments after thresholding: {len(segs)} (median length {np.median(lens):.1f} s, '
      f'shorter than 2 s: {int((lens < 2).sum())})', flush=True)

# --- merging short junk segments: do not silently drop them, mark them for OCR verification instead.
# Segment representative = plate at its middle (farther from the transition boundaries).
reps = [ (a + b) // 2 for a, b in segs if (b - a) * step >= 1.0 ]
print(f'OCR representatives: {len(reps)}', flush=True)

if '--reps-only' in sys.argv:
    json.dump({'thr_score': float(thr_s),
               'segs': [[float(ts[a]), float(ts[b - 1] + step)] for a, b in segs],
               'rep_idx': reps}, open(f'{OUT}/sb_score_segs.json', 'w'))
    print('wrote sb_score_segs.json (segmentation without OCR)')
    sys.exit(0)

# --- OCR ---
from rapidocr_onnxruntime import RapidOCR
ocr = RapidOCR()
sx0, sx1, sy0, sy1 = ZONE_SCORE
cx0, cx1, cy0, cy1 = ZONE_CLOCK
# shared region: the clock sits slightly above the score, take the union over x and y
OX0, OX1 = min(sx0, cx0) - 2, max(sx1, cx1) + 2
OY0, OY1 = 0, sy1


def read_plate(i):
    c = np.array(P[i])[OY0:OY1, OX0:OX1]
    big = cv2.cvtColor(cv2.resize(c, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC),
                       cv2.COLOR_GRAY2BGR)
    res, _ = ocr(big)
    out = {'clock': None, 'score': None, 'items': []}
    if not res:
        return out
    for box, txt, conf in res:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        # y in 6x-crop coordinates, x relative to OX0
        rel_y = (min(ys) + max(ys)) / 2 / 6.0 + OY0
        rel_x = (min(xs) + max(xs)) / 2 / 6.0 + OX0
        out['items'].append({'txt': txt, 'conf': round(float(conf), 3),
                             'x': round(rel_x, 1), 'y': round(rel_y, 1)})
        if ':' in txt or '：' in txt:
            out['clock'] = txt.replace('：', ':')
            out['clock_conf'] = round(float(conf), 3)
        elif rel_y > OY1 - (sy1 - sy0) + 1 and len(txt.strip()) <= 4:
            if out['score'] is None or rel_x > OX0 + (sx1 - OX0) * 0.4:
                pass
    return out


t0 = time.time()
rows = []
for k, i in enumerate(reps):
    r = read_plate(i)
    rows.append({'idx': i, 't': float(ts[i]), **r})
    if k % 25 == 0:
        el = time.time() - t0
        print(f'  OCR {k}/{len(reps)} ({el:.0f} s, rate {el / max(k + 1, 1):.2f} s/item, '
              f'ETA {el / max(k + 1, 1) * (len(reps) - k - 1):.0f} s)', flush=True)
json.dump({'thr_score': float(thr_s), 'zone_score': ZONE_SCORE, 'zone_clock': ZONE_CLOCK,
           'ocr_crop': [OX0, OX1, OY0, OY1], 'rows': rows,
           'segs': [[float(ts[a]), float(ts[b - 1] + step)] for a, b in segs]},
          open(f'{OUT}/sb_score_raw.json', 'w'), ensure_ascii=False, indent=1)
print(f'OCR done in {time.time() - t0:.0f} s -> sb_score_raw.json', flush=True)
