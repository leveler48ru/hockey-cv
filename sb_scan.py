#!/usr/bin/env python3
"""Scoreboard audit over the whole m1: whether the board is present all the time, where it is, and how
many distinct score states can be seen in it. No OCR: cut the plate, cluster by
pixel distance; distinct states = candidates for a score change (goal).
Output: hockey/sb_scan.json + hockey/sb_tiles/*.jpg
Run: python3 hockey/sb_scan.py [step_seconds]"""
import cv2, numpy as np, os, sys, json

SRC = './media/8802578402043.mp4'
STEP = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
OUT = os.path.dirname(os.path.abspath(__file__))
TILES = f'{OUT}/sb_tiles'
os.makedirs(TILES, exist_ok=True)


def plate(frame):
    """Scoreboard bbox in the top-left zone (HSV: red+dark, 21px morphology)."""
    h, w = frame.shape[:2]
    roi = frame[0:int(h * 0.30), 0:int(w * 0.60)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 110, 70), (12, 255, 255)) | cv2.inRange(hsv, (168, 110, 70), (180, 255, 255))
    dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 55))
    m = cv2.morphologyEx(red | dark, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    x, y, ww, hh = cv2.boundingRect(max(cs, key=cv2.contourArea))
    return (x, y, ww, hh) if ww > 80 and hh > 25 else None


cap = cv2.VideoCapture(SRC)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
dur = total / fps
step_f = max(1, int(round(STEP * fps)))
timeline, seen, sig = [], [], []
n = 0
while n < total:
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, fr = cap.read()
    if not ok:
        break
    t = n / fps
    bb = plate(fr)
    if bb:
        x, y, ww, hh = bb
        crop = fr[y:y + hh, x:x + ww]
        small = cv2.resize(crop, (240, 72)).astype(np.float32) / 255.0
        timeline.append({'t': round(t, 1), 'bbox': bb, 'area': ww * hh})
        sig.append(small)
        if len(sig) % 40 == 1 and len(seen) < 40:
            fn = f'{TILES}/{int(t):05d}.jpg'
            cv2.imwrite(fn, crop)
            seen.append(fn)
    else:
        timeline.append({'t': round(t, 1), 'bbox': None})
    n += step_f

present = [s for s in timeline if s['bbox']]
print(f'duration {dur/60:.1f} min, passes {len(timeline)} (step {STEP} s), board found in {len(present)} = {100*len(present)/max(1,len(timeline)):.1f}%')
if sig:
    sig = np.stack(sig)
    # median plate = state template; difference clusters = distinct states
    med = np.median(sig, 0)
    d = np.abs(sig - med).mean((1, 2, 3))
    thr = d.mean() + 2 * d.std()
    odd = np.where(d > thr)[0]
    print(f'plates differing from the median (>mean+2sd): {len(odd)}')
    ts = [present[i]['t'] for i in odd[:40]]
    print('t:', ts)
json.dump({'dur': dur, 'step': STEP, 'present_pct': 100*len(present)/max(1, len(timeline)),
           'timeline': timeline, 'tiles': seen}, open(f'{OUT}/sb_scan.json', 'w'))
print('sb_scan.json written, tiles:', len(seen))
