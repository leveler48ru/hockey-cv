#!/usr/bin/env python3
"""Cache of scoreboard plates over the whole m1 (one video pass, then thresholds are tuned offline).

m1 = ./media/8802578402043.mp4 (1280x720, 29.96 fps, 70:07).
The scoreboard is stable: median bbox from sb_scan = (63,40,308,60).

Does:
  1) step 0.5 s -> grayscale 308x60 plate as uint8: hockey/sb_plates.npy + sb_probe.json (t, bbox, wide?)
  2) separate: contour over the whole top band -> if the scoreboard plate merges with a wide
     overlay (bbox > 500 px along X), save the tile (up to 80) — these are event candidates
     (goal graphic, intermission, replay).
Runtime: ~20 min (frame seeking on a 70-minute file).
"""
import cv2, numpy as np, os, json

SRC = './media/8802578402043.mp4'
OUT = os.path.dirname(os.path.abspath(__file__))
X, Y, W, H = 63, 40, 308, 60
STEP = 0.5

cap = cv2.VideoCapture(SRC)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
step_f = max(1, int(round(STEP * fps)))
nprobe = total // step_f
plates = np.zeros((nprobe, H, W), np.uint8)
rows = []
top = np.zeros((nprobe, 2), np.float32)  # width/area of the top contour (for the "wide overlay" check)

def contour_stats(fr):
    h, w = fr.shape[:2]
    roi = fr[0:int(h * 0.30), 0:int(w * 0.60)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 110, 70), (12, 255, 255)) | cv2.inRange(hsv, (168, 110, 70), (180, 255, 255))
    dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 55))
    m = cv2.morphologyEx(red | dark, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return (0, 0, 0, 0)
    return cv2.boundingRect(max(cs, key=cv2.contourArea))

i = 0
n = 0
wide_tiles = 0
while n < total and i < nprobe:
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, fr = cap.read()
    if not ok:
        break
    t = n / fps
    bb = contour_stats(fr)
    plates[i] = cv2.cvtColor(fr[Y:Y + H, X:X + W], cv2.COLOR_BGR2GRAY)
    top[i] = (bb[2], bb[3])
    wide = bb[2] > 500
    rows.append({'t': round(t, 2), 'bbox': bb, 'wide': bool(wide)})
    if wide and wide_tiles < 80:
        cv2.imwrite(f'{OUT}/sb_wide/{int(t):05d}.jpg', fr[0:230, 0:768])
        wide_tiles += 1
    i += 1
    n += step_f
    if i % 500 == 0:
        print(f'{i}/{nprobe} probed', flush=True)

cap.release()
plates = plates[:i]
np.save(f'{OUT}/sb_plates.npy', plates)
json.dump({'fps': fps, 'total': total, 'dur': total / fps, 'step': STEP,
           'n': i, 'plate_bbox': [X, Y, W, H], 'probes': rows},
          open(f'{OUT}/sb_probe.json', 'w'))
print('done: probed', i, '| wide overlays', sum(1 for r in rows if r['wide']),
      '| tiles saved', wide_tiles)
