#!/usr/bin/env python3
"""Selection of wide rink shots from three games — local variant (NAS, no GPU).
Criteria: ice area >15%, no full-screen overlay.
The scoreboard is masked out when counting ice.
"""
import cv2, numpy as np, os, csv, sys

VIDS = {
    'match': './media/9281020430926.mp4',
    'm1':    './media/8802578402043.mp4',
    'm2':    './media/9387909712554.mp4',
}
SB = {  # scoreboard mask (x1,y1,x2,y2)
    'match': (40, 25, 650, 160),
    'm1':    (55, 30, 385, 115),
    'm2':    (50, 25, 720, 175),
}
STEP = 5.0
OUT = 'wide'
os.makedirs(OUT, exist_ok=True)

def ice_frac(frame, sb):
    h = frame.shape[0]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ice = cv2.inRange(hsv, (0, 0, 140), (180, 60, 255))
    x1, y1, x2, y2 = sb
    ice[y1:y2, x1:x2] = 0
    ice[:int(h*0.18), :] = 0
    return float(ice.mean()) / 255.0

def overlay_frac(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 150, 100), (12, 255, 255)) | cv2.inRange(hsv, (168, 150, 100), (180, 255, 255))
    return float(red.mean()) / 255.0

rows = []
for vid, path in VIDS.items():
    if not os.path.exists(path):
        print(f'MISSING FILE: {path}', flush=True); continue
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    t, kept, checked = 30.0, 0, 0
    while t < dur - 30:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        checked += 1
        if ok:
            icf = ice_frac(fr, SB[vid])
            ovf = overlay_frac(fr)
            if icf > 0.15 and ovf < 0.05:
                name = f'{OUT}/{vid}_{int(t):05d}.jpg'
                cv2.imwrite(name, fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                rows.append([vid, int(t), round(icf, 3), round(ovf, 3), name])
                kept += 1
        t += STEP
        if checked % 100 == 0:
            print(f'  {vid}: checked {checked}, kept {kept}', flush=True)
    cap.release()
    print(f'{vid}: {kept} wide out of {checked} checked', flush=True)

with open(f'{OUT}/index.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['vid', 't', 'ice', 'overlay', 'path']); w.writerows(rows)
print(f'TOTAL wide frames: {len(rows)}', flush=True)