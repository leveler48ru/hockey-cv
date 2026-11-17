#!/usr/bin/env python3
"""Selection of wide rink shots from three games.
Criteria: ice area >15%, no full-screen overlay, not a close-up.
The scoreboard is masked out when counting ice.
Output: /root/wide/<vid>_<t>.jpg + /root/wide/index.csv
"""
import cv2, numpy as np, os, csv

VIDS = {'match': '/root/match.mp4', 'm1': '/root/m1.mp4', 'm2': '/root/m2.mp4'}
SB = {  # scoreboard mask (x1,y1,x2,y2), per sb_detect
    'match': (40, 25, 650, 160),
    'm1':    (55, 30, 385, 115),
    'm2':    (50, 25, 720, 175),
}
STEP = 5.0  # seconds between candidates
OUT = '/root/wide'
os.makedirs(OUT, exist_ok=True)

def ice_frac(frame, sb):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # ice = bright, low saturation
    ice = cv2.inRange(hsv, (0, 0, 140), (180, 60, 255))
    x1, y1, x2, y2 = sb
    ice[y1:y2, x1:x2] = 0  # exclude the scoreboard
    ice[:int(h*0.18), :] = 0  # exclude ceiling/structures at the top
    return float(ice.mean()) / 255.0

def overlay_frac(frame):
    """share of bright-red/saturated pixels over the whole frame — full-screen overlay signal"""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 150, 100), (12, 255, 255)) | cv2.inRange(hsv, (168, 150, 100), (180, 255, 255))
    return float(red.mean()) / 255.0

rows = []
for vid, path in VIDS.items():
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    t = 30.0
    while t < dur - 30:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if not ok:
            t += STEP; continue
        icf = ice_frac(fr, SB[vid])
        ovf = overlay_frac(fr)
        if icf > 0.15 and ovf < 0.05:
            name = f'{OUT}/{vid}_{int(t):05d}.jpg'
            cv2.imwrite(name, fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append([vid, int(t), round(icf, 3), round(ovf, 3), name])
        t += STEP
    cap.release()
    print(f'{vid}: kept {sum(1 for r in rows if r[0]==vid)}', flush=True)

with open(f'{OUT}/index.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['vid', 't', 'ice', 'overlay', 'path']); w.writerows(rows)
print(f'total wide frames: {len(rows)}')