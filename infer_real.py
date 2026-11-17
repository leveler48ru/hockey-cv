#!/usr/bin/env python3
from ultralytics import YOLO
import cv2

m = YOLO('runs/rink_real-3/weights/best.pt')
cap = cv2.VideoCapture('match.mp4')
for ts in (300, 900, 1800):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * 30))
    ok, fr = cap.read()
    if not ok:
        continue
    cv2.imwrite(f'test_{ts}.png', fr)
    r = m.predict(f'test_{ts}.png', conf=0.25, device=0, verbose=False)[0]
    r.save(f'segm_{ts}.jpg')
    cls = [m.names[int(c)] for c in r.boxes.cls.tolist()]
    conf = [round(float(c), 2) for c in r.boxes.conf.tolist()]
    nm = 0 if r.masks is None else len(r.masks)
    print(f't={ts}s: {len(cls)} detections, {nm} masks')
    for c, cf in sorted(zip(cls, conf), key=lambda x: -x[1])[:10]:
        print(f'  {c:15s} {cf}')
cap.release()
print('DONE')
