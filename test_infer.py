#!/usr/bin/env python3
"""Inference of the trained model on real match frames (without ffmpeg)."""
import os, cv2
from ultralytics import YOLO

VID = '/root/match.mp4'
m = YOLO('/root/runs/rink26/weights/best.pt')
names = m.names

cap = cv2.VideoCapture(VID)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'video: fps={fps:.2f}, frames={total}, dur={total/max(fps,1)/60:.1f} min')

for ts in [600, 1200, 2400]:
    src = f'/root/test_{ts}.png'
    if not os.path.exists(src):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
        ok, frame = cap.read()
        if not ok:
            print(f't={ts}s: frame not read'); continue
        cv2.imwrite(src, frame)
    r = m.predict(src, conf=0.10, device=0, verbose=False)[0]
    cls = [names[int(c)] for c in r.boxes.cls.tolist()]
    conf = [round(float(c), 2) for c in r.boxes.conf.tolist()]
    print(f'--- t={ts}s: {len(cls)} detections')
    for c, cf in sorted(zip(cls, conf), key=lambda x: -x[1])[:15]:
        print(f'    {c:15s} {cf}')
    out = f'/root/segm_{ts}.jpg'
    r.save(out)
    print(f'    {out}')
cap.release()
print('DONE')