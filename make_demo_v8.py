#!/usr/bin/env python3
"""Demo v8: referee + player recognition on an m1 segment.
One frame: original broadcast, overlaid masks/boxes only for classes player(7)/referee(8).
Referee - cyan 'REF' label, player - no text. Bottom-left: counter of referees in frame.
Output: /work/demo_v8.mp4 (8 s, 30 fps, stride=2 sampled from the 30fps video)."""
import cv2, os
from ultralytics import YOLO

SRC = '/work/m1.mp4'
T0 = float(os.environ.get('DEMO_T0', 2400))
SECS = 8.0
OUT = '/work/demo_v8.mp4'
CONF = float(os.environ.get('DEMO_CONF', 0.25))

model = YOLO(os.environ.get('DEMO_W', '/runs/rink_ref_v8/weights/best.pt'))

cap = cv2.VideoCapture(SRC)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
cap.set(cv2.CAP_PROP_POS_MSEC, T0 * 1000)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
scale = 1280.0 / W
dim = (1280, int(H * scale))
vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'), 30, dim)

n = int(fps * SECS)
ref_total = 0
k = 0
while k < n:
    ok, frame = cap.read()
    if not ok:
        break
    k += 1
    if k % 2:  # stride 2: every second frame
        continue
    r = model.predict(frame, imgsz=1088, conf=CONF, verbose=False)[0]
    img = cv2.resize(frame, dim)
    refs = 0
    if r.boxes is not None:
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy()
        cf = r.boxes.conf.cpu().numpy()
        for b, cid, c in zip(xyxy, cls, cf):
            cid = int(cid)
            if cid == 8:
                refs += 1
                x1, y1, x2, y2 = [int(v * scale) for v in b]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(img, f'REF {c:.2f}', (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            elif cid == 7:
                x1, y1, x2, y2 = [int(v * scale) for v in b]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 215, 0), 1)
    if r.masks is not None and len(r.masks.data):
        segs = r.masks.data.cpu().numpy()
        for idx, cid in enumerate(cls):
            if int(cid) != 8 or idx >= len(segs):
                continue
            m = cv2.resize(segs[idx], (dim[0], dim[1])) > 0.3
            overlay = img.copy()
            overlay[m] = (0, 255, 255)
            img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    ref_total += refs
    cv2.putText(img, f'm1  t={T0 + k / fps:7.1f}s   referees: {refs}', (14, dim[1] - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    vw.write(img)
cap.release(); vw.release()
print('WROTE', OUT, 'frames', k // 2, 'sum_ref_frames', ref_total)
