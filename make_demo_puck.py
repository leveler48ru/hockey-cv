#!/usr/bin/env python3
"""Puck demo: puck_s1280 on an m1 segment. Players in thin green, puck as a yellow box
with a zoom around it (3x magnifier) and a PUCK conf label. Bottom: counter of frames containing the puck."""
import cv2, os
from ultralytics import YOLO

SRC = '/work/m1.mp4'
T0 = float(os.environ.get('DEMO_T0', 2118))
SECS = 8.0
OUT = '/work/demo_puck.mp4'
CONF = float(os.environ.get('DEMO_CONF', 0.20))

model = YOLO('/work/runs/puck_s1280/weights/best.pt')
cap = cv2.VideoCapture(SRC)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
cap.set(cv2.CAP_PROP_POS_MSEC, T0 * 1000)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
scale = 1280.0 / W
dim = (1280, int(H * scale))
vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'), 30, dim)

n = int(fps * SECS); k = 0
frames_with_puck = 0
written = 0
while k < n:
    ok, frame = cap.read()
    if not ok:
        break
    k += 1
    if k % 2:
        continue
    r = model.predict(frame, imgsz=1280, conf=CONF, verbose=False)[0]
    img = cv2.resize(frame, dim)
    has_puck = False
    if r.boxes is not None:
        xyxy = r.boxes.xyxy.cpu().numpy(); cls = r.boxes.cls.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
        for b, cid, c in zip(xyxy, cls, cf):
            cid = int(cid)
            x1, y1, x2, y2 = [int(v * scale) for v in b]
            if cid == 1:  # player
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 1)
            elif cid == 2:  # PUCK
                has_puck = True
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(img, f'PUCK {c:.2f}', (x1 - 20, max(16, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                # 3x magnifier around the puck
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                half = 55
                zx1, zy1 = max(0, cx - half), max(0, cy - half)
                zx2, zy2 = min(dim[0], cx + half), min(dim[1], cy + half)
                zoom = cv2.resize(img[zy1:zy2, zx1:zx2], (240, 240), interpolation=cv2.INTER_NEAREST)
                img[10:250, dim[0] - 250:dim[0] - 10] = zoom
                cv2.rectangle(img, (dim[0] - 250, 10), (dim[0] - 10, 250), (0, 255, 255), 2)
    if has_puck:
        frames_with_puck += 1
    cv2.putText(img, f'm1 t={T0 + k / fps:6.1f}s  puck_in_frame: {int(has_puck)}  total: {frames_with_puck}',
                (14, dim[1] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    vw.write(img)
    written += 1
cap.release(); vw.release()
print('WROTE', OUT, 'frames', written, 'puck_frames', frames_with_puck)
