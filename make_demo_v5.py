#!/usr/bin/env python3
"""Demo video of the current model (v5, all 9 classes) on a match segment.
CPU inference so as not to fight v6 for the GPU. Short clip ~8 s.
Output: /work/demo_v5_0811.mp4"""
import cv2, numpy as np, time
from ultralytics import YOLO

SRC = '/root/m1.mp4'
T0 = 2400.0            # seconds: segment with referees (manually annotated)
DUR = 8.0
STRIDE = 2             # every 2nd frame -> 15 fps output
OUT = '/work/demo_v5_0811.mp4'

model = YOLO('/runs/rink_referees_v5/weights/best.pt')
CLASSES = ['red_line', 'blue_line', 'center_circle', 'faceoff_spot', 'faceoff_circle',
           'goal_crease', 'board', 'player', 'referee']
# BGR colours: geometry muted, people vivid
COL = {0:(60,60,230), 1:(255,150,40), 2:(230,230,230), 3:(180,105,255), 4:(255,255,140),
       5:(255,200,80), 6:(110,110,110), 7:(0,215,0), 8:(0,255,255)}

cap = cv2.VideoCapture(SRC)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
cap.set(cv2.CAP_PROP_POS_MSEC, T0 * 1000)
W = int(cap.get(3)); H = int(cap.get(4))
scale = 1280.0 / W
outW, outH = 1280, int(H * scale)
vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'), fps / STRIDE, (outW, outH))

t_end = T0 + DUR
n = 0
t0 = time.time()
while True:
    ok = cap.grab()
    if not ok:
        break
    pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if pos > t_end:
        break
    if idx % STRIDE:
        continue
    ok, frame = cap.retrieve()
    if not ok:
        continue
    r = model.predict(frame, imgsz=960, device='cpu', verbose=False, conf=0.25)[0]
    ov = frame.copy()
    if r.masks is not None:
        for j in range(len(r.boxes)):
            cls = int(r.boxes.cls[j])
            conf = float(r.boxes.conf[j])
            m = (r.masks.data[j].cpu().numpy() > 0.5).astype(np.uint8) * 255
            m = cv2.resize(m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            col = COL.get(cls, (200, 0, 200))
            if cls in (7, 8):
                alpha = 0.35
                xy = np.argwhere(m > 0)
                if len(xy):
                    y1, x1 = xy.min(0); y2, x2 = xy.max(0)
                    cv2.rectangle(ov, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(ov, f'{CLASSES[cls]} {conf:.2f}', (x1, max(0, y1 - 6)),
                                0, 0.55, col, 1, cv2.LINE_AA)
            else:
                alpha = 0.45
            layer = ov.copy()
            layer[m > 0] = col
            cv2.addWeight(layer, alpha, ov, 1 - alpha, 0, ov)
    vis = cv2.resize(ov, (outW, outH))
    bar_h = 34
    cv2.rectangle(vis, (0, outH - bar_h), (outW, outH), (15, 15, 15), -1)
    cv2.putText(vis, f'v5 (08.11)  t={pos:.1f}s', (10, outH - 10), 0, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    xoff = 200
    for ci in range(9):
        cv2.rectangle(vis, (xoff, outH - 24), (xoff + 14, outH - 10), COL[ci], -1)
        cv2.putText(vis, CLASSES[ci][:10], (xoff + 18, outH - 12), 0, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        xoff += 18 + 62
    vw.write(vis)
    n += 1
    if n % 20 == 0:
        print(f'{n} frames, {time.time()-t0:.0f}s', flush=True)
cap.release()
vw.release()
print(f'DONE {n} frames -> {OUT} ({time.time()-t0:.0f}s)', flush=True)
