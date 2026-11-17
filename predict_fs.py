#!/usr/bin/env python3
"""CPU inference of v4 (yolov8m-seg) over fs frames on the GPU server (GPU is busy with v5).
Output: /work/out/masks.json — {stem: [{cls, conf, poly:[[u,v],...]}]},
classes 0 red_line, 1 blue_line, 2 center_circle, 5 goal_crease, 6 board."""
import glob, json, os
import cv2, numpy as np
from ultralytics import YOLO

model = YOLO('/runs/rink_players_v4/weights/best.pt')
KEEP = {0, 1, 2, 5, 6}
out = {}
paths = sorted(glob.glob('/work/data/*.jpg'))
print(f'frames: {len(paths)}', flush=True)
for i, p in enumerate(paths):
    stem = os.path.basename(p)[:-4]
    r = model.predict(p, imgsz=1088, device='cpu', verbose=False, conf=0.18)[0]
    h0, w0 = r.orig_shape
    recs = []
    if r.masks is not None:
        for j in range(len(r.boxes)):
            cls = int(r.boxes.cls[j])
            conf = float(r.boxes.conf[j])
            if cls not in KEEP:
                continue
            m = (r.masks.data[j].cpu().numpy() > 0.5).astype(np.uint8) * 255
            m = cv2.resize(m, (w0, h0), interpolation=cv2.INTER_NEAREST)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            c = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True)
            recs.append({'cls': cls, 'conf': round(conf, 3),
                         'poly': c.reshape(-1, 2).tolist()})
    out[stem] = recs
    if i % 20 == 0:
        print(f'{i}/{len(paths)}', flush=True)
os.makedirs('/work/out', exist_ok=True)
json.dump(out, open('/work/out/masks.json', 'w'))
print('DONE', len(out), flush=True)
