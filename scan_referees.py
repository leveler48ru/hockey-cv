#!/usr/bin/env python3
"""Run v6 over all wide frames (wide + wide_new) on the GPU.
Output /root/refscan/scan.json: for each frame — player(7) and referee(8) detections
with boxes and conf. Then locally we compute stripedness and assemble a labeling batch."""
import cv2, numpy as np, glob, json, os, time
from ultralytics import YOLO

model = YOLO('/runs/rink_all_v6/weights/best.pt')
OUT = {}
t0 = time.time()
paths = sorted(glob.glob('/root/wide/*.jpg')) + sorted(glob.glob('/root/wide_new/*.jpg'))
print('frames:', len(paths), flush=True)
for i, p in enumerate(paths):
    stem = os.path.basename(p)[:-4]
    r = model.predict(p, imgsz=1088, device=0, verbose=False, conf=0.10)[0]
    recs = []
    for j in range(len(r.boxes)):
        c = int(r.boxes.cls[j])
        if c not in (7, 8):
            continue
        x1, y1, x2, y2 = [float(v) for v in r.boxes.xyxy[j]]
        recs.append({'c': c, 'conf': round(float(r.boxes.conf[j]), 3),
                     'x1': round(x1, 1), 'y1': round(y1, 1),
                     'x2': round(x2, 1), 'y2': round(y2, 1)})
    OUT[stem] = recs
    if i % 200 == 0:
        print(f'{i}/{len(paths)} {time.time()-t0:.0f}s', flush=True)
json.dump(OUT, open('/root/refscan/scan.json', 'w'))
n_ref = sum(1 for v in OUT.values() for r in v if r['c'] == 8)
n_ref_hi = sum(1 for v in OUT.values() for r in v if r['c'] == 8 and r['conf'] > 0.4)
frames_ref = sum(1 for v in OUT.values() if any(r['c'] == 8 for r in v))
print(f'DONE: frames {len(OUT)}, referee detections {n_ref} (conf>0.4: {n_ref_hi}), frames with a ref {frames_ref}, {time.time()-t0:.0f}s', flush=True)
