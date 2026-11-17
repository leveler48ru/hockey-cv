#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""number_scrape.py — cutting player crops for jersey numbers (for an annotation batch).
Every 60th frame (2 s), YOLO puck_v2 classes 0/1/3 (goaltender/player/referee),
detection at imgsz=640 half (fast); the crop is taken from the original 1280
frame and upscaled x2 cubic - the annotator needs to see small numbers (20 px).
Minimum player height 44 px (below that the number is unreadable even by eye).
Manifest: /work/numbers/manifest.jsonl {f,t,cls,box,conf,img}.
"""
import json, os
import cv2
from ultralytics import YOLO

SRC = "/work/m1.mp4"
OUTD = "/work/numbers"
STEP = 60
FPS = 29.96
# Annotation batch: not 50k crops, but a sample. Write the manifest for ALL
# detections (useful for weak-label training), save images for a
# deterministic 1/8 sample (name hash) with a hard cap.
import hashlib
import os as _os
KEEP_DEN = int(_os.environ.get("KEEP_DEN", 8)); MAX_CROPS = 9000
os.makedirs(OUTD, exist_ok=True)

m = YOLO("/work/puck_v2_best.pt")
mf = open(os.path.join(OUTD, "manifest.jsonl"), "w")
nsampled = nc = 0
for i, r in enumerate(m.predict(source=SRC, stream=True, classes=[0, 1, 3],
                                conf=0.35, imgsz=640, device=0, half=True,
                                verbose=False)):
    if i % STEP:
        continue
    nsampled += 1
    im = r.orig_img
    if r.boxes is None or len(r.boxes) == 0:
        continue
    for b in r.boxes:
        cls = int(b.cls[0]); x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
        h = y2 - y1
        if h < 44 or h > 420:
            continue
        x1 = max(0, x1 - 2); y1 = max(0, y1 - 2)
        x2 = min(im.shape[1], x2 + 2); y2 = min(im.shape[0], y2 + 2)
        name = f"c_{i:06d}_{cls}_{nc}.jpg"
        keep = nc < MAX_CROPS and int(hashlib.md5(name.encode()).hexdigest()[:4], 16) % KEEP_DEN == 0
        if keep:
            crop = im[y1:y2, x1:x2]
            crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(OUTD, name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        mf.write(json.dumps({"f": i, "t": round(i / FPS, 2), "cls": cls,
                             "box": [x1, y1, x2, y2], "conf": round(float(b.conf[0]), 3),
                             "img": name, "keep": keep}) + "\n")
        nc += 1
    if nsampled % 200 == 0:
        print(f"sampled={nsampled} crops={nc}", flush=True)
mf.close()
print(f"DONE sampled={nsampled} crops={nc}", flush=True)
