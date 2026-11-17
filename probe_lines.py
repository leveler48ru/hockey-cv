#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_lines.py — honest probe: what exactly v6 segmentation gives on m1 frames.

24 timecodes across the whole match, at each a median of 7 frames -> predict ->
per class (count, conf), per red/blue pixels and per top components
(PCA elongation, length, angle). Plus overlays to disk for eyeballing.

Goal: understand why line-based calibration yields lines2 with 0 red detections.
"""
import cv2, numpy as np, json, os, time

SRC = "/work/m1.mp4"
MODEL = "/work/runs/rink_all_v6/weights/best.pt"
OUTD = "/work/out/probe"
FPS = 29.96
os.makedirs(OUTD, exist_ok=True)
N = int(os.environ.get("N", "24"))
CONF = float(os.environ.get("CONF", "0.25"))
IMGSZ = int(os.environ.get("IMGSZ", "960"))

cap = cv2.VideoCapture(SRC)
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(3)); H = int(cap.get(4))
from ultralytics import YOLO
model = YOLO(MODEL)
names = model.names


def med_frame(target):
    frames = []
    for j in range(7):
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(nframes - 1, target + j * 6))
        ok, fr = cap.read()
        if ok:
            frames.append(fr.astype(np.uint16))
    if len(frames) < 3:
        return None
    return np.median(np.stack(frames, 0), 0).astype(np.uint8)


def comps(mask, tag, vis):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for j in range(1, n):
        x, y, w, h, area = stats[j]
        if area < 200:
            continue
        ys, xs = np.nonzero(lab[y:y + h + 1, x:x + w + 1] == j)
        pts = np.stack([xs + x, ys + y], 1).astype(float)
        c = pts.mean(0)
        d = pts - c
        _, S, Vt = np.linalg.svd(d, full_matrices=False)
        el = S[0] / max(S[1], 1e-6)
        proj = pts @ Vt[0]
        ln = float(np.ptp(proj))
        ang = float(np.degrees(np.arctan2(Vt[0][1], Vt[0][0])))
        th = float(np.median(np.abs(pts @ Vt[1] + c @ Vt[1])))
        out.append(dict(area=int(area), bbox=[int(x), int(y), int(w), int(h)],
                        elong=round(el, 1), length=round(ln), angle=round(ang),
                        thick=round(th, 1), npix=int(len(pts))))
        col = (0, 0, 255) if tag == "red" else (255, 0, 0)
        for p in pts[::40].astype(int):
            cv2.circle(vis, tuple(p), 2, (0, 255, 0), -1)
        cv2.putText(vis, f"{tag}{len(out)}:e{el:.0f} L{ln:.0f}", (x, max(12, y - 4)),
                    0, 0.5, (255, 255, 255), 1)
    return sorted(out, key=lambda z: -z["area"])[:6]


rep = []
t0 = time.time()
for i in range(N):
    target = int((i + 0.5) * nframes / N)
    mf = med_frame(target)
    if mf is None:
        continue
    r = model.predict(mf, imgsz=IMGSZ, device=0, verbose=False, conf=CONF)[0]
    cls = {}
    masks = {}
    if r.boxes is not None:
        for j in range(len(r.boxes)):
            c = int(r.boxes.cls[j])
            nm = names[c] if isinstance(names, dict) else str(c)
            cf = float(r.boxes.conf[j])
            cls[nm] = [cls.get(nm, [0, 0])[0] + 1, round(max(cf, cls.get(nm, [0, 0])[1]), 2)]
            if c not in (0, 1) or r.masks is None:
                continue
            mm = (r.masks.data[j].cpu().numpy() > 0.5).astype(np.uint8) * 255
            mm = cv2.resize(mm, (W, H), interpolation=cv2.INTER_NEAREST)
            masks[c] = np.maximum(masks.get(c, np.zeros((H, W), np.uint8)), mm)
    vis = mf.copy()
    row = dict(t=round(target / FPS, 1), frame=target, cls=cls,
               red_px=int((masks.get(0, np.zeros((1, 1), np.uint8)) > 0).sum()),
               blue_px=int((masks.get(1, np.zeros((1, 1), np.uint8)) > 0).sum()))
    for c, tag in ((0, "red"), (1, "blue")):
        if c in masks:
            vis[masks[c] > 0] = (0, 0, 255) if tag == "red" else (255, 0, 0)
            row[tag] = comps(masks[c], tag, vis)
        else:
            row[tag] = []
    cv2.imwrite(f"{OUTD}/p{i:02d}_{int(target/FPS)}s.png", vis)
    rep.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
cap.release()
json.dump(rep, open(f"{OUTD}/probe.json", "w"), ensure_ascii=False, indent=1)
print(f"DONE el={time.time()-t0:.0f}s", flush=True)
