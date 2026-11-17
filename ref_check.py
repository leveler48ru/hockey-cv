#!/usr/bin/env python3
"""Who sits on the referee-GT boxes: measuring which v5 class referees land in.
For each class-8 GT box in val — the best prediction (IoU>0.3).
Plus a check of the stripes verifier: vertical edge density inside the box."""
import json, glob, os
import cv2, numpy as np
from ultralytics import YOLO

model = YOLO('/runs/rink_referees_v5/weights/best.pt')
DS = '/data'  # dataset_real_v5

def gt_boxes(txt):
    W, H = 1280, 720
    out = []
    for line in open(txt):
        p = line.split()
        if not p or p[0] != '8':
            continue
        xs = [float(p[i]) for i in range(1, 9, 2)]
        ys = [float(p[i]) for i in range(2, 9, 2)]
        out.append((min(xs) * W, min(ys) * H, max(xs) * W, max(ys) * H))
    return out

def stripe_score(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 - x1 < 8 or y2 - y1 < 15:
        return None
    crop = frame[y1:y2, x1:x2]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    return float(np.abs(edges).mean())

tally = {'referee': 0, 'player': 0, 'other': 0, 'missed': 0}
rows = []
for txt in sorted(glob.glob(f'{DS}/labels/val/*.txt')):
    name = os.path.basename(txt)[:-4]
    gts = gt_boxes(txt)
    if not gts:
        continue
    img = cv2.imread(f'{DS}/images/val/{name}.jpg')
    r = model.predict(img, imgsz=1088, device='cpu', verbose=False, conf=0.15)[0]
    preds = r.boxes.data.cpu().numpy() if len(r.boxes) else []
    for gt in gts:
        best, biou = None, 0.0
        gx1, gy1, gx2, gy2 = gt
        gw = (gx2 - gx1) * (gy2 - gy1)
        for pr in preds:
            x1, y1, x2, y2, conf, cls = pr
            iw = max(0, min(gx2, x2) - max(gx1, x1))
            ih = max(0, min(gy2, y2) - max(gy1, y1))
            inter = iw * ih
            union = gw + (x2 - x1) * (y2 - y1) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > biou:
                biou, best = iou, (int(cls), float(conf))
        if best is None or biou < 0.3:
            tally['missed'] += 1
            res = 'miss'
        else:
            c, conf = best
            key = {8: 'referee'}.get(c, {7: 'player'}.get(c, 'other'))
            tally[key] += 1
            res = f'{"ref" if c==8 else ("player" if c==7 else c)}:{conf:.2f}'
        ss = stripe_score(img, gt)
        rows.append((name, res, round(ss, 1) if ss is not None else None))

tot = sum(tally.values())
print(f'GT referee objects in val: {tot}')
for k, v in tally.items():
    print(f'  {k}: {v} ({100*v/max(1,tot):.0f}%)')
pl = [r for r in rows if r[1].startswith('player')]
ref = [r for r in rows if r[1].startswith('ref')]
oth = [r for r in rows if r[1].startswith('other')] + [r for r in rows if r[1] == 'miss']
def med(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.median(xs)), 1) if xs else None
print(f'stripe-score (median): correctly on referees {med([r[2] for r in ref])} | '
      f'consumed by players {med([r[2] for r in pl])} | other/missed {med([r[2] for r in oth])}')
json.dump({'tally': tally, 'rows': rows}, open('/out/ref_check.json', 'w'), ensure_ascii=False, indent=1)
print('done')
