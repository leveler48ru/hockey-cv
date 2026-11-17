#!/usr/bin/env python3
"""Player crop extractor + team pseudolabels (path 3: ReID).
For each class=7 (player) with a v8 mask:
  - torso = central 55% of box height AND mask>0.3
  - pseudolabel: red (H<10|H>170, S>90, V>60), dark (V<95, S<130), white (V>170, S<60), else unknown
  - the crop (raw box, without mask) goes to /root/reid/crops/<pseudo>/, metadata to jsonl
Reports balance and separability (silhouette over HSV on mask-only pixels)."""
import cv2, os, json, glob, random
import numpy as np
from ultralytics import YOLO

ROOT = '/work/dataset_ref_v8/images'
OUT = '/work/reid'
os.makedirs(OUT + '/crops', exist_ok=True)
CONF, MINH = 0.30, 28

def pseudo(hsv_px):
    import numpy as np
    Hh = hsv_px[:, 0].astype(np.float64)
    Ss = hsv_px[:, 1].astype(np.float64)
    Vv = hsv_px[:, 2].astype(np.float64)
    # circular mean of hue (degrees 0..360, OpenCV H = 0..179 -> *2)
    ang = np.deg2rad(Hh * 2.0)
    hmean = (np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360) / 2.0
    smean, vmean = Ss.mean(), Vv.mean()
    red_frac = (((Hh < 10) | (Hh > 165)) & (Ss > 70) & (Vv > 50)).mean()
    if red_frac > 0.25:
        return 'red', (hmean, smean, vmean)
    if vmean < 100 and smean < 130:
        return 'dark', (hmean, smean, vmean)
    if vmean > 160 and smean < 70:
        return 'white', (hmean, smean, vmean)
    return 'unknown', (hmean, smean, vmean)

model = YOLO('/runs/rink_ref_v8/weights/best.pt')
recs, feats = [], []
for split in ('train', 'val'):
    files = sorted(glob.glob(f'{ROOT}/{split}/*.jpg'))
    for i, f in enumerate(files):
        img = cv2.imread(f)
        if img is None:
            continue
        r = model.predict(img, imgsz=1088, conf=CONF, verbose=False)[0]
        if r.boxes is None or not len(r.boxes):
            continue
        xy = r.boxes.xyxy.cpu().numpy(); cls = r.boxes.cls.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
        masks = r.masks.data.cpu().numpy() if (r.masks is not None and len(r.masks.data)) else None
        hs, ws = img.shape[:2]
        for j, (b, c, conf) in enumerate(zip(xy, cls, cf)):
            if int(c) != 7:
                continue
            x1, y1, x2, y2 = [int(v) for v in b]
            if y2 - y1 < MINH:
                continue
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(ws, x2); y2 = min(hs, y2)
            # mask in frame coordinates
            m = None
            if masks is not None and j < len(masks):
                m = cv2.resize(masks[j], (ws, hs)) > 0.3
                m[:y1, :] = False; m[y2:, :] = False; m[:, :x1] = False; m[:, x2:] = False
            ty1 = y1 + int((y2 - y1) * 0.22); ty2 = y1 + int((y2 - y1) * 0.78)
            crop_box = img[y1:y2, x1:x2]
            reg = img[ty1:ty2, x1:x2]
            regm = None
            if m is not None:
                regm = m[ty1:ty2, x1:x2]
                if regm.sum() < 40:  # mask almost empty — take the whole region
                    regm = None
            pix_hsv = cv2.cvtColor(reg, cv2.COLOR_BGR2HSV)
            if regm is not None:
                px = pix_hsv[regm]
                px_box = None
            else:
                px = pix_hsv.reshape(-1, 3)
            lab, stat = pseudo(px)
            red_f = float((((px[:, 0] < 10) | (px[:, 0] > 165)) & (px[:, 1] > 70) & (px[:, 2] > 50)).mean())
            if regm is not None:
                feats.append([stat[0], stat[1], stat[2]])
            stem = os.path.basename(f)[:-4]
            d = OUT + '/crops/' + lab
            os.makedirs(d, exist_ok=True)
            name = f'{split}_{stem}_{j}.jpg'
            cv2.imwrite(f'{d}/{name}', crop_box)
            recs.append(dict(split=split, src=stem, idx=j, conf=round(float(conf), 3),
                             box=[x1, y1, x2, y2], pseudo=lab, hsv=[round(v, 1) for v in stat],
                             red_frac=round(red_f, 3),
                             mask_px=int(regm.sum()) if regm is not None else 0, crop=name))
        if i % 400 == 0:
            print(f'{split} {i}/{len(files)} crops={len(recs)}', flush=True)

json.dump(recs, open(OUT + '/meta.json', 'w'))
bal = {}
for rr in recs:
    bal[rr['pseudo']] = bal.get(rr['pseudo'], 0) + 1
print('BALANCE', json.dumps(bal))
F = np.array(feats)
print('n_feats', len(F))
if len(F) > 50:
    from sklearn.cluster import KMeans
    import sklearn.metrics as sm
    km = KMeans(2, n_init=10, random_state=0).fit(F)
    print('silhouette_2cl', round(sm.silhouette_score(F, km.labels_), 3))
    print('centers_HSV', km.cluster_centers_.round(1).tolist())
print('DONE')
