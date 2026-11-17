#!/usr/bin/env python3
"""Propagate calibration from the anchor frame match_00150 to all wide frames.
The camera is on a tripod, pure rotation -> inter-frame homography transfer via ORB.
H_frame(pix->world) = H_anchor @ M^-1, M: anchor_pix->frame_pix.
For each frame: project IIHF geometry -> polygons + QC (inliers, agreement with colour masks).
"""
import cv2, numpy as np, glob, os, json, time

BASE = '.'
ANCHOR = f'{BASE}/wide/match_00150.jpg'
H_A = np.load(f'{BASE}/H_pix2world.npy')          # anchor pix -> world
OUT_MASKS = f'{BASE}/real_masks'
OUT_QC = f'{BASE}/qc'
os.makedirs(OUT_MASKS, exist_ok=True)
os.makedirs(OUT_QC, exist_ok=True)

SB = {'match': (40, 25, 650, 160), 'm1': (55, 30, 385, 115), 'm2': (50, 25, 720, 175)}

# ---------- IIHF geometry 60x30 ----------
def strip_poly(ax, ay, bx, by, hw, n=24):
    dx, dy = bx-ax, by-ay
    L = np.hypot(dx, dy)
    nx, ny = -dy/L, dx/L
    pts = []
    for i in range(n+1):
        t = i/n
        pts.append((ax+dx*t+nx*hw, ay+dy*t+ny*hw))
    for i in range(n, -1, -1):
        t = i/n
        pts.append((ax+dx*t-nx*hw, ay+dy*t-ny*hw))
    return np.array(pts, np.float32)

def ring_poly(cx, cy, r, w, n=72):
    a = np.linspace(0, 2*np.pi, n, endpoint=False)
    outer = np.stack([cx+r*np.cos(a), cy+r*np.sin(a)], 1)
    inner = np.stack([cx+(r-w)*np.cos(a[::-1]), cy+(r-w)*np.sin(a[::-1])], 1)
    return np.concatenate([outer, inner]).astype(np.float32)

def disc_poly(cx, cy, r, n=24):
    a = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.stack([cx+r*np.cos(a), cy+r*np.sin(a)], 1).astype(np.float32)

def crease_poly(gx, sign):
    a = np.linspace(-np.pi/2, np.pi/2, 25)
    r1, r2 = 1.9, 1.8
    pts = [(gx+sign*r1*np.cos(ai), 15+r1*np.sin(ai)) for ai in a]
    pts += [(gx+sign*r2*np.cos(ai), 15+r2*np.sin(ai)) for ai in a[::-1]]
    return np.array(pts, np.float32)

GEOM = []
GEOM.append(('red_line', strip_poly(30, 0, 30, 30, 0.3)))
GEOM.append(('red_line', strip_poly(4, 0, 4, 30, 0.15)))
GEOM.append(('red_line', strip_poly(56, 0, 56, 30, 0.15)))
GEOM.append(('blue_line', strip_poly(22.4, 0, 22.4, 30, 0.15)))
GEOM.append(('blue_line', strip_poly(37.6, 0, 37.6, 30, 0.15)))
GEOM.append(('center_circle', ring_poly(30, 15, 4.5, 0.15)))
for dx, dy in [(24, 6), (36, 6), (24, 24), (36, 24)]:
    GEOM.append(('faceoff_spot', disc_poly(dx, dy, 0.35)))
for cx, cy in [(12, 6), (12, 24), (48, 6), (48, 24)]:
    GEOM.append(('faceoff_circle', ring_poly(cx, cy, 4.5, 0.15)))
    GEOM.append(('faceoff_spot', disc_poly(cx, cy, 0.35)))
GEOM.append(('goal_crease', crease_poly(4, +1)))
GEOM.append(('goal_crease', crease_poly(56, -1)))
GEOM.append(('board', strip_poly(0, 0, 60, 0, 0.25)))
GEOM.append(('board', strip_poly(0, 30, 60, 30, 0.25)))
GEOM.append(('board', strip_poly(0, 0, 0, 30, 0.25)))
GEOM.append(('board', strip_poly(60, 0, 60, 30, 0.25)))

CLASSES = ['red_line', 'blue_line', 'center_circle', 'faceoff_spot',
           'faceoff_circle', 'goal_crease', 'board']
CID = {c: i for i, c in enumerate(CLASSES)}

def project(H_w2p, poly):
    ph = np.hstack([poly, np.ones((len(poly), 1), np.float32)])
    pp = (H_w2p @ ph.T).T
    return (pp[:, :2] / pp[:, 2:3]).astype(np.int32)

# ---------- anchor ----------
anchor = cv2.imread(ANCHOR)
orb = cv2.ORB_create(nfeatures=2500)
akp, ades = orb.detectAndCompute(anchor, None)
flann = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
print(f'anchor: {len(akp)} keypoints', flush=True)

files = sorted(glob.glob(f'{BASE}/wide/*.jpg'))
t0 = time.time()
rows = []
for i, f in enumerate(files):
    base = os.path.splitext(os.path.basename(f))[0]
    vid = base.split('_')[0]
    img = cv2.imread(f)
    h, w = img.shape[:2]
    m = np.ones((h, w), np.uint8)
    sx1, sy1, sx2, sy2 = SB[vid]
    m[sy1:sy2, sx1:sx2] = 0
    kps, des = orb.detectAndCompute(img, m)
    rec = {'frame': base, 'inliers': 0, 'red_ok': -1, 'blue_ok': -1}
    ok = False
    if des is not None and len(kps) >= 50:
        matches = flann.knnMatch(ades, des, k=2)
        good = [a for a, b in matches if a.distance < 0.75*b.distance]
        if len(good) >= 30:
            pa = np.float32([akp[a.queryIdx].pt for a in good])
            pb = np.float32([kps[a.trainIdx].pt for a in good])
            M, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
            if M is not None:
                nin = int(mask.sum())
                rec['inliers'] = nin
                if nin >= 60:
                    H_fpw = H_A @ np.linalg.inv(M)
                    H_fwp = np.linalg.inv(H_fpw)
                    polys, labels = [], []
                    for cname, wp in GEOM:
                        pp = project(H_fwp, wp)
                        if pp[:, 0].max() < -50 or pp[:, 0].min() > w+50 or \
                           pp[:, 1].max() < -50 or pp[:, 1].min() > h+50:
                            continue
                        if np.abs(pp).max() > 1e6:
                            continue
                        polys.append(pp); labels.append(CID[cname])
                    if polys:
                        np.savez_compressed(f'{OUT_MASKS}/{base}.npz',
                                            polys=np.concatenate(polys),
                                            cls=np.array(labels, np.int32),
                                            counts=np.array([len(p) for p in polys], np.int32),
                                            H=H_fpw)
                        ok = True
                        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                        red = cv2.inRange(hsv, (0, 90, 90), (10, 255, 255)) | \
                              cv2.inRange(hsv, (170, 90, 90), (180, 255, 255))
                        blue = cv2.inRange(hsv, (100, 70, 70), (130, 255, 255))
                        red[sy1:sy2, sx1:sx2] = 0; blue[sy1:sy2, sx1:sx2] = 0
                        def line_hit(maskch, cname, hw_px=6):
                            tot = hit = 0
                            for pp, lb in zip(polys, labels):
                                if CLASSES[lb] != cname: continue
                                mm = np.zeros((h, w), np.uint8)
                                cv2.fillPoly(mm, [pp], 255)
                                mm = cv2.dilate(mm, np.ones((hw_px*2+1,)*2, np.uint8))
                                tot += int((mm > 0).sum())
                                hit += int(((mm > 0) & (maskch > 0)).sum())
                            return (hit/tot) if tot > 0 else -1
                        rec['red_ok'] = round(line_hit(red, 'red_line'), 3)
                        rec['blue_ok'] = round(line_hit(blue, 'blue_line'), 3)
    rows.append(rec)
    if (i+1) % 200 == 0:
        print(f'  {i+1}/{len(files)} in {(time.time()-t0)/60:.1f} min', flush=True)

with open(f'{OUT_QC}/propagate.json', 'w') as fp:
    json.dump(rows, fp)
good = sum(1 for r in rows if r['inliers'] >= 60)
print(f'DONE: {len(rows)} frames, anchored {good}, elapsed {(time.time()-t0)/60:.1f} min', flush=True)