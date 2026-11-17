#!/usr/bin/env python3
"""Chain propagation of calibration over jpg frames (wide/*.jpg).

Start: 533 good frames from qc/propagate.json (each with its own H in real_masks/*.npz).
Iteration: for each UNcalibrated frame, match ORB against the N temporally nearest
calibrated frames of the same video; the best by inliers (>=60) yields H =
H_seed @ inv(M). A new good frame is added to the pool and becomes available to
subsequent iterations. This way endzone frames, far from the single anchor, inherit
the calibration "by descent".
"""
import cv2, numpy as np, glob, os, json, time

BASE = '.'
WIDE = f'{BASE}/wide'
OUT_MASKS = f'{BASE}/real_masks'
SB = {'match': (40, 25, 650, 160), 'm1': (55, 30, 385, 115), 'm2': (50, 25, 720, 175)}
MIN_INL = 60
NEAR = 6          # how many nearest seed frames to try

# geometry — same as in propagate.py
def strip_poly(ax, ay, bx, by, hw, n=24):
    dx, dy = bx-ax, by-ay
    L = np.hypot(dx, dy); nx, ny = -dy/L, dx/L
    pts = [(ax+dx*t+nx*hw, ay+dy*t+ny*hw) for t in (i/n for i in range(n+1))]
    pts += [(ax+dx*t-nx*hw, ay+dy*t-ny*hw) for t in (i/n for i in range(n, -1, -1))]
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

def save_npz(base, img_shape, H_fpw):
    h, w = img_shape[:2]
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
    if not polys:
        return False
    np.savez_compressed(f'{OUT_MASKS}/{base}.npz',
                        polys=np.concatenate(polys),
                        cls=np.array(labels, np.int32),
                        counts=np.array([len(p) for p in polys], np.int32),
                        H=H_fpw)
    return True

orb = cv2.ORB_create(nfeatures=2500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# --- pool of seed frames ---
rows = json.load(open(f'{BASE}/qc/propagate.json'))
good = {r['frame']: r for r in rows if r.get('inliers', 0) >= 60}
by_vid = {}
for b in good:
    vid, t = b.split('_')
    by_vid.setdefault(vid, {})[int(t)] = b
print(f'seed good: {len(good)}', flush=True)

# descriptor cache for seed frames (LRU-like: load on demand)
feat_cache = {}
def feats(base):
    if base in feat_cache:
        return feat_cache[base]
    img = cv2.imread(f'{WIDE}/{base}.jpg')
    if img is None:
        return None
    vid = base.split('_')[0]
    m = np.ones((img.shape[0], img.shape[1]), np.uint8)
    sx1, sy1, sx2, sy2 = SB[vid]
    m[sy1:sy2, sx1:sx2] = 0
    kps, des = orb.detectAndCompute(img, m)
    if des is None:
        return None
    feat_cache[base] = (kps, des)
    if len(feat_cache) > 150:
        feat_cache.pop(next(iter(feat_cache)))
    return feat_cache[base]

all_files = sorted(glob.glob(f'{WIDE}/*.jpg'))
new_good = {}
t0 = time.time()
for i, f in enumerate(all_files):
    base = os.path.splitext(os.path.basename(f))[0]
    if base in good or base in new_good:
        continue
    vid, t = base.split('_')
    t = int(t)
    seeds = by_vid.get(vid, {})
    if not seeds:
        continue
    # nearest seeds by time
    cand = sorted(seeds.keys(), key=lambda s: abs(s - t))[:NEAR]
    img = cv2.imread(f)
    if img is None:
        continue
    m = np.ones((img.shape[0], img.shape[1]), np.uint8)
    sx1, sy1, sx2, sy2 = SB[vid]
    m[sy1:sy2, sx1:sx2] = 0
    kps, des = orb.detectAndCompute(img, m)
    if des is None or len(kps) < 40:
        continue
    best = None
    for s_t in cand:
        s_base = seeds[s_t] if isinstance(list(seeds.values())[0], str) else s_t
        sf = feats(s_base)
        if sf is None:
            continue
        skp, sdes = sf
        matches = bf.knnMatch(sdes, des, k=2)
        goodm = [a for a, b in matches if a.distance < 0.75 * b.distance]
        if len(goodm) < MIN_INL:
            continue
        pa = np.float32([skp[a.queryIdx].pt for a in goodm])
        pb = np.float32([kps[a.trainIdx].pt for a in goodm])
        M, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
        if M is None:
            continue
        nin = int(mask.sum())
        if nin >= MIN_INL and (best is None or nin > best[0]):
            best = (nin, M, s_base)
    if best is None:
        continue
    nin, M, s_base = best
    seed_H = np.load(f'{OUT_MASKS}/{s_base}.npz')['H']
    H_fpw = seed_H @ np.linalg.inv(M)
    if save_npz(base, img.shape, H_fpw):
        new_good[base] = {'seed': s_base, 'inliers': nin}
        by_vid.setdefault(vid, {})[t] = base
    if (i+1) % 200 == 0:
        print(f'  {i+1}/{len(all_files)} new: {len(new_good)} '
              f'({(time.time()-t0)/60:.1f} min)', flush=True)

print(f'TOTAL new good: {len(new_good)}, overall: {len(good)+len(new_good)}', flush=True)
with open(f'{BASE}/qc/chain_new.json', 'w') as fp:
    json.dump(new_good, fp)
