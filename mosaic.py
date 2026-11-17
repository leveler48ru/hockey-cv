#!/usr/bin/env python3
"""Build the 60x30 rink mosaic from already calibrated frames (real_masks/*.npz).

Each npz contains:
  polys/cls/counts - IIHF geometry polygons in frame pixels (visually verified)
  H                - pix->world for this frame (transferred from the anchor via ORB)
Accumulation:
  evidence  - how many times a world point was 'under' a marking (from line polygons)
  photo     - weighted-average photograph of the ice (frame warped into the world grid)
  team      - jersey colour mapping (for subsequent team detection)
Usability: frames from propagate.json with inliers>=60.
"""
import cv2, numpy as np, glob, os, json

BASE = '.'
MASKS = f'{BASE}/real_masks'
WIDE = f'{BASE}/wide'
OUT = f'{BASE}/mosaic'
os.makedirs(OUT, exist_ok=True)

PPM = 24
MARGIN = 1.0
W_M, H_M = 60.0, 30.0
GW, GH = int((W_M + 2 * MARGIN) * PPM), int((H_M + 2 * MARGIN) * PPM)
S = np.array([[PPM, 0, MARGIN * PPM], [0, PPM, MARGIN * PPM], [0, 0, 1]], np.float64)

CLASSES = ['red_line', 'blue_line', 'center_circle', 'faceoff_spot',
           'faceoff_circle', 'goal_crease', 'board']
LINE_CLS = {0, 1, 2, 4, 5}          # marking lines/circles (not boards, not dots)

qc = {r['frame']: r for r in json.load(open(f'{BASE}/qc/propagate.json'))}
files = sorted(glob.glob(f'{MASKS}/*.npz'))

evidence = np.zeros((GH, GW), np.float32)
photo = np.zeros((GH, GW, 3), np.float32)
weight = np.zeros((GH, GW), np.float32)
icesk = np.zeros((GH, GW), np.float32)   # visible-ice mask (for coverage)

n_used = 0
for fpath in files:
    base = os.path.splitext(os.path.basename(fpath))[0]
    rec = qc.get(base)
    if not rec or rec['inliers'] < 60:
        continue
    img = cv2.imread(f'{WIDE}/{base}.jpg')
    if img is None:
        continue
    d = np.load(fpath)
    H = d['H']                        # pix -> world
    G = S @ H                         # pix -> grid (world->grid = S)
    h, w = img.shape[:2]

    # rasterize the marking polygons into a frame mask
    mk = np.zeros((h, w), np.uint8)
    polys, cls, counts = d['polys'], d['cls'], d['counts']
    start = 0
    for c, cnt in zip(cls, counts):
        poly = polys[start:start + cnt]
        start += cnt
        if int(c) in LINE_CLS:
            cv2.fillPoly(mk, [poly.astype(np.int32)], 255)

    ev = cv2.warpPerspective(mk, G, (GW, GH), flags=cv2.INTER_NEAREST)
    ph = cv2.warpPerspective(img, G, (GW, GH), flags=cv2.INTER_LINEAR)
    wm = cv2.warpPerspective(np.full((h, w), 255, np.uint8), G, (GW, GH),
                             flags=cv2.INTER_NEAREST)
    wmf = (wm > 0).astype(np.float32)
    evidence += (ev > 0).astype(np.float32)
    photo += ph * wmf[..., None]
    weight += wmf
    n_used += 1

print(f'frames used: {n_used}')

w = np.clip(weight, 0, None)
img = np.nan_to_num(photo / w[..., None], 0).astype(np.uint8)
img[w == 0] = 0
cv2.imwrite(f'{OUT}/world_photo.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])

# proven markings in red over the photo
overlay = img.copy()
thr = max(2, n_used * 0.06)
mask_vis = evidence > thr
overlay[mask_vis] = (0, 0, 240)
cv2.imwrite(f'{OUT}/world_overlay.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])

# evidence heatmap
ev_vis = np.clip(evidence / max(1.0, n_used * 0.25), 0, 1)
heat = cv2.applyColorMap((ev_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
heat[ev_vis == 0] = 0
cv2.imwrite(f'{OUT}/world_evidence.jpg', heat, [cv2.IMWRITE_JPEG_QUALITY, 92])

cov = float((weight > 0).mean())
print(f'world grid coverage: {cov*100:.1f}%')
print(f'evidence threshold: {thr:.1f} frames, pixel max={evidence.max():.0f}')
print('saved:', OUT)