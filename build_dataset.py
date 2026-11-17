#!/usr/bin/env python3
"""Building the YOLO-seg dataset from real_masks/*.npz.
IIHF geometry polygons projected onto frames are CLIPPED to the frame
(Sutherland-Hodgman), coordinates normalized to [0,1].
7 classes: red_line blue_line center_circle faceoff_spot faceoff_circle goal_crease board
"""
import numpy as np, glob, os, shutil, random, cv2

BASE = '.'
MASKS = f'{BASE}/real_masks'
WIDE = f'{BASE}/wide'
DS = f'{BASE}/dataset_real'
CLASSES = ['red_line', 'blue_line', 'center_circle', 'faceoff_spot',
           'faceoff_circle', 'goal_crease', 'board']
MIN_AREA = 12          # px^2 — tiny fragments are not needed
MIN_SIDE = 2           # px — minimum extent along any axis

for sub in ('images/train', 'images/val', 'labels/train', 'labels/val'):
    os.makedirs(f'{DS}/{sub}', exist_ok=True)


def clip_poly(subject, W, H):
    """Sutherland-Hodgman: intersection of a polygon with the rectangle [0,W]x[0,H]."""
    rect = [(0, 0), (W, 0), (W, H), (0, H)]
    out = list(subject)
    for i in range(4):
        if not out:
            return []
        ax, ay = rect[i]
        bx, by = rect[(i + 1) % 4]
        # inner side of the edge: for a clockwise traversal it's the right
        ex, ey = bx - ax, by - ay
        inp = out
        out = []
        def inside(p):
            return ex * (p[1] - ay) - ey * (p[0] - ax) >= 0
        def intersect(p, q):
            px, py = p; qx, qy = q
            r1x, r1y = qx - px, qy - py
            denom = ex * r1y - ey * r1x
            if abs(denom) < 1e-9:
                return p
            t = (ex * (py - ay) - ey * (px - ax)) / denom
            return (px + t * r1x, py + t * r1y)
        for j in range(len(inp)):
            cur = inp[j]; prv = inp[j - 1]
            ci, pi = inside(cur), inside(prv)
            if ci:
                if not pi:
                    out.append(intersect(prv, cur))
                out.append(cur)
            elif pi:
                out.append(intersect(prv, cur))
    return out


def poly_to_yolo(pts):
    return ' '.join(f'{x:.5f} {y:.5f}' for x, y in pts)


files = sorted(glob.glob(f'{MASKS}/*.npz'))
random.seed(42)
random.shuffle(files)
n_val = max(20, len(files) // 8)
val_set = set(os.path.splitext(os.path.basename(f))[0] for f in files[:n_val])

counts = {c: 0 for c in CLASSES}
n_img = n_skip_empty = 0
for f in files:
    base = os.path.splitext(os.path.basename(f))[0]
    split = 'val' if base in val_set else 'train'
    src_img = f'{WIDE}/{base}.jpg'
    if not os.path.exists(src_img):
        continue
    im = cv2.imread(src_img)
    if im is None:
        continue
    H, W = im.shape[:2]

    d = np.load(f)
    polys, cls, counts_arr = d['polys'], d['cls'], d['counts']
    lines = []
    start = 0
    for c, cnt in zip(cls, counts_arr):
        poly = polys[start:start + cnt].astype(np.float64)
        start += cnt
        # drop points beyond the camera plane (wild coordinates)
        if np.abs(poly).max() > 1e5:
            continue
        clipped = clip_poly(poly, W, H)
        if len(clipped) < 3:
            continue
        arr = np.array(clipped)
        area = cv2.contourArea(arr.astype(np.float32))
        if area < MIN_AREA:
            continue
        if arr[:, 0].max() - arr[:, 0].min() < MIN_SIDE or arr[:, 1].max() - arr[:, 1].min() < MIN_SIDE:
            continue
        norm = arr / np.array([W, H], np.float64)
        norm = np.clip(norm, 0.0, 1.0)
        lines.append(f'{c} ' + poly_to_yolo(norm))
        counts[CLASSES[c]] += 1

    if not lines:
        n_skip_empty += 1
        continue
    shutil.copy(src_img, f'{DS}/images/{split}/{base}.jpg')
    with open(f'{DS}/labels/{split}/{base}.txt', 'w') as fp:
        fp.write('\n'.join(lines) + '\n')
    n_img += 1

with open(f'{DS}/data.yaml', 'w') as fp:
    fp.write(f"""path: {DS}
train: images/train
val: images/val
nc: {len(CLASSES)}
names: {CLASSES}
""")

print(f'images: {n_img} (val={len(val_set)}, empty after clipping: {n_skip_empty})')
for c, n in counts.items():
    print(f'  {c:15s} {n}')
