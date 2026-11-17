#!/usr/bin/env python3
"""Run v4: YOLO masks -> per-frame DLT from lines -> mosaic. NO ORB.

Idea: the model sees the red/blue lines as masks. We thin the mask and fit a
RANSAC line in pixels. The world line is known (x=const for vertical ones).
Correspondence: sample N points along the world line, for each find the nearest
point on the pixel line -> pairs (pix, world). DLT/RANSAC -> H_pix2world.

This is more robust than ORB on compressed video: independent of the background,
depends only on the markings.
"""
import cv2, numpy as np, os, json, time, sys
from ultralytics import YOLO

VID = sys.argv[1] if len(sys.argv) > 1 else 'match.mp4'
OUT = '/root/full_run_v4'
STEP = 3.0
PPM = 20
MARGIN = 1.0
W_M, H_M = 60.0, 30.0
GW, GH = int((W_M + 2 * MARGIN) * PPM), int((H_M + 2 * MARGIN) * PPM)
os.makedirs(OUT, exist_ok=True)

CLASSES = ['red_line', 'blue_line', 'center_circle', 'faceoff_spot',
           'faceoff_circle', 'goal_crease', 'board']

# World vertical lines: (class_name, x_world)
VERT_LINES = [
    ('red_line', 30.0),   # center line
    ('blue_line', 22.4),  # left blue line
    ('blue_line', 37.6),  # right blue line
    ('red_line', 4.0),    # left goal line
    ('red_line', 56.0),   # right goal line
]


def skeleton_and_line(mask_bin):
    """Mask thinning + RANSAC line. Returns (a,b,c) or None."""
    skel = cv2.ximgproc.thinning(mask_bin) if hasattr(cv2, 'ximgproc') else mask_bin.copy()
    ys, xs = np.where(skel > 0)
    if len(xs) < 40:
        return None
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    best_inl = []; best_line = None
    for _ in range(300):
        idx = np.random.choice(len(pts), 2, replace=False)
        p1, p2 = pts[idx]
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        if abs(dx) < 0.5 and abs(dy) < 0.5: continue
        a, b = dy, -dx; c = -(a*p1[0]+b*p1[1])
        n = np.hypot(a, b)
        if n < 1e-6: continue
        a, b, c = a/n, b/n, c/n
        d = np.abs(a*pts[:,0]+b*pts[:,1]+c)
        inl = np.where(d < 6.0)[0]
        if len(inl) > len(best_inl):
            best_inl = inl; best_line = (a, b, c)
    if best_line is None or len(best_inl) < 40:
        return None
    return best_line


def point_on_line(line, y):
    """Point on line ax+by+c=0 for a given y."""
    a, b, c = line
    if abs(a) < 1e-8:
        return None
    x = -(b*y + c) / a
    return (x, y)


def fit_H_from_masks(frame_masks, img_shape):
    """Fit H_pix2world from correspondences of pixel and world lines."""
    h, w = img_shape[:2]
    pix_pts = []
    world_pts = []

    for cls, wx in VERT_LINES:
        if cls not in frame_masks:
            continue
        line = skeleton_and_line(frame_masks[cls])
        if line is None:
            continue
        # Sample points along the world line (y from 0 to 30)
        for wy in np.linspace(2, 28, 15):
            pp = point_on_line(line, None)  # need to find a point on the pixel line
            # For a vertical world line x=wx the pixel line is also ~vertical.
            # Take points on the pixel line with y from 0 to h, match them to world y values.
            pass

    # Alternative approach: for each world point (wx, wy) find the nearest point
    # on the pixel line. But the pixel line is in frame coordinates and the world one
    # is in meters. An initial H estimate is needed for the transformation.

    # Practical approach: use mask centroids as rough correspondences,
    # then refine via ICP.
    centers_pix = []
    centers_world = []
    for cls, wx in VERT_LINES:
        if cls not in frame_masks:
            continue
        mask = frame_masks[cls]
        ys, xs = np.where(mask > 0)
        if len(xs) < 50:
            continue
        cx, cy = float(xs.mean()), float(ys.mean())
        # World point: middle of the line along y
        wmy = 15.0  # rink center along the width
        centers_pix.append([cx, cy])
        centers_world.append([wx, wmy])

    if len(centers_pix) < 4:
        return None

    pp = np.array(centers_pix, np.float32)
    pw = np.array(centers_world, np.float32)

    # Centroids are rough — use affine instead of a full homography
    # (4 points suffice for affine, 8 parameters vs 9 for homography)
    M_affine, status = cv2.estimateAffinePartial2D(pp, pw, method=cv2.RANSAC,
                                                      ransacReprojThreshold=50,
                                                      maxIters=10000)
    if M_affine is None:
        return None

    # Convert the 2x3 affine into a 3x3 homography
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = M_affine
    return H


def validate_H(H, frame_masks, img_shape):
    """Validate H: project world lines into pixels, compute overlap with masks."""
    h, w = img_shape[:2]
    H_inv = np.linalg.inv(H)
    tot = hit = 0
    for cls, wx in VERT_LINES:
        if cls not in frame_masks:
            continue
        pts = []
        for wy in np.linspace(0, 30, 40):
            p = H_inv @ np.array([wx, wy, 1.0])
            px, py = p[0]/p[2], p[1]/p[2]
            if 0 <= px < w and 0 <= py < h:
                pts.append((int(px), int(py)))
        if len(pts) < 5:
            continue
        check = np.zeros((h, w), np.uint8)
        cv2.polylines(check, [np.array(pts)], False, 255, 8)
        tot += int((check > 0).sum())
        hit += int(((check > 0) & (frame_masks[cls] > 0)).sum())
    return hit / tot if tot > 0 else 0


def main():
    model = YOLO('/root/runs/rink_real-3/weights/best.pt')
    cap = cv2.VideoCapture(VID)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_frames = max(1, int(round(STEP * fps)))
    print(f'{VID}: {total} frames, dur {total/fps/60:.1f} min, step {step_frames}', flush=True)

    evidence = np.zeros((GH, GW), np.float32)
    photo = np.zeros((GH, GW, 3), np.float32)
    weight = np.zeros((GH, GW), np.float32)
    S = np.array([[PPM, 0, MARGIN * PPM], [0, PPM, MARGIN * PPM], [0, 0, 1]], np.float64)
    recs = []
    t0 = time.time()
    k = idx = -1

    while True:
        if not cap.grab(): break
        idx += 1
        if idx % step_frames: continue
        ok, fr = cap.retrieve()
        if not ok: continue
        k += 1
        ts = idx / fps

        r = model.predict(fr, conf=0.30, device=0, verbose=False)[0]
        names = [CLASSES[int(c)] for c in r.boxes.cls.tolist()] if len(r.boxes) else []

        frame_masks = {}
        if r.masks is not None:
            for i, cls_idx in enumerate(r.boxes.cls.tolist()):
                cname = CLASSES[int(cls_idx)]
                mm = cv2.resize(r.masks.data[i].cpu().numpy(), (fr.shape[1], fr.shape[0]))
                bin_mask = (mm > 0.5).astype(np.uint8) * 255
                if cname in frame_masks:
                    frame_masks[cname] = cv2.bitwise_or(frame_masks[cname], bin_mask)
                else:
                    frame_masks[cname] = bin_mask

        n_lines = sum(1 for c in ('red_line', 'blue_line') if c in frame_masks)
        if n_lines < 2:
            recs.append({'ts': round(ts, 1), 'stage': 'few_lines', 'n': len(names)})
            continue

        H = fit_H_from_masks(frame_masks, fr.shape)
        if H is None:
            recs.append({'ts': round(ts, 1), 'stage': 'no_H', 'n': len(names)})
            continue

        match = validate_H(H, frame_masks, fr.shape)
        if match < 0.25:
            recs.append({'ts': round(ts, 1), 'stage': 'low_match', 'match': round(match, 3)})
            continue

        # Accumulation
        use_mask = np.zeros(fr.shape[:2], np.uint8)
        for cname in ('red_line', 'blue_line', 'center_circle', 'faceoff_circle', 'goal_crease'):
            if cname in frame_masks:
                use_mask = cv2.bitwise_or(use_mask, frame_masks[cname])

        G = S @ np.linalg.inv(H)
        ev = cv2.warpPerspective(use_mask, G, (GW, GH), flags=cv2.INTER_NEAREST)
        evidence += (ev > 0).astype(np.float32)

        ph = cv2.warpPerspective(fr, G, (GW, GH), flags=cv2.INTER_LINEAR)
        wm = cv2.warpPerspective(np.full(fr.shape[:2], 255, np.uint8), G, (GW, GH), flags=cv2.INTER_NEAREST)
        wmf = (wm > 0).astype(np.float32)
        photo += ph * wmf[..., None]
        weight += wmf

        recs.append({'ts': round(ts, 1), 'stage': 'good', 'match': round(match, 3), 'n': len(names)})

        if k % 50 == 0:
            el = time.time() - t0
            done = idx / max(1, total)
            eta = el / max(0.01, done) * (1 - done) / 60
            ng = sum(1 for x in recs if x['stage'] == 'good')
            print(f'  {idx}/{total} ({done*100:.0f}%) steps={k} good={ng} ETA {eta:.0f}min', flush=True)

    cap.release()
    n_good = sum(1 for x in recs if x['stage'] == 'good')
    from collections import Counter
    stages = Counter(x['stage'] for x in recs)
    matches = [x['match'] for x in recs if x['stage'] == 'good']
    print(f'stages: {dict(stages)}, good={n_good}, median match={np.median(matches) if matches else 0:.2f}', flush=True)

    ev_vis = np.clip(evidence / max(1, n_good // 4), 0, 1)
    cv2.imwrite(f'{OUT}/evidence.png', (ev_vis * 255).astype(np.uint8))
    w = np.clip(weight, 0, None)
    img = np.nan_to_num(photo / w[..., None], 0).astype(np.uint8)
    img[w[..., 0] == 0] = 0
    cv2.imwrite(f'{OUT}/world_photo.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    overlay = img.copy()
    thr = max(3, n_good * 0.08)
    overlay[evidence > thr] = (0, 0, 240)
    cv2.imwrite(f'{OUT}/world_overlay.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(f'{OUT}/frames.json', 'w') as fp:
        json.dump(recs, fp)
    cov = float((weight > 0).mean())
    print(f'coverage: {cov*100:.1f}%, elapsed {(time.time()-t0)/60:.1f} min', flush=True)

if __name__ == '__main__':
    main()
