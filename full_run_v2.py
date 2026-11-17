#!/usr/bin/env python3
"""Full run v2: YOLO masks -> per-frame H-fit from lines -> mosaic.
No ORB. The homography is fit on each frame separately from correspondences
between YOLO masks of lines (red/blue) and the known IIHF world lines.
"""
import cv2, numpy as np, os, json, time, sys
from ultralytics import YOLO

VID = sys.argv[1] if len(sys.argv) > 1 else 'match.mp4'
OUT = '/root/full_run_v2'
STEP = 3.0
PPM = 20
MARGIN = 1.0
W_M, H_M = 60.0, 30.0
GW, GH = int((W_M + 2 * MARGIN) * PPM), int((H_M + 2 * MARGIN) * PPM)
os.makedirs(OUT, exist_ok=True)

CLASSES = ['red_line', 'blue_line', 'center_circle', 'faceoff_spot',
           'faceoff_circle', 'goal_crease', 'board']

# World lines for the H fit: (class_name, x_world, y0, y1)
LINES = [
    ('red_line', 30.0, 0, 30),
    ('blue_line', 22.4, 0, 30),
    ('blue_line', 37.6, 0, 30),
    ('red_line', 4.0, 0, 30),   # goal line left
    ('red_line', 56.0, 0, 30),  # goal line right
]


def mask_centerline(mask_bin):
    """Thinning + RANSAC line through the line mask. Returns (a,b,c) or None."""
    skel = cv2.ximgproc.thinning(mask_bin) if hasattr(cv2, 'ximgproc') else mask_bin.copy()
    ys, xs = np.where(skel > 0)
    if len(xs) < 30:
        return None
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    best_inl = []; best_line = None
    for _ in range(200):
        idx = np.random.choice(len(pts), 2, replace=False)
        p1, p2 = pts[idx]
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        if abs(dx) < 1 and abs(dy) < 1: continue
        a, b = dy, -dx; c = -(a*p1[0]+b*p1[1])
        n = np.hypot(a, b)
        if n < 1e-6: continue
        a, b, c = a/n, b/n, c/n
        d = np.abs(a*pts[:,0]+b*pts[:,1]+c)
        inl = np.where(d < 5.0)[0]
        if len(inl) > len(best_inl):
            best_inl = inl; best_line = (a, b, c)
    if best_line is None or len(best_inl) < 30:
        return None
    return best_line


def fit_H_from_lines(frame_masks, img_shape):
    """Fit a homography from correspondences of pixel lines and world lines.
    frame_masks: dict class_name -> binary mask (h,w)
    Returns H_pix2world or None.
    """
    h, w = img_shape[:2]
    correspondences = []  # (pix_x, pix_y, world_x, world_y)

    for cls, wx, wy0, wy1 in LINES:
        if cls not in frame_masks:
            continue
        mask = frame_masks[cls]
        line = mask_centerline(mask)
        if line is None:
            continue
        a, b, c = line
        # Sample points along the pixel line
        for t in np.linspace(0, 1, 20):
            # point on the world line
            wy = wy0 + (wy1 - wy0) * t
            # corresponding point on the pixel line: projection of (wx, wy) onto line ax+by+c=0
            # But the correspondence along y is unknown. Instead: take a point on the pixel line
            # and match it to the world point with the same relative position.
            # Simplification: for vertical lines (x=const in world) the pixel line is also ~vertical.
            # Take points on the pixel line at even steps along y.
            pass

    # Alternative approach: DLT from line intersection points.
    # Intersection of red center (x=30) and left blue (x=22.4) is impossible (parallel).
    # But intersection of red (x=30) and a board (y=0 or y=30) is possible if a board mask exists.
    # Or: use points from masks directly as correspondences via an ICP-like approach.

    # Simplest working variant: with at least 2 non-parallel lines,
    # one can recover H with 4 degrees of freedom (affine + scale).
    # A full homography needs 4 points.

    # Practical approach: use centroids of line masks as anchor points.
    points_pix = []
    points_world = []

    for cls, wx, wy0, wy1 in LINES:
        if cls not in frame_masks:
            continue
        mask = frame_masks[cls]
        ys, xs = np.where(mask > 0)
        if len(xs) < 50:
            continue
        # Mask centroid
        cx, cy = float(xs.mean()), float(ys.mean())
        # Corresponding world point: middle of the line
        wmx, wmy = wx, (wy0 + wy1) / 2.0
        points_pix.append([cx, cy])
        points_world.append([wmx, wmy])

    if len(points_pix) < 4:
        return None

    pp = np.array(points_pix, np.float32)
    pw = np.array(points_world, np.float32)
    H, status = cv2.findHomography(pp, pw, cv2.RANSAC, 10.0)
    if H is None:
        return None
    nin = int(status.sum()) if status is not None else 0
    if nin < 3:
        return None
    return H


def main():
    model = YOLO('/root/runs/rink_real-3/weights/best.pt')
    cap = cv2.VideoCapture(VID)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / fps
    step_frames = max(1, int(round(STEP * fps)))
    print(f'{VID}: {total} frames, {dur/60:.1f} min, step={step_frames}', flush=True)

    evidence = np.zeros((GH, GW), np.float32)
    photo = np.zeros((GH, GW, 3), np.float32)
    weight = np.zeros((GH, GW), np.float32)
    S = np.array([[PPM, 0, MARGIN * PPM], [0, PPM, MARGIN * PPM], [0, 0, 1]], np.float64)
    recs = []
    t0 = time.time()
    k = 0; idx = -1

    while True:
        ok = cap.grab()
        if not ok: break
        idx += 1
        if idx % step_frames != 0: continue
        ok2, fr = cap.retrieve()
        if not ok2: continue
        ts = int(idx / fps)
        k += 1

        r = model.predict(fr, conf=0.30, device=0, verbose=False)[0]
        names = [CLASSES[int(c)] for c in r.boxes.cls.tolist()] if len(r.boxes) else []

        # Collect masks per class (merge all detections of one class)
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

        n_classes = len(frame_masks)
        if n_classes < 2:
            recs.append({'ts': ts, 'stage': 'few_cls', 'n': len(names)})
            continue

        H = fit_H_from_lines(frame_masks, fr.shape)
        if H is None:
            recs.append({'ts': ts, 'stage': 'no_H', 'n': len(names), 'cls': n_classes})
            continue

        # Check H quality: project world lines and compute overlap with masks
        G = S @ np.linalg.inv(H)  # world -> grid
        H_w2pix = np.linalg.inv(H)

        tot = hit = 0
        for cls, wx, wy0, wy1 in LINES:
            if cls not in frame_masks: continue
            pts = []
            for t in np.linspace(0, 1, 40):
                wy = wy0 + (wy1 - wy0) * t
                p = H_w2pix @ np.array([wx, wy, 1.0])
                px, py = p[0]/p[2], p[1]/p[2]
                if 0 <= px < fr.shape[1] and 0 <= py < fr.shape[0]:
                    pts.append((int(px), int(py)))
            if len(pts) < 5: continue
            check = np.zeros(fr.shape[:2], np.uint8)
            cv2.polylines(check, [np.array(pts)], False, 255, 6)
            tot += int((check > 0).sum())
            hit += int(((check > 0) & (frame_masks[cls] > 0)).sum())

        match = hit / tot if tot > 0 else 0
        if match < 0.30:
            recs.append({'ts': ts, 'stage': 'low_match', 'match': round(match, 3), 'n': len(names)})
            continue

        # Accumulation
        use_mask = np.zeros(fr.shape[:2], np.uint8)
        for cname in ('red_line', 'blue_line', 'center_circle', 'faceoff_circle', 'goal_crease'):
            if cname in frame_masks:
                use_mask = cv2.bitwise_or(use_mask, frame_masks[cname])

        ev = cv2.warpPerspective(use_mask, G, (GW, GH), flags=cv2.INTER_NEAREST)
        evidence += (ev > 0).astype(np.float32)

        ph = cv2.warpPerspective(fr, G, (GW, GH), flags=cv2.INTER_LINEAR)
        wm = cv2.warpPerspective(np.ones(fr.shape[:2], np.uint8)*255, G, (GW, GH), flags=cv2.INTER_NEAREST)
        wmf = (wm > 0).astype(np.float32)
        photo += ph * wmf[..., None]
        weight += wmf

        recs.append({'ts': ts, 'stage': 'good', 'match': round(match, 3), 'n': len(names), 'cls': n_classes})

        if k % 25 == 0:
            el = time.time() - t0
            done = idx / max(1, total)
            eta = el / max(0.01, done) * (1 - done) / 60
            ng = sum(1 for x in recs if x['stage'] == 'good')
            print(f'  idx={idx}/{total} ({done*100:.0f}%) steps={k} good={ng} {el/60:.1f}min ETA {eta:.0f}min', flush=True)

    cap.release()
    n_good = sum(1 for x in recs if x['stage'] == 'good')
    matches = [x['match'] for x in recs if x['stage'] == 'good']
    print(f'usable: {n_good}, median match={np.median(matches) if matches else 0:.2f}', flush=True)

    ev_vis = np.clip(evidence / max(1, n_good // 3), 0, 1)
    cv2.imwrite(f'{OUT}/evidence.png', (ev_vis * 255).astype(np.uint8))
    w = np.clip(weight, 0, None)
    img = np.nan_to_num(photo / w[..., None], 0).astype(np.uint8)
    cv2.imwrite(f'{OUT}/world_photo.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    overlay = img.copy()
    red = (evidence > max(3, n_good * 0.10)).astype(np.uint8) * 255
    overlay[red > 0] = (0, 0, 240)
    cv2.imwrite(f'{OUT}/world_overlay.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])

    with open(f'{OUT}/frames.json', 'w') as fp:
        json.dump(recs, fp)
    print(f'saved to {OUT}, elapsed {(time.time()-t0)/60:.1f} min', flush=True)

if __name__ == '__main__':
    main()
