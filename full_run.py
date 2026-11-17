#!/usr/bin/env python3
"""Full match run: YOLO marking masks + homography per frame -> 60x30 rink mosaic.

Single pass over the video without seek (grab+retrieve) — avoids H.264 rewinding.
Frame geometry: M (anchor_pix -> frame_pix) via ORB against the anchor frame,
H_frame_pix2world = H_anchor_pix2world @ inv(M).
Accumulation in the world grid (PPM px/m): evidence (YOLO masks) + photo (ice photomosaic).
Frame usability: ORB inliers + agreement between projected IIHF geometry and masks.
"""
import cv2, numpy as np, os, json, time, sys

VID = sys.argv[1] if len(sys.argv) > 1 else 'match.mp4'
ANCHOR_IMG = '/root/anchor.jpg'
H_ANCHOR = '/root/H_pix2world.npy'
OUT = '/root/full_run'
STEP = 3.0
PPM = 20                  # pixels per meter
MARGIN = 1.0              # meters around the rink
W_M, H_M = 60.0, 30.0
GW, GH = int((W_M + 2 * MARGIN) * PPM), int((H_M + 2 * MARGIN) * PPM)

os.makedirs(OUT, exist_ok=True)
from ultralytics import YOLO

CLASSES = ['red_line', 'blue_line', 'center_circle', 'faceoff_spot',
           'faceoff_circle', 'goal_crease', 'board']
USE = {'red_line', 'blue_line', 'center_circle', 'faceoff_circle', 'goal_crease'}

# world grid: pix -> grid px
S = np.array([[PPM, 0, MARGIN * PPM], [0, PPM, MARGIN * PPM], [0, 0, 1]], np.float64)

# IIHF geometry for validation (world lines x=const)
WORLD_LINES = {
    'red_center': (30.0, 0, 30), 'goal_l': (4.0, 0, 30), 'goal_r': (56.0, 0, 30),
    'blue_l': (22.4, 0, 30), 'blue_r': (37.6, 0, 30),
}


def line_matches(evidence_grid, Hf_pix2world):
    tot = hit = 0
    for name, (x, y0, y1) in WORLD_LINES.items():
        n = 60
        pts = np.zeros((1, n, 2), np.float32)
        for i in range(n):
            y = y0 + (y1 - y0) * i / (n - 1)
            p = Hf_pix2world @ np.array([x, y, 1.0])
            pts[0, i] = (p[0] / p[2], p[1] / p[2])
        mask = np.zeros((GH, GW), np.uint8)
        cv2.polylines(mask, [pts.astype(np.int32)], False, 255, 3)
        idx = mask > 0
        if idx.sum() < 30:
            continue
        tot += int(idx.sum())
        hit += int((evidence_grid[idx] > 0).sum())
    return (hit / tot) if tot else 0.0


def main():
    model = YOLO('/root/runs/rink_real-3/weights/best.pt')
    Ha = np.load(H_ANCHOR)
    S_h = S @ Ha                     # frame_anchor_pix -> grid px

    anchor = cv2.imread(ANCHOR_IMG)
    orb = cv2.ORB_create(nfeatures=1200)
    amask = np.ones(anchor.shape[:2], np.uint8)
    amask[25:160, 40:650] = 0
    akp, ades = orb.detectAndCompute(anchor, amask)
    print(f'anchor keypoints: {len(akp)}', flush=True)

    cap = cv2.VideoCapture(VID)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / fps
    step_frames = max(1, int(round(STEP * fps)))
    n_steps = total // step_frames
    print(f'{VID}: {total} frames, {dur/60:.1f} min, ~{n_steps} steps', flush=True)

    evidence = np.zeros((GH, GW), np.float32)
    photo = np.zeros((GH, GW, 3), np.float32)
    weight = np.zeros((GH, GW), np.float32)
    recs = []
    t0 = time.time()

    k = 0
    idx = -1
    while True:
        ok = cap.grab()
        if not ok:
            break
        idx += 1
        if idx % step_frames != 0:
            continue
        ok2, fr = cap.retrieve()
        if not ok2:
            continue
        ts = int(idx / fps)
        k += 1

        r = model.predict(fr, conf=0.30, device=0, verbose=False)[0]
        names = [CLASSES[int(c)] for c in r.boxes.cls.tolist()] if len(r.boxes) else []
        use_idx = [i for i, n in enumerate(names) if n in USE]
        if not r.masks or not use_idx:
            recs.append({'ts': ts, 'stage': 'no_det', 'n': len(names)})
            continue

        pix_mask = np.zeros((fr.shape[0], fr.shape[1]), np.uint8)
        for i in use_idx:
            mm = (cv2.resize(r.masks.data[i].cpu().numpy(), (fr.shape[1], fr.shape[0])) > 0.5)
            pix_mask[mm] = 255

        kps, des = orb.detectAndCompute(fr, None)
        if des is None or len(kps) < 40:
            recs.append({'ts': ts, 'stage': 'no_kp', 'n': len(names)})
            continue
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(ades, des, k=2)
        good = [a for a, b in matches if a.distance < 0.75 * b.distance]
        if len(good) < 40:
            recs.append({'ts': ts, 'stage': 'no_match', 'n': len(names)})
            continue
        pa = np.float32([akp[a.queryIdx].pt for a in good])
        pb = np.float32([kps[a.trainIdx].pt for a in good])
        M, mmask = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
        if M is None:
            recs.append({'ts': ts, 'stage': 'no_H', 'n': len(names)})
            continue
        nin = int(mmask.sum())
        Hf_pix2world = Ha @ np.linalg.inv(M)

        ev = cv2.warpPerspective(pix_mask, S @ Hf_pix2world, (GW, GH), flags=cv2.INTER_NEAREST)
        match = line_matches(ev, Hf_pix2world)

        if nin < 80 or match < 0.35:
            recs.append({'ts': ts, 'stage': 'reject', 'inl': nin,
                         'match': round(float(match), 3), 'n': len(names)})
            continue

        evidence += (ev > 0).astype(np.float32)
        ph = cv2.warpPerspective(fr, S @ Hf_pix2world, (GW, GH), flags=cv2.INTER_LINEAR)
        wmask = cv2.warpPerspective(np.ones((fr.shape[0], fr.shape[1]), np.uint8) * 255,
                                    S @ Hf_pix2world, (GW, GH), flags=cv2.INTER_NEAREST)
        wm = (wmask > 0).astype(np.float32)
        photo += ph * wm[..., None]
        weight += wm

        recs.append({'ts': ts, 'stage': 'good', 'inl': nin,
                     'match': round(float(match), 3), 'n': len(names)})

        if k % 25 == 0:
            el = time.time() - t0
            done_frac = idx / max(1, total)
            eta = el / max(0.01, done_frac) * (1 - done_frac) / 60
            ng = sum(1 for x in recs if x['stage'] == 'good')
            print(f'  idx={idx}/{total} ({done_frac*100:.0f}%) steps={k} good={ng} '
                  f'{el/60:.1f} min ETA {eta:.0f} min', flush=True)

    cap.release()

    n_good = sum(1 for x in recs if x['stage'] == 'good')
    matches_good = [x['match'] for x in recs if x['stage'] == 'good']
    print(f'usable frames: {n_good}, median match={np.median(matches_good):.2f}', flush=True)

    # evidence map
    ev_vis = np.clip(evidence / max(1, n_good // 3), 0, 1)
    cv2.imwrite(f'{OUT}/evidence.png', (ev_vis * 255).astype(np.uint8))

    # photo mosaic
    w = np.clip(weight, 0, None)
    img = photo / w[..., None]
    img[w[..., 0] == 0] = 0
    img = np.nan_to_num(img, 0).astype(np.uint8)
    cv2.imwrite(f'{OUT}/world_photo.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # overlay: evidenced markings in red on top of the photo
    overlay = img.copy()
    red = (evidence > max(3, n_good * 0.10)).astype(np.uint8) * 255
    overlay[red > 0] = (0, 0, 240)
    cv2.imwrite(f'{OUT}/world_overlay.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # IIHF reference on top for comparison
    ref = img.copy()
    cols = {'red_center': (0, 0, 255), 'goal_l': (0, 140, 255), 'goal_r': (0, 140, 255),
            'blue_l': (255, 0, 0), 'blue_r': (255, 0, 0)}
    for nm, (x, y0, y1) in WORLD_LINES.items():
        cv2.line(ref, (int((x + MARGIN) * PPM), int((y0 + MARGIN) * PPM)),
                 (int((x + MARGIN) * PPM), int((y1 + MARGIN) * PPM)), cols[nm], 2)
    for cx, cy in [(30, 15), (12, 6), (12, 24), (48, 6), (48, 24)]:
        cv2.circle(ref, (int((cx + MARGIN) * PPM), int((cy + MARGIN) * PPM)),
                   int(4.5 * PPM), (0, 255, 255), 2)
    cv2.imwrite(f'{OUT}/world_ref.jpg', ref, [cv2.IMWRITE_JPEG_QUALITY, 92])

    with open(f'{OUT}/frames.json', 'w') as fp:
        json.dump(recs, fp)
    print(f'saved to {OUT}: evidence.png world_photo.jpg world_overlay.jpg world_ref.jpg')
    print(f'elapsed {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
