#!/usr/bin/env python3
"""Run v5: chained ORB transfer + YOLO masks -> 60x30 rink mosaic.

Idea: instead of a single anchor — every frame is matched with the previous good frame.
Between neighboring frames (2-3 sec) the background barely changes -> ORB is reliable.
The homography accumulates along the chain: H_frame = H_prev @ M(prev->frame).
YOLO line masks are projected into the world grid and accumulated.
"""
import cv2, numpy as np, os, json, time, sys
from ultralytics import YOLO

VID = sys.argv[1] if len(sys.argv) > 1 else 'match.mp4'
OUT = '/root/full_run_v5'
STEP = 2.0
PPM = 20
MARGIN = 1.0
W_M, H_M = 60.0, 30.0
GW, GH = int((W_M + 2 * MARGIN) * PPM), int((H_M + 2 * MARGIN) * PPM)
os.makedirs(OUT, exist_ok=True)

CLASSES = ['red_line', 'blue_line', 'center_circle', 'faceoff_spot',
           'faceoff_circle', 'goal_crease', 'board']
USE = {'red_line', 'blue_line', 'center_circle', 'faceoff_circle', 'goal_crease'}
LINE_X = [4.0, 22.4, 30.0, 37.6, 56.0]
S = np.array([[PPM, 0, MARGIN * PPM], [0, PPM, MARGIN * PPM], [0, 0, 1]], np.float64)

MIN_INL = 40          # inlier threshold for a neighboring frame (lower than for the anchor)
MIN_MATCH = 0.25      # line-vs-mask overlap threshold


def grid_line_match(ev):
    tot = hit = 0
    for wx in LINE_X:
        gx = int((wx + MARGIN) * PPM)
        if gx < 2 or gx >= GW - 2:
            continue
        col = ev[max(0, int(MARGIN * PPM) - 3):min(GH, int((H_M + MARGIN) * PPM) + 3),
                 max(0, gx - 2):gx + 3]
        seg = col > 0
        tot += seg.shape[0]
        hit += int(seg.any(axis=1).sum())
    return hit / tot if tot else 0.0


def main():
    model = YOLO('/root/runs/rink_real-3/weights/best.pt')
    Ha = np.load('/root/H_pix2world.npy')
    orb = cv2.ORB_create(nfeatures=2500)

    # initial anchor
    anchor = cv2.imread('/root/anchor.jpg')
    akp, ades = orb.detectAndCompute(anchor, None)
    print(f'anchor kps={len(akp)}', flush=True)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    cap = cv2.VideoCapture(VID)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_frames = max(1, int(round(STEP * fps)))
    print(f'{VID}: {total} frames, dur {total/fps/60:.1f} min, step {step_frames}', flush=True)

    evidence = np.zeros((GH, GW), np.float32)
    photo = np.zeros((GH, GW, 3), np.float32)
    weight = np.zeros((GH, GW), np.float32)
    recs = []
    t0 = time.time()

    # chain state
    prev_kps = akp
    prev_des = ades
    prev_H = Ha.copy()
    chain_len = 0
    MAX_CHAIN = 80  # reset the chain to the anchor every N frames (drift accumulation)

    k = idx = -1
    while True:
        if not cap.grab():
            break
        idx += 1
        if idx % step_frames:
            continue
        ok, fr = cap.retrieve()
        if not ok:
            continue
        k += 1
        ts = idx / fps

        r = model.predict(fr, conf=0.30, device=0, verbose=False)[0]
        names = [CLASSES[int(c)] for c in r.boxes.cls.tolist()] if len(r.boxes) else []
        use_idx = [i for i, n in enumerate(names) if n in USE]
        if not r.masks or not use_idx:
            recs.append({'ts': round(ts, 1), 'stage': 'no_det', 'n': len(names)})
            continue

        pix_mask = np.zeros((fr.shape[0], fr.shape[1]), np.uint8)
        for i in use_idx:
            mm = cv2.resize(r.masks.data[i].cpu().numpy(), (fr.shape[1], fr.shape[0]))
            pix_mask[mm > 0.5] = 255

        kps, des = orb.detectAndCompute(fr, None)
        if des is None or len(kps) < 30:
            recs.append({'ts': round(ts, 1), 'stage': 'no_kp', 'n': len(names)})
            continue

        # match with the previous frame in the chain
        matches = bf.knnMatch(prev_des, des, k=2)
        good = [m for m, n2 in matches if m.distance < 0.75 * n2.distance]

        if len(good) < MIN_INL or chain_len >= MAX_CHAIN:
            # fallback: try the anchor directly
            matches_a = bf.knnMatch(ades, des, k=2)
            good_a = [m for m, n2 in matches_a if m.distance < 0.75 * n2.distance]
            if len(good_a) >= 60:
                pa = np.float32([akp[m.queryIdx].pt for m in good_a])
                pb = np.float32([kps[m.trainIdx].pt for m in good_a])
                M, msk = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
                if M is not None and int(msk.sum()) >= 60:
                    Hf = Ha @ np.linalg.inv(M)
                    chain_len = 0
                    prev_kps, prev_des, prev_H = kps, des, Hf
                    src = 'anchor'
                else:
                    recs.append({'ts': round(ts, 1), 'stage': 'reject', 'src': 'anchor_fallback'})
                    continue
            else:
                recs.append({'ts': round(ts, 1), 'stage': 'no_match',
                             'chain_good': len(good), 'anchor_good': len(good_a)})
                continue
        else:
            pa = np.float32([prev_kps[m.queryIdx].pt for m in good])
            pb = np.float32([kps[m.trainIdx].pt for m in good])
            M, msk = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
            if M is None or int(msk.sum()) < MIN_INL:
                recs.append({'ts': round(ts, 1), 'stage': 'no_H', 'good': len(good)})
                continue
            Hf = prev_H @ np.linalg.inv(M)
            chain_len += 1
            prev_kps, prev_des, prev_H = kps, des, Hf
            src = f'chain({chain_len})'

        G = S @ Hf
        ev = cv2.warpPerspective(pix_mask, G, (GW, GH), flags=cv2.INTER_NEAREST)
        match = grid_line_match(ev)
        if match < MIN_MATCH:
            recs.append({'ts': round(ts, 1), 'stage': 'low_match', 'match': round(match, 3), 'src': src})
            continue

        evidence += (ev > 0).astype(np.float32)
        ph = cv2.warpPerspective(fr, G, (GW, GH), flags=cv2.INTER_LINEAR)
        wm = cv2.warpPerspective(np.full(fr.shape[:2], 255, np.uint8), G, (GW, GH),
                                 flags=cv2.INTER_NEAREST)
        wmf = (wm > 0).astype(np.float32)
        photo += ph * wmf[..., None]
        weight += wmf

        recs.append({'ts': round(ts, 1), 'stage': 'good', 'match': round(match, 3),
                     'src': src, 'n': len(names)})

        if k % 50 == 0:
            el = time.time() - t0
            done = idx / max(1, total)
            eta = el / max(0.01, done) * (1 - done) / 60
            ng = sum(1 for x in recs if x['stage'] == 'good')
            print(f'  {idx}/{total} ({done*100:.0f}%) steps={k} good={ng} '
                  f'chain={chain_len} ETA {eta:.0f}min', flush=True)

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
