#!/usr/bin/env python3
"""Median ice mosaic: stitch a pan video into one clean, player-free frame.

The camera rotates -> the relation between frames is a pure-rotation homography.
Accumulate H via ORB+RANSAC, pin every frame to the frame0 view, take the median
over time: moving players/puck disappear, stationary objects remain
(ice, markings, boards).
"""
import cv2, numpy as np, json, sys, os

F = "./media/11111111---07329963-eef0-4272-a56e-a1c8980a6f9b.mp4"
OUT = os.path.dirname(os.path.abspath(__file__))
CANVAS = (3200, 1240)
STEP = 6
SAMPLE_EVERY = 20

orb = cv2.ORB_create(4000, 1.2, 8)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def step_H(g0, g1):
    s0=cv2.resize(g0,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
    s1=cv2.resize(g1,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
    return step_H_full(s0,s1,0.5)

def step_H_full(g0,g1,dscale):
    kp0, d0 = orb.detectAndCompute(g0, None)
    kp1, d1 = orb.detectAndCompute(g1, None)
    if d0 is None or d1 is None or len(d0) < 150 or len(d1) < 150:
        return None
    m = bf.knnMatch(d0, d1, k=2)
    good = [a for a, b in m if a.distance < 0.8 * b.distance]
    if len(good) < 60:
        return None
    a = np.float32([kp0[x.queryIdx].pt for x in good]).reshape(-1, 1, 2)
    b = np.float32([kp1[x.trainIdx].pt for x in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(a, b, cv2.RANSAC, 3.0)
    if H is None:
        return None
    if dscale != 1.0:
        S=np.array([[1/dscale,0,0],[0,1/dscale,0],[0,0,1]])
        H=np.linalg.inv(S)@H@S
    return H / H[2, 2]


def main():
    cap = cv2.VideoCapture(F)
    ok, f0 = cap.read()
    if not ok:
        sys.exit("no video")
    Hh, W = f0.shape[:2]
    offx, offy = (CANVAS[0] - W) // 2, (CANVAS[1] - Hh) // 2
    T = np.array([[1, 0, offx], [0, 1, offy], [0, 0, 1]], float)
    ones = np.full((Hh, W), 255, np.uint8)

    frames, masks = [], []
    Hacc = np.eye(3)

    def add(fr):
        Hi = np.linalg.inv(Hacc) @ T
        frames.append(cv2.warpPerspective(fr, Hi, (CANVAS[0], CANVAS[1])))
        masks.append(cv2.warpPerspective(ones, Hi, (CANVAS[0], CANVAS[1])) > 200)

    add(f0)
    prev = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if i % STEP == 0:
            Hs = step_H(prev, g)
            if Hs is not None:
                Hacc = Hs @ Hacc
            prev = g
        if i % SAMPLE_EVERY == 0:
            add(fr)
            if len(frames) % 10 == 0:
                print(f"warped {len(frames)} frames (at src frame {i})", flush=True)
    cap.release()
    n = len(frames)
    print("total frames:", n, flush=True)

    stack = np.stack(frames).astype(np.float32)
    mk = np.stack(masks)
    del frames, masks
    stack[~np.repeat(mk[..., None], 3, axis=3)] = np.nan
    cnt = mk.sum(axis=0).astype(np.float32)
    del mk

    med = np.zeros((CANVAS[1], CANVAS[0], 3), np.uint8)
    BLK = 160
    for y0 in range(0, CANVAS[1], BLK):
        y1 = min(y0 + BLK, CANVAS[1])
        with np.errstate(invalid="ignore"):
            m = np.nanmedian(stack[:, y0:y1], axis=0)
        med[y0:y1] = np.nan_to_num(m, nan=0.0).astype(np.uint8)
    del stack

    valid = cnt >= 3
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        sys.exit("empty mosaic")
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    print("coverage bbox px:", int(x0), int(x1), int(y0), int(y1), flush=True)
    crop = med[y0:y1 + 1, x0:x1 + 1].copy()
    crop[~valid[y0:y1 + 1, x0:x1 + 1]] = 0
    cv2.imwrite(f"{OUT}/mosaic.png", crop)
    cv2.imwrite(f"{OUT}/mosaic.jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
    json.dump(dict(x0=int(x0), y0=int(y0), offx=offx, offy=offy,
                   canvas=list(CANVAS), n=n, w=int(crop.shape[1]), h=int(crop.shape[0])),
              open(f"{OUT}/mosaic_meta.json", "w"))
    print("saved:", crop.shape, flush=True)


if __name__ == "__main__":
    main()
