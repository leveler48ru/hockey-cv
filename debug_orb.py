#!/usr/bin/env python3
"""Diagnosis: why ORB matching of frames decoded from the video failed while propagate.py worked on jpgs.
Tests the propagate parameters (nfeatures=2500, anchor without a mask, frame with a mask, >=60)
against frames decoded directly from match.mp4.
"""
import cv2, numpy as np

anchor = cv2.imread('/root/anchor.jpg')
orb = cv2.ORB_create(nfeatures=2500)
akp, ades = orb.detectAndCompute(anchor, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
print(f'anchor kps: {len(akp)}')

cap = cv2.VideoCapture('/root/match.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
SB = (40, 25, 650, 160)

for ts in (300, 303, 306, 900, 1500, 2400):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
    ok, fr = cap.read()
    if not ok:
        print(f'{ts}: read fail'); continue
    m = np.ones((fr.shape[0], fr.shape[1]), np.uint8)
    m[SB[1]:SB[3], SB[0]:SB[2]] = 0
    kps, des = orb.detectAndCompute(fr, m)
    if des is None:
        print(f'{ts}: no des'); continue
    matches = bf.knnMatch(ades, des, k=2)
    good = [a for a, b in matches if a.distance < 0.75 * b.distance]
    if len(good) < 10:
        print(f'{ts}: good={len(good)} (too few)'); continue
    pa = np.float32([akp[a.queryIdx].pt for a in good])
    pb = np.float32([kps[a.trainIdx].pt for a in good])
    M, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
    nin = int(mask.sum()) if M is not None else -1
    print(f'{ts}: kps={len(kps)} good={len(good)} inliers={nin}')
cap.release()
