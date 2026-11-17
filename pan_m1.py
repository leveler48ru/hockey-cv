#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pan_m1.py — pan compensation for m1: pixel->world homography for each sample.

Seed frame t=SEED_T (2400 s): H0 = seed_m1.npz['H'] (pixel->world, x:0..60 along
the rink, y:0..30), validated 08.11 (demo demo_geom_0811.mp4).
Every STRIDE-th frame (SEQUENTIAL reading, n_all counts real frames):
M = estimateAffinePartial2D (ORB, cur->prev), chain Hg = Hg @ M.
Weak linkage (<MIN_INL) -> full-homography fallback to prev; that fails too -> cut:
new group, anchor to the seed directly: Hg = H0 @ T (T: cur->seed).
Cannot anchor -> world=None (repeats are not converted to metres, that's honest).
CPU-only. Output: /work/out/pan_m1.npz, pan_metrics.json.
"""
import cv2, numpy as np, json, time, os, sys

SRC = "/work/m1.mp4"
SEED_NPZ = "/work/fs/seed_m1.npz"
OUTD = "/work/out"
SEED_T = 2400.0
STRIDE = 5
FPS = 29.96
NORB = 1500
RATIO = 0.78
MIN_INL = 12
MIN_INL_SEED = 25
ABS_THRESH = 4.0

os.makedirs(OUTD, exist_ok=True)
H0 = np.load(SEED_NPZ)["H"].astype(np.float64)

cap = cv2.VideoCapture(SRC)
if not cap.isOpened():
    sys.exit("no video")
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"frames={nframes} fps={cap.get(cv2.CAP_PROP_FPS)}", flush=True)

orb = cv2.ORB_create(NORB)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

def match_pts(cur, prv, ratio):
    kp1, d1 = cur
    kp2, d2 = prv
    if d1 is None or d2 is None or len(d1) < 8 or len(d2) < 8:
        return None
    m = bf.knnMatch(d1, d2, k=2)
    good = [a for a, b in m if a.distance < ratio * b.distance]
    if len(good) < 8:
        return None
    src = np.float32([kp1[a.queryIdx].pt for a in good])
    dst = np.float32([kp2[a.trainIdx].pt for a in good])
    return src, dst

def motion(cur, prv):
    """M (2x3): cur->prev, partial affine. (None, 0) if matching is poor."""
    p = match_pts(cur, prv, RATIO)
    if p is None:
        return None, 0
    M, inl = cv2.estimateAffinePartial2D(p[0], p[1], method=cv2.RANSAC,
                                         ransacReprojThreshold=ABS_THRESH)
    if M is None:
        return None, 0
    return M, int(inl.ravel().sum())

def homography_to(cur, ref, ratio=0.80, thresh=5.0, min_inl=25):
    """T (3x3): cur->ref full homography or None."""
    p = match_pts(cur, ref, ratio)
    if p is None:
        return None, 0
    T, inl = cv2.findHomography(p[0], p[1], cv2.RANSAC, thresh)
    if T is None:
        return None, 0
    return T, int(inl.ravel().sum())

def read_at_msec(ms):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap.set(cv2.CAP_PROP_POS_MSEC, ms)
    ok, fr = cap.read()
    return None if not ok else cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)

seed_gray = read_at_msec(SEED_T * 1000.0)
if seed_gray is None:
    sys.exit("seed frame read fail")
seedf = (lambda g: (lambda r: (r[0], r[1]))(orb.detectAndCompute(g, None)))(seed_gray)
print(f"seed kps={len(seedf[0])}", flush=True)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
G, SAMP, A = [], [], []
group_anch = []
gid = -1
Hg = None
prev_feats = None
n_all = -1
i = 0
t_start = time.time()
recs = []

while True:
    ok, frame = cap.read()
    if not ok:
        break
    n_all += 1
    if n_all % STRIDE:
        continue
    f = n_all
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    cur = (kp, des)
    if i == 0:
        gid += 1
        T, ni = homography_to(cur, seedf)
        if T is not None and ni >= MIN_INL_SEED:
            Hg, ya = H0 @ T, True
        else:
            Hg, ya = None, False
        group_anch.append(ya)
        recs.append((f, ni, f"init y={int(ya)}"))
    else:
        M, ni = motion(cur, prev_feats)
        if M is not None and ni >= MIN_INL:
            if Hg is not None:
                Hg = Hg @ np.vstack([M, [0.0, 0.0, 1.0]])
        else:
            Tf, ninf = homography_to(cur, prev_feats, ratio=0.85, min_inl=30)
            if Tf is not None and ninf >= 30 and Hg is not None:
                Hg = Hg @ Tf  # fast pan/zoom: partial didn't converge, full did
                recs.append((f, ni, f"homprev inl={ninf}"))
            else:
                gid += 1
                T, nis = homography_to(cur, seedf)
                if T is not None and nis >= MIN_INL_SEED:
                    Hg, ya = H0 @ T, True
                    det = f"cut y=1 inl={nis}"
                else:
                    Hg, ya = None, False
                    det = f"cut y=0 ({ni}/{nis})"
                group_anch.append(ya)
                recs.append((f, ni, det))
    G.append(gid)
    SAMP.append(f)
    A.append(Hg if Hg is not None else np.zeros((2, 3)))
    prev_feats = cur
    i += 1
    if i % 2000 == 0:
        el = time.time() - t_start
        print(f"i={i} f={f} t={f/FPS:.0f}s gid={gid} "
              f"anch={sum(group_anch)}/{gid+1} el={el:.0f}s "
              f"rate={i / max(el, .001):.1f} samp/s eta={(nframes/STRIDE - i) / max(i / max(el, .001), .01) / 60:.0f}m",
              flush=True)

cap.release()
G = np.array(G, np.int32)
SAMP = np.array(SAMP, np.int32)
A = np.stack(A)
anch = np.array(group_anch, bool)
samp_ok = anch[G]
np.savez_compressed(os.path.join(OUTD, "pan_m1.npz"),
                    samples=SAMP, G=G, A=A, group_anchored=anch,
                    H0=H0, seed_t=SEED_T, stride=STRIDE, nframes=n_all + 1)
sc = np.linalg.norm(A[:, 0, :2], axis=1)
sc_ok = sc[samp_ok > 0]
out = {"n_samples": int(len(SAMP)), "frames_read": int(n_all + 1),
       "groups": int(gid + 1), "anchored_groups": int(sum(group_anch)),
       "samples_anchored_frac": round(float(samp_ok.mean()), 4),
       "scale_med": round(float(np.median(sc_ok)), 4) if len(sc_ok) else None,
       "scale_p05": round(float(np.percentile(sc_ok, 5)), 4) if len(sc_ok) else None,
       "scale_p95": round(float(np.percentile(sc_ok, 95)), 4) if len(sc_ok) else None,
       "elapsed_s": round(time.time() - t_start, 1),
       "cut_log_last": recs[-100:]}
json.dump(out, open(os.path.join(OUTD, "pan_metrics.json"), "w"), ensure_ascii=False, indent=1)
print("DONE", json.dumps({k: v for k, v in out.items() if k != "cut_log_last"}), flush=True)
