#!/usr/bin/env python3
"""Homography propagation for the figure-skating venue (wide_fg/*) from fits.json anchors.
H_seed: pixel->world. For a new frame, ORB matching against the N temporally nearest
calibrated frames of the same video; M: new_pix->seed_pix (RANSAC),
H_new = H_seed @ M. Inlier threshold 40. Chain: a new good frame joins the pool."""
import json, glob, os
import numpy as np, cv2
from pathlib import Path

BASE = Path(".")
OUT = BASE / "fs_calib"
FRAMES = sorted((BASE / "wide_fg").glob("*.jpg"))
MIN_INL = 40
NEAR = 4

orb = cv2.ORB_create(2000)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

def kpts(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(g, None)
    return kp, des

fits = json.loads((OUT / "fits.json").read_text())
pool = {}  # stem -> (H, kp, des)
for stem, v in fits.items():
    img = cv2.imread(str(BASE / "wide_fg" / f"{stem}.jpg"))
    kp, des = kpts(img)
    pool[stem] = (np.array(v["H"], np.float64), kp, des)

def order_by_time(stems):
    return sorted(stems, key=lambda s: float(s.rsplit("_", 1)[1]))

def try_calibrate(stem, cand):
    img = cv2.imread(str(BASE / "wide_fg" / f"{stem}.jpg"))
    if img is None: return None
    kp2, des2 = kpts(img)
    if des2 is None or len(kp2) < 20: return None
    best = None
    for s in cand:
        H_seed, kp1, des1 = pool[s]
        if des1 is None: continue
        try:
            m = matcher.knnMatch(des1, des2, k=2)
        except Exception:
            continue
        good = [a for a, b in m if a.distance < 0.8 * b.distance]
        if len(good) < 12: continue
        src = np.float32([kp1[a.queryIdx].pt for a in good])   # seed pix
        dst = np.float32([kp2[a.trainIdx].pt for a in good])   # new pix
        # M: new_pix -> seed_pix
        M, mask = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
        if M is None: continue
        inl = int(mask.ravel().sum())
        if inl >= MIN_INL and (best is None or inl > best[1]):
            best = (M, inl, s)
    if best is None: return None
    M, inl, seed = best
    H_seed = pool[seed][0]
    return H_seed @ M, inl, seed

stems = [p.stem for p in FRAMES]

results = {}
calibrated = [s for s in pool]  # global pool: fs1/fs2/fs3 overlap of one skate session
by_time = {s: float(s.rsplit("_", 1)[1]) for s in stems}
for s in sorted(stems, key=lambda x: by_time[x]):
    if s in pool:
        results[s] = {"H": pool[s][0].tolist(), "anchor": True, "inl": None} if pool[s] is not None else results.get(s, {"H": fits[s]["H"], "anchor": True, "inl": None})
        continue
    if not calibrated: continue
    cand = sorted(calibrated, key=lambda c: abs(by_time[c] - by_time[s]))[:8]
    out = try_calibrate(s, cand)
    if out is not None:
        H, inl, seed = out
        img = cv2.imread(str(BASE / "wide_fg" / f"{s}.jpg"))
        kp, des = kpts(img)
        pool[s] = (H, kp, des)
        calibrated.append(s)
        results[s] = {"H": H.tolist(), "anchor": False, "inl": inl, "seed": seed}

n_anchor = sum(1 for v in results.values() if v["anchor"])
n_prop = len(results) - n_anchor
per = {}
for s, v in results.items():
    b = per.setdefault(s.rsplit("_", 1)[0], [0, 0])
    b[0 if v["anchor"] else 1] += 1
print(f"calibrated {len(results)} of {len(stems)}: anchors {n_anchor}, propagated {n_prop}")
print("per video:", {k: f"{a}+{p}" for k, (a, p) in sorted(per.items())})
np.save(OUT / "H_all.npy", np.array([results[s]["H"] for s in stems if s in results]))
(OUT / "propagate_fs.json").write_text(json.dumps({
    s: {k: v for k, v in r.items() if k != "H"} for s, r in results.items()}))
os.makedirs(OUT / "H", exist_ok=True)
for s, r in results.items():
    np.save(OUT / "H" / f"{s}.npy", np.array(r["H"]))
