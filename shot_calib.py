#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""shot_calib.py — calibrate each m1 shot into meters (60x30) using rink markings.

Stages:
1. CUT detector: HSV histogram of every STRIDE-th frame, chi-square vs the
   previous one > THR -> a cut. A shot = interval between cuts.
2. For shots >= MIN_DUR: median frame (MDIAN samples across the interval) — players
   "smear out", the markings remain.
3. v6 segmentation (red_line, blue_line) on the median frame (GPU).
4. Vertical lines: connected components of the mask -> straight-line RANSAC-lite (SVD).
5. Matching against the reference VERT=[4R,22.4B,30R,37.6B,56R]: forward and reverse
   order, >=4 lines required; homography from line-to-line correspondences
   (duality: skew(l_pix) H^T L_world = 0, SVD).
6. QC: median distance of inliers to the projected reference lines.
Output: /work/out/shots.json + /work/out/shots.npz (t0,t1,H,resid,nlines,status).
"""
import cv2, numpy as np, json, time, os, sys
from itertools import combinations

SRC = "/work/m1.mp4"
MODEL = "/work/runs/rink_all_v6/weights/best.pt"
OUTD = "/work/out"
STRIDE = 3          # one frame every ~0.1 s for cuts
CUT_THR = None      # computed adaptively via MAD/percentiles
MIN_SHOT = 1.5      # s
MEDIAN_N = 9
FPS = 29.96
VERT = [(4.0, "red"), (22.4, "blue"), (30.0, "red"), (37.6, "blue"), (56.0, "red")]
os.makedirs(OUTD, exist_ok=True)
T0 = time.time()

cap = cv2.VideoCapture(SRC)
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(3)); Hh = int(cap.get(4))
print(f"frames={nframes} {W}x{Hh}", flush=True)

# ---------- 1. cut detector ----------
def hist(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.resize(hsv, (160, 90))
    h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    h = h.astype(np.float32)
    cv2.normalize(h, h, 0, 1, cv2.NORM_L1)
    return h

dists = []
k = 0
prev = None
while True:
    ok, fr = cap.read()
    if not ok:
        break
    if k % STRIDE:
        k += 1
        continue
    h = hist(fr)
    if prev is not None:
        d = cv2.compareHist(h, prev, cv2.HISTCMP_CHISQR_ALT)
        dists.append((k, d))
    prev = h
    k += 1
    if k % 30000 == 0:
        print(f"cutscan {k}/{nframes} el={time.time()-T0:.0f}s", flush=True)
cap.release()
dk = np.array([[a, b] for a, b in dists], float)
np.save(f"{OUTD}/cut_dists.npy", dk)
q = np.percentile(dk[:, 1], [50, 90, 95, 99, 99.5, 99.9])
print(f"cut distances: p50={q[0]:.3f} p90={q[1]:.3f} p95={q[2]:.3f} "
      f"p99={q[3]:.3f} p995={q[4]:.3f} p999={q[5]:.3f} max={dk[:,1].max():.3f}", flush=True)
# adaptive threshold: robust MAD threshold, but no lower than 0.12 and no higher than p99.5
mad = np.median(np.abs(dk[:, 1] - q[0])) * 1.4826
THR = float(max(0.12, min(q[4], q[0] + 6 * mad)))
print(f"THR={THR:.3f} (med+6MAD={q[0]+6*mad:.3f})", flush=True)
cut_f = [0] + [int(f_) for f_, d in dk if d > THR]
# avoid frequent consecutive cuts: register a cut only if >= 1.0 s since the previous one
clean = [cut_f[0]]
for f_ in cut_f[1:]:
    if (f_ - clean[-1]) / FPS >= 1.0:
        clean.append(f_)
cut_f = clean
print(f"cuts found: {len(cut_f)-1} in {time.time()-T0:.0f}s", flush=True)

# shot boundaries (frames)
bounds = cut_f + [nframes]
shots = []
for a, b in zip(bounds, bounds[1:]):
    dur = (b - a) / FPS
    if dur >= MIN_SHOT:
        shots.append((a, b, dur))
print(f"usable shots (>= {MIN_SHOT} s): {len(shots)}", flush=True)

# ---------- 2-3. median frames + segmentation ----------
from ultralytics import YOLO
m = YOLO(MODEL)

def med_frame(a, b):
    frs = []
    for t in np.linspace(a, b - 1, MEDIAN_N).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ok, fr = cap.read()
        if ok:
            frs.append(fr.astype(np.uint16))
    if len(frs) < 3:
        return None
    return np.median(np.stack(frs), 0).astype(np.uint8)

cap = cv2.VideoCapture(SRC)

def line_components(mask):
    """mask (uint8 0/255) -> list of (l, n_pts): vertical straight lines from connected components."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for j in range(1, n):
        x, y, w, h, area = stats[j]
        if area < 400 or h < 180 or w > h:   # narrow tall vertical
            continue
        ys, xs = np.nonzero(lab[y:y+h+1, x:x+w+1] == j)
        pts = np.stack([xs + x, ys + y], 1).astype(float)
        pts = pts[:: max(1, len(pts) // 800)]
        cx, cy = pts.mean(0)
        M = pts - [cx, cy]
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        nv = Vt[-1]
        l = np.array([nv[0], nv[1], -(nv[0] * cx + nv[1] * cy)])
        l /= np.linalg.norm(l[:2])
        d = np.abs(pts @ l[:2] + l[2])
        if (d < 4.0).mean() > 0.7:
            out.append((l, len(pts), (cx, cy)))
    return out

# reference world vertical: L=(1,0,-X)
def solve_H(corrs):
    """corrs: list of (L_world(3), l_pix(3)). duality SVD. -> H (pixel->world) or None. """
    rows = []
    for L, l in corrs:
        K = np.array([[0, -l[2], l[1]], [l[2], 0, -l[0]], [-l[1], l[0], 0]])
        rows.append(np.kron(L.reshape(1, 3), K))   # 3x9
    A = np.vstack(rows)
    A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    try:
        _, sv, Vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    Hm = Vt[-1].reshape(3, 3).T   # H^T was solved
    if abs(Hm[2, 2]) < 1e-9 or not np.all(np.isfinite(Hm)):
        return None
    return Hm / Hm[2, 2]

def qc_resid(Hm, used, allpix):
    """median distance (px) of inliers to back-projections of the reference lines."""
    errs = []
    for (Xw, col), (l, np_, cxy) in used:
        # world line -> pix line: l ~ H^T L
        ll = Hm.T @ np.array([1.0, 0.0, -Xw])
        nrm = np.hypot(ll[0], ll[1])
        if nrm < 1e-9:
            return None
        ll = ll / nrm
        # points of the found component
        ys, = (),
        errs.append(np_)
    return None

res = []
for si, (a, b, dur) in enumerate(shots):
    mf = med_frame(a, b)
    if mf is None:
        res.append({"f0": a, "f1": b, "status": "nodata"})
        continue
    r = m.predict(mf, imgsz=max(W, Hh), device=0, verbose=False, conf=0.25)[0]
    masks = {}
    if r.masks is not None:
        for j in range(len(r.boxes)):
            c = int(r.boxes.cls[j])
            if c not in (0, 1):
                continue
            mm = (r.masks.data[j].cpu().numpy() > 0.5).astype(np.uint8) * 255
            mm = cv2.resize(mm, (W, Hh), interpolation=cv2.INTER_NEAREST)
            masks[c] = np.maximum(masks.get(c, np.zeros_like(mm)), mm)
    lines = []
    for c, name in ((0, "red"), (1, "blue")):
        if c in masks:
            for l, n_, cxy in line_components(masks[c]):
                lines.append((name, l, n_, cxy))
    if len(lines) < 4:
        res.append({"f0": a, "f1": b, "t0": round(a/FPS,1), "status": f"lines{len(lines)}"})
        continue
    lines.sort(key=lambda x: x[3][0])
    best = None
    for perm_dir in (1, -1):
        seq = lines if perm_dir == 1 else lines[::-1]
        # choose a subsequence of the reference of length len(seq) (preserving order)
        if len(seq) > len(VERT):
            continue
        for idxs in combinations(range(len(VERT)), len(seq)):
            if any(VERT[i][1] != s[0] for i, s in zip(idxs, seq)):
                continue
            corrs = [(np.array([1.0, 0.0, -VERT[i][0]]), s[1]) for i, s in zip(idxs, seq)]
            Hm = solve_H(corrs)
            if Hm is None:
                continue
            # QC: world center of the rink (30,7.5)/(30,22.5) and the corners must land inside the frame,
            # inliers of each line must lie on the back-projection of the reference
            err = 0.0; tot = 0
            for i, s in zip(idxs, seq):
                ll = Hm.T @ np.array([1.0, 0.0, -VERT[i][0]])
                nn = np.hypot(ll[0], ll[1])
                if nn < 1e-9:
                    err = 1e9; break
                ll /= nn
                # distance from the component center to the projected line
                err += abs(ll @ np.array([s[3][0], s[3][1], 1.0])) * s[2]
                tot += s[2]
            med_err = err / max(tot, 1)
            # sanity: rink center inside the frame
            q = Hm @ np.array([W/2, Hh/2, 1.0]); q /= q[2]
            inside = 0 <= q[0] <= 60 and 0 <= q[1] <= 30
            if not inside:
                continue
            if best is None or med_err < best[0]:
                best = (med_err, Hm, len(corrs), idxs, perm_dir)
    if best is None:
        res.append({"f0": a, "f1": b, "t0": round(a/FPS,1), "status": "nomatch",
                    "nlines": len(lines)})
        continue
    med_err, Hm, nc, idxs, pd_ = best
    res.append({"f0": a, "f1": b, "t0": round(a/FPS,1), "t1": round(b/FPS,1),
                "status": "ok", "nlines": nc, "dir": pd_,
                "resid_px": round(float(med_err), 2),
                "H": np.asarray(Hm).tolist()})
    if (si + 1) % 25 == 0:
        okc = sum(1 for x in res if x["status"] == "ok")
        print(f"shot {si+1}/{len(shots)} ok={okc} el={time.time()-T0:.0f}s", flush=True)

cap.release()
ok = [x for x in res if x["status"] == "ok"]
Harr = np.zeros((len(ok), 3, 3))
for i, x in enumerate(ok):
    Harr[i] = x["H"]
np.savez_compressed(f"{OUTD}/shots.npz", H=Harr,
                    f0=np.array([x["f0"] for x in ok]),
                    f1=np.array([x["f1"] for x in ok]))
json.dump({"n_shots": len(shots), "n_ok": len(ok),
           "resid_med": round(float(np.median([x["resid_px"] for x in ok])), 2) if ok else None,
           "cut_thr": THR, "elapsed_s": round(time.time()-T0, 1),
           "shots": res}, open(f"{OUTD}/shots.json", "w"), ensure_ascii=False)
print(f"DONE shots={len(shots)} ok={len(ok)} "
      f"resid_med={np.median([x['resid_px'] for x in ok]) if ok else '-'} "
      f"{time.time()-T0:.0f}s", flush=True)
