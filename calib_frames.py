#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""calib_frames.py — independent calibration of m1 windows to meters (60x30) from line markings.

No chains, no drift: every window is calibrated from scratch from the marking lines.
Every SAMPLE_STRIDE-th frame; median of BUF samples (players blur out, lines
stay); v6 segmentation (red_line, blue_line); vertical components ->
lines; matching against the VERT reference (left-to-right order, colors); H from
correspondences L^T H p = 0 (SVD); QC — median residual in METERS.

Also an honest cut diagnostics with two metrics over samples: hue+sat hist chi2
(the previous run gave all zeros: cv2.normalize(..., 0, 1, NORM_L1) zeroed the histogram —
alpha=0; fixed) and gray MAD-diff.

Output: /work/out/calib.npz (t, H, resid_m, status, nlines), calib_report.json
"""
import cv2, numpy as np, json, time, os
from itertools import combinations
from collections import Counter

SRC = "/work/m1.mp4"
MODEL = "/work/runs/rink_all_v6/weights/best.pt"
OUTD = "/work/out"
FPS = 29.96
SAMPLE_STRIDE = 6        # a sample every ~0.2 s
BUF = 7                  # samples in the median window
CALIB_EVERY = int(os.environ.get("CALIB_EVERY", 10))   # ~2 s between windows
CALIB_MAX = int(os.environ.get("CALIB_MAX", 0))        # validation: limit of windows
VERT = [(4.0, "red"), (22.4, "blue"), (30.0, "red"), (37.6, "blue"), (56.0, "red")]
os.makedirs(OUTD, exist_ok=True)
T0 = time.time()


def log(*a):
    print(*a, flush=True)


cap = cv2.VideoCapture(SRC)
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(3)); Hh = int(cap.get(4))
log(f"frames={nframes} {W}x{Hh}")

from ultralytics import YOLO
model = YOLO(MODEL)


def hue_hist(small_bgr):
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(h, h, 1.0, cv2.NORM_L1)
    return h


def line_components(mask, min_len=120, min_area=300):
    """mask uint8 0/255 -> [(l(3,), pts(N,2))].
    CRITICAL: in this broadcast the markings are tilted (camera from the side) —
    the bounding box of such a line is WIDE, so a w/h box filter
    killed all lines (verified: 0/2101 windows). The criterion is now PCA
    elongation of the component + length along the principal axis."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for j in range(1, n):
        x, y, w, h, area = stats[j]
        if area < min_area:
            continue
        ys, xs = np.nonzero(lab[y:y + h + 1, x:x + w + 1] == j)
        pts = np.stack([xs + x, ys + y], 1).astype(float)
        if len(pts) < 40:
            continue
        cx, cy = pts.mean(0)
        d = pts - [cx, cy]
        _, S, Vt = np.linalg.svd(d, full_matrices=False)
        if S[1] < 1e-6 or S[0] / S[1] < 6.0:      # elongation
            continue
        nv = Vt[-1]
        l = np.array([nv[0], nv[1], -(nv[0] * cx + nv[1] * cy)])
        l /= np.linalg.norm(l[:2])
        r = np.abs(pts @ l[:2] + l[2])
        inl = pts[r < 4.0]
        if len(inl) < 40 or (r < 4.0).mean() < 0.6:
            continue
        proj = inl @ Vt[0]
        if np.ptp(proj) < min_len:                 # length along the line
            continue
        out.append((l, inl))
    return out


def solve_H(corrs):
    """corrs: [(X_world, pts_pix(N,2))]. Rows of L^T (H p)=0 over vec(H) row-major."""
    rows = []
    for X, pts in corrs:
        L = np.array([1.0, 0.0, -X])
        idx = np.linspace(0, len(pts) - 1, min(14, len(pts))).astype(int)
        for p in pts[idx]:
            ph = np.array([p[0], p[1], 1.0])
            rows.append(np.outer(L, ph).ravel())
    A = np.asarray(rows, float)
    if A.shape[0] < 12:
        return None
    A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    try:
        _, sv, Vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    Hm = Vt[-1].reshape(3, 3)
    if not np.all(np.isfinite(Hm)) or abs(Hm[2, 2]) < 1e-9:
        return None
    return Hm / Hm[2, 2]


def qc(Hm, corrs):
    """median residual |x_world - X| in meters + sanity of the frame center."""
    errs = []
    for X, pts in corrs:
        q = (Hm @ np.c_[pts, np.ones(len(pts))].T).T
        den = np.abs(q[:, 2])
        ok = den > 1e-9
        if ok.sum() < 5:
            return None, False
        wx = q[ok, 0] / q[ok, 2]
        wy = q[ok, 1] / q[ok, 2]
        good = (wx > -40) & (wx < 100) & (wy > -40) & (wy < 70)
        if good.sum() < 5:
            return None, False
        errs.append(np.abs(wx[good] - X))
    c = Hm @ np.array([W / 2, Hh / 2, 1.0])
    if abs(c[2]) < 1e-9:
        return None, False
    c = c / c[2]
    sane = (-15 <= c[0] <= 75) and (-15 <= c[1] <= 45)
    return float(np.median(np.concatenate(errs))), sane


DBG = int(os.environ.get("DBG", "0"))   # save overlays of the first N windows
_dbg_n = 0


def try_calib(mf):
    global _dbg_n
    r = model.predict(mf, imgsz=960, device=0, verbose=False, conf=0.25)[0]
    masks = {}
    if r.masks is not None:
        for j in range(len(r.boxes)):
            c = int(r.boxes.cls[j])
            if c not in (0, 1):
                continue
            mm = (r.masks.data[j].cpu().numpy() > 0.5).astype(np.uint8) * 255
            mm = cv2.resize(mm, (W, Hh), interpolation=cv2.INTER_NEAREST)
            masks[c] = np.maximum(masks.get(c, np.zeros((Hh, W), np.uint8)), mm)
    found = []
    for c, name in ((0, "red"), (1, "blue")):
        for l, pts in line_components(masks.get(c, np.zeros((Hh, W), np.uint8))):
            found.append((name, l, pts))
    nl = len(found)
    if DBG and _dbg_n < DBG:
        _dbg_n += 1
        vis = mf.copy()
        for c in (0, 1):
            m = masks.get(c)
            if m is not None:
                vis[m > 0] = (0, 0, 255) if c == 0 else (255, 0, 0)
        for i, (nm, l, pts) in enumerate(found):
            for p in pts[::25].astype(int):
                cv2.circle(vis, tuple(p), 2, (0, 255, 0), -1)
            cv2.putText(vis, nm, tuple(pts[0].astype(int)), 0, 0.8, (255, 255, 255), 2)
        cv2.imwrite(f"{OUTD}/dbg_{_dbg_n:02d}.png", vis)
        log(f"DBG {_dbg_n} t={k/FPS:.0f}s nl={nl} "
            + " | ".join(f"{nm} n={len(p)} ext" for nm, l, p in found)
            + " || " + " | ".join(
                f"{nm} len={np.ptp(p[:,0]):.0f}x{np.ptp(p[:,1]):.0f}"
                for nm, l, p in found))
    if nl < 3 or nl > len(VERT):
        return None, (f"lines{nl}" if nl < 3 else "toomanyslow"), nl, None
    found.sort(key=lambda z: z[2][:, 0].mean())
    best = None
    for idxs in combinations(range(len(VERT)), min(nl, len(VERT))):
        if any(VERT[i][1] != z[0] for i, z in zip(idxs, found)):
            continue
        corrs = [(VERT[i][0], z[2]) for i, z in zip(idxs, found)]
        Hm = solve_H(corrs)
        if Hm is None:
            continue
        rm, sane = qc(Hm, corrs)
        if rm is None or not sane:
            continue
        if best is None or rm < best[0]:
            best = (rm, Hm)
    if best is None:
        return None, "nomatch", nl, None
    rm, Hm = best
    return Hm, "ok", nl, rm


d_hist, d_gray = [], []
hist_prev, gray_prev = None, None
buf = []
res_t, res_H, res_rm, res_status, res_nl = [], [], [], [], []
ns = 0
k = -1
while True:
    if not cap.grab():
        break
    k += 1
    if k % SAMPLE_STRIDE:
        continue
    ok, fr = cap.retrieve()
    if not ok:
        break
    ns += 1
    small = cv2.resize(fr, (160, 90))
    hh = hue_hist(small)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if hist_prev is not None:
        d_hist.append(cv2.compareHist(hh, hist_prev, cv2.HISTCMP_CHISQR_ALT))
        d_gray.append(float(np.abs(g - gray_prev).mean()))
    hist_prev, gray_prev = hh, g
    buf.append(fr.astype(np.uint16))
    if len(buf) > BUF:
        buf.pop(0)
    if ns % CALIB_EVERY == 0 and len(buf) == BUF:
        if CALIB_MAX and len(res_status) >= CALIB_MAX:
            break
        mf = np.median(np.stack(buf, 0), 0).astype(np.uint8)
        Hm, st, nl, rm = try_calib(mf)
        res_t.append(k / FPS)
        res_H.append(Hm if Hm is not None else np.zeros((3, 3)))
        res_status.append(st)
        res_nl.append(nl)
        res_rm.append(rm if rm is not None else -1.0)
        if ns % 3000 == 0:
            okc = sum(1 for s in res_status if s == "ok")
            log(f"samp={ns} t={k/FPS:.0f}s ok={okc}/{len(res_status)} "
                f"el={time.time()-T0:.0f}s eta~{(time.time()-T0)/ns*21000/60:.0f}m")
cap.release()
log(f"pass {time.time()-T0:.0f}s, calibration windows {len(res_status)}")

dh = np.array(d_hist); dg = np.array(d_gray)
rep = {
    "hist": {"p50": round(float(np.percentile(dh, 50)), 5),
             "p99": round(float(np.percentile(dh, 99)), 5),
             "p999": round(float(np.percentile(dh, 99.9)), 5),
             "max": round(float(dh.max()), 5)},
    "gray_mad": {"p50": round(float(np.percentile(dg, 50)), 2),
                 "p99": round(float(np.percentile(dg, 99)), 2),
                 "p999": round(float(np.percentile(dg, 99.9)), 2),
                 "max": round(float(dg.max()), 2)},
    "top_hist_t": [[round(float(i * SAMPLE_STRIDE / FPS), 1), round(float(dh[i]), 4)]
                   for i in np.argsort(-dh)[:20]],
    "top_gray_t": [[round(float(i * SAMPLE_STRIDE / FPS), 1), round(float(dg[i]), 2)]
                   for i in np.argsort(-dg)[:20]],
    "calib_windows": len(res_status),
    "calib_ok": sum(1 for s in res_status if s == "ok"),
    "status_counts": dict(Counter(res_status)),
    "elapsed_s": round(time.time() - T0, 1),
}
rmo = [r for r, s in zip(res_rm, res_status) if s == "ok"]
if rmo:
    rep["resid_m_med"] = round(float(np.median(rmo)), 3)
    rep["resid_m_p90"] = round(float(np.percentile(rmo, 90)), 3)
    rep["resid_m_max"] = round(float(np.max(rmo)), 3)
np.savez_compressed(f"{OUTD}/calib.npz",
                    t=np.array(res_t, float), H=np.array(res_H, float),
                    resid_m=np.array(res_rm, float), status=np.array(res_status),
                    nlines=np.array(res_nl, int))
json.dump(rep, open(f"{OUTD}/calib_report.json", "w"), ensure_ascii=False, indent=1)
log("DONE " + json.dumps({k2: v for k2, v in rep.items() if not k2.startswith("top")},
                         ensure_ascii=False))
