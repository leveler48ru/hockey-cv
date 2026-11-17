#!/usr/bin/env python3
"""Global calibration of the hockey rink on a GPU VM — v2 (robust).
Every step is wrapped in try/except, logs errors and continues.
"""
import cv2, numpy as np, json, os, sys, time, traceback

BASE = os.environ.get("WORK", "/root")
OUT = f"{BASE}/out"; os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/overlays", exist_ok=True)
F = f"{BASE}/match.mp4"
FPS = 30.0; Wd, Hd = 1920, 1080
NS, NM = 1000.0, 30.0
STEP_SEC = float(os.environ.get("STEP", 5.0))
ICE_MIN_FRAC = float(os.environ.get("ICE_MIN", 0.15))
RED_X = [4.0, 30.0, 56.0]; BLUE_X = [22.4, 37.6]
ALL_VX = sorted(BLUE_X + RED_X)
T0 = time.time()
def log(*a): print(*a, flush=True)

# ========== H_A ==========
try:
    anc = json.load(open(f"{BASE}/anchors.json"))
    m1 = np.array(anc.get('G_world2pix', [])); m2 = np.array(anc.get('H_pix2world', []))
    def is_p2w(m):
        if m.size == 0: return False
        v = m @ np.array([690/NS, 370/NS, 1.0]); w = v[:2]/v[2]
        return abs(w[0]*NM - 30) < 2 and abs(w[1]*NM - 30) < 2
    P2W = m1 if is_p2w(m1) else (m2 if is_p2w(m2) else None)
    if P2W is None: raise ValueError("cannot find pixel->world")
    W2P_n = np.linalg.inv(P2W)
    H_A = np.diag([NS, NS, 1.0]) @ W2P_n @ np.diag([1/NM, 1/NM, 1.0])
    a_val = anc.get('a', 5.80); b_val = anc.get('b', 8.60)
except Exception as e:
    log(f"H_A fallback: {e}")
    H_A = np.array([[0.0583, -0.0012, 0.247], [0.0008, 0.0412, 0.012], [0.000031, 0.00018, 1.0]])
    a_val, b_val = 5.80, 8.60

v = H_A @ np.array([30, 30, 1.]); v /= v[2]
log(f"H_A: world(30,30)->px({v[0]:.0f},{v[1]:.0f}), a={a_val}, b={b_val}")

# ========== Decode ==========
cap = cv2.VideoCapture(F)
if not cap.isOpened(): sys.exit("no video")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / FPS
times = np.arange(0, duration, STEP_SEC)
frames = []
for t in times:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * FPS))
    ok, fr = cap.read()
    if ok: frames.append((float(t), fr))
    if len(frames) % 100 == 0: log(f"  decode {len(frames)}/{len(times)}")
cap.release()
log(f"decoded {len(frames)} frames in {time.time()-T0:.0f}s")

# ========== Segmentation ==========
def segment_ice(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi = gray[Hd//3:, :]
    thr_val, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(gray, max(int(thr_val) - 20, 100), 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask[:Hd//3, :] = 0
    frac = float(mask.sum()) / (Wd * Hd)
    return mask.astype(bool), frac

def segment_lines(img, ice_mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    Hh, Ss, Vv = hsv[...,0], hsv[...,1], hsv[...,2]
    RED = (((Hh <= 12) | (Hh >= 165)) & (Ss > 50) & (Vv > 40)) & ice_mask
    BLUE = ((Hh >= 90) & (Hh <= 140) & (Ss > 25) & (Vv > 40)) & ice_mask
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    RED_m = cv2.morphologyEx(RED.astype(np.uint8)*255, cv2.MORPH_CLOSE, k) > 0
    BLUE_m = cv2.morphologyEx(BLUE.astype(np.uint8)*255, cv2.MORPH_CLOSE, k) > 0
    return RED_m, BLUE_m

def find_vlines(mask, min_len=150, tol=4.0):
    ys, xs = np.nonzero(mask)
    if len(ys) < 100: return []
    pts = np.stack([xs, ys], 1).astype(float)
    rng = np.random.default_rng(42)
    lines = []
    for _ in range(5):
        if len(pts) < 100: break
        best = None
        for _it in range(300):
            i, j = rng.choice(len(pts), 2, replace=False)
            if abs(pts[i,1] - pts[j,1]) < 60: continue
            a = (pts[i,0] - pts[j,0]) / (pts[i,1] - pts[j,1])
            b = pts[i,0] - a * pts[i,1]
            res = np.abs(pts[:,0] - (a * pts[:,1] + b))
            inl = res < tol
            n_ = int(inl.sum())
            sp = float(np.ptp(pts[inl, 1])) if n_ > 1 else 0
            sc = n_ if sp > min_len else 0
            if best is None or sc > best[0]: best = (sc, inl, a, b)
        if best is None or best[0] < 80: break
        _, inl, a, b = best
        lp = pts[inl]
        cx, cy = lp[:,0].mean(), lp[:,1].mean()
        M = np.column_stack([lp[:,0]-cx, lp[:,1]-cy])
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        n = Vt[-1]
        l = np.array([n[0], n[1], -(n[0]*cx + n[1]*cy)])
        l = l / np.linalg.norm(l[:2])
        lines.append((l, lp))
        pts = pts[np.abs(pts @ l[:2] + l[2]) >= tol]
    return lines

def x_at_y(line, y):
    a, b, c = line
    if abs(a) < 1e-9: return None
    return (-b * y - c) / a

def label_lines(vlines):
    if len(vlines) < 3: return None
    mid_y = Hd // 2
    scored = []
    for l, pts in vlines:
        x = x_at_y(l, mid_y)
        if x is not None and 0 < x < Wd:
            scored.append((x, l, pts))
    if len(scored) < 3: return None
    scored.sort(key=lambda s: s[0])
    n = len(scored)
    if n > len(ALL_VX): return None
    best_match = None; best_err = 1e9
    for start in range(len(ALL_VX) - n + 1):
        subset = ALL_VX[start:start+n]
        pix_xs = [s[0] for s in scored]
        pix_span = pix_xs[-1] - pix_xs[0]
        world_span = subset[-1] - subset[0]
        if pix_span < 50 or world_span < 1: continue
        scale = pix_span / world_span
        err = sum(abs(pix_xs[i] - (subset[i] - subset[0]) * scale - pix_xs[0]) for i in range(n))
        if err < best_err:
            best_err = err; best_match = list(zip(subset, scored))
    if best_match is None: return None
    return [(s[1][1], s[1][2], s[0]) for s in best_match]

def solve_H(labeled_lines):
    rows = []
    for line, pts, wx in labeled_lines:
        sub = pts[::max(1, len(pts)//50)]
        for px, py in sub:
            rows.append([px, py, 1, 0, 0, 0, -wx*px, -wx*py, -wx])
    if len(rows) < 8: return None
    A = np.array(rows)
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    _, sv, Vt = np.linalg.svd(A, full_matrices=False)
    h = Vt[-1]
    H = h.reshape(3, 3)
    if abs(H[2,2]) < 1e-9: return None
    return H / H[2,2]

# ========== Main loop ==========
results = []
wide_count = 0
errors = 0
for idx, (t, img) in enumerate(frames):
    try:
        ice_mask, ice_frac = segment_ice(img)
        if ice_frac < ICE_MIN_FRAC:
            results.append({"t": t, "wide": False, "ice_frac": round(ice_frac, 3)})
            continue
        wide_count += 1
        RED, BLUE = segment_lines(img, ice_mask)
        red_lines = find_vlines(RED)
        blue_lines = find_vlines(BLUE)
        all_vlines = red_lines + blue_lines
        labeled = label_lines(all_vlines)
        if labeled is None or len(labeled) < 3:
            results.append({"t": t, "wide": True, "ice_frac": round(ice_frac, 3),
                            "lines": len(all_vlines), "status": "no_label"})
            continue
        H = solve_H(labeled)
        if H is None:
            results.append({"t": t, "wide": True, "ice_frac": round(ice_frac, 3),
                            "lines": len(all_vlines), "labeled": len(labeled), "status": "no_H"})
            continue
        # Residual
        Hi = np.linalg.inv(H)
        errs = []
        for line, pts, wx in labeled:
            sub = pts[::max(1, len(pts)//100)]
            l_proj = Hi.T @ np.array([1.0, 0.0, -wx/NM])
            n = np.hypot(l_proj[0], l_proj[1])
            if n > 1e-9:
                for px, py in sub:
                    d = abs(l_proj[0]*px/NS + l_proj[1]*py/NS + l_proj[2]) / n * NS
                    errs.append(d)
        med_err = float(np.median(errs)) if errs else 999.0
        results.append({
            "t": t, "wide": True, "ice_frac": round(ice_frac, 3),
            "lines": len(all_vlines), "labeled": len(labeled),
            "med_err_px": round(med_err, 1), "H": H.tolist(),
            "world_xs": [wx for _, _, wx in labeled]
        })
        if (idx+1) % 50 == 0:
            log(f"  processed {idx+1}/{len(frames)}, wide shots {wide_count}, residual {med_err:.1f} px")
    except Exception as e:
        errors += 1
        results.append({"t": t, "error": str(e)[:200]})
        if errors <= 5:
            log(f"  ERROR @{t:.0f}s: {e}")

log(f"\nprocessing: {len(frames)} frames, {wide_count} wide shots, {errors} errors, {time.time()-T0:.0f}s")

good = [r for r in results if r.get("med_err_px", 999) < 15]
log(f"usable (residual <15 px): {len(good)}/{wide_count}")
if good:
    errs_g = [r["med_err_px"] for r in good]
    log(f"  median {np.median(errs_g):.1f} px, p90 {np.percentile(errs_g, 90):.1f} px")

json.dump(results, open(f"{OUT}/yaw_profile.json", "w"), indent=1)

# ========== Mosaic ==========
if len(good) >= 3:
    PX = 30; MW, MH = 60*PX, 30*PX
    acc = np.zeros((MH, MW, 3), np.float32); cnt = np.zeros((MH, MW), np.float32)
    ys, xs = np.mgrid[0:MH, 0:MW]
    WG = np.stack([(xs+0.5)/PX, 30.0-(ys+0.5)/PX, np.ones((MH, MW))], -1).reshape(-1, 3)
    fd = {}
    for (tt, img), r in zip(frames, results):
        if r.get("med_err_px", 999) < 15: fd[r["t"]] = img
    used = 0
    for r in good:
        t = r["t"]; H = np.array(r["H"]); img = fd.get(t)
        if img is None: continue
        Hi = np.linalg.inv(H)
        q = (Hi @ WG.T).T
        u = q[:,0]/q[:,2]*NS; w = q[:,1]/q[:,2]*NS
        ok = np.isfinite(u) & np.isfinite(w) & (u>=0) & (u<Wd-1) & (w>=0) & (w<Hd-1)
        m = ok.reshape(MH, MW)
        if m.sum() < 500: continue
        acc[m] += img[w[ok].astype(np.int32), u[ok].astype(np.int32)].astype(np.float32)
        cnt[m] += 1; used += 1
    nz = cnt > 0
    mi = np.zeros((MH, MW, 3), np.float32)
    mi[nz] = acc[nz] / cnt[nz][:, None]
    mi = mi.astype(np.uint8)
    def w2m(x, y): return (int(x*PX), int((30-y)*PX))
    for X in RED_X: cv2.line(mi, w2m(X, 0), w2m(X, 30), (0, 0, 255), 2)
    for X in BLUE_X: cv2.line(mi, w2m(X, 0), w2m(X, 30), (255, 0, 0), 2)
    for Y in [0., 30.]: cv2.line(mi, w2m(0, Y), w2m(60, Y), (0, 255, 0), 2)
    cv2.circle(mi, w2m(30, 15), int(4.5*PX), (0, 0, 255), 2)
    cov = float(nz.mean())
    cv2.putText(mi, f"GPU v2: {used} fr, cov={cov*100:.0f}%", (10, MH-12), 0, 0.65, (255,255,255), 2)
    cv2.imwrite(f"{OUT}/world_map.jpg", mi, [cv2.IMWRITE_JPEG_QUALITY, 93])
    log(f"world_map.jpg: {used} frames, coverage {cov*100:.0f}%")
else:
    log("too few usable frames for the mosaic")

log(f"DONE in {time.time()-T0:.0f}s")
