#!/usr/bin/env python3
"""Global calibration of the hockey rink on a GPU VM.

Adaptive segmentation + topological line labeling + yaw-only model.
Wide-shot filter by ice area. Mosaic from all usable frames.

Output: /root/out/world_map.jpg, /root/out/yaw_profile.json, /root/out/overlays/
"""
import cv2, numpy as np, json, os, sys, time, glob
from scipy.optimize import minimize_scalar

BASE = os.environ.get("WORK", "/root")
OUT = f"{BASE}/out"; os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/overlays", exist_ok=True)
F = f"{BASE}/match.mp4"
FPS = 30.0; Wd, Hd = 1920, 1080
NS, NM = 1000.0, 30.0
STEP_SEC = float(os.environ.get("STEP", 5.0))  # every 5 seconds
ICE_MIN_FRAC = float(os.environ.get("ICE_MIN", 0.15))  # at least 15% ice = wide shot
RED_X = [4.0, 30.0, 56.0]; BLUE_X = [22.4, 37.6]
ALL_VX = sorted(BLUE_X + RED_X)  # [4, 22.4, 30, 37.6, 56]
T0 = time.time()
def log(*a): print(*a, flush=True)

# ============================================================ 1. H_A from anchors.json
anc_path = f"{BASE}/anchors.json"
if not os.path.exists(anc_path):
    # fallback: hardcoded from final calibration
    H_A_list = [[0.0583, -0.0012, 0.247], [0.0008, 0.0412, 0.012], [0.000031, 0.00018, 1.0]]
    H_A = np.array(H_A_list)
    a_val, b_val = 5.80, 8.60
else:
    anc = json.load(open(anc_path))
    m1 = np.array(anc.get('G_world2pix', anc.get('H', [])))
    m2 = np.array(anc.get('H_pix2world', anc.get('Hi', [])))
    def is_p2w(m):
        if m.size == 0: return False
        v = m @ np.array([690/NS, 370/NS, 1.0]); w = v[:2]/v[2]
        return abs(w[0]*NM - 30) < 2 and abs(w[1]*NM - 30) < 2
    P2W = m1 if is_p2w(m1) else (m2 if is_p2w(m2) else None)
    if P2W is None: sys.exit("cannot find pixel->world in anchors.json")
    W2P_n = np.linalg.inv(P2W)
    S_winv = np.diag([1/NM, 1/NM, 1.0])
    S_pix = np.diag([NS, NS, 1.0])
    H_A = S_pix @ W2P_n @ S_winv
    a_val = anc.get('a', 5.80); b_val = anc.get('b', 8.60)

# Sanity check
v = H_A @ np.array([30, 30, 1.]); v /= v[2]
log(f"H_A: world(30,30)->px({v[0]:.0f},{v[1]:.0f}), a={a_val}, b={b_val}")

# ============================================================ 2. Frame decoding
cap = cv2.VideoCapture(F)
if not cap.isOpened(): sys.exit("no video")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / FPS
log(f"video: {total_frames} frames, {duration:.0f} s")

times = np.arange(0, duration, STEP_SEC)
frames = []
for t in times:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * FPS))
    ok, fr = cap.read()
    if ok: frames.append((float(t), fr))
    if len(frames) % 100 == 0: log(f"  decode {len(frames)}/{len(times)}")
cap.release()
log(f"decoded {len(frames)} frames in {time.time()-T0:.0f}s")

# ============================================================ 3. Adaptive segmentation
def segment_ice(img):
    """Finds the ice (bright region) and returns mask + area fraction."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold: Otsu on the lower part of the frame (where the ice is)
    roi = gray[Hd//3:, :]
    thr_val, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(gray, max(int(thr_val) - 20, 100), 255, cv2.THRESH_BINARY)
    # Morphology: remove noise, fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Remove the top third (stands/ceiling)
    mask[:Hd//3, :] = 0
    frac = mask.sum() / (Wd * Hd)
    return mask, frac

def segment_lines(img, ice_mask):
    """Finds red and blue lines inside the ice mask."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    Hh, Ss, Vv = hsv[...,0], hsv[...,1], hsv[...,2]
    
    # Red: wide range (brightness may vary)
    RED = (((Hh <= 12) | (Hh >= 165)) & (Ss > 50) & (Vv > 40)) & ice_mask
    # Blue
    BLUE = ((Hh >= 90) & (Hh <= 140) & (Ss > 25) & (Vv > 40)) & ice_mask
    
    # Morphology: join gaps
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    RED = cv2.morphologyEx(RED.astype(np.uint8)*255, cv2.MORPH_CLOSE, k) > 0
    BLUE = cv2.morphologyEx(BLUE.astype(np.uint8)*255, cv2.MORPH_CLOSE, k) > 0
    
    return RED.astype(bool), BLUE.astype(bool)

def find_vertical_lines(mask, orient='v', min_len=150, tol=4.0):
    """RANSAC for vertical lines. Returns a list of (line, inlier_pts)."""
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
            sp = np.ptp(pts[inl, 1]) if n_ > 1 else 0
            sc = n_ if sp > min_len else 0
            if best is None or sc > best[0]: best = (sc, inl, a, b)
        if best is None or best[0] < 80: break
        _, inl, a, b = best
        line_pts = pts[inl]
        # Fit a line
        cx, cy = line_pts[:,0].mean(), line_pts[:,1].mean()
        M = np.column_stack([line_pts[:,0]-cx, line_pts[:,1]-cy])
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        n = Vt[-1]
        l = np.array([n[0], n[1], -(n[0]*cx + n[1]*cy)])
        l = l / np.linalg.norm(l[:2])
        lines.append((l, line_pts))
        pts = pts[np.abs(pts @ l[:2] + l[2]) >= tol]
    return lines

def x_at_y(line, y):
    """x coordinate of the line at a given y."""
    a, b, c = line
    if abs(a) < 1e-9: return None
    return (-b * y - c) / a

# ============================================================ 4. Topological line labeling
def label_lines(vlines, img_width=Wd):
    """Labels vertical lines in left-to-right order.
    Returns [(line, world_x)] or None if labeling failed."""
    if len(vlines) < 3: return None
    
    # Sort by x at mid-height of the frame
    mid_y = Hd // 2
    scored = []
    for l, pts in vlines:
        x = x_at_y(l, mid_y)
        if x is not None and 0 < x < img_width:
            scored.append((x, l, pts))
    
    if len(scored) < 3: return None
    scored.sort(key=lambda s: s[0])
    
    # Try to match against ALL_VX = [4, 22.4, 30, 37.6, 56]
    # Take an order-preserving subset of ALL_VX of the same size
    n = len(scored)
    if n > len(ALL_VX): return None
    
    best_match = None; best_err = 1e9
    for start in range(len(ALL_VX) - n + 1):
        subset = ALL_VX[start:start+n]
        # Check: relative distances should look similar
        pix_xs = [s[0] for s in scored]
        pix_span = pix_xs[-1] - pix_xs[0]
        world_span = subset[-1] - subset[0]
        if pix_span < 50 or world_span < 1: continue
        
        scale = pix_span / world_span
        err = sum(abs(pix_xs[i] - (subset[i] - subset[0]) * scale - pix_xs[0]) for i in range(n))
        if err < best_err:
            best_err = err; best_match = list(zip(subset, scored))
    
    if best_match is None: return None
    return [(s[1][1], s[1][2], s[0]) for s in best_match]  # (line, pts, world_x)

# ============================================================ 5. Homography from lines + boards
def solve_H_from_lines(labeled_lines, board_top=None, board_bot=None):
    """Solves H (world->pixel) from line correspondences."""
    corrs = []
    for line, pts, wx in labeled_lines:
        Lw = np.array([1.0, 0.0, -wx])
        corrs.append((Lw, line, pts))
    
    if board_top is not None:
        corrs.append((np.array([0.0, 1.0, -30.0]), board_top[0], board_top[1]))
    if board_bot is not None:
        corrs.append((np.array([0.0, 1.0, -0.0]), board_bot[0], board_bot[1]))
    
    if len(corrs) < 4: return None
    
    # DLT for lines: l_pix ~ H^T L_w
    rows = []
    for Lw, lp, pts in corrs:
        for i in range(3):
            r = np.zeros(9)
            for j in range(3):
                r[j*3+i] = Lw[j] * lp[i]
            # cross product constraint: (H^T L) x l = 0
            # => eps_{ijk} (H^T L)_j l_k = 0
            pass
        # Simpler: use points on the lines
        sub = pts[::max(1, len(pts)//50)]
        for px, py in sub:
            # point (px,py) lies on world line Lw: Lw^T w = 0
            # w = H^-1 [px,py,1]^T => Lw^T H^-1 [px,py,1] = 0
            # This is nonlinear. Use a different approach:
            # For a vertical line x=X: all points have world x=X
            # => (H^-1 [px,py,1])[0] / (H^-1 [px,py,1])[2] = X
            # => h00*px + h01*py + h02 = X * (h20*px + h21*py + h22)
            rows.append([px, py, 1, 0, 0, 0, -wx*px, -wx*py, -wx])
    
    if len(rows) < 8: return None
    A = np.array(rows)
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    _, sv, Vt = np.linalg.svd(A, full_matrices=False)
    h = Vt[-1]
    H = h.reshape(3, 3)
    if abs(H[2,2]) < 1e-9: return None
    return H / H[2,2]

# ============================================================ 6. Yaw-only refinement
def yaw_residual(theta, H_base, img, ice_mask):
    """Residual when rotating the camera by theta."""
    CX, CY = 960.0, 540.0
    f_est = 2200.0  # approximate focal length
    K = np.array([[f_est, 0, CX], [0, f_est, CY], [0, 0, 1.0]])
    c, s = np.cos(theta), np.sin(theta)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    M = K @ Ry @ np.linalg.inv(K)
    H_t = M @ H_base
    
    # Project the world lines
    Hi = np.linalg.inv(H_t)
    dt_blue = cv2.distanceTransform((~segment_lines(img, ice_mask)[1]).astype(np.uint8), cv2.DIST_L2, 3)
    
    total = 0; count = 0
    for X in BLUE_X:
        for y in np.linspace(1, 29, 40):
            v = Hi @ np.array([X/NM, y/NM, 1.0])
            px = v[0]/v[2]*NS; py = v[1]/v[2]*NS
            if 0 <= px < Wd and 330 <= py < 1078:
                total += dt_blue[int(py), int(px)]
                count += 1
    return total / max(count, 1) if count > 0 else 1e9

# ============================================================ 7. Main loop
results = []
wide_count = 0
for idx, (t, img) in enumerate(frames):
    ice_mask, ice_frac = segment_ice(img)
    
    if ice_frac < ICE_MIN_FRAC:
        results.append({"t": t, "wide": False, "ice_frac": float(ice_frac)})
        continue
    
    wide_count += 1
    RED, BLUE = segment_lines(img, ice_mask)
    
    # Find vertical lines
    red_lines = find_vertical_lines(RED)
    blue_lines = find_vertical_lines(BLUE)
    all_vlines = red_lines + blue_lines
    
    # Labeling
    labeled = label_lines(all_vlines)
    
    if labeled is None or len(labeled) < 3:
        results.append({"t": t, "wide": True, "ice_frac": float(ice_frac), 
                        "lines_found": len(all_vlines), "status": "no_label"})
        continue
    
    # Solve H
    H = solve_H_from_lines(labeled)
    if H is None:
        results.append({"t": t, "wide": True, "ice_frac": float(ice_frac),
                        "lines_found": len(all_vlines), "labeled": len(labeled), "status": "no_H"})
        continue
    
    # Residual
    Hi = np.linalg.inv(H)
    errs = []
    for line, pts, wx in labeled:
        sub = pts[::max(1, len(pts)//100)]
        for px, py in sub:
            v = H @ np.array([wx/NM, 0, 1.0])  # project the world line
            # Distance from the point to the projected line
            l_proj = Hi.T @ np.array([1.0, 0.0, -wx/NM])
            n = np.hypot(l_proj[0], l_proj[1])
            if n > 1e-9:
                d = abs(l_proj[0]*px/NS + l_proj[1]*py/NS + l_proj[2]) / n * NS
                errs.append(d)
    
    med_err = float(np.median(errs)) if errs else 999.0
    
    results.append({
        "t": t, "wide": True, "ice_frac": float(ice_frac),
        "lines_found": len(all_vlines), "labeled": len(labeled),
        "med_err_px": med_err, "H": H.tolist(),
        "world_xs": [wx for _, _, wx in labeled]
    })
    
    if (idx+1) % 50 == 0:
        log(f"  processed {idx+1}/{len(frames)}, wide shots {wide_count}, last residual {med_err:.1f} px")

log(f"\nprocessing complete: {len(frames)} frames, {wide_count} wide shots, {time.time()-T0:.0f}s")

# ============================================================ 8. Statistics
good = [r for r in results if r.get("med_err_px", 999) < 15]
log(f"usable frames (residual <15 px): {len(good)}/{wide_count}")
if good:
    errs = [r["med_err_px"] for r in good]
    log(f"  median residual: {np.median(errs):.1f} px, p90: {np.percentile(errs, 90):.1f} px")

# Save the profile
json.dump(results, open(f"{OUT}/yaw_profile.json", "w"), indent=1)
log(f"saved {OUT}/yaw_profile.json")

# ============================================================ 9. Mosaic from usable frames
if len(good) >= 3:
    PX = 30; MW, MH = 60*PX, 30*PX
    acc = np.zeros((MH, MW, 3), np.float32)
    cnt = np.zeros((MH, MW), np.float32)
    ys, xs = np.mgrid[0:MH, 0:MW]
    WG = np.stack([(xs+0.5)/PX, 30.0-(ys+0.5)/PX, np.ones((MH, MW))], -1).reshape(-1, 3)
    
    fd = {r["t"]: img for (t, img), r in zip(frames, results) if r.get("med_err_px", 999) < 15}
    used = 0
    for r in good:
        t = r["t"]; H = np.array(r["H"])
        img = fd.get(t)
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
    
    # Markings
    def w2m(x, y): return (int(x*PX), int((30-y)*PX))
    for X in RED_X: cv2.line(mi, w2m(X, 0), w2m(X, 30), (0, 0, 255), 2)
    for X in BLUE_X: cv2.line(mi, w2m(X, 0), w2m(X, 30), (255, 0, 0), 2)
    for Y in [0., 30.]: cv2.line(mi, w2m(0, Y), w2m(60, Y), (0, 255, 0), 2)
    cv2.circle(mi, w2m(30, 15), int(4.5*PX), (0, 0, 255), 2)
    cov = nz.mean()
    cv2.putText(mi, f"GPU calib: {used} frames, cov={cov*100:.0f}%", (10, MH-12), 0, 0.65, (255,255,255), 2)
    cv2.imwrite(f"{OUT}/world_map.jpg", mi, [cv2.IMWRITE_JPEG_QUALITY, 93])
    log(f"world_map.jpg: {used} frames, coverage {cov*100:.0f}%")
else:
    log("too few usable frames for the mosaic")

log(f"DONE in {time.time()-T0:.0f}s")
