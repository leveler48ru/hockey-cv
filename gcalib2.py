#!/usr/bin/env python3
"""Global rink-marking calibration — v2 (line CORRESPONDENCES).

The previous run blew up: points on lines gave one equation each, l_w^T G r = 0,
and with only verticals/horizontals collected the y-row degenerates. Here every
detected line yields a LINE CORRESPONDENCE: world line L_w <-> pixel line l_p.
In the ref system:  l_ref = H_pan(u)^T l_p,  and  L_w x (G l_ref) = 0  =>  3 linear
equations on G per line. ~5 lines/frame x 45 frames = 225 rows, full rank.

pan-only model: pure rotation around the vertical axis.
  frame->ref: nx_r=(nx_f-t)/(1+t*nx_f), ny_r=ny_f/(1+t*nx_f), t=sgn*(u-u_ref)/fx
H_pan(u) = K R(t) K^-1,  in normalized coordinates it is parameterized by one t.

Unknowns: G (ref pixel -> world, 8 dof), fx, sgn.
Grid search over fx x sgn; for each pair collect all line correspondences via the current G
(to label the world lines), solve G via SVD, re-label — 3 iterations.
Final metric — distance of line points to the world line, in meters.

Output: /root/out/G.json, world_map.jpg, overlay_*.jpg, calib_report.txt
"""
import cv2, numpy as np, json, os, sys, time

BASE = os.environ.get("WORK", "/root")
OUT = f"{BASE}/out"; os.makedirs(OUT, exist_ok=True)
F = f"{BASE}/match.mp4"
FPS = 30.0; Wd, Hd = 1920, 1080; CX0, CY0 = 960.0, 540.0
Y0, Y1 = 330, 1078
NPT = float(os.environ.get("NPT", 45))
DUR = float(os.environ.get("DUR", 3603.0))
RED_X = [4.0, 30.0, 56.0]; BLUE_X = [22.4, 37.6]
T_REF = 3.0
T0 = time.time()
def log(*a): print(*a, flush=True)

# pan profile (coarse, for the initial u; refined later by the fx/sgn search itself)
try:
    pan = json.load(open(f"{BASE}/pan_cum.json"))
    SECS = np.asarray(pan["secs"], float); CUM = np.asarray(pan["cum"], float)
    def u_of(t): return float(np.interp(t, SECS, CUM))
except Exception:
    def u_of(t): return 0.0
UREF = u_of(T_REF)

# ---------------------------------------------------------------- 1. decode
def median_frame(cap, sec, half=1.0, n=11):
    frs = []
    for s in np.linspace(sec - half, sec + half, n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(s * FPS)))
        ok, fr = cap.read()
        if ok: frs.append(fr.astype(np.uint16))
    return np.median(np.stack(frs), 0).astype(np.uint8) if len(frs) >= 4 else None

cap = cv2.VideoCapture(F)
if not cap.isOpened(): sys.exit("cannot open video file")
times = np.linspace(2.0, DUR, int(NPT))
frames = []
for t in times:
    im = median_frame(cap, t)
    if im is not None: frames.append((float(t), im))
    if len(frames) % 8 == 0: log(f"  decode {len(frames)}/{len(times)} @ {t:.0f}s")
cap.release()
log(f"decode: {len(frames)} frames, {time.time()-T0:.0f}s")

# ------------------------------------------------- 2. static TV overlay
STACK = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for _, f in frames])
STATIC = STACK.std(0) < 6.0
WIN = np.zeros((Hd, Wd), bool); WIN[Y0:Y1] = True
GOOD = WIN & ~STATIC
log(f"static {STATIC.mean()*100:.0f}% | window {GOOD.mean()*100:.0f}%")

def segment(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    RED = (((h <= 10) | (h >= 168)) & (s > 70) & (v > 60)) & GOOD
    BLUE = ((h >= 95) & (h <= 135) & (s > 35) & (v > 50)) & GOOD
    ICE = ((s < 45) & (v > 170)) & GOOD
    return RED, BLUE, ICE

# ----------------------------------------------- 3. RANSAC -> lines (a,b,c)
def fit_line(pts):
    """least squares over points -> homogeneous line (a,b,c), a*x+b*y+c=0, unit norm
    PCA: center, smallest singular vector -> normal"""
    x, y = pts[:, 0], pts[:, 1]
    cx, cy = x.mean(), y.mean()
    M = np.stack([x - cx, y - cy], 1)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    n = Vt[-1]                      # normal (a,b)
    c = -(n[0] * cx + n[1] * cy)
    l = np.array([n[0], n[1], c])
    return l / np.linalg.norm(l[:2])

def ransac_lines(mask, orient="v", span_min=250, min_inl=200, tol=3.0, k=4, seed=0):
    ys, xs = np.nonzero(mask)
    if len(ys) < min_inl: return []
    pts = np.stack([xs, ys], 1).astype(float)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(k):
        if len(pts) < min_inl: break
        best = None
        for _it in range(250):
            i, j = rng.choice(len(pts), 2, replace=False)
            if orient == "v":
                if abs(pts[i, 1] - pts[j, 1]) < 80: continue
                a = (pts[i, 0] - pts[j, 0]) / (pts[i, 1] - pts[j, 1]); b = pts[i, 0] - a * pts[i, 1]
                res = np.abs(pts[:, 0] - (a * pts[:, 1] + b))
            else:
                if abs(pts[i, 0] - pts[j, 0]) < 80: continue
                a = (pts[i, 1] - pts[j, 1]) / (pts[i, 0] - pts[j, 0]); b = pts[i, 1] - a * pts[i, 0]
                res = np.abs(pts[:, 1] - (a * pts[:, 0] + b))
            inl = res < tol
            n_ = int(inl.sum())
            sp = (np.ptp(pts[inl, 1]) if orient == "v" else np.ptp(pts[inl, 0])) if n_ > 1 else 0.0
            sc = n_ if sp > span_min else 0
            if best is None or sc > best[0]: best = (sc, inl)
        if best is None or best[0] < min_inl: break
        _, inl = best
        line = fit_line(pts[inl])
        # re-fit: inliers by distance to the line
        d = np.abs(pts @ line[:2] + line[2])
        inl2 = d < tol
        if inl2.sum() < min_inl: break
        line = fit_line(pts[inl2])
        out.append((line, pts[inl2]))
        pts = pts[d >= tol]
    return out

def board_lines(img):
    _, _, ICE = segment(img)
    icem = cv2.morphologyEx((ICE.astype(np.uint8) * 255), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    top, bot = [], []
    for x in range(0, Wd, 3):
        col = np.nonzero(icem[:, x])[0]
        if len(col) > 150:
            top.append((x, float(col.min()))); bot.append((x, float(col.max())))
    out = []
    if len(top) > 300: out.append(("far", fit_line(np.array(top, float)), np.array(top, float)))
    if len(bot) > 300: out.append(("near", fit_line(np.array(bot, float)), np.array(bot, float)))
    return out

# ------------------------------------------ 4. pan homography (frame<->ref)
def Hpan(u, fx, sgn):
    """homography ref->frame for a pure pan: H = K R K^-1, parameterized by t
    in normalized coordinates (origin at the principal point, f=fx):
    x_f = (x_r + t)/(1 - t x_r);  inverse below. Homography in pixels:
    p_f = K @ Rp @ K^-1 p_r.  Rp(pan t): nx->(nx+t)/(1-t nx) is NOT linear...
    Honest pure-rotation homography around Y by angle theta, tg(theta)=t:
      Rp = [[cos,0,sin],[0,1,0],[-sin,0,cos]], sin/cos via t: cos=1/sqrt(1+t^2)"""
    t = sgn * (u - UREF) / fx
    c = 1.0 / np.sqrt(1.0 + t * t); s = t * c
    Rp = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    K = np.array([[fx, 0, CX0], [0, fx, CY0], [0, 0, 1.0]])
    return K @ Rp @ np.linalg.inv(K)   # ref -> frame

def frame_to_ref_homog(lp_frame, u, fx, sgn):
    """line in the frame -> line in the ref: l_ref = H^T l_frame"""
    H = Hpan(u, fx, sgn)
    return H.T @ lp_frame

# ------------------------------------------------------- 5. observations
obs = []   # (t, u, [(kind, l_frame, pts)])
for ti, (t, img) in enumerate(frames):
    R, B, _ = segment(img)
    items = []
    for mask, kind in ((R, "red"), (B, "blue")):
        for line, pts in ransac_lines(mask, "v", span_min=250, min_inl=200, seed=ti):
            items.append((kind, line, pts))
    for kind, line, pts in board_lines(img):
        items.append((kind, line, pts))
    obs.append((t, u_of(t), items))
    if (ti + 1) % 10 == 0: log(f"  observations {ti+1}/{len(frames)}")
log(f"observations: {sum(len(o[2]) for o in obs)} lines, {time.time()-T0:.0f}s")

# --------------------------------------------------- 6. GBOOT for labeling
GBOOT = np.load(f"{BASE}/H_center.npy").astype(float)   # ref pixel -> world
GBOOT = GBOOT / GBOOT[2, 2]

def apply_G(G, r):
    q = (G @ np.c_[r, np.ones(len(r))].T).T
    return q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]

def mark_world_line(G, kind, l_ref, pts_ref):
    """with the current G, determine the world line L_w for the detected line
    take inlier points, map to world, pick the nearest reference by median"""
    r = pts_ref
    if len(r) < 50: return None
    wx, wy = apply_G(G, r)
    m = np.isfinite(wx) & np.isfinite(wy)
    if m.sum() < 50: return None
    wx, wy = wx[m], wy[m]
    if kind in ("red", "blue"):
        exp = RED_X if kind == "red" else BLUE_X
        Xc = min(exp, key=lambda e: abs(e - float(np.median(wx))))
        if abs(Xc - float(np.median(wx))) > 7.0: return None
        return np.array([1.0, 0.0, -Xc])
    else:
        Yc = 30.0 if float(np.median(wy)) > 15.0 else 0.0
        if abs(Yc - float(np.median(wy))) > 9.0: return None
        return np.array([0.0, 1.0, -Yc])

def build_correspondences(G, fx, sgn):
    """returns a list of (L_w, l_ref) — line correspondences"""
    corr = []
    for t, u, items in obs:
        for kind, l_frame, pts in items:
            l_ref = frame_to_ref_homog(l_frame, u, fx, sgn)
            # normalize l_ref
            nrm = np.linalg.norm(l_ref[:2])
            if nrm < 1e-9: continue
            l_ref = l_ref / nrm
            # points in ref for labeling
            # take a subsample of inliers, map them to ref
            r_fr, ok = frame_to_ref_pts(pts, u, fx, sgn)
            Lw = mark_world_line(G, kind, l_ref, r_fr)
            if Lw is None: continue
            corr.append((Lw, l_ref, kind, r_fr))
    return corr

def frame_to_ref_pts(p, u, fx, sgn):
    tt = sgn * (u - UREF) / fx
    nx = (p[:, 0] - CX0) / fx; ny = (p[:, 1] - CY0) / fx
    den = 1.0 + tt * nx
    ok = np.abs(den) > 1e-3
    den = np.where(ok, den, 1.0)
    rr = np.stack([CX0 + fx * (nx - tt) / den, CY0 + fx * ny / den], 1)
    return rr, ok

def solve_G_from_lines(corr):
    """L_w x (G l_ref) = 0  =>  per component: 3 equations, but rank 2.
    Write as [L_w]_x G l_ref = 0.  G 3x3 is unknown (9), l_ref known, L_w known.
    Per line: M_k g = 0, where g = vec(G) (9), M_k is 3x9 of rank 2."""
    rows = []
    for Lw, lr, kind, _ in corr:
        Gl = None  # G l_ref is unknown, but L_w x (G l_ref)=0 is linear in G
        # (L_w x (G l_ref))_i = eps_ijk Lw_j (G l_ref)_k = eps_ijk Lw_j G_kl lr_l
        # => for each i: sum_{k,l} eps_ijk Lw_j lr_l G_kl = 0
        eps = np.zeros((3, 3, 3))
        eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1
        eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1
        for i in range(3):
            row = np.zeros(9)
            for k in range(3):
                for l in range(3):
                    coef = sum(eps[i, j, k] * Lw[j] * lr[l] for j in range(3))
                    row[k * 3 + l] = coef   # G_kl
            rows.append(row)
    A = np.array(rows)
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    _, sv, Vt = np.linalg.svd(A, full_matrices=False)
    G = Vt[-1].reshape(3, 3)
    return G / G[2, 2], float(sv[-2])   # sv[-2] — conditioning measure (the last one ~0 due to scale)

def metric_residuals(G, fx, sgn, corr):
    errs = []
    for Lw, lr, kind, r_fr in corr:
        if len(r_fr) < 30: continue
        sub = r_fr[::max(1, len(r_fr) // 200)]
        wx, wy = apply_G(G, sub)
        m = np.isfinite(wx) & np.isfinite(wy)
        wx, wy = wx[m], wy[m]
        if not len(wx): continue
        # distance to the world line Lw: |Lw . (wx,wy,1)| / ||Lw[:2]||
        d = np.abs(Lw[0] * wx + Lw[1] * wy + Lw[2]) / np.linalg.norm(Lw[:2])
        errs += list(d)
    return np.array(errs) if errs else np.array([1e9])

# -------------------------------------- 7. fx/sgn search + iterations
best = None
for sgn in (1.0, -1.0):
    for fx in (900, 1100, 1300, 1500, 1700, 1900, 2100, 2300, 2500):
        G = GBOOT.copy()
        ncorr = 0
        for it in range(3):
            corr = build_correspondences(G, float(fx), sgn)
            ncorr = len(corr)
            if ncorr < 60: break
            G, alg = solve_G_from_lines(corr)
        if ncorr < 60:
            log(f"  sgn={sgn:+.0f} fx={fx}: too few correspondences ({ncorr})")
            continue
        e = metric_residuals(G, float(fx), sgn, corr)
        med = float(np.median(e)); p90 = float(np.percentile(e, 90))
        log(f"  sgn={sgn:+.0f} fx={fx}: lines {ncorr}, median {med:.2f} m, p90 {p90:.2f} m")
        if best is None or med < best["med"]:
            best = {"med": med, "p90": p90, "fx": float(fx), "sgn": sgn,
                    "G": G.tolist(), "ncorr": ncorr, "alg": alg, "corr": corr}

if best is None: sys.exit("calibration did not converge")
G = np.array(best["G"])
corr = best["corr"]
log(f"BEST: fx={best['fx']:.0f} sgn={best['sgn']:+.0f} median {best['med']:.2f} m ({best['ncorr']} lines)")

res = metric_residuals(G, best["fx"], best["sgn"], corr)
report = {
    "fx": best["fx"], "sgn": best["sgn"], "n_lines": best["ncorr"],
    "median_err_m": best["med"], "p90_err_m": best["p90"],
    "max_err_m": float(np.percentile(res, 99)),
    "G_ref_to_world": G.tolist(),
    "G_world_to_ref": np.linalg.inv(G).tolist(),
    "frames": len(frames), "seconds_elapsed": time.time() - T0,
}
json.dump(report, open(f"{OUT}/G.json", "w"), indent=1)
log("G.json saved")

# ------------------------------------------- 8. world map (top-down view)
PX_M = 30
MW, MH = int(60 * PX_M), int(30 * PX_M)
Gwr = np.array(report["G_world_to_ref"])   # world -> ref pixel

ys, xs = np.mgrid[0:MH, 0:MW]
wx = (xs + 0.5) / PX_M
wy = 30.0 - (ys + 0.5) / PX_M
q = np.stack([wx.ravel(), wy.ravel(), np.ones(wx.size)], 1) @ Gwr.T
rx = q[:, 0] / q[:, 2]; ry = q[:, 1] / q[:, 2]

acc = np.zeros((wx.size, 3), np.float32); cnt = np.zeros(wx.size, np.float32)
step = max(1, len(frames) // 20)
frdict = dict(frames)
for t, u, _ in obs[::step]:
    fx, sgn = best["fx"], best["sgn"]
    tt = sgn * (u - UREF) / fx
    nx = (rx - CX0) / fx; ny = (ry - CY0) / fx
    den = 1.0 - tt * nx
    ok = (np.abs(den) > 1e-3) & np.isfinite(den)
    fxx = np.where(ok, CX0 + fx * (nx + tt) / np.where(ok, den, 1), -1)
    fyy = np.where(ok, CY0 + fx * ny / np.where(ok, den, 1), -1)
    vi = ok & (fxx >= 0) & (fxx < Wd - 1) & (fyy >= 0) & (fyy < Hd - 1)
    if not vi.any(): continue
    img = frdict.get(t)
    if img is None: continue
    sx = fxx[vi].astype(np.int32); sy = fyy[vi].astype(np.int32)
    acc[vi] += img[sy, sx].astype(np.float32); cnt[vi] += 1
mapimg = np.zeros((wx.size, 3), np.float32)
nz = cnt > 0
mapimg[nz] = acc[nz] / cnt[nz][:, None]
mapimg = mapimg.reshape(MH, MW, 3).astype(np.uint8)
holes = (~nz).reshape(MH, MW).astype(np.uint8)
if holes.any():
    mapimg = cv2.inpaint(mapimg, holes * 255, 5, cv2.INPAINT_TELEA)

def w2m(x, y): return (int(x * PX_M), int((30.0 - y) * PX_M))
for X in RED_X: cv2.line(mapimg, w2m(X, 0), w2m(X, 30), (0, 0, 255), 2)
for X in BLUE_X: cv2.line(mapimg, w2m(X, 0), w2m(X, 30), (255, 0, 0), 2)
for Y in (0.0, 30.0): cv2.line(mapimg, w2m(0, Y), w2m(60, Y), (0, 255, 0), 2)
cv2.circle(mapimg, w2m(30, 15), int(4.5 * PX_M), (0, 0, 255), 2)
for X, Y in [(6, 7.1), (6, 22.9), (54, 7.1), (54, 22.9)]:
    cv2.circle(mapimg, w2m(X, Y), int(4.5 * PX_M), (0, 0, 255), 2)
cv2.putText(mapimg, f"fx={best['fx']:.0f} sgn={best['sgn']:+.0f} med={best['med']:.2f}m n={best['ncorr']}",
            (10, MH - 12), 0, 0.6, (255, 255, 255), 2)
cv2.imwrite(f"{OUT}/world_map.jpg", mapimg, [cv2.IMWRITE_JPEG_QUALITY, 92])
log("world_map.jpg saved")

# ---------------------------------------------- 9. overlays on 3 frames
for nm, tsel in (("c", 3.0), ("l", 240.0), ("r", 2570.0)):
    img = frdict.get(tsel)
    if img is None: continue
    u = u_of(tsel); vis = img.copy()
    def draw_world_line(lw, col, thick=2):
        a, b, c = lw
        if abs(a) > 1e-9:
            ws = np.array([[-c / a, y, 1.0] for y in np.linspace(0, 30, 60)])
        else:
            ws = np.array([[x, -c / b, 1.0] for x in np.linspace(0, 60, 80)])
        q2 = ws @ Gwr.T
        rr = np.stack([q2[:, 0] / q2[:, 2], q2[:, 1] / q2[:, 2]], 1)
        ff, ok = frame_to_ref_pts(rr, u, best["fx"], best["sgn"])
        ff = ff[ok & np.all(np.isfinite(ff), 1)]
        if len(ff) < 2: return
        pts = ff.astype(np.int32)
        for i in range(len(pts) - 1):
            if 0 <= pts[i, 0] < Wd and 0 <= pts[i + 1, 0] < Wd:
                cv2.line(vis, tuple(pts[i]), tuple(pts[i + 1]), col, thick)
    for X in RED_X: draw_world_line(np.array([1.0, 0.0, -X]), (0, 0, 255), 2)
    for X in BLUE_X: draw_world_line(np.array([1.0, 0.0, -X]), (255, 0, 0), 2)
    for Y in (0.0, 30.0): draw_world_line(np.array([0.0, 1.0, -Y]), (0, 255, 0), 3)
    cv2.putText(vis, f"{nm} t={tsel:.0f}s fx={best['fx']:.0f} med={best['med']:.2f}m",
                (20, 40), 0, 0.9, (0, 0, 255), 2)
    cv2.imwrite(f"{OUT}/overlay_{nm}.jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
log("overlays saved")

lines_txt = [
    f"Global calibration v2 (line correspondences) — {time.strftime('%Y-%m-%d %H:%M')}",
    f"median frames: {len(frames)} across the video (0..{DUR:.0f} s)",
    f"fx = {best['fx']:.0f} px, pan sign = {best['sgn']:+.0f}",
    f"line correspondences: {best['ncorr']}",
    f"error: median {best['med']:.2f} m, p90 {best['p90']:.2f} m, p99 {report['max_err_m']:.2f} m",
    f"elapsed: {report['seconds_elapsed']:.0f} s",
    "", "G (ref pixel -> world):",
]
for row in G: lines_txt.append("  " + " ".join(f"{v:+.6f}" for v in row))
open(f"{OUT}/calib_report.txt", "w").write("\n".join(lines_txt) + "\n")
log("\n".join(lines_txt)); log("DONE")
