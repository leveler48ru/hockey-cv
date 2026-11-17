#!/usr/bin/env python3
"""Global calibration v3 — independent homography per frame.

No reliance on the pan profile at all. For each of N median frames:
1. RANSAC finds red/blue lines + boards
2. Label each line with the nearest world line (x=4/22.4/30/37.6/56, y=0/30)
3. Solve the full homography H_k (8 dof) from line correspondences (3 equations per line)
4. Recover the pan angle theta_k and a shared intrinsic matrix K from H_k

Final global calibration: K + {theta_k} -> a single H for any frame.

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
T0 = time.time()
def log(*a): print(*a, flush=True)

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

# ----------------------------------------------- 3. RANSAC -> lines
def fit_line(pts):
    x, y = pts[:, 0], pts[:, 1]
    cx, cy = x.mean(), y.mean()
    M = np.stack([x - cx, y - cy], 1)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    n = Vt[-1]
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

# ------------------------------------------------------- 4. observations
obs = []
for ti, (t, img) in enumerate(frames):
    R, B, _ = segment(img)
    items = []
    for mask, kind in ((R, "red"), (B, "blue")):
        for line, pts in ransac_lines(mask, "v", span_min=250, min_inl=200, seed=ti):
            items.append((kind, line, pts))
    for kind, line, pts in board_lines(img):
        items.append((kind, line, pts))
    obs.append((t, items))
    if (ti + 1) % 10 == 0: log(f"  observations {ti+1}/{len(frames)}")
log(f"observations: {sum(len(o[1]) for o in obs)} lines, {time.time()-T0:.0f}s")

# ------------------------------------------ 5. independent H per frame
def solve_H_from_lines(items):
    """For one frame: label the lines -> solve H (pixel->world) from line correspondences.
    L_w x (H l_p) = 0 => 3 linear equations per line, rank 2."""
    # First label every line
    marked = []
    for kind, lp, pts in items:
        # rough estimate: take the pixel median and map via H_center (bootstrap),
        # but H_center may be off — use a heuristic by position in the frame
        cx_line = pts[:, 0].mean()
        cy_line = pts[:, 1].mean()
        
        if kind in ("red", "blue"):
            # vertical line: world x = const
            # rough: left part of the frame -> small x, right -> large
            # normalize cx_line into [0, 1]
            frac = cx_line / Wd
            if kind == "red":
                exp = RED_X
            else:
                exp = BLUE_X
            # pick the nearest by frac
            # red: 4/60=0.067, 30/60=0.5, 56/60=0.933
            # blue: 22.4/60=0.373, 37.6/60=0.627
            fracs = [e / 60.0 for e in exp]
            idx = min(range(len(fracs)), key=lambda i: abs(fracs[i] - frac))
            Xc = exp[idx]
            Lw = np.array([1.0, 0.0, -Xc])
        else:
            # horizontal line: world y = const
            frac_y = cy_line / Hd
            Yc = 30.0 if frac_y < 0.5 else 0.0  # top of frame = far board = y=30
            Lw = np.array([0.0, 1.0, -Yc])
        
        marked.append((Lw, lp, kind, pts))
    
    if len(marked) < 4:
        return None, []
    
    # Solve H from line correspondences
    rows = []
    for Lw, lp, kind, pts in marked:
        eps = np.zeros((3, 3, 3))
        eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1
        eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1
        for i in range(3):
            row = np.zeros(9)
            for k in range(3):
                for l in range(3):
                    coef = sum(eps[i, j, k] * Lw[j] * lp[l] for j in range(3))
                    row[k * 3 + l] = coef
            rows.append(row)
    
    A = np.array(rows)
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    _, sv, Vt = np.linalg.svd(A, full_matrices=False)
    H = Vt[-1].reshape(3, 3)
    H = H / H[2, 2]
    
    # Check: H must be non-degenerate
    if abs(np.linalg.det(H)) < 1e-10:
        return None, []
    
    return H, marked

def apply_H(H, p):
    q = (H @ np.c_[p, np.ones(len(p))].T).T
    return q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]

def metric_residuals_frame(H, marked):
    errs = []
    for Lw, lp, kind, pts in marked:
        if len(pts) < 30: continue
        sub = pts[::max(1, len(pts) // 200)]
        wx, wy = apply_H(H, sub)
        m = np.isfinite(wx) & np.isfinite(wy)
        wx, wy = wx[m], wy[m]
        if not len(wx): continue
        d = np.abs(Lw[0] * wx + Lw[1] * wy + Lw[2]) / np.linalg.norm(Lw[:2])
        errs += list(d)
    return np.array(errs) if errs else np.array([1e9])

# Solve H for every frame
Hs = []
all_marked = []
for ti, (t, items) in enumerate(obs):
    H, marked = solve_H_from_lines(items)
    if H is not None:
        e = metric_residuals_frame(H, marked)
        med = float(np.median(e))
        Hs.append((t, H, med, len(marked)))
        all_marked.append((t, marked))
        if (ti + 1) % 10 == 0:
            log(f"  frame {ti+1}/{len(obs)} @ {t:.0f}s: {len(marked)} lines, median {med:.2f} m")
    else:
        log(f"  frame {ti+1}/{len(obs)} @ {t:.0f}s: UNSOLVED ({len(items)} lines)")

log(f"solved {len(Hs)}/{len(obs)} frames, {time.time()-T0:.0f}s")

if len(Hs) < 10:
    sys.exit(f"too few solved frames: {len(Hs)}")

# ------------------------------------------ 6. global calibration
# Average H over all frames (element-wise median)
H_stack = np.stack([h for _, h, _, _ in Hs])
H_med = np.median(H_stack, axis=0)
H_med = H_med / H_med[2, 2]

# Metric over all frames
all_errs = []
for t, marked in all_marked:
    e = metric_residuals_frame(H_med, marked)
    all_errs += list(e)
all_errs = np.array(all_errs)
med_err = float(np.median(all_errs))
p90_err = float(np.percentile(all_errs, 90))
p99_err = float(np.percentile(all_errs, 99))

log(f"global H: median {med_err:.2f} m, p90 {p90_err:.2f} m, p99 {p99_err:.2f} m")

# Recover pan angles from per-frame H
# H_k = H_global @ R_pan(theta_k)  =>  R_pan = H_global^-1 @ H_k
# For a pure pan around Y: R = [[cos,0,sin],[0,1,0],[-sin,0,cos]]
# Extract theta from H_global^-1 @ H_k
H_inv = np.linalg.inv(H_med)
thetas = []
for t, Hk, _, _ in Hs:
    M = H_inv @ Hk
    # M should be close to R_pan (up to scale)
    # normalize
    M = M / M[2, 2]
    # extract theta from M[0,0] and M[0,2]
    # R_pan = [[c,0,s],[0,1,0],[-s,0,c]]
    # M[0,0] ~ c, M[0,2] ~ s
    c_est = M[0, 0]; s_est = M[0, 2]
    theta = np.arctan2(s_est, c_est)
    thetas.append((t, float(theta)))

thetas_arr = np.array([th for _, th in thetas])
log(f"pan angles: min={np.degrees(thetas_arr.min()):.1f}° max={np.degrees(thetas_arr.max()):.1f}° span={np.degrees(thetas_arr.ptp()):.1f}°")

# Save the result
report = {
    "n_frames": len(Hs),
    "n_lines_total": sum(len(m) for _, m in all_marked),
    "median_err_m": med_err,
    "p90_err_m": p90_err,
    "p99_err_m": p99_err,
    "H_ref_to_world": H_med.tolist(),
    "H_world_to_ref": np.linalg.inv(H_med).tolist(),
    "pan_angles_deg": [(t, float(np.degrees(th))) for t, th in thetas],
    "pan_span_deg": float(np.degrees(thetas_arr.ptp())),
    "seconds_elapsed": time.time() - T0,
}
json.dump(report, open(f"{OUT}/G.json", "w"), indent=1)
log("G.json saved")

# ------------------------------------------- 7. world map (top-down view)
PX_M = 30
MW, MH = int(60 * PX_M), int(30 * PX_M)
Hwr = np.array(report["H_world_to_ref"])

ys, xs = np.mgrid[0:MH, 0:MW]
wx = (xs + 0.5) / PX_M
wy = 30.0 - (ys + 0.5) / PX_M
q = np.stack([wx.ravel(), wy.ravel(), np.ones(wx.size)], 1) @ Hwr.T
rx = q[:, 0] / q[:, 2]; ry = q[:, 1] / q[:, 2]

acc = np.zeros((wx.size, 3), np.float32); cnt = np.zeros(wx.size, np.float32)
frdict = dict(frames)
step = max(1, len(frames) // 20)
for t, items in obs[::step]:
    vi = (rx >= 0) & (rx < Wd - 1) & (ry >= 0) & (ry < Hd - 1) & np.isfinite(rx) & np.isfinite(ry)
    if not vi.any(): continue
    img = frdict.get(t)
    if img is None: continue
    sx = rx[vi].astype(np.int32); sy = ry[vi].astype(np.int32)
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
cv2.putText(mapimg, f"med={med_err:.2f}m n={report['n_lines_total']} pan={report['pan_span_deg']:.1f}deg",
            (10, MH - 12), 0, 0.6, (255, 255, 255), 2)
cv2.imwrite(f"{OUT}/world_map.jpg", mapimg, [cv2.IMWRITE_JPEG_QUALITY, 92])
log("world_map.jpg saved")

# ---------------------------------------------- 8. overlays on 3 frames
for nm, tsel in (("c", 3.0), ("l", 240.0), ("r", 2570.0)):
    img = frdict.get(tsel)
    if img is None: continue
    vis = img.copy()
    def draw_world_line(lw, col, thick=2):
        a, b, c = lw
        if abs(a) > 1e-9:
            ws = np.array([[-c / a, y, 1.0] for y in np.linspace(0, 30, 60)])
        else:
            ws = np.array([[x, -c / b, 1.0] for x in np.linspace(0, 60, 80)])
        q2 = ws @ Hwr.T
        rr = np.stack([q2[:, 0] / q2[:, 2], q2[:, 1] / q2[:, 2]], 1)
        pts = rr.astype(np.int32)
        for i in range(len(pts) - 1):
            if 0 <= pts[i, 0] < Wd and 0 <= pts[i + 1, 0] < Wd:
                cv2.line(vis, tuple(pts[i]), tuple(pts[i + 1]), col, thick)
    for X in RED_X: draw_world_line(np.array([1.0, 0.0, -X]), (0, 0, 255), 2)
    for X in BLUE_X: draw_world_line(np.array([1.0, 0.0, -X]), (255, 0, 0), 2)
    for Y in (0.0, 30.0): draw_world_line(np.array([0.0, 1.0, -Y]), (0, 255, 0), 3)
    cv2.putText(vis, f"{nm} t={tsel:.0f}s med={med_err:.2f}m",
                (20, 40), 0, 0.9, (0, 0, 255), 2)
    cv2.imwrite(f"{OUT}/overlay_{nm}.jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
log("overlays saved")

lines_txt = [
    f"Global calibration v3 (independent H per frame) — {time.strftime('%Y-%m-%d %H:%M')}",
    f"median frames: {len(frames)} across the video (0..{DUR:.0f} s)",
    f"solved frames: {len(Hs)}/{len(obs)}",
    f"lines total: {report['n_lines_total']}",
    f"error: median {med_err:.2f} m, p90 {p90_err:.2f} m, p99 {p99_err:.2f} m",
    f"pan span: {report['pan_span_deg']:.1f}°",
    f"elapsed: {report['seconds_elapsed']:.0f} s",
    "", "H (ref pixel -> world):",
]
for row in H_med: lines_txt.append("  " + " ".join(f"{v:+.6f}" for v in row))
open(f"{OUT}/calib_report.txt", "w").write("\n".join(lines_txt) + "\n")
log("\n".join(lines_txt)); log("DONE")
