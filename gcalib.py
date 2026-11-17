#!/usr/bin/env python3
"""Global rink-marking calibration over the ENTIRE video (a cloud GPU (L4)).

Input: /root/match.mp4, /root/pan_cum.json (panning profile),
       /root/H_center.npy (coarse bootstrap: reference-frame pixel -> world).
Output: /root/out/G.json, world_map.jpg (top-down 60x30 view from the whole video),
        calib_report.txt, overlay_*.jpg

Model: the camera only rotates (pan) around the vertical axis => pure rotation.
  frame->ref:  nx_r = (nx_f - t)/(1 + t*nx_f),  ny_r = ny_f/(1 + t*nx_f)
               t = sgn*(u - u_ref)/fx,  nx=(x-CX0)/fx, ny=(y-CY0)/fx
Unknowns: G (world->ref, 8 dof), fx, sgn.
Observations: marking lines (red x=4/30/56, blue x=22.4/37.6, boards y=0/30)
              extracted with RANSAC from ~45 median frames across the video.
Equation per point: l_w^T G r = 0 (homogeneous, linear in G).
"""
import cv2, numpy as np, json, os, sys, time

BASE = os.environ.get("WORK", "/root")
OUT = f"{BASE}/out"
os.makedirs(OUT, exist_ok=True)
F = f"{BASE}/match.mp4"
FPS = 30.0
Wd, Hd = 1920, 1080
CX0, CY0 = 960.0, 540.0
Y0, Y1 = 330, 1078
NPT = float(os.environ.get("NPT", 45))
T_REF = float(os.environ.get("T_REF", 3.0))
DUR = float(os.environ.get("DUR", 3603.0))
RED_X = [4.0, 30.0, 56.0]
BLUE_X = [22.4, 37.6]
T0 = time.time()

def log(*a):
    print(*a, flush=True)

pan = json.load(open(f"{BASE}/pan_cum.json"))
SECS = np.asarray(pan["secs"], float); CUM = np.asarray(pan["cum"], float)
def u_of(t): return float(np.interp(t, SECS, CUM))
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
log(f"static {STATIC.mean()*100:.0f}% | working window {GOOD.mean()*100:.0f}%")

def segment(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    RED = (((h <= 10) | (h >= 168)) & (s > 70) & (v > 60)) & GOOD
    BLUE = ((h >= 95) & (h <= 135) & (s > 35) & (v > 50)) & GOOD
    ICE = ((s < 45) & (v > 170)) & GOOD
    return RED, BLUE, ICE

# ------------------------------------------------------ 3. RANSAC on lines
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
                a = (pts[i, 0] - pts[j, 0]) / (pts[i, 1] - pts[j, 1])
                b = pts[i, 0] - a * pts[i, 1]
                res = np.abs(pts[:, 0] - (a * pts[:, 1] + b))
            else:
                if abs(pts[i, 0] - pts[j, 0]) < 80: continue
                a = (pts[i, 1] - pts[j, 1]) / (pts[i, 0] - pts[j, 0])
                b = pts[i, 1] - a * pts[i, 0]
                res = np.abs(pts[:, 1] - (a * pts[:, 0] + b))
            inl = res < tol
            n_ = int(inl.sum())
            sp = (np.ptp(pts[inl, 1]) if orient == "v" else np.ptp(pts[inl, 0])) if n_ > 1 else 0.0
            sc = n_ if sp > span_min else 0
            if best is None or sc > best[0]: best = (sc, a, b, inl)
        if best is None or best[0] < min_inl: break
        _, a, b, inl = best
        dep = pts[inl, 1] if orient == "v" else pts[inl, 0]
        tgt = pts[inl, 0] if orient == "v" else pts[inl, 1]
        A = np.stack([dep, np.ones(len(dep))], 1)
        cf, *_ = np.linalg.lstsq(A, tgt, rcond=None)
        a, b = float(cf[0]), float(cf[1])
        # re-fit with found coefficients over ALL points, then a new mask
        fdep = pts[:, 1] if orient == "v" else pts[:, 0]
        ftgt = pts[:, 0] if orient == "v" else pts[:, 1]
        inl = np.abs(ftgt - (a * fdep + b)) < tol
        out.append(pts[inl])
        pts = pts[~inl]
    return out

def board_points(img):
    _, _, ICE = segment(img)
    icem = cv2.morphologyEx((ICE.astype(np.uint8) * 255), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    top, bot = [], []
    for x in range(0, Wd, 3):
        col = np.nonzero(icem[:, x])[0]
        if len(col) > 150:
            top.append((x, float(col.min()))); bot.append((x, float(col.max())))
    return np.array(top, float), np.array(bot, float)

# ------------------------------------------------------- 4. observations
obs = []
for ti, (t, img) in enumerate(frames):
    R, B, _ = segment(img)
    items = []
    for mask, kind in ((R, "red"), (B, "blue")):
        for p in ransac_lines(mask, "v", span_min=250, min_inl=200, seed=ti):
            items.append((kind, p))
    top, bot = board_points(img)
    if len(top) > 300: items.append(("far", top))
    if len(bot) > 300: items.append(("near", bot))
    obs.append((t, u_of(t), items))
    if (ti + 1) % 10 == 0: log(f"  observations {ti+1}/{len(frames)}")
log(f"observations: {sum(len(o[2]) for o in obs)} lines, {time.time()-T0:.0f}s")

# ------------------------------------------- 5. frame <-> ref (pan-only)
def frame_to_ref(p, u, fx, sgn):
    tt = sgn * (u - UREF) / fx
    nx = (p[:, 0] - CX0) / fx; ny = (p[:, 1] - CY0) / fx
    den = 1.0 + tt * nx
    ok = np.abs(den) > 1e-3
    den = np.where(ok, den, 1.0)
    return np.stack([CX0 + fx * (nx - tt) / den, CY0 + fx * ny / den], 1), ok

def ref_to_frame(p, u, fx, sgn):
    tt = sgn * (u - UREF) / fx
    nx = (p[:, 0] - CX0) / fx; ny = (p[:, 1] - CY0) / fx
    den = 1.0 - tt * nx
    ok = np.abs(den) > 1e-3
    den = np.where(ok, den, 1.0)
    return np.stack([CX0 + fx * (nx + tt) / den, CY0 + fx * ny / den], 1), ok

GBOOT = np.load(f"{BASE}/H_center.npy").astype(float)   # ref pixel -> world (coarse)
GBOOT = GBOOT / GBOOT[2, 2]
# IMPORTANT: the solver finds G from l_w^T G r = 0, i.e. G is PIXEL->WORLD.
# So GBOOT (H_center) is taken as is, without inversion.

def apply_G(G, r):
    q = (G @ np.c_[r, np.ones(len(r))].T).T
    return q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]

def build_rows(G, fx, sgn, max_per_line=100):
    """assign each detected line to the nearest world line"""
    rows, lines_used = [], []
    for ti, (t, u, items) in enumerate(obs):
        for kind, pts in items:
            r, ok = frame_to_ref(pts, u, fx, sgn)
            r = r[ok]
            if len(r) < 80: continue
            wx, wy = apply_G(G, r)
            fin = np.isfinite(wx) & np.isfinite(wy)
            if fin.sum() < 80: continue
            wx, wy, r = wx[fin], wy[fin], r[fin]
            if kind in ("red", "blue"):
                exp = RED_X if kind == "red" else BLUE_X
                Xc = min(exp, key=lambda e: abs(e - float(np.median(wx))))
                if abs(Xc - float(np.median(wx))) > 6.0: continue
                lw = np.array([1.0, 0.0, -Xc])
            else:
                Yc = 30.0 if float(np.median(wy)) > 15.0 else 0.0
                if abs(Yc - float(np.median(wy))) > 8.0: continue
                lw = np.array([0.0, 1.0, -Yc])
            idx = np.random.default_rng(ti * 97 + len(rows)).choice(
                len(r), size=min(max_per_line, len(r)), replace=False)
            for ii in idx:
                rows.append((lw, r[ii]))
            lines_used.append((kind, float(Xc) if kind in ("red", "blue") else float(Yc)))
    return rows, lines_used

def solve_G(rows):
    A = np.empty((len(rows), 9))
    for i, (lw, r) in enumerate(rows):
        A[i] = np.outer(lw, np.array([r[0], r[1], 1.0])).ravel()
    # row scale normalization (otherwise pixel weights dominate)
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    _, sv, Vt = np.linalg.svd(A, full_matrices=False)
    G = Vt[-1].reshape(3, 3)
    return G / G[2, 2], float(sv[-1])

def metric_residuals(G, fx, sgn):
    errs = []
    RED = np.array(RED_X); BLU = np.array(BLUE_X)
    for t, u, items in obs:
        for kind, pts in items:
            if len(pts) > 300:
                pts = pts[:: max(1, len(pts) // 300)]
            r, ok = frame_to_ref(pts, u, fx, sgn); r = r[ok]
            if len(r) < 80: continue
            wx, wy = apply_G(G, r)
            m = np.isfinite(wx) & np.isfinite(wy)
            wx, wy = wx[m], wy[m]
            if not len(wx): continue
            if kind == "red":
                errs += list(np.min(np.abs(wx[:, None] - RED[None, :]), 1))
            elif kind == "blue":
                errs += list(np.min(np.abs(wx[:, None] - BLU[None, :]), 1))
            else:
                Yc = 30.0 if np.median(wy) > 15 else 0.0
                errs += list(np.abs(wy - Yc))
    return np.array(errs) if errs else np.array([1e9])

# -------------------------------------- 6. fx/sgn search + iterations
best = None
for sgn in (1.0, -1.0):
    for fx in (900, 1100, 1300, 1500, 1700, 1900, 2100, 2300):
        G = GBOOT.copy()
        nrows = 0
        for it in range(3):
            rows, used = build_rows(G, float(fx), sgn)
            nrows = len(rows)
            if nrows < 1500: break
            G, alg = solve_G(rows)
        if nrows < 1500: 
            log(f"  sgn={sgn:+.0f} fx={fx}: not enough data ({nrows})")
            continue
        e = metric_residuals(G, float(fx), sgn)
        med = float(np.median(e)); p90 = float(np.percentile(e, 90))
        log(f"  sgn={sgn:+.0f} fx={fx}: rows {nrows}, median {med:.2f} m, p90 {p90:.2f} m")
        if best is None or med < best["med"]:
            best = {"med": med, "p90": p90, "fx": float(fx), "sgn": sgn,
                    "G": G.tolist(), "rows": nrows, "alg": alg}

if best is None: sys.exit("calibration did not converge — no lines")
G = np.array(best["G"])
log(f"BEST: fx={best['fx']:.0f} sgn={best['sgn']:+.0f} median {best['med']:.2f} m")

# recomputing metrics for the final G
res = metric_residuals(G, best["fx"], best["sgn"])
report = {
    "fx": best["fx"], "sgn": best["sgn"], "n_points": best["rows"],
    "median_err_m": best["med"], "p90_err_m": best["p90"],
    "max_err_m": float(np.percentile(res, 99)),
    # the solver yields G from l_w^T G r = 0 with r=ref pixel => G: ref pixel -> world
    "G_ref_to_world": G.tolist(),
    "G_world_to_ref": np.linalg.inv(G).tolist(),
    "frames": len(frames), "seconds_elapsed": time.time() - T0,
}
json.dump(report, open(f"{OUT}/G.json", "w"), indent=1)
log("G.json saved")

# ------------------------------------------- 7. world map (top-down view)
PX_M = 30           # pixels per meter
MW, MH = int(60 * PX_M), int(30 * PX_M)
Gwr = np.array(report["G_world_to_ref"])   # world -> ref pixel

ys, xs = np.mgrid[0:MH, 0:MW]
wx = (xs + 0.5) / PX_M
wy = 30.0 - (ys + 0.5) / PX_M
q = np.stack([wx.ravel(), wy.ravel(), np.ones(wx.size)], 1) @ Gwr.T
rx = q[:, 0] / q[:, 2]; ry = q[:, 1] / q[:, 2]

acc = np.zeros((wx.size, 3), np.float32); cnt = np.zeros(wx.size, np.float32)
step = max(1, len(frames) // 20)
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
    img = dict(frames).get(t)
    if img is None: continue
    sx = fxx[vi].astype(np.int32); sy = fyy[vi].astype(np.int32)
    acc[vi] += img[sy, sx].astype(np.float32)
    cnt[vi] += 1
mapimg = np.zeros((wx.size, 3), np.float32)
nz = cnt > 0
mapimg[nz] = acc[nz] / cnt[nz][:, None]
mapimg = mapimg.reshape(MH, MW, 3).astype(np.uint8)
# soft fill of holes
holes = (~nz).reshape(MH, MW).astype(np.uint8)
if holes.any():
    mapimg = cv2.inpaint(mapimg, holes * 255, 5, cv2.INPAINT_TELEA)

# markings on top of the map
def w2m(x, y): return (int(x * PX_M), int((30.0 - y) * PX_M))
for X in RED_X:
    cv2.line(mapimg, w2m(X, 0), w2m(X, 30), (0, 0, 255), 2)
for X in BLUE_X:
    cv2.line(mapimg, w2m(X, 0), w2m(X, 30), (255, 0, 0), 2)
for Y in (0.0, 30.0):
    cv2.line(mapimg, w2m(0, Y), w2m(60, Y), (0, 255, 0), 2)
cv2.circle(mapimg, w2m(30, 15), int(4.5 * PX_M), (0, 0, 255), 2)
for X, Y in [(6, 7.1), (6, 22.9), (54, 7.1), (54, 22.9)]:
    cv2.circle(mapimg, w2m(X, Y), int(4.5 * PX_M), (0, 0, 255), 2)
cv2.putText(mapimg, f"fx={best['fx']:.0f} sgn={best['sgn']:+.0f} med={best['med']:.2f}m",
            (10, MH - 12), 0, 0.6, (255, 255, 255), 2)
cv2.imwrite(f"{OUT}/world_map.jpg", mapimg, [cv2.IMWRITE_JPEG_QUALITY, 92])
log("world_map.jpg saved")

# ---------------------------------------------- 8. overlays on 3 frames
for nm, tsel in (("c", 3.0), ("l", 240.0), ("r", 2570.0)):
    img = dict(frames).get(tsel)
    if img is None: continue
    u = u_of(tsel)
    vis = img.copy()
    def draw_world_line(lw, col, thick=2):
        # world line -> points in world -> ref -> frame
        a, b, c = lw
        if abs(a) > 1e-9:
            ws = np.array([[ -c / a, y, 1.0] for y in np.linspace(0, 30, 40)])
        else:
            ws = np.array([[x, -c / b, 1.0] for x in np.linspace(0, 60, 80)])
        q2 = ws @ Gwr.T
        rr = np.stack([q2[:, 0] / q2[:, 2], q2[:, 1] / q2[:, 2]], 1)
        ff, ok = ref_to_frame(rr, u, best["fx"], best["sgn"])
        ff = ff[ok]
        if len(ff) < 2: return
        pts = np.where(np.isfinite(ff), ff, -1).astype(np.int32)
        for i in range(len(pts) - 1):
            if (pts[i] > -1).all() and (pts[i + 1] > -1).all():
                cv2.line(vis, tuple(pts[i]), tuple(pts[i + 1]), col, thick)
    for X in RED_X: draw_world_line(np.array([1.0, 0.0, -X]), (0, 0, 255), 2)
    for X in BLUE_X: draw_world_line(np.array([1.0, 0.0, -X]), (255, 0, 0), 2)
    for Y in (0.0, 30.0): draw_world_line(np.array([0.0, 1.0, -Y]), (0, 255, 0), 3)
    cv2.putText(vis, f"{nm} t={tsel:.0f}s fx={best['fx']:.0f} med={best['med']:.2f}m",
                (20, 40), 0, 0.9, (0, 0, 255), 2)
    cv2.imwrite(f"{OUT}/overlay_{nm}.jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
log("overlays saved")

# --------------------------------------------------------- 9. report
lines_txt = [
    f"Global marking calibration — {time.strftime('%Y-%m-%d %H:%M')}",
    f"median frames: {len(frames)} across the video (0..{DUR:.0f} s)",
    f"fx = {best['fx']:.0f} px, pan sign = {best['sgn']:+.0f}",
    f"points in the system: {best['rows']}",
    f"line error: median {best['med']:.2f} m, p90 {best['p90']:.2f} m, p99 {report['max_err_m']:.2f} m",
    f"compute time: {report['seconds_elapsed']:.0f} s",
    "",
    "G (world -> ref pixel):",
]
for row in Gwr:
    lines_txt.append("  " + " ".join(f"{v:+.6f}" for v in row))
open(f"{OUT}/calib_report.txt", "w").write("\n".join(lines_txt) + "\n")
log("\n".join(lines_txt))
log("DONE")
