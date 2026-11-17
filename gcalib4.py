#!/usr/bin/env python3
"""Global calibration v4 — per-frame homography from LINE CORRESPONDENCES.

No pan profile, no manual anchors. For each median frame:
  1. RANSAC: red/blue lines + boards (lines l_pix, normalized).
  2. Labeling: vertical lines ordered by x like the world ones
     [4, 22.4, 30, 37.6, 56] (red at 0,2,4; blue at 1,3). Enumeration of order-preserving
     subsets; boards: top=y30 (far), bottom=y0 (near).
  3. H (world->pixel) from l_pix = H^T L_w: 3 linear equations per line, DLT+SVD.
  4. Selection: median line-point residual < 1.5 m.
World map: each frame warped by its own H_k into 60x30, then averaged.

Output: /root/out/G.json, world_map.jpg, overlay_*.jpg, calib_report.txt
"""
import cv2, numpy as np, json, os, sys, time, itertools

BASE = os.environ.get("WORK", "/root")
OUT = f"{BASE}/out"; os.makedirs(OUT, exist_ok=True)
F = f"{BASE}/match.mp4"
FPS = 30.0; Wd, Hd = 1920, 1080
Y0, Y1 = 330, 1078
NPT = int(float(os.environ.get("NPT", 45)))
DUR = float(os.environ.get("DUR", 3603.0))
VERT = [(4.0, "red"), (22.4, "blue"), (30.0, "red"), (37.6, "blue"), (56.0, "red")]
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
if not cap.isOpened(): sys.exit("no video")
times = np.linspace(2.0, DUR, NPT)
frames = []
for t in times:
    im = median_frame(cap, t)
    if im is not None: frames.append((float(t), im))
    if len(frames) % 15 == 0: log(f"  decode {len(frames)}/{NPT}")
cap.release()
log(f"decode: {len(frames)} frames, {time.time()-T0:.0f}s")

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

def fit_line(pts):
    x, y = pts[:, 0], pts[:, 1]
    cx, cy = x.mean(), y.mean()
    M = np.stack([x - cx, y - cy], 1)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    n = Vt[-1]; c = -(n[0] * cx + n[1] * cy)
    l = np.array([n[0], n[1], c])
    return l / np.linalg.norm(l[:2])

def ransac_lines(mask, min_inl=200, tol=3.0, k=4, seed=0):
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
            if abs(pts[i, 1] - pts[j, 1]) < 80: continue
            a = (pts[i, 0] - pts[j, 0]) / (pts[i, 1] - pts[j, 1]); b = pts[i, 0] - a * pts[i, 1]
            res = np.abs(pts[:, 0] - (a * pts[:, 1] + b))
            inl = res < tol
            n_ = int(inl.sum())
            sp = np.ptp(pts[inl, 1]) if n_ > 1 else 0.0
            sc = n_ if sp > 250 else 0
            if best is None or sc > best[0]: best = (sc, inl)
        if best is None or best[0] < min_inl: break
        _, inl = best
        line = fit_line(pts[inl])
        d = np.abs(pts @ line[:2] + line[2])
        if (d < tol).sum() < min_inl: break
        line = fit_line(pts[d < tol])
        out.append((line, pts[d < tol]))
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
    if len(top) > 300: out.append((fit_line(np.array(top, float)), np.array(top, float), 30.0))
    if len(bot) > 300: out.append((fit_line(np.array(bot, float)), np.array(bot, float), 0.0))
    return out

# ------------------------------------------------- H from line correspondences
def solve_H(corr):
    """corr: [(L_w 3-vector, l_pix 3-vector)]; l_pix = H^T L_w. DLT."""
    rows = []
    for Lw, lp in corr:
        for i in range(3):
            r = np.zeros(9); b = lp[i]
            for j in range(3):
                r[j * 3 + i] = Lw[j]
            rows.append(np.append(r, -b))
    A = np.array(rows)
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    _, sv, Vt = np.linalg.svd(A, full_matrices=False)
    h = Vt[-1]
    H = h[:9].reshape(3, 3)
    if abs(H[2, 2]) < 1e-12: return None
    return H / H[2, 2]

def line_residuals(H, corr, pts_list):
    errs = []
    Hi = np.linalg.inv(H)
    for (Lw, lp), pts in zip(corr, pts_list):
        if len(pts) < 30: continue
        sub = pts[::max(1, len(pts) // 150)]
        q = (Hi @ np.c_[sub, np.ones(len(sub))].T).T
        wx, wy = q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]
        ok = np.isfinite(wx) & np.isfinite(wy) & (wx > -20) & (wx < 80) & (wy > -20) & (wy < 50)
        if ok.sum() < 20: continue
        d = np.abs(Lw[0] * wx[ok] + Lw[1] * wy[ok] + Lw[2])
        errs += list(d)
    return np.array(errs) if errs else np.array([1e9])

def solve_frame(items):
    """items: ('v'|'b', l_pix, pts). Returns the best (H, median, corr)."""
    vlines = [it for it in items if it[0] == "v"]
    boards = [it for it in items if it[0] == "b"]
    if not vlines: return None
    # order by x at y=700
    def xat(l): return (-l[1] * 700 - l[2]) / l[0] if abs(l[0]) > 1e-9 else 1e9
    vlines = sorted(vlines, key=lambda it: xat(it[1]))
    nred = sum(1 for it in vlines if it[2] == "red")
    nblue = len(vlines) - nred
    best = None
    red_pos = [0, 2, 4]; blue_pos = [1, 3]
    for rs in itertools.combinations(red_pos, nred):
        for bs in itertools.combinations(blue_pos, nblue):
            merged = sorted([(p, "red") for p in rs] + [(p, "blue") for p in bs])
            # color consistency check
            if any(c != it[2] for (p, c), it in zip(merged, vlines)): continue
            corr, pts_list = [], []
            for (p, c), (_, lp, pts) in zip(merged, vlines):
                X = VERT[p][0]
                corr.append((np.array([1.0, 0.0, -X]), lp)); pts_list.append(pts)
            for lp, pts, Y in boards:
                corr.append((np.array([0.0, 1.0, -Y]), lp)); pts_list.append(pts)
            if len(corr) < 4: continue
            H = solve_H(corr)
            if H is None: continue
            e = line_residuals(H, corr, pts_list)
            med = float(np.median(e))
            if best is None or med < best[1]:
                best = (H, med, corr, pts_list)
    return best

# ------------------------------------------------------- per-frame run
results = []
for ti, (t, img) in enumerate(frames):
    R, B, _ = segment(img)
    items = []
    for line, pts in ransac_lines(R, seed=ti): items.append(("v", line, "red", pts))
    for line, pts in ransac_lines(B, seed=ti + 999): items.append(("v", line, "blue", pts))
    for line, pts, Y in board_lines(img): items.append(("b", line, "board", pts))
    simple = [(k, l, c, p) for k, l, c, p in items]
    # split for solve_frame
    vitems = [("v", l, c, p) for k, l, c, p in items if k == "v"]
    bitems = [("b", l, "board", p) for k, l, c, p in items if k == "b"]
    got = solve_frame([( "v", l, c, p) for _, l, c, p in items if _ == "v"] and
                      [(a, b, c, d) for a, b, c, d in vitems] or vitems)
    # careful assembly:
    mix = [(kind, lp, col, pts) for kind, lp, col, pts in (vitems + bitems)]
    got = solve_frame2(mix) if False else None
    results.append((t, items))
log("line preparation deferred — see solve_frame2")
