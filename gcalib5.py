#!/usr/bin/env python3
"""Rink markings calibration over the whole video — v5.
Each frame gets its own homography G: pixel->world, from rink lines only.
Line relation: a world point w on line L => L^T w=0; w ~ G p => l_pix ~ G^T L.
Solve G^T L_i = lam_i l_i (homogeneous SVD). Labeling by order+color.
World map = averaging warped frames into a 60x30 grid."""
import cv2, numpy as np, json, os, sys, time, itertools

BASE = os.environ.get("WORK", "/root"); OUT = f"{BASE}/out"
os.makedirs(OUT, exist_ok=True)
F = f"{BASE}/match.mp4"; FPS = 30.0; Wd, Hd = 1920, 1080
Y0, Y1 = 330, 1078; NPT = int(float(os.environ.get("NPT", 45)))
DUR = float(os.environ.get("DUR", 3603.0)); PX_M = 30
VERT = [(4.0, "red"), (22.4, "blue"), (30.0, "red"), (37.6, "blue"), (56.0, "red")]
T0 = time.time()
def log(*a): print(*a, flush=True)

def medframe(cap, sec, half=1.0, n=11):
    frs = []
    for s in np.linspace(sec - half, sec + half, n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(s * FPS)))
        ok, fr = cap.read()
        if ok: frs.append(fr.astype(np.uint16))
    return np.median(np.stack(frs), 0).astype(np.uint8) if len(frs) >= 4 else None

cap = cv2.VideoCapture(F)
if not cap.isOpened(): sys.exit("no video")
frames = []
for t in np.linspace(2.0, DUR, NPT):
    im = medframe(cap, t)
    if im is not None: frames.append((float(t), im))
    if len(frames) % 15 == 0: log(f"  decode {len(frames)}/{NPT}")
cap.release()
log(f"decode {len(frames)} frames, {time.time()-T0:.0f}s")

GRAY = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for _, f in frames])
STATIC = GRAY.std(0) < 6.0
WIN = np.zeros((Hd, Wd), bool); WIN[Y0:Y1] = True; GOOD = WIN & ~STATIC
log(f"static {STATIC.mean()*100:.0f}% | window {GOOD.mean()*100:.0f}%")

def seg(img):
    h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    Hh, Ss, Vv = h[..., 0], h[..., 1], h[..., 2]
    R = (((Hh <= 10) | (Hh >= 168)) & (Ss > 70) & (Vv > 60)) & GOOD
    B = ((Hh >= 95) & (Hh <= 135) & (Ss > 35) & (Vv > 50)) & GOOD
    I = ((Ss < 45) & (Vv > 170)) & GOOD
    return R, B, I

def fitline(pts):
    x, y = pts[:, 0], pts[:, 1]; cx, cy = x.mean(), y.mean()
    M = np.stack([x - cx, y - cy], 1); _, _, Vt = np.linalg.svd(M, full_matrices=False)
    n = Vt[-1]; l = np.array([n[0], n[1], -(n[0]*cx + n[1]*cy)])
    return l / np.linalg.norm(l[:2])

def ransac(mask, seed, min_inl=200, tol=3.0, k=4):
    ys, xs = np.nonzero(mask)
    if len(ys) < min_inl: return []
    pts = np.stack([xs, ys], 1).astype(float); rng = np.random.default_rng(seed); out = []
    for _ in range(k):
        if len(pts) < min_inl: break
        best = None
        for _it in range(250):
            i, j = rng.choice(len(pts), 2, replace=False)
            if abs(pts[i, 1] - pts[j, 1]) < 80: continue
            a = (pts[i, 0]-pts[j, 0])/(pts[i, 1]-pts[j, 1]); b = pts[i, 0]-a*pts[i, 1]
            res = np.abs(pts[:, 0]-(a*pts[:, 1]+b)); inl = res < tol; n_ = int(inl.sum())
            sp = np.ptp(pts[inl, 1]) if n_ > 1 else 0.0
            sc = n_ if sp > 250 else 0
            if best is None or sc > best[0]: best = (sc, inl)
        if best is None or best[0] < min_inl: break
        _, inl = best; line = fitline(pts[inl]); d = np.abs(pts @ line[:2] + line[2])
        if (d < tol).sum() < min_inl: break
        line = fitline(pts[d < tol]); out.append((line, pts[d < tol])); pts = pts[d >= tol]
    return out

def boards(img):
    _, _, I = seg(img)
    icem = cv2.morphologyEx((I.astype(np.uint8)*255), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    top, bot = [], []
    for x in range(0, Wd, 3):
        col = np.nonzero(icem[:, x])[0]
        if len(col) > 150: top.append((x, float(col.min()))); bot.append((x, float(col.max())))
    out = []
    if len(top) > 300: out.append((fitline(np.array(top, float)), np.array(top, float), 30.0))
    if len(bot) > 300: out.append((fitline(np.array(bot, float)), np.array(bot, float), 0.0))
    return out

def solveG(corrs):
    n = len(corrs)
    if n < 4: return None
    A = np.zeros((3*n, 9+n))
    for i, (L, l) in enumerate(corrs):
        for kk in range(3):
            for j in range(3): A[3*i+kk, j*3+kk] = L[j]
            A[3*i+kk, 9+i] = -l[kk]
    A = A/np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    try: _, sv, Vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError: return None
    G = Vt[-1, :9].reshape(3, 3)
    if abs(G[2, 2]) < 1e-9 or not np.all(np.isfinite(G)): return None
    return G/G[2, 2]

def resid(G, corrs, ptsets):
    e = []
    for (L, _), pts in zip(corrs, ptsets):
        if len(pts) < 30: continue
        sub = pts[::max(1, len(pts)//150)]
        q = (G @ np.c_[sub, np.ones(len(sub))].T).T
        d = q[:, 0]/q[:, 2]; dd = q[:, 1]/q[:, 2]
        ok = np.isfinite(d)&np.isfinite(dd)&(d>-30)&(d<90)&(dd>-30)&(dd<60)
        if ok.sum() < 20: continue
        e += list(np.abs(L[0]*d[ok]+L[1]*dd[ok]+L[2]))
    return np.array(e) if e else None

def xat(l, y=700.0): return (-l[1]*y-l[2])/l[0] if abs(l[0])>1e-9 else 1e12

def solve_frame(vlines, brd):
    if len(vlines) < 3 or not brd: return None
    order = sorted(range(len(vlines)), key=lambda i: xat(vlines[i][0]))
    vl = [vlines[i] for i in order]; cols = [c for _, _, c in vl]
    nred = cols.count("red"); nblue = cols.count("blue")
    if nred+nblue != len(vl): return None
    best = None
    for rs in itertools.combinations([0, 2, 4], nred):
        for bs in itertools.combinations([1, 3], nblue):
            merged = sorted(list(rs)+list(bs))
            if len(merged) != len(vl): continue
            if any(VERT[p][1] != cols[i] for i, p in enumerate(merged)): continue
            corrs, ptsets = [], []
            for i, p in enumerate(merged):
                corrs.append((np.array([1.0, 0.0, -VERT[p][0]]), vl[i][0])); ptsets.append(vl[i][1])
            for l, pts, Y in brd:
                corrs.append((np.array([0.0, 1.0, -Y]), l)); ptsets.append(pts)
            G = solveG(corrs)
            if G is None: continue
            e = resid(G, corrs, ptsets)
            if e is None or len(e) < 200: continue
            med = float(np.median(e))
            if len(brd) == 2:
                ys = []
                for l, pts, Y in brd:
                    q = (G @ np.c_[pts[::10], np.ones(len(pts[::10]))].T).T
                    yy = q[:, 1]/q[:, 2]
                    if np.all(np.isfinite(yy)): ys.append(float(np.median(yy)))
                if len(ys) == 2 and not (26.0 < abs(ys[0]-ys[1]) < 34.0): continue
            if best is None or med < best[1]:
                best = (G, med, float(np.percentile(e, 90)), len(corrs))
    return best

frdict = dict(frames); solved = []
for ti, (t, img) in enumerate(frames):
    R, B, _ = seg(img)
    vlines = [(l, p, "red") for l, p in ransac(R, ti)] + [(l, p, "blue") for l, p in ransac(B, ti+991)]
    res = solve_frame(vlines, boards(img))
    if res:
        solved.append((t, res))
        log(f"  @{t:5.0f}s: lines {res[3]}, med {res[1]:.2f} m, p90 {res[2]:.2f} m")
    else:
        log(f"  @{t:5.0f}s: UNSOLVED (verticals {len(vlines)})")
log(f"solved {len(solved)}/{len(frames)}, {time.time()-T0:.0f}s")
if len(solved) < 5: sys.exit("too few frames")

MW, MH = 60*PX_M, 30*PX_M
acc = np.zeros((MH, MW, 3), np.float32); cnt = np.zeros((MH, MW), np.float32)
ys, xs = np.mgrid[0:MH, 0:MW]
Wg = np.stack([(xs+0.5)/PX_M, 30.0-(ys+0.5)/PX_M, np.ones(MH*MW)], 1)
for t, (G, med, p90, nl) in solved:
    Gi = np.linalg.inv(G); q = Wg @ Gi.T
    px = q[:, 0]/q[:, 2]; py = q[:, 1]/q[:, 2]
    ok = (px>=0)&(px<Wd-1)&(py>=0)&(py<Hd-1)
    if not ok.any(): continue
    img = frdict[t]; m = ok.reshape(MH, MW)
    acc[m] += img[py[ok].astype(np.int32), px[ok].astype(np.int32)].astype(np.float32); cnt[m] += 1
nz = cnt > 0; mapimg = np.zeros((MH, MW, 3), np.float32)
mapimg[nz] = acc[nz]/cnt[nz][:, None]; mapimg = mapimg.astype(np.uint8)
holes = (~nz).astype(np.uint8)
if holes.any(): mapimg = cv2.inpaint(mapimg, holes*255, 7, cv2.INPAINT_TELEA)
def w2m(x, y): return (int(x*PX_M), int((30.0-y)*PX_M))
for X, c in [(4.0,(0,0,255)),(30.0,(0,0,255)),(56.0,(0,0,255)),(22.4,(255,0,0)),(37.6,(255,0,0))]:
    cv2.line(mapimg, w2m(X, 0), w2m(X, 30), c, 2)
for Y in (0.0, 30.0): cv2.line(mapimg, w2m(0, Y), w2m(60, Y), (0, 255, 0), 2)
cv2.circle(mapimg, w2m(30, 15), int(4.5*PX_M), (0, 0, 255), 2)
for X, Y in [(6,7.1),(6,22.9),(54,7.1),(54,22.9)]: cv2.circle(mapimg, w2m(X, Y), int(4.5*PX_M), (0,0,255), 2)
cov = nz.mean()
meds = [s[1][1] for s in solved]
cv2.putText(mapimg, f"v5: {len(solved)} fr, med={np.median(meds):.2f}m, cov={cov*100:.0f}%",
            (10, MH-12), 0, 0.6, (255, 255, 255), 2)
cv2.imwrite(f"{OUT}/world_map.jpg", mapimg, [cv2.IMWRITE_JPEG_QUALITY, 93])
log(f"world_map.jpg, coverage {cov*100:.0f}%")

for nm, tsel in (("c", 3.0), ("l", 240.0), ("r", 2570.0)):
    img = frdict.get(tsel)
    if img is None: continue
    cand = [s for s in solved if abs(s[0]-tsel) < 90]
    if not cand: continue
    t, (G, med, p90, nl) = min(cand, key=lambda s: abs(s[0]-tsel))
    Gi = np.linalg.inv(G); vis = img.copy()
    def draw(Lw, col, th=2):
        a, b, c = Lw
        ws = (np.array([[-c/a, y, 1.0] for y in np.linspace(0, 30, 60)]) if abs(a)>1e-9
              else np.array([[x, -c/b, 1.0] for x in np.linspace(0, 60, 80)]))
        q2 = ws @ Gi.T; rr = np.stack([q2[:, 0]/q2[:, 2], q2[:, 1]/q2[:, 2]], 1).astype(np.int32)
        for i in range(len(rr)-1):
            if 0<=rr[i,0]<Wd and 0<=rr[i+1,0]<Wd: cv2.line(vis, tuple(rr[i]), tuple(rr[i+1]), col, th)
    for X in [4.0, 30.0, 56.0]: draw(np.array([1.0, 0.0, -X]), (0, 0, 255), 2)
    for X in [22.4, 37.6]: draw(np.array([1.0, 0.0, -X]), (255, 0, 0), 2)
    for Y in (0.0, 30.0): draw(np.array([0.0, 1.0, -Y]), (0, 255, 0), 3)
    cv2.putText(vis, f"{nm} t={tsel:.0f}s med={med:.2f}m", (20, 40), 0, 0.9, (0, 0, 255), 2)
    cv2.imwrite(f"{OUT}/overlay_{nm}.jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
log("overlays saved")

best = min(solved, key=lambda s: s[1][1])
report = {"n_frames": len(solved), "median_err_m": float(np.median(meds)),
          "p90_err_m": float(np.percentile(np.concatenate([resid(s[1][0], [], []) or [0] for s in solved]) if False else meds, 90)) if False else float(np.max([s[1][2] for s in solved])),
          "best_frame_t": best[0], "best_med_m": best[1][1],
          "G_best_pix_to_world": best[1][0].tolist(),
          "seconds_elapsed": time.time()-T0}
json.dump(report, open(f"{OUT}/G.json", "w"), indent=1)
txt = [f"Calibration v5 — {time.strftime('%Y-%m-%d %H:%M')}",
       f"median frames {len(frames)}, solved {len(solved)}",
       f"per-frame median error: {np.median(meds):.2f} m (range {min(meds):.2f}..{max(meds):.2f})",
       f"world map coverage: {cov*100:.0f}%", f"elapsed {report['seconds_elapsed']:.0f} s",
       "", "Best frame G (pixel->world):"]
for row in best[1][0]: txt.append("  "+" ".join(f"{v:+.6f}" for v in row))
open(f"{OUT}/calib_report.txt", "w").write("\n".join(txt)+"\n")
log("\n".join(txt)); log("DONE")
