#!/usr/bin/env python3
"""Panoramic anchoring via pure yaw. The camera sits on a tripod by the near board
and rotates around the VERTICAL axis. So the rotation is applied in CAMERA space,
to the left of world->pixel, around the principal point:

    H_t = M(dt) @ H_raw,   M(dt) = K Ry(dt) K^-1

Previously I multiplied on the right - rotating the rink instead of the camera.
Hence the false zeros. Residual = median distance from projected world lines to
actual marking pixels. An empty sample (fewer than 80 points) = inf, not 0.
"""
import cv2, numpy as np, json, sys, time, os

SRC = "./media/9281020430926.mp4"
OUT = "pick"; os.makedirs(OUT, exist_ok=True)
FPS, Wd, Hd = 30.0, 1920, 1080
NS, NM = 1000.0, 30.0
NPT = int(os.environ.get("NPT", 31))
DUR = 3603.0
CX, CY = 960.0, 540.0
RED_X = [4.0, 30.0, 56.0]; BLUE_X = [22.4, 37.6]
T0 = time.time()
def log(*a): print(*a, flush=True)

# ---------- H_raw: world in METERS -> pixel
a = json.load(open('anchors.json'))
m1 = np.array(a['G_world2pix']); m2 = np.array(a['H_pix2world'])
def is_p2w(m):
    v = m @ np.array([690/NS, 370/NS, 1.0]); w = v[:2]/v[2]
    return abs(w[0]*NM - 30) < 1 and abs(w[1]*NM - 30) < 1
P2W = m1 if is_p2w(m1) else (m2 if is_p2w(m2) else None)
if P2W is None: sys.exit("pixel->world not found")
W2P_norm = np.linalg.inv(P2W)
S_winv = np.diag([1/NM, 1/NM, 1.0])
S_pix = np.diag([NS, NS, 1.0])
H_raw = S_pix @ W2P_norm @ S_winv
v = H_raw @ np.array([30, 30, 1.]); v /= v[2]
log(f"H_raw ok: world(30,30) -> px({v[0]:.0f},{v[1]:.0f})")

def Mrot(dt, f):
    K = np.array([[f, 0, CX], [0, f, CY], [0, 0, 1.0]])
    c, s = np.cos(dt), np.sin(dt)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return K @ Ry @ np.linalg.inv(K)


def Mrot2(dt, f, sgn):
    return Mrot(sgn * np.radians(dt), f)

# world points of the lines (for projection and checking)
WPTS, WKIND = [], []
for X in RED_X: WPTS += [[X, y] for y in np.linspace(1, 29, 55)]; WKIND += [0]*55
for X in BLUE_X: WPTS += [[X, y] for y in np.linspace(1, 29, 55)]; WKIND += [1]*55
for Y in [0.0, 30.0]: WPTS += [[x, Y] for x in np.linspace(1, 59, 55)]; WKIND += [2]*55
WPTS = np.array(WPTS, float); WKIND = np.array(WKIND)
WH = np.c_[WPTS, np.ones(len(WPTS))]

# ---------- decode of median frames
cap = cv2.VideoCapture(SRC)
if not cap.isOpened(): sys.exit("no video")
def medframe(sec, half=0.7, n=7):
    frs = []
    for s in np.linspace(sec-half, sec+half, n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(s*FPS)))
        ok, fr = cap.read()
        if ok: frs.append(fr.astype(np.uint16))
    return np.median(np.stack(frs), 0).astype(np.uint8) if len(frs) >= 4 else None
frames = []
for t in np.linspace(3.0, DUR-3, NPT):
    im = medframe(t)
    if im is not None: frames.append((float(t), im))
    if len(frames) % 8 == 0: log(f"  decode {len(frames)}/{NPT} {time.time()-T0:.0f}s")
cap.release()
log(f"decoded {len(frames)} frames {time.time()-T0:.0f}s")

# ---------- marking masks + distance transforms
def prep(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    Hh, Ss, Vv = hsv[...,0], hsv[...,1], hsv[...,2]
    R = (((Hh<=10)|(Hh>=168))&(Ss>70)&(Vv>60))
    B = ((Hh>=95)&(Hh<=135)&(Ss>30)&(Vv>50))
    I = ((Ss<45)&(Vv>170))
    def dt(m):
        m = m.astype(np.uint8); m[:330] = 0; m[1078:] = 0
        m = cv2.morphologyEx(m*255, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        return cv2.distanceTransform((m == 0).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
    def dtp(m):
        m = m.astype(np.uint8); m[:330] = 0; m[1078:] = 0
        m = cv2.dilate(m*255, np.ones((9,9), np.uint8))
        return cv2.distanceTransform((m == 0).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
    return dt(R), dt(B), dtp(I)
PREP = [prep(f) for _, f in frames]
log(f"masks {time.time()-T0:.0f}s")

def score(i, dt, f, sgn=1):
    Ht = Mrot(sgn * np.radians(dt), f) @ H_raw
    q = (Ht @ WH.T).T
    ok = np.abs(q[:, 2]) > 1e-9
    u = np.where(ok, q[:, 0]/np.where(ok, q[:, 2], 1), -1)
    w = np.where(ok, q[:, 1]/np.where(ok, q[:, 2], 1), -1)
    dtm = np.where(WKIND == 0, 0, np.where(WKIND == 1, 1, 2))
    dR, dB, dI = PREP[i]
    grids = [dR, dB, dI]
    tot = []
    for k in range(3):
        s_ = (WKIND == k) & (u >= 1) & (u < Wd-1) & (w >= 331) & (w < 1077)
        if s_.sum() < 20: return 1e9, 0
        g = grids[k][w[s_].astype(np.int32), u[s_].astype(np.int32)]
        tot.append(g)
    allv = np.concatenate(tot)
    return float(np.median(allv)), int(len(allv))

# ---------- global search: f and dt for each frame
FS = [2200, 2800, 3500, 4500, 6000, 9000, 16000]
DTS = np.arange(0, 34.1, 2.0)
SGNS = [1, -1]
best_f, best_sgn, best_tbl, best_med = None, 1, None, 1e9
for f in FS:
    for sgn in SGNS:
        tbl = []
        for i in range(len(frames)):
            row = min((score(i, dt, f, sgn)[0], dt) for dt in DTS)
            tbl.append((row[1], row[0]))
        meds = np.median([t[1] for t in tbl])
        log(f"  f={f} sgn={sgn:+d}: median residual {meds:.1f} px")
        if meds < best_med: best_med, best_f, best_sgn, best_tbl = meds, f, sgn, tbl

# refine dt with a 0.25 step
prof = []
for i, (t, img) in enumerate(frames):
    d0, s0 = best_tbl[i]
    for dt in np.arange(max(0, d0-2), d0+2.01, 0.25):
        s, n = score(i, dt, best_f, best_sgn)
        if s < s0: s0, d0 = s, dt
    prof.append((t, best_sgn * d0, s0))
log(f"\nglobal f = {best_f}, median {best_med:.1f} px, time {time.time()-T0:.0f}s")

ang = np.array([p[1] for p in prof]); rs = np.array([p[2] for p in prof])
log(f"yaw: min {ang.min():+.1f}° max {ang.max():+.1f}° span {np.ptp(ang):.1f}°")
log(f"residual: median {np.median(rs):.1f} p90 {np.percentile(rs,90):.1f} max {rs.max():.1f} px")
for t, d, s in prof:
    log(f"  t={t:6.0f}s yaw={d:+6.2f}° residual={s:5.1f} px")

# ---------- mosaic (only frames with residual < threshold)
THR = max(8.0, float(np.percentile(rs, 60)))
PX = 30; MW, MH = 60*PX, 30*PX
acc = np.zeros((MH, MW, 3), np.float32); cnt = np.zeros((MH, MW), np.float32)
ys, xs = np.mgrid[0:MH, 0:MW]
WG = np.stack([(xs+0.5)/PX, 30.0-(ys+0.5)/PX, np.ones((MH, MW))], -1).reshape(-1, 3)
used = 0
for (t, d, s), (tt, img) in zip(prof, frames):
    if s > THR: continue
    Ht = Mrot(np.radians(d), best_f) @ H_raw
    q = (Ht @ WG.T).T  # noqa
    ok = np.abs(q[:, 2]) > 1e-9
    u = np.where(ok, q[:, 0]/np.where(ok, q[:, 2], -1), -1)
    w = np.where(ok, q[:, 1]/np.where(ok, q[:, 2], -1), -1)
    m = ok & (u >= 0) & (u < Wd-1) & (w >= 0) & (w < Hd-1)
    if m.sum() < 1000: continue
    mm = m.reshape(MH, MW)
    acc[mm] += img[w[m].astype(np.int32), u[m].astype(np.int32)].astype(np.float32)
    cnt[mm] += 1; used += 1
nz = cnt > 0
mi = np.zeros((MH, MW, 3), np.float32); mi[nz] = acc[nz]/cnt[nz][:, None]
mi = mi.astype(np.uint8)
def w2m(x, y): return (int(x*PX), int((30-y)*PX))
for X in RED_X: cv2.line(mi, w2m(X, 0), w2m(X, 30), (0, 0, 255), 2)
for X in BLUE_X: cv2.line(mi, w2m(X, 0), w2m(X, 30), (255, 0, 0), 2)
for Y in [0., 30.]: cv2.line(mi, w2m(0, Y), w2m(60, Y), (0, 255, 0), 2)
cv2.circle(mi, w2m(30, 15), int(4.5*PX), (0, 0, 255), 2)
cv2.putText(mi, f"f={best_f} used={used} med={np.median(rs):.1f}px cov={nz.mean()*100:.0f}%",
            (10, MH-12), 0, 0.65, (255, 255, 255), 2)
cv2.imwrite(f'{OUT}/mosaic_yaw.jpg', mi, [cv2.IMWRITE_JPEG_QUALITY, 93])
log(f"mosaic_yaw.jpg: frames {used}, coverage {nz.mean()*100:.0f}%")

# ---------- overlays on 3 frames
for want in (3.0, 1800.0, 3500.0):
    i = int(np.argmin([abs(p[0]-want) for p in prof]))
    t, d, s = prof[i]; img = frames[i][1]
    Ht = Mrot(np.radians(d), best_f) @ H_raw
    q = (Ht @ WH.T).T
    u = q[:, 0]/q[:, 2]; w = q[:, 1]/q[:, 2]
    vis = img.copy()
    cols = {0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 0)}
    n = 55
    for k, (a_, b_) in enumerate([(0, 165), (165, 275), (275, 385)]):
        seg = slice(a_, b_)
        pts = np.stack([u[seg], w[seg]], 1).astype(np.int32)
        for j in range(len(pts)-1):
            if all(0 <= c < m_ for c, m_ in zip(pts[j], (Wd, Hd))) and all(0 <= c < m_ for c, m_ in zip(pts[j+1], (Wd, Hd))):
                cv2.line(vis, tuple(pts[j]), tuple(pts[j+1]), cols[k], 2)
    for X in RED_X+BLUE_X:
        v = Ht @ np.array([X, 15, 1.]); cv2.circle(vis, (int(v[0]/v[2]), int(v[1]/v[2])), 6, (0, 255, 255), -1)
    cv2.putText(vis, f"t={t:.0f}s yaw={d:+.1f}deg res={s:.1f}px", (20, 40), 0, 0.8, (0, 255, 255), 2)
    cv2.imwrite(f'{OUT}/yaw_{int(t)}.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    log(f"overlay yaw_{int(t)}.jpg")

json.dump({"f": best_f, "profile": [{"t": p[0], "yaw_deg": p[1], "res_px": p[2]} for p in prof],
           "median_res": float(np.median(rs)), "span_deg": float(np.ptp(ang))},
          open('yaw_profile.json', 'w'), indent=1)
log(f"done {time.time()-T0:.0f}s")
