#!/usr/bin/env python3
"""Global calibration check: pan profile over the whole video + rink mosaic.

Model: the camera (PTZ) performs an almost pure yaw. For a purely vertical
axis, rotation = horizontal shift in the image plane: pixel_t = pixel_A + dx_t.
So for each frame we find a single number dx minimizing the distance from the
projected world marking lines to the actual "marking" pixels.

The reference homography H_A comes from anchors.json (frame A, t=3 s).
The keys in anchors.json are swapped by name -> the matrix is identified by a point test.
"""
import cv2, numpy as np, json, os, time, sys

SRC = os.environ.get("SRC", "./media/9281020430926.mp4")
OUT = "pick"
NS, NM = 1000.0, 30.0
FPS = 30.0; Wd, Hd = 1920, 1080
NPT = int(os.environ.get("NPT", 45))
DUR = float(os.environ.get("DUR", 3603.0))
RED_X = [4.0, 30.0, 56.0]; BLUE_X = [22.4, 37.6]
T0 = time.time()
def log(*a): print(*a, flush=True)

# --------------------------------------------------------- H_A
a = json.load(open('anchors.json'))
m1 = np.array(a['G_world2pix']); m2 = np.array(a['H_pix2world'])
def test(m):           # m: pixel(normalized) -> world(normalized)?
    v = m @ np.array([690/NS, 370/NS, 1.0]); w = v[:2]/v[2]
    return abs(w[0]*NM - 30) < 1 and abs(w[1]*NM - 30) < 1
if test(m1):   P2W = m1
elif test(m2): P2W = m2
else: sys.exit("failed to identify pixel->world in anchors.json")
W2P = np.linalg.inv(P2W)
log(f"H_A identified: pixel->world. Test (690,370)->world {np.round((P2W@[690/NS,370/NS,1])[:2]/(P2W@[690/NS,370/NS,1])[2]*NM,2)}")
log(f"det P2W={np.linalg.det(P2W):.4f}")

# base pixel points of the world lines in the frame-A coordinate system (constant!)
def base_pts():
    u = []
    for X in RED_X + BLUE_X:
        for y in np.linspace(0.5, 29.5, 70):
            v = W2P @ np.array([X/NM, y/NM, 1.0]); u.append([v[0]/v[2]*NS, v[1]/v[2]*NS, 1 if X in RED_X else 2])
    return np.array(u)
BP = base_pts()

# --------------------------------------------------------- decode
def medframe(cap, sec, half=0.6, n=5):
    frs = []
    for s in np.linspace(sec-half, sec+half, n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(s*FPS)))
        ok, fr = cap.read()
        if ok: frs.append(fr.astype(np.uint16))
    return np.median(np.stack(frs), 0).astype(np.uint8) if frs else None

cap = cv2.VideoCapture(SRC)
if not cap.isOpened(): sys.exit("no video")
frames = []
for t in np.linspace(2.0, DUR, NPT):
    im = medframe(cap, t)
    if im is not None: frames.append((float(t), im))
    if len(frames) % 9 == 0: log(f"  decode {len(frames)}/{NPT}  {time.time()-T0:.0f}s")
cap.release()
log(f"decoded {len(frames)} frames, {time.time()-T0:.0f}s")

# --------------------------------------------------------- segmentation
def masks(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    Hh, Ss, Vv = hsv[...,0], hsv[...,1], hsv[...,2]
    RED = (((Hh<=10)|(Hh>=168)) & (Ss>70) & (Vv>60))
    BLUE = ((Hh>=95)&(Hh<=135)&(Ss>30)&(Vv>50))
    R = np.zeros((Hd,Wd), bool); R[:,:] = RED
    R[:330] = False; R[1078:] = False
    B = np.zeros((Hd,Wd), bool); B[:,:] = BLUE
    B[:330] = False; B[1078:] = False
    return R, B

def dt(mask):
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)

def cost_at(dx, dtR, dtB):
    p = BP.copy(); p[:,0] += dx
    ok = (p[:,0]>=0)&(p[:,0]<Wd)&(p[:,1]>=330)&(p[:,1]<1078)
    if ok.sum() < 60: return 1e9, 0
    xs = p[ok,0].astype(np.int32); ys = p[ok,1].astype(np.int32)
    kind = p[ok,2]
    d = np.where(kind==1, dtR[ys,xs], dtB[ys,xs])
    return float(np.median(d)), int(ok.sum())

prof = []
for i, (t, img) in enumerate(frames):
    R, B = masks(img)
    dtR, dtB = dt(R), dt(B)
    best = (1e9, 0, 0)
    for dx in np.arange(-900, 901, 20):
        c, n = cost_at(dx, dtR, dtB)
        if c < best[0]: best = (c, dx, n)
    for dx in np.arange(best[1]-20, best[1]+21, 2):
        c, n = cost_at(dx, dtR, dtB)
        if c < best[0]: best = (c, dx, n)
    for dx in np.arange(best[1]-2, best[1]+3, 1):
        c, n = cost_at(dx, dtR, dtB)
        if c < best[0]: best = (c, dx, n)
    c, dx, n = best
    nr = int(R.sum()); nb = int(B.sum())
    prof.append((t, float(dx), float(c), n, nr, nb))
    if (i+1) % 5 == 0 or i == 0:
        log(f"  frame {i+1:2d}/{len(frames)} t={t:6.0f}s dx={dx:+5.0f} px  median {c:5.1f} px  (points {n}, red {nr}, blue {nb})")
log(f"profile done in {time.time()-T0:.0f}s")

# --------------------------------------------------------- summary
dxs = np.array([p[1] for p in prof]); meds = np.array([p[2] for p in prof])
span = dxs.max()-dxs.min()
log(f"\npan shift: min {dxs.min():.0f}  max {dxs.max():.0f}  span {span:.0f} px")
log(f"residual: median {np.median(meds):.1f} px  p90 {np.percentile(meds,90):.1f}  max {meds.max():.1f}")
bad = [p for p in prof if p[2] > 8]
log(f"frames with a bad residual (>8 px): {len(bad)}" + (f" -> {[int(b[0]) for b in bad][:12]}" if bad else ""))

# comparison with an independent ORB profile
try:
    pp = json.load(open('match/pan_profile.json'))
    ts = np.array(pp['secs']); cx = np.array(pp['cum_x'])
    ours = np.array([p[1] for p in prof]); thet = np.array([p[0] for p in prof])
    interp = np.interp(thet, ts, cx)
    k = np.corrcoef(ours, interp)[0,1]
    log(f"\ncheck against the ORB profile (pan_profile.json): r={k:.3f}, its span {cx.max()-cx.min():.0f} px vs our {span:.0f} px")
except Exception as e:
    log(f"failed to compare with the ORB profile: {e}")

# --------------------------------------------------------- mosaic
PX = 30; MW, MH = 60*PX, 30*PX
acc = np.zeros((MH, MW, 3), np.float32); cnt = np.zeros((MH, MW), np.float32)
ys, xs = np.mgrid[0:MH, 0:MW]
WX = (xs+0.5)/PX/NM; WY = (30.0-(ys+0.5)/PX)/NM
good = [(p, f) for p, f in zip(prof, frames) if p[2] <= 8.0]
if not good: good = [(p, f) for p, f in zip(prof, frames)]
log(f"into the mosaic: {len(good)} frames")
for (t, dx, c, n, nr, nb), (tt, img) in good:
    v = (W2P @ np.c_[WX.ravel(), WY.ravel(), np.ones(MW*MH)].T).T
    px = v[:,0]/v[:,2]*NS + dx; py = v[:,1]/v[:,2]*NS
    ok = np.isfinite(px)&np.isfinite(py)&(px>=0)&(px<Wd-1)&(py>=0)&(py<Hd-1)
    m = ok.reshape(MH, MW)
    acc[m] += img[py[ok].astype(np.int32), px[ok].astype(np.int32)].astype(np.float32)
    cnt[m] += 1
nz = cnt > 0
mapimg = np.zeros((MH, MW, 3), np.float32)
mapimg[nz] = acc[nz]/cnt[nz][:, None]
mapimg = mapimg.astype(np.uint8)
cv2.imwrite(f'{OUT}/mosaic_all.jpg', mapimg, [cv2.IMWRITE_JPEG_QUALITY, 93])
log(f"mosaic_all.jpg, coverage {nz.mean()*100:.0f}%")

# overlays on three frames
fd = dict((p[0], f[1]) for p, f in zip(prof, frames))
for tsel in (3.0, 1800.0, 3500.0):
    cand = min(prof, key=lambda p: abs(p[0]-tsel))
    t, dx, c, n, nr, nb = cand
    img = fd[t]
    if img is None: continue
    vis = img.copy()
    for k, col in ((1, (0,0,255)), (2, (255,0,0))):
        sel = BP[:,2] == k
        p = BP[sel].copy(); p[:,0] += dx
        for j in range(len(p)-1):
            a1, a2 = p[j], p[j+1]
            if abs(a1[1]-a2[1]) < 40 and 0<=a1[0]<Wd and 0<=a2[0]<Wd and 330<=a1[1]<1078 and 330<=a2[1]<1078:
                cv2.line(vis, (int(a1[0]),int(a1[1])), (int(a2[0]),int(a2[1])), col, 2)
    cv2.putText(vis, f"t={t:.0f}s dx={dx:.0f}px med={c:.1f}px", (20,380), 0, 0.8, (0,255,255), 2)
    cv2.imwrite(f'{OUT}/verify_{int(tsel)}.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
log("overlays: verify_3.jpg verify_1800.jpg verify_3500.jpg")

json.dump({"ns": NS, "nm": NM, "frames": [{"t": p[0], "dx": p[1], "med_px": p[2], "npts": p[3]} for p in prof],
           "span_px": float(span), "median_overall": float(np.median(meds))},
          open('pan_global.json','w'), indent=1)
log(f"done in {time.time()-T0:.0f}s")
