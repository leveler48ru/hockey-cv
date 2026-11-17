#!/usr/bin/env python3
"""Frame A: homography from Alexander's anchors, honest parameterization.

Unknowns:
  H (9, h22=1) + d (offset of the faceoff dot from center) + yn (its y)
  + yL (y of the left blue bottom anchor) + yR (y of the right blue bottom anchor)
Anchors:
  full: 3 on the far boards (22.4/30/37.6, y=30), circle center (30,15)
  neutral-zone faceoff dots (paired, unknown d/yn):
      (300,430) -> (30-d, 30-yn)     (1910,630) -> (30+d, yn)
  bottom anchors ON the blue lines (x known, y unknown):
      (0,520) -> (22.4, yL)          (1900,565) -> (37.6, yR)
  left blue line tracked by pixels -> point constraints (1 equation each)
16+5 equations, 13 unknowns. Everything in meters, pixels normalized /1000.
"""
import cv2, numpy as np, json
from scipy.optimize import least_squares

cap = cv2.VideoCapture('./media/9281020430926.mp4')
frs = []
for s in np.linspace(2.5, 3.5, 5):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(s * 30)); ok, fr = cap.read()
    if ok: frs.append(fr.astype(np.uint16))
cap.release()
img = np.median(np.stack(frs), 0).astype(np.uint8)

FULL = [((250, 380), (22.4, 30.0)), ((690, 370), (30.0, 30.0)),
        ((1100, 360), (37.6, 30.0)), ((840, 500), (30.0, 15.0))]
DL = (300, 430); DR = (1910, 630)
BL = (0, 520); BR = (1900, 565)
LTRACK = [(159.5, 440), (115, 460), (77.5, 480), (36.5, 500)]
NS = 1000.0; NM = 30.0


def Hm(p):
    return np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]], [p[6], p[7], 1.0]])


def proj(H, wx, wy):
    """world(m) -> pixel; H operates on normalized coordinates"""
    v = H @ np.array([wx / NM, wy / NM, 1.0])
    return np.array([v[0] / v[2], v[1] / v[2]]) * NS


def w2p(H, w):
    v = H @ np.array([w[0] / NM, w[1] / NM, 1.0])
    return v[0] / v[2], v[1] / v[2]


def res(p, with_track=True):
    H = Hm(p[:8]); d, yn, yL, yR = p[8], p[9], p[10], p[11]
    out = []
    for (px, py), (wx, wy) in FULL:
        u, v = w2p(H, (wx, wy)); out += [u - px, v - py]
    for (px, py), w in [(DL, (30 - d, 30 - yn)), (DR, (30 + d, yn)),
                        (BL, (22.4, yL)), (BR, (37.6, yR))]:
        u, v = w2p(H, w); out += [u - px, v - py]
    if with_track:
        Hi = np.linalg.inv(H)
        for (px, py) in LTRACK:
            l = (Hi / NS * NM).T @ np.array([1.0, 0.0, -22.4 / NM])
            n = np.hypot(l[0], l[1])
            out.append((l[0] * px / NS + l[1] * py / NS + l[2]) / n * NM)
    return np.array(out)


# --- start: DLT on 4 boards/center anchors + 2 dots with a grid search over d,yn
P4 = np.array([p for p, _ in FULL], np.float32)
W4 = np.array([w for _, w in FULL], np.float32)
best = None
for d0 in np.arange(0.5, 9.01, 0.25):
    for yn0 in np.arange(3.0, 16.01, 0.25):
        PP = np.vstack([P4, [DL[0], DL[1]], [DR[0], DR[1]]]).astype(np.float32)
        WW = np.vstack([W4, [30 - d0, 30 - yn0], [30 + d0, yn0]]).astype(np.float32)
        Hg = cv2.findHomography(PP / NS, WW / NM, 0)[0]
        if Hg is None or abs(np.linalg.det(Hg)) < 1e-12: continue
        Hg = Hg / Hg[2, 2]
        q = (Hg @ np.c_[PP / NS, np.ones(6)].T).T * NM
        e = np.linalg.norm(q[:, :2] / q[:, 2:3] - WW, axis=1).max()
        if best is None or e < best[0]: best = (e, d0, yn0, Hg)
e0, d0, yn0, Hg = best
print(f"start: d={d0:.2f} yn={yn0:.2f} max_residual={e0:.3f} m")

# yL, yR from the line through (250,380)-(0,520): roughly take world y from the inverse H
Hi = np.linalg.inv(Hg)
p2w = lambda px, py: (lambda v: (v[0] / v[2] * NM, v[1] / v[2] * NM))((Hi @ np.array([px / NS, py / NS, 1.0])))
yL0 = np.clip(p2w(*BL)[1], 0, 30); yR0 = np.clip(p2w(*BR)[1], 0, 30)
p0 = np.concatenate([Hg.ravel()[:8] / 1.0, [d0, yn0, yL0, yR0]])
p0[:8] = Hg.ravel()[:8] / Hg.ravel()[8]
print(f"start yL={yL0:.1f} yR={yR0:.1f}")

lo = np.concatenate([np.full(8, -20.0), [0.5, 2.0, 0.5, 0.5]])
hi = np.concatenate([np.full(8, 20.0), [9.0, 16.0, 29.0, 29.0]])
sol = least_squares(res, p0, bounds=(lo, hi), method='trf',
                    x_scale='jac', ftol=1e-12, xtol=1e-12, max_nfev=8000)
p = sol.x; H = Hm(p[:8]); d, yn, yL, yR = p[8:]
rr = res(p)
print(f"\ncost={sol.cost:.1f}  max|res|={np.abs(rr).max():.1f} px/m")
print(f"faceoff dots: d={d:.2f} m from center, yn={yn:.2f} m from near boards")
print(f"bottom blue-line anchors: yL={yL:.2f} m, yR={yR:.2f} m")
for (px, py), w in [(DL, (30 - d, 30 - yn)), (DR, (30 + d, yn)), (BL, (22.4, yL)), (BR, (37.6, yR))]:
    u, v = w2p(H, w)
    print(f"  px({px:.0f},{py:.0f}) <- world({w[0]:.1f},{w[1]:.1f}) -> px({u:.0f},{v:.0f})")
for (px, py), w in FULL:
    u, v = w2p(H, w); print(f"  FULL px({px},{py}) <- ({w}) -> ({u:.0f},{v:.0f}) delta {u-px:.0f},{v-py:.0f}")

# where the boards and lines land
for X in [4.0, 22.4, 30.0, 37.6, 56.0]:
    a = w2p(H, (X, 30)); b = w2p(H, (X, 0))
    print(f"  x={X:5.1f}: far px({a[0]:7.0f},{a[1]:6.0f}) near px({b[0]:9.0f},{b[1]:7.0f})")
for Y in [0.0, 15.0, 30.0]:
    a = w2p(H, (30, Y)); print(f"  y={Y:5.1f} along X=30: px({a[0]:.0f},{a[1]:.0f})")

Hi = np.linalg.inv(H)
json.dump({"A": {"t": 3.0, "H_norm_pix2world": H.tolist(), "scale_pix": NS, "scale_m": NM,
                 "d": float(d), "yn": float(yn), "yL": float(yL), "yR": float(yR),
                 "max_res": float(np.abs(rr).max())}}, open('anchors.json', 'w'), indent=1)

vis = img.copy()
def draww(w, col, th):
    ws = ([(w, y) for y in np.linspace(0, 30, 80)] if isinstance(w, float) and abs(w) < 61 and w >= 0
          else None)
    pts = np.array([w2p(H, (x, y)) for x, y in ws], np.int32)
    for i in range(len(pts) - 1):
        if 0 <= pts[i, 0] < 1920 and 0 <= pts[i + 1, 0] < 1920 \
           and 0 <= pts[i, 1] < 1080 and 0 <= pts[i + 1, 1] < 1080:
            cv2.line(vis, tuple(pts[i]), tuple(pts[i + 1]), col, th)
for X in [4.0, 56.0]: draww(X, (0, 120, 255), 2)
draww(30.0, (0, 0, 255), 3)
for X in [22.4, 37.6]: draww(X, (255, 0, 0), 3)
pts = np.array([w2p(H, (x, 30)) for x in np.linspace(0, 60, 100)], np.int32)
for i in range(len(pts) - 1):
    if 0 <= pts[i, 0] < 1920 and 0 <= pts[i + 1, 0] < 1920:
        cv2.line(vis, tuple(pts[i]), tuple(pts[i + 1]), (0, 255, 0), 3)
for (px, py), _ in FULL: cv2.circle(vis, (px, py), 8, (0, 255, 255), -1)
for q in (DL, DR, BL, BR): cv2.circle(vis, (int(q[0]), int(q[1])), 8, (255, 255, 0), 2)
cv2.imwrite('pick/overlayA_final2.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 92])

PX = 30; MW, MH = 60 * PX, 30 * PX
ys, xs = np.mgrid[0:MH, 0:MW]
mapimg = np.zeros((MH, MW, 3), np.uint8)
cnt = np.zeros((MH, MW), np.float32)
acc = np.zeros((MH, MW, 3), np.float32)
for yv in range(MH):
    xm = (np.arange(MW) + 0.5) / PX; ym = np.full(MW, 30.0 - (yv + 0.5) / PX)
    v = (H @ np.c_[xm / NM, ym / NM, np.ones(MW)].T).T
    u = v[:, 0] / v[:, 2] * NS; ww = v[:, 1] / v[:, 2] * NS
    ok = np.isfinite(u) & np.isfinite(ww) & (u >= 0) & (u < 1919) & (ww >= 0) & (ww < 1079)
    acc[yv][ok] += img[ww[ok].astype(np.int32), u[ok].astype(np.int32)].astype(np.float32)
    cnt[yv][ok] += 1
nz = cnt > 0
mapimg[nz] = (acc / np.maximum(cnt, 1)[:, None])[nz].astype(np.uint8)
print(f"world coverage by frame A: {nz.mean()*100:.0f}%")
def w2m(x, y): return (int(x * PX), int((30 - y) * PX))
for X in [4., 30., 56.]: cv2.line(mapimg, w2m(X, 0), w2m(X, 30), (0, 0, 255), 2)
for X in [22.4, 37.6]: cv2.line(mapimg, w2m(X, 0), w2m(X, 30), (255, 0, 0), 2)
for Y in [0., 30.]: cv2.line(mapimg, w2m(0, Y), w2m(60, Y), (0, 255, 0), 2)
cv2.imwrite('pick/mapA_final2.jpg', mapimg, [cv2.IMWRITE_JPEG_QUALITY, 92])
print("ok overlayA_final2.jpg mapA_final2.jpg")
