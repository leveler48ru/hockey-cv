#!/usr/bin/env python3
"""Frame A: normalized DLT on 5 reliable anchors + checking the rest.

Reliable (world position known):
  (250,380)->(22.4,30)  (690,370)->(30,30)  (1100,360)->(37.6,30)   # far boards
  (840,500)->(30,15)                                                # center circle
  (1910,630)->(35.8,13.4)                                           # neutral-zone faceoff dot
Contested — checked only: (300,430), (0,520), (1900,565)
Pixels /1000, meters /30 -> DLT in normalized form (Hartley).
"""
import cv2, numpy as np, json

cap = cv2.VideoCapture('./media/9281020430926.mp4')
frs = []
for s in np.linspace(2.5, 3.5, 5):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(s * 30)); ok, fr = cap.read()
    if ok: frs.append(fr.astype(np.uint16))
cap.release()
img = np.median(np.stack(frs), 0).astype(np.uint8)

NS, NM = 1000.0, 30.0
ANCH = [((250, 380), (22.4, 30.0), "left blue x far boards"),
        ((690, 370), (30.0, 30.0), "red x far boards"),
        ((1100, 360), (37.6, 30.0), "right blue x far boards"),
        ((840, 500), (30.0, 15.0), "center circle"),
        ((1910, 630), (35.8, 13.4), "right neutral dot")]
CHECK = [((300, 430), "A 300,430 left far"), ((0, 520), "A 0,520 on left blue"),
         ((1900, 565), "A 1900,565 on right blue")]

P = np.array([a[0] for a in ANCH], np.float64) / NS
W = np.array([a[1] for a in ANCH], np.float64) / NM
H, _ = cv2.findHomography(P, W, 0)
H = H / H[2, 2]
print(f"DLT rank: {np.linalg.matrix_rank(np.array([[p[0],p[1],1,0,0,0,-w[0]*p[0],-w[0]*p[1],-w[0]] for p,w in zip(P,W)]+[ [0,0,0,p[0],p[1],1,-w[1]*p[0],-w[1]*p[1],-w[1]] for p,w in zip(P,W)]))}")

Hi = np.linalg.inv(H)
def p2w(px, py):
    v = H @ np.array([px / NS, py / NS, 1.0]); return np.array([v[0] / v[2], v[1] / v[2]]) * NM
def w2p(wx, wy):
    v = Hi @ np.array([wx / NM, wy / NM, 1.0]); return np.array([v[0] / v[2], v[1] / v[2]]) * NS

print("\nANCHORS:")
errs = []
for (px, py), (wx, wy), lab in ANCH:
    q = w2p(wx, wy); d = np.hypot(*(q - np.array([px, py])))
    errs.append(d); print(f"  {lab:22s} click({px},{py}) model({q[0]:.0f},{q[1]:.0f})  {d:.0f} px")
print(f"  max {max(errs):.0f} px, median {np.median(errs):.0f} px")

print("\nCONTESTED (which world point the model puts under them):")
for (px, py), lab in CHECK:
    w = p2w(px, py)
    print(f"  {lab:28s} -> world({w[0]:.1f}, {w[1]:.1f})")
    # distance to the nearest known world line
    for X in (22.4, 37.6):
        print(f"      |X-{X}| = {abs(w[0]-X):.2f} m")

# where the model expects the remaining objects
print("\nMODEL PREDICTION (world -> pixel):")
for lab, w in [("left neutral dot", (24.2, 13.4)), ("far boards center x=30", (30, 30)),
               ("near boards x=30", (30, 0)), ("near boards x=22.4", (22.4, 0)),
               ("near boards x=37.6", (37.6, 0)),
               ("corner circle near-left", (13.4, 6.7)), ("corner circle near-right", (46.6, 6.7)),
               ("corner circle far-left", (13.4, 23.3)), ("corner circle far-right", (46.6, 23.3)),
               ("left goal x=4 center", (4, 15)), ("right goal x=56 center", (56, 15))]:
    q = w2p(*w)
    inside = (0 <= q[0] < 1920 and 0 <= q[1] < 1080)
    print(f"  {lab:24s} px({q[0]:8.0f},{q[1]:8.0f}) {'visible' if inside else 'off-frame'}")

json.dump({"H_norm_pix2world": H.tolist(), "NS": NS, "NM": NM,
           "anchors": [[list(a[0]), list(a[1]), a[2]] for a in ANCH],
           "max_err_px": float(max(errs))}, open('anchors.json', 'w'), indent=1)

vis = img.copy()
def line(pts_w, col, th):
    q = np.array([w2p(*w) for w in pts_w]).astype(np.int32)
    for i in range(len(q) - 1):
        if 0 <= q[i, 0] < 1920 and 0 <= q[i+1, 0] < 1920 and 0 <= q[i, 1] < 1080 and 0 <= q[i+1, 1] < 1080:
            cv2.line(vis, tuple(q[i]), tuple(q[i+1]), col, th)
for X in (4.0, 56.0): line([(X, y) for y in np.linspace(0, 30, 60)], (0, 120, 255), 2)
line([(30.0, y) for y in np.linspace(0, 30, 60)], (0, 0, 255), 3)
for X in (22.4, 37.6): line([(X, y) for y in np.linspace(0, 30, 60)], (255, 0, 0), 3)
for Y in (0.0, 30.0): line([(x, Y) for x in np.linspace(0, 60, 80)], (0, 255, 0), 3)
line([(x, 15.0) for x in np.linspace(0, 60, 80)], (0, 200, 0), 1)
for px, py, _ in ANCH: cv2.circle(vis, (px, py), 9, (0, 255, 255), 2)
for (px, py), lab in CHECK: cv2.circle(vis, (px, py), 9, (255, 0, 255), 2)
for lab, w, col in [("neutral-L", (24.2, 13.4), (0,255,0)), ("neutral-R", (35.8, 13.4), (0,255,0)),
                    ("corner-nl", (13.4, 6.7), (255,255,0)), ("corner-nr", (46.6, 6.7), (255,255,0)),
                    ("corner-fl", (13.4, 23.3), (255,255,0)), ("corner-fr", (46.6, 23.3), (255,255,0))]:
    q = w2p(*w)
    if 0 <= q[0] < 1920 and 0 <= q[1] < 1080:
        cv2.circle(vis, (int(q[0]), int(q[1])), 11, col, 2)
        cv2.putText(vis, lab, (int(q[0]) + 14, int(q[1])), 0, 0.5, col, 2)
cv2.imwrite('pick/overlayA_norm.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 92])

# world map (broadcast bug fixed)
PX = 30; MW, MH = 60 * PX, 30 * PX
acc = np.zeros((MH, MW, 3), np.float32); cnt = np.zeros((MH, MW), np.float32)
for yv in range(MH):
    xm = (np.arange(MW) + 0.5) / PX / NM
    ym = np.full(MW, 30.0 - (yv + 0.5) / PX) / NM
    v = (Hi @ np.c_[xm, ym, np.ones(MW)].T).T
    u = v[:, 0] / v[:, 2] * NS; ww = v[:, 1] / v[:, 2] * NS
    ok = np.isfinite(u) & np.isfinite(ww) & (u >= 0) & (u < 1919) & (ww >= 0) & (ww < 1079)
    sub = img[ww[ok].astype(np.int32), u[ok].astype(np.int32)].astype(np.float32)
    acc[yv][ok] += sub; cnt[yv][ok] += 1
nz = cnt > 0
mapimg = np.zeros((MH, MW, 3), np.float32)
mapimg[nz] = acc[nz] / cnt[nz][:, None]
mapimg = mapimg.astype(np.uint8)
print(f"\nworld coverage by frame A: {nz.mean()*100:.0f}%")
def w2m(x, y): return (int(x * PX), int((30 - y) * PX))
for X in (4., 30., 56.): cv2.line(mapimg, w2m(X, 0), w2m(X, 30), (0, 0, 255), 2)
for X in (22.4, 37.6): cv2.line(mapimg, w2m(X, 0), w2m(X, 30), (255, 0, 0), 2)
for Y in (0., 30.): cv2.line(mapimg, w2m(0, Y), w2m(60, Y), (0, 255, 0), 2)
cv2.imwrite('pick/mapA_norm.jpg', mapimg, [cv2.IMWRITE_JPEG_QUALITY, 92])
print("ok: pick/overlayA_norm.jpg pick/mapA_norm.jpg")
