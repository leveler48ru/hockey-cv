#!/usr/bin/env python3
"""Demo 'how the markings are determined': split-screen (debug).
Left: broadcast frame + v5 masks. Right: 60x30 m rink map
where masks are projected by homography and accumulate (lines emerge
frame by frame), overlaid with player/referee positions as dots and the camera footprint.
H seed: real metres from second 2400 of m1 (real_masks/m1_02400.npz),
then along the video - ORB chain (neighbouring frames, 1/15 s -> reliable).
Output: /work/demo_geom_0811.mp4"""
import cv2, numpy as np, time, os
from collections import deque
from ultralytics import YOLO

SRC = '/work/m1.mp4'
SEED = '/work/seed_m1.npz'
T0 = float(os.environ.get('DEMO_T0', 2400))
DUR = 8.0
STRIDE = 2
OUT = '/work/demo_geom_0811.mp4'

model = YOLO('/runs/rink_referees_v5/weights/best.pt')
CLASSES = ['red_line', 'blue_line', 'center_circle', 'faceoff_spot', 'faceoff_circle',
           'goal_crease', 'board', 'player', 'referee']
COL = {0:(60,60,230), 1:(255,150,40), 2:(230,230,230), 3:(180,105,255), 4:(255,255,140),
       5:(255,200,80), 6:(110,110,110), 7:(0,215,0), 8:(0,255,255)}

H = np.load(SEED)['H'].astype(np.float64)   # pixel -> world, world x:0..60 y:0..30
PPM, MARGIN = 10, 1
GW, GH = int((60 + 2 * MARGIN) * PPM), int((30 + 2 * MARGIN) * PPM)
S = np.array([[PPM, 0, MARGIN * PPM], [0, PPM, MARGIN * PPM], [0, 0, 1]])

def w2g(x, y):
    q = np.array([x, y, 1.0]) @ S.T
    return float(q[0] / q[2]), float(q[1] / q[2])

def rink_base():
    img = np.full((GH, GW, 3), 22, np.uint8)
    # thin metre grid
    for x in range(0, 61, 5):
        a = w2g(x, 0); b = w2g(x, 30)
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (35, 35, 35), 1)
    for y in range(0, 31, 5):
        a = w2g(0, y); b = w2g(60, y)
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (35, 35, 35), 1)
    # reference markings - barely visible (what the model is supposed to FIND)
    ref = np.full((GH, GW, 3), 0, np.uint8)
    def vline(x, col):
        a = w2g(x, 0); b = w2g(x, 30)
        cv2.line(ref, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, 1)
    vline(30, (90, 90, 160)); vline(22.4, (160, 120, 60)); vline(37.6, (160, 120, 60))
    vline(4, (140, 140, 140)); vline(56, (140, 140, 140))
    def circle(cx, cy, r, col, th=1):
        cv2.ellipse(ref, (int((cx + MARGIN) * PPM), int((cy + MARGIN) * PPM)),
                    (int(r * PPM), int(r * PPM)), 0, 0, 360, col, th)
    circle(30, 15, 4.5, (150, 150, 150))
    for cx, cy in [(12, 6), (12, 24), (48, 6), (48, 24)]:
        circle(cx, cy, 4.5, (150, 150, 150))
    cv2.addWeighted(ref, 0.5, img, 1.0, 0, img)
    return img

def crease_pts(gx, sign):
    a = np.linspace(-np.pi / 2, np.pi / 2, 16)
    pts = [(gx - sign * 1.9 * np.cos(t), 15 + sign * 1.9 * np.sin(t)) for t in a]
    return pts

base = rink_base()
# crease on the reference map
cr = np.full((GH, GW, 3), 0, np.uint8)
for gx, sg in [(4, 1), (56, -1)]:
    pp = [w2g(x, y) for x, y in crease_pts(gx, sg)]
    cv2.polylines(cr, [np.int32(pp).reshape(-1, 1, 2)], False, (140, 140, 140), 1)
cv2.addWeighted(cr, 0.5, base, 1.0, 0, base)

acc = base.copy()                    # accumulated markings
trails = deque(maxlen=30)            # world positions of people, history

cap = cv2.VideoCapture(SRC)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
cap.set(cv2.CAP_PROP_POS_MSEC, T0 * 1000)
W = int(cap.get(3)); Hh = int(cap.get(4))

orb = cv2.ORB_create(2000)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

prev_gray = None
def propagate_to(gray):
    """ORB: current -> previous; H_cur = H_prev @ M(cur->prev)."""
    global H
    kp1, d1 = orb.detectAndCompute(prev_gray, None)
    kp2, d2 = orb.detectAndCompute(gray, None)
    if d1 is None or d2 is None or len(kp1) < 20 or len(kp2) < 20:
        return False
    m = matcher.knnMatch(d2, d1, k=2)
    good = [a for a, b in m if a.distance < 0.75 * b.distance]
    if len(good) < 15:
        return False
    src = np.float32([kp2[a.queryIdx].pt for a in good])
    dst = np.float32([kp1[a.trainIdx].pt for a in good])
    M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if M is None or int(mask.ravel().sum()) < 40:
        return False
    H = H @ M
    return True

vw = None
n = ok_cnt = fail_cnt = 0
t0 = time.time()
pos = T0
while True:
    if not cap.grab():
        break
    pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if pos > T0 + DUR:
        break
    if idx % STRIDE:
        continue
    ok, frame = cap.retrieve()
    if not ok:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None:
        if propagate_to(gray):
            ok_cnt += 1
        else:
            fail_cnt += 1
    prev_gray = gray

    # inference
    r = model.predict(frame, imgsz=960, device='cpu', verbose=False, conf=0.25)[0]
    left = frame.copy()
    people = []
    if r.masks is not None:
        for j in range(len(r.boxes)):
            cls = int(r.boxes.cls[j]); conf = float(r.boxes.conf[j])
            if cls not in COL:
                continue
            m = (r.masks.data[j].cpu().numpy() > 0.5).astype(np.uint8) * 255
            m = cv2.resize(m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            col = COL[cls]
            xy = np.argwhere(m > 0)
            if cls in (7, 8):
                if len(xy):
                    y1, x1 = xy.min(0); y2, x2 = xy.max(0)
                    cv2.rectangle(left, (x1, y1), (x2, y2), col, 2)
                    u, v = (x1 + x2) / 2, y1
                    q = np.array([u, v, 1.0]) @ H.T
                    people.append((cls, float(q[0] / q[2]), float(q[1] / q[2])))
                alpha = 0.35
            else:
                alpha = 0.45
            layer = left.copy()
            layer[m > 0] = col
            left[:] = (layer.astype(np.float32) * alpha + left.astype(np.float32) * (1 - alpha)).astype(np.uint8)

            if cls <= 5:
                G = S @ H
                g = cv2.warpPerspective(m, G, (GW, GH), flags=cv2.INTER_NEAREST)
                import sys as _s
                print(f'FRAME {n} cls={cls} pix={int((m>0).sum())} warped={int((g>0).sum())} H00={H[0][0]:.4f}', flush=True)
                wzone = g > 0
                acc[wzone] = (acc[wzone].astype(np.float32) * 0.3 +
                              np.array(col, np.float32) * 0.7).astype(np.uint8)

    # right panel: copy of accumulated + current frame
    right = acc.copy()
    # camera footprint (4 frame corners projected into world)
    corners = np.array([[0, 0], [frame.shape[1], 0], [frame.shape[1], frame.shape[0]], [0, frame.shape[0]]], float)
    qc = (np.hstack([corners, np.ones((4, 1))]) @ H.T)
    qc = (qc[:, :2] / qc[:, 2:3])
    gp = (np.hstack([qc, np.ones((4, 1))]) @ S.T)
    gp = (gp[:, :2] / gp[:, 2:3]).astype(np.int32)
    cv2.polylines(right, [gp.reshape(-1, 1, 2)], True, (255, 255, 255), 2)
    for cls, x, y in people:
        gx, gy = w2g(x, y)
        if 0 <= gx < GW and 0 <= gy < GH:
            if cls == 8:
                pts = np.array([[gx, gy - 8], [gx + 8, gy], [gx, gy + 8], [gx - 8, gy]], np.int32)
                cv2.fillPoly(right, [pts.reshape(-1,1,2)], COL[8])
                cv2.polylines(right, [pts.reshape(-1,1,2)], True, (255,255,255), 1)
            else:
                cv2.circle(right, (int(gx), int(gy)), 6, COL[cls], -1)
                cv2.circle(right, (int(gx), int(gy)), 8, (255, 255, 255), 1)

    # compose 1280x400: broadcast on the left, map on the right
    L = cv2.resize(left, (640, 360))
    R = cv2.resize(right, (640, 330))
    canvas = np.zeros((400, 1280, 3), np.uint8)
    canvas[:360, :640] = L
    canvas[:330, 640:1280] = R
    cv2.line(canvas, (640, 0), (640, 360), (60, 60, 60), 2)
    cv2.putText(canvas, f't={pos:.1f}s', (10, 22), 0, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'projection -> metres 60x30 (accumulate)', (650, 22), 0, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    # legend at the bottom
    xoff = 650
    for ci in (0, 1, 2, 5, 7, 8):
        cv2.rectangle(canvas, (xoff, 375), (xoff + 14, 391), COL[ci], -1)
        cv2.putText(canvas, CLASSES[ci][:11], (xoff + 18, 389), 0, 0.36, (220, 220, 220), 1, cv2.LINE_AA)
        xoff += 100
    if vw is None:
        vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'), fps / STRIDE, (1280, 400))
    vw.write(canvas)
    n += 1
    if n % 20 == 0:
        print(f'{n}/{120} {time.time()-t0:.0f}s (chain ok={ok_cnt} fail={fail_cnt})', flush=True)
cap.release()
vw.release()
print(f'DONE {n} -> {OUT} ok={ok_cnt} fail={fail_cnt} {time.time()-t0:.0f}s', flush=True)
