#!/usr/bin/env python3
"""Center calibration from MARKING LINES (not from points).

Root cause of all troubles found: 36-55% of frame pixels are static TV graphics
(scoreboard/overlay). ORB and phase correlation matched exactly that, so du=0 and
the "chains" lied. The ice is texturally empty -> feature matching doesn't work on it.
Only what is painted on the ice works: red/blue lines, boards.

Algebra: inverse homography G (pixel->world). A world line l_w must map from a
pixel line l_p:  l_w x (G l_p) = 0  -> 2 equations per line.
4 lines = 8 equations = exactly 8 dof of G. Remaining lines — for validation.

Output: match/G_center.json, match/center_lines.jpg
"""
import cv2, numpy as np, json, os, sys

BASE = os.path.dirname(os.path.abspath(__file__)); M = BASE + "/match"
F = "./media/9281020430926.mp4"; FPS = 30
SEC = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
TAG = sys.argv[2] if len(sys.argv) > 2 else "c"

# world lines (x 0..60 from the left boards, y 0..30; y=0 near boards, y=30 far boards)
WORLD_LINES = {"red_center": (1.0, 0.0, -30.0), "blue_farL": (1.0, 0.0, -22.4),
               "blue_farR": (1.0, 0.0, -37.6), "goalL": (1.0, 0.0, -4.0),
               "goalR": (1.0, 0.0, -56.0), "board_near": (0.0, 1.0, 0.0),
               "board_far": (0.0, 1.0, -30.0)}


def frame(sec, half=1.0, n=13):
    cap = cv2.VideoCapture(F); frs = []
    for s in np.linspace(sec - half, sec + half, n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s * FPS)); ok, fr = cap.read()
        if ok: frs.append(fr.astype(np.uint16))
    cap.release()
    return np.median(np.stack(frs), axis=0).astype(np.uint8)


img = frame(SEC)
cv2.imwrite(f"{M}/lf_{TAG}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 93])
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
Hh, Ss, Vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
RED = (((Hh <= 10) | (Hh >= 168)) & (Ss > 70) & (Vv > 60))
# marking blue color: HSV median around strong blue peaks (tuned empirically)
BLUE = ((Hh >= 95) & (Hh <= 135) & (Ss > 45) & (Vv > 50))
ICE = ((Ss < 45) & (Vv > 170))
print(f"shares: red {RED.mean()*100:.2f}%  blue {BLUE.mean()*100:.2f}%  ice {ICE.mean()*100:.2f}%")
for nm, m in (("red", RED), ("blue", BLUE)):
    col = m[400:900, :].sum(axis=0)
    tp = np.argsort(col)[-6:][::-1]
    print(f"  {nm} top cols:", [(int(x), int(col[x])) for x in tp])


def ransac_line(mask, orientation="v", span_min=250, tol=3.0, k=6):
    """orientation 'v': x = a*y + b (rink transverse lines),
       'h' along-row boundary: y = a*x + b. Returns the list of detected lines."""
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    found = []
    for _ in range(k):
        if len(pts) < 200:
            break
        best = None
        for it in range(400):
            i, j = np.random.choice(len(pts), 2, replace=False)
            if orientation == "v":
                if abs(pts[i, 1] - pts[j, 1]) < 60:
                    continue
                a = (pts[i, 0] - pts[j, 0]) / (pts[i, 1] - pts[j, 1])
                b = pts[i, 0] - a * pts[i, 1]
                pred = a * pts[:, 1] + b
                res = np.abs(pts[:, 0] - pred)
                span = pts[pts[:, 1], :][:, 1].ptp() if False else 0
            else:
                if abs(pts[i, 0] - pts[j, 0]) < 60:
                    continue
                a = (pts[i, 1] - pts[j, 1]) / (pts[i, 0] - pts[j, 0])
                b = pts[i, 1] - a * pts[i, 0]
                pred = a * pts[:, 0] + b
                res = np.abs(pts[:, 1] - pred)
            inl = res < tol
            if orientation == "v":
                sp = np.ptp(pts[inl,1]) if inl.sum() > 1 else 0
            else:
                sp = np.ptp(pts[inl,0]) if inl.sum() > 1 else 0
            score = inl.sum() if sp > span_min else 0
            if score > (best[0] if best else 0):
                best = (score, a, b, inl, sp)
        if not best or best[0] < 150:
            break
        _, a, b, inl, sp = best
        # refit on inliers (least squares)
        if orientation == "v":
            A = np.stack([pts[inl, 1], np.ones(inl.sum())], 1)
            cf, *_ = np.linalg.lstsq(A, pts[inl, 0], rcond=None)
            a, b = cf
            pred = a * pts[:, 1] + b
        else:
            A = np.stack([pts[inl, 0], np.ones(inl.sum())], 1)
            cf, *_ = np.linalg.lstsq(A, pts[inl, 1], rcond=None)
            a, b = cf
            pred = a * pts[:, 0] + b
        res = np.abs((pts[:, 0] if orientation == "v" else pts[:, 1]) - pred)
        inl = res < tol
        found.append({"a": float(a), "b": float(b), "inl": int(inl.sum()), "span": float(
            (np.ptp(pts[inl,1]) if orientation == "v" else np.ptp(pts[inl,0])))})
        pts = pts[~inl]
    return found


np.random.seed(0)
vl = ransac_line(RED, "v", span_min=250, k=5)
bl = ransac_line(BLUE, "v", span_min=200, k=5)
# boards: ice boundary at top/bottom
icem = (ICE.astype(np.uint8) * 255)
icem = cv2.morphologyEx(icem, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
top = np.zeros((1080, 1920), np.uint8); bot = np.zeros_like(top)
for x in range(0, 1920, 2):
    col = np.nonzero(icem[:, x])[0]
    if len(col) > 200:
        top[col.min(), x] = 255; top[min(1079, col.min() + 3), x] = 255
        bot[col.max(), x] = 255; bot[max(0, col.max() - 3), x] = 255
tl = ransac_line(top > 0, "h", span_min=700, k=2)
bo = ransac_line(bot > 0, "h", span_min=700, k=2)
print("\nred transverse lines:", [(round(d['a'], 3), round(d['b']), d['inl'], round(d['span'])) for d in vl])
print("blue transverse lines: ", [(round(d['a'], 3), round(d['b']), d['inl'], round(d['span'])) for d in bl])
print("far board (y=a*x+b):    ", tl)
print("near board (y=a*x+b):   ", bo)
json.dump({"red_v": vl, "blue_v": bl, "far_h": tl, "near_h": bo, "sec": SEC},
          open(f"{M}/lines_{TAG}.json", "w"), indent=1)

# --- draw the detected lines on top of the frame
vis = img.copy()
for d in vl:
    for y in (400, 1000):
        cv2.circle(vis, (int(d["a"] * y + d["b"]), y), 10, (0, 255, 255), -1)
    cv2.line(vis, (int(d["a"] * 350 + d["b"]), 350), (int(d["a"] * 1070 + d["b"]), 1070), (0, 255, 255), 2)
for d in bl:
    cv2.line(vis, (int(d["a"] * 350 + d["b"]), 350), (int(d["a"] * 1070 + d["b"]), 1070), (255, 255, 0), 2)
for d in tl + bo:
    cv2.line(vis, (0, int(d["b"])), (1919, int(d["a"] * 1919 + d["b"])), (0, 255, 0), 3)
for nm, p, col in (("c1", (850, 498), (0, 0, 255)), ("c2", (698, 374), (0, 0, 255)), ("c3", (1489, 1079), (0, 0, 255))):
    cv2.circle(vis, p, 14, col, 3)
    cv2.putText(vis, nm, (p[0] + 18, p[1]), 0, 1.0, col, 3)
cv2.imwrite(f"{M}/center_lines_{TAG}.jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("saved center_lines.jpg")
