#!/usr/bin/env python3
"""Propagation v2 for fs frames: ORB matching over ICE only (polygon of the anchor H)
+ independent control of each frame:
  a) the near board edge (sharp brightness drop) must land in [-18,-12] in world;
  b) if red/blue lines are visible (blueness/redness profile) - median x within tolerance.
We accept H only when both controls pass (b - when there is enough data)."""
import json
import numpy as np, cv2
from pathlib import Path

BASE = Path(".")
OUT = BASE / "fs_calib"
W, H = 1280, 720

RINK_CORNERS = [(-30,-15),(30,-15),(30,15),(-30,15)]

orb = cv2.ORB_create(3000)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

def kpts(gray):
    return orb.detectAndCompute(gray, None)

def proj(Hm, pts):
    q = np.array(pts, float)
    q = np.hstack([q, np.ones((len(q),1))]) @ Hm.T
    return q[:, :2] / q[:, 2:3]

fits = json.loads((OUT/"fits.json").read_text())
pool = {}
for stem, v in fits.items():
    g = cv2.cvtColor(cv2.imread(str(BASE/"wide_fg"/f"{stem}.jpg")), cv2.COLOR_BGR2GRAY)
    kp, des = kpts(g)
    pool[stem] = [np.array(v["H"], float), kp, des]

all_stems = sorted([p.stem for p in (BASE/"wide_fg").glob("*.jpg")],
                   key=lambda s: float(s.rsplit("_",1)[1]))
by_time = {s: float(s.rsplit("_",1)[1]) for s in all_stems}

MIN_INL = 60
EDGE_OK = (-18.0, -12.0)

def ice_edge_y(gray):
    h, w = gray.shape
    ys = []
    for c in range(int(w*0.2), int(w*0.8), max(1, w//15)):
        col = gray[:, c].astype(float)
        last = None
        for y in range(h-8, int(h*0.25), -2):
            if col[y] - col[max(0,y-12)] > 25: last = y
            else: break
        if last: ys.append(last)
    return float(np.median(ys)) if ys else None

def line_check(img, Hn):
    """red/blue median x in world; None if nothing visible."""
    a = img.astype(float); B,G,R = a[:,:,0],a[:,:,1],a[:,:,2]
    redness = R-(B+G)/2; blueness = B-(R+G)/2
    Hi = np.linalg.inv(Hn)
    out = {}
    for name, band, xs in (("red", redness, (0.0,)), ("blue", blueness, (-7.5, 7.5))):
        xs_found = []
        for wx in xs:
            pts = [(wx, wy) for wy in np.arange(-12, 12.5, 1.0)]
            pc = proj(Hi, pts)
            vals, wpos = [], []
            for (px, py) in pc:
                xi, yi = int(px), int(py)
                if not (0 <= xi < W and 0 <= yi < H): continue
                win = band[max(0,yi-4):yi+5, max(0,xi-30):min(W,xi+31)]
                if win.size == 0: continue
                m = win.mean(0); j = int(np.argmax(m))
                if m[j] < 3: continue
                q = np.array([max(0,xi-30)+j, yi, 1.0]) @ Hn.T
                xs_found.append(float(q[0]/q[2]))
            if len(xs_found) >= 6:
                out[name if len(xs)==1 else ("blue_L" if wx<0 else "blue_R")] = (float(np.median(xs_found)), len(xs_found))
    return out

results = {}
rejected = []
for stem in all_stems:
    if stem in pool:
        results[stem] = {"H": pool[stem][0].tolist(), "anchor": True}
        continue
    img = cv2.imread(str(BASE/"wide_fg"/f"{stem}.jpg"))
    if img is None: continue
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp2, des2 = kpts(g)
    if des2 is None or len(kp2) < 30:
        rejected.append((stem, "no features")); continue
    cand = sorted(pool, key=lambda s: abs(by_time[s]-by_time[stem]))[:8]
    best = None
    for s in cand:
        H_seed, kp1, des1 = pool[s]
        if des1 is None: continue
        # seed frame ice mask: world->pixel = inv(H_seed)
        ice = proj(np.linalg.inv(H_seed), RINK_CORNERS)
        m = np.zeros((720, 1280), np.uint8)
        cv2.fillConvexPoly(m, np.int32(ice), 1)
        m = cv2.dilate(m, np.ones((25,25), np.uint8))
        m = cv2.erode(m, np.ones((25,25), np.uint8))
        idx1 = [i for i,k in enumerate(kp1) if 0<=int(k.pt[1])<720 and 0<=int(k.pt[0])<1280 and m[int(k.pt[1]),int(k.pt[0])]>0]
        if len(idx1) < 20: continue
        d1 = des1[idx1]; k1s = [kp1[i] for i in idx1]
        m2 = matcher.knnMatch(d1, des2, k=2)
        good = [(a,b) for a,b in m2 if a.distance < 0.75*b.distance]
        if len(good) < 15: continue
        src = np.float32([k1s[a.queryIdx].pt for a,b in good])
        dst = np.float32([kp2[a.trainIdx].pt for a,b in good])
        M, mask = cv2.findHomography(dst, src, cv2.RANSAC, 3.0)
        if M is None: continue
        inl = int(mask.ravel().sum())
        if inl >= MIN_INL and (best is None or inl > best[1]):
            best = (M, inl, s)
    if best is None:
        rejected.append((stem, "no ice matching")); continue
    M, inl, seed = best
    H_new = pool[seed][0] @ M
    # ---- independent control ----
    gray = g
    ye = ice_edge_y(gray)
    ok_edge = None
    if ye is not None:
        q = np.array([640.0, ye, 1.0]) @ H_new.T
        yw = q[1]/q[2]
        ok_edge = EDGE_OK[0] <= yw <= EDGE_OK[1]
        edge_val = round(float(yw),1)
    else:
        edge_val = None
    lc = line_check(img, H_new)
    ok_lines = True
    for k, (v, n) in lc.items():
        expect = {"red":0.0, "blue_L":-7.5, "blue_R":7.5}[k]
        if abs(v-expect) > 1.2: ok_lines = False
    if ok_edge is False or (lc and not ok_lines):
        rejected.append((stem, f"control: edge={edge_val} lines={lc}")); continue
    pool[stem] = [H_new, kp2, des2]
    results[stem] = {"H": H_new.tolist(), "anchor": False, "inl": inl, "seed": seed,
                     "edge_y_world": edge_val, "lines": {k: round(v[0],2) for k,v in lc.items()}}

anch = sum(1 for v in results.values() if v["anchor"])
prop = len(results) - anch
print(f"good H: {len(results)} (anchors {anch}, propagated {prop}) | rejected {len(rejected)}")
for r in rejected[:10]: print("  -", r[0], r[1])
(OUT/"propagate_fs2.json").write_text(json.dumps(results, ensure_ascii=False))
(OUT/"rejected_fs2.json").write_text(json.dumps(dict(rejected), ensure_ascii=False))
HD = OUT/"H2"; HD.mkdir(exist_ok=True)
for s, v in results.items():
    np.save(HD/f"{s}.npy", np.array(v["H"]))
