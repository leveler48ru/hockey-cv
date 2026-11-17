#!/usr/bin/env python3
"""Honest measurement of pan displacements: ORB + phase correlation ONLY over ice,
with exclusions: (1) static TV overlay (55% of pixels unchanged between
extreme frames - it was what produced du=0), (2) motion blur (median frames)."""
import cv2, numpy as np, json, os
BASE = os.path.dirname(os.path.abspath(__file__)); M = BASE + "/match"
F = "./media/9281020430926.mp4"; FPS = 30
SEC_C, SEC_L, SEC_R = 3, 240, 2570
Y0, Y1 = 330, 1000  # ice/rink band

def med(sec, half=1.0, n=13):
    cap = cv2.VideoCapture(F); frs = []
    for s in np.linspace(sec - half, sec + half, n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s * FPS)); ok, fr = cap.read()
        if ok: frs.append(fr.astype(np.uint16))
    cap.release()
    return np.median(np.stack(frs), axis=0).astype(np.uint8)

def gray_of(sec, path):
    if os.path.exists(path):
        im = cv2.imread(path)
        if im is not None: return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    g = cv2.cvtColor(med(sec), cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path, g); return g

GC = gray_of(SEC_C, f"{M}/gy_c.png")
GL = gray_of(SEC_L, f"{M}/gy_l.png")
GR = gray_of(SEC_R, f"{M}/gy_r.png")

# static mask: barely changes between c, l, r -> overlay/logo/ticker
st = np.std(np.stack([g.astype(np.float32) for g in (GC, GL, GR)]), axis=0)
STATIC = (st < 6)
ICE = np.zeros(GC.shape, bool); ICE[Y0:Y1, :] = True
GOOD = ICE & ~STATIC
print(f"static {(STATIC).mean()*100:.0f}% | working ice window {(GOOD).mean()*100:.0f}% of frame", flush=True)
cv2.imwrite(f"{M}/ice_mask.png", (GOOD.astype(np.uint8) * 255))

orb = cv2.ORB_create(8000, 1.2, 8)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

def shift(a, b, mask):
    """scene shift a->b (point a+x lands in b) over mask pixels only"""
    mk = mask.astype(np.uint8) * 255
    ka, da = orb.detectAndCompute(a, mk)
    kb, db = orb.detectAndCompute(b, mk)
    if da is None or db is None or min(len(da), len(db)) < 100: return None
    mm = bf.knnMatch(da, db, k=2)
    good = [x for x, y in mm if x.distance < 0.75 * y.distance]
    if len(good) < 40: return None
    pa = np.float32([ka[x.queryIdx].pt for x in good])
    pb = np.float32([kb[x.trainIdx].pt for x in good])
    d = pb - pa
    m = np.median(d, axis=0)
    keep = (np.abs(d[:, 0] - m[0]) < 25) & (np.abs(d[:, 1] - m[1]) < 25)
    if keep.sum() < 30: return None
    mu = np.median(d[keep], axis=0)
    return round(float(mu[0]), 1), round(float(mu[1]), 1), int(keep.sum()), len(good)

def phase(a, b):
    z1 = a.astype(np.float32).copy(); z2 = b.astype(np.float32).copy()
    for z in (z1, z2):
        z[~GOOD] = 0.0
        z[:Y0] = 0.0; z[Y1:] = 0.0
    w = np.hanning(z1.shape[0])[:, None] * np.hanning(z1.shape[1])[None, :]
    fa = np.fft.rfft2(z1 * w); fb = np.fft.rfft2(z2 * w)
    R = fa * np.conj(fb)
    cc = np.fft.fftshift(np.fft.irfft2(R / (np.abs(R) + 1e-9), s=z1.shape))
    cc[:Y0] = 0
    dy, dx = np.unravel_index(np.argmax(cc), cc.shape)
    cy, cx = cc.shape[0] // 2, cc.shape[1] // 2
    return int(dx - cx), int(dy - cy), float(cc.max())

res = {}
for tag, g in (("l", GL), ("r", GR)):
    s = shift(GC, g, GOOD)
    px, py, pk = phase(GC, g)
    print(f"C->{tag}: ORB du/dv={s[:2]} inliers={s[2] if s else 0}/{s[3] if s else 0} | phase dx={px:+d} dy={py:+d} peak={pk:.3f}", flush=True)
    res[tag] = {"orb": s, "phase": [px, py, pk]}

# chains in small 10 s steps (no blur: single frames, but masked)
def chain(sec0, sec1, hop):
    cap = cv2.VideoCapture(F)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec0 * FPS)); ok, fr = cap.read()
    cap.release()
    if not ok: return None
    ref = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    u = v = 0.0; ok = bad = 0; s = sec0
    while s < sec1:
        s2 = min(sec1, s + hop)
        cap = cv2.VideoCapture(F)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s2 * FPS)); got, fr2 = cap.read()
        cap.release()
        if not got: bad += 1; s = s2; continue
        g2 = cv2.cvtColor(fr2, cv2.COLOR_BGR2GRAY)
        r = shift(ref, g2, GOOD)
        if r: u += r[0]; v += r[1]; ref = g2; ok += 1
        else: bad += 1
        s = s2
    return u, v, ok, bad

for tag, sec, hop in (("l", SEC_L, 10), ("r", SEC_R, 30)):
    r = chain(SEC_C, sec, hop)
    print(f"chain C->{tag} (hop {hop}s): du={r[0]:+.0f} dv={r[1]:+.0f} ok={r[2]} bad={r[3]}", flush=True)
    res[tag]["chain"] = list(r[:2])
json.dump(res, open(f"{M}/pan_measure.json", "w"), indent=1)
print("done", flush=True)
