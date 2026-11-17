#!/usr/bin/env python3
"""H for a new venue from the ice contour (classical CV), no ORB and no model.

Ice = the largest bright low-saturation connected component; its minAreaRect is
the rink rectangle (IIHF rounded corners fit exactly into ±30/±15 of the
bounding rect). 4 corners -> DLT -> H. The mirror ambiguity (world x) and
near/far are resolved by: (a) frame geometry (bottom long side = near),
(b) QC-SCORE: redness along the projected x=0 vs control strips, blueness along
x=±7.5, and agreement with 10 manual anchors when the frame is an anchor.

Output: fs_calib/Hq/<stem>.npy (pix->world), fs_calib/qc_quad.json, QC overlay."""
import json
import numpy as np, cv2
from pathlib import Path
from PIL import Image, ImageDraw

BASE = Path(".")
FG = BASE / "wide_fg"
OUT = BASE / "fs_calib"
HQ = OUT / "Hq"; HQ.mkdir(exist_ok=True)

WORLD = np.array([[-30.0, -15.0], [30.0, -15.0], [30.0, 15.0], [-30.0, 15.0]])  # near-L near-R far-R far-L
FITS = json.loads((OUT / "fits.json").read_text())

def ice_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v, s = hsv[:, :, 2], hsv[:, :, 1]
    m = ((v > 120) & (s < 90)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    m[:int(img.shape[0] * 0.18), :] = 0  # ceiling / top of the stands
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2: return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[best, cv2.CC_STAT_AREA] < 0.18 * img.shape[0] * img.shape[1]:
        return None
    return (lab == best).astype(np.uint8) * 255

def quad_corners(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    if min(rect[1]) < 60: return None
    box = cv2.boxPoints(rect)  # 4x2
    # order: sorted by y, then near/far, then x
    box = box[np.argsort(box[:, 1])]
    top, bottom = box[:2], box[2:]
    top = top[np.argsort(top[:, 0])]; bottom = bottom[np.argsort(bottom[:, 0])]
    near_L, near_R = bottom  # the bottom two (larger v)
    far_R, far_L = top[1], top[0]
    return np.array([near_L, near_R, far_R, far_L], float)

def redness_blue(img):
    a = img.astype(float); B, G, R = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    return (R - (B + G) / 2), (B - (R + G) / 2)

def line_score(H, img, wx, kind='red', yspan=(-13, 13), halfwin=2.2):
    """color signal along world vertical x=wx vs control strips at ±(4..6) m."""
    red, blue = redness_blue(img)
    band = red if kind == 'red' else blue
    Hi = np.linalg.inv(H)
    h, w = img.shape[:2]
    ys = np.linspace(*yspan, 40)
    hits, ctrl = [], []
    for wy in ys:
        q = np.array([wx, wy, 1.0]) @ Hi.T; u, v = q[0] / q[2], q[1] / q[2]
        if not (60 <= u < w - 60 and 40 <= v < h - 10): continue
        for tag, dx in (('hit', 0.0), ('c1', -halfwin - 1.0), ('c2', halfwin + 1.0)):
            q2 = np.array([wx + dx, wy, 1.0]) @ Hi.T; u2, v2 = q2[0] / q2[2], q2[1] / q2[2]
            if not (0 <= u2 < w and 0 <= v2 < h): continue
            val = band[int(max(0, v2 - 3)):int(min(h, v2 + 4)),
                        int(max(0, u2 - 2)):int(min(w, u2 + 3))]
            if val.size == 0: continue
            (hits if tag == 'hit' else ctrl).append(val.mean())
    if len(hits) < 8 or len(ctrl) < 12: return None, 0
    return float(np.median(hits) - np.median(ctrl)), len(hits)

def anchor_agree(H, stem):
    if stem not in FITS: return None
    Ha = np.array(FITS[stem]['H'])
    pts = [(0, 0), (0, -15), (0, 15), (-7.5, 15), (7.5, -15)]
    d = []
    for xy in pts:
        q1 = np.array([*xy, 1.0]) @ H.T; q1 = q1[:2] / q1[2]
        q2 = np.array([*xy, 1.0]) @ Ha.T; q2 = q2[:2] / q2[2]
        d.append(np.linalg.norm(q1 - q2))
    return float(np.median(d))

def mirror_score(H, img):
    """consistency score: blue lines at ±7.5, red line at 0."""
    sr, nr = line_score(H, img, 0.0, 'red')
    sb, nb = line_score(H, img, 7.5, 'blue')
    sbl, _ = line_score(H, img, -7.5, 'blue')
    vals = [v for v in (sr, sb, sbl) if v is not None]
    if not vals: return None, (sr, sb, sbl)
    return float(np.mean(vals)), (sr, sb, sbl)

res = {}
imgs = sorted(FG.glob('*.jpg'))
for p in imgs:
    stem = p.stem
    img = cv2.imread(str(p))
    if img is None: continue
    m = ice_mask(img)
    rec = {'fail': None}
    if m is None:
        res[stem] = {'fail': 'no ice'}; continue
    corners = quad_corners(m)
    if corners is None:
        res[stem] = {'fail': 'no quad'}; continue
    # all 4 corners inside the frame with a margin
    if not ((corners[:, 0] > 5).all() & (corners[:, 0] < 1275).all() &
            (corners[:, 1] > 5).all() & (corners[:, 1] < 715).all()):
        res[stem] = {'fail': 'corners out'}; continue
    # 2 mirror hypotheses + (near/far already fixed by frame geometry)
    best = None
    for mirror in (False, True):
        world = WORLD.copy()
        if mirror:
            world = np.array([[30, -15], [-30, -15], [-30, 15], [30, 15]])
        H, _ = cv2.findHomography(corners.astype(np.float32), world.astype(np.float32), 0)
        if H is None: continue
        # world y near: the bottom edge must give y≈-15 (not +15): check
        q = np.array([640.0, corners[[0, 1], 1].mean(), 1.0]) @ H.T
        y_near = q[1] / q[2]
        if y_near > 0:
            continue
        ms, detail = mirror_score(H, img)
        ag = anchor_agree(H, stem)
        score = (ms if ms is not None else -99) - (0 if ag is None else min(ag, 200) / 25)
        cand = (score, H, ms, ag, detail)
        if best is None or score > best[0]: best = cand
    if best is None:
        res[stem] = {'fail': 'no orient'}; continue
    score, H, ms, ag, detail = best
    ok = ms is not None and ms > 2.0
    rec = {'score': round(score, 2), 'line_ms': None if ms is None else round(ms, 1),
           'anchor_agree_px': None if ag is None else round(ag, 1),
           'per_line': [None if x is None else round(x, 1) for x in detail],
           'ok': bool(ok), 'mirror': True}
    if ok:
        np.save(HQ / f'{stem}.npy', H)
    res[stem] = rec

n_ok = sum(1 for v in res.values() if v.get('ok'))
n_anchor_ok = sum(1 for s, v in res.items() if v.get('ok') and s in FITS)
print(f"frames: {len(res)} | usable H: {n_ok}")
from collections import Counter
print("failure reasons:", dict(Counter(v.get('fail', 'low score') for v in res.values() if not v.get('ok'))))
ags = [v['anchor_agree_px'] for v in res.values() if v.get('anchor_agree_px') is not None]
if ags: print(f"anchor agreement px: n={len(ags)} median {np.median(ags):.0f} p90 {np.percentile(ags,90):.0f}")
(OUT / 'qc_quad.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))

# QC overlay: 8 usable (incl. anchors) + 4 failed with a near-orientation error
def W2P(H, x, y):
    q = np.array([x, y, 1.0]) @ np.linalg.inv(H).T
    return (float(q[0] / q[2]), float(q[1] / q[2]))
GEOM = [((0, -15, 0, 15), (0, 0, 255)), ((-7.5, -15, -7.5, 15), (255, 140, 0)),
        ((7.5, -15, 7.5, 15), (255, 140, 0))]
def crease_pts(gx, sign):
    a = np.linspace(-np.pi/2, np.pi/2, 14)
    return [W2P(H, gx - sign*1.9*np.cos(t), sign*1.9*np.sin(t)) for t in a]
H = None
tiles = []
ok_stems = [s for s, v in res.items() if v.get('ok')]
pick = ([s for s in ok_stems if s in FITS][:4] + ok_stems[:8])[:8]
for s in pick:
    H = np.load(HQ / f'{s}.npy')
    img = cv2.imread(str(FG / f'{s}.jpg'))
    for ax, ay, bx, by, col in [(0,-15,0,15,(0,0,255)),(-7.5,-15,-7.5,15,(255,140,0)),(7.5,-15,7.5,15,(255,140,0)),
                                 (-30,-15,30,-15,(0,255,0)),(-30,15,30,15,(0,255,0))]:
        pts = [W2P(H, x, y) for x, y in np.linspace(np.array([ax,ay]), np.array([bx,by]), 20)]
        pts = [p for p in pts if -50 <= p[0] <= 1330 and -50 <= p[1] <= 770]
        if len(pts) >= 2: cv2.polylines(img, [np.int32(pts).reshape(-1,1,2)], False, col, 2)
    circ = [W2P(H, 4.5*np.cos(t), 4.5*np.sin(t)) for t in np.linspace(0, 2*np.pi, 36)]
    cv2.polylines(img, [np.int32(circ).reshape(-1,1,2)], True, (255,255,255), 2)
    for gx, sg in ((-26, 1), (26, -1)):
        pts = crease_pts(gx, sg)
        pts = [p for p in pts if -50 <= p[0] <= 1330 and -50 <= p[1] <= 770]
        if len(pts) >= 2: cv2.polylines(img, [np.int32(pts).reshape(-1,1,2)], False, (0,255,255), 2)
    tiles.append(s)
if tiles:
    cols, TW, TH = 4, 480, 270
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows*(TH+18), cols*TW, 3), 15, np.uint8)
    for i, s in enumerate(tiles):
        im = cv2.resize(cv2.imread(str(FG / f'{s}.jpg')), (TW, TH))
        r, c = divmod(i, cols)
        sheet[r*(TH+18):r*(TH+18)+TH, c*TW:c*TW+TW] = im
        cv2.putText(sheet, s, (c*TW+4, r*(TH+18)+TH+13), 0, 0.45, (200,200,200), 1)
    cv2.imwrite(str(BASE / 'fs_qc_quad.jpg'), sheet)
print("QC: fs_qc_quad.jpg")
