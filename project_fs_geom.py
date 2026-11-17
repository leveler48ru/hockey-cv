#!/usr/bin/env python3
"""Project world rink geometry into YOLO-seg labels for calibrated fs frames.
Classes: 0 red_line, 1 blue_line, 2 center_circle, 5 goal_crease, 6 board.
Zone faceoff (3/4) EXCLUDED - that venue has show circles there (Alexander's rule).
Output: dataset_fs/{images/labels}/{train,val} + QC overlays."""
import json, shutil
import numpy as np, cv2
from pathlib import Path
from PIL import Image, ImageDraw

BASE = Path(".")
OUT = BASE / "dataset_fs"
CAL = {k: {"anchor": True} for k in json.loads((BASE / "fs_calib" / "fits.json").read_text())}
HDIR = BASE / "fs_calib" / "H"

W_IMG, H_IMG = 1280, 720  # wide_fg 1280x720 (will verify on the first frame)

def strip_poly(ax, ay, bx, by, hw, n=24):
    dx, dy = bx-ax, by-ay
    L = np.hypot(dx, dy); nx, ny = -dy/L, dx/L
    pts = [(ax+dx*t+nx*hw, ay+dy*t+ny*hw) for t in np.linspace(0,1,n)]
    pts += [(ax+dx*t-nx*hw, ay+dy*t-ny*hw) for t in np.linspace(1,0,n)]
    return np.array(pts, np.float64)

def ring_poly(cx, cy, r, w, n=72):
    a = np.linspace(0, 2*np.pi, n, endpoint=False)
    o = np.stack([cx+r*np.cos(a), cy+r*np.sin(a)], 1)
    i = np.stack([cx+(r-w)*np.cos(a[::-1]), cy+(r-w)*np.sin(a[::-1])], 1)
    return np.concatenate([o, i]).astype(np.float64)

def crease_poly(gx, sign):
    # centre-oriented metres: goal line x=±26, semicircle faces the centre
    a = np.linspace(-np.pi/2, np.pi/2, 25)
    p1 = [(gx - sign*1.9*np.cos(t), sign*1.9*np.sin(t)) for t in a]
    p2 = [(gx - sign*1.8*np.cos(t), sign*1.8*np.sin(t)) for t in a[::-1]]
    return np.array(p1+p2, np.float64)

# IMPORTANT: H from fs_calib carries pixel -> world with rink centre at (0,0)
# (platform convention: center=(0,0), x±30, y±15). GEOM is in the same system.
GEOM = [
    (0, strip_poly(0, -15, 0, 15, 0.3)),        # red line
    (1, strip_poly(-7.5, -15, -7.5, 15, 0.15)), # left blue line
    (1, strip_poly(7.5, -15, 7.5, 15, 0.15)),   # right blue line
    (2, ring_poly(0, 0, 4.5, 0.15)),            # center circle
    (5, crease_poly(-26, +1)),                  # left crease
    (5, crease_poly(26, -1)),                   # right crease
    (6, strip_poly(-30, -15, 30, -15, 0.25)),   # near board
    (6, strip_poly(-30, 15, 30, 15, 0.25)),     # far board
    (6, strip_poly(-30, -15, -30, 15, 0.25)),   # left board
    (6, strip_poly(30, -15, 30, 15, 0.25)),     # right board
]

MIN_PIX_AREA = 120  # smaller polygon - don't write (truncated / off frame)

for kind in ("images", "labels"):
    for split in ("train", "val"):
        p = OUT / kind / split
        p.mkdir(parents=True, exist_ok=True)

stats = {"frames": 0, "polys": 0, "by_class": {}}
skipped_small_img = 0
val_share = 0.22
qc_tiles = []
rng = np.random.default_rng(42)

calibrated = sorted([s for s in CAL if "seed" in CAL[s] or CAL[s].get("anchor")])
# fs1/fs2/fs3 are interleaved in time - take val frames from the end of each series,
# so neighbouring duplicates don't leak through the ORB chain
val_set = set()
for vid in ("fs1", "fs2", "fs3"):
    ss = [s for s in calibrated if s.startswith(vid)]
    n_val = int(len(ss) * val_share)
    val_set.update(ss[-n_val:])  # series tails go to val

for stem in calibrated:
    hf = HDIR / f"{stem}.npy"
    if not hf.exists():
        continue
    H = np.load(hf)  # pix->world
    img_p = BASE / "wide_fg" / f"{stem}.jpg"
    im = cv2.imread(str(img_p))
    if im is None:
        continue
    h, w = im.shape[:2]
    lines = []
    for cls, poly in GEOM:
        q = (np.hstack([poly, np.ones((len(poly), 1))]) @ np.linalg.inv(H).T)
        q = q[:, :2] / q[:, 2:3]
        inside = (q[:, 0] > 0) & (q[:, 0] < w) & (q[:, 1] > 0) & (q[:, 1] < h)
        if inside.sum() < len(q) * 0.85:   # partly off frame - don't write (crop artifacts)
            continue
        area = abs(cv2.contourArea(q.astype(np.float32)))
        if area < MIN_PIX_AREA:
            continue
        qc = np.clip(q, [0, 0], [w-1, h-1])
        pts = " ".join(f"{x/w:.5f} {y/h:.5f}" for x, y in qc)
        lines.append(f"{cls} {pts}")
        stats["by_class"][cls] = stats["by_class"].get(cls, 0) + 1
        stats["polys"] += 1
    if not lines:
        continue
    split = "val" if stem in val_set else "train"
    shutil.copy2(img_p, OUT / "images" / split / f"{stem}.jpg")
    (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    stats["frames"] += 1
    # QC tiles: 6 random + all with crease near the edges
    if len(qc_tiles) < 8 and rng.random() < 0.08:
        vis = im.copy()
        for cls, poly in GEOM:
            q = (np.hstack([poly, np.ones((len(poly),1))]) @ np.linalg.inv(H).T)
            q = (q[:, :2]/q[:, 2:3]).astype(np.int32)
            col = {0:(0,0,255),1:(255,120,0),2:(255,255,255),5:(0,255,255),6:(0,255,0)}[cls]
            cv2.polylines(vis, [q.reshape(-1,1,2)], True, col, 2)
        tcv = cv2.resize(vis, (480, 270))
        qc_tiles.append((stem, tcv))

print(json.dumps(stats, ensure_ascii=False))
print(f"train: {len(list((OUT/'images'/'train').glob('*.jpg')))} | val: {len(list((OUT/'images'/'val').glob('*.jpg')))}")

if qc_tiles:
    cols = 4; rows = (len(qc_tiles)+3)//4
    sheet = np.full((rows*290, cols*480, 3), 15, np.uint8)
    for i, (name, t) in enumerate(qc_tiles):
        r, c = divmod(i, cols)
        sheet[r*290:r*290+270, c*480:c*480+480] = t
        cv2.putText(sheet, name, (c*480+5, r*290+285), 0, 0.5, (200,200,200), 1)
    Image.fromarray(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB)).save(BASE/"fs_geom_qc.jpg", quality=85)
    print("QC: fs_geom_qc.jpg")
