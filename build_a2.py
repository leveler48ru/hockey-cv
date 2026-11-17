#!/usr/bin/env python3
"""Show-material of arena 1 (wide_a2: 104 frames with no markings on the ice) -> dataset_a2.

What it gives the model:
  - board (cls 6): the visible dasher band = boundary of the flooded ice contour,
    taken only where non-ice is adjacent to ice INSIDE the frame (not at the frame edge:
    the near board is not visible in the wide shot);
  - explicit geometry negatives: frames with and without board bands teach that on this ice
    there are neither lines nor circles (classes 0/1/2/3/4/5 are absent from the labels).

Output: dataset_a2/{images,labels}/{train,val}, _manifest.json, QC sheet
a2_board_qc.jpg (polygon projections overlaid on frames).
"""
import cv2, json, shutil
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

BASE = Path(".")
SRC = BASE / "wide_a2"
DS = BASE / "dataset_a2"
BAND_PX = 26          # thickness of the board band along the visible ice boundary
GRAD_K = 5
MIN_BAND_AREA = 900
MAX_POLYS = 5
VAL_EVERY = 6         # every 6th frame -> val (~17%)

for kind in ("images", "labels"):
    for split in ("train", "val"):
        (DS / kind / split).mkdir(parents=True, exist_ok=True)

def fill_holes(m):
    inv = cv2.bitwise_not(m)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    h, w = m.shape
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)
    out = m.copy()
    for lbl in range(1, n):
        if lbl not in border:
            out[lab == lbl] = 255
    return out

def ice_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v, s = hsv[:, :, 2], hsv[:, :, 1]
    m = ((v > 120) & (s < 90)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    m[:int(img.shape[0] * 0.16), :] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[best, cv2.CC_STAT_AREA] < 0.15 * img.shape[0] * img.shape[1]:
        return None
    return (lab == best).astype(np.uint8) * 255

def board_polys(m_filled, img_shape):
    h, w = img_shape[:2]
    er = cv2.erode(m_filled, np.ones((GRAD_K, GRAD_K), np.uint8))
    grad = cv2.bitwise_and(m_filled, cv2.bitwise_not(er))
    # frame borders are not a board (the near edge of the ice goes beyond the bottom of the frame)
    grad[:4, :] = 0; grad[-4:, :] = 0; grad[:, :4] = 0; grad[:, -4:] = 0
    if grad.sum() == 0:
        return []
    band = cv2.bitwise_and(m_filled,
                           cv2.dilate(grad, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BAND_PX, BAND_PX))))
    cnts, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:MAX_POLYS]
    out = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_BAND_AREA:
            continue
        c = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(c) < 4:
            continue
        pts = c.astype(float)
        pts[:, 0] /= w; pts[:, 1] /= h
        out.append(pts)
    return out

manifest = {}
n_train = n_val = n_neg = 0
qc_tiles = []
stems = sorted(p.stem for p in SRC.glob("*.jpg"))
for i, stem in enumerate(stems):
    img = cv2.imread(str(SRC / f"{stem}.jpg"))
    if img is None:
        continue
    h, w = img.shape[:2]
    m = ice_mask(img)
    if m is None:
        n_neg += 1  # almost no ice — pure negative (crowd/flag scenes)
        polys = []
    else:
        polys = board_polys(fill_holes(m), img.shape)
    split = "val" if i % VAL_EVERY == 0 else "train"
    lines = []
    for pts in polys:
        lines.append("6 " + " ".join(f"{x:.5f} {y:.5f}" for x, y in pts))
    shutil.copy2(SRC / f"{stem}.jpg", DS / "images" / split / f"{stem}.jpg")
    (DS / "labels" / split / f"{stem}.txt").write_text(("\n".join(lines) + "\n") if lines else "")
    manifest[stem] = {"split": split, "boards": len(polys)}
    if split == "train":
        n_train += 1
    else:
        n_val += 1
    if len(qc_tiles) < 8 and (polys or i % 25 == 0):
        vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(vis)
        d2 = ImageDraw.Draw(pil)
        for pts in polys:
            pp = [(x * w, y * h) for x, y in pts]
            d2.line(pp + [pp[0]], fill=(0, 255, 0), width=3)
        qc_tiles.append((stem, pil.resize((480, 270))))

(DS / "_manifest.json").write_text(json.dumps(
    {"created": "2026-11-08", "source": "wide_a2 (222222, show ice of arena 1, unmarked)",
     "train": n_train, "val": n_val, "no_ice": n_neg, "frames": manifest}, indent=1))
print(f"train {n_train} | val {n_val} | no-ice negatives {n_neg} | total frames {len(manifest)}")
total_b = sum(v["boards"] for v in manifest.values())
print("board polygons:", total_b)

cols, TW, TH = 4, 480, 270
rows = (len(qc_tiles) + 3) // 4
sheet = Image.new("RGB", (cols * TW, rows * (TH + 18)), (15, 15, 15))
d = ImageDraw.Draw(sheet)
for i, (s, t) in enumerate(qc_tiles):
    r, c = divmod(i, cols)
    sheet.paste(t, (c * TW, r * (TH + 18)))
    d.text((c * TW + 5, r * (TH + 18) + TH + 2), s, fill=(255, 210, 120))
sheet.save(BASE / "a2_board_qc.jpg", quality=86)
print("QC: a2_board_qc.jpg")
