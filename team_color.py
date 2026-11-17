#!/usr/bin/env python3
"""Team labeling by jersey color.

Idea: both teams are almost always visible in a single frame, so colors must be
clustered WITHIN the frame rather than by absolute thresholds (lighting, glare,
black equipment, gloves). Each player's torso color -> kmeans(k=2) on hue
(circular hue centering) -> two clusters = two teams. The cluster whose hue is
closer to red gets the red label, the one closer to blue - blue.
If the frame effectively has only one cluster (all players from one team),
the frame is marked ambiguous and excluded from training.

Output: teams.json {frame_name: {"red": [...idx], "blue": [...]}} + summary.
"""
import glob, os, json, argparse, collections
import numpy as np
import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(BASE, "dataset_real_v5")


def torso_color(img, poly):
    H, W = img.shape[:2]
    m = cv2.fillPoly(np.zeros((H, W), np.uint8), [poly], 1)
    ys, xs = np.nonzero(m)
    if len(xs) < 40:
        return None
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    if (y2 - y1) < 14 or (x2 - x1) < 8:
        return None
    # torso: top 45% of the box, central 55% of the width
    ty1 = y1
    ty2 = y1 + int((y2 - y1) * 0.45)
    tx1 = x1 + int((x2 - x1) * 0.22)
    tx2 = x1 + int((x2 - x1) * 0.78)
    if ty2 - ty1 < 6 or tx2 - tx1 < 4:
        return None
    cm = m[ty1:ty2, tx1:tx2]
    crop = img[ty1:ty2, tx1:tx2]
    px = crop[cm > 0]
    if len(px) < 25:
        return None
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    # take the median of each channel; hue is a circular median
    h = np.median(hsv[:, 0]) * 2.0
    s = float(np.median(hsv[:, 1]))
    v = float(np.median(hsv[:, 2]))
    return h, s, v


def circ_hue_dist(h1, h2):
    d = abs(h1 - h2) % 360
    return min(d, 360 - d)


def kmeans2(hues):
    """2-means on hue (circular). Returns centroids (c1,c2)."""
    a = np.array(hues, dtype=float)
    if len(a) == 1:
        return a[0], a[0]
    # init: the two extreme points by difference
    srt = np.sort(a)
    c1, c2 = srt[0], srt[-1]
    for _ in range(30):
        d1 = np.array([circ_hue_dist(x, c1) for x in a])
        d2 = np.array([circ_hue_dist(x, c2) for x in a])
        lab = (d2 < d1).astype(int)
        if lab.sum() == 0 or lab.sum() == len(a):
            break
        n1, n2 = a[lab == 0], a[lab == 1]
        c1n, c2n = np.mean(n1), np.mean(n2)
        if abs(c1n - c1) < 0.5 and abs(c2n - c2) < 0.5:
            c1, c2 = c1n, c2n
            break
        c1, c2 = c1n, c2n
    return c1, c2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default=DS)
    ap.add_argument("--out", default=os.path.join(BASE, "teams.json"))
    ap.add_argument("--min-sat", type=float, default=25.0, help="minimum saturation, otherwise the jersey is black/gray")
    ap.add_argument("--min-val", type=float, default=35.0)
    ap.add_argument("--sep", type=float, default=25.0, help="minimum distance between hue clusters")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = {}
    stats = collections.Counter()
    files = []
    for split in ("train", "val"):
        for lp in sorted(glob.glob(os.path.join(args.ds, "labels", split, "*.txt"))):
            files.append((split, lp))
    if args.limit:
        files = files[: args.limit]

    for split, lp in files:
        name = os.path.basename(lp)[:-4]
        ip = os.path.join(args.ds, "images", split, name + ".jpg")
        img = cv2.imread(ip)
        if img is None:
            continue
        H, W = img.shape[:2]
        feats = []          # (line_idx, hue, sat, val)
        line_idx = 0
        for line in open(lp):
            p = line.split()
            if not p or p[0] != "7":
                continue
            li = line_idx
            line_idx += 1
            v = np.array([float(x) for x in p[1:]])
            poly = (v.reshape(-1, 2) * np.array([W, H])).astype(np.int32)
            c = torso_color(img, poly)
            if c is None:
                continue
            feats.append((li, c[0], c[1], c[2]))

        if len(feats) < 2:
            stats["too few players"] += 1
            continue
        # keep only saturated (real jersey color)
        good = [f for f in feats if f[2] >= args.min_sat and f[3] >= args.min_val]
        if len(good) < 2:
            stats["no saturated colors"] += 1
            continue
        c1, c2 = kmeans2([f[1] for f in good])
        if circ_hue_dist(c1, c2) < args.sep:
            stats["clusters merged"] += 1
            continue
        red_c = c1 if circ_hue_dist(c1, 0) <= circ_hue_dist(c2, 0) else c2
        blu_c = c2 if red_c == c1 else c1
        red, blue = [], []
        for idx, h, s, v in feats:
            if s < args.min_sat or v < args.min_val:
                continue
            d_red = circ_hue_dist(h, red_c)
            d_blu = circ_hue_dist(h, blu_c)
            if min(d_red, d_blu) > 60:      # don't assign to either team
                continue
            (red if d_red < d_blu else blue).append(idx)
        if not red or not blue:
            stats["one team in frame"] += 1
            continue
        out[name] = {"red": red, "blue": blue}
        stats["labeled"] += 1
        stats["players"] += len(red) + len(blue)

    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(dict(stats), ensure_ascii=False, indent=1))
    # distribution by frame size (after clustering, if brightness drops - noise)
    print("frames:", len(out))


if __name__ == "__main__":
    main()
