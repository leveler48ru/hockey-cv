#!/usr/bin/env python3
"""Full-ice panorama: Translation stitching of median frames + homography from 7 anchors.

The camera only pans => the frame relation is ~a pure parallel translation (verified:
stabilization residual 1.4-2.4 px). So manual anchors from different views
transfer into the panorama exactly, and the homography is solved in ONE system.
Median over 25 frames removes players/puck.
"""
import cv2, numpy as np, json, os, sys

BASE = os.path.dirname(os.path.abspath(__file__))
M = BASE + "/match"
F = "./media/9281020430926.mp4"
FPS = 30
SEC_C, SEC_L, SEC_R = 3, 240, 2570


def median_frame(sec, half=2.5, n=25):
    cap = cv2.VideoCapture(F)
    frs = []
    for s in np.linspace(sec - half, sec + half, n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s * FPS))
        ok, fr = cap.read()
        if ok:
            frs.append(fr.astype(np.uint16))
    cap.release()
    return np.median(np.stack(frs), axis=0).astype(np.uint8)


def transl(g0, g1):
    """translation g1->g0: ORB over ice + median displacement of inlier pairs.
    phaseCorrelate is unusable here: pan ~1500 px > half the width -> wrap."""
    m0 = ice_mask_bgr(g0)
    m1 = ice_mask_bgr(g1)
    orb = cv2.ORB_create(8000, 1.2, 8)
    k0, d0 = orb.detectAndCompute(g0, m0)
    k1, d1 = orb.detectAndCompute(g1, m1)
    if d0 is None or d1 is None or len(d0) < 300 or len(d1) < 300:
        raise RuntimeError("too few ice keypoints")
    mm = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d0, d1, k=2)
    good = [a for a, b in mm if a.distance < 0.8 * b.distance]
    p0 = np.float32([k0[x.queryIdx].pt for x in good])
    p1 = np.float32([k1[x.trainIdx].pt for x in good])
    dif = p0 - p1  # shift by which the scene in g1 moves back toward g0
    med = np.median(dif, axis=0)
    keep = np.hypot(*(dif - med).T) < 25
    dx, dy = np.median(dif[keep], axis=0)
    return float(dx), float(dy), float(keep.sum()) / len(good)


def ice_mask_bgr(fr):
    b, g, r = cv2.split(fr.astype(np.int16))
    mx = np.maximum(np.maximum(b, g), r); mn = np.minimum(np.minimum(b, g), r)
    m = ((mx > 165) & ((mx - mn) < 55)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros(fr.shape[:2], np.uint8)
    big = max(cnts, key=cv2.contourArea)
    out = np.zeros(fr.shape[:2], np.uint8)
    cv2.fillConvexPoly(out, cv2.convexHull(big), 255)
    return out


def dlt_plain(full, linec):
    A = []
    for (x, y), (X, Y) in full:
        xn, yn, Xn, Yn = x / 1000.0, y / 1000.0, X / 60.0, Y / 60.0
        A.append([-xn, -yn, -1, 0, 0, 0, Xn * xn, Xn * yn, Xn])
        A.append([0, 0, 0, -xn, -yn, -1, Yn * xn, Yn * yn, Yn])
    for (x, y), c in linec:
        xn, yn, cn = x / 1000.0, y / 1000.0, c / 60.0
        A.append([xn, yn, 1, 0, 0, 0, -cn * xn, -cn * yn, -cn])
    _, S, Vt = np.linalg.svd(np.array(A, float))
    h = Vt[-1]
    Mm = np.array([[h[0] * 60 / 1000, h[1] * 60 / 1000, h[2] * 60],
                   [h[3] * 60 / 1000, h[4] * 60 / 1000, h[5] * 60],
                   [h[6] / 1000, h[7] / 1000, h[8]]])
    return Mm / Mm[2, 2], S


def xf(H, pts):
    p = np.asarray(pts, float).reshape(-1, 2)
    q = np.asarray(H, float) @ np.hstack([p, np.ones((len(p), 1))]).T
    return (q[:2] / q[2:3]).T


def main():
    meds = {}
    for tag, sec in [("c", SEC_C), ("l", SEC_L), ("r", SEC_R)]:
        p = f"{M}/med_{tag}.jpg"
        if os.path.exists(p):
            meds[tag] = cv2.imread(p)
        else:
            meds[tag] = median_frame(sec)
            cv2.imwrite(p, meds[tag], [cv2.IMWRITE_JPEG_QUALITY, 92])
    dxl, dyl, nl = transl(meds["c"], meds["l"])
    dxr, dyr, nr = transl(meds["c"], meds["r"])
    print(f"center->left:  dx={dxl:+.1f} dy={dyl:+.1f} inlier_share={nl:.2f} ({nl*100:.0f}%)")
    print(f"center->right: dx={dxr:+.1f} dy={dyr:+.1f} inlier_share={nr:.2f} ({nr*100:.0f}%)")

    # control check of parallelity: ORB homography over ice of center and left only
    # (if the transfer is honest, H is nearly identity with a shift)
    canvas_w = 1920  # placeholder; real value below
    # canvas: frame pixel x of src maps to panorama x = x + off_src,
    # where off_c = 0, off_l = dxl (left->center transfer already accounted by the transfer), off_r = dxr
    off_c = 0
    off_l = int(round(dxl))
    off_r = int(round(dxr))
    x0 = min(0, off_l, off_r) - 20
    off_l -= x0; off_c -= x0; off_r -= x0
    canvas_w = max(1920, off_l + 1920, off_r + 1920) + 40
    print(f"canvas_w={canvas_w} off_l={off_l} off_c={off_c} off_r={off_r}")

    canvas = np.zeros((1080 + 80, canvas_w, 3), np.uint8)
    wt = np.zeros((1080 + 80, canvas_w), np.float32)
    yoff_c = 40
    yoff_l = yoff_c + int(round(dyl))
    yoff_r = yoff_c + int(round(dyr))

    def put(img, off, yoff, w):
        h = img.shape[0]
        roi_w = wt[yoff:yoff + h, off:off + 1920]
        acc = canvas[yoff:yoff + h, off:off + 1920].astype(np.float32) * roi_w[:, :, None] \
              + img.astype(np.float32) * w
        new = roi_w + w
        np.divide(acc, new[:, :, None], out=np.zeros_like(acc), where=new[:, :, None] > 0)
        canvas[yoff:yoff + h, off:off + 1920] = np.clip(acc, 0, 255).astype(np.uint8)
        wt[yoff:yoff + h, off:off + 1920] = new

    put(meds["l"], off_l, yoff_l, 1.0)
    put(meds["r"], off_r, yoff_r, 1.0)
    put(meds["c"], off_c, yoff_c, 3.0)  # center on top: sharper ice
    cv2.imwrite(f"{M}/panorama.jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 93])
    print("panorama.jpg", canvas.shape)

    # anchors: frame pixel -> panorama (frame was pasted with offset off_*)
    toP = {"c": (off_c, yoff_c), "l": (off_l, yoff_l), "r": (off_r, yoff_r)}
    P = {
        "c1": ((850, 498), (30.0, 15.0), "c"),
        "c2": ((698, 374), (30.0, 30.0), "c"),
        "c3": ((1489, 1079), (30.0, 0.0), "c"),
        "l1": ((405, 415), (22.4, 30.0), "l"),
        "l2": ((0, 634), None, "l"),
        "r1": ((1015, 590), (37.6, 30.0), "r"),
        "r2": ((1650, 750), None, "r"),
    }
    allp = {}
    canvas_w_val = canvas_w
    for k, (px, w, src) in P.items():
        ox, oy = toP[src]
        allp[k] = ((px[0] + ox, px[1] + oy), w)
    print("anchors in the panorama:")
    for k, (px, w) in allp.items():
        print(f"   {k}: ({px[0]:.0f},{px[1]:.0f}) world={w}")
    a2 = np.array(allp["c2"][0]); b2 = np.array(allp["l1"][0]); c2 = np.array(allp["r1"][0])
    d = np.linalg.norm(np.cross(b2 - a2, c2 - a2)) / np.linalg.norm(b2 - a2)
    print(f"COLLINEARITY CHECK of far boards: {d:.1f} px")

    full = [(px, w) for px, w in allp.values() if w is not None]
    linec = [(allp["l2"][0], 22.4), (allp["r2"][0], 37.6)]
    H, S = dlt_plain(full, linec)
    np.save(f"{M}/H_pano.npy", H)
    json.dump({"H_pano": H.tolist(), "offs": {k: list(v) for k, v in toP.items()},
               "canvas_w": canvas_w, "off_l": off_l, "off_c": off_c, "off_r": off_r,
               "dxl": dxl, "dyl": dyl, "dxr": dxr, "dyr": dyr},
              open(f"{M}/H.json", "w"))
    errs = []
    for k, (px, w) in allp.items():
        p = xf(H, [px])[0]
        if w is not None:
            e = float(np.hypot(*(p - np.array(w))))
            errs.append(e)
            print(f"   {k}: world=({p[0]:.2f},{p[1]:.2f}) expected {w} -> {e*100:.0f} cm")
        else:
            cc = 22.4 if k == "l2" else 37.6
            print(f"   {k} (X={cc}): X={p[0]:.2f} Y={p[1]:.2f} -> {abs(p[0]-cc)*100:.0f} cm")
    print(f"MEAN {np.mean(errs)*100:.0f} cm | MAX {max(errs)*100:.0f} cm")

    Hi = np.linalg.inv(H)
    grid = [((x, 0.0), (x, 30.0)) for x in range(0, 61, 2)] + \
           [((0.0, y), (60.0, y)) for y in range(0, 31, 2)]
    vis = canvas.copy()
    for a3, b3 in grid:
        pa, pb = xf(Hi, [a3, b3])
        cv2.line(vis, tuple(pa.astype(int)), tuple(pb.astype(int)), (0, 200, 0), 1)
    for a3, b3, col in [((30, 0), (30, 30), (0, 0, 220)), ((22.4, 0), (22.4, 30), (220, 0, 0)),
                        ((37.6, 0), (37.6, 30), (220, 0, 0)), ((4, 0), (4, 30), (0, 0, 220)),
                        ((56, 0), (56, 30), (0, 0, 220))]:
        pa, pb = xf(Hi, [a3, b3])
        cv2.line(vis, tuple(pa.astype(int)), tuple(pb.astype(int)), col, 3)
    for (cx_, cy_) in [(30, 15), (6, 7.1), (6, 22.9), (54, 7.1), (54, 22.9),
                       (22.4, 7.1), (22.4, 22.9), (37.6, 7.1), (37.6, 22.9)]:
        pts_circle = []
        for t in np.linspace(0, 2 * np.pi, 33):
            r = 4.5 if (cx_, cy_) != (30, 15) else 4.5
            r = 2.0 if cx_ in (22.4, 37.6) else r
            pts_circle.append((cx_ + r * np.cos(t), cy_ + r * np.sin(t)))
        pp = xf(Hi, pts_circle).astype(int)
        cv2.polylines(vis, [pp], True, (0, 160, 255), 2)
        p0 = xf(Hi, [(cx_, cy_)])[0].astype(int)
        cv2.circle(vis, tuple(p0), 6, (0, 160, 255), -1)
    for k, (px, w) in allp.items():
        cv2.circle(vis, (int(px[0]), int(px[1])), 14, (0, 255, 0), 2)
    # goal circles (goal mouth 2.72 m => r=1.36 across)
    for gx in (4, 56):
        pp = xf(Hi, [(gx, 15 - 1.36), (gx, 15 + 1.36)])
        cv2.line(vis, tuple(pp[0].astype(int)), tuple(pp[1].astype(int)), (0, 255, 0), 5)
    cv2.imwrite(f"{M}/panorama_marked.jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("panorama_marked.jpg done")


if __name__ == "__main__":
    main()
