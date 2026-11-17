#!/usr/bin/env python3
"""v3: pan chains ONLY on the ice (color mask) + plain DLT.

Lesson from v2: the center->right ORB chain over the whole frame got attracted
to players/crowd (r1' drifted 256 px, the board wasn't collinear). The ice is a
plane and the background: matching on the ice mask yields an honest pure-rotation
homography.
"""
import cv2, numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solve_h import F, OUT, FPS, SEC_C, SEC_L, SEC_R, frame, xf

HOP = 15


def ice_mask(fr):
    """ice: bright and unsaturated (B>170,G>170,R>170, max-min<50)"""
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
    if cv2.contourArea(big) < 80000:
        return np.zeros(fr.shape[:2], np.uint8)
    out = np.zeros(fr.shape[:2], np.uint8)
    cv2.fillConvexPoly(out, cv2.convexHull(big), 255)
    return out


def masked_orb(fr):
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    kp, des = cv2.ORB_create(8000, 1.2, 8).detectAndCompute(g, ice_mask(fr))
    return kp, des


def pair_H_ice(fr0, fr1):
    k0, d0 = masked_orb(fr0)
    k1, d1 = masked_orb(fr1)
    if d0 is None or d1 is None or len(d0) < 150 or len(d1) < 150:
        return None, 0, 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d0, d1, k=2)
    good = [a for a, b in m if a.distance < 0.85 * b.distance]
    if len(good) < 60:
        return None, len(good), 0
    p0 = np.float32([k0[x.queryIdx].pt for x in good]).reshape(-1, 1, 2)
    p1 = np.float32([k1[x.trainIdx].pt for x in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(p0, p1, cv2.RANSAC, 4.0)
    if H is None:
        return None, len(good), 0
    return H / H[2, 2], len(good), int(mask.sum())


def frame_bgr(sec):
    cap = cv2.VideoCapture(F)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * FPS))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"no frame {sec}")
    return fr


def chain_ice(sec0, sec1, hop=HOP, tag=""):
    Hacc = np.eye(3)
    ref = frame_bgr(sec0)
    ns = int(round((sec1 - sec0) / hop))
    ok = bad = 0
    for i in range(1, ns + 1):
        s = sec1 if i == ns else sec0 + i * hop
        cur = frame_bgr(s)
        H, ng, ni = pair_H_ice(ref, cur)
        if H is not None and ni >= 40:
            Hacc = H @ Hacc
            ref = cur
            ok += 1
        else:
            bad += 1
            print(f"  {tag} step {i}/{ns} (s={s}): FAILED matches={ng} inl={ni}", flush=True)
    print(f"  {tag}: ok={ok} bad={bad}", flush=True)
    return Hacc


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
    M = np.array([[h[0] * 60 / 1000, h[1] * 60 / 1000, h[2] * 60],
                  [h[3] * 60 / 1000, h[4] * 60 / 1000, h[5] * 60],
                  [h[6] / 1000, h[7] / 1000, h[8]]])
    return M / M[2, 2], S


def main():
    print("== ice chains (hop=15s)", flush=True)
    Hcl = chain_ice(SEC_C, SEC_L, tag="L")
    Hcr = chain_ice(SEC_C, SEC_R, tag="R")

    P = {
        "c1": ((850, 498), (30.0, 15.0), "c"),
        "c2": ((698, 374), (30.0, 30.0), "c"),
        "c3": ((1489, 1079), (30.0, 0.0), "c"),
        "l1": ((405, 415), (22.4, 30.0), "l"),
        "l2": ((0, 634), None, "l"),
        "r1": ((1015, 590), (37.6, 30.0), "r"),
        "r2": ((1650, 750), None, "r"),
    }
    toC = {"c": np.eye(3), "l": np.linalg.inv(Hcl), "r": np.linalg.inv(Hcr)}
    allp = {k: (tuple(xf(toC[src], [px])[0]), w) for k, (px, w, src) in P.items()}
    print("anchors in central coordinates:")
    for k, (px, w) in allp.items():
        print(f"   {k}: ({px[0]:.0f},{px[1]:.0f}) world={w}")

    a, b, c_ = np.array(allp["c2"][0]), np.array(allp["l1"][0]), np.array(allp["r1"][0])
    d = np.linalg.norm(np.cross(b - a, c_ - a)) / np.linalg.norm(b - a)
    print(f"CONTROL: r1' off the c2-l1' line (far board) = {d:.1f} px")
    a3 = np.array(allp["c3"][0])
    d3 = np.linalg.norm(np.cross(np.array(allp["l2"][0]) - a3, allp["r1"][0] - a3)) / \
        np.linalg.norm(np.array(allp["l2"][0]) - a3) if allp["l2"][0][0] != a3[0] else -1

    full = [(px, w) for px, w in allp.values() if w is not None]
    linec = [(allp["l2"][0], 22.4), (allp["r2"][0], 37.6)]
    H, S = dlt_plain(full, linec)
    np.save(f"{OUT}/H_center.npy", H)
    json.dump({"H_center": H.tolist(), "Hcl": Hcl.tolist(), "Hcr": Hcr.tolist()},
              open(f"{OUT}/H.json", "w"))
    print("SVD (relative):", np.round(S[:8] / S[0], 4))

    print("residuals:")
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
    cap = cv2.VideoCapture(F)
    grid = [((x, 0.0), (x, 30.0)) for x in range(0, 61, 5)] + \
           [((0.0, y), (60.0, y)) for y in range(0, 31, 5)]
    for tag, sec, HW in [("center", SEC_C, Hi), ("left", SEC_L, Hcl @ Hi), ("right", SEC_R, Hcr @ Hi)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, sec * FPS)
        ok, fr = cap.read()
        if not ok:
            continue
        for a2, b2 in grid:
            pa, pb = xf(HW, [a2, b2])
            cv2.line(fr, tuple(pa.astype(int)), tuple(pb.astype(int)), (0, 255, 0), 2)
        for pt in [(22.4, 0), (22.4, 30), (37.6, 0), (37.6, 30), (4, 15), (56, 15), (30, 15)]:
            p = xf(HW, [pt])[0]
            cv2.circle(fr, tuple(p.astype(int)), 16, (0, 0, 255), 3)
        cv2.imwrite(f"{OUT}/verify_{tag}.jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cap.release()
    print("verify_*.jpg saved")


if __name__ == "__main__":
    main()
