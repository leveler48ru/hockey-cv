#!/usr/bin/env python3
"""Ice->pixel homography from 7 manual anchors taken across THREE pan views.

Logic: the camera rotates, so the relation between frames is a pure-rotation
homography, valid for the whole scene. Points from the left/right frame are
transformed into the central frame's coordinate system; we solve a single
normalized DLT: 5 full points (10 equations) + 2 "point on line X=const"
constraints (2 equations).

World: IIHF 60x30, X along the rink (0 = left goal line), Y: 0 = near boards.
"""
import cv2, numpy as np, json, os, sys

F = "./media/9281020430926.mp4"
OUT = os.path.dirname(os.path.abspath(__file__)) + "/match"
FPS = 30
SEC_C, SEC_L, SEC_R = 3, 240, 2570
HOP_L, HOP_R = 10, 30

orb = cv2.ORB_create(5000, 1.2, 8)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def frame(sec, gray_only=True):
    cap = cv2.VideoCapture(F)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * FPS))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"no frame {sec}")
    return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if gray_only else fr


def pair_H(g0, g1):
    k0, d0 = orb.detectAndCompute(g0, None)
    k1, d1 = orb.detectAndCompute(g1, None)
    if d0 is None or d1 is None or len(d0) < 200 or len(d1) < 200:
        return None, 0
    m = bf.knnMatch(d0, d1, k=2)
    good = [a for a, b in m if a.distance < 0.78 * b.distance]
    if len(good) < 80:
        return None, len(good)
    p0 = np.float32([k0[x.queryIdx].pt for x in good]).reshape(-1, 1, 2)
    p1 = np.float32([k1[x.trainIdx].pt for x in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(p0, p1, cv2.RANSAC, 4.0)
    if H is None:
        return None, 0
    inl = int(mask.sum())
    if inl < 60:
        return None, inl
    return H / H[2, 2], inl


def chain(sec0, sec1, hop):
    """sequential homography sec0->sec1; on failure the step does not move ref"""
    Hacc = np.eye(3)
    ref = frame(sec0)
    steps = ok_ = bad = 0
    s = sec0
    while True:
        s += hop
        if s >= sec1:
            cur = frame(sec1)
            H, n = pair_H(ref, cur)
            if H is not None:
                Hacc = H @ Hacc; ok_ += 1
            else:
                bad += 1
            steps += 1
            break
        cur = frame(s)
        H, n = pair_H(ref, cur)
        if H is not None:
            Hacc = H @ Hacc; ok_ += 1; ref = cur
        else:
            bad += 1
        steps += 1
    return Hacc, steps, ok_, bad


def xf(H, pts):
    p = np.asarray(pts, float).reshape(-1, 2)
    q = np.asarray(H, float) @ np.hstack([p, np.ones((len(p), 1))]).T  # (3,n)
    return (q[:2] / q[2:3]).T


def dlt(full, linec):
    """returns H: pixel -> meters (normalization inside)"""
    A = []
    for (x, y), (X, Y) in full:
        xn, yn, Xn, Yn = x / 1000.0, y / 1000.0, X / 60.0, Y / 60.0
        A.append([-xn, -yn, -1, 0, 0, 0, Xn * xn, Xn * yn, Xn])
        A.append([0, 0, 0, -xn, -yn, -1, Yn * xn, Yn * yn, Yn])
    for (x, y), c in linec:
        xn, yn, cn = x / 1000.0, y / 1000.0, c / 60.0
        A.append([xn, yn, 1, 0, 0, 0, -cn * xn, -cn * yn, -cn])
    A = np.array(A)
    _, S, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    M = np.array([[Hn[0, 0] * 60 / 1000, Hn[0, 1] * 60 / 1000, Hn[0, 2] * 60],
                  [Hn[1, 0] * 60 / 1000, Hn[1, 1] * 60 / 1000, Hn[1, 2] * 60],
                  [Hn[2, 0] / 1000,      Hn[2, 1] / 1000,      Hn[2, 2]]])
    return M / M[2, 2], S


def main():
    gc = frame(SEC_C)
    print("center->left (direct matching)...", flush=True)
    Hcl, ncl = pair_H(gc, frame(SEC_L))
    print("  inliers:", ncl, flush=True)
    if Hcl is None or ncl < 120:
        print("  -> via chain", flush=True)
        Hcl, st, ok_, bad = chain(SEC_C, SEC_L, HOP_L)
        print(f"  chain steps={st} ok={ok_} bad={bad}", flush=True)
    print("center->right (direct)...", flush=True)
    Hcr, ncr = pair_H(gc, frame(SEC_R))
    print("  inliers:", ncr, flush=True)
    if Hcr is None or ncr < 120:
        print("  -> via chain", flush=True)
        Hcr, st, ok_, bad = chain(SEC_C, SEC_R, HOP_R)
        print(f"  chain steps={st} ok={ok_} bad={bad}", flush=True)

    P = {
        "c1": ((850, 500), (30.0, 15.0), "c"),
        "c2": ((698, 374), (30.0, 30.0), "c"),
        "c3": ((1489, 1079), (30.0, 0.0), "c"),
        "l1": ((405, 415), (22.4, 30.0), "l"),
        "l2": ((0, 634), None, "l"),
        "r1": ((1015, 590), (37.6, 30.0), "r"),
        "r2": ((1650, 750), None, "r"),
    }
    toC = {"c": np.eye(3), "l": np.linalg.inv(Hcl), "r": np.linalg.inv(Hcr)}
    allp = {k: (tuple(xf(toC[src], [px])[0]), w) for k, (px, w, src) in P.items()}
    print("anchors in the central-frame coordinate system:")
    for k, (px, w) in allp.items():
        print(f"   {k}: ({px[0]:.0f},{px[1]:.0f}) world={w}")

    full = [(px, w) for px, w in allp.values() if w is not None]
    linec = [(allp["l2"][0], 22.4), (allp["r2"][0], 37.6)]
    H, S = dlt(full, linec)
    np.save(f"{OUT}/H_center.npy", H)
    json.dump({"H_center": H.tolist(), "Hcl": Hcl.tolist(), "Hcr": Hcr.tolist()},
              open(f"{OUT}/H.json", "w"))
    print("SVD spectrum:", np.round(S[:8] / S[0], 5))

    print("residuals:")
    errs = []
    for k, (px, w) in allp.items():
        p = xf(H, [px])[0]
        if w is not None:
            e = float(np.hypot(*(p - np.array(w))))
            errs.append(e)
            print(f"   {k}: world=({p[0]:.2f},{p[1]:.2f}) expected {w} -> {e*100:.0f} cm")
        else:
            c = 22.4 if k == "l2" else 37.6
            print(f"   {k} (on X={c}): X={p[0]:.2f} Y={p[1]:.2f}")
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
        for a, b in grid:
            pa, pb = xf(HW, [a, b])
            cv2.line(fr, tuple(pa.astype(int)), tuple(pb.astype(int)), (0, 255, 0), 2)
        for pt in [(22.4, 0), (22.4, 30), (37.6, 0), (37.6, 30)]:
            p = xf(HW, [pt])[0]
            cv2.circle(fr, tuple(p.astype(int)), 16, (0, 0, 255), 3)
        cv2.imwrite(f"{OUT}/verify_{tag}.jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cap.release()
    print("verify_*.jpg saved")


if __name__ == "__main__":
    main()
