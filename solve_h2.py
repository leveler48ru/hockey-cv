#!/usr/bin/env python3
"""v2: pan-homography chains with small steps + far-board collinearity check
+ robust DLT (IRLS). See solve_h.py (v1) for the anchor explanation."""
import cv2, numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solve_h import F, OUT, FPS, SEC_C, SEC_L, SEC_R, frame, pair_H, xf, dlt


def chain2(sec0, sec1, hop):
    """chain sec0->sec1; a skipped step does not move ref; returns Hacc"""
    Hacc = np.eye(3)
    ref = frame(sec0)
    ns = int(round((sec1 - sec0) / hop))
    for i in range(1, ns + 1):
        cur = frame(sec0 + i * hop if i < ns else sec1)
        H, n = pair_H(ref, cur)
        if H is not None:
            Hacc = H @ Hacc
            ref = cur
        else:
            print(f"  step {i}/{ns}: FAILED", flush=True)
    return Hacc


def irls_dlt(full, linec, iters=12):
    eqs = []
    for (x, y), (X, Y) in full:
        xn, yn, Xn, Yn = x / 1000.0, y / 1000.0, X / 60.0, Y / 60.0
        eqs.append(([-xn, -yn, -1, 0, 0, 0, Xn * xn, Xn * yn, Xn], 2))
        eqs.append(([0, 0, 0, -xn, -yn, -1, Yn * xn, Yn * yn, Yn], 2))
    for (x, y), c in linec:
        xn, yn, cn = x / 1000.0, y / 1000.0, c / 60.0
        eqs.append(([xn, yn, 1, 0, 0, 0, -cn * xn, -cn * yn, -cn], 1))
    A = np.array([e for e, _ in eqs], float)
    grp = [g for _, g in eqs]
    w = np.ones(len(A))
    Hn = None
    for it in range(iters):
        Aw = A * w[:, None]
        _, S, Vt = np.linalg.svd(Aw)
        Hn = Vt[-1]
        # meters
        Hm = np.array([[Hn[0] * 60 / 1000, Hn[1] * 60 / 1000, Hn[2] * 60],
                       [Hn[3] * 60 / 1000, Hn[4] * 60 / 1000, Hn[5] * 60],
                       [Hn[6] / 1000, Hn[7] / 1000, Hn[8]]])
        Hm = Hm / Hm[2, 2]
        # residuals: point-equations -> distance in meters; line -> |X-c|
        r = np.zeros(len(A))
        k = 0
        for (x, y), (X, Y) in full:
            p = xf(Hm, [(x, y)])[0]
            for j in range(2):
                comp = (p[0] - X) if j == 0 else (p[1] - Y)
                r[k] = abs(comp) / 30.0  # normalize by rink size
                k += 1
        for (x, y), c in linec:
            p = xf(Hm, [(x, y)])[0]
            r[k] = abs(p[0] - c) / 30.0
            k += 1
        sc = np.median(r[r > 0]) + 1e-9
        w = 1.0 / np.sqrt(1 + (r / (1.5 * sc)) ** 2)
    M = np.array([[Hn[0] * 60 / 1000, Hn[1] * 60 / 1000, Hn[2] * 60],
                  [Hn[3] * 60 / 1000, Hn[4] * 60 / 1000, Hn[5] * 60],
                  [Hn[6] / 1000, Hn[7] / 1000, Hn[8]]])
    return M / M[2, 2], w, r


def main():
    print("== chains with small steps", flush=True)
    Hcl = chain2(SEC_C, SEC_L, 20)
    Hcr = chain2(SEC_C, SEC_R, 20)

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
    print("anchors in central coordinates:")
    for k, (px, w) in allp.items():
        print(f"   {k}: ({px[0]:.0f},{px[1]:.0f}) world={w}")

    # control: c2, l1', r1' on the far board (y=30) -> collinear
    a, b, c_ = np.array(allp["c2"][0]), np.array(allp["l1"][0]), np.array(allp["r1"][0])
    d = np.abs(np.cross(b - a, c_ - a)) / np.linalg.norm(b - a)
    print(f"COLLINEARITY CONTROL of the far board: r1' deviation from the c2-l1' line: {d:.1f} px")

    full = [(px, w) for px, w in allp.values() if w is not None]
    linec = [(allp["l2"][0], 22.4), (allp["r2"][0], 37.6)]
    H, w, r = irls_dlt(full, linec)
    np.save(f"{OUT}/H_center.npy", H)
    json.dump({"H_center": H.tolist(), "Hcl": Hcl.tolist(), "Hcr": Hcr.tolist()},
              open(f"{OUT}/H.json", "w"))
    print("\nresiduals (robust solution):")
    errs = []
    for k, (px, wgt) in allp.items():
        p = xf(H, [px])[0]
        if wgt is not None:
            e = float(np.hypot(*(p - np.array(wgt))))
            errs.append(e)
            print(f"   {k}: world=({p[0]:.2f},{p[1]:.2f}) expected {wgt} -> {e*100:.0f} cm")
        else:
            cc = 22.4 if k == "l2" else 37.6
            print(f"   {k} (X={cc}): X={p[0]:.2f} Y={p[1]:.2f}")
    print(f"MEAN {np.mean(errs)*100:.0f} cm | MAX {max(errs)*100:.0f} cm")
    print("equation weights:", np.round(w, 2))

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
        for pt in [(22.4, 0), (22.4, 30), (37.6, 0), (37.6, 30), (4, 15), (56, 15)]:
            p = xf(HW, [pt])[0]
            cv2.circle(fr, tuple(p.astype(int)), 16, (0, 0, 255), 3)
        cv2.imwrite(f"{OUT}/verify_{tag}.jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cap.release()
    print("verify_*.jpg saved")


if __name__ == "__main__":
    main()
