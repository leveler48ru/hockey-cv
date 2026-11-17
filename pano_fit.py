#!/usr/bin/env python3
"""Global 'pan-only' calibration + full-ice panorama (v4).

Model: the camera rotates around the vertical axis (K is constant).
A frame with scene shift u is related to the reference (center frame) by a
pure-rotation homography; in normalized coordinates:
    nx' = (nx + t)/(1 - t*nx),  my' = ny/(1 - t*nx),  t = sgn*u/fx
World->reference: full homography H8 (8 dof) + fx => 9 unknowns for 12 equations
(5 full anchors x2 + 2 'point-on-line' x1). The sign of t is enumerated.

u is measured by an ORB chain in small steps (5 s to the left, 15 s to the right) -
bigger steps gave drift earlier; here every step is validated by the median.

Output: match/H_pan.json, match/panorama.jpg, match/panorama_marked.jpg
"""
import cv2, numpy as np, json, os
from scipy.optimize import least_squares

BASE = os.path.dirname(os.path.abspath(__file__))
M = BASE + "/match"
F = "./media/9281020430926.mp4"
FPS = 30
SEC_C, SEC_L, SEC_R = 3, 240, 2570
CX0, CY0 = 960.0, 540.0
PX_M = 35.0
MARG = 2.0
WC = int((60 + 2 * MARG) * PX_M)
HC = int((30 + 2 * MARG) * PX_M)

P = {
    "c1": ((850.0, 498.0), (30.0, 15.0), "c"),
    "c2": ((698.0, 374.0), (30.0, 30.0), "c"),
    "c3": ((1489.0, 1079.0), (30.0, 0.0), "c"),
    "l1": ((405.0, 415.0), (22.4, 30.0), "l"),
    "l2": ((0.0, 634.0), None, "l"),
    "r1": ((1015.0, 590.0), (37.6, 30.0), "r"),
    "r2": ((1650.0, 750.0), None, "r"),
}
LINEC = {"l2": 22.4, "r2": 37.6}


def median_frame(sec, path, half=1.0, n=13):
    if os.path.exists(path):
        im = cv2.imread(path)
        if im is not None and im.shape[1] == 1920:
            return im
    cap = cv2.VideoCapture(F)
    frs = []
    for s in np.linspace(sec - half, sec + half, n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s * FPS))
        ok, fr = cap.read()
        if ok:
            frs.append(fr.astype(np.uint16))
    cap.release()
    med = np.median(np.stack(frs), axis=0).astype(np.uint8)
    cv2.imwrite(path, med, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return med


orb = cv2.ORB_create(6000, 1.2, 8)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def step_shift(g0, g1):
    """a point in g0 at coordinate x0 lies at x0+du in g1 (scene shifted by du)"""
    k0, d0 = orb.detectAndCompute(g0, None)
    k1, d1 = orb.detectAndCompute(g1, None)
    if d0 is None or d1 is None or len(d0) < 200 or len(d1) < 200:
        return None
    mm = bf.knnMatch(d0, d1, k=2)
    good = [a for a, b in mm if a.distance < 0.8 * b.distance]
    if len(good) < 60:
        return None
    p0 = np.float32([k0[x.queryIdx].pt for x in good])
    p1 = np.float32([k1[x.trainIdx].pt for x in good])
    dif = p1 - p0
    m = np.median(dif, axis=0)
    keep = (np.abs(dif[:, 0] - m[0]) < 20) & (np.abs(dif[:, 1] - m[1]) < 20)
    if keep.sum() < 25:
        return None
    mu = np.median(dif[keep], axis=0)
    return float(mu[0]), float(mu[1])


def chain(sec0, sec1, hop):
    cap = cv2.VideoCapture(F)
    u = v = 0.0
    ok = bad = 0
    ref = None
    s = sec0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s * FPS))
        got, fr = cap.read()
        if got:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            if ref is None:
                ref = g
            else:
                r = step_shift(ref, g)
                if r is not None:
                    u += r[0]; v += r[1]; ref = g; ok += 1
                else:
                    bad += 1
        else:
            bad += 1
        if s >= sec1:
            break
        s = min(sec1, s + hop)
    cap.release()
    print(f"chain {sec0}->{sec1}: du={u:+.0f} dv={v:+.0f} ok={ok} bad={bad}", flush=True)
    return u, v


def ref_to_frame(p_ref, t, fx):
    nx = (p_ref[0] - CX0) / fx
    ny = (p_ref[1] - CY0) / fx
    den = 1.0 - t * nx
    if abs(den) < 1e-6:
        return None
    return np.array([CX0 + fx * (nx + t) / den, CY0 + fx * ny / den])


def frame_to_ref(p_f, t, fx):
    """inverse: nx_ref = (nx_f - t)/(1 + t*nx_f)"""
    nx = (p_f[0] - CX0) / fx
    ny = (p_f[1] - CY0) / fx
    den = 1.0 + t * nx
    if abs(den) < 1e-6:
        return None
    nr = (nx - t) / den
    return np.array([CX0 + fx * nr, CY0 + fx * ny / den])


def dlt8(pts):
    """pts ((x,y)->(u,v)) -> H world->pixel (h33=1)"""
    A = []
    for (X, Y), (u, v) in pts:
        A.append([X, Y, 1, 0, 0, 0, -u * X, -u * Y, -u])
        A.append([0, 0, 0, X, Y, 1, -v * X, -v * Y, -v])
    _, _, Vt = np.linalg.svd(np.array(A, float))
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def model_pix(H8, fx, t, x, y):
    q = H8 @ np.array([x, y, 1.0])
    return ref_to_frame(np.array([q[0] / q[2], q[1] / q[2]]), t, fx)


def residuals(par, us, sgn):
    H8 = np.array(list(par[0:8]) + [1.0]).reshape(3, 3)
    fx = par[8]
    out = []
    for k, ((px, py), w, src) in P.items():
        t = sgn * us[src] / fx
        if w is not None:
            q = model_pix(H8, fx, t, w[0], w[1])
            if q is None:
                out += [1e4, 1e4]
                continue
            out += [q[0] - px, q[1] - py]
        else:
            cc = LINEC[k]
            p1 = model_pix(H8, fx, t, cc, 5.0)
            p2 = model_pix(H8, fx, t, cc, 25.0)
            if p1 is None or p2 is None:
                out += [1e4]
                continue
            vx, vy = p2[0] - p1[0], p2[1] - p1[1]
            L = np.hypot(vx, vy) + 1e-9
            out.append(abs((px - p1[0]) * vy - (py - p1[1]) * vx) / L)
    return np.array(out)


def main():
    meds = {t: median_frame(sec, f"{M}/med13_{t}.jpg")
            for t, sec in [("c", SEC_C), ("l", SEC_L), ("r", SEC_R)]}
    print("== frame shifts (ORB chains)", flush=True)
    ul, vl = chain(SEC_C, SEC_L, 5)
    ur, vr = chain(SEC_C, SEC_R, 15)
    us = {"c": 0.0, "l": ul, "r": ur}

    fx0 = 1500.0
    best = None
    for sgn in (1.0, -1.0):
        tl = sgn * ul / fx0
        tr = sgn * ur / fx0
        pts = []
        for k, ((px, py), w, src) in P.items():
            if w is None:
                continue
            t = {"c": 0.0, "l": tl, "r": tr}[src]
            pr = frame_to_ref(np.array([px, py]), t, fx0)
            if pr is None:
                continue
            pts.append((w, tuple(pr)))
        H8s = dlt8(pts)
        p0 = np.append(H8s.flatten()[:8], fx0)  # h11..h32, h33=1
        try:
            sol = least_squares(residuals, p0, args=(us, sgn), method="lm",
                                xtol=1e-14, ftol=1e-14, max_nfev=40000)
        except Exception as ex:
            print(f"sgn={sgn:+.0f}: failure {ex}", flush=True)
            continue
        rms = float(np.sqrt(np.mean(sol.fun ** 2)))
        print(f"sgn={sgn:+.0f}: RMS={rms:.1f} px (fx={sol.x[8]:.0f})", flush=True)
        if best is None or rms < best[0]:
            best = (rms, sol, sgn)
    if best is None:
        raise SystemExit("no solution")
    rms, sol, sgn = best
    H8 = np.array(list(sol.x[0:8]) + [1.0]).reshape(3, 3)
    fx = float(sol.x[8])
    print(f"RESULT: RMS={rms:.1f} px, fx={fx:.0f}, sgn={sgn:+.0f}, ul={ul:+.0f}, ur={ur:+.0f}")
    errs = []
    for k, ((px, py), w, src) in P.items():
        t = sgn * us[src] / fx
        if w is not None:
            q = model_pix(H8, fx, t, w[0], w[1])
            e = float(np.hypot(q[0] - px, q[1] - py))
            errs.append(e)
            print(f"   {k}: pred=({q[0]:.0f},{q[1]:.0f}) actual=({px:.0f},{py:.0f}) -> {e:.0f} px")
        else:
            cc = LINEC[k]
            p1 = model_pix(H8, fx, t, cc, 5.0); p2 = model_pix(H8, fx, t, cc, 25.0)
            vx, vy = p2[0] - p1[0], p2[1] - p1[1]
            dist = abs((px - p1[0]) * vy - (py - p1[1]) * vx) / np.hypot(vx, vy)
            print(f"   {k} (X={cc}): {dist:.0f} px from the line projection")
    print(f"MEAN {np.mean(errs):.1f} px | MAX {max(errs):.1f} px")
    json.dump({"H8": H8.tolist(), "fx": fx, "cx": CX0, "cy": CY0, "sgn": sgn,
               "u": us, "rms_px": rms, "PX_M": PX_M, "MARG": MARG},
              open(f"{M}/H_pan.json", "w"))

    # == panorama
    AW = np.array([[1 / PX_M, 0, -MARG], [0, -1 / PX_M, 30 + MARG], [0, 0, 1]])
    xs = (np.arange(WC, dtype=np.float32) + 0.5)
    ys = (np.arange(HC, dtype=np.float32) + 0.5)
    GX, GY = np.meshgrid(xs, ys)
    Wm = AW @ np.stack([GX, GY, np.ones_like(GX)])
    wx, wy = Wm[0], Wm[1]
    ones = np.ones_like(wx)
    q = H8 @ np.stack([wx, wy, ones])
    with np.errstate(divide="ignore", invalid="ignore"):
        Xr = (q[0] / q[2]).astype(np.float32)
        Yr = (q[1] / q[2]).astype(np.float32)
    canvas = np.zeros((HC, WC, 3), np.uint8)
    wsum = np.zeros((HC, WC), np.float32)
    for tag in ("l", "r", "c"):
        t = sgn * us[tag] / fx
        nx = (Xr - CX0) / fx
        ny = (Yr - CY0) / fx
        den = 1.0 - t * nx
        with np.errstate(divide="ignore", invalid="ignore"):
            Xf = CX0 + fx * (nx + t) / den
            Yf = CY0 + fx * ny / den
        Xf = np.nan_to_num(Xf.astype(np.float32), nan=1e6, posinf=1e6, neginf=-1e6)
        Yf = np.nan_to_num(Yf.astype(np.float32), nan=1e6, posinf=1e6, neginf=-1e6)
        warp = cv2.remap(meds[tag], Xf, Yf, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        m = (Xf > 2) & (Xf < 1918) & (Yf > 2) & (Yf < 1078) & (warp.sum(axis=2) > 8)
        mw = m.astype(np.float32) * (2.0 if tag == "c" else 1.0)
        acc = canvas * mw[:, :, None] + warp * mw[:, :, None]
        new = wsum + mw
        np.divide(acc, new[:, :, None], out=np.zeros_like(acc), where=new[:, :, None] > 0)
        canvas = np.clip(acc, 0, 255).astype(np.uint8)
        wsum = new
    cv2.imwrite(f"{M}/panorama.jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 93])

    # == markings
    vis = canvas.copy()

    def c2(x, y):
        return (int((x + MARG) * PX_M), int((30 + MARG - y) * PX_M))
    for x in range(0, 61, 2):
        cv2.line(vis, c2(x, 0), c2(x, 30), (0, 200, 0), 1 if x % 10 else 2)
    for y in range(0, 31, 2):
        cv2.line(vis, c2(0, y), c2(60, y), (0, 200, 0), 1 if y % 10 else 2)
    cv2.line(vis, c2(30, 0), c2(30, 30), (0, 0, 220), 5)
    for x in (22.4, 37.6):
        cv2.line(vis, c2(x, 0), c2(x, 30), (220, 0, 0), 5)
    for x in (4, 56):
        cv2.line(vis, c2(x, 0), c2(x, 30), (0, 0, 220), 3)
        cv2.line(vis, c2(x, 15 - 1.36), c2(x, 15 + 1.36), (0, 255, 0), 8)
    for gx, sg in ((4, 1), (56, -1)):
        for yb in (0, 30):
            py_pole = 15 - 1.36 if yb == 0 else 15 + 1.36
            cv2.line(vis, c2(gx, yb), c2(gx + sg * 3.4, yb), (0, 0, 220), 3)
            cv2.line(vis, c2(gx + sg * 3.4, yb), c2(gx, py_pole), (0, 0, 220), 2)

    def circle(cxw, cyw, rr, col, th=2):
        pts = [c2(cxw + rr * np.cos(t), cyw + rr * np.sin(t))
               for t in np.linspace(0, 2 * np.pi, 65)]
        cv2.polylines(vis, [np.array(pts)], True, col, th)
    circle(30, 15, 4.5, (0, 160, 255), 3)
    cv2.circle(vis, c2(30, 15), 8, (0, 160, 255), -1)
    for cxw, cyw in [(6, 7.1), (6, 22.9), (54, 7.1), (54, 22.9)]:
        circle(cxw, cyw, 4.5, (0, 0, 220), 2)
        cv2.circle(vis, c2(cxw, cyw), 6, (0, 0, 220), -1)
        for dxo, dyo in [(-0.75, 0), (0.75, 0), (0, -0.6), (0, 0.6)]:
            cv2.circle(vis, c2(cxw + dxo, cyw + dyo), 4, (0, 0, 220), -1)
    for cxw, cyw in [(22.4, 7.1), (22.4, 22.9), (37.6, 7.1), (37.6, 22.9)]:
        circle(cxw, cyw, 2.0, (0, 0, 220), 2)
        cv2.circle(vis, c2(cxw, cyw), 5, (0, 0, 220), -1)
    # anchors back into world (by the actual projection)
    for k, ((px, py), w, src) in P.items():
        t = sgn * us[src] / fx
        pr = frame_to_ref(np.array([px, py]), t, fx)
        if pr is None:
            continue
        qi = np.linalg.inv(H8) @ np.array([pr[0], pr[1], 1.0])
        Xw, Yw = qi[0] / qi[2], qi[1] / qi[2]
        if -MARG < Xw < 60 + MARG and -MARG < Yw < 30 + MARG:
            pt = c2(Xw, Yw)
            cv2.circle(vis, pt, 17, (255, 255, 0), 3)
            cv2.putText(vis, k, (pt[0] + 20, pt[1] - 10), 0, 1.2, (255, 255, 0), 3)
    cv2.imwrite(f"{M}/panorama_marked.jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("panorama ready:", canvas.shape)


if __name__ == "__main__":
    main()
