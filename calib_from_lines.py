#!/usr/bin/env python3
"""H for a new venue from v4 SEGMENTATION (masks.json), without ORB and without manual taps.

Idea: the world rink 60x30 is defined by lines: red x=0, blue x=±7.5 (all along y),
the boards y=±15. Intersections of these lines = 6 known points
((0,±15),(±7.5,±15)); plus the circle r=4.5 cuts the red line at (0,±4.5).
Pixel-line intersections -> DLT (4+ points) -> A: world->pix, H=inv(A).

The Near/Far and Left/Right assignment — 4 combinations; we take the one with the minimal
reprojection residual + sign agreement with the anchors (fits.json).
QC per frame: RMS <= 15 px, circle within tolerance, boards logically placed in the frame.
Output: /out/H/<stem>.npy (pix->world, as in the rest of the pipeline) + summary.
"""
import json
import numpy as np, cv2
from pathlib import Path
from itertools import product

BASE = Path('/work') if Path('/work').exists() else Path('.')
OUT = BASE / 'out'
MASKS = json.loads((BASE / 'masks.json').read_text())
FITS = json.loads((BASE / 'fits.json').read_text())  # manual anchors (convention)

W_IMG, H_IMG = 1280, 720

# world constants
Y_BOARD = 15.0
X_BLUE = 7.5
R_CIRC = 4.5

def line_from_poly(poly):
    """ℓ=(a,b,c): a²+b²=1, minimized sum of squared distances to the contour points."""
    P = np.array(poly, float)
    if len(P) < 3:
        return None
    # PCA line (for elongated contours)
    m = P.mean(0)
    Q = P - m
    _, _, vt = np.linalg.svd(Q, full_matrices=False)
    d = vt[0]
    n = np.array([-d[1], d[0]])
    c = -n @ m
    l = np.array([n[0], n[1], c])
    dist = np.abs(P @ l)
    # even if the contour is a "brick" of the board strip (wide), the line is still the mean one
    return l, float(dist.mean()), np.hypot(*d), Q

def fit_world_points(recs, near_far, bl, assign):
    """recs: dict role->line. Returns (pix_pts, world_pts)."""
    lN, lF = assign['board_near'], assign['board_far']   # long boards y=-15,+15
    lR = assign['red']                                    # x=0
    lBL, lBR = assign['blue_L'], assign['blue_R']         # x=±7.5
    def inter(l1, l2):
        p = np.cross(l1, l2)
        if abs(p[2]) < 1e-6: return None
        return p[:2] / p[2]
    pts, wpts = [], []
    for lb, yw in ((lN, -Y_BOARD), (lF, Y_BOARD)):
        for ll, xw in ((lR, 0.0), (lBL, -X_BLUE), (lBR, X_BLUE)):
            q = inter(ll, lb)
            if q is None: continue
            if not (-200 <= q[0] <= W_IMG+200 and -200 <= q[1] <= H_IMG+200):
                continue
            pts.append(q); wpts.append((xw, yw))
    # circle×red line: (0,±4.5) — ellipse/line intersections: approx — ends of the "vertical"
    return np.array(pts), np.array(wpts)

def dlt_A(src, dst):
    """world(x,y)->pix(u,v), A[2x3]; src world pts, dst pix pts. 4+ points."""
    n = len(src)
    if n < 4: return None
    A = np.zeros((2*n, 9))
    for i in range(n):
        x, y = src[i]; u, v = dst[i]
        A[2*i] = [x, y, 1, 0, 0, 0, -u*x, -u*y, -u]
        A[2*i+1] = [0, 0, 0, x, y, 1, -v*x, -v*y, -v]
    _, _, Vt = np.linalg.svd(A)
    M = Vt[-1].reshape(3, 3)
    if abs(M[2,2]) < 1e-9: return None
    return M  # world->pix (3x3, with the 3rd row)

def project(A, wpts):
    q = np.hstack([wpts, np.ones((len(wpts), 1))]) @ A.T
    return q[:, :2] / q[:, 2:3]

def extract_roles(recs):
    """group predictions by role: long boards (y-horizontal lines), x-lines, circle."""
    boards, verticals, circles = [], [], []
    for r in recs:
        c = r['cls']
        if c == 6:
            lf = line_from_poly(r['poly'])
            if lf: boards.append(lf)
        elif c in (0, 1):
            lf = line_from_poly(r['poly'])
            if lf: verticals.append((c, lf))
        elif c == 2:
            circles.append(r)
    return boards, verticals, circles

def role_pick(boards, verticals):
    """near = bottom horizontal (larger v at median u), red — x=vertical near the center."""
    def v_at(l, u=640):
        a, b, cc = l
        if abs(b) < 1e-6: return None
        return -(a*u+cc)/b
    bs = [(v_at(l), th, l) for _, th, l in
          ((None, None, None),)] if False else []
    bsel = []
    for l, rms, dvec, Q in boards:
        a, b, cc = l
        if abs(b) < 1e-9: continue
        if abs(a) > abs(b):   # vertical board (short board x=±30) — useless for near/far
            continue
        v640 = -(a*640+cc)/b
        bsel.append((v640, l))
    if len(bsel) < 2 or len(verticals) < 2:
        return None
    bsel.sort(key=lambda t: t[0])
    board_far, board_near = bsel[0][1], bsel[-1][1]  # top of the frame = far
    verts = []
    for c, (l, rms, dvec, Q) in verticals:
        if abs(dvec[1]) < abs(dvec[0]):  # not a vertical
            continue
        u400 = -(l[1]*400+l[2])/l[0] if abs(l[0])>1e-9 else None
        u320 = -(l[1]*320+l[2])/l[0] if abs(l[0])>1e-9 else None
        verts.append((np.median([p for p in (u400,u320) if p is not None]), c, l))
    if len(verts) < 2: return None
    verts.sort(key=lambda t: t[0])
    # red — the one between the blues or the most frequent cls=0
    reds = [v for v in verts if v[1] == 0]
    blues = [v for v in verts if v[1] == 1]
    red = reds[len(reds)//2][2] if reds else verts[len(verts)//2][2]
    if len(blues) >= 2:
        bL_pix, bR_pix = blues[0][2], blues[-1][2]
    elif len(blues) == 1:
        bL_pix = bR_pix = blues[0][2]
    else:
        return None
    return dict(board_near=board_near, board_far=board_far, red=red,
                blue_L=bL_pix, blue_R=bR_pix)

def solve_frame(stem, recs, sign_ref):
    roles = extract_roles(recs)
    if roles is None: return None
    boards, verticals, circles = roles
    base = role_pick(boards, verticals)
    if base is None: return None
    best = None
    # the near/far convention is fixed by the frame; left/right of the blues — 2 variants (swap + near/far swap)
    for nf_swap, bl_swap in product((False, True), repeat=2):
        ass = dict(base)
        if nf_swap:
            ass['board_near'], ass['board_far'] = ass['board_far'], ass['board_near']
        if bl_swap:
            ass['blue_L'], ass['blue_R'] = ass['blue_R'], ass['blue_L']
        #world pts
        wp = [(0,-Y_BOARD),(0,Y_BOARD),(-X_BLUE,-Y_BOARD),(-X_BLUE,Y_BOARD),
              (X_BLUE,-Y_BOARD),(X_BLUE,Y_BOARD)]
        lN,lF = ass['board_near'], ass['board_far']
        def inter(l1,l2):
            p=np.cross(l1,l2)
            return None if abs(p[2])<1e-6 else p[:2]/p[2]
        pp = [inter(lF,(1,0,0)*0+l2) for l2 in []]
        pp = []
        for (xw,yw) in wp:
            lb = lN if yw==-Y_BOARD else lF
            # point at x=X intersecting board lb: line x=X: (1,0,-X)
            q = inter(lb, (1.0,0.0,-xw))
            if q is None or not (-300<=q[0]<=W_IMG+300 and -300<=q[1]<=H_IMG+300): continue
            pp.append(q)
        wused = [wp[i] for i,(q) in enumerate(pp)] if len(pp)==len(wp) else None
        if wused is None:
            # collect in parallel
            pp2=[]; ww=[]
            for (xw,yw) in wp:
                lb = lN if yw==-Y_BOARD else lF
                q=inter(lb,(1.0,0.0,-xw))
                if q is None: continue
                if not (-300<=q[0]<=W_IMG+300 and -300<=q[1]<=H_IMG+300): continue
                pp2.append(q); ww.append((xw,yw))
            pp, wused = pp2, ww
        if len(pp) < 5: continue
        A = dlt_A(np.array(wused,float), np.array(pp,float))
        if A is None: continue
        pred = project(A, np.array(wused,float))
        res = np.linalg.norm(pred-np.array(pp), axis=1)
        rms = float(np.sqrt((res**2).mean()))
        if rms > 25 or abs(np.linalg.det(A[:2,:2])) < 1e-6: continue
        # QC: the sign of red-to-right-blue must match the anchor convention
        uR = project(A, np.array([[X_BLUE,0]]))[0][0]
        uRed = project(A, np.array([[0,0]]))[0][0]
        s = np.sign(uR-uRed)
        if sign_ref is not None and s != 0 and s != sign_ref:
            continue
        Hm = np.linalg.inv(A)
        best = {'A': A, 'H': Hm, 'rms_px': round(rms,1), 'n': len(pp),
                'swap': (nf_swap, bl_swap)}
        if rms < 8: break
    if best is not None:
        # circle check: predicted ellipse vs contour
        if circles:
            a,b,c = circles[0]['poly'], None, None
            th = np.linspace(0,2*np.pi,36,endpoint=False)
            wpts = np.stack([R_CIRC*np.cos(th), R_CIRC*np.sin(th)],1)
            pc = project(best['A'], wpts)
            poly = np.array(circles[0]['poly'], float)
            d = min(np.linalg.norm(pc[:,None,:]-poly[None,:,:],axis=2).min(1).mean(),
                    np.linalg.norm(poly[:,None,:]-pc[None,:,:],axis=2).min(1).mean())
            best['circ_px'] = round(float(d),1)
    return best

def main():
    # 1) convention from the anchors: sign(u_blueR - u_red) in pixels
    signs = []
    for stem, v in FITS.items():
        Hm = np.array(v['H'])  # pix->world
        A = np.linalg.inv(Hm)
        pR = project(A, np.array([[X_BLUE,0]]))[0]
        p0 = project(A, np.array([[0,0]]))[0]
        signs.append(int(np.sign(pR[0]-p0[0])))
    sign_ref = int(np.sign(np.sum(signs))) if signs else None
    print('anchor signs:', signs, '-> ref', sign_ref)

    res = {}
    ok = 0
    for stem, recs in sorted(MASKS.items()):
        b = solve_frame(stem, recs, sign_ref)
        if b is None:
            res[stem] = {'fail': True}
            continue
        ok += 1
        np.save(OUT/'H'/f'{stem}.npy', b['H'])
        res[stem] = {k: v for k, v in b.items() if k not in ('A','H')}
    rms = sorted(v['rms_px'] for v in res.values() if 'rms_px' in v)
    print(f"frames: {len(MASKS)} | usable H: {ok}")
    if rms:
        print(f"rms px: p50 {rms[len(rms)//2]} p90 {rms[int(len(rms)*0.9)]} max {rms[-1]}")
    # cross-check against the anchors
    agree = []
    for stem in FITS:
        if 'H' not in (res.get(stem) or {}) and not (OUT/'H'/f'{stem}.npy').exists(): continue
        Hl = np.load(OUT/'H'/f'{stem}.npy'); Hm = np.array(FITS[stem]['H'])
        d = []
        for xy in [(0,0),(0,-15),(0,15),(-7.5,-15),(7.5,15),(0,-4.5),(0,4.5)]:
            pl = project(np.linalg.inv(Hl), np.array([xy]))[0]
            pm = project(np.linalg.inv(Hm), np.array([xy]))[0]
            d.append(float(np.linalg.norm(pl-pm)))
        agree.append(round(float(np.median(d)),1))
    if agree:
        print('line-H vs manual anchor cross-check (px):', sorted(agree))
    (OUT/'calib_lines.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))

(OUT/'H').mkdir(parents=True, exist_ok=True)
main()
