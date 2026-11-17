#!/usr/bin/env python3
"""H from predicted v4 lines (DLT + RANSAC), no ORB and no manual taps.
Constraints: red x=0, blues x=±7.5, center circle r=4.5 (cc w0).
Validation against 10 manual anchors (fits.json). QC overlay of the best frames."""
import json, glob, os
import numpy as np, cv2
from pathlib import Path
from PIL import Image

BASE = Path('/root/fs')
OUT = BASE / 'out'
HD = OUT / 'H'; HD.mkdir(exist_ok=True)
RINK = {
    "center": (0, 0), "red_near": (0, -15), "red_far": (0, 15),
    "blue_L_near": (-7.5, -15), "blue_L_far": (-7.5, 15),
    "blue_R_near": (7.5, -15), "blue_R_far": (7.5, 15),
    "faceoff_C_left": (-1.524, 0), "faceoff_C_right": (1.524, 0),
}

def fit_DLT(obs):
    rows = []
    for u, v, X in obs:
        rows.append([u, v, 1.0, 0.0, 0.0, 0.0, -X * u])
        rows.append([0.0, 0.0, 0.0, u, v, 1.0, -X * v])
    A = np.array(rows, float)
    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    h = Vt[-1]
    if abs(h[6]) < 1e-9:
        return None
    return h / h[6]

def d2l(H, u, v, X):
    p = np.array([u, v, 1.0]) @ H.T
    wx, wy = p[0] / p[2], p[1] / p[2]
    return abs(wx - X), wy

def nearest_on_poly(poly, wx, wy):
    pts = np.array(poly, float)
    if len(pts) < 2:
        return None
    d = (pts - [wx, wy]) ** 2
    return np.sqrt(d[:, 0].min() + d[:, 1].min())

def collect_lines(recs, H, world):
    obs, seen = [], []
    for rec in recs:
        c = rec['cls']
        if c not in (0, 1):
            continue
        poly = rec['poly']
        if len(poly) < 6:
            continue
        w = pix2world(H, poly)
        for X, nm in world:
            d = nearest_on_poly(w, 0.0 if X == 0 else X, 0.0)
            # distance from the line polygon to the world vertical x=X, via x-coordinates
            dx = np.abs(w[:, 0] - X)
            if float(np.median(dx)) > 3.0:
                continue
            # sampling along world y: correspondences (u,v)->X
            if nm in seen:
                continue
            ys = np.linspace(-13, 13, 24)
            # sort world points by y and interpolate the pixel ones
            order = np.argsort(w[:, 1])
            ws, ps = w[order], np.array(poly, float)[order]
            for wy in ys:
                k = np.searchsorted(ws[:, 1], wy)
                if k <= 0 or k >= len(ws):
                    continue
                t = (wy - ws[k-1, 1]) / max(1e-6, ws[k, 1] - ws[k-1, 1])
                u = ps[k-1, 0] + t * (ps[k, 0] - ps[k-1, 0])
                v = ps[k-1, 1] + t * (ps[k, 1] - ps[k-1, 1])
                obs.append((u, v, X))
            seen.append(nm)
    return obs

def pix2world(H, poly):
    q = np.array(poly, float)
    q = np.hstack([q, np.ones((len(q), 1))]) @ H.T
    return q[:, :2] / q[:, 2:3]

def calib_frame(recs):
    lines = [r for r in recs if r['cls'] in (0, 1) and len(r['poly']) >= 6 and r['conf'] > 0.25]
    cc = [r for r in recs if r['cls'] == 2 and r['conf'] > 0.25 and len(r['poly']) >= 12]
    if len(lines) < 2 and not cc:
        return None
    world_opts = [('red', 0.0), ('blueL', -7.5), ('blueR', 7.5)]
    obs_all = []
    for r in lines:
        w = pix2world(np.array([[1,0,640],[0,1,360],[0,0,1]], float), r['poly'])  # identity: world=pixels won't cut it — using only the x-range for the candidate
        # starting heuristic: a line in pixels left of the frame center -> blueL/candidate
        xs = np.array(r['poly'])[:, 0]
        cand_X = min(world_opts, key=lambda o: abs(np.median(xs) - (640 if o[1]==0 else (480 if o[1]==-7.5 else 800))))
        obs_all.append((r, cand_X[1]))
    best = None
    rng = np.random.default_rng(1)
    # minimal sets: 3 lines (or 2 + circle)
    def build(trio):
        obs = []
        for r, X in trio:
            Hs = obs  # placeholder
        return obs
    # simple approach: fixed start — if exactly 3 lines, assign red/±blue by x order,
    # then RANSAC over permutations (6 in total)
    import itertools
    if len(lines) >= 3:
        perms = list(itertools.permutations(world_opts))[:24]
    elif len(lines) == 2:
        perms = list(itertools.permutations(world_opts, 2))
    else:
        perms = [[]]
    cands = []
    for perm in perms:
        assign = list(zip(lines, [p[1] for p in perm])) if perm else []
        obs = collect_lines_assigned(assign)
        if len(obs) < 9:
            continue
        H = fit_DLT(obs)
        if H is None:
            continue
        H = H.reshape(3, 3)
        r = np.array([d2l(H, u, v, X) for u, v, X in obs])
        rms = float(np.sqrt((r[:, 0] ** 2).mean()))
        cands.append((rms, H, obs, perm))
    if not cands:
        return None
    cands.sort(key=lambda t: t[0])
    rms, H, obs, perm = cands[0]
    # refit on inliers (|dx|<0.6 m) + add the circle
    for _ in range(3):
        keep = [(u, v, X) for u, v, X in obs if d2l(H, u, v, X)[0] < 0.6]
        if len(keep) < 9:
            break
        H2 = fit_DLT(keep)
        if H2 is None:
            break
        H = H2.reshape(3, 3)
        obs = keep
    inl = sum(1 for u, v, X in obs if d2l(H, u, v, X)[0] < 0.6)
    # circle check: center and r
    circ_err = None
    if cc:
        w = pix2world(H, cc[0]['poly'])
        cx, cy = w[:, 0].mean(), w[:, 1].mean()
        rr = np.hypot(w[:, 0] - cx, w[:, 1] - cy)
        circ_err = (round(float(np.hypot(cx, cy)), 2), round(float(rr.mean()), 2))
    if rms > 1.5 or inl < 12:
        return None
    return {'H': H, 'rms_m': round(rms, 3), 'inl': inl, 'n_lines': len(lines),
            'perm': perm, 'circ': circ_err}

def collect_lines_assigned(assign):
    obs = []
    for r, X in assign:
        poly = np.array(r['poly'], float)
        w = pix2world(np.eye(3) if False else np.array([[1,0,0],[0,1,0],[0,0,1]], float), poly)
        # correspondences: take a polygon point, its projection onto the world vertical x=X
        ys = np.linspace(-13, 13, 24)
        # interpolation by world y; at startup there is no H — use v as a proxy (monotonic)
        order = np.argsort(poly[:, 1])
        P = poly[order]
        # world y from normalized v-range; the far/near sign doesn't matter for an x=const line
        for wy in ys:
            t = (wy + 13) / 26.0
            k = min(len(P) - 1, max(1, int(round(t * (len(P) - 1)))))
            u, v = P[k]
            obs.append((u, v, X))
    return obs

def main():
    masks = json.loads((OUT / 'masks.json').read_text())
    fits = json.loads((OUT / 'fits.json').read_text())  # manual anchors for validation
    summary = {'total': len(masks), 'ok': 0, 'anchor_agree': None}
    per = {}
    anchor_cmp = []
    for stem, recs in sorted(masks.items()):
        res = calib_frame(recs)
        v = stem.rsplit('_', 1)[0]
        if res is None:
            continue
        H = res['H']
        np.save(HD / f'{stem}.npy', H)
        summary['ok'] += 1
        b = per.setdefault(v, [0, 0]); b[0] += 1
        # agreement with the manual anchor
        if stem in fits:
            Hm = np.array(fits[stem]['H'])
            errs = []
            for k, xy in RINK.items():
                q = np.array([xy[0], xy[1], 1.0]) @ np.linalg.inv(H).T
                q = q[:2] / q[2]
                pm = (np.array([*fits.get('taps', {}).get(k, (0, 0))]) if False else None)
            anchor_cmp.append(stem)
    # separate anchor validation: compare H_line against the manual one on 7 structural world points
    agree = []
    for stem in anchor_cmp:
        Hl = np.load(HD / f'{stem}.npy')
        Hm = np.array(fits[stem]['H'])
        d = []
        for k, wpt in RINK.items():
            # world->pix through each H, compare pixel positions
            pl = (np.array([*wpt, 1.0]) @ np.linalg.inv(Hl).T); pl = pl[:2] / pl[2]
            pm = (np.array([*wpt, 1.0]) @ np.linalg.inv(Hm).T); pm = pm[:2] / pm[2]
            if max(abs(pl)) > 4000 or max(abs(pm)) > 4000:
                continue
            d.append(float(np.linalg.norm(pl - pm)))
        if d:
            agree.append(round(float(np.median(d)), 1))
    if agree:
        summary['anchor_agree'] = {'n': len(agree), 'median_pix': float(np.median(agree)),
                                   'p90_pix': float(np.percentile(agree, 90)), 'per_frame': agree}
    summary['per_video'] = {k: str(v[0]) for k, v in per.items()}
    (OUT / 'calib_lines.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False))

main()
