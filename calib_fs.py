#!/usr/bin/env python3
"""Calibration of the figure-skating venue (fs_*): H from platform points + estimate of
extrapolation error without goal_L/R (on those 2 frames where Alexander actually placed them).
Output: fs_calib/qc.json, fs_calib/H/<frame>.npy, summary."""
import json, math
import numpy as np, cv2
from pathlib import Path

BASE = Path(".")
CALIB = BASE / "annotate" / "data" / "calib"
OUT = BASE / "fs_calib"; (OUT / "H").mkdir(parents=True, exist_ok=True)

# meters, rink center = (0,0); x — long axis (±30), y — short axis (±15)
RINK = {
    "center": (0, 0), "red_near": (0, -15), "red_far": (0, 15),
    "blue_L_near": (-7.5, -15), "blue_L_far": (-7.5, 15),
    "blue_R_near": (7.5, -15), "blue_R_far": (7.5, 15),
    "faceoff_C_left": (-1.524, 0), "faceoff_C_right": (1.524, 0),
    "faceoff_L_near": (-19.904, -8.294), "faceoff_L_far": (-19.904, 8.294),
    "faceoff_R_near": (19.904, -8.294), "faceoff_R_far": (19.904, 8.294),
    "goal_L": (-26, 0), "goal_R": (26, 0),
}
MIN_PTS = 6
GOALS = ("goal_L", "goal_R")

def fit(pairs):
    src = np.float32([p[0] for p in pairs]); dst = np.float32([p[1] for p in pairs])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None: return None, None, None
    pred = (H @ np.hstack([src, np.ones((len(src),1))]).T).T
    pred = pred[:, :2] / pred[:, 2:3]
    rms = float(np.sqrt(((pred - dst)**2).sum(1).mean()))
    return H, rms, mask.ravel().astype(bool)

qc = {}
with_goal_frames = []
for jf in sorted(CALIB.glob("fs*.json")):
    d = json.loads(jf.read_text())
    pts = [(RINK[k], (v["x"], v["y"])) for k, v in d.items() if k in RINK]
    rec = {"n_pts": len(pts)}
    if len(pts) >= MIN_PTS:
        H, rms, inl = fit(pts)
        if H is not None:
            rec.update({"rms_px": round(rms, 1), "inliers": int(inl.sum()), "n_used": len(pts)})
            np.save(OUT / "H" / f"{jf.stem}.npy", H)
        else:
            rec["error"] = "no H"
    else:
        rec["skip"] = f"<{MIN_PTS} points"
    # frame with goal anchors — reference for estimating extrapolation
    if any(g in d for g in GOALS) and len(pts) >= MIN_PTS:
        with_goal_frames.append(jf.stem)
        if all(g in d for g in GOALS):
            base_pts = [p for p in pts if p[0][0] not in [RINK[g][0] for g in GOALS] and not (abs(p[0][0]-RINK[g][0])<1e-6 and abs(p[0][1]-RINK[g][1])<1e-6)]
            base_pts = [(w, px) for w, px in pts if not (w == RINK["goal_L"] or w == RINK["goal_R"])]
            if len(base_pts) >= MIN_PTS:
                Hb, rmsb, _ = fit(base_pts)
                if Hb is not None:
                    err = {}
                    for g in GOALS:
                        w = np.float32([*RINK[g], 1.0])
                        p = (Hb @ w); p = (p[:2]/p[2])
                        act = np.float32([d[g]["x"], d[g]["y"]])
                        err[g] = round(float(np.linalg.norm(p - act)), 1)
                    rec["goal_extrap_err_px"] = err
                    rec["goal_actually_tapped"] = True
    qc[jf.stem] = rec

good = [k for k, v in qc.items() if "rms_px" in v]
low = [k for k, v in qc.items() if "skip" in v]
r = sorted(v["rms_px"] for v in qc.values() if "rms_px" in v)
extrap = [v["goal_extrap_err_px"] for v in qc.values() if "goal_extrap_err_px" in v]
print(f"fs labels: {len(qc)} | H computed: {len(good)} | too few points: {len(low)}")
if r:
    print(f"RMS px: min {r[0]:.1f} | p50 {r[len(r)//2]:.1f} | p90 {r[int(len(r)*0.9)]:.1f} | max {r[-1]:.1f}")
if extrap:
    print(f"goals extrapolation without anchors (px): {extrap}")
print("frames with goal points:", with_goal_frames[:8])
# distribution per video
byvid = {}
for k, v in qc.items():
    vid = k.rsplit("_",1)[0]
    b = byvid.setdefault(vid, [0,0])
    b[0 if "rms_px" in v else 1] += 1
print("per video (usable / junk zooms):", byvid)
(OUT / "qc.json").write_text(json.dumps(qc, ensure_ascii=False, indent=1))
