#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""world_m1.py — convert the puck track to meters (60x30) via pan_m1.npz and honest
physics: speeds, faceoffs, shots. CPU, NAS.

Input: hockey/track_m1.jsonl.gz, hockey/pan/pan_m1.npz
Output: hockey/world_m1.json (events), stdout — QC summary.
World: x 0..60 along the rink, y 0..30. Frame f -> nearest sample (stride 5, <=0.08 s).
"""
import gzip, json, math, sys
from collections import defaultdict
import numpy as np

BASE = "."
FPS = 29.96

z = np.load(f"{BASE}/pan/pan_m1.npz")
samples = z["samples"].astype(int)          # every stride-th frame
A = z["A"].astype(np.float64)               # (n,2,3) pixel->world
H0 = z["H0"].astype(np.float64)
okmask = (np.abs(A).sum(axis=(1, 2)) > 0)
print(f"samples={len(samples)} ok={okmask.sum()} groups={z['G'].max()+1} "
      f"anchored={z['group_anchored']}")

# QC of the chain: translation and scale relative to H0
def row_scale(M):
    return float(np.hypot(M[0, 0], M[0, 1]))
s0 = row_scale(H0)
sc = np.array([row_scale(A[i]) for i in range(len(A))])
tx = A[:, 0, 2]
print(f"H0 scale={s0:.3f}; chain scale min/med/max = {sc[okmask].min():.3f}/"
      f"{np.median(sc[okmask]):.3f}/{sc[okmask].max():.3f}")
print(f"world X-offset (px->world x): min={tx[okmask].min():.1f} max={tx[okmask].max():.1f}")

# --- track: best puck detection per frame ---
pk = {}
with gzip.open(f"{BASE}/track_m1.jsonl.gz", "rt") as f:
    for line in f:
        rec = json.loads(line)
        best = None
        for cls, tid, x1, y1, x2, y2, cf in rec["d"]:
            if cls == 2 and (best is None or cf > best[1]):
                best = (((x1 + x2) / 2, (y1 + y2) / 2), cf, tid)
        if best:
            pk[rec["f"]] = best

fs = sorted(pk)
print(f"puck frames: {len(fs)} ({len(fs)/FPS/60:.1f} min of {126063/FPS/60:.1f})")

# --- into the world ---
import bisect
def H_of(fr):
    i = bisect.bisect_left(samples, fr)
    if i >= len(samples): i = len(samples) - 1
    if i > 0 and samples[i] - fr > fr - samples[i - 1]: i -= 1
    if not okmask[i]: return None
    return A[i]

W = []  # (f, x_m, y_m, conf)
out_rink = inside = None_ct = 0
for fr in fs:
    M = H_of(fr)
    if M is None:
        None_ct += 1; continue
    x, y = pk[fr][0]
    q = M @ np.array([x, y, 1.0])
    wx, wy = q[0], q[1]
    W.append((fr, wx, wy, pk[fr][1]))
    if 0 <= wx <= 60 and 0 <= wy <= 30: inside += 1
    else: out_rink += 1
print(f"converted to world: {len(W)} (no H: {None_ct}); inside the rink {inside} "
      f"({100*inside/max(1,inside+out_rink):.1f}%), outside {out_rink}")

# --- segments and speeds (m/s) ---
segs, cur = [], []
for (fr, x, y, cf) in W:
    if cur and fr - cur[-1][0] > 8:
        if len(cur) > 3: segs.append(cur)
        cur = []
    cur.append((fr, x, y))
if len(cur) > 3: segs.append(cur)

seg_stats = []
for s in segs:
    v = []
    for i in range(1, len(s)):
        dt = (s[i][0] - s[i-1][0]) / FPS
        if dt <= 0: continue
        v.append(math.hypot(s[i][1]-s[i-1][1], s[i][2]-s[i-1][2]) / dt)
    if not v: continue
    seg_stats.append({"t0": s[0][0]/FPS, "t1": s[-1][0]/FPS,
                      "dur": (s[-1][0]-s[0][0])/FPS,
                      "v_med": sorted(v)[len(v)//2], "v_max": max(v),
                      "start": (s[0][1], s[0][2]), "end": (s[-1][1], s[-1][2])})
allv = [q for s in seg_stats for q in [s["v_med"]]]
vmaxs = sorted([s["v_max"] for s in seg_stats])
print(f"segments: {len(segs)}; v_med over segments: p50={np.median(allv):.1f} "
      f"p90={np.percentile(allv,90):.1f} p99={np.percentile(allv,99):.1f} max={max(allv):.1f} m/s")
print(f"v_max over segments: p50={np.median(vmaxs):.1f} p95={np.percentile(vmaxs,95):.1f} max={vmaxs[-1]:.1f}")

print("\nTOP-12 fastest segments (shot candidates):")
for s in sorted(seg_stats, key=lambda d: -d["v_max"])[:12]:
    print(f"  t={s['t0']:7.1f}-{s['t1']:6.1f} v_max={s['v_max']:5.1f} m/s "
          f"({s['v_max']*3.6:4.0f} km/h) ({s['start'][0]:5.1f},{s['start'][1]:4.1f})->"
          f"({s['end'][0]:5.1f},{s['end'][1]:4.1f})")

# --- events in meters ---
ev = []
# faceoff: segment >0.8 s with v_med < 0.5 m/s, restart >2 m/s within 1.5 s after
for s in seg_stats:
    if s["dur"] > 0.8 and s["v_med"] < 0.5:
        rest = [q for q in seg_stats if 0 <= q["t0"]-s["t1"] < 1.5 and q["v_med"] > 2]
        if rest or True:
            ev.append({"t": round(s["t1"],1), "type": "FACEOFF?",
                       "world": [round(s["end"][0],1), round(s["end"][1],1)],
                       "still_s": round(s["dur"],1), "restart": bool(rest)})
# gaps >4 s
gaps = []
for a, b in zip(fs, fs[1:]):
    if (b-a)/FPS > 4.0:
        gaps.append({"t": round(a/FPS,1), "type": "PUCK_GONE", "dt_s": round((b-a)/FPS,1)})
ev += gaps
# shot: v_max >= 18 m/s (65 km/h) — calibrated by the distribution above
shots = [s for s in seg_stats if s["v_max"] >= 18]
for s in shots:
    ev.append({"t": round(s["t0"],1), "type": "SHOT?", "v_max": round(s["v_max"],1),
               "world": [round(s["start"][0],1), round(s["start"][1],1)],
               "end": [round(s["end"][0],1), round(s["end"][1],1)]})
ev.sort(key=lambda e: e["t"])
from collections import Counter
print(f"\nevents: {Counter(e['type'] for e in ev)}")
json.dump({"qc": {"inside_pct": round(100*inside/max(1,inside+out_rink),1),
                  "v_med_p99": round(float(np.percentile(allv,99)),1),
                  "groups": int(z['G'].max()+1)},
           "events": ev}, open(f"{BASE}/world_m1.json", "w"), ensure_ascii=False, indent=1)
print("saved", f"{BASE}/world_m1.json")
