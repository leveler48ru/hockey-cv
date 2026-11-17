#!/usr/bin/env python3
"""Match pan profile via phase correlation (fast, no ORB).
Every 2 s: dx,dy shifts. The profile is written to match/pan_profile.json -
it is later used to pick keyframes for chains and the mosaic."""
import cv2, numpy as np, json, time

F = "./media/9281020430926.mp4"
STEP = 2  # seconds
W_SMALL = 640

cap = cv2.VideoCapture(F)
fps = cap.get(cv2.CAP_PROP_FPS)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
dur = n / fps
scale = W_SMALL / 1920.0

win = np.hanning(64)[:, None] * np.ones((1, 360))  # for 640x360: need hann along width
wx = np.hanning(W_SMALL)[None, :] * np.ones((int(360), 1))

prev = None
prof = []
t0 = time.time()
sec = 0.0
idx = 0
while True:
    ok = cap.grab()
    if not ok:
        break
    if idx % int(round(STEP * fps)) == 0:
        ok2, fr2 = cap.retrieve()
        if ok2:
            small = cv2.resize(fr2, (W_SMALL, 360), interpolation=cv2.INTER_AREA)
            g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
            g = g * wx
            if prev is not None:
                sh, _ = cv2.phaseCorrelate(prev, g)
                prof.append((round(sec, 1), round(sh[0] / scale, 1), round(sh[1] / scale, 1)))
            prev = g
            sec = idx / fps
    idx += 1
cap.release()

# accumulated trajectory (in full-frame pixels)
cx, cy = [0.0], [0.0]
for s, dx, dy in prof:
    cx.append(cx[-1] + dx); cy.append(cy[-1] + dy)
prof_out = {"step_s": STEP, "n": len(prof),
            "total_dx": round(cx[-1], 1), "total_dy": round(cy[-1], 1),
            "min_dx": round(min(cx), 1), "max_dx": round(max(cx), 1),
            "secs": [p[0] for p in prof], "dx": [p[1] for p in prof],
            "cum_x": [round(v, 1) for v in cx], "cum_y": [round(v, 1) for v in cy]}
json.dump(prof_out, open("match/pan_profile.json", "w"))
print(f"done in {time.time()-t0:.0f} s; {len(prof)} sampled frames")
print(f"pan shift: min {min(cx):.0f} max {max(cx):.0f} (final {cx[-1]:.0f}) full-frame px")
# top-10 fastest pans
spd = [(abs(p[1]), p[0]) for p in prof]
spd.sort(reverse=True)
print("max pan speed (px/2s):", [(round(a), s) for a, s in spd[:8]])
