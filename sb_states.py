#!/usr/bin/env python3
"""Scoreboard as an event channel: plate state -> timestamps of discrete changes.
The plate is stable (median bbox from sb_scan: x=63 y=40 w=308 h=60). Step 1 s,
differential over the averaged norm; a jump held for >=3 passes = new state.
Output: hockey/sb_states.json + hockey/sb_tiles2/*.jpg (one tile per state).
"""
import cv2, numpy as np, os, json, sys

SRC = './media/8802578402043.mp4'
STEP = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
X, Y, W, H = 63, 40, 308, 60
OUT = os.path.dirname(os.path.abspath(__file__))
T = f'{OUT}/sb_tiles2'
os.makedirs(T, exist_ok=True)

cap = cv2.VideoCapture(SRC)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
step_f = max(1, int(round(STEP * fps)))

def plate(fr):
    c = fr[Y:Y+H, X:X+W]
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    g = (g - g.mean()) / (g.std() + 1e-6)
    return c, g

prev = None
hold = 0
states = []          # (t, mean_diff_from_prev_state)
seq = []             # each probe: state id
cur_id = 0
sid = 0
raw = []
n = 0
while n < total:
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, fr = cap.read()
    if not ok:
        break
    t = n / fps
    c, g = plate(fr)
    raw.append((t, c.copy() if sid in (0, ) else None))
    if prev is None:
        prev = g
        seq.append((t, sid))
    else:
        d = np.abs(g - prev).mean()
        if d > 0.75:
            hold += 1
            if hold >= 3:
                sid += 1
                states.append({'t': round(t, 2), 'jump': round(float(d), 3), 'state': sid})
                fn = f'{T}/{int(t):05d}.jpg'
                cv2.imwrite(fn, cv2.resize(c, (W*2, H*2), interpolation=cv2.INTER_CUBIC))
                states[-1]['tile'] = fn
                prev = g
                hold = 0
        else:
            hold = 0
            # reference adaptation: slow drift (timer) must not break the state
            prev = 0.85 * prev + 0.15 * g
        seq.append((t, sid))
    n += step_f

json.dump({'dur': total/fps, 'step': STEP, 'n_states': sid+1, 'states': states,
           'seq': [[round(t, 1), s] for t, s in seq]}, open(f'{OUT}/sb_states.json', 'w'))
print(f'passes {len(seq)}, states {sid+1}')
for s in states:
    print(f"t={s['t']:8.2f} state {s['state']:2d} jump {s['jump']:.3f} -> {s['tile']}")
