#!/usr/bin/env python3
"""Offline analysis of the scoreboard plate cache (sb_plates.npy): where the clock is, where the score is.

Idea: the plate columns split themselves by behavior:
  - clock: changes on almost every pass (0.5 s step, a tick every ~2 probes);
  - score: changes rarely, but discretely;
  - static (names, background): barely changes.
We compute a per-column "jitter" (mean |diff| between neighboring probes),
isolate the clock band (high change frequency), and inside NON-clock
columns look for rare strong jumps = candidates for a score/period change.
Plus a separate "clock stopped" metric: the share of zero-jitter in the clock band over a
sliding window = game stoppages (whistle/goal/period break is already visible here).
"""
import numpy as np, json, cv2, os

OUT = os.path.dirname(os.path.abspath(__file__))
P = np.load(f'{OUT}/sb_plates.npy').astype(np.float32)
meta = json.load(open(f'{OUT}/sb_probe.json'))
ts = np.array([r['t'] for r in meta['probes']])
n, h, w = P.shape
print(f'plates {n}, {w}x{h}, step {meta["step"]} s')

# normalize by brightness so overlay flicker does not dominate
Pn = (P - P.mean(axis=(1, 2), keepdims=True)) / (P.std(axis=(1, 2), keepdims=True) + 1e-6)

D = np.abs(np.diff(Pn, axis=0))            # (n-1, h, w)
col = D.mean(axis=1)                        # (n-1, w) jitter per column

# column change frequency: share of passes where |diff| > 8 (0..255 scale)
CH = (np.abs(np.diff(P, axis=0)) > 8).mean(axis=0)
clock_cols = np.where(CH > 0.35)[0]
print('clock columns (jitter >35% of passes):', clock_cols.min() if len(clock_cols) else '-',
      '..', clock_cols.max() if len(clock_cols) else '-', f'({len(clock_cols)} pcs)')

# rare strong jumps OUTSIDE the clock columns
mask_rest = np.ones(w, bool)
mask_rest[clock_cols] = False
rest = D[:, :, mask_rest].mean(axis=(1, 2))   # (n-1,)
thr = np.percentile(rest, 99.5)
jump = rest > max(thr, 6.0)
idx = np.where(jump)[0]
print(f'jumps outside clock: {len(idx)} (threshold {thr:.2f})')
# collapse adjacent ones into a single event
ev = []
for i in idx:
    if ev and i - ev[-1][-1] <= 4:
        ev[-1].append(i)
    else:
        ev.append([i])
print(f'after event collapsing: {len(ev)}')
for e in ev:
    t0 = ts[e[0]]
    print(f"  t={t0:8.1f} s  run {len(e)}  peak {rest[e].max():.1f}")

# "clock stopped": share of zero jitter in the clock band, 20 s window
if len(clock_cols):
    cc = slice(clock_cols.min(), clock_cols.max() + 1)
    tick = D[:, cc].mean(axis=(1, 2))
    WIN = int(15 / meta['step'])  # 15 s window
    still = np.convolve((tick < 1.0).astype(float), np.ones(WIN) / WIN, 'valid')
    stops = np.where(still > 0.9)[0]
    # collapse
    seg = []
    for i in stops:
        if seg and i - seg[-1][-1] <= 10:
            seg[-1].append(i)
        else:
            seg.append([i])
    print(f'\nclock stops (>=15 s without a tick): {len(seg)} segments')
    step = meta['step']
    for s in seg[:40]:
        print(f'  {ts[s[0]]:8.1f} .. {ts[s[-1]]:8.1f}  ({(s[-1]-s[0])*step:.0f} s)')
    np.save(f'{OUT}/sb_clocktick.npy', tick)

json.dump({'events_off_clock': [{'t': round(float(ts[e[0]]), 1), 'peak': round(float(rest[e].max()), 1)} for e in ev]},
          open(f'{OUT}/sb_jumps.json', 'w'))
np.save(f'{OUT}/sb_rest_diff.npy', rest.astype(np.float32))
print('written: sb_jumps.json, sb_rest_diff.npy')
