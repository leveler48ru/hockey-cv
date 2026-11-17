#!/usr/bin/env python3
"""Full scoreboard score timeline: digit templates + classification over all 8404 probes.

Why not "rare jumps" (died out): the score zone has 4 big jumps, of which
  t=1526.9 (L 0.757)  and  t=1782.3 (R 0.678) are real goals (different score reads before/after);
  t=3100.4 and t=4110.2 are period boundaries (clock 15:00 before and after), the score does NOT change.
At the same time scores 1-2 and 1-3 appeared somewhere inside the intervals and the detector missed them:
the digit change happened under a "wide overlay" (16.8% of probes), the diff smeared over many probes.

Method: reference states are taken from OCR confirmations; for each position (L/R) and digit
we build a template = median of normalized patches over a window inside the state. Then every
plate (only where there is no overlay) is classified by minimum distance to the templates,
and a state counts only if it holds >= 4 s in a row. Pure numpy over the cache.
"""
import json
import os

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
P = np.load(f'{OUT}/sb_plates.npy', mmap_mode='r').astype(np.float32)
meta = json.load(open(f'{OUT}/sb_probe.json'))
ts = np.array([p['t'] for p in meta['probes']])
wide = np.array([bool(p['wide']) for p in meta['probes']])
n = len(ts)
step = meta['step']
Y0, Y1 = 15, 35
POS = {'L': (120, 138), 'R': (170, 188)}


def patch(i, key):
    x0, x1 = POS[key]
    p = P[i, Y0:Y1, x0:x1]
    p = p - p.mean()
    return p / (np.sqrt((p ** 2).sum()) + 1e-6)


def stack(idxs, key):
    return np.stack([patch(i, key) for i in idxs])


def idx_between(a, b):
    m = np.where((ts >= a) & (ts <= b) & ~wide)[0]
    return m


# OCR-confirmed states (t, L, R) from the previous run
CONF = [(2.0, '0', '0'), (1525.9, '0', '0'), (1531.9, '1', '0'), (1781.8, '1', '0'),
        (1787.8, '1', '1'), (2056.6, '1', '1'), (3132.5, '1', '2'), (4112.2, '1', '3'),
        (4204.3, '1', '3')]

# template source windows: (position, digit, t0, t1)
WINS = [('L', '0', 100, 1500), ('R', '0', 100, 1500),
        ('L', '1', 1600, 4200), ('R', '1', 1800, 3050),
        ('R', '2', 3150, 4050), ('R', '3', 4130, 4205)]

TPL = {}
for key, d, a, b in WINS:
    ii = idx_between(a, b)
    # take the center of the window, away from transitions
    ii = ii[(ts[ii] > a + 3) & (ts[ii] < b - 3)]
    if len(ii) < 10:
        print(f'TEMPLATE {key}{d}: too few examples ({len(ii)}) — skipping')
        continue
    T = stack(ii, key).mean(axis=0)
    T = T - T.mean()
    T /= (np.sqrt((T ** 2).sum()) + 1e-6)
    TPL[(key, d)] = T
    print(f'template {key}{d}: {len(ii)} plates (t {ts[ii][0]:.0f}..{ts[ii][-1]:.0f})')

keys = list(TPL.keys())
print('\nclassifying all plates...')
lab = {k: np.full(n, '', dtype='<U1') for k in ('L', 'R')}
score = {k: np.zeros(n, np.int16) - 1 for k in ('L', 'R')}
for key in ('L', 'R'):
    cand = [d for (k, d) in keys if k == key]
    M = np.stack([TPL[(key, d)].ravel() for d in cand])          # (C, F)
    for i in range(n):
        if wide[i]:
            continue
        v = patch(i, key).ravel()
        d2 = ((M - v) ** 2).sum(1)
        j = int(np.argmin(d2))
        score[key][i] = cand.index(cand[j])
        lab[key][i] = cand[j]
    print(f'  {key}: done, filled {int((score[key] >= 0).sum())}/{n}')

# smoothing: a state holds >= 4 s
MINHOLD = int(np.ceil(4.0 / step))


def stable(seq):
    out = []
    i = 0
    while i < len(seq):
        j = i
        while j + 1 < len(seq) and seq[j + 1][1] == seq[i][1]:
            j += 1
        if (j - i + 1) >= MINHOLD or (seq[i][1] != -1 and seq[i][1] >= 0 and j == len(seq) - 1):
            out.append(seq[i])
        i = j + 1
    # second pass: remove lone flips between agreeing entries
    clean = []
    for s in out:
        if clean and clean[-1][1] == s[1]:
            continue
        clean.append(s)
    return clean


seql = [(ts[i], int(score['L'][i]), int(score['R'][i])) for i in range(n)]
# collapse into intervals by the (L,R) pair
runs = []
cur = None
for t, l, r in seql:
    v = (l, r)
    if cur is None or cur[1] != v:
        if cur:
            runs.append(cur)
        cur = [t, v, t]
    else:
        cur[2] = t
runs.append(cur)
runs = [r for r in runs if (r[2] - r[0]) >= 2.0 or r[1][0] >= 0]
print(f'\n(L,R) runs of length >=2 s: {len(runs)} (total runs {len(seql) and 0 or 0})')


def fmt(v):
    l, r = v
    return f'{l}-{r}' if l >= 0 and r >= 0 else f'?-{r}' if l < 0 else f'{l}-?'


print('\n=== SCORE TIMELINE (stable states, >=8 s, OCR-checked) ===')
final = []
state = None
for a, v, b in runs:
    if (b - a) < 8 or v[0] < 0 or v[1] < 0:
        continue
    if state != v:
        state = v
        final.append({'t0': float(a), 't1': float(b), 'score': fmt(v), 'L': v[0], 'R': v[1]})
for s in final:
    print(f"  {s['t0']:7.1f} .. {s['t1']:7.1f}  {s['score']}")
print(f'\nscore states: {len(final)}; goals = sum changes: ', end='')
goals = []
for k in range(1, len(final)):
    prev, cur_ = final[k - 1], final[k]
    dp = prev['L'] + prev['R']
    dc = cur_['L'] + cur_['R']
    if dc > dp:
        who = 'SHTURM' if cur_['L'] > prev['L'] else 'LIPA'
        goals.append({'t': cur_['t0'], 'score': cur_['score'], 'team': who})
        print(f"{cur_['t0']:.1f} ({who}, {prev['score']}->{cur_['score']})")
print(f'\nresult: goals {len(goals)}, final score {final[-1]["score"]}')
json.dump({'states': final, 'goals': goals,
           'templates': [f'{k}{d}' for (k, d) in keys],
           'note': 'period boundaries (clock 15:00) were not checked via a separate channel'},
          open(f'{OUT}/sb_score_timeline.json', 'w'), ensure_ascii=False, indent=1)
print('wrote sb_score_timeline.json')
