#!/usr/bin/env python3
"""Fuse channels into a single event timeline for m1 (STURM–LIPetsk).

Channels (all already computed, nothing is retrained):
  A) audio2_probs.npz -> audio_lines.json   : W/S/M peaks, thresholds tuned on 70 labels
  B) whistle_stats.json                      : classic whistle detector, 23 events
  C) siren_clusters.json                     : low-frequency clusters (80) — CHANNEL UNDER SUSPICION
  D) sb_probe.json                            : scoreboard plates, "wide overlay" flag

Principle: an event is born from channel AGREEMENT, not from a single channel.
Disagreements are not hidden — they go into a separate list, because we have no
match protocol and the only available metric = consistency between independent channels.
"""
import json
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(f'{OUT}/audio_lines.json'))
WS = json.load(open(f'{OUT}/whistle_stats.json'))
SIREN = json.load(open(f'{OUT}/siren_clusters.json'))
SB = json.load(open(f'{OUT}/sb_probe.json'))
lab = json.load(open(f'{OUT}/audio_labels.json'))

# --- D: wide-overlay clusters from scoreboard probes (0.5 s step) ---
ts = np.array([p['t'] for p in SB['probes']])
wide = np.array([bool(p['wide']) for p in SB['probes']])
step = SB['step']
clusters = []
i = 0
while i < len(wide):
    if wide[i]:
        j = i
        while j + 1 < len(wide) and wide[j + 1] and (ts[j + 1] - ts[j]) <= 2 * step:
            j += 1
        clusters.append((ts[i], ts[j]))
        i = j + 1
    else:
        i += 1
ov = [{'t0': a, 't1': b} for a, b in clusters if b - a >= 2.0]
print(f'overlay: total clusters {len(clusters)}, with duration >=2 s: {len(ov)}')


def in_overlay(t, pad=0.0):
    return any(o['t0'] - pad <= t <= o['t1'] + pad for o in ov)


def near(lst, t, key0, key1, tol):
    """nearest interval in lst to time t, if distance <= tol"""
    best, bd = None, 1e9
    for d in lst:
        lo, hi = d[key0], d[key1]
        dist = 0.0 if lo - pad_out(t, lo, hi) == 0 else abs(t - (lo if t < lo else hi))
        if dist <= tol and dist < bd:
            best, bd = d, dist
    return best, bd


def pad_out(t, lo, hi):
    return 0.0 if lo <= t <= hi else 1e9


wh_events = [(a, b) for a, b, _ in WS['events']]
siren_events = [(c['t0'], c['t1']) for c in SIREN]


def nearest_iv(t, ivs):
    if not ivs:
        return None, 1e9
    d = [0 if a <= t <= b else min(abs(t - a), abs(t - b)) for a, b in ivs]
    k = int(np.argmin(d))
    return ivs[k], float(d[k])


# --- labels as a check of the fusion ---
labc = {round(c['t0'], 1): c['label'] for c in lab}

events = []
for e in A['peaks']['S']:
    t = e['t']
    wv, wd = nearest_iv(t, wh_events)
    sv, sd = nearest_iv(t, siren_events)
    events.append({
        't': t, 'type': 'GOAL?', 'p': e['p'],
        'ev': {'whistle_dist': round(wd, 1), 'siren_dist': round(sd, 1),
               'overlay': in_overlay(t, pad=5.0)},
        'agree': int(wd <= 10) + int(sd <= 8) + int(in_overlay(t, pad=5.0)) + 1,
    })

for e in A['peaks']['W']:
    t = e['t']
    wv, wd = nearest_iv(t, wh_events)
    events.append({
        't': t, 'type': 'WHISTLE', 'p': e['p'],
        'ev': {'classic_whistle_dist': round(wd, 1), 'overlay': in_overlay(t, pad=8.0)},
        'agree': int(wd <= 6) + int(in_overlay(t, pad=8.0)) + 1,
    })

for e in A['peaks']['M']:
    events.append({'t': e['t'], 'type': 'MUSIC/ANN', 'p': e['p'],
                   'ev': {'overlay': in_overlay(e['t'], pad=5.0)}, 'agree': 1})

events.sort(key=lambda d: d['t'])

goals = [e for e in events if e['type'] == 'GOAL?']
wh = [e for e in events if e['type'] == 'WHISTLE']
print(f'\ngoal candidates: {len(goals)}, whistles: {len(wh)}')
print('\n=== GOAL candidates (channel A + cross-check B/C/D) ===')
print('   t    p    whistle±  siren±  overlay  agreement')
for g in goals:
    lab_at = [f'{k}:{v}' for k, v in labc.items() if abs(k - g['t']) <= 12]
    print(f"{g['t']:7.0f} {g['p']:.2f}  {g['ev']['whistle_dist']:>7}  {g['ev']['siren_dist']:>7}  "
          f"{'YES' if g['ev']['overlay'] else 'no':5s}  {g['agree']}/4   {','.join(lab_at) or ''}")

print('\n=== whistles: agreement between NN (A) and classic detector (B) ===')
both = sum(1 for w in wh if w['ev']['classic_whistle_dist'] <= 6)
onlyA = [round(w['t']) for w in wh if w['ev']['classic_whistle_dist'] > 6]
print(f'NN peaks {len(wh)}, confirmed by classic detector (±6 s): {both}, NN only: {len(onlyA)}')
print(f'NN only: {onlyA}')
classic_only = []
for a, b in wh_events:
    if not any(abs(w['t'] - (a + b) / 2) <= 8 for w in wh):
        classic_only.append(round(a, 1))
print(f'classic detector (23) without NN confirmation: {len(classic_only)} -> {classic_only}')

json.dump({'thr': A['thr'], 'n_overlay_clusters': len(ov),
           'overlay_clusters': ov, 'events': events,
           'agreement': {'goal_candidates': len(goals), 'whistle_nn': len(wh),
                         'whistle_confirmed_by_classic': both,
                         'classic_unmatched': classic_only}},
          open(f'{OUT}/events_m1.json', 'w'), ensure_ascii=False, indent=1)
print(f'\nwrote events_m1.json ({len(events)} events, overlay clusters {len(ov)})')
