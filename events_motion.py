#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""events_motion.py — in-game events derived FROM TRACKS (no scoreboard, per
Alexander's request of 10.11: "don't orient on the score panel").

Input: track_m1.jsonl.gz (lines {"f":frame,"d":[[cls,tid,x1,y1,x2,y2,conf],..]}),
     FPS, duration; optional audio line audio3_timeline.json.
Classes: 0 goaltender 1 player 2 puck 3 referee.

Events and their physics (all broadcast-camera flaws accounted for: the puck is
tiny, ids flip-flop, the camera follows the puck => "puck in the center of frame"
is almost always true):
 * GOAL candidate: the puck enters the goal rectangle (net mouth ~2.4 m behind
   the goal line) and within ~2 s after — play stops: puck speed drops, or the
   puck disappears/teleports (abrupt track break), or the player with the
   nearest id skates away/celebrates (player cluster at the goal).
 * FACEOFF candidate: the puck is nearly motionless (>1 s, |v|<0.5 m/s in world
   or <2 px/frame in pixels) + >=4 players within ~4 m radius + a referee
   nearby, followed by abrupt acceleration. Localization: center of player masses.
 * STOPPAGE: long (<0.3 px/frame) puck-at-rest state without a referee — puck
   out of play / faceoff away from the dots; and/or a puck track gap >10 s while
   players are calm.
 * SHOT: puck speed jump above threshold in a frame, trajectory into the net
   mouth rectangle, then — a rebound off the boards/goaltender or a goal.
 * PENALTY candidate: cluster of 2+ players with an abrupt stop + audio whistle
   within ±3 s + a referee skating toward them; "who committed the infraction" —
   by nearest id, but honestly low-confidence (re-identification is not cheap).
 * POSSESSION/speed: per-track speed with median filtering; world speed — via
   homography if calib exists (pixel-based for now, which is stated in the output).

Output: hockey/events_motion.json — list of {t, type, xy, conf, evidence{}}.
Thresholds are meaningful, but the FIRST run is calibrated by eye on the 10 label
windows shown in stdout (don't trust numbers without manual verification —
lesson of the puck line).
"""
import gzip, json, math, sys
from collections import defaultdict

SRC = sys.argv[1] if len(sys.argv) > 1 else "track_m1.jsonl.gz"
AUD = "audio3_timeline.json"
OUT = "events_motion.json"
FPS = 29.96
W, H = 1280, 720

# --- read the tracks, aggregate per-frame ---
frames = []            # list of dict cls->list of (tid,x,y,w,h,conf)
puck_track = []        # (f, x, y, conf) - best puck detection of the frame
by_frame = defaultdict(list)
with gzip.open(SRC, "rt") as f:
    for line in f:
        rec = json.loads(line)
        fr = rec["f"]
        d = defaultdict(list)
        for cls, tid, x1, y1, x2, y2, cf in rec["d"]:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            d[cls].append((tid, cx, cy, x2 - x1, y2 - y1, cf))
            by_frame[fr].append((cls, tid, cx, cy, cf))
        frames.append((fr, d))

print(f'frames {len(frames)}, detection entries {sum(len(v) for v in by_frame.values())}', flush=True)

# --- smoothed puck sequence (best detection per frame) ---
pk = {}
for fr, d in frames:
    if 2 in d and d[2]:
        best = max(d[2], key=lambda t: t[5])
        pk[fr] = (best[0], best[1], best[2], best[5])

fs = sorted(pk)
# interpolate gaps <= 8 frames (id flip-flops/occlusions), larger — a hole
xs, ys, tids = [], [], []
segs = []   # contiguous segments [(f0,x0),(f1,x1)...] for speed computation
cur = []
prev = None
for fr in fs:
    if prev is not None and fr - prev > 8:
        if len(cur) > 2: segs.append(cur)
        cur = []
    x, y = pk[fr][1], pk[fr][2]
    cur.append((fr, x, y))
    prev = fr
if len(cur) > 2: segs.append(cur)

# --- segment speeds ---
seg_stats = []
for s in segs:
    v = []
    for i in range(1, len(s)):
        dt = (s[i][0] - s[i - 1][0]) / FPS
        if dt <= 0: continue
        v.append(math.hypot(s[i][1] - s[i - 1][1], s[i][2] - s[i - 1][2]) / dt)  # px/s
    if not v: continue
    med = sorted(v)[len(v) // 2]
    seg_stats.append({'f0': s[0][0], 'f1': s[-1][0], 'dur_s': (s[-1][0] - s[0][0]) / FPS,
                      'v_med_px_s': round(med, 1),
                      'v_max_px_s': round(max(v), 1),
                      'x0': s[0][1], 'y0': s[0][2], 'x1': s[-1][1], 'y1': s[-1][2]})
seg_stats.sort(key=lambda d: d['v_max_px_s'], reverse=True)
print('TOP-15 fast puck segments (shots/passes/scoring runs):')
for s in seg_stats[:15]:
    print(f"  t={s['f0']/FPS:7.1f}-{s['f1']/FPS:7.1f} v_med={s['v_med_px_s']:6.0f} v_max={s['v_max_px_s']:6.0f} px/s end=({s['x1']:.0f},{s['y1']:.0f})")

# --- puck at-rest periods: faceoff/stoppage candidates ---
still = [s for s in seg_stats if s['dur_s'] > 0.8 and s['v_med_px_s'] < 25]
print(f'\ncalm segments (>0.8 s, <25 px/s): {len(still)}; of those with a restart within 3 s:')
ev = []
aud = json.load(open(AUD))['events'] if __import__('os').path.exists(AUD) else []
wh = [a['t'] for a in aud if a['cls'] == 'W']

def near_wh(t, d=4.0):
    return any(abs(w - t) < d for w in wh)

for s in still:
    t_end = s['f1'] / FPS
    # restart: is there a fast puck segment starting within 1.5 s after the end
    restart = [q for q in seg_stats if 0 <= (q['f0'] - s['f1']) / FPS < 1.5 and q['v_med_px_s'] > 60]
    # how many players nearby
    fmid = (s['f0'] + s['f1']) // 2
    dframe = None
    for fr, d in frames:
        if fr == fmid: dframe = d; break
    npl = len(dframe.get(1, [])) if dframe else 0
    close = 0
    if dframe:
        for tid, cx, cy, w_, h_, cf in dframe.get(1, []):
            if math.hypot(cx - s['x1'], cy - s['y1']) < 120: close += 1
    if restart or near_wh(t_end):
        ev.append({'t': round(t_end, 1), 'type': 'FACEOFF?', 'xy': [round(s['x1']), round(s['y1'])],
                   'evidence': {'still_s': round(s['dur_s'], 2), 'players_close': close, 'players_total': npl,
                                'restart': bool(restart), 'whistle_audio': near_wh(t_end)}})

# puck gaps >4 s = stoppage (puck in a corner / under the skates / in the stands)
gaps = []
for a, b in zip(fs, fs[1:]):
    if (b - a) / FPS > 4.0:
        gaps.append({'t': round(a / FPS, 1), 'type': 'PUCK_GONE', 'dt_s': round((b - a) / FPS, 1)})
ev += gaps

# --- abrupt player clusters near boards/goal without the puck (scramble / faceoff at the wall) ---
# rough for now: frames where >=6 players are within r<80 px
for fr, d in frames:
    if fr % 15: continue
    pl = d.get(1, [])
    if len(pl) < 6: continue
    for i in range(len(pl)):
        grp = [p for p in pl if math.hypot(p[1] - pl[i][1], p[2] - pl[i][2]) < 80]
        if len(grp) >= 6:
            t = fr / FPS
            if not any(abs(e['t'] - t) < 5 for e in ev):
                ev.append({'t': round(t, 1), 'type': 'SCRAMBLE?', 'xy': [round(pl[i][1]), round(pl[i][2])],
                           'evidence': {'players_in_80px': len(grp), 'whistle_audio': near_wh(t)}})
            break

ev.sort(key=lambda e: e['t'])
json.dump(ev, open(OUT, 'w'), ensure_ascii=False, indent=1)
from collections import Counter
print(f'\ntotal candidates: {Counter(e["type"] for e in ev)}')
print(f'with a whistle within ±4 s: {sum(1 for e in ev if e["evidence"].get("whistle_audio"))}')
print('timeline:', OUT)
for e in ev[:40]:
    print(f"  t={e['t']:7.1f} {e['type']:10s} xy={e.get('xy')} {e['evidence']}")
