#!/usr/bin/env python3
"""Arena goal siren matched to the REFERENCE from Alexander's ear (sample 1123.6..1129.6).

The reference spectrum (1127.5-1129.5 vs background 1120-1122) showed: the arena siren is
NOT a high tone but a low horn: a line at 70..140 Hz (fundamental ~100-110 Hz)
dominating the whole lower register. The whistle (2.3 kHz) is a separate channel.

Detector: FR=8192 (0.512 s), hop=2048 (0.128 s).
  line   = max S[60..170 Hz]
  floor  = sum S[40..450 Hz]
  ratio  = line/floor * (n_bins_low/n_bins_high)   -> share of the "line" in the roar
  loud   = line / median(line)
Flag: ratio>0.30 and loud>6. Merging gap<=0.3 s, length>=0.7 s.
Plus: pairs with whistles (whistles.json, window -6..+8 s) = GOAL candidate.
"""
import numpy as np, wave, json, os

OUT = os.path.dirname(os.path.abspath(__file__))
wf = wave.open(f'{OUT}/m1.wav', 'rb')
x = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768
sr = 16000
FR, HO = 8192, 2048
win = np.hanning(FR).astype(np.float32)
f = np.fft.rfftfreq(FR, 1 / sr)
lo_i = (f >= 60) & (f <= 170)
fl_i = (f >= 40) & (f <= 450)
share_norm = fl_i.sum() / lo_i.sum()

def scan(a, b):
    s, e = int(a * sr), int(min(b, len(x) / sr) * sr)
    seg = x[s:e]
    n = (len(seg) - FR) // HO + 1
    if n <= 0:
        return None, None, None
    st = np.lib.stride_tricks.as_strided(seg, (n, FR), (seg.strides[0] * HO, seg.strides[0]))
    S = np.abs(np.fft.rfft(st * win, axis=1)) ** 2
    line = S[:, lo_i].max(1)
    floor = S[:, fl_i].sum(1) + 1e-12
    ratio = line / floor * share_norm
    tt = np.arange(n) * HO / sr + a
    return tt, line, ratio

TT, LL, RR = [], [], []
for a in range(0, int(len(x) / sr), 240):
    t_, l_, r_ = scan(a, a + 240.5)
    if t_ is None:
        continue
    TT.append(t_); LL.append(l_); RR.append(r_)
T = np.concatenate(TT); L = np.concatenate(LL); R = np.concatenate(RR)
med = np.median(L) + 1e-12
loud = L / med
print(f'frames {len(T)}; ratio percentiles: {np.round(np.percentile(R,[50,90,99,99.9]),2)}; loud: {np.round(np.percentile(loud,[50,99,99.9]),1)}')

flag = (R > 0.30) & (loud > 6)
print('flagged frames:', int(flag.sum()), f'({100*flag.mean():.2f}%)')
ev = []
i = 0
n = len(flag)
while i < n:
    if flag[i]:
        j = i; gap = 0
        while j + 1 < n and (flag[j + 1] or gap < int(0.3 * sr / HO)):
            if not flag[j + 1]:
                gap += 1
            j += 1
        if T[j] - T[i] >= 0.7:
            ev.append({'t0': round(float(T[i]), 2), 't1': round(float(T[j]), 2),
                       'dur': round(float(T[j] - T[i]), 2),
                       'loud': round(float(loud[i:j].max()), 1)})
        i = j + 1
    else:
        i += 1
print(f'\nsiren segments (>=0.7 s): {len(ev)}')
for e in ev:
    print(f"  {e['t0']:8.1f}..{e['t1']:8.1f}  dur {e['dur']:5.2f}  loud {e['loud']:7.1f}")

wh = json.load(open(f'{OUT}/whistles.json'))['events']
pairs = []
for e in ev:
    ws = [w for w in wh if e['t0'] - 6 <= w['t0'] <= e['t1'] + 8]
    if ws:
        pairs.append({'siren': e, 'whistle_t': ws[0]['t0'], 'goal_t': round(min(ws[0]['t0'], e['t0']), 1)})
print(f'\nwhistle+siren pairs = goal candidates: {len(pairs)}')
for p in pairs:
    print(f"  goal ~{p['goal_t']:8.1f} s (whistle {p['whistle_t']}, siren {p['siren']['t0']}..{p['siren']['t1']})")
json.dump({'events': ev, 'goal_pairs': pairs}, open(f'{OUT}/sirens.json', 'w'))
print('wrote sirens.json')
