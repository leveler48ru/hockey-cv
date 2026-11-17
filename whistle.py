#!/usr/bin/env python3
"""Whistles and sirens from the m1 audio (16 kHz mono wav already extracted).

A referee whistle = a pure high tone: a narrow spectral peak at 2-6 kHz, high
tonality, sharp onset, duration 0.2-4 s. The goal siren - longer/louder, may be
two-tone; we catch it with the same detector and distinguish by length and energy.

Algorithm (no libraries besides numpy, scipy-free numpy):
  STFT frame=1024 hop=256 (16 ms), band 2000-6500 Hz.
  tone[k] = max-bin energy in band / band total energy   (spectral purity)
  band[k] = band energy / overall energy                (share of the "whistling" band)
  rel[k]  = band_energy / median(band_energy)           (loudness above the crowd background)
  flag = tone>0.55 & band_share>0.35 & rel>4  (thresholds checked via the histogram)
Merging frames into events (gap<=0.15 s), discarding <0.12 s and video-cut
artifacts (wide overlay?). Output: hockey/whistles.json + printing the distribution.
"""
import numpy as np, wave, json, os

OUT = os.path.dirname(os.path.abspath(__file__))
wf = wave.open(f'{OUT}/m1.wav', 'rb')
assert wf.getnchannels() == 1 and wf.getframerate() == 16000
x = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768
sr = 16000
print(f'audio {len(x)/sr:.1f} s')

FR, HO = 1024, 256
win = np.hanning(FR).astype(np.float32)
nf = (len(x) - FR) // HO
frames = np.lib.stride_tricks.as_strided(
    x, shape=(nf, FR), strides=(x.strides[0] * HO, x.strides[0]))
S = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2   # (nf, 513)
freqs = np.fft.rfftfreq(FR, 1.0 / sr)
band = (freqs >= 2000) & (freqs <= 6500)
low = freqs < 2000

band_e = S[:, band].sum(1)
tot_e = S.sum(1) + 1e-12
peak_e = S[:, band].max(1)
share = band_e / tot_e                    # share of the band in the overall noise
tone = peak_e / (band_e + 1e-12)          # tonal purity within the band (0..1)
med = np.median(band_e) + 1e-12
rel = band_e / med                        # band loudness above the match median

t = np.arange(nf) * HO / sr
score = share * tone
print('share percentiles:', np.round(np.percentile(share, [50, 90, 99, 99.9]), 3))
print('tone percentiles :', np.round(np.percentile(tone, [50, 90, 99, 99.9]), 3))
print('rel percentiles  :', np.round(np.percentile(rel, [50, 90, 99, 99.9]), 2))

flag = (tone > 0.45) & (share > 0.30) & (rel > 6.0)
print(f'flagged frames: {flag.sum()} ({100*flag.mean():.2f}%)')

# merging over time
ev = []
i = 0
n = len(flag)
while i < n:
    if flag[i]:
        j = i
        gap = 0
        while j + 1 < n and (flag[j + 1] or gap < int(0.15 * sr / HO)):
            if not flag[j + 1]:
                gap += 1
            j += 1
        dur = (j - i) * HO / sr
        if dur >= 0.12:
            seg = slice(i, j)
            ev.append({'t0': round(float(t[i]), 2), 't1': round(float(t[j]), 2),
                       'dur': round(float(dur), 2),
                       'peak_rel': round(float(rel[seg].max()), 1),
                       'tone': round(float(tone[seg].mean()), 2)})
        i = j + 1
    else:
        i += 1
ev = [e for e in ev if e['dur'] < 12.0]
print(f'"whistle-like tone" events: {len(ev)}')
for e in ev[:25]:
    print(f"  t={e['t0']:8.1f} dur {e['dur']:5.2f} rel {e['peak_rel']:6.1f} tone {e['tone']:.2f}")
json.dump({'sr': sr, 'n_events': len(ev), 'events': ev}, open(f'{OUT}/whistles.json', 'w'))
np.save(f'{OUT}/audio_feats.npy', np.stack([t, share, tone, np.log1p(rel)], 1).astype(np.float32))
print('wrote whistles.json, audio_feats.npy')
