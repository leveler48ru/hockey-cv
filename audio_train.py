#!/usr/bin/env python3
"""Training the audio event detector (the GPU server, GPU) on annotated m1 clips.

Classes: N=background, W=whistle, S=siren, WS=goal (whistle+siren), M=music/announcement.
Data: /work/m1.wav (16 kHz mono) + /work/audio_labels.json [{t0,dur,label}].
Feature: log-mel 64 (20-8000 Hz), STFT 1024/256 -> ~16 ms frame, 12 s window ~ 750 frames.
Model: lightweight ConvNet over time (4 conv blocks + global pooling) -> 5 classes.
Evaluation: stratified 5-fold CV BY CLIPS (important: neighboring windows of the same clip
must not overlap between train/val), per-class F1 metrics + binary "siren (S|WS)".
Inference: sliding over the whole match, 8 s window, 2 s step -> /work/audio_probs.npz and
/work/audio_timeline.json (probability peaks).
Output: model_audio_v1.pt in /work (grab it back to the NAS!).
"""
import json, wave, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SR = 16000
FR, HO = 1024, 256
N_MEL = 64
CLASSES = ['N', 'W', 'S', 'WS', 'M']
CLS = {c: i for i, c in enumerate(CLASSES)}

# ---------- features ----------
def load_wav(p):
    wf = wave.open(p, 'rb')
    x = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768
    assert wf.getframerate() == SR, wf.getframerate()
    assert wf.getnchannels() == 1, wf.getnchannels()
    return x

def melbank(sr, fr, n_fft, n_mel, lo, hi):
    def hz2mel(f): return 2595 * np.log10(1 + f / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    ms = np.linspace(hz2mel(lo), hz2mel(hi), n_mel + 2)
    fs = mel2hz(ms)
    bins = np.floor((n_fft + 1) * fs / sr).astype(int)
    W = np.zeros((n_mel, n_fft // 2 + 1), np.float32)
    for i in range(n_mel):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c > l:
            W[i, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            W[i, c:r] = (r - np.arange(c, r)) / (r - c)
    return W

MELW = melbank(SR, FR, FR, N_MEL, 20, 8000)
WINF = np.hanning(FR).astype(np.float32)

def feats(x, a, b):
    """log-mel for window [a,b) seconds: (T,64)"""
    s, e = int(a * SR), int(min(b, len(x) / SR) * SR)
    seg = x[s:e]
    if len(seg) < FR:
        seg = np.pad(seg, (0, FR - len(seg)))
    n = 1 + (len(seg) - FR) // HO
    st = np.lib.stride_tricks.as_strided(seg, (n, FR), (seg.strides[0] * HO, seg.strides[0]))
    S = np.abs(np.fft.rfft(st * WINF, axis=1)) ** 2
    M = S @ MELW.T
    return np.log(M + 1e-8).astype(np.float32)

# ---------- model ----------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(nn.Conv1d(ci, co, 5, padding=2), nn.BatchNorm1d(co),
                                 nn.ReLU(), nn.MaxPool1d(2))
        self.b = nn.Sequential(blk(N_MEL, 32), blk(32, 64), blk(64, 128), blk(128, 128))
        self.fc = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 5))

    def forward(self, z):           # (B, T, 64)
        h = self.b(z.transpose(1, 2))
        h = torch.cat([h.mean(-1), h.max(-1).values], 1)
        return self.fc(h)

# ---------- data ----------
def main():
    x = load_wav('/work/m1.wav')
    lab = json.load(open('/work/audio_labels.json'))
    dur = len(x) / SR
    print(f'audio {dur:.0f} s, clips {len(lab)}')

    Lwin = 8.0   # training window: 8 s (from the 12-second clip we take 0-8/2-10/4-10 — three views)
    xs, ys, gs = [], [], []
    for ci, c in enumerate(lab):
        y = CLS.get(c['label'])
        if y is None: continue
        for off in (0.0, 2.0, 4.0):
            if c['t0'] + off + Lwin > dur: continue
            xs.append(feats(x, c['t0'] + off, c['t0'] + off + Lwin))
            ys.append(y); gs.append(ci)
    X = np.stack(xs); Y = np.array(ys)
    u, G = np.unique(np.array(gs), return_inverse=True)
    k = len(u)
    print('views:', len(X), 'clips:', k)

    # normalization over the whole match
    mu, sd = X.mean(), X.std()
    X = (X - mu) / sd
    Xt = torch.tensor(X).to(DEV)  # (B,T,64)
    Yt = torch.tensor(Y, dtype=torch.long).to(DEV)
    Gt = torch.tensor(G)

    cnt = np.bincount(Y, minlength=5)
    wts = torch.tensor((cnt.sum() / (5 * np.maximum(cnt, 1))), dtype=torch.float32).to(DEV)

    folds = 5
    rng = np.random.default_rng(7)
    uniq = np.unique(G)
    fold_of = dict(zip(rng.permutation(uniq), np.arange(len(uniq)) % folds))
    vf = Gt.numpy().map(fold_of) if hasattr(Gt.numpy(), 'map') else np.array([fold_of[g] for g in Gt.numpy()])

    Tn = Xt.shape[1]
    def amp(z):
        z = z.clone()
        f0 = int(rng.integers(0, 40)); z[:, :, f0:f0 + 12] = 0
        t0 = int(rng.integers(0, Tn - 60)); z[:, t0:t0 + 30, :] *= 0.3
        return z

    oof = np.zeros((len(X), 5))
    for fd in range(folds):
        tr = torch.tensor(np.where(vf != fd)[0]).to(DEV)
        va = torch.tensor(np.where(vf == fd)[0]).to(DEV)
        m = Net().to(DEV)
        opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=1e-4)
        lsc = nn.CrossEntropyLoss(weight=wts)
        Ytr = Yt[tr]
        w = torch.tensor(np.bincount(Ytr.cpu(), minlength=5).astype(np.float32) + 1.)
        lsc = nn.CrossEntropyLoss(weight=(w.sum() / (5 * w)).to(DEV))
        for ep in range(80):
            m.train()
            perm = tr[torch.randperm(len(tr), device=DEV)]
            for i in range(0, len(perm), 32):
                b = perm[i:i + 32]
                opt.zero_grad()
                loss = lsc(m(amp(Xt[b])), Yt[b])
                loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                oof[vf == fd] = torch.softmax(m(Xt[va]), 1).cpu().numpy()
    pred = oof.argmax(1)
    acc = (pred == Y).mean()
    print(f'\nCV (window-level) acc {acc:.3f}')
    from collections import defaultdict
    # per-class P/R/F1
    for ci, cname in enumerate(CLASSES):
        tp = ((pred == ci) & (Y == ci)).sum(); fp = ((pred == ci) & (Y != ci)).sum(); fn = ((pred != ci) & (Y == ci)).sum()
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        print(f'  {cname:2s}: P {p:.2f} R {r:.2f} n {int((Y==ci).sum())}')
    # the "window-level" metric is optimistic (views of one clip together) — count by clips:
    clip_pred = np.zeros(k, dtype=int); clip_true = np.zeros(k, dtype=int)
    for i in range(k):
        vote = oof[G == i].mean(0).argmax(); clip_pred[i] = vote; clip_true[i] = Y[np.where(G == i)[0][0]]
    cacc = (clip_pred == clip_true).mean()
    print(f'CV (per clip, window voting) acc {cacc:.3f}')
    conf = np.zeros((5, 5), int)
    for a, b in zip(clip_true, clip_pred): conf[a, b] += 1
    print('confusion (row=truth):'); print('     ' + ' '.join(f'{c:>3}' for c in CLASSES))
    for i, c in enumerate(CLASSES): print(f'  {c:2s} ' + ' '.join(f'{v:3d}' for v in conf[i]))
    sirena_true = np.isin(clip_true, [CLS['S'], CLS['WS']]); sirena_pred = np.isin(clip_pred, [CLS['S'], CLS['WS']])
    tp = (sirena_true & sirena_pred).sum(); fp = (~sirena_true & sirena_pred).sum(); fn = (sirena_true & ~sirena_pred).sum()
    print(f'binary "siren": P {tp/max(tp+fp,1):.2f} R {tp/max(tp+fn,1):.2f} (n={int(sirena_true.sum())})')

    # ---------- final model on all data + sliding over the match ----------
    m = Net().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=1e-4)
    w = torch.tensor(np.bincount(Y, minlength=5).astype(np.float32) + 1.)
    lsc = nn.CrossEntropyLoss(weight=(w.sum() / (5 * w)).to(DEV))
    for ep in range(80):
        m.train()
        for i in range(0, len(Xt), 32):
            b = torch.arange(i, min(i + 32, len(Xt))).to(DEV)
            opt.zero_grad(); lsc(m(amp(Xt[b])), Yt[b]).backward(); opt.step()
    torch.save({'net': m.state_dict(), 'mu': mu, 'scale': sd, 'classes': CLASSES}, '/work/model_audio_v1.pt')

    print('\nsliding over the match...')
    Lslide = 8.0
    pts = np.arange(0, dur - Lslide, 2.0)
    P = np.zeros((len(pts), 5))
    m.eval()
    with torch.no_grad():
        for i in range(0, len(pts), 64):
            chunk = pts[i:i + 64]
            xb = np.stack([np.clip((feats(x, a, a + Lslide) - mu) / sd, -8, 8) for a in chunk])
            P[i:i + len(chunk)] = torch.softmax(m(torch.tensor(xb, device=DEV)), 1).cpu().numpy()
    np.savez_compressed('/work/audio_probs.npz', t=pts, P=P, classes=np.array(CLASSES))
    # peaks: P(class)>0.7, local maximum, at least 6 s apart
    peaks = []
    for ci in (1, 2, 3, 4):
        col = P[:, ci]
        for j in np.where(col > 0.7)[0]:
            if col[j] == col[max(0, j - 3):j + 4].max():
                if not peaks or pts[j] - peaks[-1]['t'] > 6:
                    peaks.append({'t': round(float(pts[j]), 1), 'class': CLASSES[ci], 'p': round(float(col[j]), 2)})
    peaks.sort(key=lambda d: d['t'])
    json.dump({'events': peaks}, open('/work/audio_timeline.json', 'w'))
    print('peaks:', len(peaks))
    for p_ in peaks[:40]: print('  ', p_)

if __name__ == '__main__':
    torch.manual_seed(0); np.random.seed(0)
    main()
