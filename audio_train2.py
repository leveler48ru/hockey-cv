#!/usr/bin/env python3
"""audio v2: 4 classes (N, W, S∪WS, M), 10 s windows ×5 views, aug, OOF threshold, sliding over the match."""
import json, wave
import numpy as np
import torch
import torch.nn as nn

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SR, FR, HO, NM = 16000, 1024, 256, 64
CLS = ['N', 'W', 'S', 'M']
CI = {c: i for i, c in enumerate(CLS)}
MERGE = {'N': 'N', 'W': 'W', 'S': 'S', 'WS': 'S', 'M': 'M'}


def load_wav(p):
    wf = wave.open(p, 'rb')
    x = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768
    assert wf.getframerate() == SR
    return x


def melbank():
    def h2m(f): return 2595 * np.log10(1 + f / 700)
    def m2h(m): return 700 * (10 ** (m / 2595) - 1)
    ms = np.linspace(h2m(20), h2m(8000), NM + 2)
    bins = np.floor((FR + 1) * m2h(ms) / SR).astype(int)
    W = np.zeros((NM, FR // 2 + 1), np.float32)
    for i in range(NM):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c > l:
            W[i, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            W[i, c:r] = (r - np.arange(c, r)) / (r - c)
    return W


MELW = melbank()
WINF = np.hanning(FR).astype(np.float32)


def feats(x, a, b):
    s, e = int(a * SR), int(min(b, len(x) / SR) * SR)
    seg = x[s:e]
    if len(seg) < FR:
        seg = np.pad(seg, (0, FR - len(seg)))
    n = 1 + (len(seg) - FR) // HO
    st = np.lib.stride_tricks.as_strided(seg, (n, FR), (seg.strides[0] * HO, seg.strides[0]))
    S = np.abs(np.fft.rfft(st * WINF, axis=1)) ** 2
    return np.log(S @ MELW.T + 1e-8).astype(np.float32)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(nn.Conv1d(ci, co, 7, padding=3), nn.BatchNorm1d(co), nn.ReLU(), nn.MaxPool1d(2))
        self.b = nn.Sequential(blk(NM, 32), blk(32, 64), blk(64, 128), blk(128, 128))
        self.fc = nn.Sequential(nn.Linear(256, 96), nn.ReLU(), nn.Dropout(0.35), nn.Linear(96, 4))

    def forward(self, z):  # (B,T,NM)
        h = self.b(z.transpose(1, 2))
        h = torch.cat([h.mean(-1), h.max(-1).values], 1)
        return self.fc(h)


def main():
    x = load_wav('/work/m1.wav')
    lab0 = json.load(open('/work/audio_labels.json'))
    dur = len(x) / SR
    Lwin, offs = 10.0, (0.0, 0.5, 1.0, 1.5, 2.0)
    xs, ys, gs = [], [], []
    names = []
    for ci_, c in enumerate(lab0):
        y = CI[MERGE[c['label']]]
        ok = False
        for off in offs:
            if c['t0'] + off + Lwin > dur:
                continue
            xs.append(feats(x, c['t0'] + off, c['t0'] + off + Lwin))
            ys.append(y); gs.append(ci_); ok = True
        if ok:
            names.append(c['t0'])
    X = np.stack(xs); Y = np.array(ys); G = np.array(gs)
    ncl = len(set(G.tolist()))
    print(f'views {len(X)} clips {ncl} class counts {np.bincount(Y)}', flush=True)

    mu, sd = float(X.mean()), float(X.std())
    Xn = np.clip((X - mu) / sd, -8, 8)
    Xt = torch.tensor(Xn, device=DEV)
    Yt = torch.tensor(Y, dtype=torch.long, device=DEV)
    Tn = Xt.shape[1]
    rng = np.random.default_rng(11)

    def amp(z):
        z = z.clone()
        f0 = int(rng.integers(0, Tn and 50))
        z[:, :, f0:f0 + 14] = 0
        t0 = int(rng.integers(0, Tn - 120))
        z[:, t0:t0 + 50, :] *= 0.2
        z += torch.randn_like(z) * 0.05
        return z

    cw = torch.tensor((len(Y) / (4 * (np.bincount(Y, minlength=4) + 1.0))), dtype=torch.float32, device=DEV)
    lsc = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    def train(fidx, ep):
        opt = torch.optim.AdamW(fidx.parameters(), 3e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
        B = 16
        N_ = len(Xt)
        for e in range(ep):
            perm = torch.randperm(N_, device=DEV)
            for i in range(0, N_, B):
                b = perm[i:i + B]
                opt.zero_grad()
                lsc(fidx(amp(Xt[b])), Yt[b]).backward()
                opt.step()
            sch.step()
        return fidx

    folds = 5
    ucl = np.unique(G)
    fold_of = dict(zip(rng.permutation(ucl), np.arange(len(ucl)) % folds))
    vf = np.array([fold_of[g] for g in G])
    oof = np.zeros((len(X), 4))
    for fd in range(folds):
        tr = torch.tensor(np.where(vf != fd)[0], device=DEV)
        va = torch.tensor(np.where(vf == fd)[0], device=DEV)
        # class weights computed on the train fold
        cwt = torch.tensor((len(tr) / (4 * (np.bincount(Y[vf != fd], minlength=4) + 1.0))), dtype=torch.float32, device=DEV)
        lsc = nn.CrossEntropyLoss(weight=cwt, label_smoothing=0.1)
        m = Net().to(DEV)
        opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 160)
        for e in range(160):
            m.train()
            perm = tr[torch.randperm(len(tr), device=DEV)]
            for i in range(0, len(perm), 16):
                b = perm[i:i + 16]
                opt.zero_grad(); lsc(m(amp(Xt[b])), Yt[b]).backward(); opt.step()
            sch.step()
        m.eval()
        with torch.no_grad():
            vai = va.cpu().numpy()
            for i in range(0, len(vai), 64):
                bb = torch.tensor(vai[i:i + 64], device=DEV)
                oof[vai[i:i + 64]] = torch.softmax(m(Xt[bb]), 1).cpu().numpy()
    pred = oof.argmax(1)
    acc = (pred == Y).mean()
    print(f'\nCV window-level acc {acc:.3f}')
    # per clip (window majority)
    clip_i = []
    for g in ucl:
        clip_i.append(oof[G == g].mean(0).argmax())
    clip_true = np.array([Y[np.where(G == g)[0][0]] for g in ucl])
    clip_i = np.array(clip_i)
    print(f'CV per-clip acc {(clip_i == clip_true).mean():.3f}')
    rep = {'clip_acc': float((clip_i == clip_true).mean())}
    for ci_, cn in enumerate(CLS):
        tp = int(((clip_i == ci_) & (clip_true == ci_)).sum())
        fp = int(((clip_i == ci_) & (clip_true != ci_)).sum())
        fn = int(((clip_i != ci_) & (clip_true == ci_)).sum())
        p_ = tp / max(tp + fp, 1); r_ = tp / max(tp + fn, 1)
        f_ = 2 * p_ * r_ / max(p_ + r_, 1e-9)
        print(f'  {cn:2s} P {p_:.2f} R {r_:.2f} F1 {f_:.2f} n {int((clip_true == ci_).sum())}')
        rep.setdefault('per_class', {})[cn] = dict(p=round(p_, 2), r=round(r_, 2), f1=round(f_, 2), n=int((clip_true == ci_).sum()))
    conf = np.zeros((4, 4), int)
    for a, b in zip(clip_true, clip_i): conf[a, b] += 1
    rep['confusion'] = conf.tolist()
    print('confusion (row=truth):'); print(conf)
    # "goal" = S: threshold picked on OOF (clip window opinion = mean)
    ps = np.array([oof[G == g][:, CI['S']].mean() for g in ucl])
    best = (0, 0.5)
    for th in np.arange(0.3, 0.92, 0.02):
        tp = int(((ps >= th) & (clip_true == CI['S'])).sum())
        fp = int(((ps >= th) & (clip_true != CI['S'])).sum())
        fn = int(((ps < th) & (clip_true == CI['S'])).sum())
        p_ = tp / max(tp + fp, 1); r_ = tp / max(tp + fn, 1)
        f_ = 2 * p_ * r_ / max(p_ + r_, 1e-9)
        if f_ > best[0]:
            best = (f_, float(th), p_, r_)
    print(f'GOAL (S) threshold {best[1]:.2f} F1 {best[0]:.2f} P {best[2]:.2f} R {best[3]:.2f}')
    rep['goal_threshold'] = {'thr': best[1], 'f1': round(best[0], 3), 'p': round(best[2], 3), 'r': round(best[3], 3)}
    json.dump(rep, open('/work/audio2_report.json', 'w'))

    # final model on all data
    lsc = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
    m = Net().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 200)
    for e in range(200):
        m.train()
        perm = torch.randperm(len(Xt), device=DEV)
        for i in range(0, len(perm), 16):
            b = perm[i:i + 16]
            opt.zero_grad(); lsc(m(amp(Xt[b])), Yt[b]).backward(); opt.step()
        sch.step()
    torch.save({'net': m.state_dict(), 'mu': mu, 'scale': sd, 'classes': CLS}, '/work/model_audio_v2.pt')

    # sliding over the whole match
    print('\nslide 10 s / step 1 s...', flush=True)
    pts = np.arange(0, dur - Lwin, 1.0)
    P = np.zeros((len(pts), 4), np.float32)
    m.eval()
    with torch.no_grad():
        for i in range(0, len(pts), 128):
            chunk = pts[i:i + 128]
            xb = np.stack([np.clip((feats(x, a, a + Lwin) - mu) / sd, -8, 8) for a in chunk])
            z = torch.tensor(xb, device=DEV)
            if z.shape[1] != Tn:
                z = nn.functional.interpolate(z.transpose(1, 2), size=Tn).transpose(1, 2)
            P[i:i + len(chunk)] = torch.softmax(m(z), 1).cpu().numpy()
    np.savez_compressed('/work/audio2_probs.npz', t=pts, P=P, classes=np.array(CLS))
    thrS = rep['goal_threshold']['thr']
    events = []
    for ci_, cn in enumerate(CLS):
        if cn == 'N': continue
        col = P[:, ci_]
        th = thrS if cn == 'S' else 0.85
        cand = np.where(col > th)[0]
        last = -99
        for j in cand:
            if col[j] >= col[max(0, j - 2):j + 3].max() and pts[j] - last > 8:
                events.append({'t': float(pts[j]), 'cls': cn, 'p': float(col[j])})
                last = pts[j]
    events.sort(key=lambda d: d['t'])
    json.dump({'thr_goal': thrS, 'events': events}, open('/work/audio2_timeline.json', 'w'))
    from collections import Counter
    print('events:', Counter(e['cls'] for e in events))
    for e in events[:50]:
        print(f"  t={e['t']:7.1f} {e['cls']:2s} p {e['p']:.2f}")


if __name__ == '__main__':
    torch.manual_seed(0)
    main()
