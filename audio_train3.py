#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audio v3: short 3.0 s windows (a ~2 s whistle fits entirely), 0.5 s stride
inside the 12-s clip => ~19 views per clip. MIL approach: the clip label is given to all
windows; evaluation and threshold use the MAX over the clip's windows (not the mean: in a long clip
the event is short). Sliding over the match with a 0.5 s step -> proper whistle localization.

Lessons from v1/v2 applied:
  * torch.save: keys {'net','mu','scale'} — no duplicates (v1/v2 lost the weights);
  * per-class threshold picked on the OOF per-clip maximum;
  * class dictionary CLS, linear layer after cat(mean,max)=2*chan.
"""
import json, wave
import numpy as np
import torch
import torch.nn as nn

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SR, FR, HO, NM = 16000, 1024, 256, 64
CLS = ['N', 'W', 'S', 'M']
CI = {c: i for i, c in enumerate(CLS)}
MERGE = {'N': 'N', 'W': 'W', 'S': 'S', 'WS': 'S', 'M': 'M'}
LW, VS = 3.0, 0.5


def load_wav(p):
    wf = wave.open(p, 'rb')
    assert wf.getframerate() == SR, wf.getframerate()
    x = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768
    return x


def melbank():
    def h2m(f): return 2595 * np.log10(1 + f / 700)
    def m2h(m): return 700 * (10 ** (m / 2595) - 1)
    ms = np.linspace(h2m(20), h2m(8000), NM + 2)
    bins = np.floor((FR + 1) * m2h(ms) / SR).astype(int)
    W = np.zeros((NM, FR // 2 + 1), np.float32)
    for i in range(NM):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c > l: W[i, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c: W[i, c:r] = (r - np.arange(c, r)) / (r - c)
    return W


MELW = melbank()
WINF = np.hanning(FR).astype(np.float32)


def feats(x, a, b):
    s, e = int(a * SR), int(min(b, len(x) / SR) * SR)
    seg = x[s:e]
    if len(seg) < FR: seg = np.pad(seg, (0, FR - len(seg)))
    n = 1 + (len(seg) - FR) // HO
    st = np.lib.stride_tricks.as_strided(seg, (n, FR), (seg.strides[0] * HO, seg.strides[0]))
    S = np.abs(np.fft.rfft(st * WINF, axis=1)) ** 2
    return np.log(S @ MELW.T + 1e-8).astype(np.float32)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(nn.Conv1d(ci, co, 7, padding=3), nn.BatchNorm1d(co),
                                 nn.ReLU(), nn.MaxPool1d(2))
        self.b = nn.Sequential(blk(NM, 32), blk(32, 64), blk(64, 128), blk(128, 128))
        self.fc = nn.Sequential(nn.Linear(256, 96), nn.ReLU(), nn.Dropout(0.35), nn.Linear(96, 4))

    def forward(self, z):  # (B,T,NM)
        h = self.b(z.transpose(1, 2))
        h = torch.cat([h.mean(-1), h.max(-1).values], 1)
        return self.fc(h)


def main():
    x = load_wav('/work/m1.wav')
    lab = json.load(open('/work/audio_labels.json'))
    dur = len(x) / SR
    xs, ys, gs = [], [], []
    for ci_, c in enumerate(lab):
        y = CI[MERGE[c['label']]]
        a = c['t0']
        while a + LW <= min(c['t0'] + 12.0, dur):
            xs.append(feats(x, a, a + LW)); ys.append(y); gs.append(ci_)
            a += VS
    X = np.stack(xs); Y = np.array(ys); G = np.array(gs)
    ucl = np.unique(G)
    print(f'views {len(X)} clips {len(ucl)} class counts {np.bincount(Y)}', flush=True)

    mu, sd = float(X.mean()), float(X.std())
    Xt = torch.tensor(np.clip((X - mu) / sd, -8, 8))
    Yt = torch.tensor(Y, dtype=torch.long)
    Tn = Xt.shape[1]
    Xt, Yt = Xt.to(DEV), Yt.to(DEV)
    Gt = torch.tensor(G, device=DEV)
    rng = np.random.default_rng(11)

    def amp(z):
        z = z.clone()
        f0 = int(rng.integers(0, max(Tn - 40, 1)))
        z[:, :, f0:f0 + 12] = 0
        t0 = int(rng.integers(0, max(z.shape[1] - 30, 1)))
        z[:, t0:t0 + 25, :] *= 0.2
        z += torch.randn_like(z) * 0.05
        return z

    def fit(idx_tr, ep, lsc):
        m = Net().to(DEV)
        opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
        for e in range(ep):
            m.train()
            perm = idx_tr[torch.randperm(len(idx_tr), device=DEV)]
            for i in range(0, len(perm), 32):
                b = perm[i:i + 32]
                opt.zero_grad(); lsc(m(amp(Xt[b])), Yt[b]).backward(); opt.step()
            sch.step()
        return m

    def probs(m, rows):
        m.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(rows), 256):
                out.append(torch.softmax(m(Xt[rows[i:i + 256]]), 1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 4))

    def cvmsew(cnt):
        return torch.tensor(len(Yt) / (4.0 * (cnt + 1.0)), dtype=torch.float32, device=DEV)

    folds = 5
    fold_of = dict(zip(rng.permutation(ucl), np.arange(len(ucl)) % folds))
    view_fold = np.array([fold_of[g] for g in G])
    oof = np.zeros((len(X), 4))
    for fd in range(folds):
        tr = torch.tensor(np.where(view_fold != fd)[0], device=DEV)
        va = np.where(view_fold == fd)[0]
        cwfd = cvmsew(np.bincount(Y[view_fold != fd], minlength=4))
        m = fit(tr, 140, nn.CrossEntropyLoss(weight=cwfd, label_smoothing=0.1))
        rows = torch.tensor(va, device=DEV)
        oof[va] = probs(m, rows)
        print(f'fold {fd} done', flush=True)

    # clip metric = max over windows (MIL)
    def clipmax(P):
        M = np.zeros((len(ucl), 4))
        for k, g in enumerate(ucl):
            M[k] = P[G == g].max(0)
        return M
    Mcv = clipmax(oof)
    true_k = np.array([Y[np.where(G == g)[0][0]] for g in ucl])
    pred_k = Mcv.argmax(1)
    acc = (pred_k == true_k).mean()
    rep = {'views': int(len(X)), 'clips': int(len(ucl)), 'cv_clip_acc_maxwin': round(float(acc), 3)}
    print(f'CV per-clip (max window) acc {acc:.3f}')
    # per-class threshold on OOF: clipmax[:,c] > th
    best_th = {}
    for c_i, c_n in enumerate(CLS):
        if c_n == 'N': continue
        bt = (0, 0.5, 0, 0)
        for th in np.arange(0.25, 0.96, 0.025):
            hit = Mcv[:, c_i] >= th
            tp = int((hit & (true_k == c_i)).sum()); fp = int((hit & (true_k != c_i)).sum())
            fn = int((~hit & (true_k == c_i)).sum())
            p_ = tp / max(tp + fp, 1); r_ = tp / max(tp + fn, 1)
            f_ = 2 * p_ * r_ / max(p_ + r_, 1e-9)
            if f_ > bt[0]: bt = (f_, float(th), p_, r_)
        best_th[c_n] = bt
        print(f'  {c_n}: threshold {bt[1]:.2f} F1 {bt[0]:.2f} P {bt[2]:.2f} R {bt[3]:.2f} n {int((true_k == c_i).sum())}')
        rep.setdefault('per_class', {})[c_n] = dict(thr=bt[1], f1=round(bt[0], 3), p=round(bt[2], 3),
                                                    r=round(bt[3], 3), n=int((true_k == c_i).sum()))
    conf = np.zeros((4, 4), int)
    for a_, b_ in zip(true_k, pred_k): conf[a_, b_] += 1
    rep['confusion'] = conf.tolist(); print(conf)
    json.dump(rep, open('/work/audio3_report.json', 'w'), indent=1)

    # final model on all data
    cw = cvmsew(np.bincount(Y, minlength=4))
    allidx = torch.arange(len(Xt), device=DEV)
    m = fit(allidx, 180, nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1))
    torch.save({'net': m.state_dict(), 'mu': mu, 'scale': sd, 'classes': CLS,
                'win': LW, 'thr': {k: v[1] for k, v in best_th.items()}}, '/work/model_audio_v3.pt')
    ck = torch.load('/work/model_audio_v3.pt', map_location='cpu')
    npar = sum(t.numel() for t in ck['net'].values())
    print(f'checkpoint: state_dict entries {len(ck["net"])}, parameters {npar}', flush=True)

    # slide over the match, 0.5 s step
    print('slide 3.0 s / step 0.5 s...', flush=True)
    pts = np.arange(0.0, dur - LW, VS)
    P = np.zeros((len(pts), 4), np.float32)
    m.eval()
    with torch.no_grad():
        for i in range(0, len(pts), 256):
            ch = pts[i:i + 256]
            xb = np.stack([np.clip((feats(x, a, a + LW) - mu) / sd, -8, 8) for a in ch])
            z = torch.tensor(xb, device=DEV)
            if z.shape[1] != Tn:
                z = nn.functional.interpolate(z.transpose(1, 2), size=Tn).transpose(1, 2)
            P[i:i + len(ch)] = torch.softmax(m(z), 1).cpu().numpy()
    np.savez_compressed('/work/audio3_probs.npz', t=pts, P=P, classes=np.array(CLS))

    GAP = {'W': 3.0, 'S': 6.0, 'M': 8.0}
    events = []
    for c_i, c_n in enumerate(CLS):
        if c_n == 'N' or c_n not in best_th: continue
        col = P[:, c_i]; th = best_th[c_n][1]
        cand = np.where(col > th)[0]; last = -9e9
        for j in cand:
            w = col[max(0, j - 3):j + 4]
            if col[j] >= w.max() and pts[j] - last >= GAP[c_n]:
                events.append({'t': round(float(pts[j]), 1), 'cls': c_n, 'p': round(float(col[j]), 3)})
                last = pts[j]
    events.sort(key=lambda d: d['t'])
    json.dump({'thr': {k: v[1] for k, v in best_th.items()}, 'events': events},
              open('/work/audio3_timeline.json', 'w'))
    from collections import Counter
    print('events:', Counter(e['cls'] for e in events))
    for e in events[:80]: print(f"  t={e['t']:7.1f} {e['cls']} p {e['p']:.2f}")


if __name__ == '__main__':
    torch.manual_seed(0)
    main()
