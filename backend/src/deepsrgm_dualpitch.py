"""
Last data-side lever (cheap, no new downloads): train DeepSRGM on BOTH pitch
versions of the CompMusic recordings - the expert .pitch contours (clean) and the
Essentia contours (deployable) - and see if that clean-pitch exposure improves
NOVEL-audio generalization vs training on Essentia contours alone.

Both caches already exist:
  data/deepsrgm_seqs.npz          (expert .pitch, ~22.5 Hz tokens)
  data/deepsrgm_essentia_seqs.npz (Essentia Melodia, ~22.5 Hz tokens)
Eval on held-out YouTube recordings (recording-aware), same as the augmentation tests.
"""
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
BIN, LO, HI = 50.0, -1200, 2400
VOCAB = int((HI - LO) / BIN) + 1
SEQ_LEN, PER_REC, EPOCHS = 800, 50, 25
SEEDS = (0, 1, 2)


def load(npz):
    z = np.load(npz, allow_pickle=True); n = int(z["n"])
    return [z[f"s{i}"].astype(np.int64) for i in range(n)], np.array(z["labels"])


def subseqs(seqs, ids, labels, per, rng):
    xs, ys = [], []
    for ri in ids:
        s = seqs[ri]
        for _ in range(per):
            if len(s) <= SEQ_LEN:
                xs.append(np.pad(s, (0, SEQ_LEN - len(s))))
            else:
                st = rng.integers(0, len(s) - SEQ_LEN); xs.append(s[st:st + SEQ_LEN])
            ys.append(int(labels[ri]))
    return np.array(xs), np.array(ys)


def main():
    import torch
    import torch.nn as nn
    classes = json.load(open(os.path.join(DATA, "classes.json")))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    ess_s, ess_y = load(os.path.join(DATA, "deepsrgm_essentia_seqs.npz"))
    exp_s, exp_y = load(os.path.join(DATA, "deepsrgm_seqs.npz"))
    yt_s, yt_y = load(os.path.join(DATA, "deepsrgm_youtube_seqs.npz"))
    print(f"Essentia-CMD {len(ess_s)}, expert-CMD {len(exp_s)}, YouTube {len(yt_s)}, dev={dev}", flush=True)

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.emb = nn.Embedding(VOCAB, 96); s.lstm = nn.LSTM(96, 192, batch_first=True, bidirectional=True)
            s.attn = nn.Linear(384, 1)
            s.fc = nn.Sequential(nn.Linear(384, 192), nn.ReLU(), nn.Dropout(0.4), nn.Linear(192, len(classes)))
        def forward(s, x):
            o, _ = s.lstm(s.emb(x)); a = torch.softmax(s.attn(o).squeeze(-1), dim=1)
            return s.fc((o * a.unsqueeze(-1)).sum(1))

    def train(seqs, labels, seed):
        torch.manual_seed(seed); rng = np.random.default_rng(1000 + seed)
        Xtr, ytr = subseqs(seqs, list(range(len(seqs))), labels, PER_REC, rng)
        net = Net().to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS); lossf = nn.CrossEntropyLoss()
        Xt = torch.tensor(Xtr); yt = torch.tensor(ytr); bs = 128
        if dev == "mps":
            torch.mps.empty_cache()
        for ep in range(EPOCHS):
            net.train(); perm = torch.randperm(len(Xt))
            for b in range(0, len(perm), bs):
                bi = perm[b:b + bs]; opt.zero_grad()
                loss = lossf(net(Xt[bi].to(dev)), yt[bi].to(dev)); loss.backward(); opt.step()
            sched.step()
        return net

    def evalrecs(net, test_ids, rng):
        net.eval(); t1 = t5 = 0
        with torch.no_grad():
            for ri in test_ids:
                Xte, _ = subseqs(yt_s, [ri], yt_y, 20, rng)
                p = net(torch.tensor(Xte).to(dev)).softmax(1).mean(0).cpu().numpy()
                top5 = np.argsort(p)[::-1][:5]
                t1 += top5[0] == int(yt_y[ri]); t5 += int(yt_y[ri]) in top5
        m = len(test_ids); return t1 / m * 100, t5 / m * 100

    A1s, A5s, B1s, B5s = [], [], [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        by_cls = {}
        for i, l in enumerate(yt_y):
            by_cls.setdefault(int(l), []).append(i)
        yt_test = []
        for l, ids in by_cls.items():
            ids = ids[:]; rng.shuffle(ids); yt_test += ids[:max(1, int(round(len(ids) * 0.3)))]

        netA = train(ess_s, ess_y, seed)                                   # Essentia-only
        a1, a5 = evalrecs(netA, yt_test, rng)
        netB = train(ess_s + exp_s, np.concatenate([ess_y, exp_y]), seed)  # Essentia + expert
        b1, b5 = evalrecs(netB, yt_test, rng)
        A1s.append(a1); A5s.append(a5); B1s.append(b1); B5s.append(b5)
        print(f"  [seed {seed}] n_test={len(yt_test)}  A(Essentia-only) {a1:.1f}/{a5:.1f}  B(Ess+expert) {b1:.1f}/{b5:.1f}", flush=True)

    print("\n=== Dual-pitch training (Essentia + expert CMD contours), eval on novel YouTube ===")
    print(f"  A  Essentia-only:   top-1 {np.mean(A1s):5.1f} +/- {np.std(A1s):.1f}   top-5 {np.mean(A5s):5.1f}")
    print(f"  B  Ess + expert:    top-1 {np.mean(B1s):5.1f} +/- {np.std(B1s):.1f}   top-5 {np.mean(B5s):5.1f}")
    print(f"  delta top-1: {np.mean(B1s)-np.mean(A1s):+.1f}   delta top-5: {np.mean(B5s)-np.mean(A5s):+.1f}")


if __name__ == "__main__":
    main()
