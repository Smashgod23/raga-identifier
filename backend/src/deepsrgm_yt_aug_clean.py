"""
Rescue: raw YouTube augmentation was ~neutral because the auto-searched labels are
noisy. Filter YT-train to recordings the CompMusic-only model CONFIRMS (its top-K
prediction contains the searched label), then augment with only those. This keeps
real concert-audio diversity while dropping mislabeled poison.

Efficient: train the CompMusic-only base model ONCE (used for both the confidence
filter and arm A), then per seed train arm B on CMD + cleaned-YT-train. Eval both
on the same held-out YT-test (recording-aware).
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
KEEP_TOPK = 3   # keep a YT recording if the base model ranks its label in top-K


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
    cmd_s, cmd_y = load(os.path.join(DATA, "deepsrgm_essentia_seqs.npz"))
    yt_s, yt_y = load(os.path.join(DATA, "deepsrgm_youtube_seqs.npz"))

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.emb = nn.Embedding(VOCAB, 96); s.lstm = nn.LSTM(96, 192, batch_first=True, bidirectional=True)
            s.attn = nn.Linear(384, 1)
            s.fc = nn.Sequential(nn.Linear(384, 192), nn.ReLU(), nn.Dropout(0.4), nn.Linear(192, len(classes)))
        def forward(s, x):
            o, _ = s.lstm(s.emb(x)); a = torch.softmax(s.attn(o).squeeze(-1), dim=1)
            return s.fc((o * a.unsqueeze(-1)).sum(1))

    def train(seqs, labels, ids, seed):
        torch.manual_seed(seed); rng = np.random.default_rng(1000 + seed)
        Xtr, ytr = subseqs(seqs, ids, labels, PER_REC, rng)
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

    def predict_topk(net, seqs, ri, rng, k=5):
        net.eval()
        with torch.no_grad():
            Xte, _ = subseqs(seqs, [ri], np.zeros(len(seqs) + 1), 20, rng)
            p = net(torch.tensor(Xte).to(dev)).softmax(1).mean(0).cpu().numpy()
        return np.argsort(p)[::-1][:k]

    def evalrecs(net, seqs, labels, ids, rng):
        t1 = t5 = 0
        for ri in ids:
            top5 = predict_topk(net, seqs, ri, rng, 5)
            t1 += top5[0] == int(labels[ri]); t5 += int(labels[ri]) in top5
        m = len(ids); return t1 / m * 100, t5 / m * 100

    print(f"CMD {len(cmd_s)}, YouTube {len(yt_s)}, device={dev}. Training CMD-only base...", flush=True)
    base = train(cmd_s, cmd_y, list(range(len(cmd_s))), seed=0)

    A1s, A5s, B1s, B5s, kept = [], [], [], [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        by_cls = {}
        for i, l in enumerate(yt_y):
            by_cls.setdefault(int(l), []).append(i)
        yt_train, yt_test = [], []
        for l, ids in by_cls.items():
            ids = ids[:]; rng.shuffle(ids); k = max(1, int(round(len(ids) * 0.3)))
            yt_test += ids[:k]; yt_train += ids[k:]

        # confidence filter: keep YT-train recordings whose label is in base top-K
        clean = [ri for ri in yt_train
                 if int(yt_y[ri]) in predict_topk(base, yt_s, ri, rng, KEEP_TOPK)]
        kept.append((len(clean), len(yt_train)))

        a1, a5 = evalrecs(base, yt_s, yt_y, yt_test, rng)
        comb_s = cmd_s + [yt_s[i] for i in clean]
        comb_y = np.concatenate([cmd_y, yt_y[clean]])
        netB = train(comb_s, comb_y, list(range(len(comb_s))), seed)
        b1, b5 = evalrecs(netB, yt_s, yt_y, yt_test, rng)
        A1s.append(a1); A5s.append(a5); B1s.append(b1); B5s.append(b5)
        print(f"  [seed {seed}] kept {len(clean)}/{len(yt_train)} YT-train, n_test={len(yt_test)}  "
              f"A {a1:.1f}/{a5:.1f}  B(clean) {b1:.1f}/{b5:.1f}", flush=True)

    print(f"\n=== Cleaned YouTube augmentation (keep label in base top-{KEEP_TOPK}) ===")
    print(f"  kept {np.mean([k[0] for k in kept]):.0f}/{np.mean([k[1] for k in kept]):.0f} YT-train on avg")
    print(f"  A  CMD-only:          top-1 {np.mean(A1s):5.1f} +/- {np.std(A1s):.1f}   top-5 {np.mean(A5s):5.1f}")
    print(f"  B  CMD+cleaned YT:    top-1 {np.mean(B1s):5.1f} +/- {np.std(B1s):.1f}   top-5 {np.mean(B5s):5.1f}")
    print(f"  delta top-1: {np.mean(B1s)-np.mean(A1s):+.1f}   delta top-5: {np.mean(B5s)-np.mean(A5s):+.1f}")


if __name__ == "__main__":
    main()
