"""
DeepSRGM proof-of-concept: does a SEQUENCE model (bi-LSTM on the tonic-normalized
pitch contour) beat TDMS's top-1 on the same in-distribution data?

TDMS is a pitch-DISTRIBUTION model and confuses allied ragas (same scale, different
phrases). A sequence model can learn phrase/gamaka order, which is the actual
discriminator. This POC trains on the expert .pitchSilIntrpPP contours (clean, no
extraction needed) with a recording-aware split. If it clearly beats TDMS (75%
top-1 in-distribution), it's worth retraining on Essentia pitch for deployment.

Stage 1 (cache): tokenize every recording's contour -> data/deepsrgm_seqs.npz
Stage 2 (train): recording-aware split, bi-LSTM, per-recording top-1/top-5.
"""
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
DATASET = os.path.join(DATA, "RagaDataset", "Carnatic")
CACHE = os.path.join(DATA, "deepsrgm_seqs.npz")

# Tokenization: tonic-normalized cents, 50-cent bins (2 per semitone -> keeps some
# gamaka detail), over a 3-octave range [-1200, +2400] cents. Silence dropped.
BIN = 50.0
LO, HI = -1200, 2400
VOCAB = int((HI - LO) / BIN) + 1   # 73
DS = 10                            # downsample voiced contour 225Hz -> ~22.5Hz


def tokenize(pitches, tonic):
    voiced = pitches[pitches > 0]
    if len(voiced) < 200:
        return None
    cents = 1200.0 * np.log2(voiced / tonic)
    idx = np.round((np.clip(cents, LO, HI) - LO) / BIN).astype(np.int64)
    return idx[::DS]


def build_cache():
    mapping = json.load(open(os.path.join(DATASET, "_info_", "ragaId_to_ragaName_mapping.json")))
    classes = json.load(open(os.path.join(DATA, "classes.json")))
    name_to_label = {n: i for i, n in enumerate(classes)}
    features_dir = os.path.join(DATASET, "features")
    seqs, labels, recids = [], [], []
    for raga_id in sorted(os.listdir(features_dir)):
        if raga_id not in mapping:
            continue
        raga_name = mapping[raga_id]
        label = name_to_label.get(raga_name)
        if label is None:
            continue
        for root, _, files in os.walk(os.path.join(features_dir, raga_id)):
            pf = next((f for f in files if f.endswith(".pitchSilIntrpPP")), None)
            tf = next((f for f in files if f.endswith(".tonicFine")), None)
            if not pf or not tf:
                continue
            try:
                arr = np.loadtxt(os.path.join(root, pf))
                tonic = float(open(os.path.join(root, tf)).read().strip())
                toks = tokenize(arr[:, 1], tonic)
            except Exception as exc:
                print(f"  skip {pf}: {exc!r}"); continue
            if toks is None:
                continue
            seqs.append(toks.astype(np.int16)); labels.append(label); recids.append(len(recids))
            print(f"  [{raga_name}] {len(toks)} tokens (total {len(seqs)})", flush=True)
    np.savez_compressed(CACHE, **{f"s{i}": s for i, s in enumerate(seqs)},
                        labels=np.array(labels), n=len(seqs))
    print(f"cached {len(seqs)} recordings -> {CACHE}")


def load_cache():
    z = np.load(CACHE, allow_pickle=True)
    n = int(z["n"]); labels = z["labels"]
    seqs = [z[f"s{i}"].astype(np.int64) for i in range(n)]
    return seqs, labels


def train_eval(seed=0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    classes = json.load(open(os.path.join(DATA, "classes.json")))
    seqs, labels = load_cache()
    n = len(seqs)
    print(f"[seed {seed}] {n} recordings, {len(set(labels.tolist()))} ragas, device={dev}")

    # recording-aware split: 80/20 per raga
    rng = np.random.default_rng(seed)
    idx_by_cls = {}
    for i, l in enumerate(labels):
        idx_by_cls.setdefault(int(l), []).append(i)
    train_ids, test_ids = [], []
    for l, ids in idx_by_cls.items():
        rng.shuffle(ids); k = max(1, int(round(len(ids) * 0.2)))
        test_ids += ids[:k]; train_ids += ids[k:]
    print(f"train recs {len(train_ids)}, test recs {len(test_ids)}")

    SEQ_LEN, PER_REC = 800, 60

    def subseqs(rec_ids, per):
        xs, ys = [], []
        for ri in rec_ids:
            s = seqs[ri]
            for _ in range(per):
                if len(s) <= SEQ_LEN:
                    sub = np.pad(s, (0, SEQ_LEN - len(s)), constant_values=0)
                else:
                    st = rng.integers(0, len(s) - SEQ_LEN); sub = s[st:st + SEQ_LEN]
                xs.append(sub); ys.append(int(labels[ri]))
        return np.array(xs), np.array(ys)

    Xtr, ytr = subseqs(train_ids, PER_REC)
    print(f"train subsequences: {Xtr.shape}")

    class Net(nn.Module):
        def __init__(self, vocab, ncls):
            super().__init__()
            self.emb = nn.Embedding(vocab, 96)
            # single layer: PyTorch's MPS multi-layer LSTM is pathologically slow
            self.lstm = nn.LSTM(96, 192, batch_first=True, bidirectional=True)
            self.attn = nn.Linear(384, 1)   # attention pooling over time
            self.fc = nn.Sequential(nn.Linear(384, 192), nn.ReLU(), nn.Dropout(0.4),
                                    nn.Linear(192, ncls))
        def forward(self, x):
            e = self.emb(x); o, _ = self.lstm(e)
            a = torch.softmax(self.attn(o).squeeze(-1), dim=1)
            pooled = (o * a.unsqueeze(-1)).sum(dim=1)
            return self.fc(pooled)

    EPOCHS = 30
    net = Net(VOCAB, len(classes)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr, device=dev); ytr_t = torch.tensor(ytr, device=dev)
    bs = 256
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(len(Xtr_t), device=dev); tot = 0.0
        for b in range(0, len(perm), bs):
            bi = perm[b:b + bs]; opt.zero_grad()
            out = net(Xtr_t[bi]); loss = lossf(out, ytr_t[bi]); loss.backward(); opt.step()
            tot += loss.item() * len(bi)
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1}/{EPOCHS} loss {tot/len(Xtr_t):.3f}", flush=True)

    # eval: per test recording, average softmax over its subsequences
    net.eval(); t1 = t5 = 0
    with torch.no_grad():
        for ri in test_ids:
            Xte, _ = subseqs([ri], 20)
            out = net(torch.tensor(Xte, device=dev)).softmax(1).mean(0).cpu().numpy()
            top5 = np.argsort(out)[::-1][:5]; true = int(labels[ri])
            t1 += top5[0] == true; t5 += true in top5
    m = len(test_ids)
    print(f"[seed {seed}] top-1 {t1/m*100:5.1f}%   top-5 {t5/m*100:5.1f}%   (n={m})", flush=True)
    return t1 / m * 100, t5 / m * 100


if __name__ == "__main__":
    if not os.path.exists(CACHE):
        print("building token cache (slow, once)...")
        build_cache()
    t1s, t5s = [], []
    for s in (0, 1, 2):
        a, b = train_eval(seed=s)
        t1s.append(a); t5s.append(b)
    print("\n=== DeepSRGM POC — 3 recording-aware splits (expert pitch, in-distribution) ===")
    print(f"  top-1 {np.mean(t1s):5.1f}% +/- {np.std(t1s):.1f}   top-5 {np.mean(t5s):5.1f}% +/- {np.std(t5s):.1f}")
    print(f"  per-seed top-1: {[round(x,1) for x in t1s]}")
    print(f"  (compare TDMS in-distribution: 75% top-1 / 91% top-5)")
