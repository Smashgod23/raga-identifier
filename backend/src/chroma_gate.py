"""
Phase 16 gate: does a tonic-normalized chromagram carry raga information?

This is deliberately a controlled swap, not a new idea competing on architecture.
Everything is copied from deepsrgm_poc.py — the recording-aware 80/20 split per
raga, SEQ_LEN 800, 60 training windows per recording, the single-layer bi-LSTM
with attention pooling, 30 epochs of Adam with cosine decay, and the
average-softmax-over-20-windows evaluation. The only thing that changes is the
input:

    poc:    one integer pitch token per frame  -> nn.Embedding(73, 96)
    here:   a 61-dim chroma+energy vector      -> nn.Linear(61, 96)

So the number this prints is directly comparable to the poc's 89.2% +/- 1.2
in-distribution top-1, and the difference is attributable to the representation.

The gate: if chroma lands anywhere near the token baseline, the representation
carries real raga information and it is worth spending the YouTube re-download
to measure it on the honest novel set. If it collapses toward chance the way the
Phase 4 mel-spectrogram CNN did (11.46%), chroma goes on the ruled-out list and
we stop.

Inputs are the .npz files written by preprocess_chroma.py. Tonic normalization
happens here at load time, as a circular roll of the 60-bin axis so that bin 0
is always Sa.

Run from backend/:
  source venv/bin/activate
  python src/chroma_gate.py
"""

import json
import os
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
CHROMA_DIR = os.path.join(DATA, "chroma")
REPORT = os.path.join(DATA, "chroma_gate_report.txt")

N_CHROMA = 60
BINS_PER_OCTAVE = 60
FMIN_HZ = 65.40639132514966          # C2, matches preprocess_chroma.FMIN_NOTE
CENTS_PER_BIN = 1200.0 / BINS_PER_OCTAVE

SEQ_LEN = 800                         # ~35 s at 22.73 fps
PER_REC = 60                          # training windows per recording
EVAL_WINDOWS = 20
EPOCHS = 30
BATCH = 256
SEEDS = (0, 1, 2)


def tonic_bin(tonic_hz):
    """Which chroma bin the tonic falls in (circular, 0..59)."""
    return int(round(1200.0 * np.log2(tonic_hz / FMIN_HZ) / CENTS_PER_BIN)) % N_CHROMA


def load_recordings():
    """Load every chroma npz, tonic-normalize, return (features, labels).

    Each feature array is (T, 61) float16: 60 tonic-rolled chroma bins plus a
    per-recording z-scored log-energy channel. Energy is z-scored per recording
    on purpose — absolute loudness is a recording fingerprint, and Phase 4
    already showed what happens when the model can memorize those.
    """
    classes = json.load(open(os.path.join(DATA, "classes.json")))
    name_to_label = {n: i for i, n in enumerate(classes)}

    paths = []
    for root, _, files in os.walk(CHROMA_DIR):
        for f in sorted(files):
            if f.endswith(".npz"):
                paths.append(os.path.join(root, f))
    paths.sort()

    feats, labels, skipped = [], [], 0
    for p in paths:
        z = np.load(p, allow_pickle=True)
        raga = str(z["raga"])
        if raga not in name_to_label:
            skipped += 1
            continue
        ch = z["chroma"].astype(np.float32)          # (T, 60), L1-normalized
        en = z["energy"].astype(np.float32)          # (T,)
        if len(ch) < SEQ_LEN // 4:
            skipped += 1
            continue

        ch = np.roll(ch, -tonic_bin(float(z["tonic"])), axis=1)

        sd = en.std()
        en = (en - en.mean()) / (sd if sd > 1e-6 else 1.0)

        feats.append(np.concatenate([ch, en[:, None]], axis=1).astype(np.float16))
        labels.append(name_to_label[raga])

    if skipped:
        print(f"  skipped {skipped} recordings (unknown raga or too short)")
    return feats, np.array(labels)


def window_index(rec_ids, per, feats, rng):
    """Fixed (recording, start) pairs, mirroring the poc's fixed subsequences."""
    pairs = []
    for ri in rec_ids:
        T = len(feats[ri])
        for _ in range(per):
            st = 0 if T <= SEQ_LEN else int(rng.integers(0, T - SEQ_LEN))
            pairs.append((ri, st))
    return pairs


def gather(pairs, feats):
    """Materialize a batch. Short recordings are zero-padded, which for a
    chroma frame is a genuine 'no energy here' vector rather than a fake pitch
    (the token pipeline needed a dedicated PAD id to get this right)."""
    out = np.zeros((len(pairs), SEQ_LEN, N_CHROMA + 1), dtype=np.float32)
    for k, (ri, st) in enumerate(pairs):
        w = feats[ri][st:st + SEQ_LEN]
        out[k, :len(w)] = w
    return out


def train_eval(feats, labels, seed):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    classes = json.load(open(os.path.join(DATA, "classes.json")))
    print(f"[seed {seed}] {len(feats)} recordings, "
          f"{len(set(labels.tolist()))} ragas, device={dev}", flush=True)

    rng = np.random.default_rng(seed)
    idx_by_cls = {}
    for i, l in enumerate(labels):
        idx_by_cls.setdefault(int(l), []).append(i)
    train_ids, test_ids = [], []
    for _, ids in idx_by_cls.items():
        rng.shuffle(ids)
        k = max(1, int(round(len(ids) * 0.2)))
        test_ids += ids[:k]
        train_ids += ids[k:]
    print(f"  train recs {len(train_ids)}, test recs {len(test_ids)}", flush=True)

    train_pairs = window_index(train_ids, PER_REC, feats, rng)
    ytr = np.array([labels[ri] for ri, _ in train_pairs])
    print(f"  train windows: {len(train_pairs)} x {SEQ_LEN} x {N_CHROMA + 1}",
          flush=True)

    class Net(nn.Module):
        """Identical to deepsrgm_poc.Net except the input projection."""
        def __init__(self, ncls):
            super().__init__()
            self.proj = nn.Linear(N_CHROMA + 1, 96)
            self.lstm = nn.LSTM(96, 192, batch_first=True, bidirectional=True)
            self.attn = nn.Linear(384, 1)
            self.fc = nn.Sequential(nn.Linear(384, 192), nn.ReLU(), nn.Dropout(0.4),
                                    nn.Linear(192, ncls))

        def forward(self, x):
            o, _ = self.lstm(torch.relu(self.proj(x)))
            a = torch.softmax(self.attn(o).squeeze(-1), dim=1)
            return self.fc((o * a.unsqueeze(-1)).sum(dim=1))

    net = Net(len(classes)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.CrossEntropyLoss()
    ytr_t = torch.tensor(ytr, dtype=torch.long)

    t0 = time.time()
    for ep in range(EPOCHS):
        net.train()
        perm = np.random.permutation(len(train_pairs))
        tot = 0.0
        for b in range(0, len(perm), BATCH):
            bi = perm[b:b + BATCH]
            # Windows are gathered per batch and moved to the device one batch at
            # a time. Holding the full window tensor on MPS blows memory.
            xb = torch.tensor(gather([train_pairs[i] for i in bi], feats),
                              device=dev)
            yb = ytr_t[bi].to(dev)
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(bi)
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"    epoch {ep+1}/{EPOCHS} loss {tot/len(train_pairs):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    net.eval()
    t1 = t5 = 0
    with torch.no_grad():
        for ri in test_ids:
            pairs = window_index([ri], EVAL_WINDOWS, feats, rng)
            xb = torch.tensor(gather(pairs, feats), device=dev)
            p = net(xb).softmax(1).mean(0).cpu().numpy()
            top5 = np.argsort(p)[::-1][:5]
            true = int(labels[ri])
            t1 += top5[0] == true
            t5 += true in top5
    m = len(test_ids)
    print(f"[seed {seed}] top-1 {t1/m*100:5.1f}%   top-5 {t5/m*100:5.1f}%   "
          f"(n={m})", flush=True)
    return t1 / m * 100, t5 / m * 100


def main():
    print(f"[{time.strftime('%H:%M:%S')}] loading chroma cache...", flush=True)
    feats, labels = load_recordings()
    mb = sum(f.nbytes for f in feats) / (1024 ** 2)
    print(f"  {len(feats)} recordings in memory, {mb:.0f} MB", flush=True)

    t1s, t5s = [], []
    for s in SEEDS:
        a, b = train_eval(feats, labels, s)
        t1s.append(a)
        t5s.append(b)

    lines = [
        "Phase 16 chroma gate (in-distribution, CMD 480)",
        "Controlled swap of deepsrgm_poc.py: same split, same windows, same",
        "architecture, same schedule. Only the input representation differs.",
        "",
        f"seeds {SEEDS}",
        f"top-1 per seed: {[round(x, 1) for x in t1s]}",
        f"top-5 per seed: {[round(x, 1) for x in t5s]}",
        f"top-1 mean {np.mean(t1s):.1f} +/- {np.std(t1s):.1f}",
        f"top-5 mean {np.mean(t5s):.1f} +/- {np.std(t5s):.1f}",
        "",
        "Reference (same protocol, pitch-token input): 89.2 +/- 1.2 top-1, 97.9 top-5",
        "Reference (Phase 4 mel-spectrogram CNN): 11.46 top-1",
        "Random baseline for 40 classes: 2.5 top-1",
    ]
    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)
    print(f"\nwrote {REPORT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
