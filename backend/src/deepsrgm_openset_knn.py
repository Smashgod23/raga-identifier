"""
Open-set upgrade: distance-based rejection in embedding space.

Instead of trusting softmax-derived scores, measure how far a clip's pooled
LSTM representation sits from the training data itself: score = negative
distance to the k-th nearest training embedding (deep k-NN OOD). Reuses the
open-set checkpoints and score cache; adds embeddings for the 768 training
contours and the 194 eval recordings.
"""
import json
import os

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
BIN, LO, HI = 50.0, -1200, 2400
N_PITCH = int((HI - LO) / BIN) + 1
PAD = N_PITCH
VOCAB = N_PITCH + 1
SEQ_LEN, EVAL_L, EVAL_W = 800, 1600, 24
SEEDS = (0, 1, 2)
N_UNSEEN = 8
K = 5


def load(npz):
    z = np.load(npz, allow_pickle=True)
    n = int(z["n"])
    return [z[f"s{i}"].astype(np.int64) for i in range(n)], np.array(z["labels"])


def crop_or_pad(s, L, rng):
    if len(s) <= L:
        return np.concatenate([s, np.full(L - len(s), PAD, dtype=np.int64)])
    st = rng.integers(0, len(s) - L)
    return s[st:st + L]


def auroc(a, b):
    x = np.concatenate([a, b])
    r = np.argsort(np.argsort(x)) + 1.0
    return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def main():
    import torch
    import torch.nn as nn
    classes = json.load(open(os.path.join(DATA, "classes.json")))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rng0 = np.random.default_rng(0)
    unseen = sorted(rng0.choice(len(classes), N_UNSEEN, replace=False).tolist())
    remap = {g: i for i, g in enumerate([i for i in range(len(classes)) if i not in unseen])}
    NC = len(remap)

    ess_s, ess_y = load(os.path.join(DATA, "deepsrgm_essentia_seqs.npz"))
    exp_s, exp_y = load(os.path.join(DATA, "deepsrgm_seqs.npz"))
    yt_s, yt_y = load(os.path.join(DATA, "deepsrgm_youtube_seqs.npz"))
    all_y = np.concatenate([ess_y, exp_y])
    tr_s = [s for s, l in zip(ess_s + exp_s, all_y) if int(l) in remap]

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.emb = nn.Embedding(VOCAB, 96, padding_idx=PAD)
            s.lstm = nn.LSTM(96, 192, batch_first=True, bidirectional=True)
            s.attn = nn.Linear(384, 1)
            s.fc = nn.Sequential(nn.Linear(384, 192), nn.ReLU(), nn.Dropout(0.4), nn.Linear(192, NC))
        def pooled(s, x):
            o, _ = s.lstm(s.emb(x))
            logits = s.attn(o).squeeze(-1).masked_fill(x == PAD, float("-inf"))
            a = torch.softmax(logits, dim=1)
            return (o * a.unsqueeze(-1)).sum(dim=1)
        def forward(s, x):
            return s.fc(s.pooled(x))

    models = []
    for sd in SEEDS:
        net = Net().to(dev)
        net.load_state_dict(torch.load(os.path.join(DATA, f"deepsrgm_os_s{sd}.pt"), map_location=dev))
        net.eval()
        models.append(net)
    print(f"loaded {len(models)} open-set checkpoints | dev={dev}", flush=True)

    def embed(seqs, L, wins):
        """Per recording: mean pooled embedding (over windows), concat over models, L2-normed."""
        out = []
        for ri in range(len(seqs)):
            rng = np.random.default_rng(7000 + ri)
            W = np.stack([crop_or_pad(seqs[ri], L, rng) for _ in range(wins)])
            x = torch.tensor(W).to(dev)
            es = []
            with torch.no_grad():
                for m in models:
                    e = m.pooled(x).mean(0).cpu().numpy()
                    es.append(e / (np.linalg.norm(e) + 1e-9))
            out.append(np.concatenate(es))
            if (ri + 1) % 100 == 0:
                print(f"  embedded {ri + 1}/{len(seqs)}", flush=True)
        return np.stack(out)

    print("embedding train contours...", flush=True)
    E_tr = embed(tr_s, SEQ_LEN, 8)
    print("embedding eval recordings...", flush=True)
    E_ev = embed(yt_s, EVAL_L, EVAL_W)

    # kNN score: negative distance to k-th nearest training embedding
    d = np.linalg.norm(E_ev[:, None, :] - E_tr[None, :, :], axis=2)
    knn_score = -np.sort(d, axis=1)[:, K - 1]

    is_seen = np.array([int(l) in remap for l in yt_y])
    z = np.load(os.path.join(DATA, "openset_scores.npz"))
    P, EN = z["P"], z["EN"]
    msp = P.max(axis=1)

    def fpr95(s):
        thr = np.percentile(s[is_seen], 5)
        return float((s[~is_seen] >= thr).mean())

    print(f"\nOOD comparison (ID={is_seen.sum()}, OOD={(~is_seen).sum()}):")
    for name, s in (("max-softmax", msp), ("energy", EN), (f"kNN-{K} embed", knn_score),
                    ("energy+kNN", (EN - EN.mean()) / EN.std() + (knn_score - knn_score.mean()) / knn_score.std())):
        print(f"  {name:14s}: AUROC {auroc(s[is_seen], s[~is_seen])*100:5.1f}   FPR@95TPR {fpr95(s)*100:5.1f}", flush=True)

    np.savez_compressed(os.path.join(DATA, "openset_knn.npz"),
                        E_tr=E_tr, E_ev=E_ev, knn=knn_score)
    print("embeddings cached -> data/openset_knn.npz")


if __name__ == "__main__":
    main()
