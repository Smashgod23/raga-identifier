"""
Open-set raga recognition experiment (the professor's problem, part 1).

Question: when the model hears a raga it was NEVER trained on, does it say so
(low similarity everywhere) instead of confidently naming a wrong raga?

Protocol:
  - Hold out 8 of the 40 ragas entirely (fixed rng(0) pick, disclosed in the
    output). Retrain the dual-pitch model (v3 arch: PAD token + attention
    masking, no frame head) on the remaining 32, 3 seeds.
  - Score ALL 194 novel YouTube recordings (never trained on, regardless of
    raga): recordings of the 32 seen ragas are in-distribution, recordings of
    the 8 held-out ragas are the "unseen raga" case.
  - Uncertainty scores compared: max softmax (MSP), negative entropy, and
    energy (logsumexp of logits). Metrics: AUROC and FPR@95TPR separating
    seen from unseen, plus closed-set top-1/top-5 on the seen recordings.

Per-recording scores are cached to data/openset_scores.npz for thresholding,
the site integration, and the writeup.
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
SEQ_LEN, PER_REC, EPOCHS = 800, 50, 25
SEEDS = (0, 1, 2)
EVAL_L, EVAL_W = 1600, 24
N_UNSEEN = 8
CACHE = os.path.join(DATA, "openset_scores.npz")


def load(npz):
    z = np.load(npz, allow_pickle=True)
    n = int(z["n"])
    return [z[f"s{i}"].astype(np.int64) for i in range(n)], np.array(z["labels"])


def crop_or_pad(s, L, rng):
    if len(s) <= L:
        return np.concatenate([s, np.full(L - len(s), PAD, dtype=np.int64)])
    st = rng.integers(0, len(s) - L)
    return s[st:st + L]


def auroc(id_scores, ood_scores):
    """Rank-based AUROC: P(id > ood). Higher score must mean more in-distribution."""
    x = np.concatenate([id_scores, ood_scores])
    r = np.argsort(np.argsort(x)) + 1.0
    r_id = r[:len(id_scores)].sum()
    n1, n2 = len(id_scores), len(ood_scores)
    return (r_id - n1 * (n1 + 1) / 2) / (n1 * n2)


def fpr_at_95tpr(id_scores, ood_scores):
    thr = np.percentile(id_scores, 5)          # accept 95% of in-distribution
    return float((ood_scores >= thr).mean())   # unseen wrongly accepted


def main():
    import torch
    import torch.nn as nn
    classes = json.load(open(os.path.join(DATA, "classes.json")))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    rng0 = np.random.default_rng(0)
    unseen = sorted(rng0.choice(len(classes), N_UNSEEN, replace=False).tolist())
    seen = [i for i in range(len(classes)) if i not in unseen]
    remap = {g: i for i, g in enumerate(seen)}
    NC = len(seen)
    print("UNSEEN (held-out) ragas:", [classes[i] for i in unseen], flush=True)

    ess_s, ess_y = load(os.path.join(DATA, "deepsrgm_essentia_seqs.npz"))
    exp_s, exp_y = load(os.path.join(DATA, "deepsrgm_seqs.npz"))
    yt_s, yt_y = load(os.path.join(DATA, "deepsrgm_youtube_seqs.npz"))
    tr_s = [s for s, l in zip(ess_s + exp_s, np.concatenate([ess_y, exp_y])) if int(l) in remap]
    tr_y = np.array([remap[int(l)] for l in np.concatenate([ess_y, exp_y]) if int(l) in remap])
    n_yt = len(yt_s)
    print(f"train {len(tr_s)} contours over {NC} seen ragas | eval {n_yt} novel recs | dev={dev}", flush=True)

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.emb = nn.Embedding(VOCAB, 96, padding_idx=PAD)
            s.lstm = nn.LSTM(96, 192, batch_first=True, bidirectional=True)
            s.attn = nn.Linear(384, 1)
            s.fc = nn.Sequential(nn.Linear(384, 192), nn.ReLU(), nn.Dropout(0.4), nn.Linear(192, NC))
        def forward(s, x):
            o, _ = s.lstm(s.emb(x))
            logits = s.attn(o).squeeze(-1).masked_fill(x == PAD, float("-inf"))
            a = torch.softmax(logits, dim=1)
            return s.fc((o * a.unsqueeze(-1)).sum(dim=1))

    FP = f"openset-u{'-'.join(map(str, unseen))}-vocab{VOCAB}-seq{SEQ_LEN}-per{PER_REC}-ep{EPOCHS}-nc{NC}-ntrain{len(tr_s)}"

    def train_or_load(seed):
        path = os.path.join(DATA, f"deepsrgm_os_s{seed}.pt")
        meta = path + ".json"
        net = Net().to(dev)
        if os.path.exists(path) and os.path.exists(meta):
            try:
                if json.load(open(meta)).get("config_fp") == FP:
                    net.load_state_dict(torch.load(path, map_location=dev))
                    print(f"  [os s{seed}] loaded checkpoint (fp ok)", flush=True)
                    return net
            except Exception:
                pass
            print(f"  [os s{seed}] fp mismatch -> retraining", flush=True)
        torch.manual_seed(seed)
        rng = np.random.default_rng(1000 + seed)
        xs, ys = [], []
        for ri in range(len(tr_s)):
            for _ in range(PER_REC):
                xs.append(crop_or_pad(tr_s[ri], SEQ_LEN, rng)); ys.append(int(tr_y[ri]))
        Xt = torch.tensor(np.array(xs)); yt_ = torch.tensor(np.array(ys))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        lossf = nn.CrossEntropyLoss()
        bs = 128
        if dev == "mps":
            torch.mps.empty_cache()
        for ep in range(EPOCHS):
            net.train()
            perm = torch.randperm(len(Xt))
            for b in range(0, len(perm), bs):
                bi = perm[b:b + bs]
                opt.zero_grad()
                loss = lossf(net(Xt[bi].to(dev)), yt_[bi].to(dev))
                loss.backward(); opt.step()
            sched.step()
            if (ep + 1) % 5 == 0:
                print(f"  [os s{seed}] epoch {ep+1}/{EPOCHS}", flush=True)
        torch.save(net.state_dict(), path)
        json.dump({"config_fp": FP, "seed": seed}, open(meta, "w"))
        print(f"  [os s{seed}] trained + saved", flush=True)
        return net

    models = [train_or_load(s) for s in SEEDS]

    if os.path.exists(CACHE) and str(np.load(CACHE)["fp"]) == FP:
        z = np.load(CACHE)
        P, EN = z["P"], z["EN"]
        print("loaded score cache", flush=True)
    else:
        P = np.zeros((n_yt, NC))       # ensemble mean softmax
        EN = np.zeros(n_yt)            # ensemble mean energy (logsumexp)
        for ri in range(n_yt):
            rng = np.random.default_rng(7000 + ri)
            W = np.stack([crop_or_pad(yt_s[ri], EVAL_L, rng) for _ in range(EVAL_W)])
            x = torch.tensor(W).to(dev)
            ps, es = [], []
            with torch.no_grad():
                for m in models:
                    m.eval()
                    lg = m(x)
                    ps.append(lg.softmax(1).mean(0).cpu().numpy())
                    es.append(torch.logsumexp(lg, dim=1).mean().item())
            P[ri] = np.mean(ps, axis=0)
            EN[ri] = float(np.mean(es))
            if (ri + 1) % 50 == 0:
                print(f"  scored {ri + 1}/{n_yt}", flush=True)
        np.savez_compressed(CACHE, P=P, EN=EN, fp=FP, unseen=np.array(unseen))
        print("score cache saved", flush=True)

    is_seen = np.array([int(l) in remap for l in yt_y])
    print(f"\neval split: {is_seen.sum()} seen-raga recs (ID) vs {(~is_seen).sum()} unseen-raga recs (OOD)", flush=True)

    # closed-set accuracy on seen recordings
    t1 = t5 = 0
    seen_idx = np.where(is_seen)[0]
    for ri in seen_idx:
        true = remap[int(yt_y[ri])]
        top5 = np.argsort(P[ri])[::-1][:5]
        t1 += int(top5[0]) == true
        t5 += true in top5
    print(f"closed-set on seen recs: top-1 {t1/len(seen_idx)*100:5.1f}  top-5 {t5/len(seen_idx)*100:5.1f}", flush=True)

    # OOD metrics
    msp = P.max(axis=1)
    ent = -np.array([-(p * np.log(p + 1e-12)).sum() for p in P])  # higher = more ID
    scores = {"max-softmax": msp, "neg-entropy": ent, "energy": EN}
    print("\nunseen-raga rejection (higher score must mean 'seen'):")
    for name, s in scores.items():
        a = auroc(s[is_seen], s[~is_seen])
        f = fpr_at_95tpr(s[is_seen], s[~is_seen])
        print(f"  {name:12s}: AUROC {a*100:5.1f}   FPR@95TPR {f*100:5.1f}   "
              f"mean seen {s[is_seen].mean():.3f} vs unseen {s[~is_seen].mean():.3f}", flush=True)

    print("\n(professor demo) mean top-similarity: "
          f"seen-raga clips {msp[is_seen].mean()*100:.1f}%  vs  unseen-raga clips {msp[~is_seen].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
