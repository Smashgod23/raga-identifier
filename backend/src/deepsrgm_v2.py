"""
v2 accuracy push on NOVEL real-world audio, three stacked levers on top of the
dual-pitch winner (expert+Essentia CMD contours, novel top-1 ~68 on n=69 splits):

  Stage 1  BASELINE   3 seeds of the dual-pitch model + their softmax ENSEMBLE.
                      Seed variance was +/-5pt, so averaging seeds converts that
                      instability into accuracy. Also re-baselines on n=194.
  Stage 2  AUGMENT    retrain 3 seeds with (a) tempo resampling of the token
                      time axis (+/-15%) and (b) global token jitter (+/-1..2
                      tokens = +/-50..100 cents), simulating small tonic error.
                      Same invariance principle that made dual-pitch win.
  Stage 3  TONIC TTA  tokens are linear in cents, so a tonic-detector error of
                      +/-500/700/1200 cents is an integer SHIFT of the token ids.
                      At inference, try the musically plausible shift hypotheses
                      and let ensemble confidence pick the frame. Attacks the
                      wrong-tonic failure mode directly, no retraining.

Eval: ALL 194 YouTube recordings (never in training -> all novel), fixed shared
windows per recording so every stage is compared on identical inputs. n=194
gives ~+/-2pt noise vs +/-5 on the old 69-recording splits.

Resumable: trained models are checkpointed to data/deepsrgm_v2_*.pt.
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
EVAL_WINDOWS = 24
# Tonic-error hypotheses in tokens (1 token = 50 cents): 0, +/-500c, +/-700c, +/-octave
TTA_SHIFTS = (0, 10, -10, 14, -14, 24, -24)


def load(npz):
    z = np.load(npz, allow_pickle=True)
    n = int(z["n"])
    return [z[f"s{i}"].astype(np.int64) for i in range(n)], np.array(z["labels"])


def crop_or_pad(s, rng):
    if len(s) <= SEQ_LEN:
        return np.pad(s, (0, SEQ_LEN - len(s)))
    st = rng.integers(0, len(s) - SEQ_LEN)
    return s[st:st + SEQ_LEN]


def subseqs(seqs, ids, labels, per, rng, aug=False):
    xs, ys = [], []
    for ri in ids:
        s = seqs[ri]
        for _ in range(per):
            sub = None
            if aug and rng.random() < 0.5:
                # tempo resample: read the source span at a stretched/compressed
                # rate so the same phrase appears faster/slower
                f = rng.uniform(0.85, 1.18)
                span = int(SEQ_LEN * f) + 1
                if len(s) > span:
                    st = rng.integers(0, len(s) - span)
                    pos = st + np.round(np.arange(SEQ_LEN) * f).astype(np.int64)
                    sub = s[np.minimum(pos, len(s) - 1)]
            if sub is None:
                sub = crop_or_pad(s, rng)
            if aug and rng.random() < 0.3:
                # small global shift = tonic slightly off (+/-50..100 cents)
                sub = np.clip(sub + rng.choice((-2, -1, 1, 2)), 0, VOCAB - 1)
            xs.append(sub)
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
    dual_s = ess_s + exp_s
    dual_y = np.concatenate([ess_y, exp_y])
    n_yt = len(yt_s)
    print(f"train: {len(dual_s)} dual-pitch CMD contours | eval: {n_yt} novel YouTube recs | dev={dev}", flush=True)

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.emb = nn.Embedding(VOCAB, 96)
            s.lstm = nn.LSTM(96, 192, batch_first=True, bidirectional=True)
            s.attn = nn.Linear(384, 1)
            s.fc = nn.Sequential(nn.Linear(384, 192), nn.ReLU(), nn.Dropout(0.4), nn.Linear(192, len(classes)))
        def forward(s, x):
            o, _ = s.lstm(s.emb(x))
            a = torch.softmax(s.attn(o).squeeze(-1), dim=1)
            return s.fc((o * a.unsqueeze(-1)).sum(1))

    # Fingerprint of everything that must match for a checkpoint to be reusable.
    # Guards against silently loading a stale .pt after a code/config change.
    CONFIG_FP = f"vocab{VOCAB}-seq{SEQ_LEN}-per{PER_REC}-ep{EPOCHS}-emb96-lstm192x1bi-attn-fc192-ntrain{len(dual_s)}"

    def train_or_load(tag, seed, aug):
        path = os.path.join(DATA, f"deepsrgm_v2_{tag}_s{seed}.pt")
        meta = path + ".json"
        net = Net().to(dev)
        if os.path.exists(path):
            fp = None
            if os.path.exists(meta):
                try:
                    fp = json.load(open(meta)).get("config_fp")
                except Exception:
                    fp = None
            if fp == CONFIG_FP:
                net.load_state_dict(torch.load(path, map_location=dev))
                print(f"  [{tag} s{seed}] loaded checkpoint (fp ok)", flush=True)
                return net
            print(f"  [{tag} s{seed}] checkpoint fp mismatch/missing -> retraining", flush=True)
        torch.manual_seed(seed)
        rng = np.random.default_rng(1000 + seed)
        Xtr, ytr = subseqs(dual_s, list(range(len(dual_s))), dual_y, PER_REC, rng, aug=aug)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        lossf = nn.CrossEntropyLoss()
        Xt = torch.tensor(Xtr); yt_ = torch.tensor(ytr); bs = 128
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
        torch.save(net.state_dict(), path)
        with open(meta, "w") as f:
            json.dump({"config_fp": CONFIG_FP, "tag": tag, "seed": seed}, f)
        print(f"  [{tag} s{seed}] trained + saved", flush=True)
        return net

    # Fixed eval windows per recording, shared across every stage/model/shift.
    windows = []
    for ri in range(n_yt):
        rng = np.random.default_rng(7000 + ri)
        windows.append(np.stack([crop_or_pad(yt_s[ri], rng) for _ in range(EVAL_WINDOWS)]))

    def ens_dist(models, W):
        """Average softmax over windows and models for one recording."""
        import torch as T
        x = T.tensor(W).to(dev)
        ps = []
        with T.no_grad():
            for m in models:
                m.eval()
                ps.append(m(x).softmax(1).mean(0).cpu().numpy())
        return np.mean(ps, axis=0)

    def evaluate(models, tta=False):
        t1 = t5 = 0
        shift_hist = {}
        oracle = 0
        for ri in range(n_yt):
            true = int(yt_y[ri])
            if not tta:
                p = ens_dist(models, windows[ri])
                chosen = 0
            else:
                best_conf, p, chosen, hit_any = -1.0, None, 0, False
                for sh in TTA_SHIFTS:
                    W = np.clip(windows[ri] + sh, 0, VOCAB - 1) if sh else windows[ri]
                    d = ens_dist(models, W)
                    if int(np.argmax(d)) == true:
                        hit_any = True
                    c = float(d.max())
                    if c > best_conf:
                        best_conf, p, chosen = c, d, sh
                oracle += hit_any
            shift_hist[chosen] = shift_hist.get(chosen, 0) + 1
            top5 = np.argsort(p)[::-1][:5]
            t1 += int(top5[0]) == true
            t5 += true in top5
        out = (t1 / n_yt * 100, t5 / n_yt * 100)
        if tta:
            print(f"    TTA chosen-shift histogram: {dict(sorted(shift_hist.items()))}", flush=True)
            print(f"    (diagnostic ORACLE any-shift-correct: {oracle / n_yt * 100:.1f}% — upper bound, not a claim)", flush=True)
        return out

    print("\n[stage 1] baseline dual-pitch, 3 seeds + ensemble (n=194)", flush=True)
    base_models = []
    for s in SEEDS:
        net = train_or_load("base", s, aug=False)
        a1, a5 = evaluate([net])
        print(f"  seed {s}: top-1 {a1:5.1f}  top-5 {a5:5.1f}", flush=True)
        base_models.append(net)
    b1, b5 = evaluate(base_models)
    print(f"  ENSEMBLE(3 base): top-1 {b1:5.1f}  top-5 {b5:5.1f}", flush=True)

    print("\n[stage 2] + tempo/tonic-jitter augmentation, 3 seeds + ensemble", flush=True)
    aug_models = []
    for s in SEEDS:
        net = train_or_load("aug", s, aug=True)
        a1, a5 = evaluate([net])
        print(f"  seed {s}: top-1 {a1:5.1f}  top-5 {a5:5.1f}", flush=True)
        aug_models.append(net)
    g1, g5 = evaluate(aug_models)
    print(f"  ENSEMBLE(3 aug): top-1 {g1:5.1f}  top-5 {g5:5.1f}", flush=True)

    print("\n[stage 3] tonic TTA on the aug ensemble", flush=True)
    t1, t5 = evaluate(aug_models, tta=True)
    print(f"  ENSEMBLE(3 aug) + tonic TTA: top-1 {t1:5.1f}  top-5 {t5:5.1f}", flush=True)

    print("\n[stage 3b] tonic TTA on ALL 6 models (base+aug)", flush=True)
    t61, t65 = evaluate(base_models + aug_models, tta=True)
    print(f"  ENSEMBLE(6) + tonic TTA: top-1 {t61:5.1f}  top-5 {t65:5.1f}", flush=True)

    print("\n=== v2 summary (all on the same 194 novel recordings) ===", flush=True)
    print(f"  base ensemble:        {b1:5.1f} / {b5:5.1f}")
    print(f"  aug ensemble:         {g1:5.1f} / {g5:5.1f}")
    print(f"  aug ensemble + TTA:   {t1:5.1f} / {t5:5.1f}")
    print(f"  6-model ens + TTA:    {t61:5.1f} / {t65:5.1f}")


if __name__ == "__main__":
    main()
