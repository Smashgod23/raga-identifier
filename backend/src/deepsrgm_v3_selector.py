"""
v3: LEARNED tonic-frame selector + pad fix.

v2's margin-gated tonic correction reached 69.6/85.6 (CV-honest) but the oracle
says ~82% of recordings have SOME correct shift hypothesis — the confidence
margin captures only part of that. Here the model itself learns to judge frames:

  - Multitask net: raga head (40-way) + frame head (binary "is this contour in
    the correct tonic frame?"). Frame negatives are training subsequences
    deliberately shifted by +/-500/700/1200 cents with octave fold — exactly the
    hypothesis set inference enumerates. Raga CE is masked to correct-frame
    samples only (teaching shifted->same-raga would create false transposition
    invariance and wreck allied-raga discrimination).
  - Pad fix (review P3): dedicated PAD token (id 73) + attention masking, so
    padding is no longer read as a real pitch and no longer moves under shifts.

Honest protocol, tightened from v2's review:
  - Attribution: v3 no-TTA ensemble vs v2 no-TTA baseline first (isolates the
    arch change before any selector claims).
  - Window length L IS in the CV config space this time (v2's disclosed debt).
  - Config space also contains "none" and v2's confidence-margin method, so the
    learned selector must beat both off-fold to win.
  - Headline = 5-fold CV, config chosen on train folds, pooled held-out score.

Resumable: model checkpoints (fingerprinted sidecars) + per-shift score cache.
"""
import json
import os

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
BIN, LO, HI = 50.0, -1200, 2400
N_PITCH = int((HI - LO) / BIN) + 1     # 73 real pitch tokens (0..72)
PAD = N_PITCH                           # dedicated pad id
VOCAB = N_PITCH + 1                     # embedding size 74
OCT = 24
SEQ_LEN, PER_REC, NEG_PER_REC, EPOCHS = 800, 50, 25, 25
SEEDS = (0, 1, 2)
SHIFTS = (0, 10, -10, 14, -14, 24, -24)
NEG_SHIFTS = (10, -10, 14, -14, 24, -24)
LENGTHS = (800, 1600)
FRAME_LOSS_W = 0.5
CACHE = os.path.join(DATA, "v3_selector_scores.npz")


def load(npz):
    z = np.load(npz, allow_pickle=True)
    n = int(z["n"])
    return [z[f"s{i}"].astype(np.int64) for i in range(n)], np.array(z["labels"])


def fold_shift(x, sh):
    """Shift pitch tokens by sh with octave fold; PAD stays PAD."""
    pad_mask = x == PAD
    y = x + sh
    y = np.where(y >= N_PITCH, y - OCT, y)
    y = np.where(y < 0, y + OCT, y)
    return np.where(pad_mask, PAD, y)


def crop_or_pad(s, L, rng):
    if len(s) <= L:
        return np.concatenate([s, np.full(L - len(s), PAD, dtype=np.int64)])
    st = rng.integers(0, len(s) - L)
    return s[st:st + L]


def main():
    import torch
    import torch.nn as nn
    classes = json.load(open(os.path.join(DATA, "classes.json")))
    NC = len(classes)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    ess_s, ess_y = load(os.path.join(DATA, "deepsrgm_essentia_seqs.npz"))
    exp_s, exp_y = load(os.path.join(DATA, "deepsrgm_seqs.npz"))
    yt_s, yt_y = load(os.path.join(DATA, "deepsrgm_youtube_seqs.npz"))
    dual_s = ess_s + exp_s
    dual_y = np.concatenate([ess_y, exp_y])
    n_yt = len(yt_s)
    print(f"train {len(dual_s)} dual-pitch contours | eval {n_yt} novel | dev={dev}", flush=True)

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.emb = nn.Embedding(VOCAB, 96, padding_idx=PAD)
            s.lstm = nn.LSTM(96, 192, batch_first=True, bidirectional=True)
            s.attn = nn.Linear(384, 1)
            s.fc = nn.Sequential(nn.Linear(384, 192), nn.ReLU(), nn.Dropout(0.4), nn.Linear(192, NC))
            s.frame = nn.Linear(384, 1)
        def forward(s, x):
            o, _ = s.lstm(s.emb(x))
            logits = s.attn(o).squeeze(-1)
            logits = logits.masked_fill(x == PAD, float("-inf"))
            a = torch.softmax(logits, dim=1)
            pooled = (o * a.unsqueeze(-1)).sum(dim=1)
            return s.fc(pooled), s.frame(pooled).squeeze(-1)

    CONFIG_FP = (f"v3-vocab{VOCAB}-pad{PAD}-seq{SEQ_LEN}-per{PER_REC}+{NEG_PER_REC}neg-"
                 f"ep{EPOCHS}-emb96-lstm192x1bi-attnmask-fc192-frame{FRAME_LOSS_W}-ntrain{len(dual_s)}")

    def build_training(rng):
        xs, ys, fs = [], [], []
        for ri in range(len(dual_s)):
            s = dual_s[ri]
            lab = int(dual_y[ri])
            for _ in range(PER_REC):
                xs.append(crop_or_pad(s, SEQ_LEN, rng)); ys.append(lab); fs.append(1.0)
            for _ in range(NEG_PER_REC):
                sub = crop_or_pad(s, SEQ_LEN, rng)
                xs.append(fold_shift(sub, int(rng.choice(NEG_SHIFTS)))); ys.append(lab); fs.append(0.0)
        return np.array(xs), np.array(ys), np.array(fs, dtype=np.float32)

    def train_or_load(seed):
        path = os.path.join(DATA, f"deepsrgm_v3_s{seed}.pt")
        meta = path + ".json"
        net = Net().to(dev)
        if os.path.exists(path) and os.path.exists(meta):
            try:
                if json.load(open(meta)).get("config_fp") == CONFIG_FP:
                    net.load_state_dict(torch.load(path, map_location=dev))
                    print(f"  [v3 s{seed}] loaded checkpoint (fp ok)", flush=True)
                    return net
            except Exception:
                pass
            print(f"  [v3 s{seed}] fp mismatch -> retraining", flush=True)
        torch.manual_seed(seed)
        rng = np.random.default_rng(1000 + seed)
        Xtr, ytr, ftr = build_training(rng)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        ce = nn.CrossEntropyLoss(reduction="none")
        bce = nn.BCEWithLogitsLoss()
        Xt = torch.tensor(Xtr); yt_ = torch.tensor(ytr); ft = torch.tensor(ftr)
        bs = 128
        if dev == "mps":
            torch.mps.empty_cache()
        for ep in range(EPOCHS):
            net.train()
            perm = torch.randperm(len(Xt))
            for b in range(0, len(perm), bs):
                bi = perm[b:b + bs]
                xb, yb, fb = Xt[bi].to(dev), yt_[bi].to(dev), ft[bi].to(dev)
                opt.zero_grad()
                cls_logits, frame_logits = net(xb)
                # raga CE only on correct-frame samples
                ce_all = ce(cls_logits, yb)
                cls_loss = (ce_all * fb).sum() / fb.sum().clamp(min=1.0)
                frame_loss = bce(frame_logits, fb)
                (cls_loss + FRAME_LOSS_W * frame_loss).backward()
                opt.step()
            sched.step()
            if (ep + 1) % 5 == 0:
                print(f"  [v3 s{seed}] epoch {ep+1}/{EPOCHS}", flush=True)
        torch.save(net.state_dict(), path)
        json.dump({"config_fp": CONFIG_FP, "seed": seed}, open(meta, "w"))
        print(f"  [v3 s{seed}] trained + saved", flush=True)
        return net

    models = [train_or_load(s) for s in SEEDS]

    # ---- score cache: per (recording, L, shift): raga dist + frame score ----
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        if z["fp"] == CONFIG_FP and tuple(int(v) for v in z["shifts"]) == SHIFTS \
                and tuple(int(v) for v in z["lengths"]) == LENGTHS:
            D, F = z["D"], z["F"]
            print("loaded score cache", flush=True)
        else:
            D = None
    else:
        D = None
    if D is None:
        D = np.zeros((n_yt, len(LENGTHS), len(SHIFTS), NC))
        F = np.zeros((n_yt, len(LENGTHS), len(SHIFTS)))
        import torch as T
        for ri in range(n_yt):
            for li, L in enumerate(LENGTHS):
                rng = np.random.default_rng(7000 + ri)
                W0 = np.stack([crop_or_pad(yt_s[ri], L, rng) for _ in range(24)])
                for si, sh in enumerate(SHIFTS):
                    W = fold_shift(W0, sh) if sh else W0
                    x = T.tensor(W).to(dev)
                    ds, fs = [], []
                    with T.no_grad():
                        for m in models:
                            m.eval()
                            cl, fr = m(x)
                            ds.append(cl.softmax(1).mean(0).cpu().numpy())
                            fs.append(T.sigmoid(fr).mean().item())
                    D[ri, li, si] = np.mean(ds, axis=0)
                    F[ri, li, si] = float(np.mean(fs))
            if (ri + 1) % 25 == 0:
                print(f"  scored {ri + 1}/{n_yt}", flush=True)
        np.savez_compressed(CACHE, D=D, F=F, fp=CONFIG_FP,
                            shifts=np.array(SHIFTS), lengths=np.array(LENGTHS))
        print("score cache saved", flush=True)

    def score(dists, idxs):
        t1 = t5 = 0
        for ri in idxs:
            top5 = np.argsort(dists[ri])[::-1][:5]
            t1 += int(top5[0]) == int(yt_y[ri])
            t5 += int(yt_y[ri]) in top5
        return t1 / len(idxs) * 100, t5 / len(idxs) * 100

    # ---- attribution: v3 no-TTA vs v2 baseline ------------------------------
    print("\n[attribution] v3 no-TTA ensemble (arch change only)", flush=True)
    for li, L in enumerate(LENGTHS):
        a1, a5 = score(D[:, li, 0], np.arange(n_yt))
        print(f"  L={L}: top-1 {a1:5.1f}  top-5 {a5:5.1f}   (v2 @1600 was 66.0/82.0)", flush=True)
    for li, L in enumerate(LENGTHS):
        orc = sum(any(int(np.argmax(D[ri, li, si])) == int(yt_y[ri]) for si in range(len(SHIFTS)))
                  for ri in range(n_yt))
        print(f"  oracle any-shift @L={L}: {orc / n_yt * 100:.1f}%", flush=True)

    # ---- config space -------------------------------------------------------
    def final_dists(li, selector, delta, mode):
        out = np.empty((n_yt, NC))
        for ri in range(n_yt):
            if selector == "frame":
                base_s = F[ri, li, 0]
                bi, bs_ = 0, base_s
                for si in range(1, len(SHIFTS)):
                    s_ = F[ri, li, si]
                    if s_ > bs_ and s_ > base_s + delta:
                        bi, bs_ = si, s_
            else:  # legacy confidence margin
                c0 = float(D[ri, li, 0].max())
                bi, bc = 0, c0
                for si in range(1, len(SHIFTS)):
                    c = float(D[ri, li, si].max())
                    if c > bc and c > c0 + delta:
                        bi, bc = si, c
            if bi == 0:
                out[ri] = D[ri, li, 0]
            elif mode == "blend":
                out[ri] = 0.5 * D[ri, li, bi] + 0.5 * D[ri, li, 0]
            else:
                out[ri] = D[ri, li, bi]
        return out

    configs = []
    for li, L in enumerate(LENGTHS):
        configs.append((f"none@L{L}", D[:, li, 0]))
        for sel, dgrid in (("frame", (0.0, 0.05, 0.10, 0.20)), ("conf", (0.15, 0.25))):
            for d in dgrid:
                for m in ("picked", "blend"):
                    configs.append((f"{sel}-d{d}-{m}@L{L}", final_dists(li, sel, d, m)))
    print(f"\n[CV] {len(configs)} configs (includes none + legacy conf-margin)", flush=True)

    rng = np.random.default_rng(42)
    order = rng.permutation(n_yt)
    folds = np.array_split(order, 5)
    pooled1 = pooled5 = 0
    picks = []
    for fi, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(order, test_idx)
        best = max(configs, key=lambda c: score(c[1], train_idx))
        f1, f5 = score(best[1], test_idx)
        pooled1 += int(round(f1 * len(test_idx) / 100))
        pooled5 += int(round(f5 * len(test_idx) / 100))
        picks.append(best[0])
        print(f"  fold {fi}: {best[0]} -> held-out {f1:5.1f} / {f5:5.1f}", flush=True)

    print("\n=== v3 learned-selector summary ===")
    print(f"  pooled held-out (headline): top-1 {pooled1 / n_yt * 100:5.1f}  top-5 {pooled5 / n_yt * 100:5.1f}")
    print(f"  fold picks: {picks}")
    print(f"  (v2 CV-honest was 69.6 / 85.6; v2 no-TTA baseline 66.0 / 82.0)")


if __name__ == "__main__":
    main()
