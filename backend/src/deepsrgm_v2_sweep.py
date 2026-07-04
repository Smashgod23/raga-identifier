"""
Eval-only sweep on the checkpointed v2 base models (no retraining).

v2 findings this sweep acts on:
  - Tonic TTA has huge headroom (oracle any-shift-correct 83.5% vs 64.4 actual)
    but max-confidence hypothesis selection is broken: it moved >50% of
    recordings off the detector's tonic when the detector is ~85-90% right,
    and np.clip at the vocab edges builds saturated token walls that look
    spuriously confident.
  Fixes tested here, all evaluation-side:
  1. OCTAVE-FOLD out-of-range tokens instead of clipping (musically correct).
  2. MARGIN-GATED selection: keep the detector's tonic (shift 0) unless a
     hypothesis beats it by a clear confidence margin delta.
  3. Restricted, musically-prioritized shift set {0, +/-Pa, +/-octave}.
  4. Longer eval windows (800 vs 1200 vs 1600 tokens = 35/53/70s of evidence).

Per-(recording, shift) ensemble distributions are computed once and cached, so
the margin/shift-set sweep is free numpy post-processing.
"""
import json
import os

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
BIN, LO, HI = 50.0, -1200, 2400
VOCAB = int((HI - LO) / BIN) + 1
OCT = 24                      # 1200 cents in tokens
EVAL_WINDOWS = 24
SEEDS = (0, 1, 2)
ALL_SHIFTS = (0, 10, -10, 14, -14, 24, -24)
DIST_CACHE = os.path.join(DATA, "v2_sweep_dists.npz")


def load(npz):
    z = np.load(npz, allow_pickle=True)
    n = int(z["n"])
    return [z[f"s{i}"].astype(np.int64) for i in range(n)], np.array(z["labels"])


def windows_for(s, L, rng):
    out = []
    for _ in range(EVAL_WINDOWS):
        if len(s) <= L:
            out.append(np.pad(s, (0, L - len(s))))
        else:
            st = rng.integers(0, len(s) - L)
            out.append(s[st:st + L])
    return np.stack(out)


def fold_shift(W, sh):
    """Shift token ids by sh, octave-folding anything that leaves the vocab
    range (musically: fold back by one octave) instead of clipping."""
    x = W + sh
    x = np.where(x >= VOCAB, x - OCT, x)
    x = np.where(x < 0, x + OCT, x)
    return x


def main():
    import torch
    import torch.nn as nn
    classes = json.load(open(os.path.join(DATA, "classes.json")))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    yt_s, yt_y = load(os.path.join(DATA, "deepsrgm_youtube_seqs.npz"))
    n_yt = len(yt_s)

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

    models = []
    for s in SEEDS:
        p = os.path.join(DATA, f"deepsrgm_v2_base_s{s}.pt")
        net = Net().to(dev)
        net.load_state_dict(torch.load(p, map_location=dev))
        net.eval()
        models.append(net)
    print(f"loaded {len(models)} base checkpoints | eval n={n_yt} | dev={dev}", flush=True)

    def ens_dist(W):
        x = torch.tensor(W).to(dev)
        ps = []
        with torch.no_grad():
            for m in models:
                ps.append(m(x).softmax(1).mean(0).cpu().numpy())
        return np.mean(ps, axis=0)

    def score(dists):
        t1 = t5 = 0
        for ri in range(n_yt):
            top5 = np.argsort(dists[ri])[::-1][:5]
            t1 += int(top5[0]) == int(yt_y[ri])
            t5 += int(yt_y[ri]) in top5
        return t1 / n_yt * 100, t5 / n_yt * 100

    # ---- Part A: window length, shift 0 only -------------------------------
    print("\n[part A] window length (shift 0, 3-model ensemble)", flush=True)
    best_L, best_t1, dists0 = None, -1, None
    for L in (800, 1200, 1600):
        d = np.stack([ens_dist(windows_for(yt_s[ri], L, np.random.default_rng(7000 + ri)))
                      for ri in range(n_yt)])
        a1, a5 = score(d)
        print(f"  L={L:4d}: top-1 {a1:5.1f}  top-5 {a5:5.1f}", flush=True)
        if a1 > best_t1:
            best_L, best_t1, dists0 = L, a1, d
    print(f"  -> best length {best_L}", flush=True)

    # ---- Part B: per-shift distributions at best_L (octave-fold), cached ----
    print(f"\n[part B] per-shift ensemble dists at L={best_L} (octave-fold)", flush=True)
    if os.path.exists(DIST_CACHE):
        z = np.load(DIST_CACHE)
        if int(z["L"]) == best_L:
            D = z["D"]
            print("  loaded cached dists", flush=True)
        else:
            D = None
    else:
        D = None
    if D is None:
        D = np.zeros((n_yt, len(ALL_SHIFTS), len(classes)), dtype=np.float64)
        for ri in range(n_yt):
            W0 = windows_for(yt_s[ri], best_L, np.random.default_rng(7000 + ri))
            for si, sh in enumerate(ALL_SHIFTS):
                D[ri, si] = ens_dist(fold_shift(W0, sh) if sh else W0)
            if (ri + 1) % 50 == 0:
                print(f"  {ri + 1}/{n_yt}", flush=True)
        np.savez_compressed(DIST_CACHE, D=D, L=best_L, shifts=np.array(ALL_SHIFTS))
        print("  dists cached", flush=True)

    oracle = 0
    for ri in range(n_yt):
        if any(int(np.argmax(D[ri, si])) == int(yt_y[ri]) for si in range(len(ALL_SHIFTS))):
            oracle += 1
    print(f"  oracle any-shift-correct (fold): {oracle / n_yt * 100:.1f}%", flush=True)

    # ---- Part C: margin-gated selection sweep (pure numpy, free) -----------
    print("\n[part C] margin-gated tonic selection sweep", flush=True)
    shift_sets = {
        "full7": list(range(len(ALL_SHIFTS))),
        "pa_oct": [ALL_SHIFTS.index(x) for x in (0, 14, -14, 24, -24)],
        "oct": [ALL_SHIFTS.index(x) for x in (0, 24, -24)],
    }
    base1, base5 = score(dists0)
    print(f"  no-TTA baseline @L={best_L}: {base1:5.1f} / {base5:5.1f}", flush=True)
    best = (None, base1, base5, None, 0)
    for name, idxs in shift_sets.items():
        for delta in (0.02, 0.05, 0.10, 0.15, 0.25):
            picked = []
            moved = 0
            for ri in range(n_yt):
                c0 = float(D[ri, 0].max())
                bi, bc = 0, c0
                for si in idxs:
                    if si == 0:
                        continue
                    c = float(D[ri, si].max())
                    if c > bc and c > c0 + delta:
                        bi, bc = si, c
                if bi != 0:
                    moved += 1
                picked.append(D[ri, bi])
            a1, a5 = score(np.stack(picked))
            print(f"  {name:7s} delta={delta:4.2f}: top-1 {a1:5.1f}  top-5 {a5:5.1f}  (moved {moved})", flush=True)
            if a1 > best[1]:
                best = (name, a1, a5, delta, moved)
    print(f"\n=== sweep summary ===")
    print(f"  best no-TTA:  L={best_L}  {base1:5.1f} / {base5:5.1f}")
    if best[0] is not None:
        print(f"  best TTA:     {best[0]} delta={best[3]}  {best[1]:5.1f} / {best[2]:5.1f}  (moved {best[4]}/194)")
    else:
        print("  TTA never beat the no-TTA baseline — detector tonic + margin gate wins")


if __name__ == "__main__":
    main()
