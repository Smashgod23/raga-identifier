"""
Honest evaluation of margin-gated tonic TTA via cross-validated config selection.

The sweep picked (shift set, margin delta) on the same 194 recordings it scored
— in-sample tuning. Here the config is chosen with 5-fold CV: pick the best
config on 4/5 of the recordings, score it on the held-out 1/5, pool the folds.
"No TTA" is in the config space, so if TTA doesn't generalize the CV falls back
to the baseline and the pooled number says so.

Also tests a BLEND mode: when the gate accepts a shifted hypothesis, the final
distribution is 0.5*chosen + 0.5*detector-frame — keeps the top-1 rescue while
protecting top-5 from a wrong hypothesis' garbage tail.

Pure numpy on the cached per-shift distributions (data/v2_sweep_dists.npz).

Disclosed caveat (review finding): the window length behind the cached dists
(L=1600) was itself selected by top-1 on the full 194 in the sweep's part A, so
the TTA delta is cross-validated but the absolute level inherits that one
in-sample choice. Folding L into the CV config space needs per-shift dists at
every L (extra model passes) and is future work.
"""
import os

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
SHIFT_SETS = {
    "none": (),
    "full7": (1, 2, 3, 4, 5, 6),
    "pa_oct": (3, 4, 5, 6),
    "oct": (5, 6),
}
DELTAS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
MODES = ("picked", "blend")
K_FOLDS = 5


def final_dists(D, set_idxs, delta, mode):
    """Apply the margin-gated selector to every recording; return (n,40) dists."""
    n = len(D)
    out = np.empty((n, D.shape[2]))
    for ri in range(n):
        c0 = float(D[ri, 0].max())
        bi, bc = 0, c0
        for si in set_idxs:
            c = float(D[ri, si].max())
            if c > bc and c > c0 + delta:
                bi, bc = si, c
        if bi == 0:
            out[ri] = D[ri, 0]
        elif mode == "blend":
            out[ri] = 0.5 * D[ri, bi] + 0.5 * D[ri, 0]
        else:
            out[ri] = D[ri, bi]
    return out


def score(dists, y, idxs):
    t1 = t5 = 0
    for ri in idxs:
        top5 = np.argsort(dists[ri])[::-1][:5]
        t1 += int(top5[0]) == int(y[ri])
        t5 += int(y[ri]) in top5
    return t1 / len(idxs) * 100, t5 / len(idxs) * 100


def main():
    z = np.load(os.path.join(DATA, "v2_sweep_dists.npz"))
    D = z["D"]
    # The shift-set indices above are positions into the sweep's ALL_SHIFTS
    # ordering; fail loudly if the cache was built with a different ordering.
    expected_shifts = (0, 10, -10, 14, -14, 24, -24)
    assert tuple(int(s) for s in z["shifts"]) == expected_shifts, \
        f"cached shift ordering {tuple(z['shifts'])} != expected {expected_shifts}"
    yz = np.load(os.path.join(DATA, "deepsrgm_youtube_seqs.npz"), allow_pickle=True)
    y = yz["labels"]
    n = len(D)
    assert len(y) == n
    print(f"cached dists: {D.shape}, L={int(z['L'])}", flush=True)

    # Precompute final dists for every config once; CV then just indexes.
    configs = []
    for sname, sidx in SHIFT_SETS.items():
        if sname == "none":
            configs.append(("none", 0.0, "picked", D[:, 0]))
            continue
        for d in DELTAS:
            for m in MODES:
                configs.append((sname, d, m, final_dists(D, sidx, d, m)))
    print(f"{len(configs)} configs precomputed", flush=True)

    rng = np.random.default_rng(42)
    order = rng.permutation(n)
    folds = np.array_split(order, K_FOLDS)

    pooled_t1 = pooled_t5 = 0
    picks = []
    for fi, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(order, test_idx)
        best = None
        for (sname, d, m, dists) in configs:
            a1, a5 = score(dists, y, train_idx)
            key = (a1, a5)
            if best is None or key > best[0]:
                best = (key, sname, d, m, dists)
        _, sname, d, m, dists = best
        f1, f5 = score(dists, y, test_idx)
        nt1 = int(round(f1 * len(test_idx) / 100))
        nt5 = int(round(f5 * len(test_idx) / 100))
        pooled_t1 += nt1
        pooled_t5 += nt5
        picks.append((sname, d, m))
        print(f"  fold {fi}: picked {sname} delta={d} {m} -> held-out {f1:5.1f} / {f5:5.1f}", flush=True)

    print("\n=== cross-validated TTA (honest, config chosen off-fold) ===")
    print(f"  pooled held-out: top-1 {pooled_t1 / n * 100:5.1f}  top-5 {pooled_t5 / n * 100:5.1f}   (n={n})")
    print(f"  baseline (no TTA): {score(D[:, 0], y, np.arange(n))[0]:5.1f} / {score(D[:, 0], y, np.arange(n))[1]:5.1f}")
    print(f"  fold picks: {picks}")

    # Transparency: in-sample best per mode (NOT the headline)
    print("\n  [transparency] in-sample bests (not honest headlines):")
    for m in MODES:
        best = max(((s, d, m, score(dists, y, np.arange(n)))
                    for (s, d, mm, dists) in configs if mm == m or s == "none"),
                   key=lambda x: x[3])
        print(f"    {m:6s}: {best[0]} delta={best[1]} -> {best[3][0]:5.1f} / {best[3][1]:5.1f}")


if __name__ == "__main__":
    main()
