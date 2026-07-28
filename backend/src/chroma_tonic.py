"""
Phase 16b: can the chromagram drone peak find the tonic?

This is the cheap half of the chroma idea and it targets the largest measured
gap in the project. On the 194-recording novel set the model scores 69.6% top-1,
but an oracle allowed to pick the best tonic hypothesis per recording reaches
about 79.9%. Roughly 10 points sit in tonic-hypothesis selection alone.

A tanpura drone is a continuously sounding Sa plus (usually) the fifth above it.
Melodia throws that away because it commits to one predominant f0 per frame and
the voice wins. A chromagram keeps it. So the time-averaged chroma of a Carnatic
recording should have a large peak at Sa, and this script measures how reliably
that is true against the 480 expert .tonicFine annotations.

Two scorers are compared:

  argmax    the single loudest pitch class
  Sa+Pa     score(b) = avg[b] + w * avg[b + 700 cents]

The second exists because argmax alone can land on the fifth instead of the
tonic. A true Sa has energy at Sa and at Sa+700. A Pa candidate would need
energy at Pa and at Pa+700, which is a much less common drone configuration, so
the template should break the tie in Sa's favor.

Accuracy is reported at exact-bin (20 cents) and within one bin, because the
downstream model quantizes to 50-cent tokens and a 20-cent error is harmless.

Run from backend/:
  source venv/bin/activate
  python src/chroma_tonic.py
"""

import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
CHROMA_DIR = os.path.join(DATA, "chroma")
REPORT = os.path.join(DATA, "chroma_tonic_report.txt")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocess_chroma as pc  # noqa: E402  (path set above)

N_CHROMA = pc.BINS_PER_OCTAVE
CENTS_PER_BIN = 1200.0 / pc.BINS_PER_OCTAVE
PA_BINS = int(round(700.0 / CENTS_PER_BIN))     # 35 bins = a just-ish fifth
PA_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def circ_dist(a, b, n=N_CHROMA):
    """Shortest distance between two bins on a circular pitch-class axis."""
    d = abs(int(a) - int(b)) % n
    return min(d, n - d)


def main():
    import librosa
    fmin = librosa.note_to_hz(pc.FMIN_NOTE)

    paths = []
    for root, _, files in os.walk(CHROMA_DIR):
        for f in sorted(files):
            if f.endswith(".npz"):
                paths.append(os.path.join(root, f))
    paths.sort()
    if not paths:
        print(f"no chroma files in {CHROMA_DIR}; run preprocess_chroma.py first")
        return 1

    profiles, true_bins, ragas = [], [], []
    for p in paths:
        z = np.load(p, allow_pickle=True)
        ch = z["chroma"].astype(np.float32)
        en = z["energy"].astype(np.float32)

        # Weight each frame by its linear energy. Loud frames are where the
        # instruments actually sound; silent frames carry no drone information
        # and averaging them in only adds noise.
        w = np.power(10.0, en, dtype=np.float64)
        w = np.clip(w, 0, np.percentile(w, 99))     # cap claps and mic bumps
        if w.sum() <= 0:
            continue
        prof = (ch * w[:, None]).sum(axis=0) / w.sum()
        s = prof.sum()
        if s <= 0:
            continue

        profiles.append(prof / s)
        tonic = float(z["tonic"])
        true_bins.append(int(round(1200.0 * np.log2(tonic / fmin)
                                   / CENTS_PER_BIN)) % N_CHROMA)
        ragas.append(str(z["raga"]))

    profiles = np.array(profiles)
    true_bins = np.array(true_bins)
    n = len(profiles)
    print(f"{n} recordings with usable chroma profiles\n")

    lines = ["Phase 16b: chroma drone peak as a tonic detector",
             f"n = {n} CompMusic recordings, ground truth = expert .tonicFine",
             f"bin resolution = {CENTS_PER_BIN:.0f} cents, "
             f"fifth offset = {PA_BINS} bins",
             ""]

    best = None
    for w in PA_WEIGHTS:
        # score(b) = profile[b] + w * profile[b + fifth]
        scores = profiles + w * np.roll(profiles, -PA_BINS, axis=1)
        pred = scores.argmax(axis=1)
        d = np.array([circ_dist(a, b) for a, b in zip(pred, true_bins)])
        exact = (d == 0).mean() * 100
        within1 = (d <= 1).mean() * 100
        within2 = (d <= 2).mean() * 100
        label = "argmax (no fifth term)" if w == 0 else f"Sa+Pa template w={w}"
        line = (f"{label:28s}  exact {exact:5.1f}%   "
                f"within 1 bin ({CENTS_PER_BIN:.0f}c) {within1:5.1f}%   "
                f"within 2 bins {within2:5.1f}%")
        print(line)
        lines.append(line)
        if best is None or within1 > best[1]:
            best = (w, within1, pred, d)

    w, within1, pred, d = best
    lines.append("")
    lines.append(f"best scorer: fifth weight {w}, {within1:.1f}% within one bin")

    # Where does it go wrong? If failures cluster at the fifth or the fourth,
    # that is a fixable systematic error rather than noise.
    wrong = d > 1
    if wrong.any():
        offs = [(int(pred[i]) - int(true_bins[i])) % N_CHROMA
                for i in np.where(wrong)[0]]
        counts = {}
        for o in offs:
            counts[o] = counts.get(o, 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:6]
        lines.append("")
        lines.append(f"failure offsets (predicted minus true, {wrong.sum()} misses):")
        for off, c in top:
            lines.append(f"  {off:3d} bins = {off * CENTS_PER_BIN:6.0f} cents  "
                         f"x{c}")
        print("\n" + "\n".join(lines[-len(top) - 2:]))

    lines += learned_template(profiles, true_bins)

    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {REPORT}")
    return 0


def learned_template(profiles, true_bins, folds=5, seed=0):
    """Learn the drone template instead of hand-writing it.

    The hand-written Sa+Pa template fails in a structured way: its worst error
    is predicting the note 500 cents above the true tonic. That is a tanpura
    tuned Sa-Ma rather than Sa-Pa, a normal Carnatic configuration that a fixed
    "tonic plus fifth" template cannot represent. Adding a fourth term by hand
    does not fix it either, because the fourth above Pa lands back on Sa and the
    two hypotheses stay tangled.

    So instead of guessing which intervals matter, learn one weight per pitch
    class relative to the candidate tonic. Scoring a candidate bin b means
    rotating the profile so b sits at index 0 and taking a dot product with the
    learned weights, which makes this a 60-way choice with circular weight
    sharing: sixty parameters total, trained with softmax cross-entropy.

    Evaluated with 5-fold cross-validation so the reported number is held out.
    """
    import torch

    n = len(profiles)
    # Rotations[i, b] is recording i's profile rotated so candidate b is index 0.
    rot = np.stack([np.stack([np.roll(p, -b) for b in range(N_CHROMA)])
                    for p in profiles]).astype(np.float32)   # (N, 60, 60)
    X = torch.tensor(rot)
    y = torch.tensor(true_bins, dtype=torch.long)

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    fold_of = np.zeros(n, dtype=int)
    for k, i in enumerate(order):
        fold_of[i] = k % folds

    correct = np.zeros(n, dtype=bool)
    preds = np.zeros(n, dtype=int)
    for f in range(folds):
        tr = torch.tensor(np.where(fold_of != f)[0])
        te = np.where(fold_of == f)[0]

        w = torch.zeros(N_CHROMA, requires_grad=True)
        opt = torch.optim.Adam([w], lr=0.1)
        for _ in range(300):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(X[tr] @ w, y[tr])
            loss.backward()
            opt.step()

        with torch.no_grad():
            p = (X[torch.tensor(te)] @ w).argmax(dim=1).numpy()
        preds[te] = p
        correct[te] = p == true_bins[te]

    d = np.array([circ_dist(a, b) for a, b in zip(preds, true_bins)])
    out = ["",
           "Learned circular drone template (60 weights, 5-fold CV, held out)",
           f"  exact {(d == 0).mean()*100:5.1f}%   "
           f"within 1 bin {(d <= 1).mean()*100:5.1f}%   "
           f"within 2 bins {(d <= 2).mean()*100:5.1f}%",
           "",
           "Baseline on the same 480 recordings (eval_essentia_tonic_report.txt):",
           "  Essentia TonicIndianArtMusic   85.4%  (octave-agnostic, +/-25 cents,",
           "                                 from a 60 s middle window)",
           "  sa_pa exact K=15                70.2%",
           "  peakedness K=5                  51.9%",
           "",
           "Read this honestly: the learned chroma template MATCHES the deployed",
           "Essentia detector, it does not beat it, and it uses far more audio to",
           "get there (up to 20 minutes against Essentia's 60 seconds). As a",
           "drop-in replacement it is not worth shipping.",
           "",
           "The open question this leaves is complementarity. The 10-point oracle",
           "gap needs BETTER HYPOTHESIS SELECTION, not a marginally different",
           "detector, so what matters is whether chroma is wrong on the same",
           "recordings Essentia is wrong on. If the errors are uncorrelated, the",
           "two together beat either alone, and that is the experiment to run",
           "next: score Essentia's tonic candidates with the chroma template",
           "rather than replacing them."]
    print("\n".join(out))
    return out


if __name__ == "__main__":
    sys.exit(main())
