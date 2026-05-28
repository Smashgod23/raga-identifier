"""
Phase 12b: tonic scorer comparison (reuses cached pitch, no pyin re-run).

Phase 12 showed the candidate ceiling at K=10 is 83.5% but the
peakedness heuristic is stuck at 51.9%. The scorer, not the candidate
pool, is the bottleneck. This script tests alternative deterministic
scorers on the cached voiced pitches from data/tonic_voiced_cache.npz,
so it runs in seconds.

Scorers under test (each picks one candidate per recording from the
top-K histogram peaks, judged octave-agnostically against .tonicFine):

  peakedness   sum of squared PCD bins under the candidate (current prod).
  sa           energy in the +/-1 bins around 0 cents (the candidate
               itself should be a strong sustained pitch — the tanpura Sa).
  sa_pa        energy at 0 cents + energy at 700 cents (Pa). The tanpura
               drones Sa and Pa together, so the true Sa uniquely has BOTH
               a strong self-peak and a strong fifth.
  sa_pa_peak   sa_pa * peakedness (combine drone cue with overall structure).

Whichever wins by the widest margin over peakedness at a fixed K is a
drop-in production change with zero retraining.

Outputs:
  data/eval_tonic_scorers_report.txt
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.dirname(__file__))
from eval_harness import load_classes, write_report

TOLERANCE_CENTS = 25.0
CACHE = "data/tonic_voiced_cache.npz"


def pcd_under(voiced: np.ndarray, cand_hz: float) -> np.ndarray:
    cents = 1200.0 * np.log2(voiced / cand_hz)
    h, _ = np.histogram(cents % 1200, bins=120, range=(0, 1200))
    return h.astype(np.float64)


def bin_energy(pcd: np.ndarray, center_bin: int, half_width: int = 1) -> float:
    """Sum PCD energy in +/-half_width bins around center, wrapping at 120."""
    idxs = [(center_bin + d) % 120 for d in range(-half_width, half_width + 1)]
    return float(pcd[idxs].sum())


def score_candidate(pcd: np.ndarray, scorer: str) -> float:
    total = pcd.sum() + 1e-9
    p = pcd / total
    if scorer == "peakedness":
        return float(np.sum(pcd ** 2))
    if scorer == "sa":
        return bin_energy(p, 0)
    if scorer == "sa_pa":
        return bin_energy(p, 0) + bin_energy(p, 70)  # 700 cents = bin 70
    if scorer == "sa_pa_peak":
        sa_pa = bin_energy(p, 0) + bin_energy(p, 70)
        return sa_pa * float(np.sum(pcd ** 2))
    raise ValueError(scorer)


def candidates(voiced: np.ndarray, k: int):
    folded = voiced.copy()
    while np.any(folded > 120):
        folded = np.where(folded > 120, folded / 2, folded)
    while np.any(folded < 60):
        folded = np.where(folded < 60, folded * 2, folded)
    hist, edges = np.histogram(folded, bins=200, range=(60, 120))
    smoothed = uniform_filter1d(hist.astype(float), size=5)
    median_pitch = float(np.median(voiced))
    out = []
    for idx in np.argsort(smoothed)[::-1][:k]:
        if smoothed[idx] == 0:
            continue
        c = (edges[idx] + edges[idx + 1]) / 2
        while c * 2 < median_pitch:
            c *= 2
        out.append(c)
    return out


def matches(cand_hz: float, expert: float) -> bool:
    if cand_hz <= 0 or expert <= 0:
        return False
    diff = 1200.0 * math.log2(cand_hz / expert)
    return abs(((diff + 600) % 1200) - 600) <= TOLERANCE_CENTS


def main() -> None:
    if not os.path.exists(CACHE):
        raise SystemExit(f"missing {CACHE}; run eval_tonic_candidates.py first")
    z = np.load(CACHE, allow_pickle=True)
    voiced_list = list(z["voiced"])
    experts = z["experts"]
    valid = [(v, e) for v, e in zip(voiced_list, experts) if v is not None and e > 0]
    n = len(valid)

    scorers = ["peakedness", "sa", "sa_pa", "sa_pa_peak"]
    Ks = [5, 10]

    lines = [
        "=" * 78,
        "Phase 12b: tonic scorer comparison (octave-agnostic top-1, cached 60s pitch)",
        f"Recordings: {n}   Baseline peakedness@K5 is current production",
        "=" * 78,
        "",
        f"  {'scorer':14s} {'K=5 top-1':12s} {'K=10 top-1':12s}",
        f"  {'-'*14} {'-'*12} {'-'*12}",
    ]

    results = {}
    for scorer in scorers:
        row = f"  {scorer:14s} "
        for k in Ks:
            hits = 0
            for voiced, expert in valid:
                cands = candidates(voiced, k)
                if not cands:
                    continue
                best = max(cands, key=lambda c: score_candidate(pcd_under(voiced, c), scorer))
                if matches(best, expert):
                    hits += 1
            acc = 100 * hits / n
            results[(scorer, k)] = acc
            row += f"{acc:6.1f}%{'':6s}"
        lines.append(row)

    base = results[("peakedness", 5)]
    lines += [
        "",
        f"Production baseline (peakedness, K=5): {base:.1f}%",
        "Deltas vs baseline:",
    ]
    for scorer in scorers:
        for k in Ks:
            d = results[(scorer, k)] - base
            lines.append(f"  {scorer:14s} K={k:2d}:  {d:+5.1f} pp")

    write_report("data/eval_tonic_scorers_report.txt", lines)
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
