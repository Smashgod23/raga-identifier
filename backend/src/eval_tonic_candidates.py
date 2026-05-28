"""
Phase 12 diagnostic: what actually raises the tonic ceiling?

The README previously claimed the upward-only octave fold caps the
candidate ceiling at 60.4%. That reasoning is wrong: tonic correctness
is judged octave-agnostically (cent difference wrapped mod 1200), and
the downstream TDMS feature is octave-folded, so the *octave* a
candidate lands in is irrelevant. Only its pitch class matters. The
ceiling is therefore a pitch-class-coverage problem — how often the
true Sa's pitch class is among the top-K histogram peaks — not a fold
problem.

This script tests that empirically. It extracts the pyin pitch contour
once per recording (60s @ offset 10s, matching production tonic
detection), caches it, then sweeps:
  * candidate count K in {5, 8, 10, 15}
  * fold strategy: upward-only (current) vs bidirectional vs none

For each combination it reports, octave-agnostically against the
expert .tonicFine:
  * ceiling: fraction of recordings where some top-K candidate is
    within +/-25 cents (mod octave) of the expert tonic
  * heuristic top-1: fraction where the peakedness-best candidate matches

If bidirectional fold == upward-only fold (as predicted), the fold is a
non-lever. If raising K raises the ceiling, that's the cheap production
fix.

Outputs:
  data/tonic_voiced_cache.npz      (cached per-recording voiced pitches)
  data/eval_tonic_candidates_report.txt
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.dirname(__file__))
from eval_audio_ab import audio_path_for
from eval_harness import load_classes, write_report

EXTRACT_OFFSET = 10.0
EXTRACT_DUR = 60.0
TOLERANCE_CENTS = 25.0


def find_tonic_fine(recording_id: str):
    features_root = os.path.join(
        os.path.dirname(__file__), "..", "data", "RagaDataset", "Carnatic", "features"
    )
    rec_dir = os.path.join(features_root, recording_id)
    if not os.path.isdir(rec_dir):
        return None
    for name in os.listdir(rec_dir):
        if name.endswith(".tonicFine"):
            with open(os.path.join(rec_dir, name)) as f:
                return float(f.read().strip())
    return None


def extract_voiced(args: tuple) -> dict:
    idx, recording_id = args
    import librosa

    path = audio_path_for(recording_id)
    if not os.path.exists(path):
        return {"idx": idx, "error": "missing audio"}
    try:
        y, sr = librosa.load(path, sr=16000, mono=True, offset=EXTRACT_OFFSET, duration=EXTRACT_DUR)
    except Exception as exc:
        return {"idx": idx, "error": str(exc)}
    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    if rms > 1e-6:
        y = y * (0.1 / rms)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=60, fmax=800, sr=sr, frame_length=1024, hop_length=256
    )
    voiced = f0[voiced_flag & (voiced_probs > 0.3)]
    if len(voiced) < 30:
        voiced = f0[voiced_flag]
    voiced = voiced[~np.isnan(voiced)]
    if len(voiced) < 30:
        return {"idx": idx, "error": "too little voiced"}
    return {"idx": idx, "voiced": voiced.astype(np.float32)}


def generate_candidates(voiced: np.ndarray, k: int, fold: str):
    """Return list of (hz, peakedness) for top-k smoothed histogram peaks.

    fold: 'up' (current production), 'bi' (emit both octaves bracketing
    median), or 'none' (leave candidate in [60,120]). Because matching is
    octave-agnostic, 'none' and 'up' should produce identical ceilings —
    that's the control proving the fold is a non-lever.
    """
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
        base = (edges[idx] + edges[idx + 1]) / 2
        if fold == "none":
            cands = [base]
        elif fold == "up":
            c = base
            while c * 2 < median_pitch:
                c *= 2
            cands = [c]
        elif fold == "bi":
            c = base
            while c * 2 < median_pitch:
                c *= 2
            cands = [c, c * 2]  # octave below + above the median bracket
        else:
            cands = [base]
        for c in cands:
            cents = 1200.0 * np.log2(voiced / c)
            h, _ = np.histogram(cents % 1200, bins=120, range=(0, 1200))
            out.append((float(c), float(np.sum(h ** 2))))
    return out


def matches(cand_hz: float, expert: float) -> bool:
    if cand_hz <= 0 or expert <= 0:
        return False
    diff = 1200.0 * math.log2(cand_hz / expert)
    diff_mod = ((diff + 600) % 1200) - 600
    return abs(diff_mod) <= TOLERANCE_CENTS


def main() -> None:
    classes = load_classes()
    with open("data/tdms_meta.json") as f:
        meta = json.load(f)

    cache_path = "data/tonic_voiced_cache.npz"
    if os.path.exists(cache_path):
        print("Loading cached voiced pitches...")
        z = np.load(cache_path, allow_pickle=True)
        voiced_list = list(z["voiced"])
        experts = z["experts"]
    else:
        jobs = [(i, m["recording_id"]) for i, m in enumerate(meta)]
        print(f"Extracting pyin pitch for {len(jobs)} recordings (60s @ +10s)...")
        voiced_list = [None] * len(jobs)
        t0 = time.time()
        done = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
            for r in pool.map(extract_voiced, jobs, chunksize=1):
                done += 1
                if "error" not in r:
                    voiced_list[r["idx"]] = r["voiced"]
                if done % 50 == 0 or done == len(jobs):
                    rate = done / max(1e-6, time.time() - t0)
                    print(f"  [{done}/{len(jobs)}] {rate:.2f} rec/s   ETA {(len(jobs)-done)/max(1e-6,rate)/60:.1f} min")
        experts = np.array([find_tonic_fine(m["recording_id"]) or 0.0 for m in meta], dtype=np.float64)
        np.savez(cache_path, voiced=np.array(voiced_list, dtype=object), experts=experts)
        print(f"Cached to {cache_path}")

    valid = [(v, e) for v, e in zip(voiced_list, experts) if v is not None and e > 0]
    n = len(valid)
    print(f"\n{n} recordings with voiced pitch + expert tonic\n")

    lines = [
        "=" * 78,
        "Phase 12 diagnostic: tonic candidate ceiling vs K and fold strategy",
        f"Window: 60s @ +10s   Matching: octave-agnostic, +/-25 cents   Recordings: {n}",
        "=" * 78,
        "",
        f"  {'strategy':28s} {'ceiling (Sa in top-K)':24s} {'heuristic top-1':16s}",
        f"  {'-'*28} {'-'*24} {'-'*16}",
    ]

    for fold in ["up", "none", "bi"]:
        for k in [5, 8, 10, 15]:
            ceil_hits = 0
            top1_hits = 0
            for voiced, expert in valid:
                cands = generate_candidates(voiced, k, fold)
                if not cands:
                    continue
                if any(matches(hz, expert) for hz, _ in cands):
                    ceil_hits += 1
                best_hz = max(cands, key=lambda c: c[1])[0]
                if matches(best_hz, expert):
                    top1_hits += 1
            label = f"fold={fold:4s} K={k:2d}"
            lines.append(
                f"  {label:28s} {100*ceil_hits/n:6.1f}%{'':17s} {100*top1_hits/n:6.1f}%"
            )
        lines.append("")

    lines += [
        "Reading:",
        "  - If fold=up and fold=none give identical ceilings at each K, the octave",
        "    fold is confirmed a non-lever (matching is octave-agnostic).",
        "  - If ceiling rises with K, more candidates is the cheap production fix.",
        "  - heuristic top-1 is what production's _detect_tonic actually achieves.",
    ]
    write_report("data/eval_tonic_candidates_report.txt", lines)
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
