"""
Phase 10b: multi-window query aggregation.

Phase 10a established that longer templates (5min vs 60s) buy +9.3 pp
top-1 with the query side held at 60s. This script tests the matched-
cost variant: keep the long templates, but extract THREE 60s windows
from each query recording (at 25%, 50%, 75% through) and average their
TDMSs before the 1-NN lookup. The total inference work is roughly 3x
(three pyin passes instead of one) but the templates are the same long
ones.

Expected behavior: averaging multiple TDMSs reduces the variance of any
single window's sparse distribution. If the dominant remaining gap to
the Phase 1 ceiling is per-window noise rather than absolute window
length, this should add another 2-5 pp on top of Phase 10a.

Setup:
  Templates: 5-min window, pyin + expert tonic. Reuses
             data/X_tdms_long.npy from Phase 10a.
  Queries:   Three 60s windows at 25/50/75% through each recording,
             pyin + expert tonic. Three TDMSs averaged into one.
  Classifier: 1-NN with symmetric KL.
  Eval: 5-fold stratified CV at the recording level, 5 seeds.

Outputs:
  data/X_tdms_audio_multiwindow.npy
  data/eval_multiwindow_queries_report.txt
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from build_tdms import compute_tdms
from eval_audio_ab import TDMSKnnDirect, audio_path_for
from eval_audio_expert_tonic import find_tonic_fine
from eval_harness import load_classes, write_report

QUERY_DURATION = 60.0
WINDOW_FRACTIONS = (0.25, 0.50, 0.75)


def extract_multiwindow_query(args: tuple) -> dict:
    """Extract and average TDMSs from three windows of the same recording."""
    idx, recording_id, label, expert_tonic = args
    import librosa

    if expert_tonic is None or expert_tonic <= 0:
        return {"idx": idx, "error": "no expert tonic"}
    path = audio_path_for(recording_id)
    if not os.path.exists(path):
        return {"idx": idx, "error": f"missing audio: {path}"}

    try:
        total = librosa.get_duration(path=path)
    except Exception as exc:
        return {"idx": idx, "error": f"duration probe failed: {exc}"}

    tdmss: list[np.ndarray] = []
    for frac in WINDOW_FRACTIONS:
        # Center a QUERY_DURATION-second window at `frac` through the recording,
        # clipped so the window stays within the file.
        center = total * frac
        offset = max(0.0, min(center - QUERY_DURATION / 2.0, total - QUERY_DURATION))
        duration = min(QUERY_DURATION, total)

        try:
            y, sr = librosa.load(path, sr=16000, mono=True, offset=offset, duration=duration)
        except Exception:
            continue

        rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
        if rms > 1e-6:
            y = y * (0.1 / rms)

        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=60, fmax=800, sr=sr, frame_length=1024, hop_length=256
        )
        if voiced_flag.sum() < 30:
            continue

        hop_seconds = 256.0 / sr
        pitches_full = np.where(voiced_flag, f0, 0.0)
        pitches_full = np.where(np.isfinite(pitches_full), pitches_full, 0.0)
        timestamps = np.arange(len(pitches_full)) * hop_seconds

        # Per-window tonic fold: each window's median voiced pitch sets the octave.
        voiced_pitches = f0[voiced_flag]
        median_pitch = float(np.median(voiced_pitches))
        tonic = float(expert_tonic)
        while tonic * 2 <= median_pitch * 1.5:
            tonic *= 2
        while tonic >= median_pitch * 1.5:
            tonic /= 2

        tdms = compute_tdms(timestamps, pitches_full, tonic)
        if tdms is not None:
            tdmss.append(tdms.ravel())

    if not tdmss:
        return {"idx": idx, "error": "no usable windows"}

    # Average then re-L1-normalize (averaging preserves L1=1, this is a no-op
    # but defensive against floating-point drift).
    avg = np.mean(np.stack(tdmss, axis=0), axis=0)
    avg /= max(float(avg.sum()), 1e-12)

    return {
        "idx": idx,
        "recording_id": recording_id,
        "label": label,
        "n_windows": len(tdmss),
        "tdms": avg.astype(np.float32),
    }


def asymmetric_cv(X_template, X_query, y, n_classes, *, n_splits=5, seeds=(0, 1, 2, 3, 4)):
    from sklearn.model_selection import StratifiedKFold

    top_k = (1, 5)
    per_fold = np.zeros((len(seeds), n_splits, len(top_k)), dtype=np.float64)
    for s_idx, seed in enumerate(seeds):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for f_idx, (train_idx, test_idx) in enumerate(skf.split(X_query, y)):
            clf = TDMSKnnDirect(n_classes)
            clf.fit(X_template[train_idx], y[train_idx])
            proba = clf.predict_proba(X_query[test_idx])
            for k_idx, k in enumerate(top_k):
                top = np.argsort(-proba, axis=1)[:, :k]
                hits = (top == y[test_idx][:, None]).any(axis=1)
                per_fold[s_idx, f_idx, k_idx] = hits.mean()
    mean = per_fold.mean(axis=(0, 1))
    std = per_fold.mean(axis=1).std(axis=0)
    return {"top_k_mean": mean, "top_k_std": std}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    classes = load_classes()
    n_classes = len(classes)

    with open("data/tdms_meta.json") as f:
        meta = json.load(f)
    y_full = np.load("data/y_tdms.npy")
    assert len(meta) == len(y_full) == 480

    # ---- Build multi-window queries -----------------------------------
    if args.skip_extract and os.path.exists("data/X_tdms_audio_multiwindow.npy"):
        print("Loading cached multi-window queries from data/X_tdms_audio_multiwindow.npy")
        X_query = np.load("data/X_tdms_audio_multiwindow.npy")
    else:
        jobs = []
        for i, m in enumerate(meta):
            t = find_tonic_fine(m["recording_id"])
            jobs.append((i, m["recording_id"], int(y_full[i]), t))
        print(
            f"Extracting 3-window-average queries for {len(jobs)} recordings, "
            f"{args.workers} workers"
        )
        # 3 pyin calls per recording at ~5s each = ~15s per record
        # Parallelized 6-way: 480 * 15 / 6 = ~20 min
        X_query = np.zeros((len(jobs), 14400), dtype=np.float32)
        t0 = time.time()
        done = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for r in pool.map(extract_multiwindow_query, jobs, chunksize=1):
                done += 1
                if "error" in r:
                    if done <= 5 or done % 50 == 0:
                        print(f"  [{done}/{len(jobs)}] ERROR idx={r['idx']}: {r['error']}")
                    continue
                X_query[r["idx"]] = r["tdms"]
                if done % 25 == 0 or done == len(jobs):
                    rate = done / max(1e-6, time.time() - t0)
                    eta = (len(jobs) - done) / max(1e-6, rate)
                    print(f"  [{done}/{len(jobs)}] {rate:.2f} rec/s   ETA {eta / 60:.1f} min")
        np.save("data/X_tdms_audio_multiwindow.npy", X_query)
        print(f"Saved data/X_tdms_audio_multiwindow.npy")

    # ---- Load Phase 10a templates -------------------------------------
    X_template = np.load("data/X_tdms_long.npy")
    X_single = np.load("data/X_tdms_audio_experttonic.npy")  # Phase 9a single-window query
    print(f"\nTemplates: {X_template.shape}    Single-window queries: {X_single.shape}    "
          f"Multi-window queries: {X_query.shape}")

    template_ok = ~np.all(X_template == 0, axis=1)
    single_ok = ~np.all(X_single == 0, axis=1)
    multi_ok = ~np.all(X_query == 0, axis=1)
    keep = template_ok & single_ok & multi_ok
    X_template = X_template[keep]
    X_single = X_single[keep]
    X_query = X_query[keep]
    y = y_full[keep]
    print(f"All three extractions OK: {int(keep.sum())}/480 recordings")

    # ---- Evaluate -----------------------------------------------------
    print("\n[1/2] Long template + single 60s query (Phase 10a check)...")
    t0 = time.time()
    p10a = asymmetric_cv(X_template, X_single, y, n_classes)
    print(f"    done in {time.time() - t0:.1f}s")

    print("\n[2/2] Long template + 3-window-averaged query (Phase 10b)...")
    t0 = time.time()
    p10b = asymmetric_cv(X_template, X_query, y, n_classes)
    print(f"    done in {time.time() - t0:.1f}s")

    am, as_ = p10a["top_k_mean"], p10a["top_k_std"]
    bm, bs = p10b["top_k_mean"], p10b["top_k_std"]
    d1 = (bm[0] - am[0]) * 100
    d5 = (bm[1] - am[1]) * 100

    lines = [
        "=" * 78,
        "Phase 10b: multi-window query aggregation",
        f"Templates: 5-min pyin + expert tonic     Queries: 3 x 60s averaged",
        f"Recordings: {int(keep.sum())}/480",
        "=" * 78,
        "",
        f"  Phase 10a single 60s query             top1 {am[0]*100:5.2f}% +/- {as_[0]*100:.2f}   top5 {am[1]*100:5.2f}% +/- {as_[1]*100:.2f}",
        f"  Phase 10b 3-window avg query           top1 {bm[0]*100:5.2f}% +/- {bs[0]*100:.2f}   top5 {bm[1]*100:5.2f}% +/- {bs[1]*100:.2f}",
        f"  Delta vs Phase 10a                     +{d1:5.2f} pp top-1     +{d5:5.2f} pp top-5",
        "",
        "Comparison ladder:",
        "  Phase 1   expert pitch + expert tonic, full recording        top1 85.67%   top5 97.58%",
        f"  Phase 10b pyin   + expert tonic, 5min template + 3x60s avg   top1 {bm[0]*100:5.2f}%   top5 {bm[1]*100:5.2f}%",
        "  Phase 10a pyin   + expert tonic, 5min template + 60s query   top1 55.10%   top5 88.74%",
        "  Phase 9b  CREPE  + expert tonic, 60s template + 60s query    top1 51.51%   top5 87.36%",
        "  Phase 9a  pyin   + expert tonic, 60s template + 60s query    top1 45.81%   top5 82.93%",
        "  Phase 8   pyin   + heuristic tonic, 60s + 60s                 top1 37.36%   top5 67.91%",
        "  v1 deployed audio                                              top1  8.32%   top5 28.57%",
    ]
    write_report("data/eval_multiwindow_queries_report.txt", lines)
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
