"""
Phase 12c: end-to-end raga eval with the sa_pa tonic detector.

Phase 12b showed the sa_pa drone scorer lifts tonic top-1 from 51.9% to
65.0% (K=10). This script confirms whether that tonic gain translates to
raga accuracy in the full multi-window TDMS pipeline. It is identical to
eval_multiwindow_heuristic.py (Phase 11a) except _detect_tonic now uses
its new defaults (scorer="sa_pa", k=10).

Compare against:
  Phase 10b (expert tonic, perfect):                      73.23% / 95.31%
  Phase 11a (old heuristic tonic, peakedness K=5):        35.83% / 63.67%
  Phase 12c (this, sa_pa tonic K=10):                      ?

Outputs:
  data/X_tdms_audio_multiwindow_sapa.npy
  data/eval_multiwindow_sapa_report.txt
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
from eval_audio_ab import audio_path_for
from eval_harness import load_classes, write_report
from eval_multiwindow_queries import asymmetric_cv

QUERY_DURATION = 60.0
WINDOW_FRACTIONS = (0.25, 0.50, 0.75)


def extract_one(args: tuple) -> dict:
    idx, recording_id, label = args
    import librosa
    from predict import _detect_tonic  # new defaults: sa_pa, k=10

    path = audio_path_for(recording_id)
    if not os.path.exists(path):
        return {"idx": idx, "error": "missing audio"}
    try:
        total = librosa.get_duration(path=path)
    except Exception as exc:
        return {"idx": idx, "error": str(exc)}

    tdmss = []
    for frac in WINDOW_FRACTIONS:
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
        f0, vflag, vprob = librosa.pyin(y, fmin=60, fmax=800, sr=sr, frame_length=1024, hop_length=256)
        voiced = f0[vflag & (vprob > 0.3)]
        if len(voiced) < 30:
            voiced = f0[vflag]
        voiced = voiced[~np.isnan(voiced)]
        if len(voiced) < 30:
            continue
        tonic = _detect_tonic(voiced)
        if tonic is None or tonic <= 0:
            continue
        hop_seconds = 256.0 / sr
        pitches_full = np.where(vflag, f0, 0.0)
        pitches_full = np.where(np.isfinite(pitches_full), pitches_full, 0.0)
        timestamps = np.arange(len(pitches_full)) * hop_seconds
        tdms = compute_tdms(timestamps, pitches_full, tonic)
        if tdms is not None:
            tdmss.append(tdms.ravel())

    if not tdmss:
        return {"idx": idx, "error": "no usable windows"}
    avg = np.mean(np.stack(tdmss, axis=0), axis=0)
    avg /= max(float(avg.sum()), 1e-12)
    return {"idx": idx, "label": label, "tdms": avg.astype(np.float32)}


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

    out_path = "data/X_tdms_audio_multiwindow_sapa.npy"
    if args.skip_extract and os.path.exists(out_path):
        X_query = np.load(out_path)
    else:
        jobs = [(i, m["recording_id"], int(y_full[i])) for i, m in enumerate(meta)]
        print(f"Extracting 3x60s queries with sa_pa tonic, {args.workers} workers")
        X_query = np.zeros((len(jobs), 14400), dtype=np.float32)
        t0 = time.time()
        done = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for r in pool.map(extract_one, jobs, chunksize=1):
                done += 1
                if "error" not in r:
                    X_query[r["idx"]] = r["tdms"]
                if done % 25 == 0 or done == len(jobs):
                    rate = done / max(1e-6, time.time() - t0)
                    print(f"  [{done}/{len(jobs)}] {rate:.2f} rec/s  ETA {(len(jobs)-done)/max(1e-6,rate)/60:.1f} min")
        np.save(out_path, X_query)

    X_template = np.load("data/X_tdms_long.npy")
    keep = (~np.all(X_template == 0, axis=1)) & (~np.all(X_query == 0, axis=1))
    X_template, X_query, y = X_template[keep], X_query[keep], y_full[keep]
    print(f"\nKept {int(keep.sum())}/480\nRunning asymmetric 5-fold CV...")
    res = asymmetric_cv(X_template, X_query, y, n_classes)
    m, s = res["top_k_mean"], res["top_k_std"]

    lines = [
        "=" * 78,
        "Phase 12d: end-to-end raga eval with exact-bin sa_pa tonic (K=15)",
        f"Templates: 5-min pyin + EXPERT tonic   Queries: 3x60s avg, pyin + sa_pa exact K=15",
        f"Recordings: {int(keep.sum())}/480",
        "=" * 78,
        "",
        f"  Phase 12d (sa_pa exact, K=15)  top1 {m[0]*100:5.2f}% +/- {s[0]*100:.2f}   top5 {m[1]*100:5.2f}% +/- {s[1]*100:.2f}",
        "",
        "Comparison ladder:",
        "  Phase 10b  expert tonic              top1 73.23%   top5 95.31%",
        f"  Phase 12d  sa_pa exact (K=15)        top1 {m[0]*100:5.2f}%   top5 {m[1]*100:5.2f}%",
        "  Phase 12c  sa_pa +/-1bin (K=10)      top1 41.29%   top5 69.58%",
        "  Phase 11a  peakedness (K=5)          top1 35.83%   top5 63.67%",
        "  v1 deployed                          top1  8.32%   top5 28.57%",
    ]
    write_report("data/eval_multiwindow_sapa_report.txt", lines)
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
