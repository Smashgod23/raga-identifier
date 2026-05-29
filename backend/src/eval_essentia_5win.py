"""
Phase 13d: 5-window queries against the cached full-recording templates.

Phase 13c hit 70.00% with 3x60s queries vs full-recording Melodia
templates. The query side is the remaining limiter (180s of query audio
against dense full-recording templates). Going 1->3 windows was +18 pp
(Phase 10b), so denser queries are the cheapest lever left.

This reuses the cached full-recording templates from Phase 13c
(X_tdms_essfull_template.npy) and only re-extracts queries with FIVE 60s
windows at 10/30/50/70/90% through the recording, one shared Essentia
tonic detected on a 180s middle window.

Outputs:
  data/X_tdms_essfull_query5.npy
  data/eval_essentia_5win_report.txt
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

SR = 44100
TONIC_WIN = 180.0
# Overridable via CLI (--nwin, --wdur) for the Phase 13e density push.
QUERY_DUR = 60.0
WINDOW_FRACTIONS = (0.10, 0.30, 0.50, 0.70, 0.90)


def _melodia_tdms(audio, tonic):
    import essentia.standard as es
    pitch, _ = es.PredominantPitchMelodia(frameSize=2048, hopSize=128)(audio)
    timestamps = np.arange(len(pitch)) * (128.0 / SR)
    pitches = np.where(pitch > 0, pitch, 0.0).astype(np.float64)
    if np.count_nonzero(pitches) < 30:
        return None
    tdms = compute_tdms(timestamps, pitches, float(tonic))
    return tdms.ravel() if tdms is not None else None


def extract_query(args):
    idx, recording_id = args
    import warnings; warnings.filterwarnings("ignore")
    import essentia.standard as es
    path = audio_path_for(recording_id)
    if not os.path.exists(path):
        return {"idx": idx, "error": "missing audio"}
    try:
        full = es.MonoLoader(filename=path, sampleRate=SR)()
        total = len(full)
        half_t = int(TONIC_WIN * SR // 2)
        mid = total // 2
        tchunk = full[max(0, mid - half_t): mid + half_t] if total > TONIC_WIN * SR else full
        tonic = float(es.TonicIndianArtMusic()(tchunk))
        if tonic <= 0:
            return {"idx": idx, "error": "tonic<=0"}
        wlen = int(QUERY_DUR * SR)
        tdmss = []
        for frac in WINDOW_FRACTIONS:
            center = int(total * frac)
            lo = max(0, min(center - wlen // 2, total - wlen))
            chunk = full[lo: lo + wlen]
            if len(chunk) < SR * 5:
                continue
            row = _melodia_tdms(chunk, tonic)
            if row is not None:
                tdmss.append(row)
        if not tdmss:
            return {"idx": idx, "error": "no windows"}
        avg = np.mean(np.stack(tdmss), axis=0)
        avg /= max(float(avg.sum()), 1e-12)
        return {"idx": idx, "tdms": avg.astype(np.float32)}
    except Exception as exc:
        return {"idx": idx, "error": str(exc)[:80]}


def main():
    global QUERY_DUR, WINDOW_FRACTIONS
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--nwin", type=int, default=5)
    parser.add_argument("--wdur", type=float, default=60.0)
    parser.add_argument("--tag", default="query5")
    args = parser.parse_args()
    QUERY_DUR = args.wdur
    # Evenly spaced window centers across the recording.
    WINDOW_FRACTIONS = tuple(
        (i + 1) / (args.nwin + 1) for i in range(args.nwin)
    )

    classes = load_classes(); n_classes = len(classes)
    with open("data/tdms_meta.json") as f:
        meta = json.load(f)
    y_full = np.load("data/y_tdms.npy")
    X_t = np.load("data/X_tdms_essfull_template.npy")  # cached Phase 13c templates

    jobs = [(i, m["recording_id"]) for i, m in enumerate(meta)]
    qpath = f"data/X_tdms_essfull_{args.tag}.npy"
    print(f"Extracting {args.nwin}x{args.wdur:.0f}s Melodia queries + shared Essentia tonic...")
    X_q = np.zeros((len(jobs), 14400), dtype=np.float32)
    t0 = time.time(); done = errors = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(extract_query, jobs, chunksize=1):
            done += 1
            if "error" in r:
                errors += 1
            else:
                X_q[r["idx"]] = r["tdms"]
            if done % 50 == 0 or done == len(jobs):
                rate = done / max(1e-6, time.time() - t0)
                print(f"  [{done}/{len(jobs)}] {rate:.2f} rec/s  ETA {(len(jobs)-done)/max(1e-6,rate)/60:.1f} min  ({errors} err)")
    np.save(qpath, X_q)

    keep = (~np.all(X_t == 0, axis=1)) & (~np.all(X_q == 0, axis=1))
    X_t2, X_q2, y = X_t[keep], X_q[keep], y_full[keep]
    print(f"\nKept {int(keep.sum())}/480\nRunning asymmetric 5-fold CV...")
    res = asymmetric_cv(X_t2, X_q2, y, n_classes)
    m, s = res["top_k_mean"], res["top_k_std"]

    lines = [
        "=" * 78,
        f"Phase 13 query-density push: full-rec Melodia templates + {args.nwin}x{args.wdur:.0f}s queries",
        f"Queries: {args.nwin}x{args.wdur:.0f}s evenly spaced, 1 shared Essentia tonic (180s)",
        f"Recordings: {int(keep.sum())}/480",
        "=" * 78,
        "",
        f"  {args.nwin}x{args.wdur:.0f}s query                   top1 {m[0]*100:5.2f}% +/- {s[0]*100:.2f}   top5 {m[1]*100:5.2f}% +/- {s[1]*100:.2f}",
        "",
        "Comparison ladder:",
        "  Phase 1   expert .pitch + expert tonic    top1 85.67%   top5 97.58%",
        f"  THIS      Melodia full + {args.nwin}x{args.wdur:.0f}s query       top1 {m[0]*100:5.2f}%   top5 {m[1]*100:5.2f}%",
        "  Phase 13d Melodia full + 5x60s query      top1 73.62%   top5 91.21%",
        "  Phase 13c Melodia full + 3x60s query      top1 70.00%   top5 86.58%",
        "  Phase 12d pyin + sa_pa tonic              top1 42.25%   top5 70.42%",
        "  v1 deployed                               top1  8.32%   top5 28.57%",
    ]
    write_report(f"data/eval_essentia_{args.tag}_report.txt", lines)
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
