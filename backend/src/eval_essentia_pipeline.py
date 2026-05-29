"""
Phase 13b: full Essentia pipeline — the expert extractors on real audio.

Essentia's TonicIndianArtMusic hit 85.4% tonic (Phase 13) vs sa_pa's
70.2%, and PredominantPitchMelodia is the exact algorithm that produced
the CompMusic .pitch files. Running BOTH on raw audio replicates the
expert feature pipeline (Melodia + Gulati tonic) the 86.7% papers used,
but on arbitrary uploads with no dataset annotations.

This script rebuilds templates AND queries with Melodia pitch so the two
sides share a pitch extractor (Phase 8 showed cross-extractor mismatch is
catastrophic):
  Templates: Melodia pitch + EXPERT tonic (.tonicFine, offline-available),
             5-minute middle window.
  Queries:   Melodia pitch + ESSENTIA tonic (runtime, no labels),
             three 60s windows averaged.
  Classifier: 1-NN symmetric KL, asymmetric 5-fold recording-aware CV.

Compare to:
  Phase 1   expert .pitch + expert tonic (full rec)   85.67% / 97.58%
  Phase 12d pyin + sa_pa tonic                          42.25% / 70.42%
  Phase 13b Melodia + Essentia tonic (this)            ?

Outputs:
  data/X_tdms_ess_template.npy
  data/X_tdms_ess_query.npy
  data/eval_essentia_pipeline_report.txt
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
from eval_essentia_tonic import find_tonic_fine
from eval_harness import load_classes, write_report
from eval_multiwindow_queries import asymmetric_cv

SR = 44100
TEMPLATE_DUR = 300.0
QUERY_DUR = 60.0
WINDOW_FRACTIONS = (0.25, 0.50, 0.75)


def _melodia_tdms(audio, sr, tonic):
    """audio (float32 @ 44.1k) -> TDMS using Melodia pitch + given tonic."""
    import essentia.standard as es
    pitch, _ = es.PredominantPitchMelodia(frameSize=2048, hopSize=128)(audio)
    # Melodia hop is 128/44100 s. Build a timestamp axis for compute_tdms.
    hop_s = 128.0 / sr
    timestamps = np.arange(len(pitch)) * hop_s
    pitches = np.where(pitch > 0, pitch, 0.0).astype(np.float64)
    if np.count_nonzero(pitches) < 30:
        return None
    tdms = compute_tdms(timestamps, pitches, float(tonic))
    return tdms.ravel() if tdms is not None else None


def extract_template(args):
    idx, recording_id, expert_tonic = args
    import warnings; warnings.filterwarnings("ignore")
    import essentia.standard as es
    if expert_tonic is None or expert_tonic <= 0:
        return {"idx": idx, "error": "no expert tonic"}
    path = audio_path_for(recording_id)
    if not os.path.exists(path):
        return {"idx": idx, "error": "missing audio"}
    try:
        audio = es.MonoLoader(filename=path, sampleRate=SR)()
        if len(audio) > TEMPLATE_DUR * SR:
            mid = len(audio) // 2
            half = int(TEMPLATE_DUR * SR // 2)
            audio = audio[mid - half: mid + half]
        row = _melodia_tdms(audio, SR, expert_tonic)
    except Exception as exc:
        return {"idx": idx, "error": str(exc)[:80]}
    if row is None:
        return {"idx": idx, "error": "tdms none"}
    return {"idx": idx, "tdms": row}


def extract_query(args):
    idx, recording_id, _ = args
    import warnings; warnings.filterwarnings("ignore")
    import essentia.standard as es
    path = audio_path_for(recording_id)
    if not os.path.exists(path):
        return {"idx": idx, "error": "missing audio"}
    try:
        full = es.MonoLoader(filename=path, sampleRate=SR)()
        total = len(full)
        tdmss = []
        for frac in WINDOW_FRACTIONS:
            center = int(total * frac)
            half = int(QUERY_DUR * SR // 2)
            lo = max(0, min(center - half, total - int(QUERY_DUR * SR)))
            hi = min(total, lo + int(QUERY_DUR * SR))
            chunk = full[lo:hi]
            if len(chunk) < SR * 5:
                continue
            # Essentia tonic per window (runtime — no expert label).
            tonic = float(es.TonicIndianArtMusic()(chunk))
            if tonic <= 0:
                continue
            row = _melodia_tdms(chunk, SR, tonic)
            if row is not None:
                tdmss.append(row)
        if not tdmss:
            return {"idx": idx, "error": "no windows"}
        avg = np.mean(np.stack(tdmss), axis=0)
        avg /= max(float(avg.sum()), 1e-12)
        return {"idx": idx, "tdms": avg.astype(np.float32)}
    except Exception as exc:
        return {"idx": idx, "error": str(exc)[:80]}


def run_extract(fn, jobs, label, n_workers, dim=14400):
    out = np.zeros((len(jobs), dim), dtype=np.float32)
    t0 = time.time(); done = 0; errors = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        for r in pool.map(fn, jobs, chunksize=1):
            done += 1
            if "error" in r:
                errors += 1
                if errors <= 5:
                    print(f"  {label} err idx={r['idx']}: {r['error']}")
            else:
                out[r["idx"]] = r["tdms"]
            if done % 50 == 0 or done == len(jobs):
                rate = done / max(1e-6, time.time() - t0)
                print(f"  {label} [{done}/{len(jobs)}] {rate:.2f} rec/s  ETA {(len(jobs)-done)/max(1e-6,rate)/60:.1f} min")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    classes = load_classes()
    n_classes = len(classes)
    with open("data/tdms_meta.json") as f:
        meta = json.load(f)
    y_full = np.load("data/y_tdms.npy")
    jobs = [(i, m["recording_id"], find_tonic_fine(m["recording_id"])) for i, m in enumerate(meta)]

    tpath, qpath = "data/X_tdms_ess_template.npy", "data/X_tdms_ess_query.npy"
    if args.skip_extract and os.path.exists(tpath) and os.path.exists(qpath):
        X_t, X_q = np.load(tpath), np.load(qpath)
    else:
        print("Extracting Essentia Melodia templates (5-min + expert tonic)...")
        X_t = run_extract(extract_template, jobs, "tmpl", args.workers)
        np.save(tpath, X_t)
        print("Extracting Essentia Melodia queries (3x60s + Essentia tonic)...")
        X_q = run_extract(extract_query, jobs, "qry", args.workers)
        np.save(qpath, X_q)

    keep = (~np.all(X_t == 0, axis=1)) & (~np.all(X_q == 0, axis=1))
    X_t, X_q, y = X_t[keep], X_q[keep], y_full[keep]
    print(f"\nKept {int(keep.sum())}/480\nRunning asymmetric 5-fold CV...")
    res = asymmetric_cv(X_t, X_q, y, n_classes)
    m, s = res["top_k_mean"], res["top_k_std"]

    lines = [
        "=" * 78,
        "Phase 13b: full Essentia pipeline (Melodia pitch + Gulati tonic)",
        f"Templates: Melodia 5-min + EXPERT tonic   Queries: Melodia 3x60s + ESSENTIA tonic",
        f"Recordings: {int(keep.sum())}/480",
        "=" * 78,
        "",
        f"  Phase 13b (Essentia full)     top1 {m[0]*100:5.2f}% +/- {s[0]*100:.2f}   top5 {m[1]*100:5.2f}% +/- {s[1]*100:.2f}",
        "",
        "Comparison ladder:",
        "  Phase 1   expert .pitch + expert tonic   top1 85.67%   top5 97.58%",
        f"  Phase 13b Melodia + Essentia tonic       top1 {m[0]*100:5.2f}%   top5 {m[1]*100:5.2f}%",
        "  Phase 12d pyin + sa_pa tonic             top1 42.25%   top5 70.42%",
        "  v1 deployed                              top1  8.32%   top5 28.57%",
    ]
    write_report("data/eval_essentia_pipeline_report.txt", lines)
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
