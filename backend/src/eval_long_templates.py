"""
Phase 10a: long-template TDMS, short queries.

Phase 9 found the last unexplained piece of the audio-pipeline gap is
window length: the 86.67% expert ceiling uses full-recording pitch
contours (~30 minutes each), but Phase 8-9's audio benchmarks all used
60s middle windows for both train and test. This script tests the
"asymmetric template" hypothesis: build TDMS templates from a longer
window of each recording (5 minutes around the midpoint), then query
with the same 60s middle TDMSs Phase 9a produced.

The 5-fold split is at the recording level, so training never sees the
held-out recording's audio in any form. Within a recording the
template and the query do overlap by 60s, but recordings are
recording-aware split, so the model never sees that overlap during
training.

Setup:
  Templates: 5-minute window centered at midpoint of each recording,
             pyin pitch, EXPERT tonic from .tonicFine, TDMS.
  Queries:   60s window centered at midpoint, pyin pitch, EXPERT tonic.
             Reuses data/X_tdms_audio_experttonic.npy from Phase 9a.
  Classifier: 1-NN with symmetric KL.
  Eval: 5-fold stratified CV at the recording level, 5 seeds.

Comparison points:
  Phase 7 ceiling (expert pitch + expert tonic, full recording): 85.67%
  Phase 9b (CREPE + expert tonic, 60s):                          51.51%
  Phase 9a (pyin + expert tonic, 60s):                           45.81%
  Phase 10a (pyin + expert tonic, 5min template + 60s query):     ?
  Phase 8 (pyin + heuristic tonic, 60s):                         37.36%

If Phase 10a clears Phase 9a by a lot, the window-length hypothesis is
confirmed and the deployment recipe is: build long templates once on
HF, accept short user queries, do 1-NN with symmetric KL.

Outputs:
  data/X_tdms_long.npy
  data/eval_long_templates_report.txt
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

TEMPLATE_DURATION = 300.0  # 5 minutes


def extract_long_template(args: tuple) -> dict:
    """Extract a TDMS from a 5-minute window centered at midpoint."""
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

    duration = min(TEMPLATE_DURATION, total)
    offset = max(0.0, (total - duration) / 2.0)

    try:
        y, sr = librosa.load(path, sr=16000, mono=True, offset=offset, duration=duration)
    except Exception as exc:
        return {"idx": idx, "error": f"load failed: {exc}"}

    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    if rms > 1e-6:
        y = y * (0.1 / rms)

    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=60, fmax=800, sr=sr, frame_length=1024, hop_length=256
    )
    if voiced_flag.sum() < 30:
        return {"idx": idx, "error": "too little voiced pitch"}

    hop_seconds = 256.0 / sr
    pitches_full = np.where(voiced_flag, f0, 0.0)
    pitches_full = np.where(np.isfinite(pitches_full), pitches_full, 0.0)
    timestamps = np.arange(len(pitches_full)) * hop_seconds

    voiced_pitches = f0[voiced_flag]
    median_pitch = float(np.median(voiced_pitches))
    tonic = float(expert_tonic)
    while tonic * 2 <= median_pitch * 1.5:
        tonic *= 2
    while tonic >= median_pitch * 1.5:
        tonic /= 2

    tdms = compute_tdms(timestamps, pitches_full, tonic)
    if tdms is None:
        return {"idx": idx, "error": "tdms compute returned None"}

    return {
        "idx": idx,
        "recording_id": recording_id,
        "label": label,
        "tonic_hz": tonic,
        "duration_used": duration,
        "tdms": tdms.ravel(),
    }


# ---------------------------------------------------------------------------
# Asymmetric template eval: train on one source (long), test on another (short).
# Aligned 1:1 by recording. Recording-aware k-fold.
# ---------------------------------------------------------------------------

def asymmetric_cv(
    X_template: np.ndarray,
    X_query: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    *,
    n_splits: int = 5,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
):
    from sklearn.model_selection import StratifiedKFold

    top_k = (1, 5)
    per_fold = np.zeros((len(seeds), n_splits, len(top_k)), dtype=np.float64)
    n_correct_top1_total = 0
    n_total = 0

    for s_idx, seed in enumerate(seeds):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for f_idx, (train_idx, test_idx) in enumerate(skf.split(X_query, y)):
            clf = TDMSKnnDirect(n_classes)
            # Train uses LONG templates of the training recordings.
            clf.fit(X_template[train_idx], y[train_idx])
            # Test uses SHORT queries of the held-out recordings.
            proba = clf.predict_proba(X_query[test_idx])
            for k_idx, k in enumerate(top_k):
                top = np.argsort(-proba, axis=1)[:, :k]
                hits = (top == y[test_idx][:, None]).any(axis=1)
                per_fold[s_idx, f_idx, k_idx] = hits.mean()
                if k == 1 and seed == seeds[0]:
                    n_correct_top1_total += hits.sum()
                    n_total += len(hits)

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

    # ---- Build templates ----------------------------------------------
    if args.skip_extract and os.path.exists("data/X_tdms_long.npy"):
        print("Loading cached templates from data/X_tdms_long.npy")
        X_template = np.load("data/X_tdms_long.npy")
    else:
        jobs = []
        for i, m in enumerate(meta):
            t = find_tonic_fine(m["recording_id"])
            jobs.append((i, m["recording_id"], int(y_full[i]), t))
        print(f"Extracting 5-min templates for {len(jobs)} recordings, {args.workers} workers")
        print(f"Expected runtime: ~{len(jobs) * 25.0 / args.workers / 60:.0f} min")

        X_template = np.zeros((len(jobs), 14400), dtype=np.float32)
        keep_t = np.ones(len(jobs), dtype=bool)
        t0 = time.time()
        done = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for r in pool.map(extract_long_template, jobs, chunksize=1):
                done += 1
                if "error" in r:
                    keep_t[r["idx"]] = False
                    if done <= 5 or done % 50 == 0:
                        print(f"  [{done}/{len(jobs)}] ERROR idx={r['idx']}: {r['error']}")
                    continue
                X_template[r["idx"]] = r["tdms"]
                if done % 25 == 0 or done == len(jobs):
                    rate = done / max(1e-6, time.time() - t0)
                    eta = (len(jobs) - done) / max(1e-6, rate)
                    print(f"  [{done}/{len(jobs)}] {rate:.2f} rec/s   ETA {eta / 60:.1f} min")

        np.save("data/X_tdms_long.npy", X_template)
        print(f"Saved data/X_tdms_long.npy; kept {int(keep_t.sum())}/{len(jobs)}")

    # ---- Load Phase 9a queries ----------------------------------------
    X_query = np.load("data/X_tdms_audio_experttonic.npy")
    print(f"\nTemplates: {X_template.shape}    Queries: {X_query.shape}")

    # Drop recordings missing from either side.
    template_ok = ~np.all(X_template == 0, axis=1)
    query_ok = ~np.all(X_query == 0, axis=1)
    keep = template_ok & query_ok
    X_template = X_template[keep]
    X_query = X_query[keep]
    y = y_full[keep]
    print(f"Both extractions OK: {int(keep.sum())}/480 recordings")

    # ---- Evaluate -----------------------------------------------------
    print("\n[1/2] Symmetric baseline: 60s template + 60s query (Phase 9a check)...")
    t0 = time.time()
    sym = asymmetric_cv(X_query, X_query, y, n_classes)
    print(f"    done in {time.time() - t0:.1f}s")

    print("\n[2/2] Asymmetric: 5-min template + 60s query (Phase 10a)...")
    t0 = time.time()
    asym = asymmetric_cv(X_template, X_query, y, n_classes)
    print(f"    done in {time.time() - t0:.1f}s")

    sm, ss = sym["top_k_mean"], sym["top_k_std"]
    am, as_ = asym["top_k_mean"], asym["top_k_std"]
    delta_top1 = (am[0] - sm[0]) * 100
    delta_top5 = (am[1] - sm[1]) * 100

    lines = [
        "=" * 78,
        "Phase 10a: long-template asymmetric eval",
        f"Templates: 5-min pyin + expert tonic   Queries: 60s pyin + expert tonic",
        f"Recordings: {int(keep.sum())}/480",
        "=" * 78,
        "",
        f"  Symmetric  (Phase 9a check, 60s+60s)   top1 {sm[0]*100:5.2f}% +/- {ss[0]*100:.2f}   top5 {sm[1]*100:5.2f}% +/- {ss[1]*100:.2f}",
        f"  Asymmetric (Phase 10a, 5min+60s)       top1 {am[0]*100:5.2f}% +/- {as_[0]*100:.2f}   top5 {am[1]*100:5.2f}% +/- {as_[1]*100:.2f}",
        f"  Delta vs symmetric                     +{delta_top1:5.2f} pp top-1     +{delta_top5:5.2f} pp top-5",
        "",
        "Comparison ladder:",
        "  Phase 1   expert pitch + expert tonic, full recording      top1 85.67%   top5 97.58%",
        "  Phase 9b  CREPE  + expert tonic, 60s template + 60s query  top1 51.51%   top5 87.36%",
        f"  Phase 10a pyin   + expert tonic, 5min template + 60s query top1 {am[0]*100:5.2f}%   top5 {am[1]*100:5.2f}%",
        "  Phase 9a  pyin   + expert tonic, 60s template + 60s query  top1 45.81%   top5 82.93%",
        "  Phase 8   pyin   + heuristic tonic, 60s + 60s               top1 37.36%   top5 67.91%",
        "  v1 deployed audio                                            top1  8.32%   top5 28.57%",
    ]
    write_report("data/eval_long_templates_report.txt", lines)
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
