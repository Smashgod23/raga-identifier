"""
Diagnostic: audio TDMS with EXPERT tonic (from .tonicFine), pyin pitch.

Phase 2 step 1 found the runtime pipeline at 37.36% top-1 vs the expert
pipeline at 85.67%. The gap could come from pitch quality (pyin vs the
Salamon-Gomez predominant-melody method CompMusic used) or tonic
quality (the predict.py heuristic vs the hand-verified .tonicFine).

This script isolates the two by rerunning the audio TDMS extraction with
the EXPERT tonic baked in. Same 60s window, same pyin pitch, same TDMS
algorithm. The only thing that changes vs eval_audio_ab.py is the tonic.

If accuracy jumps back close to 86%, the tonic detector is the bottleneck
and wiring tonic_detector_v1.pt is the cheapest win. If it stays low, the
pitch extractor (pyin) is the bottleneck and we need crepe or basic-pitch.

Outputs:
  data/X_tdms_audio_experttonic.npy
  data/eval_audio_expert_tonic_report.txt
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
from eval_harness import recording_level_cv, load_classes, write_report
from eval_audio_ab import make_tdms_knn_factory, audio_path_for, CLIP_DURATION


def find_tonic_fine(recording_id: str) -> float | None:
    """Look up the expert .tonicFine value for a recording.

    recording_id has the form <raga_id>/<artist>/<album>/<piece> matching
    both the audio dir and the features dir.
    """
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


def extract_one(args: tuple) -> dict:
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

    offset = max(0.0, (total - CLIP_DURATION) / 2.0)
    duration = min(CLIP_DURATION, total)

    try:
        y, sr = librosa.load(path, sr=16000, mono=True, offset=offset, duration=duration)
    except Exception as exc:
        return {"idx": idx, "error": f"load failed: {exc}"}

    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    if rms > 1e-6:
        y = y * (0.1 / rms)

    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=60, fmax=800, sr=sr, frame_length=1024, hop_length=256
    )

    if voiced_flag.sum() < 30:
        return {"idx": idx, "error": "too little voiced pitch"}

    hop_seconds = 256.0 / sr
    pitches_full = np.where(voiced_flag, f0, 0.0)
    pitches_full = np.where(np.isfinite(pitches_full), pitches_full, 0.0)
    timestamps = np.arange(len(pitches_full)) * hop_seconds

    # The expert tonic might be in a different octave than the singer's
    # median; fold it into the singer's range to keep cents math sensible.
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
        "tdms": tdms.ravel(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    classes = load_classes()
    n_classes = len(classes)

    with open("data/tdms_meta.json") as f:
        meta = json.load(f)
    y_full = np.load("data/y_tdms.npy")
    assert len(meta) == len(y_full) == 480

    jobs = []
    missing_tonic = 0
    for i, m in enumerate(meta):
        t = find_tonic_fine(m["recording_id"])
        if t is None:
            missing_tonic += 1
        jobs.append((i, m["recording_id"], int(y_full[i]), t))
    print(f"Prepared {len(jobs)} jobs ({missing_tonic} missing expert tonic)")

    n = len(jobs)
    X = np.zeros((n, 14400), dtype=np.float32)
    keep = np.ones(n, dtype=bool)
    t0 = time.time()
    done = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(extract_one, jobs, chunksize=1):
            done += 1
            if "error" in result:
                keep[result["idx"]] = False
                if done <= 5 or done % 50 == 0:
                    print(f"  [{done}/{n}] ERROR idx={result['idx']}: {result['error']}")
                continue
            X[result["idx"]] = result["tdms"]
            if done % 25 == 0 or done == n:
                rate = done / max(1e-6, time.time() - t0)
                eta = (n - done) / max(1e-6, rate)
                print(f"  [{done}/{n}] {rate:.2f} rec/s   ETA {eta / 60:.1f} min")

    np.save("data/X_tdms_audio_experttonic.npy", X)
    print(f"\nSaved data/X_tdms_audio_experttonic.npy; kept {keep.sum()}/{n}")

    X = X[keep]
    y = y_full[keep]

    print("\nRunning 5-fold CV with kNN (sym KL)...")
    cv = recording_level_cv(X, y, make_tdms_knn_factory(n_classes), verbose=False)
    m, s = cv["top_k_mean"], cv["top_k_std"]

    lines = [
        "=" * 78,
        "Audio TDMS with EXPERT tonic (Phase 2 step 2: bottleneck isolation)",
        f"Pitch: pyin (runtime), Tonic: .tonicFine (expert), Window: 60s middle",
        f"Recordings: {int(keep.sum())}/{n}",
        "=" * 78,
        "",
        f"  TDMS k-NN (sym KL), audio + expert tonic     top1 {m[0]*100:5.2f}% +/- {s[0]*100:.2f}   top5 {m[1]*100:5.2f}% +/- {s[1]*100:.2f}",
        "",
        "Comparison:",
        "  Phase 2 step 1 (audio + heuristic tonic)     top1 37.36% +/- 1.00   top5 67.91% +/- 0.81",
        "  Phase 1 expert templates                     top1 85.67% +/- 0.40   top5 97.58% +/- 0.21",
        "  v1 audio (production reality)                top1  8.32% +/- 0.93   top5 28.57% +/- 1.60",
        "",
        "Interpretation:",
        "  - If audio+expert_tonic >= 70%, tonic is the dominant bottleneck.",
        "  - If audio+expert_tonic in 40-60%, both tonic and pitch contribute.",
        "  - If audio+expert_tonic <= 45%, pitch (pyin) is the bottleneck.",
    ]
    write_report("data/eval_audio_expert_tonic_report.txt", lines)
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    # macOS defaults to "spawn" which forces every worker to re-import the
    # whole module tree and intermittently hangs on librosa. Force "fork" so
    # workers inherit the parent's loaded modules directly.
    mp.set_start_method("fork", force=True)
    main()
