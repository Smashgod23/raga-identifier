"""
Diagnostic: audio TDMS with CREPE pitch + EXPERT tonic.

Phase 2 step 2 found that switching from heuristic to expert tonic only
recovers +8.5 pp top-1 (37.36% -> 45.81%), leaving a 40-pp gap to the
expert-pitch ceiling. That implicates pyin's pitch quality as the bigger
bottleneck. The SPD paper (AIMC 2023, current SOTA on CMD at 88.13%)
explicitly recommends CREPE for runtime inference; this script tests
whether switching pyin -> CREPE closes the gap.

Setup: same 60s middle window, same expert tonic, same TDMS algorithm.
Only the pitch extractor changes.

TensorFlow inside CREPE doesn't fork cleanly, so the script runs
sequentially. CREPE with model_capacity='small' processes a 60s clip
in roughly 4-8 seconds on M-series, so 480 recordings take ~30-60
minutes single-threaded.

Outputs:
  data/X_tdms_audio_crepe.npy
  data/eval_crepe_audio_report.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from build_tdms import compute_tdms
from eval_audio_ab import audio_path_for, make_tdms_knn_factory, CLIP_DURATION
from eval_audio_expert_tonic import find_tonic_fine
from eval_harness import recording_level_cv, load_classes, write_report


def extract_crepe_tdms(
    audio_path: str,
    expert_tonic: float,
    *,
    offset: float = 0.0,
    duration: float = 60.0,
    model_capacity: str = "small",
    step_ms: float = 30.0,
    confidence_threshold: float = 0.5,
) -> Optional[np.ndarray]:
    """Extract pitch via CREPE, normalize by expert tonic, build TDMS."""
    import librosa
    import crepe

    y, sr = librosa.load(audio_path, sr=16000, mono=True, offset=offset, duration=duration)
    if len(y) < sr * 5:  # need at least 5 seconds
        return None

    # RMS normalize so the loud-quiet variance doesn't shift confidence
    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    if rms > 1e-6:
        y = y * (0.1 / rms)

    # CREPE expects 16 kHz mono float audio in [-1, 1].
    t, f0, confidence, _ = crepe.predict(
        y, sr,
        viterbi=True,
        step_size=int(step_ms),
        verbose=0,
        model_capacity=model_capacity,
    )
    if len(f0) < 30:
        return None

    # Mask unvoiced (low confidence) frames as 0.0 so compute_tdms ignores
    # them. The default confidence threshold for CREPE is ~0.5.
    voiced_mask = confidence > confidence_threshold
    if voiced_mask.sum() < 30:
        # Fall back to a looser threshold rather than failing the recording.
        voiced_mask = confidence > 0.3
    if voiced_mask.sum() < 30:
        return None

    pitches_full = np.where(voiced_mask, f0, 0.0)

    # Fold expert tonic into the singer's median octave (mirrors what
    # eval_audio_expert_tonic.py does).
    voiced_pitches = f0[voiced_mask]
    median_pitch = float(np.median(voiced_pitches))
    tonic = float(expert_tonic)
    while tonic * 2 <= median_pitch * 1.5:
        tonic *= 2
    while tonic >= median_pitch * 1.5:
        tonic /= 2

    timestamps = t.astype(np.float64)  # seconds
    tdms = compute_tdms(timestamps, pitches_full, tonic)
    return tdms.ravel() if tdms is not None else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small", choices=["tiny", "small", "medium", "large", "full"])
    parser.add_argument("--limit", type=int, default=None, help="Process only first N recordings")
    args = parser.parse_args()

    classes = load_classes()
    n_classes = len(classes)

    with open("data/tdms_meta.json") as f:
        meta = json.load(f)
    y_full = np.load("data/y_tdms.npy")
    if args.limit:
        meta = meta[: args.limit]
        y_full = y_full[: args.limit]
    print(f"Processing {len(meta)} recordings with CREPE model='{args.model}'")

    X = np.zeros((len(meta), 14400), dtype=np.float32)
    keep = np.ones(len(meta), dtype=bool)

    t0 = time.time()
    for i, m in enumerate(meta):
        tonic = find_tonic_fine(m["recording_id"])
        if tonic is None:
            keep[i] = False
            continue
        path = audio_path_for(m["recording_id"])
        if not os.path.exists(path):
            keep[i] = False
            continue
        try:
            import librosa as _librosa
            total = _librosa.get_duration(path=path)
            offset = max(0.0, (total - CLIP_DURATION) / 2.0)
            duration = min(CLIP_DURATION, total)
            row = extract_crepe_tdms(path, tonic, offset=offset, duration=duration, model_capacity=args.model)
            if row is None:
                keep[i] = False
                continue
            X[i] = row
        except Exception as exc:
            keep[i] = False
            print(f"  [{i + 1}/{len(meta)}] ERROR: {exc}")
            continue

        if (i + 1) % 10 == 0 or i == len(meta) - 1:
            rate = (i + 1) / max(1e-6, time.time() - t0)
            eta = (len(meta) - i - 1) / max(1e-6, rate)
            print(
                f"  [{i + 1}/{len(meta)}] {rate:.2f} rec/s   ETA {eta / 60:.1f} min   "
                f"kept {int(keep[: i + 1].sum())}"
            )

    np.save("data/X_tdms_audio_crepe.npy", X)
    print(f"\nSaved data/X_tdms_audio_crepe.npy; kept {int(keep.sum())}/{len(meta)}")

    X = X[keep]
    y = y_full[keep]
    if len(X) < 100:
        print("Too few recordings for a meaningful eval.")
        return

    print("\nRunning 5-fold CV with kNN (sym KL)...")
    cv = recording_level_cv(X, y, make_tdms_knn_factory(n_classes), verbose=False)
    m_, s_ = cv["top_k_mean"], cv["top_k_std"]

    lines = [
        "=" * 78,
        "CREPE pitch + EXPERT tonic (Phase 2 step 3: pitch-extractor diagnostic)",
        f"Pitch: CREPE model={args.model} step=30ms, Tonic: .tonicFine (expert)",
        f"Window: 60s middle, Recordings: {int(keep.sum())}/{len(meta)}",
        "=" * 78,
        "",
        f"  TDMS k-NN (sym KL), CREPE + expert tonic       top1 {m_[0]*100:5.2f}% +/- {s_[0]*100:.2f}   top5 {m_[1]*100:5.2f}% +/- {s_[1]*100:.2f}",
        "",
        "Comparison ladder:",
        "  Phase 1 expert (.pitch + .tonicFine, full rec)   top1 85.67% +/- 0.40   top5 97.58%",
        f"  Phase 2 step 3: CREPE + expert tonic, 60s        top1 {m_[0]*100:5.2f}% +/- {s_[0]*100:.2f}   top5 {m_[1]*100:5.2f}%",
        "  Phase 2 step 2: pyin + expert tonic, 60s         top1 45.81% +/- 1.55   top5 82.93%",
        "  Phase 2 step 1: pyin + heuristic tonic, 60s      top1 37.36% +/- 1.00   top5 67.91%",
        "  v1 audio (deployed reality)                      top1  8.32% +/- 0.93   top5 28.57%",
        "",
        "Interpretation:",
        "  - If CREPE+expert_tonic >= 70%, pitch extractor was the bottleneck.",
        "  - If CREPE+expert_tonic ~ 50-65%, both pitch and window length contribute.",
        "  - If CREPE+expert_tonic ~ 45-50%, window length / sparsity dominates.",
    ]
    write_report("data/eval_crepe_audio_report.txt", lines)
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
