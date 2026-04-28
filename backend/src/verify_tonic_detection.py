"""
Tonic detection diagnostic: compare predict.py's auto-detected Sa against
the expert-annotated .tonicFine values for 5 random Carnatic recordings.

Usage (from backend/):
  source venv/bin/activate
  python src/verify_tonic_detection.py

Read-only — writes nothing to data/ or models/.
"""

import json
import math
import os
import random
import sys

import librosa
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import _detect_tonic

AUDIO_DIR   = os.path.expanduser("~/raga-data-audio/RagaDataset/Carnatic/audio")
FEAT_DIR    = os.path.join(os.path.dirname(__file__), "..", "data",
                           "RagaDataset", "Carnatic", "features")
MAPPING_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                            "RagaDataset", "Carnatic", "_info_",
                            "ragaId_to_ragaName_mapping.json")

DIAGNOSTIC_DURATION = 90.0  # seconds of audio to load per recording


def tonicfine_path(audio_path):
    """Given an audio .mp3 path, return the parallel .tonicFine path in features/."""
    rel = os.path.relpath(audio_path, AUDIO_DIR)          # ragaId/artist/album/song/song.mp3
    rel_no_ext = os.path.splitext(rel)[0]                 # ragaId/artist/album/song/song
    return os.path.join(FEAT_DIR, rel_no_ext + ".tonicFine")


def load_expert_tonic(tonicfine_path):
    with open(tonicfine_path) as f:
        return float(f.read().strip())


def detect_tonic_from_audio(audio_path, duration):
    """Load `duration` seconds of audio and run the same pyin + _detect_tonic
    pipeline as predict.py. Returns detected tonic in Hz."""
    y, sr = librosa.load(audio_path, sr=16000, mono=True, duration=duration)

    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    if rms > 1e-6:
        y = y * (0.1 / rms)

    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=60, fmax=800, sr=sr,
        frame_length=1024, hop_length=256
    )
    confident_mask = voiced_flag & (voiced_probs > 0.3)
    voiced = f0[confident_mask]
    if len(voiced) < 30:
        voiced = f0[voiced_flag]
    if len(voiced) < 30:
        raise ValueError("Not enough voiced pitches")

    return _detect_tonic(voiced)


def cents_diff(detected, expert):
    """Signed difference in cents: positive means detected is sharp."""
    return 1200.0 * math.log2(detected / expert)


def main():
    with open(MAPPING_PATH, encoding="utf-8") as f:
        id_to_name = json.load(f)

    # Collect all (raga_id, audio_path) pairs that have a matching .tonicFine
    pairs_by_raga = {}
    for raga_id in sorted(os.listdir(AUDIO_DIR)):
        raga_name = id_to_name.get(raga_id)
        if raga_name is None:
            continue
        for root, _, files in os.walk(os.path.join(AUDIO_DIR, raga_id)):
            for fname in files:
                if not fname.lower().endswith(".mp3"):
                    continue
                apath = os.path.join(root, fname)
                tpath = tonicfine_path(apath)
                if os.path.exists(tpath):
                    pairs_by_raga.setdefault(raga_id, []).append((raga_name, apath, tpath))

    available_ragas = sorted(pairs_by_raga.keys())
    rng = random.Random(42)
    chosen_ragas = rng.sample(available_ragas, min(5, len(available_ragas)))

    results = []
    for raga_id in chosen_ragas:
        raga_name, apath, tpath = rng.choice(pairs_by_raga[raga_id])
        rel = os.path.relpath(apath, AUDIO_DIR)
        print(f"Processing {raga_name}: {rel} ...", flush=True)

        expert = load_expert_tonic(tpath)
        try:
            detected = detect_tonic_from_audio(apath, DIAGNOSTIC_DURATION)
            diff = cents_diff(detected, expert)
            results.append((raga_name, rel, expert, detected, diff))
        except ValueError as e:
            results.append((raga_name, rel, expert, None, None))
            print(f"  WARNING: {e}", flush=True)

    # Table
    print()
    header = f"{'Raga':<22}  {'Expert Hz':>9}  {'Detected Hz':>11}  {'Error (cents)':>13}  {'Status':<12}  Recording"
    print(header)
    print("-" * len(header))
    errors = []
    for raga_name, rel, expert, detected, diff in results:
        if detected is None:
            print(f"{raga_name:<22}  {expert:>9.2f}  {'FAILED':>11}  {'—':>13}  {'ERROR':<12}  {rel}")
            continue
        abs_diff = abs(diff)
        status = "GOOD (≤25¢)" if abs_diff <= 25 else ("OK (≤50¢)" if abs_diff <= 50 else "BROKEN (>50¢)")
        print(f"{raga_name:<22}  {expert:>9.2f}  {detected:>11.2f}  {diff:>+13.1f}  {status:<12}  {rel}")
        errors.append(abs_diff)

    # Summary
    if errors:
        print()
        print("Summary:")
        print(f"  Recordings tested:     {len(results)}")
        print(f"  Mean absolute error:   {sum(errors)/len(errors):.1f} cents")
        print(f"  Median absolute error: {sorted(errors)[len(errors)//2]:.1f} cents")
        print(f"  Max absolute error:    {max(errors):.1f} cents")
        print(f"  Within ±25¢ (good):   {sum(1 for e in errors if e <= 25)}/{len(errors)}")
        print(f"  Within ±50¢ (ok):     {sum(1 for e in errors if e <= 50)}/{len(errors)}")
        print(f"  Over 50¢ (broken):    {sum(1 for e in errors if e > 50)}/{len(errors)}")


if __name__ == "__main__":
    main()
