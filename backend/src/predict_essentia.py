"""
Production inference with the Essentia expert extractors (Phase 13).

Replaces the pyin + heuristic-tonic query path with the same algorithms
that produced the CompMusic expert features:
  * PredominantPitchMelodia — the Salamon-Gomez melody extractor behind
    the dataset's .pitch files.
  * TonicIndianArtMusic — Gulati's multipitch tonic detector (85.4%
    octave-agnostic on the 480-recording set vs 70.2% for the hand-built
    sa_pa scorer).

Query path (matches the Phase 13d/e eval that hit 73.62%+ top-1):
  1. Load the upload at 44.1 kHz (Essentia MonoLoader).
  2. Detect ONE tonic on a 180s middle window (TonicIndianArtMusic).
  3. Extract N evenly-spaced windows of Melodia pitch, build a TDMS per
     window under the shared tonic, average them.
  4. 1-NN with symmetric KL against the full-recording template index.

Templates (built offline, shipped as a fixed asset) use Melodia pitch +
the expert .tonicFine tonic, full recording. The 1-NN lookup reuses
predict_tdms.TDMSIndex unchanged.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from build_tdms import compute_tdms

SR = 44100
MELODIA_FRAME = 2048
MELODIA_HOP = 128
TONIC_WINDOW_S = 180.0
# Query density. Phase 13 sweep: 3x60s -> 70.0%, 5x60s -> 73.62%,
# 7x90s -> 75.62% top-1 (clears the 75% target). Pinned to the best
# validated config. For short uploads the windows overlap and effective
# density drops (a 2-3 min clip lands closer to the 3-window 70% number),
# but it never does worse than fewer windows.
N_WINDOWS = 7
WINDOW_S = 90.0
MIN_AUDIO_S = 5.0
# Cap how much of a long upload we decode into RAM. 20 min is far more than
# the 7x90s=630s the windows can use, and bounds peak memory at ~212 MB
# (44.1kHz float32) for a pathologically long file. Median CMD recording is
# 9.75 min, so this is a no-op for essentially all real input.
MAX_ANALYZE_S = 1200.0


def _melodia_tdms(audio: np.ndarray, tonic_hz: float):
    """audio float32 @ 44.1k -> flattened TDMS under tonic_hz, or None."""
    import essentia.standard as es

    pitch, _ = es.PredominantPitchMelodia(frameSize=MELODIA_FRAME, hopSize=MELODIA_HOP)(audio)
    timestamps = np.arange(len(pitch)) * (MELODIA_HOP / SR)
    pitches = np.where(pitch > 0, pitch, 0.0).astype(np.float64)
    if np.count_nonzero(pitches) < 30:
        return None
    tdms = compute_tdms(timestamps, pitches, float(tonic_hz))
    return tdms.ravel() if tdms is not None else None


def detect_tonic(audio: np.ndarray, tonic_override: float | None = None) -> float:
    """Tonic via TonicIndianArtMusic on a 180s middle window (or override)."""
    import essentia.standard as es

    if tonic_override is not None and float(tonic_override) > 0:
        return float(tonic_override)
    total = len(audio)
    win = int(TONIC_WINDOW_S * SR)
    if total > win:
        mid = total // 2
        chunk = audio[mid - win // 2: mid + win // 2]
    else:
        chunk = audio
    if len(chunk) < MIN_AUDIO_S * SR:
        return 0.0
    return float(es.TonicIndianArtMusic()(chunk))


def compute_query_tdms(audio_path: str, *, tonic_override: float | None = None):
    """Detect tonic once, average up to N_WINDOWS Melodia-pitch TDMSs.

    Loads at most MAX_ANALYZE_S of audio in a single decode (one EasyLoader
    call capped to startTime=0..min(total, MAX_ANALYZE_S)) then slices in
    memory. This keeps a typical upload fast (one decode) while bounding peak
    RAM for a pathologically long file — at 44.1 kHz float32, the 20-minute
    cap is ~212 MB regardless of how long the upload actually is.

    Returns (tdms_vector float32 (14400,), tonic_hz). Raises ValueError if no
    usable window can be extracted.
    """
    import librosa
    import essentia.standard as es

    try:
        total_s = float(librosa.get_duration(path=audio_path))
    except Exception as exc:
        raise ValueError(f"could not read audio: {exc}") from exc
    if total_s < MIN_AUDIO_S:
        raise ValueError("audio too short for raga detection (need >= 5s)")

    load_s = min(total_s, MAX_ANALYZE_S)
    audio = es.EasyLoader(filename=audio_path, sampleRate=SR, startTime=0.0, endTime=load_s)()
    return query_tdms_from_audio(audio, tonic_override=tonic_override)


def query_tdms_from_audio(audio: np.ndarray, *, tonic_override: float | None = None):
    """Core query path on an in-memory audio array at SR.

    Shared by compute_query_tdms (production) and the length-curve eval so
    both measure identical logic. Detects tonic once, averages up to
    N_WINDOWS Melodia-pitch TDMSs. Returns (tdms float32 (14400,), tonic_hz).
    """
    total = len(audio)
    if total < MIN_AUDIO_S * SR:
        raise ValueError("audio too short for raga detection (need >= 5s)")

    tonic = detect_tonic(audio, tonic_override)
    if tonic <= 0:
        raise ValueError("tonic detection failed")

    wlen = int(WINDOW_S * SR)
    if total <= wlen:
        fractions = [0.5]
    else:
        fractions = [(i + 1) / (N_WINDOWS + 1) for i in range(N_WINDOWS)]

    tdmss = []
    for frac in fractions:
        center = int(total * frac)
        lo = max(0, min(center - wlen // 2, total - wlen))
        chunk = audio[lo: lo + wlen] if total > wlen else audio
        if len(chunk) < MIN_AUDIO_S * SR:
            continue
        row = _melodia_tdms(chunk, tonic)
        if row is not None:
            tdmss.append(row)

    if not tdmss:
        raise ValueError("no usable windows for TDMS extraction")
    avg = np.mean(np.stack(tdmss, axis=0), axis=0)
    avg /= max(float(avg.sum()), 1e-12)
    return avg.astype(np.float32), tonic


def predict_top_k(index, audio_path: str, tonic_override: float | None = None):
    """audio path -> (per-class proba (n_classes,), tonic_hz) via 1-NN."""
    q_tdms, tonic = compute_query_tdms(audio_path, tonic_override=tonic_override)
    return index.query(q_tdms), tonic
