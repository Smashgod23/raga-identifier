"""
TDMS extraction from raw audio.

Mirrors the deployed pyin + tonic-detection pipeline from
`backend/src/predict.py`, so a TDMS built here uses the same pitch
estimate the live API would compute. The Phase-1 templates were built
from CompMusic's expert `.pitchSilIntrpPP` files and their `.tonicFine`
tonics; this module is what makes TDMS deployable.

Two public functions:
  extract_pitch_and_tonic(audio_path, ...) - shared by both v1 and TDMS
    feature extraction; returns (timestamps, pitches_hz, tonic_hz).
  compute_tdms_from_audio(audio_path, ...) - convenience wrapper that
    chains pitch extraction into compute_tdms.

Returning the raw pitch contour lets a single librosa.pyin call serve
both the v1 360-D PCD path and the TDMS path during A/B benchmarks.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from build_tdms import compute_tdms
from predict import _detect_tonic, _fold_override_to_tonic


SR = 16000
HOP = 256
FRAME = 1024
FMIN = 60
FMAX = 800
HOP_SECONDS = HOP / SR


def extract_pitch_and_tonic(
    audio_path: str,
    *,
    offset: float = 0.0,
    duration: Optional[float] = None,
    tonic_override: Optional[float] = None,
):
    """Run the predict.py pyin pipeline, return aligned arrays plus tonic.

    Returns:
        timestamps (1D float64, seconds)
        pitches_hz (1D float64, Hz, NaN for unvoiced frames)
        voiced_pitches_hz (1D, Hz, used for tonic detection)
        tonic_hz (float)
    """
    import librosa

    load_kwargs = {"sr": SR, "mono": True, "offset": float(offset)}
    if duration is not None:
        load_kwargs["duration"] = float(duration)
    y, sr = librosa.load(audio_path, **load_kwargs)

    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    if rms > 1e-6:
        y = y * (0.1 / rms)

    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=FMIN, fmax=FMAX, sr=sr, frame_length=FRAME, hop_length=HOP
    )

    confident = voiced_flag & (voiced_probs > 0.3)
    voiced_pitches = f0[confident]
    if len(voiced_pitches) < 30:
        voiced_pitches = f0[voiced_flag]
    if len(voiced_pitches) < 30:
        raise ValueError("Not enough pitched audio detected")

    if tonic_override is not None and float(tonic_override) > 0:
        tonic = _fold_override_to_tonic(tonic_override, voiced_pitches)
    else:
        tonic = _detect_tonic(voiced_pitches)

    # Build full-length contour with NaN for unvoiced (TDMS needs this so
    # tau-offset alignment is correct). Use voiced_flag (looser than the
    # confident mask) to match the v1 feature path.
    pitches_full = np.where(voiced_flag, f0, np.nan)
    pitches_full = np.where(np.isfinite(pitches_full), pitches_full, np.nan)
    timestamps = np.arange(len(pitches_full)) * HOP_SECONDS

    return timestamps, pitches_full, voiced_pitches, float(tonic)


def compute_tdms_from_audio(
    audio_path: str,
    *,
    offset: float = 0.0,
    duration: Optional[float] = None,
    tonic_override: Optional[float] = None,
):
    """Extract pitch, normalize by tonic, return (TDMS, tonic_hz)."""
    timestamps, pitches_full, _, tonic = extract_pitch_and_tonic(
        audio_path,
        offset=offset,
        duration=duration,
        tonic_override=tonic_override,
    )
    # compute_tdms expects unvoiced as <= 0; convert NaN sentinel.
    pitches_for_tdms = np.where(np.isnan(pitches_full), 0.0, pitches_full)
    tdms = compute_tdms(timestamps, pitches_for_tdms, tonic)
    return tdms, tonic
