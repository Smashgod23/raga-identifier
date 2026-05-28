"""
Production inference for the Phase 10b TDMS k-NN candidate model.

Replicates the eval pipeline at deployment time: extract three 60s
windows from the user's audio (centered at 25/50/75% through the clip),
run pyin + heuristic tonic + compute_tdms on each, average the three
TDMSs, then 1-NN against a precomputed template index using symmetric
KL distance.

The template index is loaded once at API startup (~28 MB for 480
recordings x 14400 float32) and served from memory. Each query runs
three short pyin passes plus 480 symmetric-KL distance computations —
end-to-end about 15-25 seconds on CPU for a 1-2 minute upload.
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from build_tdms import compute_tdms, symmetric_kl_distance
from predict import _detect_tonic, _fold_override_to_tonic, hz_to_note_name

# Eval-time settings. Keep in sync with eval_multiwindow_queries.py.
QUERY_DURATION = 60.0
WINDOW_FRACTIONS = (0.25, 0.50, 0.75)
PYIN_FMIN = 60
PYIN_FMAX = 800
PYIN_FRAME = 1024
PYIN_HOP = 256


class TDMSIndex:
    """In-memory template index loaded once at API startup.

    Stores the 480 long-template TDMS rows (eta*eta = 14400 float32 each)
    and their integer labels. Exposes a single `query()` method that
    accepts a query TDMS (1D, 14400) and returns a length-`n_classes`
    probability distribution over ragas.
    """

    def __init__(self, X_templates: np.ndarray, y_labels: np.ndarray, n_classes: int):
        if X_templates.ndim != 2 or X_templates.shape[1] != 14400:
            raise ValueError(f"templates must be (N, 14400), got {X_templates.shape}")
        if len(y_labels) != len(X_templates):
            raise ValueError("templates and labels must align 1:1")
        self.X = X_templates.astype(np.float32)
        self.y = np.asarray(y_labels, dtype=np.int64)
        self.n_classes = n_classes

    def query(self, q_tdms: np.ndarray) -> np.ndarray:
        """Return per-class scores summing to 1.

        Implementation: compute symmetric KL from `q_tdms` to every
        template, sort templates by distance, and score each class as
        `1 / (rank + 1)` where `rank` is the position of that class's
        first template in the sorted list. Matches the eval-time
        TDMSKnnDirect classifier exactly.
        """
        q = q_tdms.ravel().astype(np.float32)
        n = len(self.X)
        dists = np.empty(n, dtype=np.float64)
        for i in range(n):
            dists[i] = symmetric_kl_distance(q, self.X[i])
        order = np.argsort(dists, kind="stable")
        scores = np.zeros(self.n_classes, dtype=np.float64)
        seen: set[int] = set()
        rank = 0
        for j in order:
            cls = int(self.y[j])
            if cls in seen:
                continue
            scores[cls] = 1.0 / (rank + 1)
            seen.add(cls)
            rank += 1
            if rank >= self.n_classes:
                break
        s = scores.sum()
        if s > 0:
            scores /= s
        return scores


def _pyin_with_tonic(audio_path: str, offset: float, duration: float,
                     tonic_override: Optional[float]):
    """Shared chunk: librosa.load + pyin + tonic detect (or fold override)."""
    import librosa

    y, sr = librosa.load(audio_path, sr=16000, mono=True, offset=float(offset), duration=float(duration))
    if len(y) < sr * 5:
        return None  # less than 5s of audio
    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    if rms > 1e-6:
        y = y * (0.1 / rms)

    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=PYIN_FMIN, fmax=PYIN_FMAX, sr=sr,
        frame_length=PYIN_FRAME, hop_length=PYIN_HOP,
    )
    confident = voiced_flag & (voiced_probs > 0.3)
    voiced = f0[confident]
    if len(voiced) < 30:
        voiced = f0[voiced_flag]
    if len(voiced) < 30:
        return None

    if tonic_override is not None and float(tonic_override) > 0:
        tonic = _fold_override_to_tonic(tonic_override, voiced)
    else:
        tonic = _detect_tonic(voiced)

    hop_seconds = PYIN_HOP / sr
    pitches_full = np.where(voiced_flag, f0, 0.0)
    pitches_full = np.where(np.isfinite(pitches_full), pitches_full, 0.0)
    timestamps = np.arange(len(pitches_full)) * hop_seconds
    return timestamps, pitches_full, tonic


def compute_query_tdms(
    audio_path: str,
    *,
    tonic_override: Optional[float] = None,
    window_fractions: tuple[float, ...] = WINDOW_FRACTIONS,
    window_seconds: float = QUERY_DURATION,
) -> tuple[np.ndarray, float]:
    """Build a multi-window-averaged TDMS for a single audio file.

    For each fraction in `window_fractions`, center a `window_seconds`
    window at `fraction * total_duration` and compute its TDMS. Average
    all successful windows into one 14400-D vector. Returns the averaged
    TDMS and the tonic detected on the first successful window. If
    fewer than three windows can be extracted the function still
    returns whatever is available.
    """
    import librosa

    try:
        total = float(librosa.get_duration(path=audio_path))
    except Exception as exc:
        raise ValueError(f"could not probe audio duration: {exc}") from exc

    tdmss: list[np.ndarray] = []
    used_tonic: Optional[float] = None

    if total <= window_seconds:
        # Single short clip: one window covers the whole thing.
        loaded = _pyin_with_tonic(audio_path, 0.0, total, tonic_override)
        if loaded is None:
            raise ValueError("not enough pitched audio to compute TDMS")
        timestamps, pitches_full, tonic = loaded
        tdms = compute_tdms(timestamps, pitches_full, tonic)
        if tdms is None:
            raise ValueError("TDMS computation failed for short clip")
        return tdms.ravel().astype(np.float32), tonic

    for frac in window_fractions:
        center = total * frac
        offset = max(0.0, min(center - window_seconds / 2.0, total - window_seconds))
        duration = min(window_seconds, total)
        loaded = _pyin_with_tonic(audio_path, offset, duration, tonic_override)
        if loaded is None:
            continue
        timestamps, pitches_full, tonic = loaded
        tdms = compute_tdms(timestamps, pitches_full, tonic)
        if tdms is None:
            continue
        tdmss.append(tdms.ravel())
        if used_tonic is None:
            used_tonic = tonic

    if not tdmss or used_tonic is None:
        raise ValueError("no usable windows for TDMS extraction")

    avg = np.mean(np.stack(tdmss, axis=0), axis=0)
    avg /= max(float(avg.sum()), 1e-12)
    return avg.astype(np.float32), used_tonic


def predict_top_k(index: TDMSIndex, audio_path: str,
                  tonic_override: Optional[float] = None,
                  top_k: int = 5):
    """End-to-end: audio path -> top-k raga predictions with scores.

    Returns (proba: np.ndarray of shape (n_classes,), tonic_hz: float).
    """
    q_tdms, tonic = compute_query_tdms(audio_path, tonic_override=tonic_override)
    proba = index.query(q_tdms)
    return proba, tonic
