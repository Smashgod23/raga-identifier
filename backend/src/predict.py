import numpy as np
import os
from scipy.ndimage import uniform_filter1d

def extract_features_from_audio(audio_path):
    import librosa
    y, sr = librosa.load(audio_path, sr=16000, mono=True, duration=60)
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=60, fmax=800, sr=sr,
        frame_length=1024, hop_length=256
    )
    voiced = f0[voiced_flag]
    if len(voiced) < 30:
        raise ValueError("Not enough pitched audio detected")

    folded_pitches = voiced.copy()
    while np.any(folded_pitches > 120):
        folded_pitches = np.where(folded_pitches > 120, folded_pitches / 2, folded_pitches)
    while np.any(folded_pitches < 60):
        folded_pitches = np.where(folded_pitches < 60, folded_pitches * 2, folded_pitches)

    hist, bin_edges = np.histogram(folded_pitches, bins=200, range=(60, 120))
    smoothed = uniform_filter1d(hist.astype(float), size=3)
    tonic_idx = np.argmax(smoothed)
    tonic_base = (bin_edges[tonic_idx] + bin_edges[tonic_idx + 1]) / 2

    median_pitch = np.median(voiced)
    tonic = tonic_base
    while tonic * 2 < median_pitch:
        tonic *= 2

    cents = 1200 * np.log2(voiced / tonic)
    cents = cents[(cents > -2400) & (cents < 2400)]
    folded = cents % 1200
    hist, _ = np.histogram(folded, bins=120, range=(0, 1200), density=True)
    return hist
