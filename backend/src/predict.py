import numpy as np
import os
from scipy.ndimage import uniform_filter1d

def extract_features_from_audio(audio_path):
    import librosa
    from scipy.ndimage import uniform_filter1d

    y, sr = librosa.load(audio_path, sr=16000, mono=True, duration=60)
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=60, fmax=800, sr=sr,
        frame_length=1024, hop_length=256
    )

    voiced = f0[voiced_flag]
    if len(voiced) < 30:
        raise ValueError("Not enough pitched audio detected")

    # Tonic detection
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

    # All voiced frames in cents
    hop = 256 / sr
    all_cents = 1200 * np.log2(voiced / tonic)

    # Stable pitch filtering — slope < 1500 cents/sec
    slope = np.abs(np.gradient(all_cents, hop))
    stable_mask = slope < 1500
    min_frames = int(0.1 / hop)
    stable_filtered = []
    i = 0
    while i < len(stable_mask):
        if stable_mask[i]:
            j = i
            while j < len(stable_mask) and stable_mask[j]:
                j += 1
            if (j - i) >= min_frames:
                stable_filtered.extend(all_cents[i:j])
            i = j
        else:
            i += 1

    stable_cents = np.array(stable_filtered)

    # Feature 1: stable pitch class distribution
    if len(stable_cents) < 10:
        stable_cents = all_cents  # fallback
    folded_stable = stable_cents % 1200
    pcd_stable, _ = np.histogram(folded_stable, bins=120, range=(0, 1200), density=True)

    # Feature 2: duration-weighted distribution
    folded_all = all_cents % 1200
    pcd_duration, _ = np.histogram(folded_all, bins=120, range=(0, 1200), density=True)

    return np.concatenate([pcd_stable, pcd_duration])
