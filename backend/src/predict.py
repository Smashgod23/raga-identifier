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

    # Tonic detection
    hop = 256 / sr
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

    all_cents = 1200 * np.log2(voiced / tonic)

    # --- Feature 1: nyas simulation (stable pitch filtering, strict) ---
    slope = np.abs(np.gradient(all_cents, hop))
    stable_mask = slope < 1500
    min_frames = int(0.1 / hop)
    nyas_cents = []
    i = 0
    while i < len(stable_mask):
        if stable_mask[i]:
            j = i
            while j < len(stable_mask) and stable_mask[j]:
                j += 1
            if (j - i) >= min_frames:
                nyas_cents.extend(all_cents[i:j])
            i = j
        else:
            i += 1
    nyas_cents = np.array(nyas_cents) if len(nyas_cents) >= 10 else all_cents
    folded_nyas = nyas_cents % 1200
    pcd_nyas, _ = np.histogram(folded_nyas, bins=120, range=(0, 1200), density=True)

    # --- Feature 2: duration-weighted distribution ---
    folded_all = all_cents % 1200
    pcd_duration, _ = np.histogram(folded_all, bins=120, range=(0, 1200), density=True)

    # --- Feature 3: stable pitch distribution (looser threshold) ---
    stable_mask2 = slope < 3000
    stable_cents = all_cents[stable_mask2]
    if len(stable_cents) < 10:
        stable_cents = all_cents
    folded_stable = stable_cents % 1200
    pcd_stable, _ = np.histogram(folded_stable, bins=120, range=(0, 1200), density=True)

    # 360 features total
    return np.concatenate([pcd_nyas, pcd_duration, pcd_stable])


if __name__ == "__main__":
    import sys
    import pickle
    import json

    if len(sys.argv) < 2:
        print("Usage: python src/predict.py path/to/audio.wav")
        sys.exit(1)

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base, "models", "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(base, "data", "classes.json")) as f:
        classes = json.load(f)
    with open(os.path.join(base, "models", "raga_sklearn.pkl"), "rb") as f:
        model = pickle.load(f)

    features = extract_features_from_audio(sys.argv[1])
    print(f"Feature shape: {features.shape}")
    features_scaled = scaler.transform([features])
    probs = model.predict_proba(features_scaled)[0]
    top5_idx = np.argsort(probs)[::-1][:5]
    print("\nTop 5 predictions:")
    for i in top5_idx:
        bar = "█" * int(probs[i] * 50)
        print(f"  {classes[i]:<25} {probs[i]*100:5.1f}% {bar}")