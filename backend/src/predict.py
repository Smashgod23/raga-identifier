import numpy as np
import pickle
import json
import os
from scipy.ndimage import uniform_filter1d

def load_model():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base, "models", "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(base, "data", "classes.json")) as f:
        classes = json.load(f)
    with open(os.path.join(base, "models", "raga_sklearn.pkl"), "rb") as f:
        model = pickle.load(f)
    return model, scaler, classes

def extract_features_from_audio(audio_path):
    import librosa
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    f0, voiced_flag, _ = librosa.pyin(y, fmin=60, fmax=800, sr=sr, frame_length=2048)
    voiced = f0[voiced_flag]
    if len(voiced) < 50:
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

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py path/to/audio.wav")
        sys.exit(1)
    model, scaler, classes = load_model()
    features = extract_features_from_audio(sys.argv[1])
    features_scaled = scaler.transform([features])
    probs = model.predict_proba(features_scaled)[0]
    top5_idx = np.argsort(probs)[::-1][:5]
    print("\nTop 5 predictions:")
    for i in top5_idx:
        bar = "█" * int(probs[i] * 50)
        print(f"  {classes[i]:<25} {probs[i]*100:5.1f}% {bar}")
