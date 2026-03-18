import numpy as np
import torch
import torch.nn as nn
import pickle
import json
import os

class RagaNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def load_model():
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("data/classes.json") as f:
        classes = json.load(f)
    model = RagaNet(input_dim=120, num_classes=len(classes))
    model.load_state_dict(torch.load("models/raga_model_best.pt", map_location="cpu"))
    model.eval()
    return model, scaler, classes

def extract_features_from_audio(audio_path):
    import librosa
    from scipy.ndimage import uniform_filter1d

    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    # Extract pitch
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=60, fmax=800,
        sr=sr, frame_length=2048
    )
    voiced = f0[voiced_flag]
    if len(voiced) < 50:
        raise ValueError("Not enough pitched audio detected")

    # Fold all pitches into one octave (60-120 Hz range)
    # This collapses Sa regardless of which octave it's sung in
    folded_pitches = voiced.copy()
    while np.any(folded_pitches > 120):
        folded_pitches = np.where(folded_pitches > 120, folded_pitches / 2, folded_pitches)
    while np.any(folded_pitches < 60):
        folded_pitches = np.where(folded_pitches < 60, folded_pitches * 2, folded_pitches)

    # Find most common pitch in that octave = tonic
    hist, bin_edges = np.histogram(folded_pitches, bins=200, range=(60, 120))
    smoothed = uniform_filter1d(hist.astype(float), size=3)
    tonic_idx = np.argmax(smoothed)
    tonic_base = (bin_edges[tonic_idx] + bin_edges[tonic_idx + 1]) / 2

    # Bring tonic up to match median octave of the performance
    median_pitch = np.median(voiced)
    tonic = tonic_base
    while tonic * 2 < median_pitch:
        tonic *= 2

    print(f"Estimated tonic: {tonic:.1f} Hz")

    # Convert to cents relative to tonic
    cents = 1200 * np.log2(voiced / tonic)
    cents = cents[(cents > -2400) & (cents < 2400)]

    folded = cents % 1200
    hist, _ = np.histogram(folded, bins=120, range=(0, 1200), density=True)
    return hist

def predict(audio_path):
    model, scaler, classes = load_model()
    features = extract_features_from_audio(audio_path)
    features = scaler.transform([features])
    x = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        top5 = torch.topk(probs, 5)

    results = [
        {
            "raga": classes[i],
            "confidence": round(p.item() * 100, 1)
        }
        for i, p in zip(top5.indices, top5.values)
    ]
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py path/to/audio.wav")
        sys.exit(1)
    results = predict(sys.argv[1])
    print("\nTop 5 predictions:")
    for r in results:
        bar = "█" * int(r["confidence"] / 2)
        print(f"  {r['raga']:<25} {r['confidence']:5.1f}% {bar}")