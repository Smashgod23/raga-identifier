import numpy as np
import os
from scipy.ndimage import uniform_filter1d

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def hz_to_note_name(hz):
    """Convert Hz to nearest piano note name with cents offset, e.g. 'C#3 (+12¢)'."""
    if hz is None or hz <= 0:
        return ''
    midi = 69 + 12 * np.log2(hz / 440.0)
    midi_round = int(round(midi))
    note = NOTE_NAMES[midi_round % 12]
    octave = midi_round // 12 - 1
    cents = int(round((midi - midi_round) * 100))
    return f'{note}{octave} ({cents:+d}¢)' if abs(cents) >= 3 else f'{note}{octave}'


def _detect_tonic(voiced):
    """Pick Sa from voiced pitches. Candidates come from a folded-Hz histogram;
    the winner is whichever produces the most concentrated cents-mod-1200
    distribution (Krumhansl-style L2 peakedness)."""
    folded_pitches = voiced.copy()
    while np.any(folded_pitches > 120):
        folded_pitches = np.where(folded_pitches > 120, folded_pitches / 2, folded_pitches)
    while np.any(folded_pitches < 60):
        folded_pitches = np.where(folded_pitches < 60, folded_pitches * 2, folded_pitches)

    hist, bin_edges = np.histogram(folded_pitches, bins=200, range=(60, 120))
    smoothed = uniform_filter1d(hist.astype(float), size=5)
    median_pitch = np.median(voiced)

    candidate_indices = np.argsort(smoothed)[::-1][:5]
    best_tonic, best_score = None, -1
    for idx in candidate_indices:
        if smoothed[idx] == 0:
            continue
        cand = (bin_edges[idx] + bin_edges[idx + 1]) / 2
        while cand * 2 < median_pitch:
            cand *= 2
        cents = 1200 * np.log2(voiced / cand)
        h, _ = np.histogram(cents % 1200, bins=120, range=(0, 1200))
        score = float(np.sum(h ** 2))
        if score > best_score:
            best_score = score
            best_tonic = cand

    if best_tonic is None:
        tonic_idx = int(np.argmax(smoothed))
        best_tonic = (bin_edges[tonic_idx] + bin_edges[tonic_idx + 1]) / 2
        while best_tonic * 2 < median_pitch:
            best_tonic *= 2
    return float(best_tonic)


def _fold_override_to_tonic(tonic_hz, voiced):
    """A user-supplied Sa can be given in any octave. Shift it to the octave
    closest to the singer's median pitch so cents math lines up."""
    median_pitch = float(np.median(voiced))
    t = float(tonic_hz)
    while t * 2 <= median_pitch * 1.5:
        t *= 2
    while t >= median_pitch * 1.5:
        t /= 2
    return t


def extract_features_from_audio(audio_path, tonic_override=None, offset=0.0, duration=None):
    """Load an audio segment, detect (or accept) Sa, build a 360-d pitch-class feature vector.
    Returns (features, tonic_hz). `tonic_override` in Hz skips auto-detection."""
    import librosa

    load_kwargs = {'sr': 16000, 'mono': True, 'offset': float(offset)}
    if duration is not None:
        load_kwargs['duration'] = float(duration)
    y, sr = librosa.load(audio_path, **load_kwargs)

    # RMS normalization — robust to transient spikes (coughs, mic bumps, applause)
    # that would otherwise dominate a peak-normalized signal.
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
        raise ValueError("Not enough pitched audio detected")

    hop = 256 / sr

    if tonic_override is not None and float(tonic_override) > 0:
        tonic = _fold_override_to_tonic(tonic_override, voiced)
    else:
        tonic = _detect_tonic(voiced)

    all_cents = 1200 * np.log2(voiced / tonic)

    # Feature 1: nyas-style stable pitches (slope < 1500 cents/sec, held ≥ 100ms)
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
    pcd_nyas, _ = np.histogram(nyas_cents % 1200, bins=120, range=(0, 1200), density=True)

    # Feature 2: duration-weighted distribution across all voiced pitches
    pcd_duration, _ = np.histogram(all_cents % 1200, bins=120, range=(0, 1200), density=True)

    # Feature 3: loose-stable distribution (catches notes that aren't fully held)
    stable_cents = all_cents[slope < 3000]
    if len(stable_cents) < 10:
        stable_cents = all_cents
    pcd_stable, _ = np.histogram(stable_cents % 1200, bins=120, range=(0, 1200), density=True)

    features = np.concatenate([pcd_nyas, pcd_duration, pcd_stable])
    return features, float(tonic)


if __name__ == "__main__":
    import sys
    import pickle
    import json

    if len(sys.argv) < 2:
        print("Usage: python src/predict.py path/to/audio.wav [tonic_hz]")
        sys.exit(1)

    override = float(sys.argv[2]) if len(sys.argv) >= 3 else None
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base, "models", "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(base, "data", "classes.json")) as f:
        classes = json.load(f)
    with open(os.path.join(base, "models", "raga_sklearn.pkl"), "rb") as f:
        model = pickle.load(f)

    features, tonic = extract_features_from_audio(sys.argv[1], tonic_override=override)
    print(f"Feature shape: {features.shape}")
    print(f"Tonic: {tonic:.2f} Hz ({hz_to_note_name(tonic)})")
    features_scaled = scaler.transform([features])
    probs = model.predict_proba(features_scaled)[0]
    top5_idx = np.argsort(probs)[::-1][:5]
    print("\nTop 5 predictions:")
    for i in top5_idx:
        bar = "█" * int(probs[i] * 50)
        print(f"  {classes[i]:<25} {probs[i]*100:5.1f}% {bar}")
