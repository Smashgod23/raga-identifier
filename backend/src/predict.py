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


def _detect_tonic(voiced, k=15, scorer="sa_pa"):
    """Pick Sa from voiced pitches.

    Candidates are the top-K peaks of a smoothed folded-Hz histogram.
    The winner is the candidate that best satisfies the chosen scorer:

      scorer="sa_pa"  (default) — the tanpura drones Sa and its fifth Pa
        continuously, so the true Sa uniquely has strong energy at the
        exact Sa (0 cents) and Pa (+700 cents) bins of its tonic-relative
        pitch-class distribution. Scoring on that drone signature beats
        the old peakedness metric by a wide margin (Phase 12 diagnostics).
        The exact single-bin form (not +/-10 cents) tested best — the
        drone is a precise sustained pitch, so a tight tolerance is more
        discriminative against Pa-lock / Ri-lock wrong tonics.

      scorer="peakedness" — legacy: most concentrated cents-mod-1200
        distribution (Krumhansl-style L2). Kept for comparison/eval.

    K defaults to 15: Phase 12 showed the correct Sa pitch-class is in the
    top-15 histogram peaks 89.8% of the time (vs 83.5% at K=10, 62.7% at
    K=5), and the sa_pa scorer can exploit the larger pool (peakedness
    could not — it was pinned at 51.9% for any K). Octave choice is
    irrelevant downstream because the TDMS/PCD feature is octave-folded;
    candidates are folded up to the singer's register only for a sensible
    reported Hz value.

    Tonic top-1 (octave-agnostic, 60s window, 480 CompMusic recordings):
    peakedness 51.9%  ->  sa_pa+/-1bin@K10 65.0%  ->  sa_pa exact@K15 70.2%.
    """
    folded_pitches = voiced.copy()
    while np.any(folded_pitches > 120):
        folded_pitches = np.where(folded_pitches > 120, folded_pitches / 2, folded_pitches)
    while np.any(folded_pitches < 60):
        folded_pitches = np.where(folded_pitches < 60, folded_pitches * 2, folded_pitches)

    hist, bin_edges = np.histogram(folded_pitches, bins=200, range=(60, 120))
    smoothed = uniform_filter1d(hist.astype(float), size=5)
    median_pitch = np.median(voiced)

    candidate_indices = np.argsort(smoothed)[::-1][:k]
    best_tonic, best_score = None, -1.0
    for idx in candidate_indices:
        if smoothed[idx] == 0:
            continue
        cand = (bin_edges[idx] + bin_edges[idx + 1]) / 2
        while cand * 2 < median_pitch:
            cand *= 2
        h, _ = np.histogram((1200 * np.log2(voiced / cand)) % 1200, bins=120, range=(0, 1200))
        h = h.astype(float)
        if scorer == "sa_pa":
            p = h / (h.sum() + 1e-9)
            # Exact Sa (bin 0) + Pa (bin 70) energy. Tight single-bin
            # tolerance tested better than +/-1 bin (Phase 12d).
            score = float(p[0] + p[70])
        else:  # peakedness
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

    # The deployed sklearn model was trained on features from a 60s window with
    # peak normalization (git 970d390). Inference MUST match that pipeline, or the
    # feature vectors drift away from what the model learned — a train/serve skew
    # that silently wrecks accuracy (measured: cosine 0.44 vs the trained features,
    # confident wrong predictions). So default to a 60s window and peak-normalize.
    # Callers that pass an explicit duration (multi-segment averaging) set their
    # own window; offset lets them sample different parts of a clip.
    load_kwargs = {'sr': 16000, 'mono': True, 'offset': float(offset),
                   'duration': float(duration) if duration is not None else 60}
    y, sr = librosa.load(audio_path, **load_kwargs)

    # Peak normalization to match training. (An RMS-norm change shipped later
    # broke alignment with the trained model and is reverted here.)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0:
        y = y / peak

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
        # Top-5 peakedness scorer: the exact tonic detector the deployed model
        # was trained with. The newer sa_pa scorer (Phase 12) picks a different
        # Sa, which shifts every cent value and breaks alignment with this v1
        # model, so v1 inference pins the training-time behavior. _detect_tonic
        # keeps its sa_pa default for the offline eval scripts that rely on it.
        tonic = _detect_tonic(voiced, k=5, scorer="peakedness")

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
