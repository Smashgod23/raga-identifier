import numpy as np
import os
import json
from scipy.ndimage import uniform_filter1d

DATASET_DIR = os.path.expanduser(
    "~/raga-identifier/backend/data/RagaDataset/Carnatic"
)

def load_pitch_and_tonic(recording_dir):
    pitch_file = None
    tonic_file = None
    for f in os.listdir(recording_dir):
        if f.endswith(".pitchSilIntrpPP"):
            pitch_file = os.path.join(recording_dir, f)
        elif f.endswith(".tonicFine"):
            tonic_file = os.path.join(recording_dir, f)
    if not pitch_file or not tonic_file:
        return None, None
    pitch_data = np.loadtxt(pitch_file)
    timestamps = pitch_data[:, 0]
    pitches = pitch_data[:, 1]
    tonic = float(open(tonic_file).read().strip())
    return timestamps, pitches, tonic

def filter_stable_pitches(timestamps, pitches, tonic,
                           tslope=1500, ttime=0.1):
    """
    Keep only stable pitch regions — low slope, held for minimum duration.
    Based on Koduri et al. 2016 thesis approach.
    tslope: max allowed pitch slope in cents/sec
    ttime: minimum duration in seconds for stable region
    """
    if len(pitches) < 2:
        return np.array([])

    dt = np.mean(np.diff(timestamps))
    voiced = pitches > 0

    # Convert to cents relative to tonic
    cents = np.where(voiced, 1200 * np.log2(np.where(voiced, pitches, 1) / tonic), 0)

    # Compute slope in cents/sec
    slope = np.abs(np.gradient(cents, dt))

    # Stable = voiced + low slope
    stable = voiced & (slope < tslope)

    # Filter by minimum duration — remove isolated stable frames
    min_frames = int(ttime / dt)
    stable_filtered = np.zeros_like(stable)

    i = 0
    while i < len(stable):
        if stable[i]:
            j = i
            while j < len(stable) and stable[j]:
                j += 1
            if (j - i) >= min_frames:
                stable_filtered[i:j] = True
            i = j
        else:
            i += 1

    return cents[stable_filtered]

def compute_pitch_class_distribution(cents, bins=120):
    if len(cents) < 10:
        return None
    folded = cents % 1200
    hist, _ = np.histogram(folded, bins=bins, range=(0, 1200), density=True)
    return hist

def compute_duration_weighted_distribution(timestamps, pitches, tonic, bins=120):
    """
    Duration-weighted pitch class distribution.
    Each pitch is weighted by how long it is held.
    """
    voiced = pitches > 0
    if np.sum(voiced) < 10:
        return None

    dt = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 0.0044
    cents = 1200 * np.log2(pitches[voiced] / tonic)
    folded = cents % 1200

    # Each frame contributes dt seconds of weight
    hist, _ = np.histogram(folded, bins=bins, range=(0, 1200),
                           weights=np.ones(len(folded)) * dt, density=True)
    return hist

def extract_features(recording_dir):
    result = load_pitch_and_tonic(recording_dir)
    if result[0] is None:
        return None
    timestamps, pitches, tonic = result

    # Feature 1: stable pitch class distribution
    stable_cents = filter_stable_pitches(timestamps, pitches, tonic)
    pcd_stable = compute_pitch_class_distribution(stable_cents)

    # Feature 2: duration-weighted distribution (all voiced frames)
    pcd_duration = compute_duration_weighted_distribution(timestamps, pitches, tonic)

    if pcd_stable is None or pcd_duration is None:
        return None

    # Concatenate both — 240 features total
    return np.concatenate([pcd_stable, pcd_duration])

def build_dataset():
    mapping_path = os.path.join(DATASET_DIR, "_info_", "ragaId_to_ragaName_mapping.json")
    with open(mapping_path) as f:
        id_to_name = json.load(f)

    features_dir = os.path.join(DATASET_DIR, "features")
    X, y, raga_names = [], [], []

    raga_ids = sorted(os.listdir(features_dir))

    for raga_id in raga_ids:
        if raga_id not in id_to_name:
            continue
        raga_name = id_to_name[raga_id]
        raga_dir = os.path.join(features_dir, raga_id)
        label = len(raga_names)
        raga_names.append(raga_name)
        count = 0

        for root, dirs, files in os.walk(raga_dir):
            has_pitch = any(f.endswith(".pitchSilIntrpPP") for f in files)
            if not has_pitch:
                continue
            feat = extract_features(root)
            if feat is not None:
                X.append(feat)
                y.append(label)
                count += 1

        print(f"{raga_name}: {count} recordings")

    X = np.array(X)
    y = np.array(y)
    print(f"\nTotal: {len(X)} samples, {len(raga_names)} ragas")
    print(f"Feature shape: {X.shape}")

    np.save("data/X.npy", X)
    np.save("data/y.npy", y)
    with open("data/classes.json", "w") as f:
        json.dump(raga_names, f, ensure_ascii=False, indent=2)
    print("Saved X.npy, y.npy, classes.json")
    return X, y, raga_names

if __name__ == "__main__":
    build_dataset()