import numpy as np
import os
import json

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
    pitches = pitch_data[:, 1]
    tonic = float(open(tonic_file).read().strip())
    return pitches, tonic

def pitch_to_cents(pitches, tonic):
    voiced = pitches[pitches > 0]
    if len(voiced) < 100:
        return None
    cents = 1200 * np.log2(voiced / tonic)
    return cents

def compute_pitch_class_distribution(cents, bins=120):
    # fold into one octave (0-1200 cents), 120 bins = 10 cents each
    folded = cents % 1200
    hist, _ = np.histogram(folded, bins=bins, range=(0, 1200), density=True)
    return hist

def extract_features(recording_dir):
    pitches, tonic = load_pitch_and_tonic(recording_dir)
    if pitches is None:
        return None
    cents = pitch_to_cents(pitches, tonic)
    if cents is None:
        return None
    pcd = compute_pitch_class_distribution(cents)
    return pcd  # 120-dimensional feature vector

def build_dataset():
    # Load raga name mapping
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

        # Walk all recording subdirectories
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

    # Save
    np.save("data/X.npy", X)
    np.save("data/y.npy", y)
    with open("data/classes.json", "w") as f:
        json.dump(raga_names, f, ensure_ascii=False, indent=2)
    print("Saved X.npy, y.npy, classes.json")
    return X, y, raga_names

if __name__ == "__main__":
    build_dataset()