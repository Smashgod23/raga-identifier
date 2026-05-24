"""
Extract TDMS features for the 480 CompMusic Carnatic recordings.

Walks `data/RagaDataset/Carnatic/features/<raga_id>/.../<piece>.pitchSilIntrpPP`,
loads each pitch contour and its `.tonicFine` tonic, computes a 120x120 TDMS,
flattens to 14400-D, and saves:

  data/X_tdms.npy     (N, 14400) float32
  data/y_tdms.npy     (N,)       int64
  data/tdms_meta.json list[{raga, raga_id, recording_id, path}]

The class label order is the same as data/classes.json (40 ragas) so the
TDMS features align with X.npy.

Run from backend/ (so relative `data/` paths resolve correctly):

    python src/extract_tdms_features.py

Takes ~10 minutes on a M-series Mac.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from build_tdms import compute_tdms

DATASET_DIR = os.path.join(
    "data", "RagaDataset", "Carnatic"
)
FEATURES_DIR = os.path.join(DATASET_DIR, "features")
INFO_DIR = os.path.join(DATASET_DIR, "_info_")


def load_pitch_and_tonic(recording_dir: str):
    pitch_file = None
    tonic_file = None
    for name in os.listdir(recording_dir):
        if name.endswith(".pitchSilIntrpPP"):
            pitch_file = os.path.join(recording_dir, name)
        elif name.endswith(".tonicFine"):
            tonic_file = os.path.join(recording_dir, name)
    if pitch_file is None or tonic_file is None:
        return None
    pitch_data = np.loadtxt(pitch_file)
    if pitch_data.ndim != 2 or pitch_data.shape[1] != 2:
        return None
    with open(tonic_file) as f:
        tonic = float(f.read().strip())
    return pitch_data[:, 0], pitch_data[:, 1], tonic


def main() -> None:
    with open(os.path.join(INFO_DIR, "ragaId_to_ragaName_mapping.json")) as f:
        id_to_name = json.load(f)

    with open("data/classes.json") as f:
        classes = json.load(f)
    name_to_label = {name: i for i, name in enumerate(classes)}

    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    meta: list[dict] = []

    raga_ids = sorted(os.listdir(FEATURES_DIR))
    t0 = time.time()
    total_processed = 0
    total_skipped = 0

    for raga_id in raga_ids:
        if raga_id not in id_to_name:
            continue
        raga_name = id_to_name[raga_id]
        if raga_name not in name_to_label:
            print(f"  WARN: raga {raga_name} not in classes.json, skipping")
            continue
        label = name_to_label[raga_name]
        raga_dir = os.path.join(FEATURES_DIR, raga_id)
        count = 0

        for root, _, files in os.walk(raga_dir):
            if not any(f.endswith(".pitchSilIntrpPP") for f in files):
                continue
            loaded = load_pitch_and_tonic(root)
            if loaded is None:
                total_skipped += 1
                continue
            timestamps, pitches, tonic = loaded
            tdms = compute_tdms(timestamps, pitches, tonic)
            if tdms is None:
                total_skipped += 1
                continue
            X_rows.append(tdms.ravel())
            y_rows.append(label)
            meta.append(
                {
                    "raga": raga_name,
                    "raga_id": raga_id,
                    "recording_id": os.path.relpath(root, FEATURES_DIR),
                    "path": root,
                }
            )
            count += 1
            total_processed += 1

        elapsed = time.time() - t0
        rate = total_processed / max(1e-6, elapsed)
        print(
            f"  {raga_name:30s} {count:3d} recs  "
            f"({total_processed} total, {rate:.1f} rec/s)"
        )

    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.int64)
    print(f"\nProcessed {total_processed} recordings, skipped {total_skipped}")
    print(f"X_tdms shape: {X.shape}  dtype: {X.dtype}")
    print(f"y_tdms shape: {y.shape}")

    np.save("data/X_tdms.npy", X)
    np.save("data/y_tdms.npy", y)
    with open("data/tdms_meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nSaved data/X_tdms.npy, data/y_tdms.npy, data/tdms_meta.json")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
