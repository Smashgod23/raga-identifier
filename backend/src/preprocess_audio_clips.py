"""
Slice raw Carnatic audio recordings into overlapping 30-second clips and extract
360-dimensional pitch-class features for each clip.

Feature extraction uses extract_features_from_audio from predict.py exactly —
same pyin pipeline, same tonic detection, same 3-channel 360-d feature space
as inference. No changes to that function.

Outputs:
  backend/data/X_audio_clips.npy       — (N, 360) float64 feature matrix
  backend/data/y_audio_clips.npy       — (N,) int64 raga labels
  backend/data/audio_clips_meta.json   — per-row metadata for recording-aware splitting

Run from backend/:
  source venv/bin/activate
  python src/preprocess_audio_clips.py
"""

import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter

import numpy as np

AUDIO_DIR = os.path.expanduser("~/raga-data-audio/RagaDataset/Carnatic/audio")
MAPPING_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data",
    "RagaDataset", "Carnatic", "_info_", "ragaId_to_ragaName_mapping.json"
)
CLASSES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "classes.json")
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
PREDICT_DIR  = os.path.dirname(os.path.abspath(__file__))

CLIP_DURATION     = 30.0  # seconds per clip
CLIP_HOP          = 10.0  # seconds between clip start times
MIN_CLIP_DURATION = 20.0  # minimum clip length to attempt feature extraction


def _build_clip_offsets(total_dur):
    """Return list of (offset, actual_duration) for 30s clips with 10s hop.
    Final clip may be shorter than 30s if the recording ends early."""
    clips = []
    offset = 0.0
    while offset + MIN_CLIP_DURATION <= total_dur:
        dur = min(CLIP_DURATION, total_dur - offset)
        clips.append((offset, dur))
        offset += CLIP_HOP
    return clips


def _process_recording(args):
    """Worker function — runs in a spawned subprocess so all imports are local."""
    audio_path, raga_id, raga_name, label = args

    # Local imports so spawned workers start cleanly without inheriting parent state.
    import librosa
    import sys as _sys
    _sys.path.insert(0, PREDICT_DIR)
    from predict import extract_features_from_audio

    rel_path = os.path.relpath(audio_path, AUDIO_DIR)
    recording_id = os.path.splitext(os.path.basename(audio_path))[0]

    try:
        total_dur = librosa.get_duration(path=audio_path)
    except Exception as e:
        return [], 0, 1  # (results, n_skipped, n_errors)

    clip_specs = _build_clip_offsets(total_dur)
    if not clip_specs:
        return [], 1, 0  # recording too short

    results = []
    n_skipped = 0
    n_errors  = 0

    for offset, dur in clip_specs:
        try:
            features, _ = extract_features_from_audio(audio_path, offset=offset, duration=dur)
            results.append((
                features,
                label,
                {
                    "recording_id": recording_id,
                    "raga_id":      raga_id,
                    "raga_name":    raga_name,
                    "label":        label,
                    "clip_start":   round(offset, 2),
                    "clip_dur":     round(dur, 2),
                    "audio_path":   rel_path,
                }
            ))
        except ValueError:
            # Not enough voiced audio (silence, applause, tanpura drone) — expected
            n_skipped += 1
        except Exception:
            n_errors += 1

    return results, n_skipped, n_errors


def main():
    with open(MAPPING_PATH, encoding="utf-8") as f:
        id_to_name = json.load(f)
    with open(CLASSES_PATH, encoding="utf-8") as f:
        classes = json.load(f)
    name_to_label = {n: i for i, n in enumerate(classes)}

    # Build full work list: one item per recording file
    work_items = []
    for raga_id in sorted(os.listdir(AUDIO_DIR)):
        if raga_id not in id_to_name:
            continue
        raga_name = id_to_name[raga_id]
        if raga_name not in name_to_label:
            print(f"SKIP {raga_name}: not in classes.json", flush=True)
            continue
        label = name_to_label[raga_name]
        raga_dir = os.path.join(AUDIO_DIR, raga_id)
        for root, _, files in os.walk(raga_dir):
            for fname in sorted(files):
                if fname.lower().endswith(".mp3"):
                    work_items.append((
                        os.path.join(root, fname),
                        raga_id, raga_name, label,
                    ))

    nproc = max(1, min(mp.cpu_count() - 1, 8))
    print(
        f"[{time.strftime('%H:%M:%S')}] {len(work_items)} recordings, "
        f"{nproc} workers. Starting...",
        flush=True,
    )

    X_all, y_all, meta_all = [], [], []
    total_skipped = total_errors = completed = 0
    t_start = time.time()

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=nproc) as pool:
        for item_results, n_skip, n_err in pool.imap_unordered(
            _process_recording, work_items, chunksize=1
        ):
            completed  += 1
            total_skipped += n_skip
            total_errors  += n_err

            for features, label, meta in item_results:
                X_all.append(features)
                y_all.append(label)
                meta_all.append(meta)

            elapsed = time.time() - t_start
            rate = completed / elapsed if elapsed > 0 else 1e-9
            eta   = (len(work_items) - completed) / rate
            if completed % 24 == 0 or completed == len(work_items):
                print(
                    f"[{time.strftime('%H:%M:%S')}] {completed}/{len(work_items)} recordings | "
                    f"{len(X_all):,} clips | ETA {eta/60:.0f} min",
                    flush=True,
                )

    X = np.array(X_all, dtype=np.float64)
    y = np.array(y_all, dtype=np.int64)

    np.save(os.path.join(OUTPUT_DIR, "X_audio_clips.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y_audio_clips.npy"), y)
    with open(os.path.join(OUTPUT_DIR, "audio_clips_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_all, f, ensure_ascii=False, indent=2)

    print(f"\n[{time.strftime('%H:%M:%S')}] === DONE ===", flush=True)
    print(f"Total clips:     {len(X):,}", flush=True)
    print(f"Feature shape:   {X.shape}", flush=True)
    print(f"Skipped (silence/short): {total_skipped:,}", flush=True)
    print(f"Errors:          {total_errors}", flush=True)

    print("\nPer-raga clip counts:", flush=True)
    counts = Counter(y_all)
    for lbl, cnt in sorted(counts.items()):
        print(f"  [{lbl:2d}] {classes[lbl]:<30} {cnt:,}", flush=True)


if __name__ == "__main__":
    main()
