"""
Pre-compute log-Mel spectrograms for the 480 CompMusic audio recordings.

One .npy per recording, full duration. Training (train_v4.py) loads these
into memory and slices windows on the fly — much faster than re-computing
spectrograms during every epoch.

Spectrogram parameters chosen so every recording fits comfortably in memory:
  sr=16000, n_fft=2048, hop_length=512, n_mels=128
  → ~1,875 mel frames per 60 seconds of audio
  → ~9,375 frames per 10-minute recording
  → ~4.8 MB per recording at float32
  → ~2.3 GB total for the 480-recording set

Outputs:
  backend/data/melspec/{relative_recording_path}.npy  (one file per recording)

Run from backend/:
  source venv/bin/activate
  python src/preprocess_melspec.py
"""

import json
import multiprocessing as mp
import os
import time

import numpy as np

AUDIO_DIR = os.path.expanduser("~/raga-data-audio/RagaDataset/Carnatic/audio")
FEAT_DIR  = os.path.join(
    os.path.dirname(__file__), "..", "data",
    "RagaDataset", "Carnatic", "features"
)
MAPPING_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data",
    "RagaDataset", "Carnatic", "_info_", "ragaId_to_ragaName_mapping.json"
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "melspec")

SR         = 16000
N_FFT      = 2048
HOP_LENGTH = 512
N_MELS     = 128


def tonicfine_path(audio_path):
    """Same convention as preprocess_audio_clips.py — gate by .tonicFine
    presence so we only process recordings the rest of the pipeline knows
    about."""
    rel_no_ext = os.path.splitext(os.path.relpath(audio_path, AUDIO_DIR))[0]
    return os.path.join(FEAT_DIR, rel_no_ext + ".tonicFine")


def out_path_for(audio_path):
    rel_no_ext = os.path.splitext(os.path.relpath(audio_path, AUDIO_DIR))[0]
    return os.path.join(OUT_DIR, rel_no_ext + ".npy")


def _process_recording(args):
    """Worker — load the MP3, compute log-Mel, save as .npy."""
    audio_path, raga_name = args
    out = out_path_for(audio_path)

    # Skip if already computed (idempotent).
    if os.path.exists(out):
        return audio_path, "skipped (exists)", 0.0

    import librosa  # local import for spawn workers
    try:
        y, sr = librosa.load(audio_path, sr=SR, mono=True)
    except Exception as e:
        return audio_path, f"FAIL load: {e}", 0.0

    if len(y) < SR * 5:
        return audio_path, "FAIL too short (<5s)", 0.0

    # Normalize amplitude (matches predict.py's RMS normalization at inference).
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms > 1e-6:
        y = y * (0.1 / rms)

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, log_mel)
    return audio_path, f"ok ({log_mel.shape[1]} frames)", float(log_mel.nbytes) / (1024 * 1024)


def main():
    with open(MAPPING_PATH, encoding="utf-8") as f:
        id_to_name = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)

    work_items = []
    missing_tonic = []
    for raga_id in sorted(os.listdir(AUDIO_DIR)):
        if raga_id not in id_to_name:
            continue
        raga_name = id_to_name[raga_id]
        raga_dir = os.path.join(AUDIO_DIR, raga_id)
        for root, _, files in os.walk(raga_dir):
            for fname in sorted(files):
                if not fname.lower().endswith(".mp3"):
                    continue
                apath = os.path.join(root, fname)
                if not os.path.exists(tonicfine_path(apath)):
                    missing_tonic.append(os.path.relpath(apath, AUDIO_DIR))
                    continue
                work_items.append((apath, raga_name))

    if missing_tonic:
        print(f"ABORT: {len(missing_tonic)} audio files have no matching "
              f".tonicFine. First 5: {missing_tonic[:5]}", flush=True)
        raise SystemExit(2)

    print(f"[{time.strftime('%H:%M:%S')}] {len(work_items)} recordings to process. "
          f"Output dir: {OUT_DIR}", flush=True)

    nproc = max(1, min(mp.cpu_count() - 1, 7))
    print(f"[{time.strftime('%H:%M:%S')}] Using {nproc} workers.", flush=True)

    completed = 0
    total_mb  = 0.0
    n_skipped = 0
    n_failed  = 0
    t_start = time.time()

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=nproc) as pool:
        for audio_path, status, mb in pool.imap_unordered(
            _process_recording, work_items, chunksize=1
        ):
            completed += 1
            total_mb  += mb
            if status.startswith("FAIL"):
                n_failed += 1
                print(f"  FAIL {os.path.relpath(audio_path, AUDIO_DIR)}: {status}",
                      flush=True)
            elif status.startswith("skipped"):
                n_skipped += 1

            if completed % 24 == 0 or completed == len(work_items):
                elapsed = time.time() - t_start
                rate = completed / elapsed if elapsed > 0 else 1e-9
                eta = (len(work_items) - completed) / rate
                print(f"[{time.strftime('%H:%M:%S')}] {completed}/{len(work_items)} | "
                      f"{total_mb:.0f} MB written | ETA {eta/60:.0f} min",
                      flush=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] === DONE ===", flush=True)
    print(f"  Processed: {len(work_items) - n_skipped - n_failed}", flush=True)
    print(f"  Already had output (skipped): {n_skipped}", flush=True)
    print(f"  Failed: {n_failed}", flush=True)
    print(f"  Total mel-spec data on disk: {total_mb:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
