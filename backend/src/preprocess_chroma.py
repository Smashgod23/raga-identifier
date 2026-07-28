"""
Phase 16: high-resolution, tonic-normalizable CQT chromagrams for the 480
CompMusic recordings.

Why this is not a repeat of Phase 4 (mel-spectrogram CNN, 11.46%) or Phase 5
(AST embeddings, 3.12%). Both of those failed the same way: the representation
carried the singer's timbre, the room, and the microphone, so a model with 384
training recordings learned "which concert is this" instead of "which raga is
this". A chromagram removes exactly those confounders:

  - octave folding collapses register, so a male and a female singer performing
    the same raga land on the same axis
  - tonic normalization (a circular roll, applied at load time) removes the
    performer's chosen Sa
  - energy is pitch-class energy, not spectral envelope, so vowel and timbre
    information is mostly gone

What it keeps that the current best model (DeepSRGM on Melodia tokens) throws
away: polyphony. Melodia commits to one predominant f0 per frame; a chromagram
keeps the tanpura drone, the violin, and any harmony still visible in the same
frame. The drone in particular is a direct tonic cue, and tonic-hypothesis
selection is where the measured oracle headroom sits (69.6 actual vs 79.9
oracle on the 194-recording novel set).

Parameters are chosen to line up with the existing token pipeline so the two
representations are directly comparable and can be fused later:

  SR 16000, hop 704  -> 22.73 frames/sec, matching the ~22.5 Hz token rate
  CQT fmin C2 (65.4 Hz), 60 bins/octave (20 cents), 5 octaves (300 bins)
  folded to a 60-bin chroma, plus a per-frame log-energy channel

Storage is float16, roughly 1.6 MB per 10-minute recording, ~800 MB total.

Outputs:
  backend/data/chroma/{recording_id}/{file}.npz  with keys:
      chroma  (T, 60) float16, per-frame L1-normalized
      energy  (T,)    float16, log10 of the frame's total CQT magnitude
      tonic   scalar  float32, the expert .tonicFine value in Hz
      raga    str

Run from backend/:
  source venv/bin/activate
  python src/preprocess_chroma.py
"""

import json
import multiprocessing as mp
import os
import time

import numpy as np

AUDIO_DIR = os.path.expanduser("~/raga-data-audio/RagaDataset/Carnatic/audio")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FEAT_DIR = os.path.join(DATA, "RagaDataset", "Carnatic", "features")
MAPPING_PATH = os.path.join(
    DATA, "RagaDataset", "Carnatic", "_info_", "ragaId_to_ragaName_mapping.json"
)
OUT_DIR = os.path.join(DATA, "chroma")

SR = 16000
HOP = 704                 # 16000/704 = 22.73 fps, matches the token pipeline
BINS_PER_OCTAVE = 60      # 20 cents per bin
N_OCTAVES = 5
N_BINS = BINS_PER_OCTAVE * N_OCTAVES
FMIN_NOTE = "C2"          # 65.4 Hz, below the lowest Carnatic vocal Sa
MAX_ANALYZE_S = 1200.0    # same 20-minute cap the Essentia contour builder uses


def tonicfine_path(audio_path):
    rel_no_ext = os.path.splitext(os.path.relpath(audio_path, AUDIO_DIR))[0]
    return os.path.join(FEAT_DIR, rel_no_ext + ".tonicFine")


def out_path_for(audio_path):
    rel_no_ext = os.path.splitext(os.path.relpath(audio_path, AUDIO_DIR))[0]
    return os.path.join(OUT_DIR, rel_no_ext + ".npz")


def chroma_from_audio(y, sr=SR):
    """Fold a high-resolution CQT into a 60-bin pitch-class profile per frame.

    Returns (chroma (T,60) float32 L1-normalized, energy (T,) float32 log10).
    """
    import librosa

    cqt = np.abs(
        librosa.cqt(
            y=y, sr=sr, hop_length=HOP,
            fmin=librosa.note_to_hz(FMIN_NOTE),
            n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE,
        )
    )
    # Fold octaves: (300, T) -> (5, 60, T) -> sum over octaves -> (60, T)
    folded = cqt.reshape(N_OCTAVES, BINS_PER_OCTAVE, -1).sum(axis=0)

    total = folded.sum(axis=0)
    # Silence guard: frames with no energy would divide by ~0 and produce noise.
    quiet = total < 1e-8
    safe = np.where(quiet, 1.0, total)
    chroma = (folded / safe).T.astype(np.float32)
    chroma[quiet] = 0.0
    energy = np.log10(np.maximum(total, 1e-8)).astype(np.float32)
    return chroma, energy


def _process_recording(args):
    audio_path, raga_name = args
    out = out_path_for(audio_path)
    if os.path.exists(out):
        return audio_path, "skipped (exists)", 0.0

    import librosa
    try:
        y, _ = librosa.load(audio_path, sr=SR, mono=True)
    except Exception as e:
        return audio_path, f"FAIL load: {e}", 0.0

    if len(y) < SR * 5:
        return audio_path, "FAIL too short (<5s)", 0.0

    if len(y) > MAX_ANALYZE_S * SR:
        mid, half = len(y) // 2, int(MAX_ANALYZE_S * SR // 2)
        y = y[mid - half: mid + half]

    try:
        tonic = float(open(tonicfine_path(audio_path)).read().strip())
    except Exception as e:
        return audio_path, f"FAIL tonic: {e}", 0.0
    if not (50.0 < tonic < 500.0):
        return audio_path, f"FAIL tonic out of range: {tonic}", 0.0

    try:
        chroma, energy = chroma_from_audio(y)
    except Exception as e:
        return audio_path, f"FAIL cqt: {e}", 0.0

    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out,
        chroma=chroma.astype(np.float16),
        energy=energy.astype(np.float16),
        tonic=np.float32(tonic),
        raga=raga_name,
    )
    mb = os.path.getsize(out) / (1024 * 1024)
    return audio_path, f"ok ({chroma.shape[0]} frames)", mb


def collect_work():
    with open(MAPPING_PATH, encoding="utf-8") as f:
        id_to_name = json.load(f)

    work, missing = [], []
    for raga_id in sorted(os.listdir(AUDIO_DIR)):
        if raga_id not in id_to_name:
            continue
        raga_name = id_to_name[raga_id]
        for root, _, files in os.walk(os.path.join(AUDIO_DIR, raga_id)):
            for fname in sorted(files):
                if not fname.lower().endswith(".mp3"):
                    continue
                apath = os.path.join(root, fname)
                if not os.path.exists(tonicfine_path(apath)):
                    missing.append(os.path.relpath(apath, AUDIO_DIR))
                    continue
                work.append((apath, raga_name))
    return work, missing


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    work, missing = collect_work()
    if missing:
        print(f"ABORT: {len(missing)} audio files have no .tonicFine. "
              f"First 5: {missing[:5]}", flush=True)
        raise SystemExit(2)

    print(f"[{time.strftime('%H:%M:%S')}] {len(work)} recordings -> {OUT_DIR}",
          flush=True)
    nproc = max(1, min(mp.cpu_count() - 1, 7))
    print(f"[{time.strftime('%H:%M:%S')}] {nproc} workers", flush=True)

    done = skipped = failed = 0
    total_mb = 0.0
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=nproc) as pool:
        for apath, status, mb in pool.imap_unordered(_process_recording, work,
                                                     chunksize=1):
            done += 1
            total_mb += mb
            if status.startswith("FAIL"):
                failed += 1
                print(f"  FAIL {os.path.relpath(apath, AUDIO_DIR)}: {status}",
                      flush=True)
            elif status.startswith("skipped"):
                skipped += 1
            if done % 24 == 0 or done == len(work):
                el = time.time() - t0
                rate = done / el if el > 0 else 1e-9
                print(f"[{time.strftime('%H:%M:%S')}] {done}/{len(work)} | "
                      f"{total_mb:.0f} MB | ETA {(len(work)-done)/rate/60:.0f} min",
                      flush=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] === DONE ===", flush=True)
    print(f"  processed {len(work)-skipped-failed}, skipped {skipped}, "
          f"failed {failed}, {total_mb:.0f} MB on disk", flush=True)


if __name__ == "__main__":
    main()
