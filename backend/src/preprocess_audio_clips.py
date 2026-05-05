"""
Slice raw Carnatic audio recordings into overlapping fixed-length windows
and extract 360-dimensional pitch-class features for each window.

Feature extraction uses extract_features_from_audio from predict.py exactly —
same pyin pipeline, same 3-channel 360-d feature space as inference. The tonic
is supplied per-recording from CompMusic's expert-annotated .tonicFine files
(via tonic_override) instead of auto-detection, so all clips from a recording
share the singer's true Sa.

Outputs (paths derived from --output-prefix):
  backend/data/X_{prefix}.npy       — (N, 360) float64 feature matrix
  backend/data/y_{prefix}.npy       — (N,) int64 raga labels
  backend/data/{prefix}_meta.json   — per-row metadata for recording-aware splitting

Run from backend/:
  source venv/bin/activate
  # 30s/10s clips (the original v2 dataset):
  python src/preprocess_audio_clips.py --window-sec 30 --hop-sec 10 \
      --min-sec 20 --output-prefix audio_clips

  # 15s/5s windows (Phase 3 multi-scale):
  python src/preprocess_audio_clips.py --window-sec 15 --hop-sec 5 \
      --min-sec 10 --output-prefix 15s
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from collections import Counter

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
CLASSES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "classes.json")
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
PREDICT_DIR  = os.path.dirname(os.path.abspath(__file__))

def tonicfine_path(audio_path):
    """Map an audio .mp3 path to its parallel .tonicFine path under FEAT_DIR.
    Audio and features share the same relative tree; only base dir + extension differ."""
    rel_no_ext = os.path.splitext(os.path.relpath(audio_path, AUDIO_DIR))[0]
    return os.path.join(FEAT_DIR, rel_no_ext + ".tonicFine")


def _build_clip_offsets(total_dur, clip_duration, clip_hop, min_clip_duration):
    """Return list of (offset, actual_duration) for clips of `clip_duration` seconds
    with `clip_hop` seconds between starts. Final clip may be shorter than
    `clip_duration` if the recording ends early. A clip is only emitted if at
    least `min_clip_duration` seconds remain at its start."""
    clips = []
    offset = 0.0
    while offset + min_clip_duration <= total_dur:
        dur = min(clip_duration, total_dur - offset)
        clips.append((offset, dur))
        offset += clip_hop
    return clips


def _process_recording(args):
    """Worker function — runs in a spawned subprocess so all imports are local.

    The clip_specs list is precomputed in the parent so workers don't need to
    pull window/hop constants from this module's globals (under spawn the
    module is re-imported and CLI overrides wouldn't propagate)."""
    audio_path, raga_id, raga_name, label, expert_tonic, clip_specs = args

    # Local imports so spawned workers start cleanly without inheriting parent state.
    import sys as _sys
    _sys.path.insert(0, PREDICT_DIR)
    from predict import extract_features_from_audio

    rel_path = os.path.relpath(audio_path, AUDIO_DIR)
    # Use relative path (not just basename) as the unique recording ID.
    # Many recordings share the same song title across artists/ragas.
    recording_id = os.path.splitext(rel_path)[0]

    if not clip_specs:
        return [], 1, 0  # recording too short for this scale

    results = []
    n_skipped = 0
    n_errors  = 0

    for offset, dur in clip_specs:
        try:
            features, _ = extract_features_from_audio(
                audio_path,
                tonic_override=expert_tonic,
                offset=offset,
                duration=dur,
            )
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
                    "expert_tonic_hz": expert_tonic,
                }
            ))
        except ValueError:
            # Not enough voiced audio (silence, applause, tanpura drone) — expected
            n_skipped += 1
        except Exception:
            n_errors += 1

    return results, n_skipped, n_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-sec", type=float, required=True,
                        help="Window length in seconds (e.g. 30 for the v2 30s clips)")
    parser.add_argument("--hop-sec", type=float, required=True,
                        help="Hop between window starts in seconds")
    parser.add_argument("--min-sec", type=float, required=True,
                        help="Minimum remaining audio at the start of a window")
    parser.add_argument("--output-prefix", required=True,
                        help="Output names: X_<prefix>.npy, y_<prefix>.npy, "
                             "<prefix>_meta.json (e.g. 15s, 1min, 3min, audio_clips)")
    args = parser.parse_args()

    if not (args.hop_sec > 0 and args.window_sec > 0 and args.min_sec > 0):
        raise SystemExit("ABORT: window/hop/min must all be positive")
    if args.min_sec > args.window_sec:
        raise SystemExit("ABORT: --min-sec must be ≤ --window-sec")

    x_out    = os.path.join(OUTPUT_DIR, f"X_{args.output_prefix}.npy")
    y_out    = os.path.join(OUTPUT_DIR, f"y_{args.output_prefix}.npy")
    meta_out = os.path.join(OUTPUT_DIR, f"{args.output_prefix}_meta.json")

    # Refuse to silently overwrite a previous run's outputs. The user can
    # delete or rename them explicitly if a re-run is intended.
    for p in (x_out, y_out, meta_out):
        if os.path.exists(p):
            raise SystemExit(
                f"ABORT: {p} already exists. Move or delete it before re-running "
                f"(prefix={args.output_prefix})."
            )

    print(f"[{time.strftime('%H:%M:%S')}] Scale: window={args.window_sec}s "
          f"hop={args.hop_sec}s min={args.min_sec}s → prefix='{args.output_prefix}'",
          flush=True)

    with open(MAPPING_PATH, encoding="utf-8") as f:
        id_to_name = json.load(f)
    with open(CLASSES_PATH, encoding="utf-8") as f:
        classes = json.load(f)
    name_to_label = {n: i for i, n in enumerate(classes)}

    # Probe duration up-front so we can pre-build clip_specs per recording in
    # the parent process. Workers don't need module-level scale constants.
    import librosa  # local: not needed in workers

    work_items = []
    missing = []
    too_short = 0
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
                if not fname.lower().endswith(".mp3"):
                    continue
                apath = os.path.join(root, fname)
                tpath = tonicfine_path(apath)
                if not os.path.exists(tpath):
                    missing.append(os.path.relpath(apath, AUDIO_DIR))
                    continue
                with open(tpath) as f:
                    expert_tonic = float(f.read().strip())
                try:
                    total_dur = librosa.get_duration(path=apath)
                except Exception:
                    total_dur = 0.0
                clip_specs = _build_clip_offsets(
                    total_dur, args.window_sec, args.hop_sec, args.min_sec
                )
                if not clip_specs:
                    too_short += 1
                    # Still send the work item so we get an explicit "skipped" count.
                work_items.append(
                    (apath, raga_id, raga_name, label, expert_tonic, clip_specs)
                )

    if missing:
        print(
            f"ABORT: {len(missing)} audio files have no matching .tonicFine. "
            f"First 5: {missing[:5]}",
            flush=True,
        )
        raise SystemExit(2)

    print(
        f"[{time.strftime('%H:%M:%S')}] Loaded {len(work_items)}/480 expert tonics. "
        f"All recordings matched.",
        flush=True,
    )
    if too_short:
        print(
            f"[{time.strftime('%H:%M:%S')}] {too_short} recordings are shorter "
            f"than --min-sec={args.min_sec}s and will produce zero windows.",
            flush=True,
        )
    expected_clips = sum(len(w[5]) for w in work_items)
    print(
        f"[{time.strftime('%H:%M:%S')}] Expected total windows before silence "
        f"skips: {expected_clips:,}",
        flush=True,
    )

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

    np.save(x_out, X)
    np.save(y_out, y)
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta_all, f, ensure_ascii=False, indent=2)

    print(f"\n[{time.strftime('%H:%M:%S')}] === DONE ({args.output_prefix}) ===",
          flush=True)
    print(f"Total clips:     {len(X):,}", flush=True)
    print(f"Feature shape:   {X.shape}", flush=True)
    print(f"Skipped (silence/short): {total_skipped:,}", flush=True)
    print(f"Errors:          {total_errors}", flush=True)
    print(f"Outputs:", flush=True)
    print(f"  {x_out}", flush=True)
    print(f"  {y_out}", flush=True)
    print(f"  {meta_out}", flush=True)

    print("\nPer-raga clip counts:", flush=True)
    counts = Counter(y_all)
    for lbl, cnt in sorted(counts.items()):
        print(f"  [{lbl:2d}] {classes[lbl]:<30} {cnt:,}", flush=True)


if __name__ == "__main__":
    main()
