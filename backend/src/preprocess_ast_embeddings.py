"""
Pre-compute AST (Audio Spectrogram Transformer) embeddings for the 480
CompMusic audio recordings.

This is the feature-extraction step for v5. Instead of training a CNN from
scratch (v4 — failed at this dataset size), we use a foundation model
(MIT/ast-finetuned-audioset-10-10-0.4593) pretrained on 2 million 10-second
AudioSet clips. The pretrained model already knows what voice, instruments,
room acoustics, and microphones sound like — so when we then train a small
classifier head on its 768-dim embeddings, the head only has to learn
raga-specific patterns from our 480 recordings.

Pipeline:
  - Load each MP3 at 16 kHz mono.
  - Slice into 10.24-second windows with 5-second hop (10.24s matches AST's
    pretrained positional embeddings exactly; hop=5s keeps clips
    overlapping enough to handle gamakam patterns at the boundary).
  - For each batch of windows, run AST.forward() and grab the pooled CLS
    embedding (768-dim).
  - Save data/X_ast.npy + y_ast.npy + ast_meta.json.

Outputs:
  data/X_ast.npy        (N_windows, 768) float32
  data/y_ast.npy        (N_windows,) int64
  data/ast_meta.json    per-window {recording_id, label, raga_name,
                                     window_start_sec}

Run from backend/:
  source venv/bin/activate
  python src/preprocess_ast_embeddings.py
"""

import json
import os
import sys
import time

import numpy as np
import torch
import librosa
from transformers import ASTModel, ASTFeatureExtractor

THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR  = os.path.dirname(THIS_DIR)
DATA_DIR     = os.path.join(BACKEND_DIR, "data")
AUDIO_DIR    = os.path.expanduser("~/raga-data-audio/RagaDataset/Carnatic/audio")
FEAT_DIR     = os.path.join(DATA_DIR, "RagaDataset", "Carnatic", "features")
MAPPING_PATH = os.path.join(DATA_DIR, "RagaDataset", "Carnatic", "_info_",
                            "ragaId_to_ragaName_mapping.json")
CLASSES_PATH = os.path.join(DATA_DIR, "classes.json")

X_OUT     = os.path.join(DATA_DIR, "X_ast.npy")
Y_OUT     = os.path.join(DATA_DIR, "y_ast.npy")
META_OUT  = os.path.join(DATA_DIR, "ast_meta.json")

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR             = 16000
WINDOW_SEC     = 10.24      # exact AST positional embedding length
HOP_SEC        = 5.0
WINDOW_SAMPLES = int(WINDOW_SEC * SR)
HOP_SAMPLES    = int(HOP_SEC    * SR)
BATCH_SIZE     = 16         # tunable; bigger = faster on MPS but more memory


def tonicfine_path(audio_path):
    rel_no_ext = os.path.splitext(os.path.relpath(audio_path, AUDIO_DIR))[0]
    return os.path.join(FEAT_DIR, rel_no_ext + ".tonicFine")


def main():
    if os.path.exists(X_OUT):
        print(f"ABORT: {X_OUT} already exists. Move/delete it before re-running.")
        sys.exit(2)

    with open(MAPPING_PATH, encoding="utf-8") as f:
        id_to_name = json.load(f)
    with open(CLASSES_PATH, encoding="utf-8") as f:
        classes = json.load(f)
    name_to_label = {n: i for i, n in enumerate(classes)}

    # Build the recording list (identical filter to preprocess_audio_clips).
    recordings = []
    missing = []
    for raga_id in sorted(os.listdir(AUDIO_DIR)):
        if raga_id not in id_to_name:
            continue
        raga_name = id_to_name[raga_id]
        if raga_name not in name_to_label:
            continue
        label = name_to_label[raga_name]
        raga_dir = os.path.join(AUDIO_DIR, raga_id)
        for root, _, files in os.walk(raga_dir):
            for fname in sorted(files):
                if not fname.lower().endswith(".mp3"):
                    continue
                apath = os.path.join(root, fname)
                if not os.path.exists(tonicfine_path(apath)):
                    missing.append(os.path.relpath(apath, AUDIO_DIR))
                    continue
                rel_no_ext = os.path.splitext(os.path.relpath(apath, AUDIO_DIR))[0]
                recordings.append({
                    "audio_path":   apath,
                    "recording_id": rel_no_ext,
                    "raga_name":    raga_name,
                    "label":        label,
                })
    if missing:
        print(f"ABORT: {len(missing)} audio files have no matching .tonicFine. "
              f"First 5: {missing[:5]}")
        sys.exit(2)

    print(f"[{time.strftime('%H:%M:%S')}] {len(recordings)} recordings to process",
          flush=True)

    # Load AST model once.
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[{time.strftime('%H:%M:%S')}] Loading {MODEL_NAME} → {device}",
          flush=True)
    feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_NAME)
    model = ASTModel.from_pretrained(MODEL_NAME).to(device).eval()
    print(f"[{time.strftime('%H:%M:%S')}] Model loaded.", flush=True)

    X_all = []
    y_all = []
    meta_all = []
    total_windows = 0
    t_start = time.time()

    with torch.no_grad():
        for r_idx, rec in enumerate(recordings):
            try:
                audio, _ = librosa.load(rec["audio_path"], sr=SR, mono=True)
            except Exception as e:
                print(f"  FAIL load {rec['recording_id']}: {e}", flush=True)
                continue

            if len(audio) < WINDOW_SAMPLES:
                # Recording shorter than window — skip.
                continue

            # Build window offsets for this recording.
            offsets = []
            s = 0
            while s + WINDOW_SAMPLES <= len(audio):
                offsets.append(s)
                s += HOP_SAMPLES

            # Process windows in batches.
            for b_start in range(0, len(offsets), BATCH_SIZE):
                batch_offsets = offsets[b_start:b_start + BATCH_SIZE]
                batch_audio = [audio[off:off + WINDOW_SAMPLES] for off in batch_offsets]

                # Feature extractor expects list of 1D float arrays.
                inputs = feature_extractor(
                    batch_audio, sampling_rate=SR, return_tensors="pt"
                )
                input_values = inputs.input_values.to(device)

                outputs = model(input_values)
                # AST returns last_hidden_state and pooler_output. We use the
                # pooler (CLS-token-based, 768-dim).
                emb = outputs.pooler_output.cpu().numpy()  # (batch, 768)

                for j, off in enumerate(batch_offsets):
                    X_all.append(emb[j])
                    y_all.append(rec["label"])
                    meta_all.append({
                        "recording_id":     rec["recording_id"],
                        "raga_name":        rec["raga_name"],
                        "label":            rec["label"],
                        "window_start_sec": off / SR,
                        "window_dur_sec":   WINDOW_SEC,
                    })
                total_windows += len(batch_offsets)

            if (r_idx + 1) % 24 == 0 or (r_idx + 1) == len(recordings):
                elapsed = time.time() - t_start
                rate = (r_idx + 1) / elapsed
                eta = (len(recordings) - r_idx - 1) / rate
                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"{r_idx + 1}/{len(recordings)} recordings | "
                      f"{total_windows:,} windows | ETA {eta/60:.0f} min",
                      flush=True)

    X = np.asarray(X_all, dtype=np.float32)
    y = np.asarray(y_all, dtype=np.int64)
    np.save(X_OUT, X)
    np.save(Y_OUT, y)
    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(meta_all, f, ensure_ascii=False)

    print(f"\n[{time.strftime('%H:%M:%S')}] === DONE ===", flush=True)
    print(f"Total windows: {len(X):,}", flush=True)
    print(f"Embedding shape: {X.shape}", flush=True)
    print(f"Outputs:")
    print(f"  {X_OUT}")
    print(f"  {Y_OUT}")
    print(f"  {META_OUT}")


if __name__ == "__main__":
    main()
