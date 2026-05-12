"""
Extract tonic-detection training data from the 480 CompMusic audio recordings.

For each recording:
  1. Load 60 s of audio at 16 kHz (skipping the first 10 s, which is often
     pure tanpura intro and gives misleading pitch statistics).
  2. Run pyin to get the pitch contour — the *same* pipeline production
     uses at inference time.
  3. Compute the histogram-based candidate-generation logic from
     predict.py._detect_tonic: fold voiced pitches into [60, 120] Hz, take
     the top 5 histogram peaks as candidate tonics, fold back to the
     singer's octave.
  4. For each candidate, compute features:
       - folded 120-bin pitch class distribution (under this candidate as Sa)
       - candidate frequency (Hz)
       - candidate's L2 peakedness score (existing heuristic's metric)
       - fraction of voiced frames within ±25 cents of the candidate
  5. Label each candidate as "correct" if it's within ±25 cents of the
     expert .tonicFine value (modulo octave), else "wrong".

Output:
  data/tonic_candidates.npz     Compressed array of features + labels
  data/tonic_candidates_meta.json   Per-recording metadata

This is fast for the 480 recordings (~5 s of pyin per recording at 16 kHz
on 60 s of audio, multiprocessing-parallel = ~10 min wall clock with 7
workers). The output is small (~5 MB) so the trainer can iterate quickly.

Run from backend/:
  source venv/bin/activate
  python src/extract_tonic_candidates.py
"""

import json
import math
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
CLASSES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "classes.json")
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")

OUT_NPZ  = os.path.join(OUTPUT_DIR, "tonic_candidates.npz")
OUT_META = os.path.join(OUTPUT_DIR, "tonic_candidates_meta.json")

SR              = 16000
EXTRACT_OFFSET  = 10.0   # skip first 10 s (tanpura-only intro)
EXTRACT_DUR     = 60.0   # use 60 s for tonic estimation
TOP_K           = 5      # number of candidate tonics per recording
N_BINS          = 120    # pitch class distribution resolution
TOLERANCE_CENTS = 25.0   # candidate is "correct" if within ±25 ¢ of expert


def tonicfine_path(audio_path):
    rel_no_ext = os.path.splitext(os.path.relpath(audio_path, AUDIO_DIR))[0]
    return os.path.join(FEAT_DIR, rel_no_ext + ".tonicFine")


def _candidates_from_voiced(voiced):
    """Replicate predict.py._detect_tonic exactly. Returns the top-K
    smoothed-histogram peaks as candidate tonics with their production
    peakedness scores. Keep this in lockstep with `_detect_tonic` — the
    learned re-ranker has to see the same candidate set at training time
    that production produces at inference time."""
    from scipy.ndimage import uniform_filter1d

    # Fold all voiced pitches into [60, 120] Hz octave for histogram.
    folded = voiced.copy()
    while np.any(folded > 120):
        folded = np.where(folded > 120, folded / 2, folded)
    while np.any(folded < 60):
        folded = np.where(folded < 60, folded * 2, folded)

    # Production: 200-bin histogram + uniform-filter smoothing (size=5).
    hist, edges = np.histogram(folded, bins=200, range=(60, 120))
    smoothed = uniform_filter1d(hist.astype(float), size=5)
    if smoothed.max() == 0:
        return []

    # Top K from the *smoothed* histogram, not raw.
    top_idx = np.argsort(smoothed)[::-1][:TOP_K]
    candidates = []
    median_pitch = float(np.median(voiced))

    for idx in top_idx:
        if smoothed[idx] == 0:
            continue
        cand_hz = float((edges[idx] + edges[idx + 1]) / 2)
        # Production folds the candidate UP toward the singer's octave only —
        # never down. Folding down was wrong; the singer rarely sings below
        # their Sa, so folding a low candidate down lands on a non-Sa pitch.
        while cand_hz * 2 < median_pitch:
            cand_hz *= 2

        # Production peakedness score: sum of squared bin counts on the
        # 120-bin cents-mod-1200 histogram (raw counts, no density flag).
        cents = 1200.0 * np.log2(voiced / cand_hz)
        h, _ = np.histogram(cents % 1200, bins=N_BINS, range=(0, 1200))
        peakedness = float(np.sum(h ** 2))
        candidates.append({
            "hz":         cand_hz,
            "peakedness": peakedness,
            "hist_bin":   int(idx),
            "hist_count": int(smoothed[idx]),  # smoothed weight at peak
        })
    return candidates


def _process_recording(args):
    """Extract candidates and features for one recording. Returns a dict."""
    audio_path, raga_id, raga_name, label, expert_tonic = args

    import librosa
    try:
        y, _ = librosa.load(audio_path, sr=SR, mono=True,
                            offset=EXTRACT_OFFSET, duration=EXTRACT_DUR)
    except Exception as e:
        return {"recording_id": "FAIL", "error": str(e)}

    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    if rms > 1e-6:
        y = y * (0.1 / rms)

    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=60, fmax=800, sr=SR,
        frame_length=1024, hop_length=256,
    )
    confident_mask = voiced_flag & (voiced_probs > 0.3)
    voiced = f0[confident_mask]
    if len(voiced) < 30:
        voiced = f0[voiced_flag]
    if len(voiced) < 30:
        return {"recording_id": "FAIL", "error": "not enough voiced"}

    voiced = voiced[~np.isnan(voiced)]
    if len(voiced) < 30:
        return {"recording_id": "FAIL", "error": "all NaN voiced"}

    candidates = _candidates_from_voiced(voiced)
    if not candidates:
        return {"recording_id": "FAIL", "error": "no candidates"}

    rel_no_ext = os.path.splitext(os.path.relpath(audio_path, AUDIO_DIR))[0]

    # For each candidate, compute features and a correctness label.
    cand_records = []
    for cand in candidates:
        # Folded 120-bin pitch class distribution under this candidate as Sa.
        cents = 1200.0 * np.log2(voiced / cand["hz"])
        cents_mod = cents % 1200
        pcd, _ = np.histogram(cents_mod, bins=N_BINS, range=(0, 1200), density=True)

        # Fraction of voiced time within ±25 cents of the candidate (i.e.
        # "voiced frames that look like Sa").
        sa_mask = np.abs(cents_mod - 0) < TOLERANCE_CENTS
        sa_mask |= np.abs(cents_mod - 1200) < TOLERANCE_CENTS
        sa_fraction = float(sa_mask.mean())

        # Correctness label: cent-distance to expert tonic, modulo octave.
        if cand["hz"] > 0 and expert_tonic > 0:
            diff = 1200.0 * math.log2(cand["hz"] / expert_tonic)
            diff_mod = ((diff + 600) % 1200) - 600  # nearest pc-difference
            is_correct = bool(abs(diff_mod) <= TOLERANCE_CENTS)
        else:
            diff_mod = float("nan")
            is_correct = False

        cand_records.append({
            "hz":          cand["hz"],
            "peakedness":  cand["peakedness"],
            "hist_count":  cand["hist_count"],
            "sa_fraction": sa_fraction,
            "pcd":         pcd.tolist(),
            "diff_cents":  diff_mod,
            "is_correct":  is_correct,
        })

    return {
        "recording_id": rel_no_ext,
        "raga_id":      raga_id,
        "raga_name":    raga_name,
        "label":        int(label),
        "expert_tonic": float(expert_tonic),
        "candidates":   cand_records,
    }


def main():
    with open(MAPPING_PATH, encoding="utf-8") as f:
        id_to_name = json.load(f)
    with open(CLASSES_PATH, encoding="utf-8") as f:
        classes = json.load(f)
    name_to_label = {n: i for i, n in enumerate(classes)}

    work_items = []
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
                tpath = tonicfine_path(apath)
                if not os.path.exists(tpath):
                    missing.append(os.path.relpath(apath, AUDIO_DIR))
                    continue
                with open(tpath) as f:
                    expert_tonic = float(f.read().strip())
                work_items.append((apath, raga_id, raga_name, label, expert_tonic))

    if missing:
        raise SystemExit(
            f"ABORT: {len(missing)} audio files missing .tonicFine: {missing[:5]}"
        )

    print(f"[{time.strftime('%H:%M:%S')}] {len(work_items)} recordings to process",
          flush=True)

    nproc = max(1, min(mp.cpu_count() - 1, 7))
    print(f"[{time.strftime('%H:%M:%S')}] Using {nproc} workers", flush=True)

    results = []
    completed = 0
    n_failed = 0
    t_start = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=nproc) as pool:
        for r in pool.imap_unordered(_process_recording, work_items, chunksize=1):
            completed += 1
            if r.get("recording_id") == "FAIL":
                n_failed += 1
                print(f"  FAIL: {r.get('error')}", flush=True)
            else:
                results.append(r)
            if completed % 24 == 0 or completed == len(work_items):
                elapsed = time.time() - t_start
                rate = completed / elapsed
                eta = (len(work_items) - completed) / rate
                print(f"[{time.strftime('%H:%M:%S')}] {completed}/{len(work_items)} | "
                      f"ok {len(results)} fail {n_failed} | ETA {eta/60:.0f} min",
                      flush=True)

    # Pack into numpy arrays for the trainer.
    X_rows = []           # per-candidate feature vectors
    y_rows = []           # per-candidate labels (1 if correct, 0 else)
    rec_idx_rows = []     # which recording this candidate came from
    cand_rank_rows = []   # 0..K-1, position in this recording's top-K
    rec_meta = []         # per-recording metadata

    for r in results:
        rec_idx = len(rec_meta)
        rec_meta.append({
            "recording_id": r["recording_id"],
            "raga_id":      r["raga_id"],
            "raga_name":    r["raga_name"],
            "label":        r["label"],
            "expert_tonic": r["expert_tonic"],
        })
        for ci, cand in enumerate(r["candidates"]):
            # Feature vector: [hz_log, peakedness, hist_count_log, sa_fraction]
            # + 120-bin folded pitch class distribution.
            feat = np.concatenate([
                np.array([
                    math.log(max(cand["hz"], 1e-3)),
                    cand["peakedness"],
                    math.log(max(cand["hist_count"], 1)),
                    cand["sa_fraction"],
                ], dtype=np.float32),
                np.array(cand["pcd"], dtype=np.float32),
            ])
            X_rows.append(feat)
            y_rows.append(1 if cand["is_correct"] else 0)
            rec_idx_rows.append(rec_idx)
            cand_rank_rows.append(ci)

    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.int64)
    rec_idx_arr   = np.asarray(rec_idx_rows, dtype=np.int64)
    cand_rank_arr = np.asarray(cand_rank_rows, dtype=np.int64)

    np.savez(OUT_NPZ, X=X, y=y, rec_idx=rec_idx_arr, cand_rank=cand_rank_arr)
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(rec_meta, f, ensure_ascii=False)

    # Diagnostic: how often is the top-by-peakedness candidate correct?
    # That's the existing heuristic's behavior.
    heuristic_correct = 0
    n_recordings_with_candidates = 0
    for r in results:
        if not r["candidates"]:
            continue
        n_recordings_with_candidates += 1
        # Existing heuristic picks the top peakedness.
        best = max(r["candidates"], key=lambda c: c["peakedness"])
        if best["is_correct"]:
            heuristic_correct += 1

    print(f"\n[{time.strftime('%H:%M:%S')}] === DONE ===", flush=True)
    print(f"  recordings:           {len(results)}", flush=True)
    print(f"  total candidates:     {len(X)}", flush=True)
    print(f"  positive (correct):   {int(y.sum())}", flush=True)
    print(f"  feature dim:          {X.shape[1]}", flush=True)
    print(f"  failed:               {n_failed}", flush=True)
    print(file=__import__('sys').stdout)
    print(f"  heuristic top-peakedness accuracy: "
          f"{heuristic_correct}/{n_recordings_with_candidates} = "
          f"{100*heuristic_correct/max(1, n_recordings_with_candidates):.1f}%", flush=True)
    # Upper bound: if a perfect re-ranker could pick the right candidate when one exists
    # in the top-K, what would the ceiling be?
    upper = 0
    for r in results:
        if any(c["is_correct"] for c in r["candidates"]):
            upper += 1
    print(f"  perfect re-ranker ceiling (correct ∈ top-{TOP_K}): "
          f"{upper}/{len(results)} = "
          f"{100*upper/max(1, len(results)):.1f}%", flush=True)


if __name__ == "__main__":
    main()
