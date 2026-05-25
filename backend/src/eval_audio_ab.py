"""
A/B benchmark: TDMS-from-audio vs v1's 360-D feature, on identical audio.

For each of the 480 CompMusic recordings, extracts a 60-second middle
window of the actual mp3, runs the deployed pyin+tonic pipeline once,
then computes both:
  * v1's 360-D PCD feature (the deployed model's input)
  * a 120x120 TDMS

Saves the resulting matrices and then evaluates four configurations
under the Phase-0 5-fold harness:

  1. v1 model (MLP) on v1 audio features
  2. TDMS k-NN (symmetric KL) on audio TDMSs
  3. TDMS k-NN (symmetric KL) on audio TDMSs, but with EXPERT-pitch
     templates as the training set (cross-source check)
  4. Same as 3 but vice versa (train on audio, test on expert) - sanity

Configuration 2 is the deployable target. Configuration 3 measures how
robust the templates are to runtime pitch-extraction differences.

Multi-process extraction is used because pyin is the bottleneck (~10s
per 60s clip). Re-running just the eval block without re-extraction is
supported by passing --skip-extract.

Outputs:
  data/X_tdms_audio.npy   (480, 14400) float32
  data/X_v1_audio.npy     (480, 360)   float32
  data/audio_features_meta.json
  data/eval_audio_ab_report.txt
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from build_tdms import compute_tdms, symmetric_kl_distance
from eval_harness import recording_level_cv, load_classes, write_report

AUDIO_ROOT = os.path.expanduser("~/raga-data-audio/RagaDataset/Carnatic/audio")
CLIP_DURATION = 60.0  # seconds


def audio_path_for(recording_id: str) -> str:
    piece = os.path.basename(recording_id)
    return os.path.join(AUDIO_ROOT, recording_id, f"{piece}.mp3")


def extract_features_one(args: tuple) -> Optional[dict]:
    """Worker: extract pitch once, compute both v1 and TDMS features.

    Imports librosa inside the function so subprocess startup doesn't
    serialize the whole numpy/librosa stack.
    """
    idx, recording_id, label = args
    import librosa
    from scipy.ndimage import uniform_filter1d

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build_tdms import compute_tdms
    from predict import _detect_tonic

    path = audio_path_for(recording_id)
    if not os.path.exists(path):
        return {"idx": idx, "error": f"missing audio: {path}"}

    try:
        total = librosa.get_duration(path=path)
    except Exception as exc:
        return {"idx": idx, "error": f"duration probe failed: {exc}"}

    offset = max(0.0, (total - CLIP_DURATION) / 2.0)
    duration = min(CLIP_DURATION, total)

    try:
        y, sr = librosa.load(path, sr=16000, mono=True, offset=offset, duration=duration)
    except Exception as exc:
        return {"idx": idx, "error": f"load failed: {exc}"}

    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    if rms > 1e-6:
        y = y * (0.1 / rms)

    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=60, fmax=800, sr=sr, frame_length=1024, hop_length=256
    )

    confident = voiced_flag & (voiced_probs > 0.3)
    voiced = f0[confident]
    if len(voiced) < 30:
        voiced = f0[voiced_flag]
    if len(voiced) < 30:
        return {"idx": idx, "error": "too little voiced pitch"}

    tonic = _detect_tonic(voiced)
    hop_seconds = 256.0 / sr

    # ---- v1 360-D PCD feature, identical to predict.py ----
    all_cents = 1200 * np.log2(voiced / tonic)
    slope = np.abs(np.gradient(all_cents, hop_seconds))
    stable_mask = slope < 1500
    min_frames = int(0.1 / hop_seconds)
    nyas_cents: list[float] = []
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
    nyas_cents_arr = np.array(nyas_cents) if len(nyas_cents) >= 10 else all_cents
    pcd_nyas, _ = np.histogram(nyas_cents_arr % 1200, bins=120, range=(0, 1200), density=True)
    pcd_duration, _ = np.histogram(all_cents % 1200, bins=120, range=(0, 1200), density=True)
    stable_cents = all_cents[slope < 3000]
    if len(stable_cents) < 10:
        stable_cents = all_cents
    pcd_stable, _ = np.histogram(stable_cents % 1200, bins=120, range=(0, 1200), density=True)
    v1_features = np.concatenate([pcd_nyas, pcd_duration, pcd_stable]).astype(np.float32)

    # ---- TDMS feature ----
    pitches_full = np.where(voiced_flag, f0, 0.0)
    pitches_full = np.where(np.isfinite(pitches_full), pitches_full, 0.0)
    timestamps = np.arange(len(pitches_full)) * hop_seconds
    tdms = compute_tdms(timestamps, pitches_full, tonic)
    if tdms is None:
        return {"idx": idx, "error": "tdms compute returned None"}

    return {
        "idx": idx,
        "recording_id": recording_id,
        "label": label,
        "tonic_hz": float(tonic),
        "v1_features": v1_features,
        "tdms": tdms.ravel(),
    }


def extract_all(meta: list[dict], n_workers: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Run extract_features_one for every recording, in parallel."""
    n = len(meta)
    X_v1 = np.zeros((n, 360), dtype=np.float32)
    X_tdms = np.zeros((n, 14400), dtype=np.float32)
    tonics = np.zeros(n, dtype=np.float32)
    errors: list[dict] = []
    out_meta: list[dict] = []

    args = [(i, m["recording_id"], m.get("label", -1)) for i, m in enumerate(meta)]
    t0 = time.time()
    done = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        for result in pool.map(extract_features_one, args, chunksize=1):
            done += 1
            if "error" in result:
                errors.append(result)
                print(f"  [{done}/{n}] ERROR idx={result['idx']}: {result['error']}")
                continue
            idx = result["idx"]
            X_v1[idx] = result["v1_features"]
            X_tdms[idx] = result["tdms"]
            tonics[idx] = result["tonic_hz"]
            out_meta.append(
                {
                    "idx": idx,
                    "recording_id": result["recording_id"],
                    "label": result["label"],
                    "tonic_hz": result["tonic_hz"],
                }
            )
            if done % 25 == 0 or done == n:
                rate = done / max(1e-6, time.time() - t0)
                eta = (n - done) / max(1e-6, rate)
                print(
                    f"  [{done}/{n}] {rate:.2f} rec/s   ETA {eta / 60:.1f} min"
                )

    if errors:
        print(f"\n{len(errors)} extraction errors. Continuing with successful subset.")
        for e in errors[:10]:
            print(f"  idx={e['idx']}: {e['error']}")

    return X_v1, X_tdms, out_meta


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class TDMSKnnDirect:
    """1-NN with symmetric KL, using directly stored TDMS feature matrices.

    Differs from the indices-based version in train_tdms.py: this one is fed
    feature vectors and is used when train and test come from different
    sources (so we cannot reuse a single precomputed distance matrix).
    """

    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.classes_ = np.arange(n_classes)

    def fit(self, X, y):
        self._train_X = np.asarray(X)
        self._train_y = np.asarray(y, dtype=np.int64)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        out = np.zeros((len(X), self.n_classes), dtype=np.float64)
        for i in range(len(X)):
            d = np.empty(len(self._train_X))
            for j in range(len(self._train_X)):
                d[j] = symmetric_kl_distance(X[i], self._train_X[j])
            order = np.argsort(d, kind="stable")
            seen: list[int] = []
            for j in order:
                cls = int(self._train_y[j])
                if cls in seen:
                    continue
                out[i, cls] = 1.0 / (len(seen) + 1)
                seen.append(cls)
                if len(seen) >= self.n_classes:
                    break
            s = out[i].sum()
            if s > 0:
                out[i] /= s
        return out


def make_tdms_knn_factory(n_classes: int):
    return lambda: TDMSKnnDirect(n_classes)


def make_v1_factory():
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return lambda: Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(256, 128, 64),
                    activation="relu",
                    alpha=0.01,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=42,
                    learning_rate="adaptive",
                    learning_rate_init=0.001,
                ),
            ),
        ]
    )


def cross_source_eval(X_train, y_train, X_test, y_test, classifier_factory, n_splits=5, seeds=(0, 1, 2, 3, 4)):
    """Train on full source A, test on each fold of source B (paired).

    Pratham's audio and expert features are aligned 1:1 by recording, so
    each fold of source B holds out the matching recordings from source A.
    """
    from sklearn.model_selection import StratifiedKFold

    n_classes = int(max(y_train.max(), y_test.max())) + 1
    top_k = (1, 5)
    per_fold = np.zeros((len(seeds), n_splits, len(top_k)), dtype=np.float64)

    for s_idx, seed in enumerate(seeds):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for f_idx, (train_idx, test_idx) in enumerate(skf.split(X_test, y_test)):
            clf = classifier_factory()
            # Held-out audio recordings are test_idx; train uses the
            # remaining ALIGNED expert-pitch templates plus the matching
            # train_idx from source B is excluded entirely. To keep the
            # cross-source story clean we use only source A train_idx.
            clf.fit(X_train[train_idx], y_train[train_idx])
            proba = clf.predict_proba(X_test[test_idx])
            cls = getattr(clf, "classes_", np.arange(n_classes))
            full = np.zeros((proba.shape[0], n_classes))
            full[:, cls] = proba
            for k_idx, k in enumerate(top_k):
                top = np.argsort(-full, axis=1)[:, :k]
                hits = (top == y_test[test_idx][:, None]).any(axis=1)
                per_fold[s_idx, f_idx, k_idx] = hits.mean()

    mean = per_fold.mean(axis=(0, 1))
    std = per_fold.mean(axis=1).std(axis=0)
    return {"top_k_mean": mean, "top_k_std": std}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    classes = load_classes()
    n_classes = len(classes)

    # Use tdms_meta as the canonical 480-recording list, since X_tdms is
    # indexed against it.
    with open("data/tdms_meta.json") as f:
        tdms_meta = json.load(f)
    y_expert = np.load("data/y_tdms.npy")
    assert len(tdms_meta) == len(y_expert) == 480

    # Inject label into meta for the worker
    for i, m in enumerate(tdms_meta):
        m["label"] = int(y_expert[i])

    if args.skip_extract and os.path.exists("data/X_tdms_audio.npy"):
        print("Loading cached audio features...")
        X_v1_audio = np.load("data/X_v1_audio.npy")
        X_tdms_audio = np.load("data/X_tdms_audio.npy")
    else:
        print(f"Extracting features from audio for {len(tdms_meta)} recordings, {args.workers} workers")
        X_v1_audio, X_tdms_audio, audio_meta = extract_all(tdms_meta, args.workers)
        np.save("data/X_v1_audio.npy", X_v1_audio)
        np.save("data/X_tdms_audio.npy", X_tdms_audio)
        with open("data/audio_features_meta.json", "w") as f:
            json.dump(audio_meta, f, ensure_ascii=False, indent=2)
        print("Saved data/X_v1_audio.npy, data/X_tdms_audio.npy, data/audio_features_meta.json")

    # Drop any rows that failed (all-zero features)
    keep = ~(np.all(X_tdms_audio == 0, axis=1) | np.all(X_v1_audio == 0, axis=1))
    X_v1_audio = X_v1_audio[keep]
    X_tdms_audio = X_tdms_audio[keep]
    X_tdms_expert = np.load("data/X_tdms.npy")[keep]
    y = y_expert[keep]
    n_kept = int(keep.sum())
    print(f"\nKept {n_kept}/480 recordings after dropping extraction failures")

    lines: list[str] = [
        "=" * 78,
        "TDMS-from-audio A/B benchmark vs v1 (Phase 2 step 1)",
        f"Source: live pyin + tonic heuristic on 60s middle window of each mp3",
        f"Recordings: {n_kept}/480",
        "=" * 78,
        "",
    ]

    print("\n[1/4] v1 MLP on audio-derived 360-D features (5-fold CV)...")
    t0 = time.time()
    cv_v1 = recording_level_cv(X_v1_audio, y, make_v1_factory(), verbose=False)
    print(f"    done in {time.time() - t0:.1f}s")
    m1, s1 = cv_v1["top_k_mean"], cv_v1["top_k_std"]
    lines.append(
        f"  v1 MLP on audio 360-D features                top1 {m1[0] * 100:5.2f}% +/- {s1[0] * 100:.2f}   "
        f"top5 {m1[1] * 100:5.2f}% +/- {s1[1] * 100:.2f}"
    )

    print("\n[2/4] TDMS k-NN (sym KL) on audio TDMS, 5-fold CV...")
    t0 = time.time()
    cv_tdms_audio = recording_level_cv(
        X_tdms_audio, y, make_tdms_knn_factory(n_classes), verbose=False
    )
    print(f"    done in {time.time() - t0:.1f}s")
    m2, s2 = cv_tdms_audio["top_k_mean"], cv_tdms_audio["top_k_std"]
    lines.append(
        f"  TDMS k-NN (sym KL) on audio TDMS              top1 {m2[0] * 100:5.2f}% +/- {s2[0] * 100:.2f}   "
        f"top5 {m2[1] * 100:5.2f}% +/- {s2[1] * 100:.2f}"
    )

    print("\n[3/4] Cross-source: train expert, test audio (5-fold CV)...")
    t0 = time.time()
    cs1 = cross_source_eval(
        X_tdms_expert, y, X_tdms_audio, y, make_tdms_knn_factory(n_classes)
    )
    print(f"    done in {time.time() - t0:.1f}s")
    m3, s3 = cs1["top_k_mean"], cs1["top_k_std"]
    lines.append(
        f"  TDMS k-NN: train EXPERT, test AUDIO           top1 {m3[0] * 100:5.2f}% +/- {s3[0] * 100:.2f}   "
        f"top5 {m3[1] * 100:5.2f}% +/- {s3[1] * 100:.2f}"
    )

    print("\n[4/4] Cross-source: train audio, test expert (5-fold CV)...")
    t0 = time.time()
    cs2 = cross_source_eval(
        X_tdms_audio, y, X_tdms_expert, y, make_tdms_knn_factory(n_classes)
    )
    print(f"    done in {time.time() - t0:.1f}s")
    m4, s4 = cs2["top_k_mean"], cs2["top_k_std"]
    lines.append(
        f"  TDMS k-NN: train AUDIO, test EXPERT           top1 {m4[0] * 100:5.2f}% +/- {s4[0] * 100:.2f}   "
        f"top5 {m4[1] * 100:5.2f}% +/- {s4[1] * 100:.2f}"
    )

    lines += [
        "",
        "Reference (Phase-0 / Phase-1, expert pitch, 5-fold CV):",
        "  v1 baseline (MLP on expert 360-D)             top1 71.79% +/- 0.84   top5 94.25% +/- 0.21",
        "  TDMS k-NN sym KL (expert templates)           top1 85.67% +/- 0.40   top5 97.58% +/- 0.21",
    ]
    write_report("data/eval_audio_ab_report.txt", lines)
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
