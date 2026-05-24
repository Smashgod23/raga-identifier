"""
Shared evaluation harness for raga-identification experiments.

Two modes:
  * Recording-level evaluation: one feature vector per full recording
    (CompMusic 480-recording setup). 5-fold stratified CV with multiple
    seeds. Reports top-1 and top-5 accuracy with mean and std across seeds.
  * Clip-to-recording evaluation: feature vectors are short clips with a
    `recording_id` field in metadata. Train recording-aware (no leakage),
    test predictions are aggregated per recording by averaging softmax
    probabilities. Reports per-clip top-1, per-recording-vote top-1 and
    top-5.

The recording-level mode is used in Phase 0 to re-baseline v1 and in
Phase 1 to evaluate TDMS. The clip-to-recording mode mirrors what the
deployed API does at inference time.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Callable

import numpy as np
from sklearn.model_selection import StratifiedKFold


def recording_level_cv(
    X: np.ndarray,
    y: np.ndarray,
    classifier_factory: Callable[[], "ClassifierProtocol"],
    *,
    n_splits: int = 5,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    top_k: tuple[int, ...] = (1, 5),
    verbose: bool = True,
) -> dict:
    """Stratified k-fold CV at the recording level.

    `classifier_factory` returns a fresh classifier each call. The classifier
    must implement `fit(X, y)` and `predict_proba(X)`.

    Returns a dict with keys: `top_k_mean`, `top_k_std`, `per_fold` (n_seeds x
    n_splits x len(top_k) array), `confusion` (summed across folds and seeds,
    using seed 0's last fold to pin order).
    """
    n_classes = int(y.max()) + 1
    per_fold = np.zeros((len(seeds), n_splits, len(top_k)), dtype=np.float64)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)

    for s_idx, seed in enumerate(seeds):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for f_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            clf = classifier_factory()
            clf.fit(X[train_idx], y[train_idx])
            proba = clf.predict_proba(X[test_idx])
            # Align proba columns to global class order (some classifiers drop
            # classes that are absent from a given training fold).
            cls = getattr(clf, "classes_", np.arange(n_classes))
            full = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
            full[:, cls] = proba

            for k_idx, k in enumerate(top_k):
                top = np.argsort(-full, axis=1)[:, :k]
                hits = (top == y[test_idx][:, None]).any(axis=1)
                per_fold[s_idx, f_idx, k_idx] = hits.mean()

            preds = full.argmax(axis=1)
            for t, p in zip(y[test_idx], preds):
                confusion[t, p] += 1

            if verbose:
                line = "  ".join(
                    f"top{k}={per_fold[s_idx, f_idx, k_idx] * 100:5.2f}%"
                    for k_idx, k in enumerate(top_k)
                )
                print(f"  seed={seed} fold={f_idx + 1}/{n_splits}  {line}")

    mean = per_fold.mean(axis=(0, 1))
    std = per_fold.mean(axis=1).std(axis=0)  # std across seeds, after averaging folds
    return {
        "top_k": top_k,
        "top_k_mean": mean,
        "top_k_std": std,
        "per_fold": per_fold,
        "confusion": confusion,
    }


def recording_aware_split(
    recording_ids: np.ndarray,
    y: np.ndarray,
    *,
    test_frac: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split clip indices so no recording_id appears in both train and test.

    Stratifies recordings by their raga label (a recording's label is the
    majority label across its clips). Returns (train_idx, test_idx).
    """
    rng = np.random.default_rng(seed)
    by_rec: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(recording_ids):
        by_rec[r].append(i)

    rec_to_label: dict[str, int] = {}
    for r, idxs in by_rec.items():
        labels, counts = np.unique(y[idxs], return_counts=True)
        rec_to_label[r] = int(labels[counts.argmax()])

    recs_by_label: dict[int, list[str]] = defaultdict(list)
    for r, lbl in rec_to_label.items():
        recs_by_label[lbl].append(r)

    test_recs: set[str] = set()
    for lbl, recs in recs_by_label.items():
        recs = list(recs)
        rng.shuffle(recs)
        n_test = max(1, int(round(len(recs) * test_frac)))
        test_recs.update(recs[:n_test])

    train_idx, test_idx = [], []
    for i, r in enumerate(recording_ids):
        (test_idx if r in test_recs else train_idx).append(i)
    return np.array(train_idx), np.array(test_idx)


def recording_vote_accuracy(
    proba_clip: np.ndarray,
    recording_ids: np.ndarray,
    y_clip: np.ndarray,
    *,
    top_k: tuple[int, ...] = (1, 5),
) -> dict:
    """Aggregate clip-level softmax probabilities per recording and score.

    A recording's prediction is the argmax of mean clip probabilities. Its
    true label is the majority label across its clips. Returns top-k accuracy
    over unique recordings.
    """
    by_rec: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(recording_ids):
        by_rec[r].append(i)

    n_classes = proba_clip.shape[1]
    per_rec_proba = []
    per_rec_label = []
    for r, idxs in by_rec.items():
        per_rec_proba.append(proba_clip[idxs].mean(axis=0))
        labels, counts = np.unique(y_clip[idxs], return_counts=True)
        per_rec_label.append(int(labels[counts.argmax()]))

    per_rec_proba = np.asarray(per_rec_proba)
    per_rec_label = np.asarray(per_rec_label)

    out = {}
    for k in top_k:
        top = np.argsort(-per_rec_proba, axis=1)[:, :k]
        hits = (top == per_rec_label[:, None]).any(axis=1)
        out[f"top{k}"] = float(hits.mean())
    out["n_recordings"] = len(per_rec_label)
    return out


def load_classes(path: str = "data/classes.json") -> list[str]:
    with open(path) as f:
        return json.load(f)


def write_report(path: str, lines: list[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
