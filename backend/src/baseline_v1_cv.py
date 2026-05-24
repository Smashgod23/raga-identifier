"""
Honest re-baseline of v1 (sklearn MLP on 360-D pitch features).

The deployed v1 reports 84.4% on a random 80/20 split of 689 samples
(CompMusic + YouTube). That number is inflated relative to a true
held-out evaluation because it doesn't enforce that test recordings
were unseen at training time — and at the full-recording level, with
one feature per recording, a random split is fine in principle but
combining CompMusic and YouTube versions of the same raga doesn't
matter much for leakage. The more useful comparison is to evaluate
the *same training recipe* under k-fold stratified CV on the 480
CompMusic recordings, which is the dataset every published baseline
in the field uses (Chordia & Şentürk PCD, Gulati TDMS, etc.). That is
what this script does.

Output: data/baseline_v1_cv_report.txt
"""

from __future__ import annotations

import json
import time

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from eval_harness import recording_level_cv, load_classes, write_report


def make_v1_classifier():
    """Same hyperparameters as backend/src/train.py."""
    return Pipeline(
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


def main() -> None:
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    classes = load_classes()
    print(f"X={X.shape}  y={y.shape}  {len(classes)} ragas")

    t0 = time.time()
    result = recording_level_cv(
        X, y, make_v1_classifier, n_splits=5, seeds=(0, 1, 2, 3, 4)
    )
    elapsed = time.time() - t0

    top1_mean, top5_mean = result["top_k_mean"]
    top1_std, top5_std = result["top_k_std"]

    # Per-class accuracy from the summed confusion matrix.
    cm = result["confusion"]
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    ranked = np.argsort(per_class_acc)

    lines = [
        "=" * 72,
        "v1 baseline (sklearn MLP on 360-D pitch features), 5-fold CV",
        "Dataset: CompMusic Carnatic, 480 recordings × 40 ragas",
        f"Seeds: 5  |  Folds: 5  |  Elapsed: {elapsed:.1f}s",
        "=" * 72,
        "",
        "HEADLINE",
        f"  Top-1 accuracy: {top1_mean * 100:5.2f}% ± {top1_std * 100:.2f}%",
        f"  Top-5 accuracy: {top5_mean * 100:5.2f}% ± {top5_std * 100:.2f}%",
        "",
        "WORST 5 RAGAS (per-class top-1, summed across seeds and folds)",
    ]
    for idx in ranked[:5]:
        lines.append(
            f"  {classes[idx]:30s}  {per_class_acc[idx] * 100:5.1f}% "
            f"({cm[idx, idx]}/{cm[idx].sum()})"
        )
    lines += ["", "BEST 5 RAGAS"]
    for idx in ranked[::-1][:5]:
        lines.append(
            f"  {classes[idx]:30s}  {per_class_acc[idx] * 100:5.1f}% "
            f"({cm[idx, idx]}/{cm[idx].sum()})"
        )

    # Top confusion pairs.
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    flat = np.dstack(np.unravel_index(np.argsort(-cm_off, axis=None), cm.shape))[0]
    lines += ["", "TOP 10 CONFUSION PAIRS (true → predicted)"]
    seen = 0
    for i, j in flat:
        if cm_off[i, j] == 0:
            break
        lines.append(f"  {classes[i]:25s} → {classes[j]:25s}  {cm_off[i, j]}")
        seen += 1
        if seen >= 10:
            break

    write_report("data/baseline_v1_cv_report.txt", lines)
    print("\n".join(lines))
    print(f"\nSaved data/baseline_v1_cv_report.txt")


if __name__ == "__main__":
    main()
