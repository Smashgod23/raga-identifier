"""
Train and evaluate TDMS-based raga classifiers.

Five variants, all evaluated under the Phase-0 harness (5-fold stratified
CV with 5 seeds) so numbers are directly comparable to the v1 baseline:

  1. k-NN with Frobenius distance (paper's M_F)
  2. k-NN with symmetric KL    (paper's M_KL, jointly best in the paper)
  3. k-NN with Bhattacharyya   (paper's M_B,  jointly best in the paper)
  4. sklearn MLP on flat TDMS
  5. sklearn MLP on TDMS + 360-D pitch features (concat)

For the k-NN variants we also report leave-one-out accuracy to match the
paper's exact protocol, since LOO is essentially free here (no training:
just one 480x480 distance matrix per metric reused across all splits).

Output: data/train_tdms_report.txt
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from build_tdms import (
    bhattacharyya_distance,
    frobenius_distance,
    symmetric_kl_distance,
)
from eval_harness import recording_level_cv, load_classes, write_report


# ---------------------------------------------------------------------------
# k-NN classifier with custom matrix distance, supporting top-k rankings.
# ---------------------------------------------------------------------------

class TDMSNearestNeighbor:
    """1-NN by raw distance, but with a real top-k ranking.

    For a test point, sort all training points by distance and assign each
    class a score equal to `1 / (encounter_rank + 1)` based on the first
    training point of that class encountered in the sorted list. This gives
    paper-matching top-1 (nearest neighbor decides) while yielding a useful
    top-k for downstream comparison.

    Distances are precomputed against the entire labeled set once via the
    `set_global_distances` helper and reused across folds. fit() then just
    stores indices.
    """

    def __init__(self, dist_matrix: np.ndarray, all_labels: np.ndarray, n_classes: int):
        self.dist_matrix = dist_matrix
        self.all_labels = all_labels
        self.n_classes = n_classes
        self.classes_ = np.arange(n_classes)

    def fit(self, X_indices: np.ndarray, y: np.ndarray) -> "TDMSNearestNeighbor":
        self._train_idx = np.asarray(X_indices, dtype=np.int64).ravel()
        return self

    def predict_proba(self, X_indices: np.ndarray) -> np.ndarray:
        test_idx = np.asarray(X_indices, dtype=np.int64).ravel()
        # distances shape: (n_test, n_train_total) — slice to train rows only.
        d = self.dist_matrix[test_idx][:, self._train_idx]
        out = np.zeros((len(test_idx), self.n_classes), dtype=np.float64)
        for i in range(len(test_idx)):
            order = np.argsort(d[i], kind="stable")
            seen = set()
            rank = 0
            for j in order:
                cls = int(self.all_labels[self._train_idx[j]])
                if cls in seen:
                    continue
                out[i, cls] = 1.0 / (rank + 1)
                seen.add(cls)
                rank += 1
                if rank >= self.n_classes:
                    break
            s = out[i].sum()
            if s > 0:
                out[i] /= s
        return out


def make_factory_for_knn(dist_matrix, all_labels, n_classes):
    """Build a classifier_factory that returns fresh TDMSNearestNeighbor.

    The factory accepts indices instead of feature matrices in fit/predict,
    so we route X = indices through the harness. We achieve that by passing
    `np.arange(N).reshape(N, 1)` to the harness as the feature matrix.
    """
    def factory():
        return TDMSNearestNeighbor(dist_matrix, all_labels, n_classes)
    return factory


def build_distance_matrix(X: np.ndarray, metric) -> np.ndarray:
    n = len(X)
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            v = metric(X[i], X[j])
            d[i, j] = v
            d[j, i] = v
    return d


def leave_one_out_knn(dist_matrix: np.ndarray, y: np.ndarray) -> dict:
    """Paper's exact protocol: each recording tested against the other 479."""
    n = len(y)
    n_classes = int(y.max()) + 1
    correct_top1 = 0
    correct_top5 = 0
    for i in range(n):
        d = dist_matrix[i].copy()
        d[i] = np.inf  # exclude self
        order = np.argsort(d, kind="stable")
        seen: list[int] = []
        for j in order:
            cls = int(y[j])
            if cls not in seen:
                seen.append(cls)
                if len(seen) >= 5:
                    break
        if seen[0] == int(y[i]):
            correct_top1 += 1
        if int(y[i]) in seen[:5]:
            correct_top5 += 1
    return {"top1": correct_top1 / n, "top5": correct_top5 / n}


# ---------------------------------------------------------------------------
# MLP factories.
# ---------------------------------------------------------------------------

def make_mlp_factory(input_dim: int):
    def factory():
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(512, 128, 64),
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
    return factory


def summarize(name: str, result: dict, lines: list[str]) -> None:
    top1_m, top5_m = result["top_k_mean"]
    top1_s, top5_s = result["top_k_std"]
    lines.append(
        f"  {name:50s}  top1 {top1_m * 100:5.2f}% +/- {top1_s * 100:.2f}   "
        f"top5 {top5_m * 100:5.2f}% +/- {top5_s * 100:.2f}"
    )


def main() -> None:
    classes = load_classes()
    n_classes = len(classes)
    X_tdms = np.load("data/X_tdms.npy")
    y = np.load("data/y_tdms.npy")
    print(f"X_tdms={X_tdms.shape}  y={y.shape}  {n_classes} ragas")

    # Reshape rows into (eta, eta) matrices for matrix-aware distances.
    eta = int(np.sqrt(X_tdms.shape[1]))
    assert eta * eta == X_tdms.shape[1], "TDMS must be square"
    X_mat = X_tdms.reshape(-1, eta, eta)

    lines: list[str] = [
        "=" * 78,
        "TDMS evaluation, Phase 1",
        f"Dataset: CompMusic Carnatic, {len(X_tdms)} recordings x {n_classes} ragas",
        "Protocol: 5-fold stratified CV, 5 seeds (same harness as v1 baseline)",
        "=" * 78,
        "",
    ]

    # --- k-NN variants --------------------------------------------------
    metrics = [
        ("Frobenius", frobenius_distance),
        ("symmetric KL", symmetric_kl_distance),
        ("Bhattacharyya", bhattacharyya_distance),
    ]
    indices = np.arange(len(X_tdms)).reshape(-1, 1)  # placeholder for harness

    lines.append("k-NN (1-NN with top-k ranking by nearest-class-encounter):")
    for name, metric in metrics:
        t0 = time.time()
        print(f"\nBuilding distance matrix for {name}...")
        D = build_distance_matrix(X_mat, metric)
        print(f"  built in {time.time() - t0:.1f}s")

        factory = make_factory_for_knn(D, y, n_classes)
        cv = recording_level_cv(indices, y, factory, n_splits=5, seeds=(0, 1, 2, 3, 4), verbose=False)
        summarize(f"k-NN ({name})  5-fold CV", cv, lines)

        loo = leave_one_out_knn(D, y)
        lines.append(
            f"  {'k-NN (' + name + ')  leave-one-out':50s}  "
            f"top1 {loo['top1'] * 100:5.2f}%               "
            f"top5 {loo['top5'] * 100:5.2f}%"
        )

    # --- MLPs -----------------------------------------------------------
    lines += ["", "MLP variants:"]

    print("\nTraining MLP on flat TDMS (14400-D)...")
    t0 = time.time()
    cv = recording_level_cv(X_tdms, y, make_mlp_factory(X_tdms.shape[1]), verbose=False)
    print(f"  done in {time.time() - t0:.1f}s")
    summarize("MLP on flat TDMS", cv, lines)

    print("\nTraining MLP on TDMS + 360-D concat...")
    X_pcd = np.load("data/X.npy").astype(np.float32)
    if len(X_pcd) != len(X_tdms):
        raise SystemExit(
            f"Dimension mismatch: X.npy has {len(X_pcd)} rows but X_tdms has {len(X_tdms)}. "
            f"They must be aligned by recording. Re-run preprocess.py and extract_tdms_features.py."
        )
    X_concat = np.concatenate([X_tdms, X_pcd], axis=1)
    t0 = time.time()
    cv = recording_level_cv(X_concat, y, make_mlp_factory(X_concat.shape[1]), verbose=False)
    print(f"  done in {time.time() - t0:.1f}s")
    summarize("MLP on TDMS + 360-D concat", cv, lines)

    # --- Comparison to Phase 0 -----------------------------------------
    lines += [
        "",
        "Comparison to Phase 0 baseline:",
        f"  {'v1 (MLP on 360-D pitch)  5-fold CV':50s}  "
        f"top1 71.79% +/- 0.84   top5 94.25% +/- 0.21",
        "",
        "Paper reference (Gulati et al. ISMIR 2016, leave-one-out):",
        "  k-NN (symmetric KL)            CMD             top1 86.7%",
        "  k-NN (Bhattacharyya)           CMD             top1 86.7%",
        "  PCD baseline (Chordia+Senturk) CMD             top1 73.1%",
    ]

    write_report("data/train_tdms_report.txt", lines)
    print("\n" + "\n".join(lines))
    print("\nSaved data/train_tdms_report.txt")


if __name__ == "__main__":
    main()
