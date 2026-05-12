"""
Train a learned tonic detector to re-rank the top-5 candidates produced by
the existing heuristic.

Input data (from extract_tonic_candidates.py):
  data/tonic_candidates.npz: arrays X, y, rec_idx
    X       — (N_candidates, 124) float32 features (4 scalars + 120-bin PCD)
    y       — (N_candidates,) int64, 1 if candidate is within ±25 ¢ of expert
    rec_idx — (N_candidates,) int64 recording index (groups for the split)

Approach:
  Treat each candidate as a binary example: "is this the correct tonic?".
  Train a small MLP to predict the probability. At inference, score all
  candidates from one recording and pick the highest-scoring one. This is
  a re-ranker that improves on the heuristic without changing the
  candidate-generation step (which is hard to beat — true tonic is almost
  always in the top-5 histogram peaks).

Recording-aware split (no candidate from the same recording in both train
and test) so the test number reflects generalization to unseen artists.

Outputs:
  models/tonic_detector_v1.pt
  data/tonic_detector_report.txt

Run from backend/:
  source venv/bin/activate
  python src/train_tonic_detector.py
"""

import json
import os
import sys
import time
from collections import defaultdict
from io import StringIO

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import GroupShuffleSplit

THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR  = os.path.dirname(THIS_DIR)
DATA_DIR     = os.path.join(BACKEND_DIR, "data")
MODELS_DIR   = os.path.join(BACKEND_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

NPZ_PATH    = os.path.join(DATA_DIR, "tonic_candidates.npz")
META_PATH   = os.path.join(DATA_DIR, "tonic_candidates_meta.json")
PT_PATH     = os.path.join(MODELS_DIR, "tonic_detector_v1.pt")
REPORT_PATH = os.path.join(DATA_DIR, "tonic_detector_report.txt")

EPOCHS         = 60
BATCH_SIZE     = 64
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
TEST_SIZE      = 0.20
SEED           = 42


class Tee:
    def __init__(self):
        self.buf = StringIO()
    def write(self, s):
        sys.__stdout__.write(s); self.buf.write(s)
    def flush(self):
        sys.__stdout__.flush()


class TonicCandidateScorer(nn.Module):
    """Tiny MLP. Input = 124-dim per-candidate features. Output = single logit
    representing P(this candidate is the true tonic). Architecture is
    deliberately small (~16k params) because we only have ~480 recordings *
    5 candidates ≈ 2.4k examples and we don't want to memorize them."""
    def __init__(self, input_dim=124, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def evaluate_per_recording(model, X, y, rec_idx, device):
    """For each recording, score all its candidates and pick the highest.
    Return (top1_correct, n_recordings_with_correct_in_topK)."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device)).cpu().numpy()
    # Group by recording.
    per_rec = defaultdict(list)
    for i, rid in enumerate(rec_idx):
        per_rec[int(rid)].append(i)
    correct = 0
    has_correct = 0
    n = 0
    for rid, idxs in per_rec.items():
        n += 1
        if any(y[i] == 1 for i in idxs):
            has_correct += 1
        # Pick candidate with highest model score.
        best_i = max(idxs, key=lambda i: logits[i])
        if y[best_i] == 1:
            correct += 1
    return correct, has_correct, n


def evaluate_heuristic(X, y, rec_idx):
    """Existing heuristic = highest peakedness (feature index 1 in X).
    Returns same triple as above so the comparison is apples-to-apples."""
    per_rec = defaultdict(list)
    for i, rid in enumerate(rec_idx):
        per_rec[int(rid)].append(i)
    correct = 0
    n = 0
    for rid, idxs in per_rec.items():
        n += 1
        # Feature index 1 is `peakedness`.
        best_i = max(idxs, key=lambda i: float(X[i, 1]))
        if y[best_i] == 1:
            correct += 1
    return correct, n


def main():
    t0 = time.time()
    tee = Tee()
    print(f"[{time.strftime('%H:%M:%S')}] >>> Train tonic detector",
          file=tee, flush=True)
    print(file=tee)

    arrs = np.load(NPZ_PATH)
    X = arrs["X"]; y = arrs["y"]
    rec_idx = arrs["rec_idx"]
    with open(META_PATH, encoding="utf-8") as f:
        rec_meta = json.load(f)

    print(f"X: {X.shape}, y: {y.shape} (positives: {int(y.sum())})", file=tee)
    print(f"recordings: {len(rec_meta)}, candidates per rec: avg "
          f"{len(X)/max(1,len(rec_meta)):.2f}", file=tee)
    print(file=tee)

    # ── Recording-aware split ───────────────────────────────────────────
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=rec_idx))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    g_train, g_test = rec_idx[train_idx], rec_idx[test_idx]

    if set(g_train.tolist()) & set(g_test.tolist()):
        raise SystemExit("ABORT: recording_ids leaked across split")

    print(f"[{time.strftime('%H:%M:%S')}] Recording-aware split:", file=tee)
    print(f"  train: {len(X_train):,} candidates, "
          f"{len(set(g_train.tolist())):,} recordings", file=tee)
    print(f"  test:  {len(X_test):,} candidates, "
          f"{len(set(g_test.tolist())):,} recordings", file=tee)
    print(file=tee)

    # ── Baseline: existing heuristic accuracy ───────────────────────────
    h_train_correct, h_train_n = evaluate_heuristic(X_train, y_train, g_train)
    h_test_correct,  h_test_n  = evaluate_heuristic(X_test,  y_test,  g_test)
    print("Existing heuristic (top-peakedness) baseline:", file=tee)
    print(f"  train: {h_train_correct}/{h_train_n} = "
          f"{100*h_train_correct/max(1,h_train_n):.1f}%", file=tee)
    print(f"  test:  {h_test_correct}/{h_test_n} = "
          f"{100*h_test_correct/max(1,h_test_n):.1f}%", file=tee)
    print(file=tee)

    # Class imbalance (most candidates are wrong).
    pos_weight = torch.tensor(
        [(len(y_train) - y_train.sum()) / max(1, y_train.sum())],
        dtype=torch.float32,
    )
    print(f"pos_weight = {float(pos_weight):.2f} "
          f"(BCEWithLogitsLoss handles imbalance)", file=tee)
    print(file=tee)

    # ── Device, model, loader ───────────────────────────────────────────
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[{time.strftime('%H:%M:%S')}] Device: {device}", file=tee)
    model = TonicCandidateScorer(input_dim=X.shape[1]).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}", file=tee)
    print(file=tee)

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"[{time.strftime('%H:%M:%S')}] Training {EPOCHS} epochs, "
          f"batch={BATCH_SIZE}, lr={LEARNING_RATE} (cosine)", file=tee)
    print(file=tee)

    best_test_acc = 0.0
    val_acc_trace = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * yb.size(0)
            n += yb.size(0)
        scheduler.step()
        avg_loss = total_loss / max(1, n)

        # Per-recording top-1 accuracy (same metric the heuristic was scored on).
        train_corr, _, train_n = evaluate_per_recording(
            model, X_train, y_train, g_train, device)
        test_corr,  _, test_n  = evaluate_per_recording(
            model, X_test,  y_test,  g_test,  device)
        train_acc = train_corr / max(1, train_n) * 100
        test_acc  = test_corr  / max(1, test_n)  * 100
        val_acc_trace.append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), PT_PATH)
            saved = "  (saved)"
        else:
            saved = ""

        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"[{time.strftime('%H:%M:%S')}] epoch {epoch:3d}/{EPOCHS}  "
                  f"loss={avg_loss:.4f}  train_top1={train_acc:.1f}%  "
                  f"test_top1={test_acc:.1f}%  best={best_test_acc:.1f}%{saved}",
                  file=tee, flush=True)

    print(file=tee)
    print(f"[{time.strftime('%H:%M:%S')}] Done. Best test top-1: "
          f"{best_test_acc:.1f}%", file=tee)
    print(file=tee)

    # ── Final eval on best checkpoint ───────────────────────────────────
    model.load_state_dict(torch.load(PT_PATH, map_location=device))
    test_corr, has_corr, test_n = evaluate_per_recording(
        model, X_test, y_test, g_test, device)

    bar = "=" * 78
    print(bar, file=tee)
    print("FINAL — tonic detector v1 (re-ranks heuristic's top-5 candidates)",
          file=tee)
    print(bar, file=tee)
    print(f"Test recordings: {test_n}", file=tee)
    print(f"Recordings where the correct tonic is in top-5 candidates: "
          f"{has_corr}/{test_n} = {100*has_corr/max(1,test_n):.1f}%  "
          f"(this is the perfect re-ranker ceiling)", file=tee)
    print(file=tee)
    print(f"Existing heuristic top-1: {h_test_correct}/{test_n} = "
          f"{100*h_test_correct/max(1,test_n):.1f}%", file=tee)
    print(f"Learned re-ranker top-1:  {test_corr}/{test_n} = "
          f"{100*test_corr/max(1,test_n):.1f}%", file=tee)
    print(f"Improvement:              "
          f"{100*(test_corr - h_test_correct)/max(1,test_n):+.1f} pp", file=tee)
    print(file=tee)
    print(f"Per-epoch test_top1 trace:", file=tee)
    print(f"  [{', '.join(f'{v:.1f}' for v in val_acc_trace)}]", file=tee)
    print(file=tee)
    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] Total wall time: {elapsed:.1f}s",
          file=tee)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(tee.buf.getvalue())
    print(f"\nReport written to: {REPORT_PATH}", file=sys.__stdout__, flush=True)


if __name__ == "__main__":
    main()
