"""
Step D: train v2 model on the combined dataset with recording-aware splitting
and class balancing.

Key differences from train.py (v1):
  - Trains on data/X_combined.npy (44,760 rows) instead of X.npy + X_yt.npy
  - GroupShuffleSplit(80/20) on recording_id from combined_meta.json — all
    rows from one MP3 (the single CompMusic X.npy row + ~92 audio clips)
    stay on the same side of the split. A naive random split would leak
    overlapping 30s clips from the same recording across train/test.
  - PyTorch trains with weighted CrossEntropyLoss (class_weights computed
    as n_samples / (n_classes * bincount(y_train))) to handle the 8.25×
    class imbalance.
  - sklearn MLPClassifier uses early_stopping=False (its default 15%
    validation_fraction would carve random rows from training, ignoring
    recording boundaries). Class balance is handled by oversampling
    minority classes to the median train count, since MLPClassifier
    doesn't support class_weight or sample_weight.
  - StandardScaler is fit on TRAIN ONLY, not on the full combined matrix —
    fitting on all data would leak test-set variance into the scaler.
  - Outputs go to *_v2 paths so v1 artifacts are preserved.

Outputs:
  models/raga_model_best_v2.pt
  models/raga_sklearn_v2.pkl
  models/scaler_v2.pkl
  data/train_v2_report.txt          (full text report — same content as stdout)

Run from backend/:
  source venv/bin/activate
  python src/train_v2.py
"""

import json
import os
import pickle
import sys
import time
from collections import Counter
from io import StringIO

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(THIS_DIR)
DATA_DIR    = os.path.join(BACKEND_DIR, "data")
MODELS_DIR  = os.path.join(BACKEND_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

X_PATH       = os.path.join(DATA_DIR, "X_combined.npy")
Y_PATH       = os.path.join(DATA_DIR, "y_combined.npy")
META_PATH    = os.path.join(DATA_DIR, "combined_meta.json")
CLASSES_PATH = os.path.join(DATA_DIR, "classes.json")

PT_PATH      = os.path.join(MODELS_DIR, "raga_model_best_v2.pt")
SK_PATH      = os.path.join(MODELS_DIR, "raga_sklearn_v2.pkl")
SCALER_PATH  = os.path.join(MODELS_DIR, "scaler_v2.pkl")
REPORT_PATH  = os.path.join(DATA_DIR, "train_v2_report.txt")

EPOCHS         = 200
BATCH_SIZE     = 128         # bigger than v1's 32 — 44k rows benefits from larger batches
LEARNING_RATE  = 0.001
WEIGHT_DECAY   = 1e-4
LR_STEP        = 50
LR_GAMMA       = 0.5
TEST_SIZE      = 0.20
SEED           = 42


# ── helpers ────────────────────────────────────────────────────────────────

class Tee:
    def __init__(self):
        self.buf = StringIO()
    def write(self, s):
        sys.__stdout__.write(s); self.buf.write(s)
    def flush(self):
        sys.__stdout__.flush()


class RagaNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, device):
    """Return (accuracy, all_preds, all_labels)."""
    model.eval()
    correct = total = 0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)
            preds_all.append(preds.cpu().numpy())
            labels_all.append(yb.cpu().numpy())
    return (correct / total * 100,
            np.concatenate(preds_all),
            np.concatenate(labels_all))


def oversample_to_median(X, y, seed):
    """Bring every minority class up to the median per-class count by random
    resampling with replacement. Workaround for MLPClassifier's missing
    class_weight/sample_weight support."""
    counts = np.bincount(y)
    target = int(np.median(counts[counts > 0]))
    Xs, ys = [], []
    for cls in range(len(counts)):
        mask = (y == cls)
        n = int(mask.sum())
        if n == 0:
            continue
        if n < target:
            X_more = resample(X[mask], n_samples=(target - n),
                              random_state=seed + cls, replace=True)
            Xs.append(X[mask]); Xs.append(X_more)
            ys.append(np.full(n, cls)); ys.append(np.full(target - n, cls))
        else:
            Xs.append(X[mask]); ys.append(np.full(n, cls))
    Xb = np.concatenate(Xs); yb = np.concatenate(ys)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xb))
    return Xb[perm], yb[perm]


# ── main ───────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    tee = Tee()
    print(f"[{time.strftime('%H:%M:%S')}] >>> Step D: train_v2 — recording-aware split + class balancing",
          file=tee, flush=True)
    print(file=tee)

    # ── Load ────────────────────────────────────────────────────────────
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    with open(CLASSES_PATH, encoding="utf-8") as f:
        classes = json.load(f)

    if not (len(X) == len(y) == len(meta)):
        raise SystemExit(
            f"ABORT: row count mismatch — X={len(X)}, y={len(y)}, meta={len(meta)}"
        )

    n_classes = len(classes)
    groups   = np.array([m["recording_id"] for m in meta])
    sources  = np.array([m["source"]       for m in meta])

    print(f"X:        {X.shape}", file=tee)
    print(f"y:        {y.shape}", file=tee)
    print(f"classes:  {n_classes}", file=tee)
    print(f"sources:  {dict(Counter(sources.tolist()))}", file=tee)
    print(f"unique recording_ids: {len(set(groups))}", file=tee)
    print(file=tee)

    # ── Recording-aware split ───────────────────────────────────────────
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    g_train, g_test = groups[train_idx], groups[test_idx]
    s_train, s_test = sources[train_idx], sources[test_idx]

    leak = set(g_train) & set(g_test)
    if leak:
        raise SystemExit(f"ABORT: {len(leak)} recording_ids leaked across split")

    print(f"[{time.strftime('%H:%M:%S')}] Recording-aware split:", file=tee)
    print(f"  train: {len(X_train):,} rows, {len(set(g_train)):,} unique recordings",
          file=tee)
    print(f"  test:  {len(X_test):,} rows, {len(set(g_test)):,} unique recordings",
          file=tee)
    print(f"  test fraction (rows): {len(X_test) / len(X):.3f}", file=tee)
    print(file=tee)

    # ── Scale (train only) ──────────────────────────────────────────────
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[{time.strftime('%H:%M:%S')}] Saved scaler → {SCALER_PATH}", file=tee)
    print(file=tee)

    # ── Class weights for PyTorch ───────────────────────────────────────
    # n_samples / (n_classes * bincount(y_train)) — the spec the user gave.
    train_counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    safe_counts  = np.where(train_counts == 0, 1, train_counts)
    class_w_np   = len(y_train) / (n_classes * safe_counts)
    class_w_t    = torch.tensor(class_w_np, dtype=torch.float32)

    print(f"Class weights (min/max/mean): "
          f"{class_w_np.min():.3f} / {class_w_np.max():.3f} / {class_w_np.mean():.3f}",
          file=tee)
    print(file=tee)

    # ── PyTorch training loop ───────────────────────────────────────────
    device = torch.device("cpu")
    Xtr_t = torch.tensor(X_train_s, dtype=torch.float32)
    ytr_t = torch.tensor(y_train,    dtype=torch.long)
    Xte_t = torch.tensor(X_test_s,  dtype=torch.float32)
    yte_t = torch.tensor(y_test,     dtype=torch.long)

    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t),
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(TensorDataset(Xte_t, yte_t),
                              batch_size=BATCH_SIZE, shuffle=False)

    model = RagaNet(input_dim=360, num_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=class_w_t.to(device))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_STEP, gamma=LR_GAMMA)

    print(f"[{time.strftime('%H:%M:%S')}] Training PyTorch model — "
          f"{EPOCHS} epochs, batch={BATCH_SIZE}, lr={LEARNING_RATE}", file=tee)
    print(f"  logging every 5 epochs (loss / train acc / val acc)", file=tee)
    print(file=tee)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        train_correct = train_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss     += loss.item() * yb.size(0)
            train_correct  += (logits.argmax(dim=1) == yb).sum().item()
            train_total    += yb.size(0)
        scheduler.step()

        train_acc = train_correct / train_total * 100
        avg_loss  = total_loss / train_total

        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            val_acc, _, _ = evaluate(model, test_loader, device)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), PT_PATH)
                tag = "  (saved)"
            else:
                tag = ""
            print(f"[{time.strftime('%H:%M:%S')}] epoch {epoch:3d}/{EPOCHS}  "
                  f"loss={avg_loss:.4f}  train_acc={train_acc:.2f}%  "
                  f"val_acc={val_acc:.2f}%  best={best_acc:.2f}%{tag}",
                  file=tee, flush=True)

    print(file=tee)
    print(f"[{time.strftime('%H:%M:%S')}] PyTorch training done. Best val acc: "
          f"{best_acc:.2f}%", file=tee)
    print(f"  saved → {PT_PATH}", file=tee)
    print(file=tee)

    # Reload best checkpoint for the final eval.
    model.load_state_dict(torch.load(PT_PATH, map_location=device))
    final_acc, preds, labels = evaluate(model, test_loader, device)
    print(f"[{time.strftime('%H:%M:%S')}] PyTorch final test acc: {final_acc:.2f}%",
          file=tee)
    print(file=tee)

    # ── sklearn MLPClassifier for deployment ───────────────────────────
    print(f"[{time.strftime('%H:%M:%S')}] Training sklearn MLPClassifier (deployment copy)",
          file=tee)
    print(f"  early_stopping=False, oversample minority classes to median train count",
          file=tee)
    X_train_bal, y_train_bal = oversample_to_median(X_train_s, y_train, SEED)
    print(f"  oversampled train: {len(X_train_bal):,} rows "
          f"(from {len(X_train_s):,})", file=tee)

    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        alpha=0.01,
        max_iter=200,
        early_stopping=False,
        random_state=SEED,
        learning_rate='adaptive',
        learning_rate_init=LEARNING_RATE,
    )
    clf.fit(X_train_bal, y_train_bal)
    with open(SK_PATH, "wb") as f:
        pickle.dump(clf, f)
    sk_test_acc = clf.score(X_test_s, y_test) * 100
    print(f"  sklearn test acc: {sk_test_acc:.2f}%", file=tee)
    print(f"  saved → {SK_PATH}", file=tee)
    print(file=tee)

    # ── Final report ───────────────────────────────────────────────────
    bar = "=" * 78
    print(bar, file=tee)
    print("FINAL TEST METRICS (PyTorch best_v2 checkpoint)", file=tee)
    print(bar, file=tee)
    print(f"Total dataset:  {len(X):,} rows  (train {len(X_train):,} / "
          f"test {len(X_test):,})", file=tee)
    print(f"Unique recordings:  train {len(set(g_train)):,}  /  "
          f"test {len(set(g_test)):,}  (no leakage)", file=tee)
    print(f"Overall PyTorch test accuracy:  {final_acc:.2f}%", file=tee)
    print(f"Overall sklearn  test accuracy: {sk_test_acc:.2f}%", file=tee)
    print(file=tee)

    # Per-raga test accuracy (PyTorch), sorted ascending → worst 10 first.
    print("Per-raga test accuracy (PyTorch, sorted ascending):", file=tee)
    per_raga = []
    for cls in range(n_classes):
        mask = labels == cls
        n = int(mask.sum())
        if n == 0:
            per_raga.append((classes[cls], None, 0))
            continue
        acc = (preds[mask] == cls).mean() * 100
        per_raga.append((classes[cls], acc, n))
    per_raga_sorted = sorted(
        per_raga,
        key=lambda r: (1e9 if r[1] is None else r[1])
    )
    print(f"  {'rank':>4}  {'raga':<28}  {'acc':>7}  {'n_test':>6}", file=tee)
    print(f"  {'-'*4}  {'-'*28}  {'-'*7}  {'-'*6}", file=tee)
    for rank, (name, acc, n) in enumerate(per_raga_sorted, 1):
        acc_s = "  N/A " if acc is None else f"{acc:>6.2f}%"
        print(f"  {rank:>4}  {name:<28}  {acc_s:>7}  {n:>6}", file=tee)
    print(file=tee)

    # Accuracy by data source.
    print("Test accuracy by data source:", file=tee)
    print(f"  {'source':<11}  {'rows':>7}  {'acc':>7}", file=tee)
    print(f"  {'-'*11}  {'-'*7}  {'-'*7}", file=tee)
    for src in ("compmusic", "youtube", "audio_clip"):
        mask = s_test == src
        n = int(mask.sum())
        if n == 0:
            print(f"  {src:<11}  {n:>7}  {'  N/A ':>7}", file=tee)
            continue
        acc = (preds[mask] == labels[mask]).mean() * 100
        print(f"  {src:<11}  {n:>7,}  {acc:>6.2f}%", file=tee)
    print(file=tee)

    # Confusion matrix for the bottom 10 ragas (only show non-zero off-diagonals
    # so the print stays readable for 40-class data).
    print("Confusion matrix — bottom 10 ragas by test accuracy", file=tee)
    print("(rows = true raga, columns = top mispredictions)", file=tee)
    bottom = [r for r in per_raga_sorted[:10] if r[1] is not None]
    for true_name, acc, n in bottom:
        true_cls = classes.index(true_name)
        mask = labels == true_cls
        if mask.sum() == 0:
            continue
        pred_counts = Counter(preds[mask].tolist())
        ordered = pred_counts.most_common()
        line = f"  {true_name:<28} ({acc:.1f}%, n={n})  → "
        parts = []
        for p_cls, cnt in ordered[:6]:
            tag = "✓" if p_cls == true_cls else "✗"
            parts.append(f"{tag} {classes[p_cls]} ({cnt})")
        print(line + "  ".join(parts), file=tee)
    print(file=tee)

    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] Total wall time: {elapsed/60:.1f} min",
          file=tee)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(tee.buf.getvalue())
    print(f"\nReport written to: {REPORT_PATH}", file=sys.__stdout__, flush=True)


if __name__ == "__main__":
    main()
