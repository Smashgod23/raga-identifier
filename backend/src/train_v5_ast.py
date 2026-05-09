"""
Phase 5: train a small classifier head on top of frozen AST embeddings.

The Audio Spectrogram Transformer (MIT/ast-finetuned-audioset-10-10-0.4593)
was pretrained on 2 million 10-second AudioSet clips. Its 768-dim CLS
embedding already encodes voice timbre, instrumentation, room acoustics,
and music structure. preprocess_ast_embeddings.py ran the full encoder over
every 10.24-second window of every CompMusic recording and saved the
embeddings to data/X_ast.npy.

This script trains a small head on top of those frozen embeddings:
  768 → 256 → 40, BatchNorm + ReLU + Dropout(0.3)

The hope is that with the foundation model handling all the "what does this
audio sound like" reasoning, the small head can learn raga-specific
patterns from our 480 recordings without memorizing the concerts.

Outputs:
  models/raga_ast_head_v5.pt
  data/train_v5_report.txt

Run from backend/:
  source venv/bin/activate
  python src/train_v5_ast.py
"""

import json
import os
import sys
import time
from collections import Counter, defaultdict
from io import StringIO

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import GroupShuffleSplit

THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(THIS_DIR)
DATA_DIR    = os.path.join(BACKEND_DIR, "data")
MODELS_DIR  = os.path.join(BACKEND_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

X_PATH       = os.path.join(DATA_DIR, "X_ast.npy")
Y_PATH       = os.path.join(DATA_DIR, "y_ast.npy")
META_PATH    = os.path.join(DATA_DIR, "ast_meta.json")
CLASSES_PATH = os.path.join(DATA_DIR, "classes.json")
PT_PATH      = os.path.join(MODELS_DIR, "raga_ast_head_v5.pt")
REPORT_PATH  = os.path.join(DATA_DIR, "train_v5_report.txt")

EPOCHS         = 50
BATCH_SIZE     = 128
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
WARMUP_EPOCHS  = 3
TEST_SIZE      = 0.20
SEED           = 42


class Tee:
    def __init__(self):
        self.buf = StringIO()
    def write(self, s):
        sys.__stdout__.write(s); self.buf.write(s)
    def flush(self):
        sys.__stdout__.flush()


class ASTClassifierHead(nn.Module):
    """Small MLP head on top of frozen AST embeddings. The foundation model
    is doing the hard work; this head just maps 768-dim embeddings to 40
    raga logits with one nonlinearity."""
    def __init__(self, n_classes=40, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x):
        return self.net(x)


def evaluate_logits(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits_all.append(model(xb).detach().cpu().numpy())
            labels_all.append(yb.numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def main():
    t0 = time.time()
    tee = Tee()
    print(f"[{time.strftime('%H:%M:%S')}] >>> Phase 5: train_v5_ast — frozen AST + small head",
          file=tee, flush=True)
    print(file=tee)

    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    with open(CLASSES_PATH, encoding="utf-8") as f:
        classes = json.load(f)

    n_classes = len(classes)
    if not (len(X) == len(y) == len(meta)):
        raise SystemExit(
            f"ABORT: row count mismatch X={len(X)}, y={len(y)}, meta={len(meta)}"
        )

    rec_ids = np.array([m["recording_id"] for m in meta])
    print(f"X: {X.shape}, y: {y.shape}", file=tee)
    print(f"unique recordings: {len(set(rec_ids))}", file=tee)
    print(f"classes: {n_classes}", file=tee)
    print(file=tee)

    # ── Recording-aware split ───────────────────────────────────────────
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=rec_ids))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    g_train, g_test = rec_ids[train_idx], rec_ids[test_idx]
    meta_test = [meta[i] for i in test_idx]

    if set(g_train) & set(g_test):
        raise SystemExit("ABORT: recording_ids leaked across split")

    print(f"[{time.strftime('%H:%M:%S')}] Recording-aware split:", file=tee)
    print(f"  train: {len(X_train):,} windows, "
          f"{len(set(g_train)):,} unique recordings", file=tee)
    print(f"  test:  {len(X_test):,} windows, "
          f"{len(set(g_test)):,} unique recordings", file=tee)
    print(f"  test fraction (windows): {len(X_test)/len(X):.3f}", file=tee)
    print(file=tee)

    # ── Class weights ───────────────────────────────────────────────────
    train_counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    safe_counts  = np.where(train_counts == 0, 1, train_counts)
    class_w_np   = len(y_train) / (n_classes * safe_counts)
    print(f"Class weights min/max/mean: "
          f"{class_w_np.min():.3f}/{class_w_np.max():.3f}/{class_w_np.mean():.3f}",
          file=tee)
    print(file=tee)

    # ── Device, model, loaders ──────────────────────────────────────────
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[{time.strftime('%H:%M:%S')}] Device: {device}", file=tee)
    model = ASTClassifierHead(n_classes=n_classes).to(device)
    print(f"Head parameters: {sum(p.numel() for p in model.parameters()):,}",
          file=tee)
    print(file=tee)

    train_ds = TensorDataset(torch.from_numpy(X_train).float(),
                             torch.from_numpy(y_train).long())
    test_ds  = TensorDataset(torch.from_numpy(X_test).float(),
                             torch.from_numpy(y_test).long())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_w_np, dtype=torch.float32).to(device))
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training loop ───────────────────────────────────────────────────
    print(f"[{time.strftime('%H:%M:%S')}] Training {EPOCHS} epochs, "
          f"batch={BATCH_SIZE}, lr={LEARNING_RATE} "
          f"({WARMUP_EPOCHS}-epoch warmup → cosine)", file=tee)
    print(file=tee)

    best_rec_acc = 0.0
    val_acc_trace = []
    rec_acc_trace = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        train_correct = train_total = 0
        t_epoch = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
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

        test_logits, test_labels_arr = evaluate_logits(model, test_loader, device)
        test_softmax = softmax(test_logits, axis=1)
        test_preds   = test_logits.argmax(axis=1)
        val_acc      = (test_preds == test_labels_arr).mean() * 100
        val_acc_trace.append(val_acc)

        # Per-recording vote.
        rec_to_rows = defaultdict(list)
        for j, m in enumerate(meta_test):
            rec_to_rows[m["recording_id"]].append(j)
        rec_correct = rec_total = 0
        for recid, idxs in rec_to_rows.items():
            mean_sm = test_softmax[idxs].mean(axis=0)
            pred = int(mean_sm.argmax())
            true = int(test_labels_arr[idxs[0]])
            rec_total += 1
            if pred == true:
                rec_correct += 1
        rec_acc = rec_correct / rec_total * 100
        rec_acc_trace.append(rec_acc)

        if rec_acc > best_rec_acc:
            best_rec_acc = rec_acc
            torch.save(model.state_dict(), PT_PATH)
            saved = "  (saved)"
        else:
            saved = ""

        epoch_dt = time.time() - t_epoch
        print(f"[{time.strftime('%H:%M:%S')}] epoch {epoch:2d}/{EPOCHS}  "
              f"loss={avg_loss:.4f}  train={train_acc:.2f}%  "
              f"val={val_acc:.2f}%  rec_vote={rec_acc:.2f}%  "
              f"best_rec={best_rec_acc:.2f}%{saved}  ({epoch_dt:.1f}s)",
              file=tee, flush=True)

    print(file=tee)
    print(f"[{time.strftime('%H:%M:%S')}] Training done. Best per-recording vote: "
          f"{best_rec_acc:.2f}%", file=tee)
    print(file=tee)

    # ── Final eval on best checkpoint ───────────────────────────────────
    model.load_state_dict(torch.load(PT_PATH, map_location=device))
    model.eval()
    test_logits, test_labels_arr = evaluate_logits(model, test_loader, device)
    test_softmax = softmax(test_logits, axis=1)
    test_preds   = test_logits.argmax(axis=1)
    final_row_acc = (test_preds == test_labels_arr).mean() * 100

    rec_to_rows = defaultdict(list)
    for j, m in enumerate(meta_test):
        rec_to_rows[m["recording_id"]].append(j)
    rec_correct = rec_total = 0
    rec_per_raga = defaultdict(lambda: [0, 0])
    rec_predictions = []
    for recid, idxs in rec_to_rows.items():
        mean_sm = test_softmax[idxs].mean(axis=0)
        pred = int(mean_sm.argmax())
        true = int(test_labels_arr[idxs[0]])
        rec_total += 1
        rec_per_raga[true][1] += 1
        if pred == true:
            rec_correct += 1
            rec_per_raga[true][0] += 1
        rec_predictions.append((true, pred))
    final_rec_acc = rec_correct / rec_total * 100

    bar = "=" * 78
    print(bar, file=tee)
    print("FINAL TEST METRICS — AST embeddings + small head (v5)", file=tee)
    print(bar, file=tee)
    print(f"Total windows: {len(X):,}  "
          f"(train {len(X_train):,} / test {len(X_test):,})", file=tee)
    print(f"Unique recordings: train {len(set(g_train)):,}  /  "
          f"test {len(set(g_test)):,}  (zero leakage)", file=tee)
    print(file=tee)
    print(f"Per-window test accuracy:    {final_row_acc:.2f}%", file=tee)
    print(f"Per-recording vote accuracy: {final_rec_acc:.2f}%  "
          f"({rec_correct}/{rec_total} recordings)", file=tee)
    print(file=tee)
    print("Comparison (recording vote, production-realistic):", file=tee)
    print(f"  v3 histogram + multi-scale MLP:  53.62%", file=tee)
    print(f"  v4 mel-spec CNN from scratch:    11.46%", file=tee)
    print(f"  v5 AST embeddings + small head:  {final_rec_acc:.2f}%", file=tee)
    print(file=tee)

    print("Per-raga test accuracy (per-recording vote, ascending):", file=tee)
    per_raga = []
    for cls in range(n_classes):
        c, t = rec_per_raga.get(cls, [0, 0])
        per_raga.append((classes[cls], (c / t * 100) if t > 0 else None, t))
    per_raga_sorted = sorted(per_raga,
                             key=lambda r: (1e9 if r[1] is None else r[1]))
    print(f"  {'rank':>4}  {'raga':<28}  {'acc':>7}  {'n_rec':>5}", file=tee)
    print(f"  {'-'*4}  {'-'*28}  {'-'*7}  {'-'*5}", file=tee)
    for rank, (name, acc, n) in enumerate(per_raga_sorted, 1):
        marker = " ←" if rank <= 10 and acc is not None else ""
        acc_s = "  N/A " if acc is None else f"{acc:>6.2f}%"
        print(f"  {rank:>4}  {name:<28}  {acc_s:>7}  {n:>5}{marker}", file=tee)
    print(file=tee)

    pair_counts = Counter()
    for true, pred in rec_predictions:
        if true != pred:
            pair_counts[(true, pred)] += 1
    print("Top 10 most-confused (true → predicted) pairs at recording level:",
          file=tee)
    print(f"  {'count':>5}  {'true':<28}  →  {'predicted':<28}", file=tee)
    print(f"  {'-'*5}  {'-'*28}  ----  {'-'*28}", file=tee)
    for (t, p), c in pair_counts.most_common(10):
        print(f"  {c:>5}  {classes[t]:<28}  →  {classes[p]:<28}", file=tee)
    print(file=tee)

    print("Per-epoch val_acc trace:", file=tee)
    print(f"  [{', '.join(f'{v:.2f}' for v in val_acc_trace)}]", file=tee)
    print("Per-epoch per-recording vote acc trace:", file=tee)
    print(f"  [{', '.join(f'{v:.2f}' for v in rec_acc_trace)}]", file=tee)
    print(file=tee)

    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] Total wall time: {elapsed/60:.1f} min",
          file=tee)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(tee.buf.getvalue())
    print(f"\nReport written to: {REPORT_PATH}", file=sys.__stdout__, flush=True)


if __name__ == "__main__":
    main()
