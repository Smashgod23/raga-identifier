"""
Phase 3 step 3: train v3 multi-scale model.

Trains one model on the 72,997-row multi-scale dataset (480 full + 209 YouTube
+ 6,833 3-min + 21,825 1-min + 21,825 30s + 21,825 15s, all from raw audio
with expert .tonicFine values).

Key choices vs v1/v2:
  - Trains on data/X_multiscale.npy (six sources, scale-balanced via subsample)
  - Recording-aware GroupShuffleSplit(80/20) on recording_id, so all features
    derived from one recording at every scale stay on the same side of the
    train/test boundary.
  - PyTorch with weighted CrossEntropyLoss; weights = n_samples / (n_classes *
    bincount(y_train)) — the user's exact spec.
  - sklearn deployment copy is converted from the trained PyTorch model by
    folding BatchNorm into the preceding Linear layers (no separate sklearn
    fit). Ensures inference parity between deployment and training.
  - StandardScaler fit on TRAIN ONLY.
  - Outputs go to *_v3 paths so v1/v2 artifacts are preserved.

Reporting at end of training:
  - Overall test accuracy, per-row and per-recording (mean-softmax vote
    across all clips/windows of the same recording — what predict.py does
    in production).
  - Per-scale test accuracy (compmusic_full / youtube_full / scale_3min /
    scale_1min / scale_30s / scale_15s) so we can see where 15s windows
    drag or hold up.
  - Per-raga test accuracy sorted ascending, worst 10 highlighted.
  - Confusion matrix: top-10 most-confused (true, predicted) pairs.

Outputs:
  models/raga_model_best_v3.pt
  models/raga_sklearn_v3.pkl
  models/scaler_v3.pkl
  data/train_v3_report.txt

Run from backend/:
  source venv/bin/activate
  python src/train_v3.py
"""

import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from io import StringIO

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(THIS_DIR)
DATA_DIR    = os.path.join(BACKEND_DIR, "data")
MODELS_DIR  = os.path.join(BACKEND_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

X_PATH       = os.path.join(DATA_DIR, "X_multiscale.npy")
Y_PATH       = os.path.join(DATA_DIR, "y_multiscale.npy")
META_PATH    = os.path.join(DATA_DIR, "multiscale_meta.json")
CLASSES_PATH = os.path.join(DATA_DIR, "classes.json")

PT_PATH      = os.path.join(MODELS_DIR, "raga_model_best_v3.pt")
SK_PATH      = os.path.join(MODELS_DIR, "raga_sklearn_v3.pkl")
SCALER_PATH  = os.path.join(MODELS_DIR, "scaler_v3.pkl")
REPORT_PATH  = os.path.join(DATA_DIR, "train_v3_report.txt")

EPOCHS         = 200
BATCH_SIZE     = 128
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
    """Same architecture as v1 train.py."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),  # indices 0-3
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),  # indices 4-7
            nn.Linear(128, 64),  nn.ReLU(),                   # indices 8-9
            nn.Linear(64, num_classes),                       # index 10
        )
    def forward(self, x):
        return self.net(x)


def evaluate_logits(model, loader, device):
    """Return concatenated (logits, labels) over the whole loader."""
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            logits_all.append(model(xb).cpu().numpy())
            labels_all.append(yb.cpu().numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def fold_bn_into_linear(W, b, bn_w, bn_b, bn_mean, bn_var, eps=1e-5):
    """Fuse BatchNorm following a Linear layer. PyTorch BN1d formula:
        y = bn_w * (x - bn_mean) / sqrt(bn_var + eps) + bn_b
    Composed with Linear(W, b):
        y = scale * (W x + b - bn_mean) + bn_b
        y = (scale * W) x + (scale * (b - bn_mean) + bn_b)
    Returns new (W', b') matching the Linear shape."""
    scale = bn_w / np.sqrt(bn_var + eps)
    new_W = W * scale[:, None]                     # (out, in)
    new_b = scale * (b - bn_mean) + bn_b           # (out,)
    return new_W, new_b


def convert_pytorch_to_sklearn(model, n_classes, n_features_in):
    """Convert trained PyTorch RagaNet to an sklearn MLPClassifier with
    identical inference behavior. BatchNorm layers are folded into the
    preceding Linear layers; Dropout is a no-op at inference and is
    dropped. Returns the populated MLPClassifier."""
    model.eval()
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    # Layer 1: Linear (net.0) + BN1d (net.1)
    W1, b1 = fold_bn_into_linear(
        sd["net.0.weight"], sd["net.0.bias"],
        sd["net.1.weight"], sd["net.1.bias"],
        sd["net.1.running_mean"], sd["net.1.running_var"]
    )
    # Layer 2: Linear (net.4) + BN1d (net.5)
    W2, b2 = fold_bn_into_linear(
        sd["net.4.weight"], sd["net.4.bias"],
        sd["net.5.weight"], sd["net.5.bias"],
        sd["net.5.running_mean"], sd["net.5.running_var"]
    )
    # Layer 3: Linear (net.8) — no BN
    W3, b3 = sd["net.8.weight"], sd["net.8.bias"]
    # Output layer: Linear (net.10)
    W4, b4 = sd["net.10.weight"], sd["net.10.bias"]

    # Build a fresh MLPClassifier and run a tiny fit so its internal
    # bookkeeping (coefs_, intercepts_, classes_, n_layers_, etc.) is set up.
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        max_iter=1,
        random_state=SEED,
    )
    rng = np.random.default_rng(SEED)
    X_dummy = rng.standard_normal((n_classes * 2, n_features_in))
    y_dummy = np.repeat(np.arange(n_classes), 2)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence warning is expected
        clf.fit(X_dummy, y_dummy)

    # PyTorch Linear: weight (out, in); sklearn MLP: coefs_[i] (in, out).
    clf.coefs_[0] = W1.T;  clf.intercepts_[0] = b1
    clf.coefs_[1] = W2.T;  clf.intercepts_[1] = b2
    clf.coefs_[2] = W3.T;  clf.intercepts_[2] = b3
    clf.coefs_[3] = W4.T;  clf.intercepts_[3] = b4

    return clf


# ── main ───────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    tee = Tee()
    print(f"[{time.strftime('%H:%M:%S')}] >>> Phase 3 step 3: train_v3 — multi-scale + recording-aware",
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
    meta_test = [meta[i] for i in test_idx]

    leak = set(g_train) & set(g_test)
    if leak:
        raise SystemExit(f"ABORT: {len(leak)} recording_ids leaked across split")

    print(f"[{time.strftime('%H:%M:%S')}] Recording-aware split:", file=tee)
    print(f"  train: {len(X_train):,} rows, {len(set(g_train)):,} unique recordings",
          file=tee)
    print(f"  test:  {len(X_test):,} rows,  {len(set(g_test)):,} unique recordings",
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
    train_counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    safe_counts  = np.where(train_counts == 0, 1, train_counts)
    class_w_np   = len(y_train) / (n_classes * safe_counts)
    class_w_t    = torch.tensor(class_w_np, dtype=torch.float32)
    print(f"Class weights min/max/mean: "
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
    print(f"  val_acc computed every epoch (saved to report); console "
          f"prints every 5", file=tee)
    print(file=tee)

    best_acc = 0.0
    val_acc_trace = []  # full per-epoch val_acc, written to report at the end
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        train_correct = train_total = 0
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

        # Compute val_acc EVERY epoch (cheap; gives us the full plateau trace
        # the user wants to inspect).
        test_logits, test_labels = evaluate_logits(model, test_loader, device)
        val_acc = (test_logits.argmax(axis=1) == test_labels).mean() * 100
        val_acc_trace.append(val_acc)

        # Save best whenever it improves, not only on print-aligned epochs.
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), PT_PATH)
            saved_this_epoch = True
        else:
            saved_this_epoch = False

        # Console only every 5 epochs (plus first and last).
        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            tag = "  (saved)" if saved_this_epoch else ""
            print(f"[{time.strftime('%H:%M:%S')}] epoch {epoch:3d}/{EPOCHS}  "
                  f"loss={avg_loss:.4f}  train_acc={train_acc:.2f}%  "
                  f"val_acc={val_acc:.2f}%  best={best_acc:.2f}%{tag}",
                  file=tee, flush=True)

    print(file=tee)
    print(f"[{time.strftime('%H:%M:%S')}] PyTorch training done. Best per-row val acc: "
          f"{best_acc:.2f}%", file=tee)
    print(f"  saved → {PT_PATH}", file=tee)
    print(file=tee)

    # Reload best checkpoint for the final eval & conversion.
    model.load_state_dict(torch.load(PT_PATH, map_location=device))
    model.eval()
    test_logits, test_labels = evaluate_logits(model, test_loader, device)
    test_softmax = softmax(test_logits, axis=1)
    test_preds   = test_logits.argmax(axis=1)
    final_row_acc = (test_preds == test_labels).mean() * 100

    # ── Convert PyTorch → sklearn (BN folded) ───────────────────────────
    AGREEMENT_THRESHOLD = 99.5
    print(f"[{time.strftime('%H:%M:%S')}] Converting PyTorch checkpoint to sklearn (BN folded)",
          file=tee)
    clf = convert_pytorch_to_sklearn(model, n_classes=n_classes, n_features_in=360)

    # Sanity: predictions from the converted sklearn model must match the
    # PyTorch model on the test set (modulo float precision). If they don't
    # match closely enough, the BN fold is miscomputed and the sklearn copy
    # would deploy a model that disagrees with the trained one — refuse to
    # save it.
    sk_preds = clf.predict(X_test_s)
    match = (sk_preds == test_preds).mean() * 100
    print(f"  sklearn vs PyTorch agreement on test set: {match:.4f}% "
          f"(threshold {AGREEMENT_THRESHOLD}%)", file=tee)

    sklearn_saved = False
    if match >= AGREEMENT_THRESHOLD:
        with open(SK_PATH, "wb") as f:
            pickle.dump(clf, f)
        print(f"  saved → {SK_PATH}", file=tee)
        sklearn_saved = True
    else:
        print(f"  CONVERSION FAILED: agreement {match:.4f}% < {AGREEMENT_THRESHOLD}%.",
              file=tee)
        print(f"  Refusing to save raga_sklearn_v3.pkl — would deploy a model that "
              f"disagrees with the trained PyTorch model on >0.5% of test rows.",
              file=tee)
        print(f"  raga_model_best_v3.pt is still saved; investigate BN fold "
              f"before attempting deployment.", file=tee)
    print(file=tee)

    # ── Final report ───────────────────────────────────────────────────
    bar = "=" * 78
    print(bar, file=tee)
    print("FINAL TEST METRICS (PyTorch best_v3 checkpoint)", file=tee)
    print(bar, file=tee)
    print(f"Total dataset: {len(X):,} rows  (train {len(X_train):,} / "
          f"test {len(X_test):,})", file=tee)
    print(f"Unique recordings:  train {len(set(g_train)):,}  /  "
          f"test {len(set(g_test)):,}  (zero leakage)", file=tee)
    sklearn_status = (
        f"saved (sklearn vs PyTorch agreement {match:.4f}%)"
        if sklearn_saved else
        f"NOT saved (BN-fold conversion failed at {match:.4f}% agreement, "
        f"threshold {AGREEMENT_THRESHOLD}%)"
    )
    print(f"sklearn deployment copy: {sklearn_status}", file=tee)
    print(file=tee)

    # 1) Per-row accuracy.
    print(f"Per-row test accuracy:  {final_row_acc:.2f}%  "
          f"(over {len(test_preds):,} test rows)", file=tee)
    print(file=tee)

    # 2) Per-recording vote (mean softmax across all rows of the same recording).
    rec_to_rows = defaultdict(list)
    for i, m in enumerate(meta_test):
        rec_to_rows[m["recording_id"]].append(i)

    rec_correct = rec_total = 0
    rec_per_raga = defaultdict(lambda: [0, 0])  # [correct, total]
    rec_predictions = []  # list of (true_label, pred_label)
    for recid, idxs in rec_to_rows.items():
        mean_sm = test_softmax[idxs].mean(axis=0)
        pred = int(mean_sm.argmax())
        # All rows in this recording share the same label.
        true = int(test_labels[idxs[0]])
        rec_total += 1
        rec_per_raga[true][1] += 1
        if pred == true:
            rec_correct += 1
            rec_per_raga[true][0] += 1
        rec_predictions.append((true, pred))

    final_rec_acc = rec_correct / rec_total * 100
    print(f"Per-recording test accuracy (mean-softmax vote across all "
          f"clips/windows of one recording): {final_rec_acc:.2f}%  "
          f"({rec_correct:,}/{rec_total:,} recordings)", file=tee)
    print(f"  This is the deployment-relevant metric — predict.py averages "
          f"clip predictions in production.", file=tee)
    print(file=tee)

    # 3) Per-scale (per-source) test accuracy on rows.
    print("Per-scale test accuracy (per-row):", file=tee)
    print(f"  {'source':<16}  {'rows':>7}  {'acc':>7}", file=tee)
    print(f"  {'-'*16}  {'-'*7}  {'-'*7}", file=tee)
    for src in ("compmusic_full", "youtube_full", "scale_3min", "scale_1min",
                "scale_30s", "scale_15s"):
        mask = s_test == src
        n = int(mask.sum())
        if n == 0:
            print(f"  {src:<16}  {n:>7}  {'  N/A ':>7}", file=tee)
            continue
        acc = (test_preds[mask] == test_labels[mask]).mean() * 100
        print(f"  {src:<16}  {n:>7,}  {acc:>6.2f}%", file=tee)
    print(file=tee)

    # 4) Per-raga test accuracy (per-recording vote), sorted ascending.
    print("Per-raga test accuracy (per-recording vote, sorted ascending — "
          "worst 10 first):", file=tee)
    per_raga = []
    for cls in range(n_classes):
        c, t = rec_per_raga.get(cls, [0, 0])
        if t == 0:
            per_raga.append((classes[cls], None, 0))
        else:
            per_raga.append((classes[cls], c / t * 100, t))
    per_raga_sorted = sorted(
        per_raga,
        key=lambda r: (1e9 if r[1] is None else r[1])
    )
    print(f"  {'rank':>4}  {'raga':<28}  {'acc':>7}  {'n_rec':>6}", file=tee)
    print(f"  {'-'*4}  {'-'*28}  {'-'*7}  {'-'*6}", file=tee)
    for rank, (name, acc, n) in enumerate(per_raga_sorted, 1):
        marker = " ←" if rank <= 10 and acc is not None else ""
        acc_s = "  N/A " if acc is None else f"{acc:>6.2f}%"
        print(f"  {rank:>4}  {name:<28}  {acc_s:>7}  {n:>6}{marker}", file=tee)
    print(file=tee)

    # 5) Confusion matrix — top 10 most-confused (true, predicted) pairs at
    #    the per-recording level.
    pair_counts = Counter()
    for true, pred in rec_predictions:
        if true == pred:
            continue
        pair_counts[(true, pred)] += 1

    print("Confusion matrix — top 10 most-confused (true → predicted) pairs at "
          "the recording level:", file=tee)
    print(f"  {'count':>5}  {'true':<28}  →  {'predicted':<28}", file=tee)
    print(f"  {'-'*5}  {'-'*28}  ----  {'-'*28}", file=tee)
    for (t, p), c in pair_counts.most_common(10):
        print(f"  {c:>5}  {classes[t]:<28}  →  {classes[p]:<28}", file=tee)
    print(file=tee)

    # Per-epoch val_acc trace — diagnostic, lets us see whether the model
    # plateaus like v2 did or keeps improving. Printed at the end of the
    # report (not gating).
    print("Per-epoch val_acc trace (diagnostic):", file=tee)
    print(f"  epochs={len(val_acc_trace)}, "
          f"min={min(val_acc_trace):.2f}%, max={max(val_acc_trace):.2f}%, "
          f"final={val_acc_trace[-1]:.2f}%",
          file=tee)
    # Find the epoch where best val_acc was first reached (where saving stopped).
    best_v   = max(val_acc_trace)
    best_ep  = val_acc_trace.index(best_v) + 1
    print(f"  best  val_acc {best_v:.2f}% first reached at epoch {best_ep}",
          file=tee)
    # Compare best to last to flag plateau.
    plateau_gap = best_v - val_acc_trace[-1]
    print(f"  best - final = {plateau_gap:+.2f} pp "
          f"({'plateaued early' if plateau_gap > 1 else 'still tracking near best'})",
          file=tee)
    # Full trace as comma-separated list.
    print(f"  per-epoch val_acc list:", file=tee)
    print(f"  [{', '.join(f'{v:.2f}' for v in val_acc_trace)}]", file=tee)
    print(file=tee)

    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] Total wall time: {elapsed/60:.1f} min",
          file=tee)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(tee.buf.getvalue())
    print(f"\nReport written to: {REPORT_PATH}", file=sys.__stdout__, flush=True)


if __name__ == "__main__":
    main()
