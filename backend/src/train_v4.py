"""
Phase 4: train a 2D CNN on log-Mel spectrograms.

Replaces the 360-dim pitch-class histogram features (which throw away
temporal/phrase information) with full 2D log-Mel spectrograms. The CNN can
see *order* of swaras and gamakam patterns, which is what separates similar
ragas (Mohanam vs Bilahari, Bhairavi vs Mukhari) — the cases histogram
methods fundamentally cannot solve.

Key choices:
  - Inputs: 60-second windows of pre-computed log-Mel spectrograms from
    data/melspec/, sliced with a 30-second hop. ~14,000 windows from 480
    CompMusic recordings (no YouTube — we don't have raw YouTube audio
    stored, that integration is a separate effort).
  - Recording-aware GroupShuffleSplit(80/20) on recording_id, seed 42 (same
    seed as v3 so the train/test partition is comparable).
  - 2D CNN: 4 conv blocks + AdaptiveAvgPool + 2-layer head. ~210k params.
  - Apple Silicon MPS backend for ~5x speedup over CPU on conv layers.
  - PyTorch with weighted CrossEntropyLoss (n_samples / (n_classes *
    bincount)).
  - Best checkpoint selected by per-recording vote accuracy (the
    deployment-relevant metric), not per-window accuracy.

Outputs:
  models/raga_cnn_v4.pt
  data/train_v4_report.txt

Run from backend/:
  source venv/bin/activate
  python src/train_v4.py
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
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import GroupShuffleSplit

THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(THIS_DIR)
DATA_DIR    = os.path.join(BACKEND_DIR, "data")
MODELS_DIR  = os.path.join(BACKEND_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MELSPEC_DIR  = os.path.join(DATA_DIR, "melspec")
CLASSES_PATH = os.path.join(DATA_DIR, "classes.json")
MAPPING_PATH = os.path.join(DATA_DIR, "RagaDataset", "Carnatic", "_info_",
                            "ragaId_to_ragaName_mapping.json")
PT_PATH      = os.path.join(MODELS_DIR, "raga_cnn_v4.pt")
REPORT_PATH  = os.path.join(DATA_DIR, "train_v4_report.txt")

# Spectrogram parameters: preprocess_melspec.py wrote files at n_mels=128,
# hop=512. Earlier training runs choked because the resulting 6.7 GB cache
# (3.3 GB even at float16) doesn't fit alongside MPS buffers and other apps
# on this machine — the OS kept paging it out, training stalled at 11 % CPU.
# We subsample 2x in both axes at load time, dropping the cache to ~420 MB.
# n_mels 64 still spans the pitch range cleanly; 64 ms time resolution still
# captures gamakam patterns (which are 100-500 ms).
SR             = 16000
DISK_HOP       = 512
DISK_N_MELS    = 128
SUBSAMPLE_FREQ = 2          # 128 → 64 mel bins
SUBSAMPLE_TIME = 2          # 32 ms → 64 ms per frame
EFFECTIVE_HOP  = DISK_HOP * SUBSAMPLE_TIME
N_MELS         = DISK_N_MELS // SUBSAMPLE_FREQ

WINDOW_SEC = 60.0
HOP_SEC    = 30.0
WINDOW_FRAMES = int(WINDOW_SEC * SR / EFFECTIVE_HOP)   # 937
HOP_FRAMES    = int(HOP_SEC    * SR / EFFECTIVE_HOP)   # 468

EPOCHS         = 30
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-3   # higher than v1's 3e-4 because cosine warms up over 2 epochs first
WEIGHT_DECAY   = 1e-4
WARMUP_EPOCHS  = 2
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


class MelSpecCNN(nn.Module):
    """4-block 2D CNN on log-Mel spectrograms. AdaptiveAvgPool at the end
    makes it input-size invariant, so inference can use any window length
    without architectural changes."""
    def __init__(self, n_classes=40):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Squeeze to (128,)
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class MelSpecWindowDataset(Dataset):
    """Yields (mel_window, label) pairs. Mel-specs are eager-loaded into a
    dict at startup (~6.7 GB total) so __getitem__ is pure RAM access. The
    earlier mmap-based version stalled when the OS file cache evicted under
    memory pressure: 32 batches × 32 windows × 10 MB recordings can't all
    stay cached, so every batch ended up reading from disk."""
    def __init__(self, window_index, melspec_cache):
        self.window_index = window_index
        self.cache = melspec_cache  # {path: np.ndarray (n_mels, n_frames)}

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx):
        path, start, end, label, _recording_id = self.window_index[idx]
        # Cache holds float16; cast back to float32 on slice for the model.
        window = self.cache[path][:, start:end].astype(np.float32)
        # Per-window z-score (typical for spectrogram CNN).
        mu  = window.mean()
        sig = window.std() + 1e-6
        window = (window - mu) / sig
        return torch.from_numpy(window).unsqueeze(0), int(label)


def preload_melspecs(window_index):
    """Read every unique mel-spec path into a dict, subsample 2x in both
    axes (frequency and time), and downcast to float16. The on-disk format
    is float32 at n_mels=128/hop=512 (~6.7 GB). After 2x/2x subsample +
    float16 the in-memory cache is ~420 MB, which keeps the system out of
    swap on a memory-pressured Mac."""
    paths = sorted({w[0] for w in window_index})
    cache = {}
    total_bytes = 0
    t0 = time.time()
    for i, p in enumerate(paths):
        arr = np.load(p)
        # Subsample frequency and time, then quantize.
        arr = arr[::SUBSAMPLE_FREQ, ::SUBSAMPLE_TIME]
        arr = arr.astype(np.float16, copy=False)
        cache[p] = arr
        total_bytes += arr.nbytes
        if (i + 1) % 50 == 0 or (i + 1) == len(paths):
            print(f"  loaded {i+1:>3}/{len(paths)}  "
                  f"({total_bytes / (1024**3):.3f} GB, "
                  f"{time.time() - t0:.0f}s)", flush=True)
    return cache


def build_window_index(classes):
    """Walk data/melspec/ and produce (path, start_frame, end_frame, label,
    recording_id) tuples. recording_id matches the rest of the pipeline."""
    with open(MAPPING_PATH, encoding="utf-8") as f:
        id_to_name = json.load(f)
    name_to_label = {n: i for i, n in enumerate(classes)}

    windows = []
    skipped_short = 0
    for raga_id in sorted(os.listdir(MELSPEC_DIR)):
        if raga_id not in id_to_name:
            continue
        raga_name = id_to_name[raga_id]
        if raga_name not in name_to_label:
            continue
        label = name_to_label[raga_name]
        raga_dir = os.path.join(MELSPEC_DIR, raga_id)
        for root, _, files in os.walk(raga_dir):
            for fname in sorted(files):
                if not fname.endswith(".npy"):
                    continue
                path = os.path.join(root, fname)
                # Read shape via mmap header. On-disk frame count is at the
                # full hop_length=512 resolution; after subsampling the
                # effective frame count is half. Window start/end indices
                # below are in *post-subsample* frames, matching how the
                # dataset slices the cache.
                shape = np.load(path, mmap_mode="r").shape  # (DISK_N_MELS, disk_frames)
                n_frames = shape[1] // SUBSAMPLE_TIME
                if n_frames < WINDOW_FRAMES:
                    skipped_short += 1
                    continue
                rel_no_ext = os.path.splitext(os.path.relpath(path, MELSPEC_DIR))[0]
                recording_id = rel_no_ext  # matches multiscale_meta convention
                start = 0
                while start + WINDOW_FRAMES <= n_frames:
                    windows.append((path, start, start + WINDOW_FRAMES, label, recording_id))
                    start += HOP_FRAMES
    return windows, skipped_short


def evaluate(model, loader, device):
    """Return (logits_array, labels_array) over the loader."""
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


# ── main ───────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    tee = Tee()
    print(f"[{time.strftime('%H:%M:%S')}] >>> Phase 4: train_v4 — log-Mel CNN",
          file=tee, flush=True)
    print(file=tee)

    with open(CLASSES_PATH, encoding="utf-8") as f:
        classes = json.load(f)
    n_classes = len(classes)

    print(f"[{time.strftime('%H:%M:%S')}] Building window index "
          f"(window={WINDOW_SEC}s, hop={HOP_SEC}s, n_mels={N_MELS})", file=tee)
    windows, skipped_short = build_window_index(classes)
    print(f"  total windows: {len(windows):,}", file=tee)
    print(f"  recordings shorter than window: {skipped_short}", file=tee)
    print(file=tee)

    labels  = np.array([w[3] for w in windows])
    rec_ids = np.array([w[4] for w in windows])

    # ── Recording-aware split ───────────────────────────────────────────
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(gss.split(np.zeros(len(labels)), labels, groups=rec_ids))

    train_windows = [windows[i] for i in train_idx]
    test_windows  = [windows[i] for i in test_idx]

    leak = set(rec_ids[train_idx]) & set(rec_ids[test_idx])
    if leak:
        raise SystemExit(f"ABORT: {len(leak)} recording_ids leaked across split")

    print(f"[{time.strftime('%H:%M:%S')}] Recording-aware split:", file=tee)
    print(f"  train: {len(train_windows):,} windows, "
          f"{len(set(rec_ids[train_idx])):,} unique recordings", file=tee)
    print(f"  test:  {len(test_windows):,} windows, "
          f"{len(set(rec_ids[test_idx])):,} unique recordings", file=tee)
    print(f"  test fraction (windows): {len(test_idx) / len(windows):.3f}", file=tee)
    print(file=tee)

    # Eager-load every mel-spec into RAM so __getitem__ is pure memory
    # access. ~6.7 GB total — verified to fit, otherwise we'd see swap.
    print(f"[{time.strftime('%H:%M:%S')}] Pre-loading mel-specs into RAM...",
          file=tee, flush=True)
    melspec_cache = preload_melspecs(windows)
    print(f"[{time.strftime('%H:%M:%S')}] Cache built ({len(melspec_cache)} files).",
          file=tee, flush=True)
    print(file=tee)

    train_ds = MelSpecWindowDataset(train_windows, melspec_cache)
    test_ds  = MelSpecWindowDataset(test_windows,  melspec_cache)

    # num_workers=0 because mmap'd numpy arrays and DataLoader workers
    # have ergonomic issues on macOS (spawn-mode workers re-import this
    # module, lose the cache, and re-mmap).
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Class weights ───────────────────────────────────────────────────
    train_labels = labels[train_idx]
    train_counts = np.bincount(train_labels, minlength=n_classes).astype(np.float64)
    safe_counts  = np.where(train_counts == 0, 1, train_counts)
    class_w_np   = len(train_labels) / (n_classes * safe_counts)
    print(f"Class weights min/max/mean: "
          f"{class_w_np.min():.3f} / {class_w_np.max():.3f} / "
          f"{class_w_np.mean():.3f}", file=tee)
    print(file=tee)

    # ── Device & model ──────────────────────────────────────────────────
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[{time.strftime('%H:%M:%S')}] Device: {device}", file=tee)
    model = MelSpecCNN(n_classes=n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}", file=tee)
    print(file=tee)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_w_np, dtype=torch.float32).to(device))

    # Linear warmup for WARMUP_EPOCHS, then cosine annealing across the rest.
    # Earlier run with cosine-from-3e-4 was too gentle; warmup avoids the
    # learning-rate cold-start that wasted the first few epochs.
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

    test_meta = [(rec_ids[i], int(labels[i])) for i in test_idx]
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

        # Eval on test set every epoch.
        test_logits, test_labels_arr = evaluate(model, test_loader, device)
        test_softmax = softmax(test_logits, axis=1)
        test_preds   = test_logits.argmax(axis=1)
        val_acc      = (test_preds == test_labels_arr).mean() * 100
        val_acc_trace.append(val_acc)

        # Per-recording vote.
        rec_to_rows = defaultdict(list)
        for j, (recid, _) in enumerate(test_meta):
            rec_to_rows[recid].append(j)
        rec_correct = rec_total = 0
        for recid, idxs in rec_to_rows.items():
            mean_sm = test_softmax[idxs].mean(axis=0)
            pred = int(mean_sm.argmax())
            true = test_meta[idxs[0]][1]
            rec_total += 1
            if pred == true:
                rec_correct += 1
        rec_acc = rec_correct / rec_total * 100
        rec_acc_trace.append(rec_acc)

        # Save best by per-recording vote.
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
              f"best_rec={best_rec_acc:.2f}%{saved}  ({epoch_dt:.0f}s)",
              file=tee, flush=True)

    print(file=tee)
    print(f"[{time.strftime('%H:%M:%S')}] Training done. Best per-recording vote: "
          f"{best_rec_acc:.2f}%", file=tee)
    print(file=tee)

    # ── Final eval on best checkpoint ───────────────────────────────────
    model.load_state_dict(torch.load(PT_PATH, map_location=device))
    model.eval()
    test_logits, test_labels_arr = evaluate(model, test_loader, device)
    test_softmax = softmax(test_logits, axis=1)
    test_preds   = test_logits.argmax(axis=1)
    final_row_acc = (test_preds == test_labels_arr).mean() * 100

    rec_to_rows = defaultdict(list)
    for j, (recid, _) in enumerate(test_meta):
        rec_to_rows[recid].append(j)

    rec_correct = rec_total = 0
    rec_per_raga = defaultdict(lambda: [0, 0])
    rec_predictions = []
    for recid, idxs in rec_to_rows.items():
        mean_sm = test_softmax[idxs].mean(axis=0)
        pred = int(mean_sm.argmax())
        true = test_meta[idxs[0]][1]
        rec_total += 1
        rec_per_raga[true][1] += 1
        if pred == true:
            rec_correct += 1
            rec_per_raga[true][0] += 1
        rec_predictions.append((true, pred))
    final_rec_acc = rec_correct / rec_total * 100

    # ── Final report ───────────────────────────────────────────────────
    bar = "=" * 78
    print(bar, file=tee)
    print("FINAL TEST METRICS — Mel-spec CNN (v4)", file=tee)
    print(bar, file=tee)
    print(f"Total windows: {len(windows):,}  "
          f"(train {len(train_windows):,} / test {len(test_windows):,})", file=tee)
    print(f"Unique recordings: train {len(set(rec_ids[train_idx])):,}  /  "
          f"test {len(set(rec_ids[test_idx])):,}  (zero leakage)", file=tee)
    print(file=tee)
    print(f"Per-window test accuracy:    {final_row_acc:.2f}%", file=tee)
    print(f"Per-recording vote accuracy: {final_rec_acc:.2f}%  "
          f"({rec_correct}/{rec_total} recordings)", file=tee)
    print(file=tee)
    print("Comparison vs prior models on the same kind of test set:", file=tee)
    print(f"  v1 (histogram, expert-pipeline test): 84.4%  (paper number, not "
          f"directly comparable — expert features at test time)", file=tee)
    print(f"  v3 (histogram, multi-scale, recording vote): 53.62%  "
          f"(production-realistic)", file=tee)
    print(f"  v4 (mel-spec CNN, recording vote): {final_rec_acc:.2f}%", file=tee)
    print(file=tee)

    # Per-raga (recording-level), sorted ascending.
    print("Per-raga test accuracy (per-recording vote, ascending — worst first):",
          file=tee)
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

    # Top-10 confused pairs at recording level.
    pair_counts = Counter()
    for true, pred in rec_predictions:
        if true != pred:
            pair_counts[(true, pred)] += 1
    print("Top 10 most-confused (true → predicted) pairs at the recording level:",
          file=tee)
    print(f"  {'count':>5}  {'true':<28}  →  {'predicted':<28}", file=tee)
    print(f"  {'-'*5}  {'-'*28}  ----  {'-'*28}", file=tee)
    for (t, p), c in pair_counts.most_common(10):
        print(f"  {c:>5}  {classes[t]:<28}  →  {classes[p]:<28}", file=tee)
    print(file=tee)

    # Per-epoch traces.
    print("Per-epoch val_acc trace (per-window):", file=tee)
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
