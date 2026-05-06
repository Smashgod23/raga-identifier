"""
Step 2 of Phase 3: combine the six feature sets into one multi-scale dataset.

Inputs (treated read-only):
  data/X.npy             (480, 360)   CompMusic full-recording features
  data/y.npy             (480,)
  data/X_yt.npy          (209, 360)   YouTube full-recording features
  data/y_yt.npy          (209,)
  data/X_3min.npy        (6833, 360)  180s/60s windows
  data/y_3min.npy        (6833,)
  data/3min_meta.json
  data/X_1min.npy        (21825, 360) 60s/20s windows  ← cap source
  data/y_1min.npy        (21825,)
  data/1min_meta.json
  data/X_audio_clips.npy (44071, 360) 30s/10s windows from Phase 2
  data/y_audio_clips.npy (44071,)
  data/audio_clips_meta.json
  data/X_15s.npy         (88183, 360) 15s/5s windows
  data/y_15s.npy         (88183,)
  data/15s_meta.json

Outputs:
  data/X_multiscale.npy
  data/y_multiscale.npy
  data/multiscale_meta.json   per-row {source, recording_id, label, raga_name, ...}

Sub-sampling:
  - Cap each clip-scale at the 1-minute row count (the natural balance point).
  - Stratified by raga label, with fixed seed 42, on the 30s and 15s sources.
  - Full-recording (689) and 3-min (6833) are kept as-is — already under cap.
  - 1-min (21825) is kept as-is — it IS the cap.

Recording-aware grouping is preserved at the meta level: every CompMusic
recording's full-recording row + its derived clips at every scale share the
same recording_id, so train_v3.py's GroupShuffleSplit will keep them on the
same side of the train/test boundary.

Run from backend/:
  source venv/bin/activate
  python src/build_multiscale_dataset.py
"""

import json
import os
import sys
from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(THIS_DIR)
DATA_DIR    = os.path.join(BACKEND_DIR, "data")
FEATURES_DIR = os.path.join(DATA_DIR, "RagaDataset", "Carnatic", "features")
MAPPING_PATH = os.path.join(DATA_DIR, "RagaDataset", "Carnatic", "_info_",
                            "ragaId_to_ragaName_mapping.json")
CLASSES_PATH = os.path.join(DATA_DIR, "classes.json")

X_OUT       = os.path.join(DATA_DIR, "X_multiscale.npy")
Y_OUT       = os.path.join(DATA_DIR, "y_multiscale.npy")
META_OUT    = os.path.join(DATA_DIR, "multiscale_meta.json")

SEED = 42


def build_compmusic_recording_ids(classes, y_old):
    """Walk features dir to recover the X.npy row → recording_id mapping.
    Same logic as combine_datasets.py — abort on misalignment because we
    cannot safely group recordings across scales without it."""
    with open(MAPPING_PATH, encoding="utf-8") as f:
        id_to_name = json.load(f)
    name_to_label = {n: i for i, n in enumerate(classes)}

    rec_ids   = []
    raga_names = []
    walk_labels = []
    for raga_id in sorted(os.listdir(FEATURES_DIR)):
        if raga_id not in id_to_name:
            continue
        raga_name = id_to_name[raga_id]
        if raga_name not in name_to_label:
            continue
        label = name_to_label[raga_name]
        raga_dir = os.path.join(FEATURES_DIR, raga_id)
        for root, _dirs, files in os.walk(raga_dir):
            pitch_files = [f for f in files if f.endswith(".pitchSilIntrpPP")]
            if not pitch_files:
                continue
            base = os.path.splitext(pitch_files[0])[0]
            rel  = os.path.relpath(root, FEATURES_DIR)
            rec_ids.append(f"{rel}/{base}")
            raga_names.append(raga_name)
            walk_labels.append(label)

    if len(rec_ids) != len(y_old):
        raise SystemExit(
            f"ABORT: features-dir walk produced {len(rec_ids)} recordings but "
            f"y.npy has {len(y_old)} rows."
        )
    if not np.array_equal(np.asarray(walk_labels), y_old):
        raise SystemExit(
            "ABORT: walk label sequence does not match y.npy. "
            "X.npy row → recording_id mapping is unreliable on this filesystem."
        )
    return rec_ids, raga_names


def stratified_subsample(X, y, target, seed):
    """Return indices of `target` rows, stratified by y. Falls back to
    random subsample if any class has fewer than 2 rows."""
    if len(X) <= target:
        return np.arange(len(X))
    counts = np.bincount(y)
    if counts.min() < 2:
        rng = np.random.default_rng(seed)
        return rng.choice(len(X), size=target, replace=False)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=target, random_state=seed)
    keep_idx, _ = next(sss.split(np.zeros(len(X)), y))
    return keep_idx


def main():
    print(">>> Phase 3 step 2: build multi-scale combined dataset", flush=True)
    print(flush=True)

    # ── Load classes & all six sources ───────────────────────────────────
    with open(CLASSES_PATH, encoding="utf-8") as f:
        classes = json.load(f)

    X_full   = np.load(os.path.join(DATA_DIR, "X.npy"))
    y_full   = np.load(os.path.join(DATA_DIR, "y.npy"))
    X_yt     = np.load(os.path.join(DATA_DIR, "X_yt.npy"))
    y_yt     = np.load(os.path.join(DATA_DIR, "y_yt.npy"))
    X_3m     = np.load(os.path.join(DATA_DIR, "X_3min.npy"))
    y_3m     = np.load(os.path.join(DATA_DIR, "y_3min.npy"))
    X_1m     = np.load(os.path.join(DATA_DIR, "X_1min.npy"))
    y_1m     = np.load(os.path.join(DATA_DIR, "y_1min.npy"))
    X_30     = np.load(os.path.join(DATA_DIR, "X_audio_clips.npy"))
    y_30     = np.load(os.path.join(DATA_DIR, "y_audio_clips.npy"))
    X_15     = np.load(os.path.join(DATA_DIR, "X_15s.npy"))
    y_15     = np.load(os.path.join(DATA_DIR, "y_15s.npy"))

    with open(os.path.join(DATA_DIR, "3min_meta.json"), encoding="utf-8") as f:
        meta_3m = json.load(f)
    with open(os.path.join(DATA_DIR, "1min_meta.json"), encoding="utf-8") as f:
        meta_1m = json.load(f)
    with open(os.path.join(DATA_DIR, "audio_clips_meta.json"), encoding="utf-8") as f:
        meta_30 = json.load(f)
    with open(os.path.join(DATA_DIR, "15s_meta.json"), encoding="utf-8") as f:
        meta_15 = json.load(f)

    # Sanity: meta lengths must match feature matrices.
    for name, X, m in [("3min", X_3m, meta_3m), ("1min", X_1m, meta_1m),
                       ("30s", X_30, meta_30), ("15s", X_15, meta_15)]:
        if len(X) != len(m):
            raise SystemExit(
                f"ABORT: {name} feature rows ({len(X)}) != meta rows ({len(m)})"
            )

    # Sanity: feature widths.
    for name, X in [("X", X_full), ("X_yt", X_yt), ("X_3m", X_3m), ("X_1m", X_1m),
                    ("X_30", X_30), ("X_15", X_15)]:
        if X.shape[1] != 360:
            raise SystemExit(f"ABORT: {name} has {X.shape[1]} dims, expected 360")

    print(f"Source row counts:")
    print(f"  full (X.npy)            {len(X_full):>7,}")
    print(f"  youtube (X_yt.npy)      {len(X_yt):>7,}")
    print(f"  3min                    {len(X_3m):>7,}")
    print(f"  1min                    {len(X_1m):>7,}   ← cap source")
    print(f"  30s (X_audio_clips)     {len(X_30):>7,}")
    print(f"  15s                     {len(X_15):>7,}")
    print(flush=True)

    cap = len(X_1m)
    print(f"Cap = {cap:,} (the 1-minute row count). 30s and 15s "
          f"will be stratified-subsampled to this size with seed {SEED}.",
          flush=True)
    print(flush=True)

    # ── Build per-row records for each source ───────────────────────────
    cm_rec_ids, cm_raga_names = build_compmusic_recording_ids(classes, y_full)

    rows_X   = []
    rows_y   = []
    rows_meta = []

    # 1) CompMusic full-recording features (480 rows, kept as-is).
    for i in range(len(X_full)):
        rows_X.append(X_full[i])
        rows_y.append(int(y_full[i]))
        rows_meta.append({
            "source":       "compmusic_full",
            "recording_id": cm_rec_ids[i],
            "label":        int(y_full[i]),
            "raga_name":    cm_raga_names[i],
        })

    # 2) YouTube full-recording features (209 rows, kept as-is, synthetic IDs).
    for i in range(len(X_yt)):
        label = int(y_yt[i])
        rows_X.append(X_yt[i])
        rows_y.append(label)
        rows_meta.append({
            "source":       "youtube_full",
            "recording_id": f"youtube_row_{i}",
            "label":        label,
            "raga_name":    classes[label],
        })

    def append_scale(X_src, y_src, meta_src, scale_label, do_subsample):
        if do_subsample and len(X_src) > cap:
            keep = stratified_subsample(X_src, y_src, cap, SEED)
        else:
            keep = np.arange(len(X_src))
        for i in keep:
            rows_X.append(X_src[i])
            label = int(y_src[i])
            rows_y.append(label)
            m = meta_src[i]
            rows_meta.append({
                "source":            scale_label,
                "recording_id":      m["recording_id"],
                "label":             label,
                "raga_name":         m.get("raga_name", classes[label]),
                "window_start_sec":  m.get("clip_start"),
                "window_dur_sec":    m.get("clip_dur"),
                "expert_tonic_hz":   m.get("expert_tonic_hz"),
            })
        return len(keep)

    n_3min = append_scale(X_3m, y_3m, meta_3m, "scale_3min", do_subsample=False)
    n_1min = append_scale(X_1m, y_1m, meta_1m, "scale_1min", do_subsample=False)
    n_30s  = append_scale(X_30, y_30, meta_30, "scale_30s",  do_subsample=True)
    n_15s  = append_scale(X_15, y_15, meta_15, "scale_15s",  do_subsample=True)

    X_combined = np.vstack(rows_X).astype(np.float64)
    y_combined = np.array(rows_y, dtype=np.int64)

    np.save(X_OUT, X_combined)
    np.save(Y_OUT, y_combined)
    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(rows_meta, f, ensure_ascii=False)

    print(f"Wrote {X_OUT}")
    print(f"Wrote {Y_OUT}")
    print(f"Wrote {META_OUT}")
    print(flush=True)

    # ── Approval-gate stats ──────────────────────────────────────────────
    print(">>> Approval gate stats", flush=True)
    print()

    # Reload to make sure the persisted artifacts agree with what we built.
    Xc = np.load(X_OUT)
    yc = np.load(Y_OUT)
    with open(META_OUT, encoding="utf-8") as f:
        md = json.load(f)

    print(f"  X_multiscale shape: {Xc.shape}")
    print(f"  y_multiscale shape: {yc.shape}")
    print(f"  meta rows:          {len(md):,}    (must match X_multiscale rows)")
    print()

    by_source = {}
    for row_idx, m in enumerate(md):
        by_source.setdefault(m["source"], []).append(row_idx)

    print(f"  {'source':<16}  {'rows':>7}  {'unique recordings':>18}")
    print(f"  {'-'*16}  {'-'*7}  {'-'*18}")
    for src in ("compmusic_full", "youtube_full", "scale_3min", "scale_1min",
                "scale_30s", "scale_15s"):
        idxs = by_source.get(src, [])
        unique = len({md[i]["recording_id"] for i in idxs})
        print(f"  {src:<16}  {len(idxs):>7,}  {unique:>18,}")
    print()

    # Per-raga counts in the combined set.
    raga_counts = Counter(yc.tolist())
    cmin = min(raga_counts.values()) if raga_counts else 0
    cmax = max(raga_counts.values()) if raga_counts else 0
    print(f"  Per-raga combined counts: min={cmin:,}  median={int(np.median(list(raga_counts.values()))):,}  max={cmax:,}")
    print(f"  Imbalance ratio (max/min): {cmax/cmin if cmin else float('inf'):.2f}×")
    print()

    # Verify recording_id grouping property: every CompMusic recording_id
    # should appear under compmusic_full AND under each scale_* (since clips
    # from CompMusic recordings exist at every clip scale).
    cm_ids = {md[i]["recording_id"] for i in by_source["compmusic_full"]}
    for src in ("scale_3min", "scale_1min", "scale_30s", "scale_15s"):
        scale_ids = {md[i]["recording_id"] for i in by_source[src]}
        overlap = cm_ids & scale_ids
        print(f"  compmusic_full ∩ {src:<12} recording_ids: "
              f"{len(overlap):>3} / {len(cm_ids)} CompMusic, "
              f"{len(scale_ids):>3} unique in {src}")
    print()

    # NaN/Inf scan.
    nan_rows = int(np.isnan(Xc).any(axis=1).sum())
    inf_rows = int(np.isinf(Xc).any(axis=1).sum())
    print(f"  NaN rows: {nan_rows}    Inf rows: {inf_rows}")
    print()

    # Hard checks.
    failures = []
    if Xc.shape[1] != 360:
        failures.append(f"X_multiscale dim {Xc.shape[1]} != 360")
    if len(md) != len(Xc):
        failures.append(f"meta rows {len(md)} != X rows {len(Xc)}")
    if n_3min != len(X_3m):
        failures.append(f"3min: appended {n_3min}, expected {len(X_3m)}")
    if n_1min != len(X_1m):
        failures.append(f"1min: appended {n_1min}, expected {len(X_1m)}")
    if n_30s != cap:
        failures.append(f"30s: appended {n_30s}, expected {cap} (cap)")
    if n_15s != cap:
        failures.append(f"15s: appended {n_15s}, expected {cap} (cap)")
    if nan_rows or inf_rows:
        failures.append(f"feature matrix has {nan_rows} NaN / {inf_rows} Inf rows")

    if failures:
        print("APPROVAL GATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(2)

    print("APPROVAL GATE PASSED — safe to proceed to train_v3.py.")


if __name__ == "__main__":
    main()
