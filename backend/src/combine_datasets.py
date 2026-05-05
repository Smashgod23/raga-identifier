"""
Step C: combine the three feature sets into one training dataset.

Inputs (untouched, treated as read-only working backups):
  data/X.npy             (480, 360)  — CompMusic pitch-file features
  data/y.npy             (480,)
  data/X_yt.npy          (209, 360)  — YouTube extracted features
  data/y_yt.npy          (209,)
  data/X_audio_clips.npy (44071, 360) — 30s clips from CompMusic raw MP3s
  data/y_audio_clips.npy (44071,)
  data/audio_clips_meta.json — recording_id per clip row

Outputs:
  data/X_combined.npy        (44760, 360)
  data/y_combined.npy        (44760,)
  data/combined_meta.json    [{source, recording_id, label, raga_name, ...}, ...]

The combined_meta records the source (compmusic / youtube / audio_clip) and
recording_id for every row so train_v2.py can do a GroupShuffleSplit at the
recording level — clips from the same MP3 must stay on the same side of the
train/test boundary.

CompMusic X.npy rows and audio_clip rows that came from the same recording
get the same recording_id, so all features for one recording group together
in the split.

The script ends with a hard sanity check (per-source counts, meta length,
unique recording counts). It exits non-zero if anything looks wrong, so a
caller can gate training on its success.

Run from backend/:
  source venv/bin/activate
  python src/combine_datasets.py
"""

import json
import os
import sys

import numpy as np

THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(THIS_DIR)
DATA_DIR    = os.path.join(BACKEND_DIR, "data")
FEATURES_DIR = os.path.join(DATA_DIR, "RagaDataset", "Carnatic", "features")
MAPPING_PATH = os.path.join(DATA_DIR, "RagaDataset", "Carnatic", "_info_",
                            "ragaId_to_ragaName_mapping.json")
CLASSES_PATH = os.path.join(DATA_DIR, "classes.json")

X_OLD       = os.path.join(DATA_DIR, "X.npy")
Y_OLD       = os.path.join(DATA_DIR, "y.npy")
X_YT        = os.path.join(DATA_DIR, "X_yt.npy")
Y_YT        = os.path.join(DATA_DIR, "y_yt.npy")
X_CLIPS     = os.path.join(DATA_DIR, "X_audio_clips.npy")
Y_CLIPS     = os.path.join(DATA_DIR, "y_audio_clips.npy")
META_CLIPS  = os.path.join(DATA_DIR, "audio_clips_meta.json")

X_OUT       = os.path.join(DATA_DIR, "X_combined.npy")
Y_OUT       = os.path.join(DATA_DIR, "y_combined.npy")
META_OUT    = os.path.join(DATA_DIR, "combined_meta.json")


def build_compmusic_recording_ids(classes, y_old):
    """Walk the features dir in the same order preprocess.py used and return
    one recording_id per X.npy row. Aborts on any walk-order mismatch — for
    train_v2 we *must* match audio_clip recording_ids exactly so a CompMusic
    recording and its derived clips group together in the recording-aware
    split."""
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
            f"y.npy has {len(y_old)} rows. Re-run preprocess.py to regenerate "
            f"X.npy in the current walk order, or investigate dataset drift."
        )
    if not np.array_equal(np.asarray(walk_labels), y_old):
        n_diff = int((np.asarray(walk_labels) != y_old).sum())
        raise SystemExit(
            f"ABORT: walk label sequence has {n_diff} mismatches against y.npy. "
            f"X.npy row → recording_id mapping is unreliable on this filesystem."
        )
    return rec_ids, raga_names


def main():
    print(">>> Step C: combine X.npy + X_yt.npy + X_audio_clips.npy", flush=True)
    print(flush=True)

    with open(CLASSES_PATH, encoding="utf-8") as f:
        classes = json.load(f)
    X_old   = np.load(X_OLD);   y_old   = np.load(Y_OLD)
    X_yt    = np.load(X_YT);    y_yt    = np.load(Y_YT)
    X_clips = np.load(X_CLIPS); y_clips = np.load(Y_CLIPS)
    with open(META_CLIPS, encoding="utf-8") as f:
        clips_meta = json.load(f)

    print(f"Loaded:")
    print(f"  X.npy            {X_old.shape}   y.npy   {y_old.shape}")
    print(f"  X_yt.npy         {X_yt.shape}    y_yt    {y_yt.shape}")
    print(f"  X_audio_clips    {X_clips.shape} y_clips {y_clips.shape}")
    print(f"  audio_clips_meta {len(clips_meta):,} entries")
    print(f"  classes          {len(classes)}")
    print(flush=True)

    # Sanity: feature widths match before we concat.
    if not (X_old.shape[1] == X_yt.shape[1] == X_clips.shape[1] == 360):
        raise SystemExit(
            f"ABORT: feature dim mismatch — X.npy={X_old.shape[1]}, "
            f"X_yt.npy={X_yt.shape[1]}, X_audio_clips={X_clips.shape[1]}"
        )
    if len(clips_meta) != len(X_clips):
        raise SystemExit(
            f"ABORT: audio_clips_meta has {len(clips_meta)} rows but "
            f"X_audio_clips has {len(X_clips)}"
        )

    # CompMusic recording_ids — must match the audio_clip recording_id format
    # so the same physical recording groups across both sources.
    cm_rec_ids, cm_raga_names = build_compmusic_recording_ids(classes, y_old)

    # ── Build the combined arrays + per-row metadata ─────────────────────
    X_combined = np.concatenate([X_old, X_yt, X_clips], axis=0)
    y_combined = np.concatenate([y_old, y_yt, y_clips], axis=0)

    meta = []
    # CompMusic rows: 0 .. 480
    for i in range(len(X_old)):
        meta.append({
            "source":       "compmusic",
            "recording_id": cm_rec_ids[i],
            "label":        int(y_old[i]),
            "raga_name":    cm_raga_names[i],
        })
    # YouTube rows: 480 .. 689
    # Each YouTube row is its own unique recording — no clip overlap with
    # anything else — so a synthetic-but-unique recording_id is sufficient
    # for the recording-aware split. The original videos.json doesn't
    # reliably map to X_yt row order (some downloads failed silently in
    # the original run), so we don't try to recover the actual video IDs.
    for i in range(len(X_yt)):
        label = int(y_yt[i])
        meta.append({
            "source":       "youtube",
            "recording_id": f"youtube_row_{i}",
            "label":        label,
            "raga_name":    classes[label],
        })
    # Audio-clip rows: 689 .. 44760
    for i, m in enumerate(clips_meta):
        meta.append({
            "source":       "audio_clip",
            "recording_id": m["recording_id"],
            "label":        int(y_clips[i]),
            "raga_name":    m["raga_name"],
            "clip_start":   m.get("clip_start"),
            "clip_dur":     m.get("clip_dur"),
            "expert_tonic_hz": m.get("expert_tonic_hz"),
        })

    if len(meta) != len(X_combined):
        raise SystemExit(
            f"ABORT: meta length {len(meta)} != X_combined rows {len(X_combined)}"
        )

    # Write outputs.
    np.save(X_OUT, X_combined)
    np.save(Y_OUT, y_combined)
    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    print(f"Wrote {X_OUT}")
    print(f"Wrote {Y_OUT}")
    print(f"Wrote {META_OUT}")
    print(flush=True)

    # ── Sanity check ─────────────────────────────────────────────────────
    print(">>> Sanity check")
    print()

    # Reload from disk to make sure the persisted artifacts agree with what
    # we just built (catches a partial-write or path bug).
    Xc = np.load(X_OUT)
    yc = np.load(Y_OUT)
    with open(META_OUT, encoding="utf-8") as f:
        meta_disk = json.load(f)

    expected_total = len(X_old) + len(X_yt) + len(X_clips)
    print(f"  X_combined shape:           {Xc.shape}    "
          f"(expected ({expected_total}, 360))")
    print(f"  y_combined shape:           {yc.shape}    "
          f"(expected ({expected_total},))")
    print(f"  combined_meta row count:    {len(meta_disk):,} "
          f"(must match X_combined rows {len(Xc):,})")

    by_source = {"compmusic": [], "youtube": [], "audio_clip": []}
    for row_idx, m in enumerate(meta_disk):
        by_source.setdefault(m["source"], []).append(row_idx)

    print()
    print(f"  {'source':<11}  {'rows':>7}  {'unique recording_ids':>22}")
    print(f"  {'-'*11}  {'-'*7}  {'-'*22}")
    for src, idxs in by_source.items():
        unique = len({meta_disk[i]["recording_id"] for i in idxs})
        print(f"  {src:<11}  {len(idxs):>7,}  {unique:>22,}")
    print()

    # Hard checks — abort before training if anything is off.
    expected = {
        ("rows", "compmusic"):  len(X_old),
        ("rows", "youtube"):    len(X_yt),
        ("rows", "audio_clip"): len(X_clips),
    }
    failures = []
    for src, n_expected in (("compmusic", len(X_old)),
                            ("youtube", len(X_yt)),
                            ("audio_clip", len(X_clips))):
        n_actual = len(by_source.get(src, []))
        if n_actual != n_expected:
            failures.append(f"{src} rows: got {n_actual}, expected {n_expected}")

    if Xc.shape != (expected_total, 360):
        failures.append(f"X_combined shape {Xc.shape} != ({expected_total}, 360)")
    if yc.shape != (expected_total,):
        failures.append(f"y_combined shape {yc.shape} != ({expected_total},)")
    if len(meta_disk) != len(Xc):
        failures.append(
            f"meta length {len(meta_disk)} != X_combined rows {len(Xc)}"
        )

    # Each YouTube row must have a unique recording_id (1:1 with rows).
    yt_unique = len({meta_disk[i]["recording_id"] for i in by_source["youtube"]})
    if yt_unique != len(by_source["youtube"]):
        failures.append(
            f"youtube source: {yt_unique} unique recording_ids "
            f"but {len(by_source['youtube'])} rows (must be 1:1)"
        )

    # CompMusic and audio_clip recording_ids should overlap heavily — same
    # physical recording on both sides — though not necessarily 100% if
    # any audio file is missing. Just print, don't fail on this.
    cm_ids = {meta_disk[i]["recording_id"] for i in by_source["compmusic"]}
    ac_ids = {meta_disk[i]["recording_id"] for i in by_source["audio_clip"]}
    overlap = cm_ids & ac_ids
    print(f"  compmusic ∩ audio_clip recording_id overlap: "
          f"{len(overlap)} (of {len(cm_ids)} compmusic, {len(ac_ids)} audio_clip)")
    print()

    if failures:
        print("SANITY CHECK FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(2)

    print("SANITY CHECK PASSED — safe to run train_v2.py.")


if __name__ == "__main__":
    main()
