"""
Gate 1: read-only sanity check on the audio-clip features before training.

Compares the 44,071 newly-extracted clip features (X_audio_clips.npy, built
from raw MP3s + expert .tonicFine values) against the original 480-row X.npy
(built from CompMusic's expert pitch + nyas annotations) to confirm the new
data is in the same feature space and recognizably the same ragas.

Outputs:
  data/gate1_report.txt                  Full text report
  data/gate1_kalyani_comparison.png      Overlaid distribution plot
  (both go to backend/data/, nothing else is touched)

Run from backend/:
  source venv/bin/activate
  python src/gate1_report.py
"""

import json
import os
import random
import sys
from collections import Counter
from io import StringIO

import numpy as np

# Headless plotting — no display required.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(THIS_DIR)
DATA_DIR    = os.path.join(BACKEND_DIR, "data")
FEATURES_DIR = os.path.join(DATA_DIR, "RagaDataset", "Carnatic", "features")
MAPPING_PATH = os.path.join(DATA_DIR, "RagaDataset", "Carnatic", "_info_",
                            "ragaId_to_ragaName_mapping.json")
CLASSES_PATH = os.path.join(DATA_DIR, "classes.json")
AUDIO_DIR    = os.path.expanduser("~/raga-data-audio/RagaDataset/Carnatic/audio")

X_OLD_PATH    = os.path.join(DATA_DIR, "X.npy")
Y_OLD_PATH    = os.path.join(DATA_DIR, "y.npy")
X_CLIPS_PATH  = os.path.join(DATA_DIR, "X_audio_clips.npy")
Y_CLIPS_PATH  = os.path.join(DATA_DIR, "y_audio_clips.npy")
META_PATH     = os.path.join(DATA_DIR, "audio_clips_meta.json")

REPORT_PATH = os.path.join(DATA_DIR, "gate1_report.txt")
PLOT_PATH   = os.path.join(DATA_DIR, "gate1_kalyani_comparison.png")


# ── helpers ────────────────────────────────────────────────────────────────

class Tee:
    """Write everything to both stdout and an in-memory buffer for the report."""
    def __init__(self):
        self.buf = StringIO()
    def write(self, s):
        sys.__stdout__.write(s)
        self.buf.write(s)
    def flush(self):
        sys.__stdout__.flush()


def section(tee, title):
    bar = "=" * 78
    print(bar, file=tee)
    print(title, file=tee)
    print(bar, file=tee)


def cos_sim(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def top_k_peaks(channel, k=5):
    """Return list of (bin_idx, value) sorted by value desc."""
    idx = np.argsort(channel)[::-1][:k]
    return [(int(i), float(channel[i])) for i in idx]


def fmt_peaks(peaks):
    return ", ".join(f"bin{i:3d} ({i*10:>4d}¢)={v:.4f}" for i, v in peaks)


def build_xnpy_recording_index(classes, y_old):
    """Walk features dir in the same order preprocess.py used and produce
    X.npy row → (recording_id, raga_name) mapping. The recording_id format
    matches audio_clips_meta.json: '{raga_id}/{artist}/{album}/{song}/{song}'.
    Audio paths nest one extra dir for the song filename, so we append the
    pitchSilIntrpPP basename (without ext) to the features relpath.

    Returns the mapping only if the walk reproduces the exact label sequence
    in y.npy. Returns None otherwise — the caller should skip the cross-
    comparison sections rather than risk pointing at the wrong row.

    The walk can have more entries than y.npy: preprocess.build_dataset()
    skips any recording where extract_features() returns None (missing
    .tonicFine, too little voiced pitch, missing channel). In that case
    we can't recover which specific recording was dropped within a raga,
    so we don't try to guess."""
    with open(MAPPING_PATH, encoding="utf-8") as f:
        id_to_name = json.load(f)

    name_to_label = {n: i for i, n in enumerate(classes)}

    raga_ids = sorted(os.listdir(FEATURES_DIR))
    out = []           # row index → (recording_id, raga_name)
    walk_labels = []
    for raga_id in raga_ids:
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
            recording_id = f"{rel}/{base}"
            out.append((recording_id, raga_name))
            walk_labels.append(label)

    if (len(out) == len(y_old)
            and np.array_equal(np.asarray(walk_labels), y_old)):
        return out
    return None


def find_match_in_clips(recording_id, meta):
    """Return list of clip indices in X_audio_clips for this recording_id."""
    return [i for i, m in enumerate(meta) if m["recording_id"] == recording_id]


def _tonicfine_path_for_audio(audio_path):
    """Map an audio .mp3 path to its parallel .tonicFine path under FEATURES_DIR.
    Mirrors preprocess_audio_clips.tonicfine_path so we sample the same set
    of recordings as verify_tonic_detection.py (which only includes MP3s
    with a matching .tonicFine annotation)."""
    rel_no_ext = os.path.splitext(os.path.relpath(audio_path, AUDIO_DIR))[0]
    return os.path.join(FEATURES_DIR, rel_no_ext + ".tonicFine")


# ── sections ───────────────────────────────────────────────────────────────

def section1_dataset_summary(tee, X_clips, y_clips, classes):
    section(tee, "SECTION 1 — Dataset summary")
    print(f"X_audio_clips.npy shape: {X_clips.shape}", file=tee)
    print(f"y_audio_clips.npy shape: {y_clips.shape}", file=tee)
    print(f"Total clips: {len(y_clips):,}", file=tee)
    print(f"Distinct labels present: {len(np.unique(y_clips))} / {len(classes)}", file=tee)
    print(file=tee)

    counts = Counter(y_clips.tolist())
    print("Per-raga clip counts:", file=tee)
    print(f"  {'#':>2}  {'raga':<28}  {'clips':>7}", file=tee)
    print(f"  {'-'*2}  {'-'*28}  {'-'*7}", file=tee)
    for label in range(len(classes)):
        n = counts.get(label, 0)
        print(f"  {label:>2}  {classes[label]:<28}  {n:>7,}", file=tee)
    print(file=tee)

    if counts:
        cmax = max(counts.values()); cmin = min(counts.values())
        print(f"Class imbalance ratio (max/min): {cmax}/{cmin} = {cmax/cmin:.2f}×", file=tee)
        print(f"Median clips per raga: {int(np.median(list(counts.values())))}", file=tee)
        print(f"Mean clips per raga:   {np.mean(list(counts.values())):.1f}", file=tee)
    print(file=tee)


def section2_tonic_verification(tee, meta):
    section(tee, "SECTION 2 — Tonic verification on the original 5 recordings")
    print("(Using the same selection logic as verify_tonic_detection.py:", file=tee)
    print(" seed=42, walk AUDIO_DIR, sort raga_ids, rng.sample(5) then rng.choice.)", file=tee)
    print(file=tee)

    with open(MAPPING_PATH, encoding="utf-8") as f:
        id_to_name = json.load(f)

    # Match verify_tonic_detection.py exactly: only include MP3s that have
    # a corresponding .tonicFine annotation. Skipping this filter would let
    # extra/unannotated audio files into the seeded RNG pool and pick
    # recordings that aren't in audio_clips_meta.json.
    pairs_by_raga = {}
    for raga_id in sorted(os.listdir(AUDIO_DIR)):
        raga_name = id_to_name.get(raga_id)
        if raga_name is None:
            continue
        for root, _, files in os.walk(os.path.join(AUDIO_DIR, raga_id)):
            for fname in files:
                if not fname.lower().endswith(".mp3"):
                    continue
                apath = os.path.join(root, fname)
                if not os.path.exists(_tonicfine_path_for_audio(apath)):
                    continue
                pairs_by_raga.setdefault(raga_id, []).append((raga_name, apath))

    rng = random.Random(42)
    available = sorted(pairs_by_raga.keys())
    chosen_ragas = rng.sample(available, min(5, len(available)))

    by_recid = {}
    for m in meta:
        by_recid.setdefault(m["recording_id"], []).append(m)

    print(f"  {'raga':<22}  {'expert tonic Hz':>15}  {'clips':>6}  {'consistent':>10}",
          file=tee)
    print(f"  {'-'*22}  {'-'*15}  {'-'*6}  {'-'*10}", file=tee)
    for raga_id in chosen_ragas:
        raga_name, apath = rng.choice(pairs_by_raga[raga_id])
        rel_no_ext = os.path.splitext(os.path.relpath(apath, AUDIO_DIR))[0]
        clips = by_recid.get(rel_no_ext, [])

        if not clips:
            print(f"  {raga_name:<22}  {'(none in clips)':>15}  {0:>6}  {'-':>10}",
                  file=tee)
            continue

        tonics = {round(c["expert_tonic_hz"], 6) for c in clips}
        consistent = "yes" if len(tonics) == 1 else f"NO ({len(tonics)})"
        tonic_hz = clips[0]["expert_tonic_hz"]
        print(f"  {raga_name:<22}  {tonic_hz:>15.4f}  {len(clips):>6}  {consistent:>10}",
              file=tee)
        print(f"    recording_id: {rel_no_ext}", file=tee)
    print(file=tee)


def per_channel_compare(tee, x_old_row, clip_rows):
    """Per-channel top-5 peaks and cosine similarity between X.npy row and 3 clip rows."""
    chan_names = [("nyas",     0,   120),
                  ("duration", 120, 240),
                  ("stable",   240, 360)]
    for cname, lo, hi in chan_names:
        old = x_old_row[lo:hi]
        print(f"  channel '{cname}' [{lo}:{hi}]", file=tee)
        print(f"    X.npy        peaks: {fmt_peaks(top_k_peaks(old))}", file=tee)
        for ci, c in enumerate(clip_rows):
            cc = c[lo:hi]
            print(f"    clip{ci+1:<8} peaks: {fmt_peaks(top_k_peaks(cc))}", file=tee)
        for ci, c in enumerate(clip_rows):
            cc = c[lo:hi]
            print(f"    cos(X, clip{ci+1}) = {cos_sim(old, cc):+.4f}", file=tee)
    print(f"  full-360 cosine vs clip-mean: "
          f"{cos_sim(x_old_row, np.mean(np.asarray(clip_rows), axis=0)):+.4f}",
          file=tee)
    print(file=tee)


def section3_kalyani(tee, X_old, X_clips, meta, x_index):
    section(tee, "SECTION 3 — Kalyāṇi feature comparison (the critical check)")

    # Find a Kalyāṇi recording that exists in BOTH X.npy and X_audio_clips.
    target = None
    for row_idx, (recid, raga_name) in enumerate(x_index):
        if "Kalyāṇi" in raga_name or raga_name == "Kalyāṇi":
            clips = find_match_in_clips(recid, meta)
            if clips:
                target = (row_idx, recid, raga_name, clips)
                break
    if target is None:
        print("FAIL: no Kalyāṇi recording present in both X.npy and X_audio_clips", file=tee)
        return None

    row_idx, recid, raga_name, clip_idxs = target
    print(f"Matched recording: {recid}", file=tee)
    print(f"  raga:           {raga_name}", file=tee)
    print(f"  X.npy row:      {row_idx}", file=tee)
    print(f"  clips in X_audio_clips: {len(clip_idxs)}", file=tee)
    print(f"  expert tonic:   {meta[clip_idxs[0]]['expert_tonic_hz']:.4f} Hz", file=tee)
    print(file=tee)

    # 3 sample clips spread across the recording.
    sample_picks = [clip_idxs[0],
                    clip_idxs[len(clip_idxs)//2],
                    clip_idxs[-1]]
    sample_rows = [X_clips[i] for i in sample_picks]
    sample_starts = [meta[i]["clip_start"] for i in sample_picks]

    print(f"Sampled clip start times: {sample_starts}", file=tee)
    print(file=tee)
    per_channel_compare(tee, X_old[row_idx], sample_rows)
    return row_idx, recid, raga_name, sample_picks, sample_rows


def make_kalyani_plot(X_old_row, sample_rows, sample_starts, recid):
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    chan_names = [("nyas (bins 0–119)",     0,   120),
                  ("duration (bins 120–239)", 120, 240),
                  ("stable (bins 240–359)",   240, 360)]
    bins = np.arange(120) * 10  # cents per bin

    for ax, (cname, lo, hi) in zip(axes, chan_names):
        ax.plot(bins, X_old_row[lo:hi], label="X.npy (expert pitch+nyas)",
                color="black", linewidth=2.0)
        colors = ["tab:blue", "tab:orange", "tab:green"]
        for c, row, t, color in zip(range(3), sample_rows, sample_starts, colors):
            ax.plot(bins, row[lo:hi], label=f"clip @ {t:.0f}s",
                    color=color, alpha=0.75, linewidth=1.0)
        ax.set_title(f"channel: {cname}")
        ax.set_ylabel("density")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("cents above tonic (Sa = 0)")
    fig.suptitle(f"Kalyāṇi — X.npy row vs 3 audio clips\n{recid}", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=120)
    plt.close(fig)


def section4_sanity(tee, X_old, X_clips):
    section(tee, "SECTION 4 — Whole-dataset sanity checks")
    nan_rows = int(np.isnan(X_clips).any(axis=1).sum())
    inf_rows = int(np.isinf(X_clips).any(axis=1).sum())
    print(f"Clips with any NaN: {nan_rows}", file=tee)
    print(f"Clips with any Inf: {inf_rows}", file=tee)

    chan_slices = [("nyas", slice(0,120)), ("duration", slice(120,240)),
                   ("stable", slice(240,360))]
    zero_total = 0
    for cname, sl in chan_slices:
        n_zero = int((X_clips[:, sl].sum(axis=1) == 0).sum())
        zero_total += n_zero
        print(f"Clips with all-zero {cname:>8} channel: {n_zero}", file=tee)
    print(f"Clips with at least one all-zero channel: {zero_total} (sum across channels — may double-count)",
          file=tee)
    print(file=tee)

    print(f"  {'channel':<8}  {'clips mean':>11}  {'clips std':>10}  "
          f"{'X.npy mean':>11}  {'X.npy std':>10}", file=tee)
    print(f"  {'-'*8}  {'-'*11}  {'-'*10}  {'-'*11}  {'-'*10}", file=tee)
    for cname, sl in chan_slices:
        cm = X_clips[:, sl].mean(); cs = X_clips[:, sl].std()
        om = X_old[:, sl].mean();   os = X_old[:, sl].std()
        print(f"  {cname:<8}  {cm:>11.6f}  {cs:>10.6f}  {om:>11.6f}  {os:>10.6f}",
              file=tee)
    print(file=tee)
    print(f"Overall mean (clips):  {X_clips.mean():.6f}", file=tee)
    print(f"Overall mean (X.npy):  {X_old.mean():.6f}", file=tee)
    print(f"Overall std  (clips):  {X_clips.std():.6f}", file=tee)
    print(f"Overall std  (X.npy):  {X_old.std():.6f}", file=tee)
    print(file=tee)


def section5_todi(tee, X_old, X_clips, meta, x_index):
    section(tee, "SECTION 5 — Tōḍi feature comparison (different swara set)")
    target = None
    for row_idx, (recid, raga_name) in enumerate(x_index):
        if "Tōḍi" in raga_name or raga_name == "Tōḍi":
            clips = find_match_in_clips(recid, meta)
            if clips:
                target = (row_idx, recid, raga_name, clips)
                break
    if target is None:
        print("FAIL: no Tōḍi recording present in both X.npy and X_audio_clips", file=tee)
        return

    row_idx, recid, raga_name, clip_idxs = target
    print(f"Matched recording: {recid}", file=tee)
    print(f"  raga:           {raga_name}", file=tee)
    print(f"  X.npy row:      {row_idx}", file=tee)
    print(f"  clips in X_audio_clips: {len(clip_idxs)}", file=tee)
    print(f"  expert tonic:   {meta[clip_idxs[0]]['expert_tonic_hz']:.4f} Hz", file=tee)
    print(file=tee)

    sample_picks = [clip_idxs[0],
                    clip_idxs[len(clip_idxs)//2],
                    clip_idxs[-1]]
    sample_rows = [X_clips[i] for i in sample_picks]
    sample_starts = [meta[i]["clip_start"] for i in sample_picks]
    print(f"Sampled clip start times: {sample_starts}", file=tee)
    print(file=tee)
    per_channel_compare(tee, X_old[row_idx], sample_rows)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    tee = Tee()

    # Load everything once.
    with open(CLASSES_PATH, encoding="utf-8") as f:
        classes = json.load(f)
    X_old   = np.load(X_OLD_PATH)
    y_old   = np.load(Y_OLD_PATH)
    X_clips = np.load(X_CLIPS_PATH)
    y_clips = np.load(Y_CLIPS_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)

    print(f"Loaded:", file=tee)
    print(f"  X.npy            {X_old.shape}", file=tee)
    print(f"  y.npy            {y_old.shape}", file=tee)
    print(f"  X_audio_clips    {X_clips.shape}", file=tee)
    print(f"  y_audio_clips    {y_clips.shape}", file=tee)
    print(f"  meta entries     {len(meta):,}", file=tee)
    print(f"  classes          {len(classes)}", file=tee)
    print(file=tee)

    if len(meta) != len(X_clips):
        print(f"WARNING: meta length ({len(meta)}) != X_audio_clips rows ({len(X_clips)})",
              file=tee)

    section1_dataset_summary(tee, X_clips, y_clips, classes)
    section2_tonic_verification(tee, meta)

    # Returns None if the features-dir walk doesn't exactly reproduce the
    # label sequence in y.npy (e.g. preprocess skipped a recording on this
    # filesystem). In that case the X.npy row → recording_id mapping isn't
    # trustworthy and we skip the per-recording comparison sections rather
    # than silently compare unrelated recordings.
    x_index = build_xnpy_recording_index(classes, y_old)
    if x_index is None:
        section(tee, "SECTIONS 3 & 5 — SKIPPED")
        print("Could not align X.npy rows to recording_ids on this filesystem.",
              file=tee)
        print("(features-dir walk did not match y.npy label sequence.)", file=tee)
        print(file=tee)
        section4_sanity(tee, X_old, X_clips)
    else:
        kal = section3_kalyani(tee, X_old, X_clips, meta, x_index)
        if kal is not None:
            row_idx, recid, _raga_name, sample_picks, sample_rows = kal
            sample_starts = [meta[i]["clip_start"] for i in sample_picks]
            make_kalyani_plot(X_old[row_idx], sample_rows, sample_starts, recid)
            print(f"Saved plot: {PLOT_PATH}", file=tee)
            print(file=tee)

        section4_sanity(tee, X_old, X_clips)
        section5_todi(tee, X_old, X_clips, meta, x_index)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(tee.buf.getvalue())
    print(f"\nReport written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
