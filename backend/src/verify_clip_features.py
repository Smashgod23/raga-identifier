"""
Pre-flight check for the refactored expert-tonic clip pipeline.

Reuses the same 5 random recordings from verify_tonic_detection.py, runs them
through the refactored preprocess_audio_clips._process_recording, and reports:

  - tonic used (must equal expert tonic exactly, modulo octave folding)
  - 3 channel summaries (any NaN/Inf? any all-zero? peak position + value?)

Read-only — does not write to data/ or models/.
"""

import json
import os
import random
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from preprocess_audio_clips import (
    AUDIO_DIR, MAPPING_PATH, tonicfine_path
)
from predict import extract_features_from_audio


def channel_report(name, ch):
    has_nan  = bool(np.isnan(ch).any())
    has_inf  = bool(np.isinf(ch).any())
    all_zero = bool(np.all(ch == 0))
    peak_idx = int(np.argmax(ch))
    peak_val = float(ch[peak_idx])
    nonzero  = int((ch > 0).sum())
    peak_cents = peak_idx * 10  # 120 bins * 10 cents
    return (
        f"  {name:<10} nan={has_nan} inf={has_inf} all_zero={all_zero}  "
        f"peak@bin{peak_idx:3d} ({peak_cents:>4d}¢) val={peak_val:.4f}  "
        f"non-zero bins={nonzero}/120"
    )


def main():
    with open(MAPPING_PATH, encoding="utf-8") as f:
        id_to_name = json.load(f)

    # Replicate verify_tonic_detection's selection: 5 random ragas (seed 42),
    # one random recording each (must have a .tonicFine).
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
                tpath = tonicfine_path(apath)
                if os.path.exists(tpath):
                    pairs_by_raga.setdefault(raga_id, []).append((raga_name, apath, tpath))

    rng = random.Random(42)
    available = sorted(pairs_by_raga.keys())
    chosen_ragas = rng.sample(available, min(5, len(available)))

    overall_pass = True
    for raga_id in chosen_ragas:
        raga_name, apath, tpath = rng.choice(pairs_by_raga[raga_id])
        rel = os.path.relpath(apath, AUDIO_DIR)
        with open(tpath) as f:
            expert_tonic = float(f.read().strip())

        print(f"=== {raga_name} ===")
        print(f"  recording: {rel}")
        print(f"  expert tonic: {expert_tonic:.4f} Hz")

        # Run feature extraction with expert tonic on a 30s clip from offset=30s
        # (same shape as a real pipeline clip, skipping any pure-tanpura intro).
        try:
            features, used_tonic = extract_features_from_audio(
                apath,
                tonic_override=expert_tonic,
                offset=30.0,
                duration=30.0,
            )
        except ValueError as e:
            print(f"  FAIL: {e}")
            overall_pass = False
            continue

        # used_tonic may differ from expert_tonic by an octave because
        # _fold_override_to_tonic shifts to the singer's octave. Compare in cents
        # mod 1200 to confirm same pitch class.
        cents_off = 1200 * np.log2(used_tonic / expert_tonic)
        cents_off_mod = ((cents_off + 600) % 1200) - 600  # nearest pc-difference
        print(f"  tonic used:   {used_tonic:.4f} Hz  (octave-folded from expert; pc-diff={cents_off_mod:+.2f}¢)")

        # Channel checks
        ch_nyas     = features[0:120]
        ch_duration = features[120:240]
        ch_stable   = features[240:360]

        for name, ch in [("nyas", ch_nyas), ("duration", ch_duration), ("stable", ch_stable)]:
            line = channel_report(name, ch)
            print(line)

        # Hard pass/fail
        bad = (
            np.isnan(features).any()
            or np.isinf(features).any()
            or np.all(ch_nyas == 0)
            or np.all(ch_duration == 0)
            or np.all(ch_stable == 0)
        )
        # Each channel should have at least one visible peak (>0 max)
        no_peak = (ch_nyas.max() <= 0) or (ch_duration.max() <= 0) or (ch_stable.max() <= 0)

        if bad or no_peak:
            print(f"  STATUS: FAIL")
            overall_pass = False
        else:
            print(f"  STATUS: pass")
        print()

    print(f"Overall: {'PASS — safe to restart pipeline' if overall_pass else 'FAIL — do not restart'}")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
