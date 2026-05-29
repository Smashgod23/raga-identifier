"""
Phase 13 validation: Essentia's expert tonic detector vs our heuristics.

The CompMusic expert tonic (.tonicFine) was produced by Gulati's
multipitch method, which ships in Essentia as TonicIndianArtMusic. If we
run it at inference on raw audio, we may get expert-grade tonic without
any dataset annotations — directly attacking the dominant bottleneck
(tonic detection capped the audio pipeline; sa_pa reached only 70.2%).

This script runs TonicIndianArtMusic on a 60s middle window of every
recording and scores octave-agnostically against .tonicFine. Compares to
the sa_pa exact scorer (70.2%) and old peakedness (51.9%).

Outputs: data/eval_essentia_tonic_report.txt
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_audio_ab import audio_path_for
from eval_harness import write_report

TOL = 25.0


def find_tonic_fine(recording_id: str):
    root = os.path.join(os.path.dirname(__file__), "..", "data",
                        "RagaDataset", "Carnatic", "features")
    rec_dir = os.path.join(root, recording_id)
    if not os.path.isdir(rec_dir):
        return None
    for name in os.listdir(rec_dir):
        if name.endswith(".tonicFine"):
            with open(os.path.join(rec_dir, name)) as f:
                return float(f.read().strip())
    return None


def matches(cand, expert):
    if cand <= 0 or expert <= 0:
        return False
    return abs(((1200 * math.log2(cand / expert) + 600) % 1200) - 600) <= TOL


def essentia_tonic_one(args):
    idx, recording_id, expert = args
    import warnings
    warnings.filterwarnings("ignore")
    import essentia.standard as es

    path = audio_path_for(recording_id)
    if not os.path.exists(path) or expert is None or expert <= 0:
        return {"idx": idx, "error": "missing"}
    try:
        audio = es.MonoLoader(filename=path, sampleRate=44100)()
        if len(audio) < 44100 * 10:
            chunk = audio
        else:
            mid = len(audio) // 2
            half = min(30 * 44100, mid)
            chunk = audio[mid - half: mid + half]
        tonic = float(es.TonicIndianArtMusic()(chunk))
    except Exception as exc:
        return {"idx": idx, "error": str(exc)[:80]}
    return {"idx": idx, "tonic": tonic, "correct": matches(tonic, expert)}


def main():
    with open("data/tdms_meta.json") as f:
        meta = json.load(f)
    jobs = []
    for i, m in enumerate(meta):
        jobs.append((i, m["recording_id"], find_tonic_fine(m["recording_id"])))

    print(f"Running Essentia TonicIndianArtMusic on {len(jobs)} recordings...")
    correct = 0
    n = 0
    errors = 0
    t0 = time.time()
    done = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
        for r in pool.map(essentia_tonic_one, jobs, chunksize=1):
            done += 1
            if "error" in r:
                errors += 1
            else:
                n += 1
                correct += int(r["correct"])
            if done % 50 == 0 or done == len(jobs):
                rate = done / max(1e-6, time.time() - t0)
                print(f"  [{done}/{len(jobs)}] {rate:.1f} rec/s  acc so far {100*correct/max(1,n):.1f}%")

    acc = 100 * correct / max(1, n)
    lines = [
        "=" * 70,
        "Phase 13: Essentia TonicIndianArtMusic tonic accuracy",
        f"60s middle window, octave-agnostic +/-25 cents, {n} recordings ({errors} errors)",
        "=" * 70,
        "",
        f"  Essentia TonicIndianArtMusic   {acc:.1f}%  ({correct}/{n})",
        "",
        "Comparison (same 480-recording set, octave-agnostic top-1):",
        "  Essentia tonic (this)          " + f"{acc:.1f}%",
        "  sa_pa exact, K=15 (shipped)    70.2%",
        "  peakedness, K=5 (old prod)     51.9%",
        "",
        "If Essentia tonic >> 70%, it replaces sa_pa as the tonic detector.",
        "Next: full pipeline with Essentia Melodia pitch + Essentia tonic.",
    ]
    write_report("data/eval_essentia_tonic_report.txt", lines)
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
