"""
Phase 11b: Phase 11a + tonic re-ranker.

50-recording A/B test of the re-ranker showed no consistent lift on
this offset, but the only honest test is to plug it into the full
multi-window audio pipeline and re-evaluate end-to-end. This script
does exactly that.

Setup:
  Templates: 5-min pyin + EXPERT tonic (reuses X_tdms_long.npy from
             Phase 10a — expert tonic for templates is realistic since
             they're built offline from .tonicFine annotations).
  Queries:   Three 60s windows averaged, pyin + heuristic candidates +
             RE-RANKER pick.
  Classifier: 1-NN with symmetric KL.
  Eval: 5-fold stratified CV, 5 seeds, recording-aware.

If this beats Phase 11a's 35.83% top-1 / 63.67% top-5, we wire the
re-ranker. If not, we keep the heuristic-only path and move on to
CREPE / bidirectional octave fold.

Outputs:
  data/X_tdms_audio_multiwindow_reranker.npy
  data/eval_multiwindow_reranker_report.txt
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from build_tdms import compute_tdms
from eval_audio_ab import audio_path_for
from eval_harness import load_classes, write_report
from eval_multiwindow_queries import asymmetric_cv

QUERY_DURATION = 60.0
WINDOW_FRACTIONS = (0.25, 0.50, 0.75)
RERANKER_NPZ = "models/tonic_detector_v1.npz"


class _LocalTonicReRanker:
    """Self-contained copy of the trained tonic-detector forward pass.

    Phase 11b found the re-ranker regresses end-to-end so the wiring in
    predict.py was reverted before merge. This class lives here so the
    eval script remains self-reproducible against HEAD; merging it into
    predict.py is deferred until the re-ranker is retrained on
    production-offset candidates (see Phase 11b notes in the README).
    """

    def __init__(self, npz_path: str):
        z = np.load(npz_path)
        self.fc0_W = z["fc0_W"]; self.fc0_b = z["fc0_b"]
        self.bn1_gamma = z["bn1_gamma"]; self.bn1_beta = z["bn1_beta"]
        self.bn1_mean = z["bn1_mean"]; self.bn1_var = z["bn1_var"]
        self.fc4_W = z["fc4_W"]; self.fc4_b = z["fc4_b"]
        self.bn5_gamma = z["bn5_gamma"]; self.bn5_beta = z["bn5_beta"]
        self.bn5_mean = z["bn5_mean"]; self.bn5_var = z["bn5_var"]
        self.fc8_W = z["fc8_W"]; self.fc8_b = z["fc8_b"]
        self.eps = float(z["bn_eps"])

    @staticmethod
    def _bn(x, gamma, beta, mean, var, eps):
        return (x - mean) / np.sqrt(var + eps) * gamma + beta

    def score(self, features: np.ndarray) -> np.ndarray:
        x = features @ self.fc0_W.T + self.fc0_b
        x = self._bn(x, self.bn1_gamma, self.bn1_beta, self.bn1_mean, self.bn1_var, self.eps)
        x = np.maximum(x, 0.0)
        x = x @ self.fc4_W.T + self.fc4_b
        x = self._bn(x, self.bn5_gamma, self.bn5_beta, self.bn5_mean, self.bn5_var, self.eps)
        x = np.maximum(x, 0.0)
        x = x @ self.fc8_W.T + self.fc8_b
        return x.squeeze(-1)


def _detect_tonic_with_reranker(voiced, reranker: "_LocalTonicReRanker") -> float | None:
    """Mirrors predict._detect_tonic's candidate generation, then scores the
    5 candidates with the re-ranker instead of the peakedness heuristic."""
    from scipy.ndimage import uniform_filter1d

    folded = voiced.copy()
    while np.any(folded > 120):
        folded = np.where(folded > 120, folded / 2, folded)
    while np.any(folded < 60):
        folded = np.where(folded < 60, folded * 2, folded)
    hist, bin_edges = np.histogram(folded, bins=200, range=(60, 120))
    smoothed = uniform_filter1d(hist.astype(float), size=5)
    median_pitch = float(np.median(voiced))

    candidate_indices = np.argsort(smoothed)[::-1][:5]
    feats, cand_hzs = [], []
    for idx in candidate_indices:
        if smoothed[idx] == 0:
            continue
        cand = (bin_edges[idx] + bin_edges[idx + 1]) / 2
        while cand * 2 < median_pitch:
            cand *= 2
        cents = 1200.0 * np.log2(voiced / cand)
        cents_mod = cents % 1200
        h_raw, _ = np.histogram(cents_mod, bins=120, range=(0, 1200))
        peakedness = float(np.sum(h_raw ** 2))
        sa_mask = (np.abs(cents_mod) < 25.0) | (np.abs(cents_mod - 1200) < 25.0)
        sa_fraction = float(sa_mask.mean())
        hist_count = float(smoothed[idx])
        pcd, _ = np.histogram(cents_mod, bins=120, range=(0, 1200), density=True)
        feat = np.concatenate([
            np.array([
                np.log(max(cand, 1e-3)),
                peakedness,
                np.log(max(hist_count, 1)),
                sa_fraction,
            ], dtype=np.float32),
            pcd.astype(np.float32),
        ])
        feats.append(feat)
        cand_hzs.append(cand)

    if not feats:
        return None
    logits = reranker.score(np.stack(feats))
    return float(cand_hzs[int(np.argmax(logits))])


def extract_one(args: tuple) -> dict:
    """Same as eval_multiwindow_heuristic.extract_one but with the re-ranker."""
    idx, recording_id, label = args
    import librosa

    # Each worker loads the re-ranker once. Small numpy bundle (~45KB).
    reranker = _LocalTonicReRanker(RERANKER_NPZ)

    path = audio_path_for(recording_id)
    if not os.path.exists(path):
        return {"idx": idx, "error": f"missing audio: {path}"}

    try:
        total = librosa.get_duration(path=path)
    except Exception as exc:
        return {"idx": idx, "error": f"duration probe failed: {exc}"}

    tdmss: list[np.ndarray] = []
    for frac in WINDOW_FRACTIONS:
        center = total * frac
        offset = max(0.0, min(center - QUERY_DURATION / 2.0, total - QUERY_DURATION))
        duration = min(QUERY_DURATION, total)

        try:
            y, sr = librosa.load(path, sr=16000, mono=True, offset=offset, duration=duration)
        except Exception:
            continue

        rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
        if rms > 1e-6:
            y = y * (0.1 / rms)

        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=60, fmax=800, sr=sr, frame_length=1024, hop_length=256
        )
        confident = voiced_flag & (voiced_probs > 0.3)
        voiced = f0[confident]
        if len(voiced) < 30:
            voiced = f0[voiced_flag]
        voiced = voiced[~np.isnan(voiced)]
        if len(voiced) < 30:
            continue

        tonic = _detect_tonic_with_reranker(voiced, reranker)
        if tonic is None or tonic <= 0:
            continue

        hop_seconds = 256.0 / sr
        pitches_full = np.where(voiced_flag, f0, 0.0)
        pitches_full = np.where(np.isfinite(pitches_full), pitches_full, 0.0)
        timestamps = np.arange(len(pitches_full)) * hop_seconds

        tdms = compute_tdms(timestamps, pitches_full, tonic)
        if tdms is not None:
            tdmss.append(tdms.ravel())

    if not tdmss:
        return {"idx": idx, "error": "no usable windows"}

    avg = np.mean(np.stack(tdmss, axis=0), axis=0)
    avg /= max(float(avg.sum()), 1e-12)
    return {"idx": idx, "label": label, "tdms": avg.astype(np.float32)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    classes = load_classes()
    n_classes = len(classes)
    with open("data/tdms_meta.json") as f:
        meta = json.load(f)
    y_full = np.load("data/y_tdms.npy")
    assert len(meta) == len(y_full) == 480

    if args.skip_extract and os.path.exists("data/X_tdms_audio_multiwindow_reranker.npy"):
        X_query = np.load("data/X_tdms_audio_multiwindow_reranker.npy")
    else:
        jobs = [(i, m["recording_id"], int(y_full[i])) for i, m in enumerate(meta)]
        print(f"Extracting 3x60s queries with RE-RANKER tonic, {args.workers} workers")
        X_query = np.zeros((len(jobs), 14400), dtype=np.float32)
        t0 = time.time()
        done = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for r in pool.map(extract_one, jobs, chunksize=1):
                done += 1
                if "error" in r:
                    if done <= 5 or done % 50 == 0:
                        print(f"  [{done}/{len(jobs)}] ERROR idx={r['idx']}: {r['error']}")
                    continue
                X_query[r["idx"]] = r["tdms"]
                if done % 25 == 0 or done == len(jobs):
                    rate = done / max(1e-6, time.time() - t0)
                    eta = (len(jobs) - done) / max(1e-6, rate)
                    print(f"  [{done}/{len(jobs)}] {rate:.2f} rec/s   ETA {eta / 60:.1f} min")
        np.save("data/X_tdms_audio_multiwindow_reranker.npy", X_query)

    X_template = np.load("data/X_tdms_long.npy")
    keep = (~np.all(X_template == 0, axis=1)) & (~np.all(X_query == 0, axis=1))
    X_template = X_template[keep]
    X_query = X_query[keep]
    y = y_full[keep]
    print(f"\nKept {int(keep.sum())}/480")

    print("\nRunning asymmetric 5-fold CV...")
    res = asymmetric_cv(X_template, X_query, y, n_classes)
    m, s = res["top_k_mean"], res["top_k_std"]

    lines = [
        "=" * 78,
        "Phase 11b: production-faithful eval with tonic RE-RANKER",
        f"Templates: 5-min pyin + EXPERT tonic     Queries: 3 x 60s avg, pyin + RE-RANKER tonic",
        f"Recordings: {int(keep.sum())}/480",
        "=" * 78,
        "",
        f"  Phase 11b (re-ranker tonic on queries)  top1 {m[0]*100:5.2f}% +/- {s[0]*100:.2f}   top5 {m[1]*100:5.2f}% +/- {s[1]*100:.2f}",
        "",
        "Comparison ladder:",
        "  Phase 10b  pyin + EXPERT      tonic, 5min + 3x60s avg   top1 73.23%   top5 95.31%",
        f"  Phase 11b  pyin + RE-RANKER   tonic, 5min + 3x60s avg   top1 {m[0]*100:5.2f}%   top5 {m[1]*100:5.2f}%",
        "  Phase 11a  pyin + heuristic   tonic, 5min + 3x60s avg   top1 35.83%   top5 63.67%",
        "  Phase 8    pyin + heuristic   tonic, 60s + 60s            top1 37.36%   top5 67.91%",
        "  v1 deployed                                                top1  8.32%   top5 28.57%",
        "",
        "Phase 11b vs Phase 11a tells us whether wiring the re-ranker is worth shipping.",
    ]
    write_report("data/eval_multiwindow_reranker_report.txt", lines)
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
