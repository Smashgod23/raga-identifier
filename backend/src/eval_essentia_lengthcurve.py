"""
Phase 13f: accuracy vs upload length — the honest short-clip floor.

The 75.62% headline used 7x90s = 630s of query audio. Real users upload
short clips. This measures top-1/top-5 as a function of upload length by
simulating an L-second upload: extract the middle L seconds of each
recording and run the EXACT production query path
(predict_essentia.query_tdms_from_audio) on it — same tonic detection,
same windowing, same averaging the deployed /predict-tdms uses.

For L <= 90s the production path uses a single window covering the whole
clip; for longer L it uses up to 7 windows within the clip. So this curve
is what a user actually gets for a clip of length L.

Queries run against the cached full-recording Essentia templates
(X_tdms_essfull_template.npy). Lengths: 30, 60, 90, 120, 180, 300s.

Output: data/eval_essentia_lengthcurve_report.txt
"""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_audio_ab import audio_path_for
from eval_harness import load_classes, write_report
from eval_multiwindow_queries import asymmetric_cv

SR = 44100
LENGTHS = [30, 60, 90, 120, 180, 300]


def extract_one(args):
    idx, recording_id, length_s = args
    import warnings; warnings.filterwarnings("ignore")
    import essentia.standard as es
    import librosa
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import predict_essentia

    path = audio_path_for(recording_id)
    if not os.path.exists(path):
        return {"idx": idx, "error": "missing"}
    try:
        total = float(librosa.get_duration(path=path))
        lo = max(0.0, (total - length_s) / 2.0)
        hi = min(total, lo + length_s)
        seg = es.EasyLoader(filename=path, sampleRate=SR, startTime=lo, endTime=hi)()
        row, _ = predict_essentia.query_tdms_from_audio(seg)
        return {"idx": idx, "tdms": row}
    except ValueError:
        return {"idx": idx, "error": "no-usable"}
    except Exception as exc:
        return {"idx": idx, "error": str(exc)[:80]}


def main():
    classes = load_classes(); n_classes = len(classes)
    with open("data/tdms_meta.json") as f:
        meta = json.load(f)
    y_full = np.load("data/y_tdms.npy")
    X_t = np.load("data/X_tdms_essfull_template.npy")

    lines = [
        "=" * 72,
        "Phase 13f: accuracy vs simulated upload length (production query path)",
        "Templates: full-recording Melodia + expert tonic. Query: middle L seconds.",
        "=" * 72,
        "",
        f"  {'upload length':16s} {'top-1':10s} {'top-5':10s} {'usable':8s}",
        f"  {'-'*16} {'-'*10} {'-'*10} {'-'*8}",
    ]

    for length_s in LENGTHS:
        jobs = [(i, m["recording_id"], length_s) for i, m in enumerate(meta)]
        X_q = np.zeros((len(jobs), 14400), dtype=np.float32)
        t0 = time.time(); done = usable = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
            for r in pool.map(extract_one, jobs, chunksize=1):
                done += 1
                if "error" not in r:
                    X_q[r["idx"]] = r["tdms"]; usable += 1
        keep = (~np.all(X_t == 0, axis=1)) & (~np.all(X_q == 0, axis=1))
        res = asymmetric_cv(X_t[keep], X_q[keep], y_full[keep], n_classes)
        m, s = res["top_k_mean"], res["top_k_std"]
        lines.append(f"  {length_s:>4d}s ({length_s//60}m{length_s%60:02d}s)     "
                     f"{m[0]*100:5.1f}%     {m[1]*100:5.1f}%     {usable}/{len(jobs)}")
        print(f"L={length_s}s: top1 {m[0]*100:.1f}% top5 {m[1]*100:.1f}% "
              f"({usable} usable, {time.time()-t0:.0f}s)")

    lines += [
        "",
        "Reference (7x90s = 630s of query audio): top1 75.62%  top5 91.17%",
        "v1 deployed: top1 8.32%  top5 28.57%",
        "",
        "This is what a user gets for a single clip of the given length, using",
        "the exact /predict-tdms query path. Short clips use one window; the",
        "curve shows how accuracy grows with more audio.",
    ]
    write_report("data/eval_essentia_lengthcurve_report.txt", lines)
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
