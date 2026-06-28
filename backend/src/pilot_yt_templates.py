"""
Pilot: does expanding the TDMS template index with YouTube recordings fix the
out-of-distribution failures on novel YouTube audio?

For a handful of ragas that have several YouTube videos, download a middle
window of each, build (a) a full-window Melodia template with an Essentia tonic
and (b) a 3-window query TDMS. Then leave-one-out compare:
  - baseline:  YT query vs CMD-only template index
  - expanded:  YT query vs (CMD templates + the OTHER YT templates)
If the expanded index lifts top-1 a lot, the data expansion is worth scaling.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import essentia.standard as es
import predict_essentia as pe
from build_tdms import symmetric_kl_distance

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
SR = 44100
WINDOW_SEC = 240  # 4-min middle window per recording (keeps downloads ~20MB)

PILOT_RAGAS = ["Kāpi", "Bhairavi", "Madhyamāvati", "Bilahari", "Mōhanaṁ", "Sencuruṭṭi"]
PER_RAGA = 4


def _probe_duration(vid):
    try:
        r = subprocess.run([sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-warnings",
                            "--skip-download", "--print", "%(duration)s",
                            f"https://www.youtube.com/watch?v={vid}"],
                           capture_output=True, text=True, timeout=60)
        return int(r.stdout.strip().split("\n")[-1]) if r.returncode == 0 else 0
    except Exception:
        return 0


def _download_window(vid, dst):
    dur = _probe_duration(vid)
    start = max(60, dur // 3) if dur > WINDOW_SEC + 120 else 0
    args = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-warnings",
            "-x", "--audio-format", "wav",
            "--download-sections", f"*{start}-{start + WINDOW_SEC}",
            "-o", dst, f"https://www.youtube.com/watch?v={vid}"]
    for attempt in range(2):
        try:
            subprocess.run(args, check=True, capture_output=True, text=True, timeout=240)
            return True
        except subprocess.TimeoutExpired:
            if attempt == 1:
                return False
        except subprocess.CalledProcessError:
            return False
    return False


def build_for_video(vid):
    """Return (template_tdms, query_tdms, tonic) or None on failure."""
    with tempfile.TemporaryDirectory() as td:
        tmpl = os.path.join(td, "a.%(ext)s")
        if not _download_window(vid, tmpl):
            return None
        wavs = [f for f in os.listdir(td) if f.startswith("a.")]
        if not wavs:
            return None
        path = os.path.join(td, wavs[0])
        try:
            audio = es.MonoLoader(filename=path, sampleRate=SR)()
            tonic = pe.detect_tonic(audio)                       # Essentia tonic
            template = pe._melodia_tdms(audio, tonic)            # full-window Melodia template
            query, _ = pe.query_tdms_from_audio(audio, tonic_override=tonic)
        except Exception as exc:
            print(f"    essentia failed for {vid}: {exc!r}")
            return None
    return np.asarray(template, dtype=np.float64).ravel(), np.asarray(query, dtype=np.float64).ravel(), tonic


def topk_labels(q, T, y, k=5, exclude=None):
    d = np.array([symmetric_kl_distance(q, T[i]) for i in range(len(T))])
    if exclude is not None:
        d[exclude] = np.inf
    order = np.argsort(d, kind="stable")
    seen = []
    for j in order:
        c = int(y[j])
        if c not in seen:
            seen.append(c)
        if len(seen) >= k:
            break
    return seen


def main():
    classes = json.load(open(os.path.join(DATA, "classes.json")))
    cls_idx = {c: i for i, c in enumerate(classes)}
    cmd_T = np.load(os.path.join(DATA, "X_tdms_essfull_template.npy")).astype(np.float64)
    cmd_y = np.load(os.path.join(DATA, "y_tdms.npy"))
    videos = json.load(open(os.path.join(DATA, "youtube_videos.json")))

    yt_templates, yt_queries, yt_labels, yt_ids = [], [], [], []
    for raga in PILOT_RAGAS:
        for v in videos.get(raga, [])[:PER_RAGA]:
            vid = v["id"]
            print(f"[{raga}] {vid} ...", flush=True)
            out = build_for_video(vid)
            if out is None:
                print("    skip (download/essentia failed)")
                continue
            tmpl, query, tonic = out
            yt_templates.append(tmpl); yt_queries.append(query)
            yt_labels.append(cls_idx[raga]); yt_ids.append(vid)
            print(f"    built (tonic {tonic:.1f} Hz)")

    n = len(yt_templates)
    print(f"\nBuilt {n} YouTube recordings across {len(set(yt_labels))} ragas\n")
    if n < 4:
        print("too few to evaluate"); return
    yt_T = np.stack(yt_templates); yt_q = np.stack(yt_queries); yt_y = np.array(yt_labels)
    np.save(os.path.join(DATA, "pilot_yt_templates.npy"), yt_T)
    np.save(os.path.join(DATA, "pilot_yt_queries.npy"), yt_q)
    np.save(os.path.join(DATA, "pilot_yt_labels.npy"), yt_y)

    # Baseline: YT query vs CMD-only index
    base_t1 = base_t5 = 0
    # Expanded: YT query vs CMD + other YT templates (leave-one-out on the YT side)
    exp_t1 = exp_t5 = 0
    comb_T = np.concatenate([cmd_T, yt_T], axis=0)
    comb_y = np.concatenate([cmd_y, yt_y], axis=0)
    for i in range(n):
        true = yt_y[i]
        b = topk_labels(yt_q[i], cmd_T, cmd_y, k=5)
        base_t1 += (b[0] == true); base_t5 += (true in b[:5])
        # exclude this recording's own template (index len(cmd_T)+i)
        e = topk_labels(yt_q[i], comb_T, comb_y, k=5, exclude=len(cmd_T) + i)
        exp_t1 += (e[0] == true); exp_t5 += (true in e[:5])

    print("=== PILOT RESULT (YouTube queries, leave-one-out) ===")
    print(f"  CMD-only index ({len(cmd_T)} tmpl):       top-1 {base_t1/n*100:5.1f}%   top-5 {base_t5/n*100:5.1f}%")
    print(f"  CMD + YouTube index ({len(comb_T)} tmpl): top-1 {exp_t1/n*100:5.1f}%   top-5 {exp_t5/n*100:5.1f}%")
    print(f"  n = {n} YouTube recordings")


if __name__ == "__main__":
    main()
