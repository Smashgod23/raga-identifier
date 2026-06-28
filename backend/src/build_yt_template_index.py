"""
Scale-up of the validated pilot: build TDMS templates for every YouTube
recording in data/youtube_videos.json and write them to a YouTube template
index that gets concatenated onto the CompMusic index at deploy time.

Templates use a long middle window + Essentia Melodia + Essentia tonic
(TonicIndianArtMusic), matching how the deployed query path detects tonic, so
novel YouTube queries have same-style references to match against.

Resumable: progress is saved after every recording to
data/yt_tdms_templates.npz (templates, labels, ids). Re-running skips ids
already present, so a flaky-network batch can be restarted freely.
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

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(DATA, "yt_tdms_templates.npz")
SR = 44100
WINDOW_SEC = 300  # 5-min middle window: better templates than 4-min, ~25MB each


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
    for attempt in range(3):  # flaky network: a few retries
        try:
            subprocess.run(args, check=True, capture_output=True, text=True, timeout=300)
            return True
        except subprocess.TimeoutExpired:
            continue
        except subprocess.CalledProcessError:
            continue
    return False


def build_template(vid):
    with tempfile.TemporaryDirectory() as td:
        if not _download_window(vid, os.path.join(td, "a.%(ext)s")):
            return None
        wavs = [f for f in os.listdir(td) if f.startswith("a.")]
        if not wavs:
            return None
        try:
            audio = es.MonoLoader(filename=os.path.join(td, wavs[0]), sampleRate=SR)()
            tonic = pe.detect_tonic(audio)
            tmpl = pe._melodia_tdms(audio, tonic)
        except Exception as exc:
            print(f"    essentia failed: {exc!r}")
            return None
    return np.asarray(tmpl, dtype=np.float32).ravel()


def load_progress():
    if os.path.exists(OUT):
        z = np.load(OUT, allow_pickle=True)
        return list(z["templates"]), list(z["labels"]), list(z["ids"])
    return [], [], []


def save_progress(templates, labels, ids):
    np.savez(OUT, templates=np.array(templates, dtype=np.float32),
             labels=np.array(labels, dtype=np.int64), ids=np.array(ids, dtype=object))


def main():
    classes = json.load(open(os.path.join(DATA, "classes.json")))
    cls_idx = {c: i for i, c in enumerate(classes)}
    videos = json.load(open(os.path.join(DATA, "youtube_videos.json")))

    templates, labels, ids = load_progress()
    done = set(ids)
    print(f"Resuming: {len(done)} templates already built.\n", flush=True)

    total = sum(len(v) for v in videos.values())
    seen = 0
    for raga, vids in videos.items():
        if raga not in cls_idx:
            continue
        for v in vids:
            seen += 1
            vid = v["id"]
            if vid in done:
                continue
            print(f"[{seen}/{total}] {raga} {vid} ...", flush=True)
            tmpl = build_template(vid)
            if tmpl is None:
                print("    skip", flush=True)
                continue
            templates.append(tmpl); labels.append(cls_idx[raga]); ids.append(vid)
            done.add(vid)
            save_progress(templates, labels, ids)  # checkpoint after every success
            print(f"    built ({len(templates)} total)", flush=True)

    print(f"\nDONE: {len(templates)} YouTube templates across {len(set(labels))} ragas -> {OUT}")


if __name__ == "__main__":
    main()
