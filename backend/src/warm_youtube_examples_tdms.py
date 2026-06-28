"""Re-seed data/youtube_examples.json with TDMS/Essentia predictions.

The /predict-youtube endpoint now runs the TDMS model, so the cached examples
must be TDMS predictions too. Essentia has no macOS wheel, so we cannot run TDMS
locally. Instead, download each example's analysis window from a residential IP
(where yt-dlp works) and POST it to the LIVE /predict-tdms endpoint on the Space
(where Essentia runs), then keep only the videos that predict their correct raga.
This matches what a fresh /predict-youtube fetch produces (same window, same
single-window TDMS query).

Run from a home network:

    cd backend && venv/bin/python src/warm_youtube_examples_tdms.py
"""

import glob
import json
import os
import subprocess
import sys
import tempfile

import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from warm_youtube_examples import EXAMPLES, SEGMENT_LEN, _probe_duration, _window_start

API = "https://smashgod23-raga-identifier-api.hf.space"
OUT = os.path.join(BASE_DIR, "data", "youtube_examples.json")


def _download_window(vid, dest_template):
    start = _window_start(_probe_duration(vid))
    args = [
        sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-warnings",
        "-x", "--audio-format", "wav",
        "--download-sections", f"*{start}-{start + SEGMENT_LEN}",
        "-o", dest_template, f"https://www.youtube.com/watch?v={vid}",
    ]
    for attempt in range(2):
        try:
            subprocess.run(args, check=True, capture_output=True, text=True, timeout=240)
            return
        except subprocess.TimeoutExpired:
            if attempt == 1:
                raise


def _predict_tdms_live(wav_path):
    """POST the clip to the live /predict-tdms (Essentia runs on the Space)."""
    for attempt in range(4):
        try:
            with open(wav_path, "rb") as f:
                r = requests.post(API + "/predict-tdms",
                                  files={"file": ("clip.wav", f, "audio/wav")}, timeout=120)
            if r.status_code == 200:
                return r.json()
            print(f"    /predict-tdms HTTP {r.status_code} (attempt {attempt + 1})")
        except Exception as exc:
            print(f"    upload error (attempt {attempt + 1}): {str(exc)[-80:]}")
    return None


def main():
    results = {}
    for ex in EXAMPLES:
        vid = ex["id"]
        print(f"[{vid}] {ex['title']}", flush=True)
        with tempfile.TemporaryDirectory() as td:
            template = os.path.join(td, "a.%(ext)s")
            try:
                _download_window(vid, template)
            except Exception as exc:
                print(f"    download failed: {str(exc)[-120:]}")
                continue
            wavs = glob.glob(os.path.join(td, "a.*"))
            if not wavs:
                print("    no file downloaded")
                continue
            payload = _predict_tdms_live(wavs[0])
        if not payload:
            print("    no TDMS payload")
            continue
        ok = payload["top_raga"] == ex["raga"]
        print(f"    -> {payload['top_raga']} ({payload['confidence']}%)  [{'OK' if ok else 'wrong — skip'}]")
        if ok:
            results[vid] = {
                "url": f"https://www.youtube.com/watch?v={vid}",
                "title": ex["title"],
                "expected_raga": ex["raga"],
                "payload": payload,
            }
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(results)} TDMS examples to {OUT}")


if __name__ == "__main__":
    main()
