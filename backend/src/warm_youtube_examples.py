"""Precompute raga predictions for a curated set of YouTube example videos.

The live /predict-youtube endpoint runs on a datacenter IP that YouTube
throttles, so an open "paste any link" box fails intermittently and looks
broken to a first-time visitor. This script runs the EXACT inference pipeline
from a residential IP (where yt-dlp works fine), once, offline, and writes the
results to backend/data/youtube_examples.json. The backend loads that file at
startup and seeds its in-memory cache, so the curated examples always return
instantly and correctly regardless of the server's IP reputation.

Run it from a home/residential network whenever you want to refresh or change
the example set:

    cd backend && venv/bin/python src/warm_youtube_examples.py

It is offline tooling, not part of the deployed image. Predictions are
deterministic for a given (model, audio window), and the model + sklearn
version here match the Space, so the committed results equal what the server
would have produced.
"""

import json
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np

# Reuse the real feature pipeline so precomputed results match the server.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from predict import extract_features_from_audio, hz_to_note_name  # noqa: E402

from huggingface_hub import hf_hub_download  # noqa: E402

REPO_ID = "Smashgod23/raga-identifier"
OUT_PATH = os.path.join(BASE_DIR, "data", "youtube_examples.json")

# Curated alapana (pure raga exposition) recordings: clean raga content, no
# percussion or lyrics to confuse the model, recognizable artists. Video ids
# are drawn from the existing training index (data/youtube_videos.json). The
# `raga` label is the expected answer, used only to sanity-check each result.
# More candidates than needed: any that the live pipeline mispredicts on its
# first 60s are skipped at seed time, so the shipped set is whatever actually
# works. All are first-listed videos for their raga in data/youtube_videos.json.
EXAMPLES = [
    {"id": "Ol51u5A6yK4", "raga": "Kalyāṇi",          "title": "Amrutha Venkatesh - Kalyani Alapana"},
    {"id": "0a91szM1Ivw", "raga": "Bhairavi",         "title": "Raga Bhairavi in Carnatic Music"},
    {"id": "uFO3whwMaX8", "raga": "Tōḍi",             "title": "MS Subbulakshmi - Todi Alapana"},
    {"id": "t3KSXVci5o4", "raga": "Kāpi",             "title": "Amrutha Venkatesh - Kapi Alapana"},
    {"id": "kl9fIlp2Xtk", "raga": "Harikāmbhōji",     "title": "Harikambhoji Raga Alapana"},
    {"id": "o_hcBabLYUU", "raga": "Madhyamāvati",     "title": "Nedunuri Krishnamurthy - Madhyamavati"},
    {"id": "nIfA-bcwFFI", "raga": "Mōhanaṁ",          "title": "Dr L Subramaniam - Mohanam"},
    {"id": "hECvvLVgDIU", "raga": "Śankarābharaṇaṁ",  "title": "MS Subbulakshmi - Shankarabharanam"},
    {"id": "ZdCNoF1pQCk", "raga": "Sāvēri",           "title": "TM Krishna - Saveri Alapana"},
    {"id": "IapzpEvpob8", "raga": "Bilahari",         "title": "Ranjani Gayatri - Bilahari"},
    {"id": "ciRoR0CF0FI", "raga": "Sindhubhairavi",   "title": "Sindhu Bhairavi - Venkatachala Nilayam"},
    {"id": "rlqEFjTxvZI", "raga": "Ānandabhairavi",   "title": "Sandeep Narayan - Anandabhairavi"},
]

# Mirror the live /predict-youtube path exactly: it downloads the first
# SEGMENT_LEN seconds and runs a single-window extraction (which loads the first
# 60s, the training window). Matching it means the cached example equals what a
# real fetch would produce for that video, so the examples are honest, not a
# different code path that happens to look better.
SEGMENT_LEN = 90


def _load_artifacts():
    model_path = hf_hub_download(repo_id=REPO_ID, filename="raga_sklearn.pkl", local_dir=os.path.join(BASE_DIR, "models"))
    scaler_path = hf_hub_download(repo_id=REPO_ID, filename="scaler.pkl", local_dir=os.path.join(BASE_DIR, "models"))
    classes_path = hf_hub_download(repo_id=REPO_ID, filename="classes.json", local_dir=os.path.join(BASE_DIR, "data"))
    with open(classes_path) as f:
        classes = json.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model, scaler, classes


def _predict_single(model, scaler, audio_path):
    """Single-window inference, identical to api.main._do_youtube_fetch: one
    extract_features_from_audio call (which loads the first 60s training window)
    on the downloaded clip. Returns (probs, tonic_hz)."""
    features, tonic = extract_features_from_audio(audio_path)
    probs = model.predict_proba(scaler.transform([features]))[0]
    return probs, tonic


def _format_response(classes, probs, tonic_hz):
    """Port of api.main._format_response (tonic never overridden for examples)."""
    top5_idx = np.argsort(probs)[::-1][:5]
    predictions = [
        {"raga": classes[i], "confidence": round(float(probs[i]) * 100, 1)}
        for i in top5_idx
    ]
    return {
        "top_raga": predictions[0]["raga"],
        "confidence": predictions[0]["confidence"],
        "predictions": predictions,
        "tonic_hz": round(float(tonic_hz), 2) if tonic_hz else None,
        "tonic_note": hz_to_note_name(tonic_hz) if tonic_hz else "",
        "tonic_overridden": False,
    }


def _probe_duration(video_id):
    """Metadata-only duration probe (no media download)."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        r = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-warnings",
             "--skip-download", "--print", "%(duration)s", url],
            capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            return int(r.stdout.strip().split("\n")[-1])
    except Exception:
        pass
    return 0


def _window_start(duration):
    """Sample a SEGMENT_LEN window 1/3 into the recording, the same window the
    model was trained on (download_youtube_data.py). The opening of a concert
    (tuning, sparse alapana, applause) is not characteristic of the raga, so the
    first-90s window the endpoint used to grab predicts poorly; a middle window
    matches training and is far more accurate."""
    return max(60, duration // 3) if duration and duration > 180 else 0


def _download(video_id, dest_template):
    """Download a representative SEGMENT_LEN window as wav, matching both the
    training window and the (now middle-window) live endpoint. Residential IP,
    so no EJS/cookies needed. Retries once on the flaky-network timeout."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    start = _window_start(_probe_duration(video_id))
    args = [
        sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-warnings",
        "-x", "--audio-format", "wav",
        "--download-sections", f"*{start}-{start + SEGMENT_LEN}",
        "-o", dest_template, url,
    ]
    for attempt in range(2):
        try:
            subprocess.run(args, check=True, capture_output=True, text=True, timeout=240)
            return
        except subprocess.TimeoutExpired:
            if attempt == 1:
                raise


def main():
    model, scaler, classes = _load_artifacts()
    print(f"Loaded model with {len(classes)} ragas\n")

    results = {}
    for ex in EXAMPLES:
        vid = ex["id"]
        print(f"[{vid}] {ex['title']}")
        with tempfile.TemporaryDirectory() as tmp:
            template = os.path.join(tmp, "audio.%(ext)s")
            try:
                _download(vid, template)
            except subprocess.CalledProcessError as e:
                print(f"    download FAILED: {(e.stderr or '')[-200:]}\n")
                continue
            except subprocess.TimeoutExpired:
                print("    download FAILED: timed out\n")
                continue
            wavs = [f for f in os.listdir(tmp) if f.startswith("audio.")]
            if not wavs:
                print("    download produced no file\n")
                continue
            probs, tonic = _predict_single(model, scaler, os.path.join(tmp, wavs[0]))
            payload = _format_response(classes, probs, tonic)

        correct = payload["top_raga"] == ex["raga"]
        flag = "OK" if correct else f"!! expected {ex['raga']} — SKIPPED"
        print(f"    -> {payload['top_raga']} ({payload['confidence']}%)  Sa={payload['tonic_note']}  [{flag}]")
        # Only ship examples the live pipeline actually gets right, so a visitor
        # never sees a curated example labeled with the wrong raga.
        if not correct:
            print()
            continue
        results[vid] = {
            "url": f"https://www.youtube.com/watch?v={vid}",
            "title": ex["title"],
            "expected_raga": ex["raga"],
            "payload": payload,
        }
        print()

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(results)} examples to {OUT_PATH}")


if __name__ == "__main__":
    main()
