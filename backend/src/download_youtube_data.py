"""
Download YouTube audio for all 40 ragas and extract features for training.
Processes each video by downloading audio, extracting pitch features using
the same pipeline as predict.py, and saving to data/X_yt.npy + data/y_yt.npy.
"""

import subprocess
import json
import os
import sys
import tempfile
import numpy as np
import glob as globmod

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predict import extract_features_from_audio

VIDEOS_JSON = "/tmp/raga_youtube_videos.json"
CLASSES_JSON = os.path.join(os.path.dirname(__file__), "..", "data", "classes.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def download_and_extract(video_id, duration):
    """Download a segment from a YouTube video and extract features."""
    with tempfile.TemporaryDirectory() as tmpdir:
        url = f"https://www.youtube.com/watch?v={video_id}"
        output_template = os.path.join(tmpdir, "audio.%(ext)s")

        # For longer videos, grab a 90s segment from the middle
        dl_args = [
            "yt-dlp", "--no-playlist",
            "-x", "--audio-format", "wav",
            "-o", output_template,
        ]
        if duration > 180:
            start = max(60, duration // 3)
            dl_args += ["--download-sections", f"*{start}-{start + 90}"]

        dl_args.append(url)

        try:
            result = subprocess.run(
                dl_args, capture_output=True, text=True, timeout=120
            )
        except subprocess.TimeoutExpired:
            return None
        if result.returncode != 0:
            return None

        wav_files = globmod.glob(os.path.join(tmpdir, "audio.*"))
        if not wav_files:
            return None

        try:
            features = extract_features_from_audio(wav_files[0])
            return features
        except (ValueError, Exception) as e:
            return None


def main():
    with open(VIDEOS_JSON) as f:
        videos = json.load(f)
    with open(CLASSES_JSON) as f:
        classes = json.load(f)

    raga_to_label = {name: i for i, name in enumerate(classes)}

    X_all, y_all = [], []
    for raga, vids in videos.items():
        label = raga_to_label.get(raga)
        if label is None:
            print(f"  SKIP {raga}: not in classes.json")
            continue

        success = 0
        for vid in vids:
            vid_id = vid["id"]
            dur = vid["duration"]
            print(f"  [{raga}] {vid_id} ({dur}s) ... ", end="", flush=True)

            features = download_and_extract(vid_id, dur)
            if features is not None:
                X_all.append(features)
                y_all.append(label)
                success += 1
                print("OK")
            else:
                print("FAIL")

        print(f"  {raga}: {success}/{len(vids)} videos processed\n")

    X_yt = np.array(X_all)
    y_yt = np.array(y_all)
    print(f"\nTotal YouTube samples: {len(X_yt)}")
    print(f"Feature shape: {X_yt.shape}")

    np.save(os.path.join(OUTPUT_DIR, "X_yt.npy"), X_yt)
    np.save(os.path.join(OUTPUT_DIR, "y_yt.npy"), y_yt)
    print("Saved X_yt.npy, y_yt.npy")

    # Show per-raga counts
    from collections import Counter
    counts = Counter(y_yt)
    for label, count in sorted(counts.items()):
        print(f"  {classes[label]}: {count}")


if __name__ == "__main__":
    main()
