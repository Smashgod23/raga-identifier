"""
Extract Essentia pitch contours for the YouTube recordings so they can go into
DeepSRGM TRAINING (not just TDMS templates). The whole point: the sequence model
only ever trained on clean CompMusic studio audio; adding real YouTube/concert
recordings teaches it the real-world distribution it currently fails on. A trained
net tolerates the auto-label noise far better than the 1-NN TDMS did.

Same tokenization as build_essentia_contours.py (so CMD + YouTube contours share a
representation). Resumable: checkpoints every 20 recordings, skips already-done ids.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import essentia.standard as es
import predict_essentia as pe

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(DATA, "deepsrgm_youtube_seqs.npz")
BIN, LO, HI = 50.0, -1200, 2400
DS = max(1, round((pe.SR / pe.MELODIA_HOP) / 22.5))
WINDOW_SEC = 300   # 5-min middle window per recording


def tokenize(pitch, tonic):
    voiced = pitch[pitch > 0]
    if tonic <= 0 or len(voiced) < 200:
        return None
    cents = 1200.0 * np.log2(voiced / tonic)
    return np.round((np.clip(cents, LO, HI) - LO) / BIN).astype(np.int64)[::DS].astype(np.int16)


def _download(vid, dst):
    try:
        r = subprocess.run([sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-warnings", "--skip-download",
                            "--print", "%(duration)s", f"https://www.youtube.com/watch?v={vid}"],
                           capture_output=True, text=True, timeout=60)
        dur = int(r.stdout.strip().split("\n")[-1]) if r.returncode == 0 else 0
    except Exception:
        dur = 0
    start = max(60, dur // 3) if dur > WINDOW_SEC + 120 else 0
    args = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-warnings", "-x", "--audio-format", "wav",
            "--download-sections", f"*{start}-{start + WINDOW_SEC}", "-o", dst, f"https://www.youtube.com/watch?v={vid}"]
    for attempt in range(2):
        try:
            subprocess.run(args, check=True, capture_output=True, text=True, timeout=300); return True
        except subprocess.TimeoutExpired:
            if attempt == 1:
                return False
        except subprocess.CalledProcessError:
            return False
    return False


def contour_for(vid):
    with tempfile.TemporaryDirectory() as td:
        dst = os.path.join(td, "a.%(ext)s")
        if not _download(vid, dst):
            return None
        files = [f for f in os.listdir(td) if f.startswith("a.")]
        if not files:
            return None
        audio = es.MonoLoader(filename=os.path.join(td, files[0]), sampleRate=pe.SR)()
        tonic = pe.detect_tonic(audio)
        pitch, _ = es.PredominantPitchMelodia(frameSize=pe.MELODIA_FRAME, hopSize=pe.MELODIA_HOP)(audio)
    return tokenize(np.asarray(pitch, dtype=np.float64), tonic)


def save(seqs, labels, ids):
    np.savez_compressed(OUT, **{f"s{i}": s for i, s in enumerate(seqs)},
                        labels=np.array(labels), ids=np.array(ids), n=len(seqs))


def main():
    classes = json.load(open(os.path.join(DATA, "classes.json")))
    name_to_label = {n: i for i, n in enumerate(classes)}
    videos = json.load(open(os.path.join(DATA, "youtube_videos.json")))

    done = set()
    seqs, labels, ids = [], [], []
    if os.path.exists(OUT):
        z = np.load(OUT, allow_pickle=True); k = int(z["n"])
        for i in range(k):
            seqs.append(z[f"s{i}"]);
        labels = list(z["labels"]); ids = list(z["ids"]); done = set(ids)
        print(f"resuming: {len(done)} already done", flush=True)

    ok = len(seqs)
    total = sum(len(v) for v in videos.values())
    i = 0
    for raga, vids in videos.items():
        label = name_to_label.get(raga)
        if label is None:
            continue
        for v in vids:
            i += 1; vid = v["id"]
            if vid in done:
                continue
            toks = None
            try:
                toks = contour_for(vid)
            except Exception as exc:
                print(f"[{i}/{total}] {raga} {vid} fail: {exc!r}", flush=True)
            if toks is None or len(toks) < 200:
                print(f"[{i}/{total}] {raga} {vid} no contour", flush=True); continue
            seqs.append(toks); labels.append(label); ids.append(vid); ok += 1
            print(f"[{i}/{total}] {raga} {vid}: {len(toks)} tokens (ok {ok})", flush=True)
            if ok % 20 == 0:
                save(seqs, labels, ids); print(f"  checkpoint {ok}", flush=True)
    save(seqs, labels, ids)
    print(f"\nDONE: {ok} YouTube contours -> {OUT}")


if __name__ == "__main__":
    main()
