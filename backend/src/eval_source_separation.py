"""
Experiment: does Demucs source separation (isolate the lead voice from tanpura,
percussion, and audience noise) improve TDMS raga ID on novel YouTube audio?

For a set of YouTube recordings, build a TDMS query two ways and classify each
against the CompMusic-only index:
  - RAW:        Essentia tonic + Melodia on the downloaded audio.
  - SEPARATED:  Demucs vocals stem, then Essentia tonic + Melodia on it.
Report top-1 / top-5 for both. If separation lifts accuracy, it's the lever to
pursue for real-world (concert/phone) audio quality.
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
WINDOW_SEC = 180
PILOT_RAGAS = ["Kāpi", "Bhairavi", "Madhyamāvati", "Bilahari", "Mōhanaṁ", "Sencuruṭṭi"]
PER_RAGA = 2

_model = None
_device = None


def get_model_cached():
    global _model, _device
    if _model is None:
        import torch
        from demucs.pretrained import get_model
        _device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"  loading demucs htdemucs (device={_device})...", flush=True)
        _model = get_model("htdemucs")
        _model.eval()
    return _model


def _probe(vid):
    try:
        r = subprocess.run([sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-warnings",
                            "--skip-download", "--print", "%(duration)s",
                            f"https://www.youtube.com/watch?v={vid}"],
                           capture_output=True, text=True, timeout=60)
        return int(r.stdout.strip().split("\n")[-1]) if r.returncode == 0 else 0
    except Exception:
        return 0


def _download(vid, dst):
    dur = _probe(vid)
    start = max(60, dur // 3) if dur > WINDOW_SEC + 120 else 0
    args = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-warnings", "-x",
            "--audio-format", "wav", "--download-sections", f"*{start}-{start + WINDOW_SEC}",
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


def separated_vocals(path):
    """Return Demucs vocals stem as a mono float32 numpy array at model SR (44.1k)."""
    import torch
    from demucs.apply import apply_model
    from demucs.audio import AudioFile
    model = get_model_cached()
    wav = AudioFile(path).read(streams=0, samplerate=model.samplerate,
                               channels=model.audio_channels)  # [channels, samples]
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        sources = apply_model(model, wav[None], device=_device, progress=False)[0]
    sources = sources * ref.std() + ref.mean()
    vocals = sources[model.sources.index("vocals")]  # [channels, samples] @ model.samplerate
    return vocals.mean(0).cpu().numpy().astype(np.float32)


def query_for(audio):
    tonic = pe.detect_tonic(audio)
    q, _ = pe.query_tdms_from_audio(audio, tonic_override=tonic)
    return np.asarray(q, dtype=np.float64).ravel()


def topk(q, T, y, k=5):
    eps = 1e-12
    qx = q + eps; lq = np.log(qx); Tx = T + eps
    d = (qx * lq).sum() - qx @ np.log(Tx).T + (Tx * np.log(Tx)).sum(1) - (Tx @ lq)
    order = np.argsort(d, kind="stable"); seen = []
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

    raw_t1 = raw_t5 = sep_t1 = sep_t5 = n = 0
    for raga in PILOT_RAGAS:
        for v in videos.get(raga, [])[:PER_RAGA]:
            vid = v["id"]; true = cls_idx[raga]
            print(f"[{raga}] {vid} ...", flush=True)
            with tempfile.TemporaryDirectory() as td:
                wav = os.path.join(td, "a.%(ext)s")
                if not _download(vid, wav):
                    print("    download failed"); continue
                files = [f for f in os.listdir(td) if f.startswith("a.")]
                if not files:
                    print("    no file"); continue
                path = os.path.join(td, files[0])
                try:
                    raw_audio = es.MonoLoader(filename=path, sampleRate=SR)()
                    q_raw = query_for(raw_audio)
                    voc = separated_vocals(path)
                    q_sep = query_for(voc)
                except Exception as exc:
                    print(f"    failed: {exc!r}"); continue
            r = topk(q_raw, cmd_T, cmd_y); s = topk(q_sep, cmd_T, cmd_y)
            n += 1
            raw_t1 += r[0] == true; raw_t5 += true in r[:5]
            sep_t1 += s[0] == true; sep_t5 += true in s[:5]
            print(f"    raw -> {classes[r[0]]} {'OK' if r[0]==true else 'X'} | sep -> {classes[s[0]]} {'OK' if s[0]==true else 'X'}")

    if n == 0:
        print("no recordings processed"); return
    print(f"\n=== Source-separation A/B on {n} novel YouTube recordings (vs CMD index) ===")
    print(f"  RAW audio:        top-1 {raw_t1/n*100:5.1f}%   top-5 {raw_t5/n*100:5.1f}%")
    print(f"  Demucs vocals:    top-1 {sep_t1/n*100:5.1f}%   top-5 {sep_t5/n*100:5.1f}%")
    print(f"  n={n}")


if __name__ == "__main__":
    main()
