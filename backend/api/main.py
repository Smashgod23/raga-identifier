from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os, sys
import numpy as np
import pickle
import json
from huggingface_hub import hf_hub_download
from supabase import create_client
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import extract_features_from_audio, hz_to_note_name

app = FastAPI(title="Raga Identifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_ID = "Smashgod23/raga-identifier"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

model_path  = hf_hub_download(repo_id=REPO_ID, filename="raga_sklearn.pkl", local_dir=os.path.join(BASE_DIR, "models"))
scaler_path = hf_hub_download(repo_id=REPO_ID, filename="scaler.pkl",       local_dir=os.path.join(BASE_DIR, "models"))
classes_path= hf_hub_download(repo_id=REPO_ID, filename="classes.json",     local_dir=os.path.join(BASE_DIR, "data"))

with open(classes_path) as f:
    CLASSES = json.load(f)

with open(scaler_path, "rb") as f:
    SCALER = pickle.load(f)

with open(model_path, "rb") as f:
    MODEL = pickle.load(f)

print(f"Model loaded — {len(CLASSES)} ragas")

# Multi-segment threshold: videos/clips longer than this get sampled in three
# 90s windows and averaged, matching how the YouTube pipeline already works.
LONG_CLIP_THRESHOLD = 180
SEGMENT_LEN = 90


def _predict_multi_segment(audio_path, tonic_override=None):
    """Run inference over an audio file, using multi-segment averaging for long
    clips. For >180s clips we detect Sa once on a representative middle window
    (or use the override), then extract features from three segments spaced
    across the recording and average the probabilities. Short clips get a
    single full-file pass. Returns (avg_probs, tonic_hz)."""
    import librosa

    try:
        total_dur = float(librosa.get_duration(path=audio_path))
    except Exception:
        total_dur = 0.0

    if total_dur <= LONG_CLIP_THRESHOLD or total_dur <= 0:
        features, detected_tonic = extract_features_from_audio(
            audio_path, tonic_override=tonic_override, duration=total_dur if total_dur > 0 else None
        )
        probs = MODEL.predict_proba(SCALER.transform([features]))[0]
        # Report the user-supplied Sa as-is so the UI doesn't show an octave-shifted note.
        return probs, (float(tonic_override) if tonic_override else detected_tonic)

    # Long clip: anchor Sa once so per-segment features share the same cents reference.
    if tonic_override is not None and float(tonic_override) > 0:
        anchor_tonic = float(tonic_override)
    else:
        anchor_offset = max(0.0, (total_dur - LONG_CLIP_THRESHOLD) / 2)
        _, anchor_tonic = extract_features_from_audio(
            audio_path, offset=anchor_offset, duration=LONG_CLIP_THRESHOLD
        )

    quarter = total_dur / 4
    segments = [
        (quarter, SEGMENT_LEN),
        (2 * quarter, SEGMENT_LEN),
        (3 * quarter, SEGMENT_LEN),
    ]

    all_probs = []
    for offset, dur in segments:
        if offset + dur > total_dur:
            dur = max(30.0, total_dur - offset)
        if dur < 30.0:
            continue
        try:
            feats, _ = extract_features_from_audio(
                audio_path, tonic_override=anchor_tonic, offset=offset, duration=dur
            )
            all_probs.append(MODEL.predict_proba(SCALER.transform([feats]))[0])
        except (ValueError, Exception):
            continue

    if not all_probs:
        # Fallback: one pass on the middle window. If even this fails, surface a 422
        # rather than a 500 — typical cause is a corrupt or fully-silent file.
        try:
            feats, _ = extract_features_from_audio(
                audio_path, tonic_override=anchor_tonic,
                offset=max(0.0, (total_dur - LONG_CLIP_THRESHOLD) / 2),
                duration=LONG_CLIP_THRESHOLD,
            )
            all_probs = [MODEL.predict_proba(SCALER.transform([feats]))[0]]
        except Exception:
            raise ValueError("Could not extract features from audio")

    return np.mean(all_probs, axis=0), anchor_tonic


def _format_response(probs, tonic_hz, tonic_overridden):
    top5_idx = np.argsort(probs)[::-1][:5]
    predictions = [
        {"raga": CLASSES[i], "confidence": round(float(probs[i]) * 100, 1)}
        for i in top5_idx
    ]
    return {
        "top_raga": predictions[0]["raga"],
        "confidence": predictions[0]["confidence"],
        "predictions": predictions,
        "tonic_hz": round(float(tonic_hz), 2) if tonic_hz else None,
        "tonic_note": hz_to_note_name(tonic_hz) if tonic_hz else '',
        "tonic_overridden": bool(tonic_overridden),
    }


@app.get("/health")
def health():
    return {"status": "ok", "ragas": len(CLASSES)}

@app.get("/ragas")
def list_ragas():
    return {"ragas": CLASSES, "count": len(CLASSES)}

@app.post("/predict")
async def predict_raga(
    file: UploadFile = File(...),
    tonic_hz: Optional[float] = Form(None),
):
    allowed = {".wav", ".mp3", ".m4a", ".webm", ".ogg"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format: {ext}. Use {allowed}")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    override = tonic_hz if tonic_hz and tonic_hz > 0 else None
    try:
        probs, used_tonic = _predict_multi_segment(tmp_path, tonic_override=override)
        return _format_response(probs, used_tonic, tonic_overridden=override is not None)
    except ValueError as e:
        raise HTTPException(422, str(e))
    finally:
        os.unlink(tmp_path)


class YouTubeRequest(BaseModel):
    url: str
    tonic_hz: Optional[float] = None


@app.post("/predict-youtube")
async def predict_youtube(request: YouTubeRequest):
    import subprocess
    import glob as globmod

    url = request.url
    if not any(d in url for d in ['youtube.com', 'youtu.be']):
        raise HTTPException(400, "Please provide a valid YouTube URL")

    override = request.tonic_hz if request.tonic_hz and request.tonic_hz > 0 else None

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            dur_result = subprocess.run(
                ["yt-dlp", "--no-playlist", "--print", "duration", url],
                capture_output=True, text=True, timeout=30
            )
            duration = int(dur_result.stdout.strip()) if dur_result.returncode == 0 else 0
        except (subprocess.TimeoutExpired, ValueError):
            duration = 0

        if duration > LONG_CLIP_THRESHOLD:
            quarter = duration // 4
            segments = [
                (quarter, quarter + SEGMENT_LEN),
                (2 * quarter, 2 * quarter + SEGMENT_LEN),
                (3 * quarter, 3 * quarter + SEGMENT_LEN),
            ]
        else:
            segments = [None]

        all_probs = []
        anchor_tonic = None
        for seg in segments:
            seg_dir = os.path.join(tmpdir, f"seg_{seg[0] if seg else 'full'}")
            os.makedirs(seg_dir, exist_ok=True)
            output_template = os.path.join(seg_dir, "audio.%(ext)s")

            dl_args = [
                "yt-dlp", "--no-playlist",
                "-x", "--audio-format", "wav",
                "-o", output_template,
            ]
            if seg is not None:
                dl_args += ["--download-sections", f"*{seg[0]}-{seg[1]}"]
            dl_args.append(url)

            try:
                result = subprocess.run(
                    dl_args, capture_output=True, text=True, timeout=120
                )
            except subprocess.TimeoutExpired:
                continue
            if result.returncode != 0:
                continue

            wav_files = globmod.glob(os.path.join(seg_dir, "audio.*"))
            if not wav_files:
                continue

            # First successful segment sets Sa so later segments share the same cents reference.
            seg_override = override if override is not None else anchor_tonic
            try:
                features, tonic = extract_features_from_audio(
                    wav_files[0], tonic_override=seg_override
                )
                if anchor_tonic is None:
                    anchor_tonic = tonic
                all_probs.append(MODEL.predict_proba(SCALER.transform([features]))[0])
            except (ValueError, Exception):
                continue

        if not all_probs:
            raise HTTPException(422, "Could not extract audio from this video")

        avg_probs = np.mean(all_probs, axis=0)
        used_tonic = override if override is not None else anchor_tonic
        return _format_response(avg_probs, used_tonic, tonic_overridden=override is not None)


class FeedbackRequest(BaseModel):
    predicted_raga: str
    actual_raga: str
    was_correct: bool
    confidence: float
    audio_filename: str = ""

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    try:
        supabase.table("feedback").insert({
            "predicted_raga": feedback.predicted_raga,
            "actual_raga": feedback.actual_raga,
            "was_correct": feedback.was_correct,
            "confidence": feedback.confidence,
            "audio_filename": feedback.audio_filename
        }).execute()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(500, str(e))
