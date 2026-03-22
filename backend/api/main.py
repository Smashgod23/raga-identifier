from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os, sys
import numpy as np
import pickle
import json
from huggingface_hub import hf_hub_download
from supabase import create_client
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import extract_features_from_audio

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

@app.get("/health")
def health():
    return {"status": "ok", "ragas": len(CLASSES)}

@app.get("/ragas")
def list_ragas():
    return {"ragas": CLASSES, "count": len(CLASSES)}

@app.post("/predict")
async def predict_raga(file: UploadFile = File(...)):
    allowed = {".wav", ".mp3", ".m4a", ".webm", ".ogg"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format: {ext}. Use {allowed}")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        features = extract_features_from_audio(tmp_path)
        features_scaled = SCALER.transform([features])

        probs = MODEL.predict_proba(features_scaled)[0]
        top5_idx = np.argsort(probs)[::-1][:5]

        predictions = [
            {"raga": CLASSES[i], "confidence": round(probs[i] * 100, 1)}
            for i in top5_idx
        ]

        return {
            "top_raga": predictions[0]["raga"],
            "confidence": predictions[0]["confidence"],
            "predictions": predictions
        }

    except ValueError as e:
        raise HTTPException(422, str(e))
    finally:
        os.unlink(tmp_path)
class YouTubeRequest(BaseModel):
    url: str

@app.post("/predict-youtube")
async def predict_youtube(request: YouTubeRequest):
    import subprocess
    import glob as globmod

    url = request.url
    if not any(d in url for d in ['youtube.com', 'youtu.be']):
        raise HTTPException(400, "Please provide a valid YouTube URL")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_template = os.path.join(tmpdir, "audio.%(ext)s")

        try:
            result = subprocess.run(
                [
                    "yt-dlp",
                    "--no-playlist",
                    "-x",
                    "--audio-format", "wav",
                    "-o", output_template,
                    url
                ],
                capture_output=True,
                text=True,
                timeout=120
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(422, "Download timed out — try a shorter video")

        if result.returncode != 0:
            raise HTTPException(422, "Could not download audio from this URL")

        wav_files = globmod.glob(os.path.join(tmpdir, "audio.*"))
        if not wav_files:
            raise HTTPException(422, "Could not extract audio from this video")

        audio_path = wav_files[0]

        try:
            features = extract_features_from_audio(audio_path)
            features_scaled = SCALER.transform([features])

            probs = MODEL.predict_proba(features_scaled)[0]
            top5_idx = np.argsort(probs)[::-1][:5]

            predictions = [
                {"raga": CLASSES[i], "confidence": round(probs[i] * 100, 1)}
                for i in top5_idx
            ]

            return {
                "top_raga": predictions[0]["raga"],
                "confidence": predictions[0]["confidence"],
                "predictions": predictions
            }

        except ValueError as e:
            raise HTTPException(422, str(e))

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