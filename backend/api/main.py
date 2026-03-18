from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os, sys
import numpy as np
import torch
import torch.nn as nn
import pickle
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import RagaNet, extract_features_from_audio

app = FastAPI(title="Raga Identifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, "data", "classes.json")) as f:
    CLASSES = json.load(f)

with open(os.path.join(BASE_DIR, "models", "scaler.pkl"), "rb") as f:
    SCALER = pickle.load(f)

MODEL = RagaNet(input_dim=120, num_classes=len(CLASSES))
MODEL.load_state_dict(torch.load(
    os.path.join(BASE_DIR, "models", "raga_model_best.pt"),
    map_location="cpu"
))
MODEL.eval()

print(f"Model loaded — {len(CLASSES)} ragas")

@app.get("/health")
def health():
    return {"status": "ok", "ragas": len(CLASSES)}

@app.get("/ragas")
def list_ragas():
    return {"ragas": CLASSES, "count": len(CLASSES)}

@app.post("/predict")
async def predict_raga(file: UploadFile = File(...)):
    # Accept wav, mp3, m4a, webm
    allowed = {".wav", ".mp3", ".m4a", ".webm", ".ogg"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format: {ext}. Use {allowed}")

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        features = extract_features_from_audio(tmp_path)
        features_scaled = SCALER.transform([features])
        x = torch.tensor(features_scaled, dtype=torch.float32)

        with torch.no_grad():
            logits = MODEL(x)
            probs = torch.softmax(logits, dim=1)[0]
            top5 = torch.topk(probs, 5)

        predictions = [
            {
                "raga": CLASSES[i],
                "confidence": round(p.item() * 100, 1)
            }
            for i, p in zip(top5.indices, top5.values)
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