from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os, sys
import numpy as np
import torch
import torch.nn as nn
import pickle
import json
from huggingface_hub import hf_hub_download

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import RagaNet, extract_features_from_audio

app = FastAPI(title="Raga Identifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download model files from Hugging Face at startup
REPO_ID = "Smashgod23/raga-identifier"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

model_path  = hf_hub_download(repo_id=REPO_ID, filename="raga_model_best.pt", local_dir=os.path.join(BASE_DIR, "models"))
scaler_path = hf_hub_download(repo_id=REPO_ID, filename="scaler.pkl",          local_dir=os.path.join(BASE_DIR, "models"))
classes_path= hf_hub_download(repo_id=REPO_ID, filename="classes.json",        local_dir=os.path.join(BASE_DIR, "data"))

with open(classes_path) as f:
    CLASSES = json.load(f)

with open(scaler_path, "rb") as f:
    SCALER = pickle.load(f)

MODEL = RagaNet(input_dim=120, num_classes=len(CLASSES))
MODEL.load_state_dict(torch.load(model_path, map_location="cpu"))
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
        x = torch.tensor(features_scaled, dtype=torch.float32)

        with torch.no_grad():
            logits = MODEL(x)
            probs = torch.softmax(logits, dim=1)[0]
            top5 = torch.topk(probs, 5)

        predictions = [
            {"raga": CLASSES[i], "confidence": round(p.item() * 100, 1)}
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