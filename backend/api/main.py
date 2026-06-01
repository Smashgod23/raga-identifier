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
from predict_tdms import TDMSIndex
import predict_essentia

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

# --- Phase 13 TDMS k-NN index, Essentia full-recording templates.
# Templates are full-recording Melodia pitch + expert .tonicFine tonic.
# /predict-tdms queries them with Essentia Melodia + TonicIndianArtMusic
# (predict_essentia). Validated 75.62% top-1 / 91.17% top-5 on the 480-
# recording CMD set with 7x90s queries (vs v1's 8.32% / 28.57%). The
# 480x14400 float32 array is ~28 MB. Index optional — if neither HF nor a
# local copy is available, /predict-tdms returns 503 and v1 keeps working.
def _load_tdms_index() -> Optional["TDMSIndex"]:
    data_dir = os.path.join(BASE_DIR, "data")
    local_X = os.path.join(data_dir, "X_tdms_essfull_template.npy")
    local_y = os.path.join(data_dir, "y_tdms.npy")
    # Prefer local if both files already exist (covers dev environments and
    # any redeploy where HF auth is rate-limited or temporarily down).
    if os.path.exists(local_X) and os.path.exists(local_y):
        try:
            X = np.load(local_X)
            y = np.load(local_y)
            return TDMSIndex(X, y, n_classes=len(CLASSES))
        except Exception as exc:
            print(f"Local TDMS files failed to load: {exc!r}. Falling back to HF.")
    try:
        x_path = hf_hub_download(repo_id=REPO_ID, filename="X_tdms_essfull_template.npy", local_dir=data_dir)
        y_path = hf_hub_download(repo_id=REPO_ID, filename="y_tdms.npy", local_dir=data_dir)
        return TDMSIndex(np.load(x_path), np.load(y_path), n_classes=len(CLASSES))
    except Exception as exc:
        print(f"TDMS index unavailable: {exc!r}. /predict-tdms will return 503.")
        return None


TDMS_INDEX = _load_tdms_index()
if TDMS_INDEX is not None:
    print(f"TDMS index loaded — {len(TDMS_INDEX.X)} Essentia templates")

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


@app.post("/predict-tdms")
async def predict_raga_tdms(
    file: UploadFile = File(...),
    tonic_hz: Optional[float] = Form(None),
):
    """Phase 13 TDMS k-NN inference with Essentia expert extractors.
    Detects the tonic once via TonicIndianArtMusic, averages Melodia-pitch
    TDMSs from up to 7 windows, and does 1-NN (symmetric KL) against the
    full-recording template index. Validated 75.62% top-1 / 91.17% top-5
    on the 480-recording CMD set (7x90s queries) vs v1's 8.32% / 28.57%.
    Short uploads sample fewer independent windows and land closer to 70%."""
    if TDMS_INDEX is None:
        raise HTTPException(503, "TDMS index not loaded")

    allowed = {".wav", ".mp3", ".m4a", ".webm", ".ogg"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format: {ext}. Use {allowed}")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    override = tonic_hz if tonic_hz and tonic_hz > 0 else None
    try:
        probs, used_tonic = predict_essentia.predict_top_k(TDMS_INDEX, tmp_path, tonic_override=override)
        return _format_response(probs, used_tonic, tonic_overridden=override is not None)
    except ValueError as e:
        raise HTTPException(422, str(e))
    finally:
        os.unlink(tmp_path)


# --- YouTube extraction helpers ------------------------------------------------
# This backend runs on a datacenter IP (HF Space), which YouTube blocks for
# yt-dlp unless its JS signature/n-token challenges are solved. We solve them
# two ways, in order of strength: (1) the EJS challenge solver, which needs the
# Deno runtime baked into the Docker image, and (2) an optional cookie jar from
# a logged-in account, supplied via the YT_COOKIES secret. Either alone unblocks
# most videos; together they are the most reliable combination on a cloud IP.

YT_DLP_CACHE = os.path.join(BASE_DIR, ".cache", "yt-dlp")


def _materialize_cookies() -> Optional[str]:
    """Write the YT_COOKIES secret (Netscape cookie-file format) to a private
    temp file once at startup and return its path, or None if unset. yt-dlp
    needs cookies as a file, not an env var. Failures degrade to no-cookies
    rather than taking the endpoint down."""
    raw = os.getenv("YT_COOKIES", "").strip()
    if not raw:
        return None
    try:
        fd, path = tempfile.mkstemp(prefix="yt_cookies_", suffix=".txt")
        with os.fdopen(fd, "w") as f:
            # yt-dlp rejects cookie files without the Netscape header line, so
            # add it for users who paste only the cookie rows.
            if not raw.lstrip().startswith("#"):
                f.write("# Netscape HTTP Cookie File\n")
            f.write(raw + "\n")
        os.chmod(path, 0o600)
        print("YouTube cookies loaded from YT_COOKIES secret.")
        return path
    except Exception as exc:
        print(f"Could not write YT_COOKIES cookie file: {exc!r}. Continuing without cookies.")
        return None


YT_COOKIE_FILE = _materialize_cookies()


def _ytdlp_base(strat: list) -> list:
    """Base yt-dlp argv shared by every call: EJS challenge solving, a writable
    cache dir, optional cookies, and the per-attempt extractor strategy."""
    args = [
        "yt-dlp", "--no-playlist",
        "--remote-components", "ejs:github",
        "--cache-dir", YT_DLP_CACHE,
    ]
    if YT_COOKIE_FILE:
        args += ["--cookies", YT_COOKIE_FILE]
    return args + strat


def _youtube_error_detail(stderr: str) -> str:
    """Classify a raw yt-dlp stderr into a user-facing 422 message so the UI can
    say something useful instead of a generic failure. The truncated stderr is
    appended for debugging."""
    s = (stderr or "").lower()
    if "sign in to confirm" in s or "not a bot" in s or "confirm you" in s:
        msg = ("YouTube is blocking this server with a bot check right now. "
               "Try a different video, or upload the audio file directly.")
    elif "private video" in s or "members-only" in s or "join this channel" in s:
        msg = "This video is private or members-only, so its audio can't be fetched."
    elif "video unavailable" in s or "removed" in s or "is not available" in s or "age" in s:
        msg = "This video is unavailable (removed, region-locked, or age-restricted). Try another."
    elif "timed out" in s or "timeout" in s:
        msg = ("Fetching this video took too long and timed out. Try a shorter "
               "video, or upload the audio file directly.")
    else:
        msg = "Could not extract audio from this video. Try uploading the audio file directly."
    if stderr:
        msg += f" (yt-dlp: {stderr.strip()[-300:]})"
    return msg


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

    # Extractor strategies tried in order until one returns metadata. With the
    # EJS challenge solver enabled (and Deno in the image), the default web
    # client handles most videos; the alternative player clients are fallbacks
    # for the occasional video that rejects it. We lock onto the FIRST strategy
    # that works during the duration probe and reuse it for every segment
    # download, so a failing video bails out in well under two minutes instead
    # of grinding through every client x every segment (the old design could
    # hang for ~18 minutes before giving up).
    EXTRACTOR_STRATEGIES = [
        [],  # default (web) — EJS solves its JS challenges
        ["--extractor-args", "youtube:player_client=tv"],
        ["--extractor-args", "youtube:player_client=android_vr"],
    ]
    last_stderr = ""

    with tempfile.TemporaryDirectory() as tmpdir:
        # Probe duration, locking onto the first strategy that returns it. The
        # very first probe after a cold start also downloads + caches the EJS
        # components, so the 40s budget is deliberately generous.
        duration = 0
        working_strat = None
        for strat in EXTRACTOR_STRATEGIES:
            try:
                dur_result = subprocess.run(
                    _ytdlp_base(strat) + ["--print", "duration", url],
                    capture_output=True, text=True, timeout=40,
                )
            except subprocess.TimeoutExpired:
                last_stderr = "yt-dlp metadata probe timed out (40s)"
                continue
            out_lines = (dur_result.stdout or "").strip().splitlines()
            if dur_result.returncode == 0 and out_lines:
                try:
                    # yt-dlp prints duration as a float; take the last line in
                    # case warnings leaked onto stdout.
                    duration = int(float(out_lines[-1]))
                    working_strat = strat
                    break
                except ValueError:
                    pass
            last_stderr = (dur_result.stderr or "")[-500:]

        # No strategy could even read the video's metadata -> fail fast with a
        # classified message instead of attempting doomed downloads.
        if working_strat is None:
            raise HTTPException(422, _youtube_error_detail(last_stderr))

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

            # Reuse the strategy that won the probe — no per-segment client loop.
            dl_args = _ytdlp_base(working_strat) + [
                "-x", "--audio-format", "wav",
                "-o", output_template,
            ]
            if seg is not None:
                dl_args += ["--download-sections", f"*{seg[0]}-{seg[1]}"]
            dl_args.append(url)

            try:
                result = subprocess.run(dl_args, capture_output=True, text=True, timeout=90)
            except subprocess.TimeoutExpired:
                last_stderr = "yt-dlp download timed out (90s)"
                continue
            if result.returncode != 0:
                last_stderr = (result.stderr or "")[-500:]
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
            except Exception:
                continue

        if not all_probs:
            # Metadata read but every download/feature pass failed. Surface the
            # classified yt-dlp failure so the UI can guide the user.
            raise HTTPException(422, _youtube_error_detail(last_stderr))

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
