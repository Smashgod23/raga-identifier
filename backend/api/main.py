from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os, sys, atexit, threading
from urllib.parse import urlparse, parse_qs
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
try:
    from predict_tdms import TDMSIndex
    import predict_essentia
    TDMS_AVAILABLE = True
except Exception as _tdms_exc:
    # Essentia is a heavy native dependency. If its wheel fails to import on the
    # host, keep the whole API up on the v1 path rather than crashing at startup;
    # /predict-tdms then returns 503 and record/upload/youtube still work.
    print(f"TDMS/Essentia import failed: {_tdms_exc!r}. /predict-tdms disabled; v1 endpoints still work.")
    TDMSIndex = None
    predict_essentia = None
    TDMS_AVAILABLE = False

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
    if not TDMS_AVAILABLE:
        return None
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
    single full-file pass.

    Returns (avg_probs, tonic_hz, feature_vectors), where feature_vectors is the
    list of raw pre-scaler vectors actually fed to the model, one per segment.
    They are returned so the caller can log exactly what the model saw; scaled
    values would be useless for retraining because the scaler is refit each
    time."""
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
        return probs, (float(tonic_override) if tonic_override else detected_tonic), [features]

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
    all_feats = []
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
            all_feats.append(feats)
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
            all_feats = [feats]
        except Exception:
            raise ValueError("Could not extract features from audio")

    return np.mean(all_probs, axis=0), anchor_tonic, all_feats


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


# Bump this whenever extract_features_from_audio changes shape or meaning.
# Without it, rows logged under different pipelines look identical and silently
# poison any retraining that mixes them.
FEATURE_VERSION = "pcd-v1"


def _log_prediction(result, features, source):
    """Write one row to `predictions` and attach its id to the response.

    Best-effort by design: a logging outage must never turn a working
    prediction into a 500.
    """
    try:
        row = supabase.table("predictions").insert({
            "source": source,
            "predicted_raga": result["top_raga"],
            "confidence": result["confidence"],
            "top_k": result["predictions"],
            # Raw, pre-scaler. The scaler is refit on every retrain, so scaled
            # values would be frozen against a scaler that no longer exists.
            "features": [[float(x) for x in f] for f in features] if features else None,
            "feature_version": FEATURE_VERSION,
            "n_segments": len(features) if features else 0,
            "tonic_hz": result["tonic_hz"],
            "tonic_overridden": result["tonic_overridden"],
        }).execute()
        result["prediction_id"] = row.data[0]["id"]
    except Exception as exc:
        print(f"prediction log failed: {exc!r}")
    return result


@app.get("/health")
def health():
    return {"status": "ok", "ragas": len(CLASSES)}

@app.get("/ragas")
def list_ragas():
    return {"ragas": CLASSES, "count": len(CLASSES)}

@app.get("/youtube-examples")
def youtube_examples():
    """Curated example videos whose predictions are precomputed and cached, so
    the frontend can offer 'try one of these' links that always work even when
    YouTube is throttling the server's datacenter IP. YT_EXAMPLES is populated
    at startup (see _load_youtube_examples)."""
    return {"examples": YT_EXAMPLES}

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
        probs, used_tonic, feats = _predict_multi_segment(tmp_path, tonic_override=override)
        result = _format_response(probs, used_tonic, tonic_overridden=override is not None)
        return _log_prediction(result, feats, source="upload")
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
        result = _format_response(probs, used_tonic, tonic_overridden=override is not None)
        # features=None on purpose: a TDMS is 14,400 floats, roughly a megabyte
        # per row as JSON. Log the prediction, skip the vector.
        return _log_prediction(result, features=None, source="tdms")
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        # Essentia is imported lazily inside the TDMS path, so a broken/missing
        # native wheel surfaces here at call time, not at startup. Degrade to a
        # clean 503 (v1 endpoints keep working) instead of a 500 stack trace.
        print(f"TDMS inference failed: {e!r}")
        raise HTTPException(503, "The TDMS model is temporarily unavailable. Try the standard prediction instead.")
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


def _is_youtube_url(url: str) -> bool:
    """True only if the URL's actual host is youtube.com/youtu.be (or a
    subdomain like m./music.youtube.com). A substring check would wave through
    things like https://evil.com/?x=youtube.com, so parse the host instead."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return host in ("youtube.com", "youtu.be") or host.endswith(".youtube.com")


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
        # Best-effort cleanup so the cookie jar doesn't linger on a long-lived host.
        atexit.register(lambda: os.path.exists(path) and os.unlink(path))
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
        # Google throttles this Space's datacenter IP and drops media
        # connections mid-stream (SSL EOF). A couple of retries with a short
        # socket timeout recovers transient drops without hanging on a dead one.
        "--retries", "2", "--fragment-retries", "2", "--socket-timeout", "20",
    ]
    if YT_COOKIE_FILE:
        args += ["--cookies", YT_COOKIE_FILE]
    return args + strat


def _youtube_error_detail(stderr: str) -> str:
    """Classify a raw yt-dlp stderr into a user-facing 422 message so the UI can
    say something useful instead of a generic failure. The truncated stderr is
    appended for debugging."""
    s = (stderr or "").lower()
    # Match the bot-check phrasing specifically. YouTube's age gate is the very
    # similar "Sign in to confirm your age", so a loose "sign in to confirm"
    # match would misroute age-restricted videos to the bot-check message.
    if "not a bot" in s or "sign in to confirm you're not a bot" in s:
        msg = ("YouTube is blocking this server with a bot check right now. "
               "Try a different video, or upload the audio file directly.")
    elif "private video" in s or "members-only" in s or "join this channel" in s:
        msg = "This video is private or members-only, so its audio can't be fetched."
    elif ("video unavailable" in s or "removed" in s or "is not available" in s
          or "age-restricted" in s or "confirm your age" in s or "inappropriate for some users" in s):
        msg = "This video is unavailable (removed, region-locked, or age-restricted). Try another."
    elif any(sig in s for sig in (
            "unexpected_eof", "eof occurred", "ssl", "timed out", "timeout",
            "429", "too many requests", "connection reset", "connection aborted",
            "read timed out", "handshake", "rate limit", "rate-limit")):
        # The shared datacenter IP got rate limited or had its download dropped
        # mid-stream. This is the common transient failure, so keep the message
        # clean (no raw SSL noise) and hand the user a concrete workaround.
        return ("YouTube is rate-limiting this server right now. It blocks shared "
                "cloud servers after a few requests. Wait a few minutes and try the "
                "same link again (results are cached once one works). If it keeps "
                "failing, download the audio yourself and use the Upload button "
                "with the file instead.")
    else:
        msg = "Could not extract audio from this video. Try uploading the audio file directly."
    if stderr:
        msg += f" (yt-dlp: {stderr.strip()[-300:]})"
    return msg


def _looks_throttled(stderr: str) -> bool:
    """True if the failure is an IP-level block (Google dropping the connection /
    429), which kills every player client equally. When this is the cause there's
    no point spending ~60s per remaining extractor strategy — bail and surface
    the rate-limit message immediately."""
    s = (stderr or "").lower()
    return any(sig in s for sig in (
        "unexpected_eof", "eof occurred", "ssl", "429", "too many requests",
        "connection reset", "connection aborted", "read timed out", "handshake"))


# Per-video result cache + single-flight lock. The shared datacenter IP only
# succeeds occasionally and throttles harder under concurrent load, so: (1) once
# a link succeeds we cache the result and never hit YouTube for it again, and
# (2) only one fetch runs at a time. In-memory, so it resets on Space restart.
_YT_CACHE = {}
_YT_CACHE_MAX = 256
_yt_fetch_lock = threading.Lock()


def _load_youtube_examples():
    """Curated example videos with predictions precomputed offline from a
    residential IP (src/warm_youtube_examples.py). YouTube throttles this
    Space's datacenter IP, so an open 'paste any link' box fails intermittently
    for first-time visitors. These examples are served straight from the cache,
    so the demo always works regardless of the server's IP reputation. Returns
    a list of {url, title, raga} for the frontend's example buttons."""
    path = os.path.join(BASE_DIR, "data", "youtube_examples.json")
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"No YouTube examples loaded ({exc!r}). The example buttons will be empty.")
        return []
    listing = []
    for vid, entry in data.items():
        try:
            payload = entry["payload"]
            # Seed the cache under the same (video_id, tonic) key the endpoint
            # uses, so a click (or pasted link to the same video) is an instant hit.
            _YT_CACHE[(vid, None)] = payload
            listing.append({
                "url": entry["url"],
                "title": entry["title"],
                "raga": payload["top_raga"],
            })
        except (KeyError, TypeError) as exc:
            # One malformed entry shouldn't take down startup.
            print(f"Skipping malformed YouTube example {vid!r}: {exc!r}")
    print(f"Loaded {len(listing)} precomputed YouTube examples.")
    return listing


YT_EXAMPLES = _load_youtube_examples()


def _youtube_video_id(url: str) -> Optional[str]:
    """Extract the canonical video id from a YouTube URL so cache hits work
    regardless of extra query params (playlist, timestamp, si=...)."""
    try:
        u = urlparse(url)
    except ValueError:
        return None
    host = (u.hostname or "").lower()
    if host == "youtu.be":
        vid = u.path.lstrip("/").split("/")[0]
        return vid or None
    if host == "youtube.com" or host.endswith(".youtube.com"):
        v = parse_qs(u.query).get("v")
        if v and v[0]:
            return v[0]
        parts = [p for p in u.path.split("/") if p]
        if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v"):
            return parts[1]
    return None


class YouTubeRequest(BaseModel):
    url: str
    tonic_hz: Optional[float] = None


def _do_youtube_fetch(url, override):
    """Download one analysis window and run inference. Raises HTTPException on
    failure. Pulled out of the endpoint so the single-flight lock wraps only this
    blocking work."""
    import subprocess
    import glob as globmod

    # Extractor strategies tried in order until one downloads. With the EJS
    # challenge solver enabled (and Deno in the image), the default web client
    # handles most videos; the alternative player clients are fallbacks for the
    # occasional video that rejects it.
    EXTRACTOR_STRATEGIES = [
        [],  # default (web) — EJS solves its JS challenges
        ["--extractor-args", "youtube:player_client=tv"],
        ["--extractor-args", "youtube:player_client=android_vr"],
    ]
    # One fixed analysis window, no duration probe. Each yt-dlp call on a
    # datacenter IP pays a steep EJS-solve + throttle cost, so the old "probe
    # duration, then download three 90s segments" flow meant four round-trips
    # and ~4-5 minutes per request (over the frontend timeout) plus heavy
    # throttling. A single SEGMENT_LEN window from the start of the video is one
    # round-trip — fast enough for the UI and far less likely to get rate
    # limited. File upload still does full multi-segment averaging for accuracy.
    last_stderr = ""

    # Sample a window 1/3 into the recording — the window the model was trained on
    # (download_youtube_data.py). The opening of a concert (tuning, sparse alapana,
    # applause) does not characterize the raga, so the old first-90s window
    # predicted the wrong raga even after the feature pipeline was realigned. A
    # metadata-only duration probe is not subject to the media-CDN throttle, so it
    # usually succeeds even when downloads are being dropped; fall back to the
    # opening window if it fails.
    seg_start = 0
    try:
        probe = subprocess.run(
            _ytdlp_base([]) + ["--skip-download", "--print", "%(duration)s", url],
            capture_output=True, text=True, timeout=30)
        if probe.returncode == 0:
            dur = int((probe.stdout or "0").strip().split("\n")[-1])
            if dur > 180:
                seg_start = max(60, dur // 3)
    except Exception:
        seg_start = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        output_template = os.path.join(tmpdir, "audio.%(ext)s")
        download_ok = False
        for strat in EXTRACTOR_STRATEGIES:
            dl_args = _ytdlp_base(strat) + [
                "-x", "--audio-format", "wav",
                "-o", output_template,
                # yt-dlp clamps the window to the real length for short clips.
                "--download-sections", f"*{seg_start}-{seg_start + SEGMENT_LEN}",
                url,
            ]
            try:
                result = subprocess.run(dl_args, capture_output=True, text=True, timeout=60)
            except subprocess.TimeoutExpired:
                # A stalled download is the IP throttle; other clients won't help.
                last_stderr = "yt-dlp download timed out (60s)"
                break
            if result.returncode == 0 and globmod.glob(os.path.join(tmpdir, "audio.*")):
                download_ok = True
                break
            last_stderr = (result.stderr or "")[-500:]
            # IP-level block (SSL drop / 429) fails every client, so stop trying
            # the rest instead of burning ~60s each. Per-video errors (private,
            # unavailable) don't match this and still fall through to the next.
            if _looks_throttled(last_stderr):
                break

        if not download_ok:
            raise HTTPException(422, _youtube_error_detail(last_stderr))

        wav_files = globmod.glob(os.path.join(tmpdir, "audio.*"))

        # Prefer the TDMS/Essentia model — far more accurate than v1 on real audio
        # (70-75% vs ~17% top-1). On a single downloaded window it runs the
        # single-window TDMS query. Fall back to v1 if TDMS is unavailable or
        # errors on this clip, so the feature still returns a prediction.
        if TDMS_INDEX is not None:
            try:
                probs, tonic = predict_essentia.predict_top_k(
                    TDMS_INDEX, wav_files[0], tonic_override=override
                )
                result = _format_response(probs, tonic, tonic_overridden=override is not None)
                # source stays "youtube" (the entry point) on both paths so the
                # column still counts YouTube traffic. Which model served it is
                # recoverable as source='youtube' AND features IS NULL, since
                # only the v1 fallback below has a vector worth storing.
                return _log_prediction(result, features=None, source="youtube")
            except Exception as exc:
                print(f"YouTube TDMS inference failed, falling back to v1: {exc!r}")

        try:
            features, tonic = extract_features_from_audio(
                wav_files[0], tonic_override=override
            )
            probs = MODEL.predict_proba(SCALER.transform([features]))[0]
        except Exception:
            raise HTTPException(422, _youtube_error_detail(last_stderr or "feature extraction failed"))

        used_tonic = override if override is not None else tonic
        result = _format_response(probs, used_tonic, tonic_overridden=override is not None)
        return _log_prediction(result, [features], source="youtube")


@app.post("/predict-youtube")
def predict_youtube(request: YouTubeRequest):
    # Sync (not async) on purpose: the fetch does blocking work (yt-dlp subprocess
    # + librosa). As a plain def, FastAPI runs it in a worker thread, so a slow
    # YouTube fetch can't freeze /health and every other request.
    url = request.url
    if not _is_youtube_url(url):
        raise HTTPException(400, "Please provide a valid YouTube URL")

    override = request.tonic_hz if request.tonic_hz and request.tonic_hz > 0 else None

    # Cache hit -> instant, no YouTube call. This is what makes already-seen links
    # dependable: the first fetch may need a retry, but after one success it is
    # served from here.
    vid = _youtube_video_id(url)
    cache_key = (vid, override) if vid else None
    if cache_key is not None and cache_key in _YT_CACHE:
        # Strip the logged row id. Without this the second visitor to a cached
        # link would attach their feedback to the first visitor's prediction row.
        # The precomputed YT_EXAMPLES seed this same cache and carry no id, so
        # both paths behave identically.
        return {k: v for k, v in _YT_CACHE[cache_key].items() if k != "prediction_id"}

    # Single-flight: a second fetch arriving mid-fetch is told to wait rather than
    # piling onto the shared IP and throttling both into failure.
    if not _yt_fetch_lock.acquire(blocking=False):
        raise HTTPException(429, "The server is busy fetching another YouTube link "
            "right now. Wait a moment and try again, or use the Upload button with "
            "an audio file instead.")
    try:
        payload = _do_youtube_fetch(url, override)
    finally:
        _yt_fetch_lock.release()

    if cache_key is not None:
        if len(_YT_CACHE) >= _YT_CACHE_MAX:
            _YT_CACHE.clear()
        _YT_CACHE[cache_key] = payload
    return payload


class FeedbackRequest(BaseModel):
    predicted_raga: str
    actual_raga: str
    was_correct: bool
    confidence: float
    audio_filename: str = ""
    prediction_id: str = ""

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    try:
        row = {
            "predicted_raga": feedback.predicted_raga,
            "actual_raga": feedback.actual_raga,
            "was_correct": feedback.was_correct,
            "confidence": feedback.confidence,
            "audio_filename": feedback.audio_filename,
        }
        # Only set when the client actually sent one. The column is a uuid FK,
        # so writing "" would fail, and older clients that predate this field
        # must keep working.
        if feedback.prediction_id:
            row["prediction_id"] = feedback.prediction_id
        supabase.table("feedback").insert(row).execute()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(500, str(e))
