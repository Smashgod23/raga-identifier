# Raga Identifier

A Carnatic music raga recognition system built by Pratham Aithal, a high school student at Rock Hill High School in Frisco, TX (PISD).

Live site: https://raga-identifier.vercel.app
GitHub: https://github.com/Smashgod23/raga-identifier
Contact: theprathamaithal@gmail.com

---

## What This Is

Raga Identifier is a web application that listens to Carnatic music, either recorded live from a microphone, uploaded as an audio file, or provided via a YouTube link, and identifies which raga is being performed. Think of it as Shazam for Carnatic ragas. The system currently recognizes 40 Carnatic ragas, trained on 689 samples from the CompMusic research dataset and YouTube recordings.

I built this project from scratch to connect two of my personal interests: Carnatic vocal music and machine learning. It is not a wrapper around a pre-existing API. I designed, trained, and deployed the model entirely from the ground up.

---

## Background and Motivation

Carnatic music is one of the two main subgenres of Indian classical music, originating in South India. The central concept is the raga, a melodic framework defined not just by a scale (a set of ascending and descending notes called arohanam and avarohanam) but by characteristic phrases, ornaments (gamakam), and the emotional mood (rasa) it evokes. There are hundreds of ragas in the Carnatic tradition, each with a distinct identity.

Identifying a raga by ear is a skill that takes trained musicians years to develop. I wanted to see whether a machine learning model could approximate this ability, and ultimately build something that helps learners, musicians, and enthusiasts identify ragas from recordings. As someone who studies Carnatic vocal music myself, this felt like a meaningful problem to work on.

---

## Academic Foundation

The feature extraction approach I used in this project is directly informed by the PhD thesis:

Koduri, G.K. (2016). Towards a multimodal knowledge base for Indian art music: A case study with melodic intonation. Universitat Pompeu Fabra, Barcelona. Supervised by Dr. Xavier Serra Casals, Music Technology Group.

This thesis was produced as part of the CompMusic project (ERC grant 267583), which built one of the largest annotated corpora of Indian classical music. The full thesis is available at: http://compmusic.upf.edu/phd-thesis-gkoduri

### Key Findings from the Thesis That Influenced This Project

**1. Pitch Distribution as Raga Identity**

The thesis establishes that the pitch class distribution, a histogram of how often each pitch relative to the tonic appears, is an effective and shruti-independent feature for raga recognition. Because Carnatic music is performed in different shrutis (tonic frequencies) by different artists, normalizing pitches relative to the tonic and folding them into a single octave is essential.

I implemented a 120-bin pitch class distribution where each bin represents 10 cents, covering one octave (0 to 1200 cents relative to the tonic). This is consistent with the Koduri thesis finding that finer bin resolutions around 10 cents outperform coarser bins for Carnatic music classification.

**2. Stable Pitch Filtering**

A critical insight from the thesis is that naively including all voiced pitch frames, including ornaments and gamakas, introduces noise that actually hurts classification accuracy. The thesis proposes filtering to only stable pitch regions using two thresholds: a maximum allowed pitch slope (Tslope, measured in cents per second) and a minimum duration for a stable region (Ttime, measured in seconds). This removes passing notes and ornaments, keeping only the pitches where a performer is truly resting on a swara.

I implemented this with Tslope = 1500 cents/sec and Ttime = 0.1 seconds, consistent with the optimal values reported in the thesis. This change improved my model accuracy from 80.2% to 84.4%.

**3. Duration-Weighted Distribution**

The thesis also proposes weighting pitch contributions by how long they are held rather than simply counting frames. A sustained note held for 2 seconds contributes more to the pitch distribution than a passing note. I implemented this as a second feature channel alongside the stable pitch distribution, giving the model a richer representation of each recording.

**4. Nyas Segments**

The dataset included .flatSegNyas files which mark timestamps of flat or sustained notes, the moments when performers rest on characteristic swaras. These matter most for raga identification because the vadi (dominant swara) and samvadi (sub-dominant swara) are most clearly expressed during sustained notes. By extracting pitch distributions specifically from these segments, I was able to give the model cleaner and more informative training examples.

**5. Tani Avartanam Exclusion**

The dataset also included .taniSegKNN files marking where the percussion solo (tani avartanam) begins. These sections contain no melodic content and would add noise to pitch-based features. I exclude these sections during feature extraction.

---

## Dataset

The core training data came from the Indian Art Music Raga Recognition Dataset (features), available at https://zenodo.org/records/7278505. This dataset was produced by the CompMusic research group and contains pre-extracted pitch features for 480 Carnatic music recordings across 40 ragas, with 12 recordings per raga. An additional 209 samples were sourced from YouTube recordings of all 40 ragas (using yt-dlp search and the inference pipeline for feature extraction), bringing the total to 689 training samples. The CompMusic features include:

- .pitch: raw pitch contour with timestamps
- .pitchSilIntrpPP: pitch with silences interpolated
- .tonicFine: the estimated tonic frequency of each recording
- .flatSegNyas: timestamps of flat/sustained note segments
- .taniSegKNN: timestamps of percussion solo sections

Each recording is a full-length concert performance by professional Carnatic musicians, labeled with the raga performed.

The 40 ragas in the dataset are: Sanmukhapriya, Kapi, Bhairavi, Madhyamavati, Bilahari, Mohanam, Sencurutti, Sriranjani, Ritigaula, Husseni, Dhanyasi, Atana, Behag, Surati, Kamavardani, Mukhari, Sindhubhairavi, Sahana, Kanada, Mayamalavagaula, Nata, Shankarabharanam, Saveri, Kamas, Todi, Begada, Harikambhoji, Sri, Kalyani, Sama, Natakurinji, Purvikalyani, Yadukula Kamboji, Devagandhari, Kedaragaula, Anandabhairavi, Gaula, Varali, Kambhoji, and Karaharapriya.

---

## System Architecture

The project is split into three layers: the ML pipeline, the backend API, and the frontend web app.

### ML Pipeline (Python, local)

**Feature Extraction (src/preprocess.py)**

For each recording in the dataset, I:
1. Load the pitch contour and tonic from the .pitchSilIntrpPP and .tonicFine files
2. Exclude any tani avartanam sections using .taniSegKNN timestamps
3. Extract pitches from flat/nyas segments using .flatSegNyas timestamps
4. Convert all pitches to cents relative to the tonic
5. Apply stable pitch filtering (Tslope = 1500 cents/sec, Ttime = 0.1 sec)
6. Compute three 120-bin pitch class distributions: nyas-based, duration-weighted, and stable-filtered
7. Concatenate them into a single 360-dimensional feature vector

**Audio Clip Pipeline (src/preprocess_audio_clips.py)**

A second preprocessing pipeline that slices the full raw audio recordings (MP3 files from the CompMusic audio dataset) into 30-second clips with a 10-second hop and extracts features from each clip directly via the pyin pitch estimator. This is the same inference path used at runtime in predict.py. The tonic for each clip is taken from the corresponding expert-annotated .tonicFine file, not auto-detected, because auto-detection fails too often on short windows. The pipeline produced 44,071 clips across 480 recordings, which will be used to expand training data from 689 samples to roughly 44,760. A multiprocessing pool with 7 workers is used because pyin (not disk I/O) is the bottleneck.

**Model (src/train.py)**

I trained a feedforward neural network using PyTorch with the following architecture:
- Input: 360-dimensional feature vector
- Hidden layer 1: 256 units, BatchNorm, ReLU, 30% dropout
- Hidden layer 2: 128 units, BatchNorm, ReLU, 30% dropout
- Hidden layer 3: 64 units, ReLU
- Output: 40 classes (one per raga)

Training uses Adam optimizer with learning rate 0.001 and weight decay 1e-4, with a step learning rate scheduler. Training runs for 200 epochs and the best checkpoint is saved. The model achieves 84.4% accuracy on a held-out 20% test set.

For deployment, the PyTorch model is converted to a scikit-learn MLPClassifier (same architecture) to avoid the 2GB PyTorch dependency on the server.

**Inference (src/predict.py)**

For a new audio file:
1. Load audio at 16kHz using librosa, normalize amplitude
2. Extract pitch contour using pyin (probabilistic YIN algorithm), filter by voiced probability
3. Detect the tonic by folding all pitches into a single octave, evaluating the top 5 frequency candidates, and selecting the one that produces the most concentrated (peaked) pitch-class distribution
4. Apply the same three-channel feature extraction as training
5. Scale features using the saved StandardScaler
6. Run inference and return the top 5 predictions with confidence scores

### Backend API (FastAPI, Hugging Face Spaces)

The backend is a FastAPI application deployed on Hugging Face Spaces at smashgod23-raga-identifier-api.hf.space.

On startup it downloads the model, scaler, and class list from Hugging Face Hub (Smashgod23/raga-identifier) to avoid storing large files in the git repository.

Endpoints:
- GET /health: returns status and number of ragas
- GET /ragas: returns the full list of 40 ragas
- POST /predict: accepts an audio file, runs inference, saves the audio to Supabase Storage, returns top 5 predictions and a unique audio ID
- POST /predict-youtube: accepts a YouTube URL, downloads audio using yt-dlp. For videos longer than 3 minutes, it samples 3 segments from different parts of the video (at 25%, 50%, and 75% through) and averages the predictions, which avoids tuning sections and intros that would throw off the model. Returns top 5 predictions.
- POST /feedback: accepts user feedback (predicted raga, actual raga, correctness, confidence, audio filename) and stores it in the Supabase feedback table

The backend uses Supabase for storage and the feedback database. Environment variables SUPABASE_URL and SUPABASE_KEY are set in the Hugging Face Spaces secrets configuration.

### Frontend (React + Vite, Vercel)

The frontend is a React application deployed on Vercel at raga-identifier.vercel.app.

Features:
- Live microphone recording with real-time waveform visualization using the Web Audio API AnalyserNode
- File upload for .wav, .mp3, and .m4a files
- YouTube link input: paste a YouTube URL and the backend extracts and analyzes the audio
- Step-by-step processing status messages that keep the user informed during analysis (e.g. "Detecting pitch contour...", "Estimating tonic (Sa)..."), with extra context for YouTube downloads
- Results display showing the top raga name, confidence percentage, arohanam, avarohanam, and a confidence bar chart for the top 5 predictions
- Similar ragas panel based on shared swara sets
- Human feedback loop: after each prediction, the user is asked whether the result was correct. If not, a searchable dropdown lets them select the actual raga. This feedback, along with the audio file ID, is sent to the backend and stored in Supabase for future retraining.
- About section with links to GitHub and contact email

### Infrastructure

- Model storage: Hugging Face Hub (free tier)
- Backend hosting: Hugging Face Spaces (free tier)
- Frontend hosting: Vercel (free tier)
- Database and file storage: Supabase (free tier)
- Version control: GitHub

---

## Obstacles and How I Solved Them

**Python environment issues**

The first major obstacle was getting Python 3.11.9 installed correctly on macOS via pyenv. The .zshrc file was owned by root, preventing writes. I fixed this with sudo chown to reclaim ownership. Then the Python build was missing lzma support, causing librosa to fail on import. I fixed this by installing xz via Homebrew and rebuilding Python with the correct flags pointing to the xz library.

**Tonic detection**

Getting the tonic of a new recording right is the single hardest part of the inference pipeline. An incorrect tonic shifts every pitch by the wrong amount and the model fails completely. My first attempt used the median voiced pitch, which was often wrong. My second attempt used frequency histogram peak detection but got confused by high-frequency harmonics. The final approach folds all pitches into a single octave to collapse the tonic signature regardless of which octave the performer sings in, then finds the dominant pitch in that octave and scales it back to match the median pitch of the performance.

**Auto-tonic failure on short clips**

When I built the audio clip preprocessing pipeline, I initially planned to auto-detect the tonic for each 30-second clip the same way inference does it at runtime. I ran a diagnostic on five random recordings before starting the overnight pipeline run, and found that 3 out of 5 were completely wrong - one had an octave error where the algorithm detected 91.95 Hz instead of 183.51 Hz (the median voiced pitch sat right on the boundary of the octave-folding loop), one locked onto Pa instead of Sa because the tanpura drone sustains both notes equally and the early part of the recording had more melodic activity on the fifth, and one locked onto Ri during the alapana intro where the performer was ornamenting that swara heavily. The per-clip failure rate on 30-second windows would have been around 60%, producing garbage features after a multi-hour processing run. I killed the pipeline before it wrote any output and refactored it to pre-load the expert-annotated .tonicFine values (one per recording) and pass them through as a tonic override. All 480 recordings have matching .tonicFine files, and the pipeline now hard-aborts at startup if any are missing.

**YouTube download issues**

yt-dlp initially failed because it needed a JavaScript runtime to solve YouTube's challenge system. Installing Deno and updating yt-dlp resolved this. Many video URLs were also unavailable, requiring multiple retries with different videos.

**Data augmentation failure**

I attempted to augment the training data by adding Gaussian noise to feature vectors. This reduced accuracy from 78% to 3%. The reason is that pitch class distributions are normalized histograms and adding noise to them destroys their mathematical properties. I later tried more conservative approaches (small circular shifts to simulate tonic errors, tiny noise with scale jitter) but these also hurt accuracy - even a 3-bin shift (30 cents) distorts the subtle pitch patterns that distinguish similar ragas. With only 12 samples per class, the model doesn't have enough signal to learn through the augmentation noise. The real fix turned out to be more data, not augmentation.

**Recording ID collisions**

During development of the audio clip pipeline, I initially used the filename (basename only) as the recording ID for each clip, which is used to keep clips from the same recording in the same train/test split. I found that 42 recordings share the same song title across different artists, for example Sri_Madhava.mp3 appears under two different Behag artists. This would have incorrectly grouped clips from separate recordings together and possibly leaked correlated data across the split boundary. I fixed this by using the full relative path from the audio root directory as the recording ID instead of just the filename.

**PyTorch deployment size**

Railway's free tier has a 4GB Docker image limit and PyTorch alone is over 2GB. I solved this by converting the trained PyTorch model to a scikit-learn MLPClassifier, which has equivalent inference behavior but requires only scikit-learn as a dependency, keeping the Docker image under 1GB.

**sklearn and numpy version mismatches**

After converting to sklearn, the deployment crashed because the local sklearn version (1.5.2) did not match the server's default (1.5.0). Pickle files are not forward or backward compatible across sklearn or numpy versions. I fixed this by pinning both sklearn and numpy to the exact versions used during training in requirements-deploy.txt.

**React hooks error**

The feedback state variables were accidentally placed outside the App component function, causing an invalid hook call. React requires all hooks to be called inside a function component. The fix was moving them inside the component body.

---

## Accuracy and Model Performance

| Metric | Value |
|---|---|
| Number of ragas | 40 |
| Training samples | 689 (480 CompMusic + 209 YouTube) |
| Feature dimensions | 360 |
| Baseline (random guessing) | 2.5% |

The model performs best on ragas with very distinctive swara sets. On real recordings, Todi came in at 97.9% confidence, Kalyani at 97.3%, and Shankarabharanam at 97.6% (from the middle of an MS Subbulakshmi recording). The hardest cases are pentatonic ragas that share many swaras, like Mohanam and Bilahari, where the difference lies in specific ornamental patterns rather than the swara set alone.

---

## Human Feedback Loop

Every time someone uses the app and submits feedback, the following data is stored in Supabase:
- The predicted raga
- The actual raga (if the user corrects it)
- Whether the prediction was correct
- The confidence score
- The audio file ID (the audio itself is stored in Supabase Storage)

When enough feedback accumulates (target: 50+ corrections per raga), I plan to download the corrected audio files, extract features from them, add them to the training set, and retrain the model. This creates a loop where real-world usage directly improves the model over time.

The original 84.4% model is backed up on Hugging Face as raga_sklearn_v1_84pct.pkl so I can always revert if a retrained version performs worse.

---

## Next Steps

**Expanded training data from full audio recordings**

The audio clip pipeline (src/preprocess_audio_clips.py) has already run and produced 44,071 clips from the 480 CompMusic concert recordings. The next step is to combine these with the existing 689-sample dataset and retrain using a recording-aware train/test split, where all clips from the same recording stay on the same side of the split. This is required because clips with a 10-second hop from the same concert are heavily correlated, and a naive random split would leak them across train and test, inflating accuracy numbers. The retrained model is expected to improve substantially on hard cases like similar pentatonic ragas.

**Improved tonic detection**

Tonic detection now evaluates multiple candidates and picks the one producing the most concentrated pitch distribution, which improved accuracy on live recordings. Further improvements could include a dedicated tonic detection model trained specifically for Carnatic music.

**More ragas**

The current 40 ragas represent a good cross-section of common Carnatic ragas but there are hundreds more. As I collect more data, the goal is to expand to at least 100 ragas.

**Mobile app**

The web app is built in React which can be wrapped into a native iOS and Android app using Capacitor. I plan to publish on the App Store and Google Play Store once the model accuracy is higher. iOS requires an Apple Developer account ($99/year) and Android requires a Google Play Developer account ($25 one-time fee).

**Automatic retraining pipeline**

Currently retraining is a manual process. The long-term goal is a script that runs on a schedule, checks how much new feedback has accumulated in Supabase, downloads the audio, retrains the model if there is enough new data, evaluates it against the held-out test set, and only deploys if accuracy improves.

**Gamakam analysis**

The current approach treats gamakam (ornaments) as noise to be filtered out. But gamakam is actually deeply characteristic of specific ragas. For example, the oscillation pattern on Ga in Todi is completely different from the oscillation on Ga in Bhairavi even though both use the same komal Ga. A more advanced model would analyze ornamental patterns as a feature rather than discarding them.

**Desktop app**

Using Electron, the same React codebase can be packaged as a Mac, Windows, and Linux desktop application for offline use.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Audio processing | librosa, scipy, yt-dlp |
| Model training | PyTorch |
| Model inference (deployed) | scikit-learn MLPClassifier |
| Backend API | FastAPI, Python 3.11 |
| Backend hosting | Hugging Face Spaces |
| Model storage | Hugging Face Hub |
| Database and file storage | Supabase |
| Frontend | React, Vite |
| Frontend hosting | Vercel |
| Version control | GitHub |

---

## Project Structure

```
raga-identifier/
├── backend/
│   ├── api/
│   │   └── main.py                        FastAPI app
│   ├── src/
│   │   ├── preprocess.py                  Feature extraction from CompMusic pitch files
│   │   ├── preprocess_audio_clips.py      Clip slicing pipeline for raw audio
│   │   ├── train.py                       Model training (PyTorch + sklearn)
│   │   ├── predict.py                     Inference for live audio
│   │   ├── download_youtube_data.py       YouTube data collection
│   │   ├── verify_tonic_detection.py      Diagnostic: auto-tonic vs expert on 5 recordings
│   │   └── verify_clip_features.py        Pre-flight check for clip feature quality
│   ├── data/
│   │   ├── X.npy                          CompMusic training features (480 samples)
│   │   ├── X_yt.npy                       YouTube training features (209 samples)
│   │   ├── y.npy / y_yt.npy              Training labels
│   │   ├── classes.json                   Raga names
│   │   └── youtube_videos.json            Video index for deduplication
│   ├── models/
│   │   ├── raga_model_best.pt             PyTorch model
│   │   ├── raga_sklearn.pkl               Deployed sklearn model
│   │   └── scaler.pkl                     Feature scaler
│   ├── requirements.txt                   Full local dependencies
│   ├── requirements-deploy.txt            Slim deployment dependencies
│   └── Dockerfile
└── frontend/
    ├── src/
    │   ├── App.jsx                        Main React component
    │   └── ragas.json                     Raga knowledge base
    ├── index.html
    └── package.json
```

---

## Running Locally

**Backend**

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at http://localhost:5173 and expects the backend at http://localhost:8000. To point the frontend at the local backend instead of the deployed API, change API_URL in src/App.jsx.

---

## Retraining the Model

```bash
cd backend
source venv/bin/activate

# (Optional) Collect more YouTube training data
python src/download_youtube_data.py

# Extract features from CompMusic pitch files
python src/preprocess.py

# (Optional) Slice raw audio recordings into 30s clips and extract features
# Requires the CompMusic audio dataset at ~/raga-data-audio/
python src/preprocess_audio_clips.py

# Train - produces both PyTorch model and sklearn model for deployment
python src/train.py

# Upload to Hugging Face
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(path_or_fileobj='models/raga_sklearn.pkl', path_in_repo='raga_sklearn.pkl', repo_id='Smashgod23/raga-identifier')
api.upload_file(path_or_fileobj='models/scaler.pkl', path_in_repo='scaler.pkl', repo_id='Smashgod23/raga-identifier')
"
```

Then push to GitHub to trigger a Hugging Face Spaces redeploy.

---

Built by Pratham Aithal
Rock Hill High School, Frisco, TX (PISD)
theprathamaithal@gmail.com
https://github.com/Smashgod23/raga-identifier
