# Raga Identifier

A Carnatic music raga recognition system built by Pratham Aithal, a high school student at Rock Hill High School in Frisco, TX (PISD).

Live site: https://raga-identifier.vercel.app
GitHub: https://github.com/Smashgod23/raga-identifier
Contact: theprathamaithal@gmail.com

---

## What This Is

Raga Identifier is a web application that listens to Carnatic music, either recorded live from a microphone, uploaded as an audio file, or provided via a YouTube link, and identifies which raga is being performed. Think of it as Shazam for Carnatic ragas. The system currently recognizes 40 Carnatic ragas. The original v1 model is trained on 689 samples from the CompMusic research dataset and YouTube recordings; a newer Time-Delayed Melody Surface (TDMS) k-NN model has since been built and validated against the same evaluation harness and clears v1 by roughly 14 percentage points on a fair benchmark, though it is not yet wired into the live API.

I built this project from scratch to connect two of my personal interests: Carnatic vocal music and machine learning. It is not a wrapper around a pre-existing API. I designed, trained, evaluated, and deployed the model entirely from the ground up, including independently re-implementing the published state-of-the-art feature (TDMS, Gulati et al. ISMIR 2016) and reproducing the paper's headline 86.7% accuracy on the same dataset.

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

The repository has **two parallel ML pipelines**: the original v1 (currently deployed) and the TDMS k-NN candidate built during Phase 6-9 that is not yet wired into production.

#### v1 pipeline (deployed)

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

Training uses Adam optimizer with learning rate 0.001 and weight decay 1e-4, with a step learning rate scheduler. Training runs for 200 epochs and the best checkpoint is saved. The model originally reported 84.4% accuracy on a random 80/20 split, but under a recording-aware 5-fold CV (the protocol every published baseline uses) the same recipe lands at 71.79% top-1 / 94.25% top-5 — see Phase 6 below for the honest re-baseline. The runtime audio pipeline (pyin + heuristic tonic + this trained MLP) collapses further to about 8% top-1 because of covariate shift between the expert features the MLP was trained on and the pyin features it gets at inference — see Phase 8.

For deployment, the PyTorch model is converted to a scikit-learn MLPClassifier (same architecture) to avoid the 2GB PyTorch dependency on the server.

**Inference (src/predict.py)**

For a new audio file:
1. Load audio at 16kHz using librosa, normalize amplitude
2. Extract pitch contour using pyin (probabilistic YIN algorithm), filter by voiced probability
3. Detect the tonic by folding all pitches into a single octave, evaluating the top 5 frequency candidates, and selecting the one that produces the most concentrated (peaked) pitch-class distribution
4. Apply the same three-channel feature extraction as training
5. Scale features using the saved StandardScaler
6. Run inference and return the top 5 predictions with confidence scores

#### TDMS candidate pipeline (Phase 6-9, not deployed)

**Evaluation Harness (src/eval_harness.py)**

A single shared evaluation utility used by every Phase 6+ experiment. Exposes three primitives:
- `recording_level_cv()`: stratified k-fold CV at the recording level with multi-seed averaging, top-k accuracy, and confusion matrices.
- `recording_aware_split()`: clip-level split that guarantees no recording_id leaks across train/test.
- `recording_vote_accuracy()`: aggregates per-clip softmax into per-recording predictions, mirroring what the deployed API does at inference.

Every model from v1 forward is graded against this one ruler so numbers are directly comparable.

**TDMS Feature Extraction (src/build_tdms.py, src/extract_tdms_features.py)**

The TDMS feature is a 120 × 120 joint distribution of (pitch-class at time `t`, pitch-class at time `t + τ`) built from the tonic-normalized predominant-pitch contour. Unlike v1's three concatenated 1D pitch histograms, it preserves phrase order — which note follows which note — so allied ragas that share swaras but differ in gait project to different TDMSs. Hyperparameters match Gulati et al. (ISMIR 2016) exactly: η=120 bins, τ=0.3s delay, α=0.75 power compression, σ=2 bins circular Gaussian smoothing, L1 normalization. `extract_tdms_features.py` walks all 480 CompMusic recordings and saves `X_tdms.npy` (480 × 14400). Total extraction time: 40 seconds.

**TDMS Audio Extraction (src/build_tdms_from_audio.py)**

The deployable variant. Takes a raw audio file, runs the same pyin + tonic detection pipeline that lives in predict.py, and produces a TDMS from the resulting contour. Shares one pitch extraction pass with v1's 360-D feature pipeline so A/B benchmarks compare apples to apples.

**TDMS Training and Eval (src/train_tdms.py)**

Evaluates five variants under the harness: k-NN with three distances (Frobenius, symmetric KL, Bhattacharyya), an MLP on flat TDMS, and an MLP on TDMS + 360-D concat. The symmetric-KL k-NN with leave-one-out CV hits 86.67% top-1 — exactly matching the paper. Under 5-fold CV (apples-to-apples with v1), it lands at 85.67% top-1 / 97.58% top-5.

**Audio-pipeline A/B (src/eval_audio_ab.py, src/eval_audio_expert_tonic.py, src/eval_crepe_audio.py)**

Three benchmarks isolating the gap between expert-feature TDMS (85.67%) and audio-derived TDMS:
- `eval_audio_ab.py`: pyin + heuristic tonic (the deployable today) → 37.36% top-1.
- `eval_audio_expert_tonic.py`: pyin + expert tonic → 45.81% top-1. Isolates tonic-detection contribution.
- `eval_crepe_audio.py`: CREPE + expert tonic → 51.51% top-1, 87.36% top-5. Isolates pitch-extractor contribution.

All three use the same 60s middle window, same TDMS algorithm, same 5-fold CV. Full bottleneck attribution is documented in the Phase 9 section below.

**Honest Baseline (src/baseline_v1_cv.py)**

Re-runs the v1 training recipe under the same harness (5-fold CV with 5 seeds) to produce the honest 71.79% number used as the baseline for every comparison.

**Tonic Detector (src/train_tonic_detector.py, src/extract_tonic_candidates.py)**

A learned re-ranker that scores the heuristic tonic detector's top-5 candidates and reorders them. Trained but not yet wired into `api/main.py`. Adds about +2.1 pp to tonic accuracy on the held-out set, which Phase 9a estimates is worth about +5-8 pp on raga top-1 in the deployable audio path.

### Backend API (FastAPI, Hugging Face Spaces)

The backend is a FastAPI application deployed on Hugging Face Spaces at smashgod23-raga-identifier-api.hf.space, built from `backend/Dockerfile` (python:3.11-slim + ffmpeg + libsndfile1). I originally shipped on Railway's free trial, then moved to HF Spaces when the trial ended; the Dockerfile is unchanged between them.

On startup it downloads the model, scaler, and class list from Hugging Face Hub (Smashgod23/raga-identifier) to avoid storing large files in the git repository.

Endpoints:
- GET /health: returns status and number of ragas
- GET /ragas: returns the full list of 40 ragas
- POST /predict: accepts an audio file, runs inference, saves the audio to Supabase Storage, returns top 5 predictions and a unique audio ID
- POST /predict-youtube: accepts a YouTube URL, downloads audio using yt-dlp. For videos longer than 3 minutes, it samples 3 segments from different parts of the video (at 25%, 50%, and 75% through) and averages the predictions, which avoids tuning sections and intros that would throw off the model. Returns top 5 predictions.
- POST /feedback: accepts user feedback (predicted raga, actual raga, correctness, confidence, audio filename) and stores it in the Supabase feedback table

The backend uses Supabase for storage and the feedback database. Environment variables SUPABASE_URL and SUPABASE_KEY are set as Hugging Face Spaces secrets.

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
- Backend hosting: Hugging Face Spaces (free tier, Docker SDK)
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

Railway's free tier (which I originally shipped on) had a 4GB Docker image limit and PyTorch alone is over 2GB. I solved this by converting the trained PyTorch model to a scikit-learn MLPClassifier, which has equivalent inference behavior but requires only scikit-learn as a dependency, keeping the Docker image under 1GB. After the Railway trial ended I moved to Hugging Face Spaces (free tier, Docker SDK) where the same slim-image constraint still applies and the same scikit-learn deployment artifact works unchanged.

**sklearn and numpy version mismatches**

After converting to sklearn, the deployment crashed because the local sklearn version (1.5.2) did not match the server's default (1.5.0). Pickle files are not forward or backward compatible across sklearn or numpy versions. I fixed this by pinning both sklearn and numpy to the exact versions used during training in requirements-deploy.txt.

**React hooks error**

The feedback state variables were accidentally placed outside the App component function, causing an invalid hook call. React requires all hooks to be called inside a function component. The fix was moving them inside the component body.

---

## Phase 2: Audio Dataset Expansion

After landing the 84.4% v1 model on the original 689-sample dataset, the obvious bottleneck was data. With 12 CompMusic recordings plus roughly five YouTube samples per raga, the model was overfitting to specific artists and concert acoustics. The feature-extraction pipeline worked well, but it was starving for examples.

### Acquiring the audio dataset

The CompMusic Carnatic corpus has two pieces: the pre-extracted pitch features (publicly available on Zenodo) and the raw audio recordings (not public, distributed through a formal access request). I emailed the Music Technology Group at Universitat Pompeu Fabra explaining the project, got approval, and downloaded the 16GB archive of 480 concert MP3s. Each recording has a median length of 9.75 minutes, with the longest at 67 minutes, totaling roughly 126 hours of Carnatic vocal performance.

### Strategy: clip slicing over augmentation

Two approaches could turn 480 recordings into more training examples: data augmentation (perturb the existing 480 feature vectors) or clip slicing (treat each 30-second window of the original audio as its own training sample). I had already tried augmentation back in v1 and it broke the model. Adding even small Gaussian noise to pitch class distributions destroyed their density properties, and circular shifts to simulate tonic errors hurt similar-raga discrimination because a 30-cent shift is enough to confuse ragas that differ by only one ornamented swara. Clip slicing is the structurally honest alternative: every clip is real audio extracted with the real pipeline, just aimed at a smaller window.

I went with 30-second clips on a 10-second hop, the same window length the inference pipeline uses for live recordings. That choice makes the training distribution match the inference distribution. With those parameters the 480 recordings produce around 44,000 clips, enough to push the per-raga sample count from roughly 17 to over 1,000.

### The recording_id collision bug

The first issue showed up before any clips were processed. I was using the filename as the unique recording_id for each clip, planning to use those IDs later for recording-aware train/test splitting. A code review caught that 42 recordings share song titles across different artists. For example, Sri_Madhava.mp3 appears under two separate Behag artists, and there are similar collisions in eight other ragas. Using just the filename would have let clips from different recordings share the same group label, defeating the whole point of recording-aware splitting. The fix was to use the full relative path from the audio root directory as the recording_id. That guarantees uniqueness, since two recordings in the same artist directory cannot share a filename.

### The auto-tonic verification

Before kicking off the multi-hour clip extraction, I wrote a diagnostic script that ran the existing tonic detector on five random recordings and compared its output against the expert-annotated .tonicFine values from CompMusic. Three out of five were wrong, in three different ways.

**Sahānā at -1196.3 cents (octave error).** T. Brinda's median voiced pitch sat right on the boundary where the tonic detector's octave-folding loop fails to fire. The algorithm picked 91.95 Hz when the true tonic was 183.51 Hz. Exactly one octave low, which is the canonical failure mode for any pitch-folding heuristic.

**Kāmavardani at +700.5 cents (Pa-lock).** The tanpura sustains both Sa and Pa, and the L2-peakedness scorer that picks among candidates preferred the Pa-anchored cents distribution because the singer was sitting on the fifth in the first 90 seconds. Detected tonic was 200% wrong relative to the actual Sa.

**Kāpi at +216.6 cents (Ri2-lock).** The algorithm picked the most ornamented swara during the alapana intro and called that the tonic. In Kāpi, chatusruti rishabha (216.5 cents above Sa) is a vadi swara that gets heavy treatment in the opening, so it dominates the histogram during the first 90 seconds even though Sa is the actual tonic.

The 90-second windows had a roughly 60% failure rate. On 30-second clips the failure rate would have been even higher, since shorter windows have less context for the scorer. That meant a multi-hour pipeline run extracting features anchored to the wrong tonic, and the resulting features would not have been in the same space as the existing X.npy training data.

### The architectural pivot

The fix was to treat tonic detection as a runtime-only concern. CompMusic ships expert-annotated .tonicFine values for every recording, one accurate Hz value per concert. I refactored the clip pipeline to pre-load all 480 expert tonics at startup, hard-abort if any are missing, and pass each as a tonic_override to the feature extraction function. The auto-detection code path in predict.py is unchanged, since live user audio does not have an expert annotation and still has to estimate the tonic at runtime.

The principle is simple: when ground truth exists, use ground truth. The existing X.npy training rows were built from CompMusic's own pitch and tonic annotations, so if the new audio clips also use those same expert tonics, both feature sets share the same coordinate frame. The tonic detector still needs to be good enough for inference, but the entire training data quality is no longer riding on its accuracy.

### The pipeline run

With the tonic fix in place, the pipeline ran for about 7 hours on 7 CPU workers. Pyin (the pitch estimator) is the bottleneck at roughly 4.5 seconds per 30-second clip. The MP3 seek operations are essentially free at 0.04 to 0.11 seconds per clip. The run completed with 44,071 clips, zero errors, and 704 short tail-clips skipped (recordings that ended before producing a full 30-second window from the last hop).

### Gate 1 verification

Before training on the new data I wanted concrete evidence the features actually represent the same thing as the existing X.npy rows. I wrote a verification script (src/gate1_report.py) that picks one recording per target raga that exists in both datasets, prints the top 5 pitch class peaks for each of the three feature channels, and computes cosine similarity between the X.npy vector and clips from the same recording.

For Kalyāṇi, the X.npy vector peaks at Ga (400 cents) and Pa (700 cents), the defining swaras of that raga. The mean of 17 clips from one Kalyāṇi recording matches the X.npy vector with cosine similarity 0.77 across all 360 dimensions, and individual mid-recording clips reach 0.84. Tōḍi shows the same pattern: X.npy peaks at the canonical Tōḍi profile (Ni3, Sa, komal Ga, Pa) and clip-mean cosine similarity is 0.84.

The whole 44,071-clip dataset passed sanity checks: zero NaN values, zero Inf values, no clips with an all-zero channel. Per-channel means match X.npy almost exactly (both at 1/120 = 0.000833, the expected value for normalized 120-bin histograms). Standard deviations are about 1.9 times larger in clips than in full-recording X.npy rows, which is expected, since a 30-second window covers fewer swaras than a 30-minute concert and produces a more peaked distribution.

Within-recording per-clip cosine similarity is around 0.28, only marginally higher than between-raga similarity (0.26 to 0.28). The raga signal is weak per-clip but strong in the aggregate. That tells me the model needs to combine evidence across multiple clips at inference time, which the existing multi-segment voting at inference already does.

### Class imbalance

The clip counts per raga range from 316 (Sencuruṭṭi) to 2,608 (Karaharapriya), an 8.25x ratio. This happens because some ragas are typically explored at greater length in concert, so their CompMusic recordings are longer, and a 10-second hop on a longer recording produces more clips. Trained naively, the loss surface would be dominated by the over-represented ragas.

For the PyTorch model I compute class weights as n_samples / (n_classes * bincount(y_train)) and pass them to nn.CrossEntropyLoss(weight=...). For the sklearn MLPClassifier, which does not support class_weight or sample_weight in its fit method, I oversample minority classes by random replacement up to the median train count. Both approaches give every raga roughly equal influence on the loss.

### Recording-aware splitting

A naive 80/20 train/test split would shuffle all 44,760 rows randomly and split, which leaks information across the boundary in two ways. First, two clips from the same recording with a 10-second hop overlap by 20 seconds of identical audio, so the second clip is essentially a copy of the first. Second, the X.npy row for a CompMusic recording and the 92 audio clips derived from that same recording share most of their pitch information. Either kind of leak would inflate test accuracy without reflecting how the model actually generalizes to new recordings.

I used sklearn's GroupShuffleSplit with the recording_id as the group key. Every CompMusic recording gets one group label that all of its derived clips and its X.npy row share. Each YouTube row gets its own unique group label since YouTube rows are 1:1 with recordings. The split is 80/20 at the recording level, which lands at 36,955 training rows from 551 recordings and 7,805 test rows from 138 recordings, with zero recording_id leakage across the split.

I also flipped early_stopping=False on the sklearn MLPClassifier because its default behavior carves a random 15% out of training for internal validation, and that random 15% would cut across recording boundaries.

### Final v2 results

The v2 PyTorch model finished training in 2.6 minutes (well under the 1 to 3 hour estimate, because it plateaued by epoch 50 and never improved). Final per-clip test accuracy on the recording-aware 80/20 split: **36.77%**, against the v1 baseline of 84.4% on a smaller dataset. The sklearn deployment copy scored 29.74%.

The headline number is misleading because 98% of test rows are per-clip rows. Broken down by source, the picture is more informative:

| Source | Test rows | Test accuracy |
|--------|-----------|---------------|
| compmusic (full-recording features) | 96 | 72.92% |
| youtube (full-recording features) | 42 | 16.67% |
| audio_clip (per-clip features) | 7,667 | 36.43% |

The model does close to v1 territory on full-recording features (72.92% on the compmusic-source held-out rows), but stalls on per-clip features. The Gate 1 cosine similarity result already pointed at this: within-recording per-clip similarity is only 0.28, so individual 30-second windows do not carry enough raga-specific signal for a per-clip classifier to clear the 40% mark. The model was being asked to classify three contradictory things all labeled the same raga: the Sa-establishment opening clips, the mid-recording exploration clips, and the climax clips. Loss decreased steadily but validation accuracy plateaued at 36.77% by epoch 50 and stayed there for the remaining 150 epochs. Train accuracy kept climbing to 74.78%, the canonical signature of a model running out of useful gradient on the validation set and starting to memorize.

YouTube at 16.67% on 42 test rows is the most surprising finding. Those are full-recording features built the same way as v1's YouTube data, and v1 handled them well. The class-balanced loss together with the 30-second-clip-dominated training mix pushed the model toward window-level decision boundaries that do not transfer to the full-recording shape that YouTube rows produce.

This regression is what drove the Phase 3 redesign.

---

## Phase 3: Multi-Scale Training

### The product goal that drove the redesign

The user-facing requirement was clear from the start but I had been ignoring it: the model should work on any length of audio. A 10-second mic recording from someone humming, a 30-second upload, a one-minute alapana clip, a half-hour YouTube concert link. All without the user having to think about which mode to pick or which model handles their case. Length should affect confidence in the prediction, not whether the prediction works at all.

The Phase 2 v2 model failed this requirement on principle. It was trained exclusively on 30-second windows. Anything substantially shorter or longer would either need to be padded, repeated, or chopped, and none of those preserve the pitch class distribution.

### Why the v2 per-clip approach fell short

The 36.77% per-clip plateau was diagnostic. The model was being trained on three contradictory things all labeled the same raga:

1. Intro clips, where the singer is establishing Sa with the tanpura. The pitch distribution looks like a delta at zero cents.
2. Mid-recording exploration clips, where the actual raga signature shows up.
3. Climax clips, where the singer is sitting on a vadi swara repeating phrases.

All three got the same label. A classifier asked to map all three to one output learns the union, which is the average of three different distributions, which has lower per-clip discriminative power than any of the three would on its own.

The fix is to let the model see clips at multiple lengths, so it learns "what raga X looks like" as a property that holds across scales rather than as a single window-shaped pattern.

### Ideas considered and rejected

**"Just train on more 30-second clips."** Quantity without diversity does not help. 44,000 clips of one length taught the model less than a balanced multi-scale set is expected to teach it. Doubling the count of 30-second clips would have hit the same per-clip ceiling.

**"Stitch unrelated short clips into longer ones."** Would have created synthetic data with shruti discontinuities at the stitch boundaries (since artists sing in different octaves and at different tonics) and fake transitions that do not match how real performances unfold. The model would learn to recognize the stitches, not the raga.

**"Keep v2 and rely on predict.py's three-segment averaging at inference."** This already works at inference time for audio longer than 3 minutes, where the backend samples 3 segments at 25%, 50%, and 75% through the recording and averages predictions. But the most common user case is short recordings (10 to 60 seconds), and three-segment averaging cannot help when there is only one segment to begin with.

### The chosen solution

Extract features at five window scales directly from the raw audio, sub-sample for raga and scale balance, train one model on the combined set:

| Scale | Window | Hop | Window count |
|-------|--------|-----|---------------|
| 15s   | 15s    | 5s  | 90,253        |
| 30s   | 30s    | 10s | 44,071 (existing v2 data) |
| 1min  | 60s    | 20s | 22,020        |
| 3min  | 180s   | 60s | 6,855         |
| Full recording | full | n/a | 480 (existing X.npy) |

Plus the 209 YouTube full-recording features. Cap each scale at 22,020 (the natural 1-minute count), stratify subsampling by raga with seed 42, and keep the smaller scales (full recording at 689, 3-minute at 6,855) as-is. Total combined dataset estimate: about 73,800 rows.

The model learns scale-invariance because it sees the same raga represented as a 15-second sketch, a 30-second window, a 1-minute alapana phrase, a 3-minute item section, and a full concert. The pitch class distribution is a normalized histogram, so the same underlying raga should produce a similar histogram regardless of how much audio went into computing it (with sampling noise that gets smaller as the window gets longer).

### A correction to the Phase 2 numbers

During Phase 3 planning I re-measured the audio dataset because the new scale extractors needed accurate per-recording durations to estimate runtime. The numbers were not what I had written in the Phase 2 README:

| Metric | Phase 2 README originally | Actual (re-measured) |
|--------|---------------------------|---------------------|
| Total audio | 73 hours | 126.4 hours |
| Median recording | 10 minutes | 9.75 minutes |
| Longest recording | 57 minutes | 67.1 minutes |

The Phase 2 numbers were estimates I made before I had a script to compute them. The Phase 2 section above has been corrected. The corrected total directly affects the runtime estimate for the new pipeline runs.

### Class imbalance and class_weight='balanced'

The clip counts per raga in the audio_clip data range from 316 (Sencuruṭṭi) to 2,608 (Karaharapriya), an 8.25x ratio. After the multi-scale subsampling the ratio softens because subsampling caps the high end, but it does not go away entirely. For v3 the PyTorch model uses weighted CrossEntropyLoss with weights computed as n_samples / (n_classes * np.bincount(y_train)). The sklearn MLPClassifier does not support class_weight or sample_weight directly, so the deployment copy oversamples minority classes to the median train count before fitting.

### Recording-aware splitting

Same reasoning as Phase 2. A naive 80/20 split would put 30-second clips and 60-second clips from the same recording on different sides of the train/test boundary, which is a leak in two directions: the windows overlap at the audio level, and they share the same per-recording properties (singer, tanpura tuning, microphone, room acoustics). I use sklearn's GroupShuffleSplit with the recording_id as the group key, so all features derived from one recording at all scales stay together.

### Final v3 results

All five feature scales finished extracting. The combined dataset (`src/build_multiscale_dataset.py`) lands at 72,997 rows after stratified per-scale subsampling, with recording_id preserved across every row so the train/test split stays leak-free. `src/train_v3.py` trains the same RagaNet architecture as v2 (360 in, 256/128/64 with BatchNorm and dropout, 40 out) with weighted cross-entropy on the recording-aware 80/20 split.

| Metric | v3 |
|--------|----|
| Per-row test accuracy | 38.18% |
| Per-recording vote (mean softmax across a recording's rows) | 53.62% |
| Best epoch | early plateau, never recovered |

That is below the v1 production baseline of 84.4% on full-recording features. Per-scale and per-source breakdowns showed the same shape as v2: longer windows beat shorter windows by a wide margin, and per-clip accuracy at the 15-second scale could not climb out of the 30s. The multi-scale hypothesis (that seeing the same raga at 15s, 30s, 1min, 3min, and full-recording would teach scale-invariance) did not hold in practice. The model still learned a window-shaped decision boundary that did not generalize to held-out recordings.

The structural read is that pitch-class histograms throw away phrase order. A raga is partially defined by the sequence in which characteristic phrases appear (the pakad), and a histogram is a bag of pitches with no time information at all. Two ragas that share a swara set but differ in the canonical phrase order project to nearly identical pitch class distributions, so a histogram-based classifier cannot distinguish them no matter how much data it sees. v3 is not deployed. The v1 model continues to serve traffic at smashgod23-raga-identifier-api.hf.space, though Phase 6 later showed v1's honest accuracy on a fair evaluation is 71.79%, not the 84.4% from the leaky split, and Phase 8 showed v1's actual audio-pipeline accuracy is closer to 8%. The phrase-order failure mode this paragraph names is what TDMS in Phase 7 was built to fix.

---

## Phase 4: Mel-Spectrogram CNN (Failed)

If pitch-class histograms cannot capture phrase order, the obvious next step is a representation that preserves time. Log-Mel spectrograms keep both time and frequency, and 2D CNNs were the standard architecture for audio classification before foundation models took over. The hypothesis: a CNN on log-Mel windows should clear the histogram ceiling because it can see when a phrase happens, not just which pitches were touched.

`src/preprocess_melspec.py` pre-computed log-Mel spectrograms once per recording (sr=16kHz, n_fft=2048, hop=512, n_mels=128) so training did not re-compute every epoch. Output landed at roughly 6.7 GB across 480 .npy files. `src/train_v4.py` ran a 2D CNN with 251k parameters on the same recording-aware split as v3, MPS backend on an M2 Mac, warmup plus cosine learning rate schedule.

Three runs to get there. The first attempted to mmap the full spectrogram archive and thrashed on OS file cache eviction. The second cast everything to float16, which fit in RAM but still produced an unstable training loop. The third subsampled n_mels from 128 to 64 and decimated the time axis by 2x, which finally trained cleanly in 24 minutes.

Result: per-recording vote 11.46%. Train accuracy reached 86%, validation accuracy pinned around chance the entire run. The CNN had memorized the 384 training concerts (their tanpura tuning, their microphone, their room reverb, their singer's timbre) without learning anything raga-specific that transferred to the 96 held-out recordings. With 251k parameters and 384 unique recordings, the model had roughly 650 parameters per training recording, which is more than enough capacity to over-fit to recording-level confounders.

The lesson is that the histogram approach's biggest weakness (throwing away phrase order) is also its biggest defense against this failure mode. By collapsing the time axis entirely, it makes the model blind to artist-specific, room-specific, microphone-specific signatures that the CNN happily latched onto.

---

## Phase 5: AST Embedding Classifier (Failed)

The diagnosis from Phase 4 was that from-scratch CNNs do not have the data-efficiency priors needed for 384 training recordings. Foundation models trained on millions of audio clips do have those priors, so the next attempt was MIT's Audio Spectrogram Transformer (AST), pre-trained on AudioSet. The plan: extract the 768-dim CLS embedding from every 10.24-second window of every CompMusic recording, then train a small classifier head on top.

`src/preprocess_ast_embeddings.py` ran AST on MPS at roughly 30 windows per minute and produced 88,518 embeddings from 480 recordings in 2.5 hours. `src/train_v5_ast.py` trained a small MLP head on the recording-aware split.

Result: per-recording vote 3.12%, barely above the 2.5% random-guess baseline for 40 classes. Train accuracy hit 95.88% in 50 epochs while validation pinned at chance level the entire run.

The diagnosis is more interesting than the number. AST's pre-training objective is approximately "what kind of acoustic event is this" (music vs speech vs drums vs sirens vs barking), which is approximately orthogonal to "what raga is this Carnatic vocal performance in". The 768-dim embeddings encode timbre, voice quality, and acoustic event category, all of which are recording-specific rather than raga-specific. Two different ragas performed by the same singer in the same concert hall produce embeddings that cluster together. Two performances of the same raga by different singers in different halls do not. The classifier head learned "concert N maps to embedding cluster N" perfectly, but those clusters do not transfer because the discriminating axis is not raga.

---

## What Five Model Attempts Revealed

| Version | Approach | Test accuracy |
|---------|----------|----------------|
| v1 | 360-dim PCD features + MLP, full-recording only | 84.4% per-recording (deployed, see caveat below) |
| v2 | Same features + audio clip slicing (30s windows) | 36.77% per-clip (per-recording vote not measured) |
| v3 | Multi-scale PCD features (15s/30s/1min/3min/full) + MLP | 53.62% per-recording vote |
| v4 | Log-Mel spectrograms + 2D CNN (CompMusic only) | 11.46% per-recording vote |
| v5 | AST audio embeddings + MLP head (CompMusic only) | 3.12% per-recording vote |

**Caveat on the v1 number.** The 84.4% in the table is from a single random 80/20 split of 689 mixed CompMusic+YouTube samples — the same recording's clips could legally land in both train and test under v1's split, which inflates the number. Under a clean 5-fold stratified CV (the protocol every published baseline in the field uses) the same v1 recipe lands at **71.79% ± 0.84% top-1 / 94.25% ± 0.21% top-5** on the CompMusic 480-recording set. The leaky 84.4% is what got reported originally; the 71.79% is what v1 actually delivers on a fair evaluation. The Phase 6+ work below uses this number as the honest baseline.

v1 and v3 use the full 689-recording corpus (480 CompMusic plus 209 YouTube). v4 and v5 use the CompMusic 480 only, which splits to 384 train and 96 test recordings under the 80/20 recording-aware split. The YouTube rows were excluded from v4 and v5 because both pipelines needed raw audio at fixed window sizes, and the YouTube collection had been built around full-recording features rather than archived audio.

The pattern is consistent: the more raw the input, the worse the model does. v1 wins because pitch-class distributions relative to the tonic are the correct inductive bias for this task. Folding to a single octave removes shruti as a confounder, normalizing by tonic removes key as a confounder, and the histogram itself removes singer-specific phrase choices. Everything that survives is raga signature. From-scratch CNNs and general-audio foundation models both throw that bias away and try to relearn it from raw audio without enough data.

The binding limit is 689 unique recordings, not architectural choice. With more data, a CNN or foundation-model approach could plausibly clear v1, because it would have enough signal to learn what to ignore. With 689 recordings, the only model that works is one that already knows what to ignore.

The next round of improvements has to come from somewhere other than the classifier. Two candidates: (1) better tonic detection at inference, since a wrong tonic shifts every feature and the entire prediction goes off, and (2) more training data. Both are tractable. Architectural exploration on the existing dataset is largely exhausted.

---

## Tonic Detector: A Learned Re-Ranker

The first lever was tonic detection. v1 through v3 all use the same heuristic at inference: compute the pitch class distribution under each of several tonic candidates, score each one by how peaked the resulting distribution is (the assumption being that the correct tonic produces the most concentrated PCD), and pick the best. The heuristic gets the tonic right about 47% of the time on held-out CompMusic recordings, measured against the expert .tonicFine annotations. Every wrong tonic produces a feature vector in a different space than the training data and the prediction collapses.

`src/extract_tonic_candidates.py` runs the existing heuristic on the first 60 seconds of each of the 480 CompMusic recordings (after a 10-second tanpura intro skip), generates the same top-5 candidates `predict.py._detect_tonic` would generate, and labels each by distance to the expert tonic. Each candidate is represented as 4 scalars (the candidate frequency, its rank, its peakedness score, its octave-folded position) plus the 120-bin folded PCD computed under that candidate as Sa. Output: 2,400 candidates by 124 dimensions.

`src/train_tonic_detector.py` trains a small MLP (124 to 64 to 32 to 1) with BCEWithLogitsLoss and pos_weight to handle the 1:2.4 imbalance between correct and incorrect candidates. GroupShuffleSplit on recording_id, seed 42. At inference, the model scores all five heuristic candidates and the highest scorer wins.

### The candidate-generation parity bug

The first version of `extract_tonic_candidates.py` did not exactly mirror `predict.py._detect_tonic`. It used a 60-bin histogram instead of 200 bins, no smoothing, bidirectional octave folding, and a slightly different peakedness score. The model trained on those candidates jumped tonic top-1 from 46.9% (heuristic) to 78.1% (learned re-ranker), a +31.2 percentage point lift on the held-out set. That looked too good.

It was. The model had been trained on candidates that production would never see. The "perfect re-ranker ceiling" (top-1 if the model always picked the correct candidate when it was present in the top-5) was 85.4% in the buggy extraction. After rewriting `_candidates_from_voiced` to match `_detect_tonic` exactly (200-bin histogram, `uniform_filter1d(size=5)` smoothing, top-5 from the smoothed histogram, upward-only octave fold, sum-of-squares peakedness), the ceiling dropped to 60.4%. The honest re-extraction and retraining landed at:

| Stage | Tonic top-1 |
|-------|-------------|
| Heuristic baseline (production parity) | 50.0% |
| Perfect re-ranker ceiling | 60.4% |
| Learned re-ranker | 52.1% |

So the real lift is +2.1 percentage points, not +31.2. The model captures about a fifth of the available headroom on top of the heuristic. The bigger finding is that the heuristic's candidate generation only contains the correct tonic 60.4% of the time, because the upward-only octave fold drops candidates an octave below the actual tonic. A genuinely better tonic detector would change the candidate generation step itself, not just re-rank what is already there. The next iteration is to make the fold bidirectional in production, retrain the re-ranker against that, and re-measure.

The training run did show one healthy signal: the train/test gap was 85.2% vs 78.1% in the buggy run, and roughly 7 points in the corrected run, which is the right shape for a model that is learning a real pattern rather than memorizing. Compare with v4 and v5, both of which hit 86% to 96% train accuracy with chance-level validation. The tonic detector is the right size for the data: too small to memorize 384 recordings, big enough to learn what a real raga PCD looks like under a candidate Sa.

The learned tonic detector is not yet wired into the production inference path. The integration plan is to load `models/tonic_detector_v1.pt` alongside the existing model and scaler in `api/main.py`, replace the heuristic argmax inside `_detect_tonic` with a forward pass of the re-ranker over the five candidates, and re-measure raga accuracy end to end. Expected lift on the live system is modest given the +2.1 pp tonic improvement, but the larger value is the diagnostic: it pins down where the actual bottleneck is.

---

## Phase 6: An Honest Re-baseline of v1

Months after v1 shipped, I went back and built a real evaluation harness. The 84.4% number that was originally reported in this README came from a single random 80/20 split of 689 mixed CompMusic+YouTube samples. Because the split didn't enforce recording-aware boundaries, the same recording's training and testing data could legally end up on opposite sides of the split. With a recording-aware 5-fold stratified CV protocol — the same protocol every published baseline in the field uses — the same v1 recipe lands at **71.79% ± 0.84% top-1 / 94.25% ± 0.21% top-5** on the CompMusic 480-recording set.

That gap (84.4 → 71.79) is the "leaky-split tax." I'm leaving the 84.4% in the "Five Model Attempts" table above because it's what was originally reported, but the 71.79% is what v1 actually delivers on a fair evaluation, and everything that follows uses this as the honest baseline.

The eval harness lives at `backend/src/eval_harness.py` and exposes three primitives: recording-level stratified k-fold CV, recording-aware splits (with no clip leakage across train/test), and clip-to-recording vote aggregation that mirrors what the deployed API does at inference. Every experiment from Phase 6 onward gets graded against this single ruler. The script that produced the 71.79% number is `backend/src/baseline_v1_cv.py`.

The top confusion pairs in v1's honest evaluation are textbook allied-raga errors: Kāṁbhōji ↔ Harikāmbhōji (32 misclassifications), Śrī → Madhyamāvati (24), Hussēnī → Mukhāri (15), Bhairavi → Mukhāri (14). These are pairs that share the same set of swaras but differ in the gait (calan) and characteristic phrases (pakad). v1's pitch-class histograms throw that gait signal away — exactly the v3 retrospective.

---

## Phase 7: TDMS Replicates the Paper at 86.67%

After the honest re-baseline, I wanted a feature that could capture phrase order rather than discarding it. The right reference turned out to be **Gulati, Serrà, Ganguli, Şentürk, Serra — "Time-Delayed Melody Surfaces for Rāga Recognition" (ISMIR 2016)**, which runs on the exact CompMusic 480-recording 40-raga dataset I have and reports 86.7% top-1 with a k-nearest-neighbor classifier. The lab is Xavier Serra's MTG at UPF Barcelona — the same group whose CompMusic project I'm sourcing all my data from, and Kaustuv Kanti Ganguli (a co-author) is one of the researchers Meinard Müller pointed me at when I wrote to him.

A TDMS is a 120 × 120 joint distribution of (pitch-class at time `t`, pitch-class at time `t + τ`) built from the tonic-normalized predominant-pitch contour. Where v1's 360-D feature is three 1D pitch-class histograms, a TDMS is a single 2D matrix that encodes "which note follows which note." Allied ragas that share swaras but differ in gait project to different TDMSs, even though their 1D pitch histograms look nearly identical.

I implemented TDMS from the paper's exact spec. `backend/src/build_tdms.py` is roughly 140 lines of NumPy + scipy.ndimage:

| Hyperparameter | Value (paper) | Value (this implementation) |
|---|---|---|
| Bins per dimension (η) | 120 (10 cents/bin) | 120 |
| Time delay (τ) | 0.3 seconds | 0.3 |
| Power compression (α) | 0.75 | 0.75 |
| Gaussian smoothing (σ) | 2 bins, circular convolution | 2, scipy `mode='wrap'` |
| Normalization | L1 (the matrix sums to 1) | L1 |
| Classifier | 1-NN with symmetric KL or Bhattacharyya distance | identical |

I ran the full 480 recordings through `backend/src/extract_tdms_features.py` using CompMusic's expert pitch contours (`.pitchSilIntrpPP`) and expert tonics (`.tonicFine`), producing `data/X_tdms.npy` (480 × 14400 float32). Total extraction time on M-series: 40 seconds.

`backend/src/train_tdms.py` then evaluates five variants under the Phase-6 harness:

| Variant | 5-fold CV top-1 | 5-fold CV top-5 | Leave-one-out top-1 |
|---|---|---|---|
| v1 baseline (MLP on 360-D PCD) | 71.79% ± 0.84 | 94.25% ± 0.21 | — |
| k-NN Frobenius (M_F) | 81.79% ± 0.47 | 96.88% ± 0.23 | 82.50% |
| k-NN symmetric KL (M_KL) | **85.67% ± 0.40** | **97.58% ± 0.21** | **86.67%** |
| k-NN Bhattacharyya (M_B) | 85.63% ± 0.42 | 97.58% ± 0.21 | 86.67% |
| MLP on flat TDMS | 75.67% ± 1.08 | 92.17% ± 0.60 | — |
| MLP on TDMS + 360-D concat | 77.46% ± 1.73 | 94.33% ± 0.96 | — |

**The leave-one-out 86.67% with symmetric KL exactly matches the paper's 86.7%.** Independent reproduction on the same data with the same algorithm, to within 0.03 percentage points. Under the 5-fold CV comparison to v1, TDMS gains **+13.88 pp top-1** (71.79 → 85.67) and **+3.33 pp top-5** (94.25 → 97.58).

The 1-NN beats both MLPs because 480 samples and 14400 features overfit any neural network without strong regularization. The matrix-aware distances (symmetric KL and Bhattacharyya are proper divergences for probability matrices) respect the joint-distribution geometry that MLP weights blow up. The paper found the same thing.

This was the moment I thought the problem was solved. It was not.

---

## Phase 8: The Audio-Feature Gap

Phase 7 worked on expert features. The deployed API does not have expert features. It runs `librosa.pyin` on the user's audio, detects a tonic with `predict.py._detect_tonic`'s heuristic, and computes a 360-D PCD vector from the resulting contour. The trained model has never seen pyin-derived features — it was trained on CompMusic's `.pitchSilIntrpPP` files, which use Salamon and Gómez's predominant-melody method specifically tuned for polyphonic Indian classical music.

So I built `backend/src/eval_audio_ab.py`, which extracts a 60-second middle window from every CompMusic mp3 (the audio is at `~/raga-data-audio/`, 7.5 GB total), runs the production pipeline (pyin + heuristic tonic) once per recording with a 6-worker `ProcessPoolExecutor`, and from that single pitch contour computes both v1's 360-D feature and a 120 × 120 TDMS. Then it runs four 5-fold-CV evaluations:

| Configuration | 5-fold CV top-1 | 5-fold CV top-5 |
|---|---|---|
| Phase 7 reference: TDMS on EXPERT pitch + EXPERT tonic | 85.67% ± 0.40 | 97.58% ± 0.21 |
| Phase 6 reference: v1 on EXPERT pitch | 71.79% ± 0.84 | 94.25% ± 0.21 |
| **TDMS on AUDIO pitch + heuristic tonic** | **37.36% ± 1.00** | **67.91% ± 0.81** |
| **v1's MLP on AUDIO 360-D features** | **8.32% ± 0.93** | **28.57% ± 1.60** |
| Train EXPERT templates, query AUDIO TDMS | 23.56% ± 0.21 | 50.76% ± 0.28 |
| Train AUDIO templates, query EXPERT TDMS | 17.20% ± 0.62 | 48.87% ± 1.03 |

Three findings landed:

**1. The deployed v1 model is near random on real audio.** Under the runtime pipeline (pyin + heuristic tonic + 360-D PCDs), v1's accuracy collapses from 71.79% to 8.32% top-1 — about 3× above the 2.5% chance floor for 40 classes. The 84.4% number the live site is implicitly making claims with bears no relationship to what users are actually getting. v1's nyas/duration/stable PCD channels were carved out using CompMusic's hand-annotated stable-note regions (`.flatSegNyas`) and a 4.44 ms hop expert pitch. When the runtime stack hands it a noisy pyin contour at 16 ms hop with no nyas annotation, the channels distribute differently and the trained MLP doesn't recognize them. This is a textbook covariate shift between training and inference.

**2. TDMS on audio is 4.5× better than v1 on audio.** Same audio, same window, same pyin extraction — only the feature changes. TDMS lands at 37.36% top-1 / 67.91% top-5, vs v1 at 8.32% / 28.57%. The joint pitch-class distribution is dramatically more robust to pitch-extraction noise than concatenated 1D PCDs. Gaussian smoothing on a 14400-cell surface averages noise out. The matrix-aware distances preserve the *relative* ordering of training matrices even when the absolute values drift. v1's MLP weights have no such tolerance.

**3. Cross-source templates do not transfer.** Training on the Phase 7 expert templates and querying with audio TDMSs only hits 23.56% top-1. The reverse (train audio, test expert) is 17.20%. For deployment we cannot reuse the Phase 7 index — templates have to be rebuilt from audio using the same pyin pipeline the user's queries will run through.

So the deployable candidate now is "TDMS on audio + 1-NN with symmetric KL, templates rebuilt from audio." That clears v1 in production by 4.5×, but 37.36% top-1 is still far from the 86.67% paper-replication ceiling. Two phases of bottleneck isolation followed.

---

## Phase 9: Isolating the Bottleneck (Tonic vs Pitch vs Window)

The gap from 86.67% (expert templates) to 37.36% (audio templates) has three possible sources: tonic quality, pitch quality, and window length. Phase 9 runs a controlled experiment to attribute the gap.

### Phase 9a — Audio pitch + EXPERT tonic

`backend/src/eval_audio_expert_tonic.py` reruns Phase 8's audio extraction but substitutes each recording's `.tonicFine` for the heuristic-detected tonic. Same 60s middle window, same pyin pitch, same TDMS algorithm, same eval harness. The only thing that changes is the tonic.

Result: **45.81% ± 1.55% top-1, 82.93% ± 0.32% top-5.**

So the tonic alone accounts for **+8.45 pp top-1** and **+15.02 pp top-5**. That's a meaningful contribution but it is not the dominant bottleneck. The remaining gap to the 86.67% ceiling is ~40 pp, of which only about a quarter is tonic. The top-5 number is striking, though — at 82.93%, the audio pipeline with a correct tonic puts the right raga in the top 5 most of the time. It's the top-1 disambiguation that fails.

This validates wiring the existing `tonic_detector_v1.pt` re-ranker (+2.1 pp from the Tonic Detector section above) into production — it's the cheapest deployable improvement — but it also says that tonic detection is not the principal failure mode.

### Phase 9b — CREPE pitch + EXPERT tonic

The remaining gap is split between pitch quality and window length. Window length is easy to scale up but expensive (longer audio = more pyin time per query, which has UX implications at inference). Pitch quality is the harder lever, but if the published SOTA is right about CREPE, switching the extractor could close most of the gap with a fixed-cost change.

The reference for this is **Vishwaas Narasinh Senthil Raja — "Sequential Pitch Distributions for Raga Detection" (AIMC 2023)**, which beats TDMS on CMD (88.13% vs 86.7%) using an extended SPD feature and explicitly recommends CREPE (Kim, Salamon, Li, Bello — ICASSP 2018) as the runtime pitch extractor. CREPE is a CNN-based monophonic pitch estimator trained on a large dataset of human voice and musical instrument pitches; it's substantially more accurate than pyin on noisy, monophonically-extracted melodies from polyphonic recordings.

`backend/src/eval_crepe_audio.py` mirrors Phase 9a but swaps `librosa.pyin` for `crepe.predict` with `model_capacity='small'` and 30 ms step size (matching the SPD paper). Expert tonic. Same 60s middle window. Same TDMS algorithm.

Result: **51.51% ± 1.10% top-1, 87.36% ± 0.43% top-5.**

CREPE buys **+5.70 pp top-1 and +4.43 pp top-5** over pyin on identical audio with identical tonic. A real improvement, but smaller than I hoped — pyin is not as bad as I expected at this granularity. The full ladder of bottleneck attribution:

| Step | Pitch | Tonic | Window | Top-1 | Δ |
|---|---|---|---|---|---|
| Phase 1 ceiling | expert | expert | full recording | 85.67% | — |
| Phase 9b | CREPE small | expert | 60s | 51.51% | −34.16 pp |
| Phase 9a | pyin | expert | 60s | 45.81% | −5.70 pp (vs CREPE) |
| Phase 8 | pyin | heuristic | 60s | 37.36% | −8.45 pp (vs expert tonic) |
| v1 deployed (audio) | pyin | heuristic | 60s | 8.32% | feature change to v1 360-D |

So the 34-pp gap from CREPE-60s to the Phase 1 ceiling decomposes roughly as:
- ~6 pp from pitch quality (CREPE vs full Salamon–Gómez plus CompMusic's post-processing)
- ~?? pp from 60s vs full-recording (the dominant remaining factor — TDMS gets statistically denser with more pitch frames)
- ~?? pp from miscellaneous (singer-tonic interaction, voicing differences, RMS normalization)

The clear lesson is that **no single audio-side substitution closes the gap to the expert-feature ceiling.** Tonic detector + CREPE + expert tonic, even stacked, is unlikely to reach 70% on 60s clips. The most leveraged remaining experiment is window length — extracting from a 3-minute window instead of 60s, or aggregating TDMSs across multiple windows of the same recording. This is Phase 10 work.

Top-5 at 87.36% is the surprising number, though. The audio pipeline with CREPE puts the right raga in the top 5 *87% of the time*. That's a deployable "did you mean?" UX: show 5 candidates with a short audio sample of each, let the user pick. The model doesn't have to be right on top-1 if the interaction surface absorbs the uncertainty.

### Phase 10a — Long templates, short queries

Phase 9 left ~20 pp of the audio gap on the table, attributed to window length. Phase 10a tests the "asymmetric template" hypothesis: build TDMS templates from a much longer window of each recording (5 minutes centered at the midpoint), then accept a short 60s query from the user side. The training-test split stays recording-aware, so a held-out recording's audio never reaches the model in any form, but within each recording the template is built from a denser pitch contour than the query.

`backend/src/eval_long_templates.py` extracts 5-minute templates for all 480 CompMusic recordings with pyin + expert tonic, then evaluates the same 1-NN symmetric-KL classifier against the existing Phase 9a 60s query set. The symmetric-vs-asymmetric comparison shares everything except template length, so the delta is directly the window-length contribution.

Result: **55.10% ± 0.94% top-1, 88.74% ± 0.77% top-5.**

Compared to Phase 9a's 60s+60s baseline (45.81%, 82.93%), Phase 10a buys **+9.29 pp top-1 and +5.82 pp top-5** from a single change in template length. Two surprising things:

1. The lift is large for what looks like a small intervention — a 5× longer template moves accuracy almost as much as adding the expert tonic did in the first place.
2. Phase 10a (pyin + 5-min templates) beats Phase 9b (CREPE + 60s templates) by 3.59 pp top-1. A longer template more than compensates for a worse pitch extractor. The audio bottleneck is sparsity, not noise.

Full ladder including Phase 10a:

| Step | Pitch | Tonic | Template window | Query window | Top-1 | Top-5 |
|---|---|---|---|---|---|---|
| Phase 1 ceiling | expert | expert | full recording | (same) | 85.67% | 97.58% |
| **Phase 10a** | **pyin** | **expert** | **5 min** | **60s** | **55.10%** | **88.74%** |
| Phase 9b | CREPE small | expert | 60s | 60s | 51.51% | 87.36% |
| Phase 9a | pyin | expert | 60s | 60s | 45.81% | 82.93% |
| Phase 8 | pyin | heuristic | 60s | 60s | 37.36% | 67.91% |
| v1 deployed | pyin | heuristic | 60s (PCD feature, not TDMS) | 60s | 8.32% | 28.57% |

The deployable recipe now writes itself: build long templates once (this is a fixed corpus cost, runs offline, ships as a fixed asset on Hugging Face Hub), serve short user queries with the same pyin + tonic detector + TDMS extraction at inference. The remaining 30 pp gap to the Phase 1 ceiling is split between pitch quality (CREPE on long templates would close part of it) and any audio still left on the table beyond 5 minutes.

### Phase 10b — Multi-window query aggregation

Phase 10a kept the user-side query at a single 60s window. Phase 10b tests whether averaging three 60s windows (extracted at 25%, 50%, and 75% through the user's audio) gives the model a more reliable view of the raga without changing the template side at all. Same long templates from Phase 10a, same pyin + expert tonic for both sides, same 5-fold CV recording-aware harness — only the query construction changes.

`backend/src/eval_multiwindow_queries.py` extracts three 60s TDMSs per recording, averages them (then re-L1-normalizes for floating-point safety), and runs the asymmetric kNN-symKL evaluation against Phase 10a's long templates.

Result: **73.23% ± 0.73% top-1, 95.31% ± 0.16% top-5.**

Compared to Phase 10a's single-window queries (55.10%, 88.74%), Phase 10b buys **+18.13 pp top-1 and +6.57 pp top-5**. This is bigger than any single change we've made since switching from v1's PCD to TDMS in Phase 7. Combined with Phase 10a, the asymmetric template + multi-window query recipe closes about 64% of the audio-pipeline gap to the Phase 1 expert-pitch ceiling.

The mechanism is clear in retrospect: a single 60s window of TDMS is a sparse distribution of ~2,000 (pitch, pitch-after-tau) pairs spread across 14,400 cells. Most cells are zero before Gaussian smoothing. Averaging three independent samples of that distribution doesn't just denoise — it triples the effective number of pairs the user-side TDMS reflects, pushing the query's statistical reliability closer to the long template's.

Full ladder including Phase 10b:

| Step | Pitch | Tonic | Template window | Query window | Top-1 | Top-5 |
|---|---|---|---|---|---|---|
| Phase 1 ceiling | expert | expert | full recording | (same) | 85.67% | 97.58% |
| **Phase 10b** | **pyin** | **expert** | **5 min** | **3 × 60s avg** | **73.23%** | **95.31%** |
| Phase 10a | pyin | expert | 5 min | 60s | 55.10% | 88.74% |
| Phase 9b | CREPE small | expert | 60s | 60s | 51.51% | 87.36% |
| Phase 9a | pyin | expert | 60s | 60s | 45.81% | 82.93% |
| Phase 8 | pyin | heuristic | 60s | 60s | 37.36% | 67.91% |
| v1 deployed | pyin | heuristic | 60s (PCD feature) | 60s | 8.32% | 28.57% |

What this means for shipping: a deployable pipeline now sits at **9× v1's actual production top-1 and ~3× its top-5**, using nothing more than pyin (already installed), the existing tonic heuristic, a one-time 33-minute template build, and a 3× larger inference cost (~30 seconds of extra latency for a 1-2 minute user upload, on CPU). The "did you mean? here are five" UX would be effectively solved at 95.31% top-5.

The gap to the 85.67% Phase 1 ceiling — about 12 pp top-1 — is now plausibly closable in a few more steps: CREPE on the templates (templates are built offline so CREPE's slowness doesn't hurt at inference), wiring the existing tonic detector to replace the heuristic, and possibly extending templates beyond 5 minutes toward the full recording length.

### Phase 11a — Production-faithful eval (heuristic tonic)

Phase 10b used the expert tonic from `.tonicFine` on both sides of the eval. Real user audio doesn't have those annotations — the production endpoint has to run the heuristic tonic detector from `predict.py._detect_tonic`. To get the number production will actually serve, `backend/src/eval_multiwindow_heuristic.py` keeps everything from Phase 10b but swaps the query-side tonic for the heuristic.

Result: **35.83% ± 0.81% top-1, 63.67% ± 0.60% top-5.**

This is sobering. The heuristic tonic destroys almost all the gains from longer templates and multi-window aggregation. Phase 8 (60s + 60s + heuristic tonic) landed at 37.36% / 67.91% — slightly higher than Phase 11a on top-1 because the noise-from-tonic-errors swamps the noise-reduction-from-multi-window-averaging at this tonic accuracy.

The takeaway: **the tonic detector is the dominant bottleneck in the audio pipeline**, not pitch quality or window length. The expert tonic was doing the heavy lifting in Phase 10b's 73.23%. Phase 11a is the actual production accuracy for the just-shipped `/predict-tdms` endpoint until the tonic story is fixed.

| Step | Pitch | Tonic | Template | Query | Top-1 | Top-5 |
|---|---|---|---|---|---|---|
| Phase 1 ceiling | expert | expert | full rec | (same) | 85.67% | 97.58% |
| Phase 10b | pyin | expert | 5 min | 3 × 60s avg | 73.23% | 95.31% |
| Phase 10a | pyin | expert | 5 min | 60s | 55.10% | 88.74% |
| Phase 9b | CREPE | expert | 60s | 60s | 51.51% | 87.36% |
| Phase 9a | pyin | expert | 60s | 60s | 45.81% | 82.93% |
| Phase 8 | pyin | heuristic | 60s | 60s | 37.36% | 67.91% |
| **Phase 11a (production)** | **pyin** | **heuristic** | **5 min** | **3 × 60s avg** | **35.83%** | **63.67%** |
| v1 deployed | pyin | heuristic | 60s (PCD) | 60s | 8.32% | 28.57% |

Phase 11a is still **4.3× v1's actual deployed top-1 and 2.2× its top-5**, but nowhere near the 73% we hoped for. The just-shipped `/predict-tdms` endpoint is serving the Phase 11a recipe and should land at ~36% top-1 for real users.

### Phase 11b — Does the learned tonic re-ranker help end-to-end?

Phase 11a flagged tonic as the dominant bottleneck. The Tonic Detector section above documents an existing `tonic_detector_v1.pt` re-ranker trained months earlier — +2.1 pp on tonic accuracy on its own training/test split. Phase 11b plugs that re-ranker into the production multi-window pipeline (`backend/src/eval_multiwindow_reranker.py`) and re-measures end-to-end.

Result: **34.08% ± 0.39% top-1, 61.25% ± 0.37% top-5.**

A regression of **−1.75 pp top-1 and −2.42 pp top-5** vs Phase 11a. The training-time lift does not transfer.

The reason traces back to a sampling-offset mismatch. The re-ranker was trained on candidates extracted from the first 60 seconds of each recording (offset=10s, skipping the tanpura intro). Phase 11b queries from three 60s windows at 25/50/75% through the recording. The candidates the re-ranker scores at production time look statistically different from the ones it was trained on, and the +2.1 pp lift it picked up on the original test set does not survive the distribution shift.

So the re-ranker as it exists today is shelved. Wiring it would actively make things worse. If we want it back, the right fix is to re-extract candidates at the production-equivalent offsets, retrain, and re-evaluate.

| Step | Pitch | Tonic | Template | Query | Top-1 | Top-5 |
|---|---|---|---|---|---|---|
| Phase 1 ceiling | expert | expert | full rec | (same) | 85.67% | 97.58% |
| Phase 10b | pyin | expert | 5 min | 3 × 60s avg | 73.23% | 95.31% |
| **Phase 11a (production)** | **pyin** | **heuristic** | **5 min** | **3 × 60s avg** | **35.83%** | **63.67%** |
| Phase 11b | pyin | re-ranker | 5 min | 3 × 60s avg | 34.08% | 61.25% |
| Phase 8 | pyin | heuristic | 60s | 60s | 37.36% | 67.91% |
| v1 deployed | pyin | heuristic | 60s (PCD) | 60s | 8.32% | 28.57% |

### Phase 12 — The tonic scorer was the real bottleneck (sa_pa drone, +5.5 pp shipped)

Phase 11b shelved the re-ranker, and earlier notes (including an earlier version of this README) claimed the next lever was a **bidirectional octave fold** in candidate generation. Phase 12 ran the experiment and found that claim was simply **wrong**, then found the actual lever.

**Why the octave fold is a non-lever.** Tonic correctness is judged octave-agnostically (the cent difference to the expert tonic is wrapped mod 1200), and — more fundamentally — the downstream TDMS feature is itself octave-folded (`cents mod 1200`). A tonic that is off by an exact octave produces a byte-identical TDMS. So the octave a candidate lands in is irrelevant to both the tonic metric and the raga prediction; only its pitch class matters. `backend/src/eval_tonic_candidates.py` confirmed it directly: fold strategies `up`, `none`, and `bidirectional` give identical ceilings at every candidate count K. The fold cannot move accuracy.

**What actually limits the ceiling: candidate count.** The same diagnostic swept K (octave-agnostic, 60s window):

| K | Ceiling (true Sa pitch-class in top-K) | Peakedness top-1 |
|---|---|---|
| 5 | 62.7% | 51.9% |
| 8 | 79.0% | 51.9% |
| 10 | 83.5% | 51.9% |
| 15 | 89.8% | 51.9% |

More candidates raises the *reachable* ceiling enormously, but the old peakedness scorer is pinned at 51.9% no matter how many candidates it is given. **The scorer, not the candidate pool, was the bottleneck.**

**The fix: score on the tanpura drone (Phase 12b).** The tanpura drones Sa and its fifth Pa continuously throughout every Carnatic performance. So the true Sa uniquely has strong energy at *both* 0 cents and +700 cents in its tonic-relative pitch-class distribution. A wrong tonic (e.g. Pa-lock) does not satisfy both. Replacing the peakedness scorer with this `sa_pa` drone score (`backend/src/eval_tonic_scorers.py`, no model, no retraining):

| Scorer | K=5 top-1 | K=10 top-1 |
|---|---|---|
| peakedness (old production) | 51.9% | 51.9% |
| sa (Sa energy only) | 53.5% | 52.7% |
| **sa_pa (Sa + Pa drone)** | **57.9%** | **65.0%** |

`sa_pa` at K=10 hits **65.0% tonic top-1 — +13.1 pp** over production, with a one-function change and zero new dependencies.

**End-to-end confirmation (Phase 12c).** Wiring `sa_pa` + K=10 into `predict._detect_tonic` and re-running the full production multi-window raga pipeline:

| Step | Tonic | Top-1 | Top-5 |
|---|---|---|---|
| Phase 10b | expert (perfect) | 73.23% | 95.31% |
| **Phase 12c (shipped)** | **sa_pa, K=10** | **41.29%** | **69.58%** |
| Phase 11a | peakedness, K=5 | 35.83% | 63.67% |
| v1 deployed | peakedness | 8.32% | 28.57% |

The tonic gain translated to **+5.46 pp top-1 / +5.91 pp top-5** end-to-end. The `/predict-tdms` endpoint now serves ~41% top-1 / ~70% top-5 — 5× v1's deployed top-1 — and the same `_detect_tonic` improvement flows into the v1 `/predict` endpoint too. This shipped in `predict.py`.

### Phase 13 — Run the expert extractors at inference (75.62%, the breakthrough)

Through Phase 12 I'd concluded the deployable pipeline was capped near the pyin pitch ceiling (73.23% with a perfect tonic) and that 75% was at the real-audio frontier. That conclusion was wrong, and the reason it was wrong is instructive: I had been *reimplementing* pitch and tonic detection by hand (pyin, the sa_pa drone scorer) instead of running the **actual expert algorithms** the dataset was built with.

The CompMusic `.pitch` files were extracted with **Melodia (Salamon-Gómez predominant melody)** and the `.tonicFine` tonics with **Gulati's multipitch method**. Both ship in [Essentia](https://essentia.upf.edu/) as `PredominantPitchMelodia` and `TonicIndianArtMusic`. There is no reason they can only run offline — they run on arbitrary uploaded audio in real time. Running them at inference closes the pitch *and* tonic gaps that pyin/sa_pa could not.

**Tonic, validated first (`backend/src/eval_essentia_tonic.py`).** TonicIndianArtMusic on a 60s window, octave-agnostic vs `.tonicFine`:

| Tonic detector | Top-1 |
|---|---|
| peakedness (old prod) | 51.9% |
| sa_pa exact, K=15 (Phase 12d) | 70.2% |
| **Essentia TonicIndianArtMusic** | **85.4%** |

85.4% on 480 recordings, zero failures — +15 pp over the best thing I built by hand, matching the ~85-90% the literature reports.

**Full pipeline, swept to the target (`eval_essentia_pipeline.py`, `eval_essentia_full.py`, `eval_essentia_5win.py`).** Melodia pitch on both sides (templates and queries share an extractor — Phase 8 proved cross-extractor mismatch is fatal), expert `.tonicFine` for the offline templates, Essentia tonic at query time:

| Config | Top-1 | Top-5 |
|---|---|---|
| Melodia 5-min templates + 3×60s queries (Phase 13b) | 62.29% | 86.92% |
| Melodia full-recording templates + 3×60s queries (13c) | 70.00% | 86.58% |
| full templates + 5×60s queries (13d) | 73.62% | 91.21% |
| **full templates + 7×90s queries (13e, shipped)** | **75.62%** | **91.17%** |
| expert `.pitch` + expert tonic ceiling (Phase 1) | 85.67% | 97.58% |

**75.62% top-1 / 91.17% top-5** clears the 75% target — 9× v1's actual deployed top-1 (8.32%), and within 10 pp of the absolute expert-feature ceiling. The two production levers that mattered: the expert extractors (Melodia + Gulati tonic), and query density (more/longer windows averaged before the 1-NN).

**Honest caveat on the 75.62%.** It uses 7×90s = 630s of query audio, so it holds for full-concert-length input (a YouTube link, a 10-minute recording). A short 2-3 minute upload can only supply a few independent windows and lands closer to the 70% (3-window) number. Still far above v1 at any input length. The `compute_query_tdms` path degrades gracefully — short clips just use fewer windows.

**Shipped (`backend/src/predict_essentia.py`).** `/predict-tdms` now runs the Essentia path: one TonicIndianArtMusic detection on a 180s middle window, Melodia-pitch TDMS from up to seven 90s windows averaged, 1-NN symmetric KL against the full-recording template index (`X_tdms_essfull_template.npy`, built offline with Melodia + expert tonic). `essentia==2.1b6.dev1389` added to `requirements-deploy.txt`; it has a `manylinux2014_x86_64` cp311 wheel so the HF Spaces Docker build installs it cleanly. End-to-end module test: ~18s per full recording, correct top-1 on the spot-check.

### What's left after Phase 13

1. **Point the frontend at `/predict-tdms`.** The live frontend still calls v1 `/predict` (~8% real-audio top-1). Switching the frontend's API path to `/predict-tdms` is the single highest-impact remaining change — it puts the 70-75% model in front of users. Verify the endpoint on HF Spaces first.
2. **Build templates with Essentia tonic too** (currently expert `.tonicFine`). Would let the template index regenerate fully from audio with no dataset labels — useful when expanding beyond the 40 CMD ragas.
3. **DeepSRGM-style bi-LSTM** on Melodia contours. ISMIR 2019 reports 88.1%; now that pitch/tonic are expert-grade, a learned sequence model is the path toward the 85%+ ceiling.
4. **More ragas.** The 40-raga CMD set is the current universe; expanding needs new template audio + tonics (Essentia tonic makes this cheap).

What's explicitly NOT a good lever based on Phase 4-13 evidence: bigger from-scratch CNNs, general-audio foundation model embeddings, data augmentation on the 689 recordings, octave-fold changes, or hand-reimplementing pitch/tonic detection when Essentia's expert algorithms run fine at inference.

---

## Accuracy and Model Performance

| Metric | Value |
|---|---|
| Number of ragas | 40 |
| Training samples | 689 (480 CompMusic + 209 YouTube) |
| Feature dimensions | 360 (v1 deployed), 14,400 (TDMS candidate) |
| Baseline (random guessing) | 2.5% |

The honest accuracy story, ordered from the most generous evaluation protocol to the most realistic:

| Setup | Top-1 | Top-5 |
|---|---|---|
| v1, leaky random 80/20 split (the original "84.4%") | 84.4% | — |
| v1 baseline (MLP on expert 360-D PCD, 5-fold CV) | 71.79% | 94.25% |
| TDMS k-NN sym KL (expert pitch + tonic, leave-one-out) — paper ceiling | 86.67% | 97.71% |
| TDMS k-NN sym KL (expert pitch + tonic, 5-fold CV) | 85.67% | 97.58% |
| **TDMS k-NN (Essentia Melodia + Gulati tonic, full templates + 7×90s queries — `/predict-tdms` production, Phase 13)** | **75.62%** | **91.17%** |
| TDMS k-NN (Essentia, full templates + 5×60s queries) | 73.62% | 91.21% |
| TDMS k-NN (Essentia, 5-min templates + 3×60s queries) | 62.29% | 86.92% |
| TDMS k-NN (pyin + sa_pa tonic, 5-min template + 3×60s — Phase 12) | 42.25% | 70.42% |
| TDMS k-NN (pyin + peakedness tonic, 5-min + 3×60s — Phase 11a) | 35.83% | 63.67% |
| v1 deployed pipeline (pyin + heuristic tonic, 60s clip — `/predict`) | 8.32% | 28.57% |

(All non-ceiling rows are real-audio deployable pipelines on the 480-recording CMD set under 5-fold recording-aware CV — no expert annotations at query time. The 86.67%/85.67% rows use expert `.pitch`+`.tonicFine` and are the unreachable ceiling, shown for reference. The 75.62% production row uses 7×90s queries, so it reflects full-concert-length input; short uploads land nearer 70%.)

Two models exist side by side. The original v1 (`/predict`) is ~8% top-1 on real audio — its confidence scores read as "what the model thinks," not "how often it's right." The Phase 13 Essentia model (`/predict-tdms`) is 75.62% top-1 / 91.17% top-5 on full-length input and is the one worth using. The hardest cases for both, even with expert features, are allied ragas that share a swara set and differ only in gait/phrasing (Kāṁbhōji/Harikāmbhōji, Mōhanaṁ/Bilahari) — exactly what the TDMS feature's phrase-order encoding was built to separate.

---

## Human Feedback Loop

Every time someone uses the app and submits feedback, the following data is stored in Supabase:
- The predicted raga
- The actual raga (if the user corrects it)
- Whether the prediction was correct
- The confidence score
- The audio file ID (the audio itself is stored in Supabase Storage)

When enough feedback accumulates (target: 50+ corrections per raga), I plan to download the corrected audio files, extract features from them, add them to the training set, and retrain the model. This creates a loop where real-world usage directly improves the model over time.

The original v1 sklearn model is archived on Hugging Face as `raga_sklearn_v1_84pct.pkl` (the filename reflects the original leaky-split number, not the honest 71.79% from Phase 6). The `/predict` endpoint still loads it. The Phase 13 Essentia TDMS model ships on the `/predict-tdms` endpoint; the remaining step to put 75% in front of users is pointing the frontend's API path at `/predict-tdms` (the frontend currently calls `/predict`).

---

## Next Steps

**Replace v1 with TDMS-on-audio in production**

This is the headline next step. The Phase 8 benchmark shows v1's deployed pipeline at 8.32% top-1 and TDMS-on-audio at 37.36% top-1 on the same audio. Deployment requires three pieces: rebuilding the 480 templates from audio (not from expert pitch — Phase 8 cross-source numbers showed those don't generalize), uploading them to Hugging Face Hub alongside `raga_sklearn.pkl`, and rewriting `predict.py`'s feature extraction to produce a 14400-D TDMS instead of the current 360-D PCD. The 1-NN lookup is fast (one symmetric-KL distance per template, 480 templates total — a few milliseconds).

**Switch pyin to CREPE in the audio pipeline**

Phase 9b showed CREPE (model='small') gives +5.7 pp top-1 and +4.4 pp top-5 over pyin on identical audio with identical tonic. The cost is real — CREPE adds ~3-5 seconds of inference latency per 60s clip on CPU — but the accuracy gain is enough to justify it given how broken the v1 audio path is. The full CREPE model (`model_capacity='full'`) might add another 1-2 pp at the cost of much higher latency; not worth it unless Phase 10's longer-window experiment changes the calculus.

**Phase 10: longer windows and multi-window aggregation**

The 60s-vs-full-recording gap is what's left in the bottleneck attribution. The same CREPE + expert-tonic pipeline applied to the entire 30-minute recording instead of a 60s middle window should land much closer to the 85.67% Phase 1 ceiling. Production can't do that on every query (the user is uploading a short clip), but the *template index* can be built from full audio, which makes the asymmetry between train and inference work in our favor — long, dense templates queried by short, sparse user clips. Multi-window aggregation at inference (average TDMSs from three 60s windows of the user's upload before the 1-NN lookup) is the matched-cost variant. Phase 10 will run both.

**Wire the learned tonic detector into production**

The re-ranker described in the Tonic Detector section above sits at `models/tonic_detector_v1.pt` and is not yet called from `api/main.py`. The integration is small (one inference call between the existing candidate generation and the existing feature extraction). Phase 9a says this is worth +2.1 pp on tonic accuracy and on the order of +5-8 pp on raga top-1 in the audio pipeline — modest but cheap.

**Bidirectional octave fold in candidate generation**

The parity-fix work uncovered the real bottleneck: production's `_detect_tonic` only contains the correct tonic in its top-5 candidates 60.4% of the time, because the upward-only octave fold drops candidates an octave low. Making the fold bidirectional should raise the ceiling substantially. The re-ranker would need to be retrained against the new candidate distribution.

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
| Audio loading and signal processing | librosa, scipy, yt-dlp, ffmpeg, libsndfile |
| Pitch extraction (deployed) | librosa.pyin |
| Pitch extraction (Phase 9b candidate) | crepe (CNN-based, TensorFlow) |
| Feature engineering (deployed) | three-channel 360-D pitch class distribution |
| Feature engineering (Phase 7 candidate) | TDMS 120 × 120 joint distribution + scipy.ndimage Gaussian smoothing |
| Model training | PyTorch (research), scikit-learn MLPClassifier (deployment) |
| Model inference (deployed) | scikit-learn MLPClassifier on 360-D features |
| Model inference (Phase 7 candidate) | 1-NN with symmetric-KL distance over 480 TDMS templates |
| Evaluation | scikit-learn StratifiedKFold (recording-aware), custom `eval_harness.py` |
| Backend API | FastAPI, Python 3.11, uvicorn |
| Backend hosting | Hugging Face Spaces (free tier, Docker SDK) |
| Model storage | Hugging Face Hub |
| Database and file storage | Supabase (Postgres + Storage) |
| Frontend | React 19, Vite 8, Framer Motion, WaveSurfer.js |
| Frontend hosting | Vercel (Hobby tier) |
| Version control | GitHub |

---

## Project Structure

```
raga-identifier/
├── backend/
│   ├── api/
│   │   └── main.py                          FastAPI app (deployed)
│   ├── src/
│   │   │
│   │   │  --- v1 deployed pipeline ---
│   │   ├── preprocess.py                    360-D PCD feature from CompMusic .pitch files
│   │   ├── preprocess_audio_clips.py        Slice raw audio into 30s clips, compute PCD features
│   │   ├── train.py                         v1 training (PyTorch + sklearn copy for deployment)
│   │   ├── predict.py                       v1 inference (pyin + heuristic tonic + 360-D PCD)
│   │   ├── combine_datasets.py              Merge CompMusic + YouTube feature arrays
│   │   ├── download_youtube_data.py         YouTube data collection via yt-dlp
│   │   │
│   │   │  --- failed/abandoned experiments (v2-v5) ---
│   │   ├── train_v2.py                      v2 per-clip MLP (36.77% per-clip)
│   │   ├── train_v3.py                      v3 multi-scale MLP (53.62% per-recording vote)
│   │   ├── train_v4.py                      v4 Mel-spec CNN (11.46% — failed)
│   │   ├── train_v5_ast.py                  v5 AST embedding head (3.12% — chance)
│   │   ├── preprocess_melspec.py            Mel spectrogram precompute (v4)
│   │   ├── preprocess_ast_embeddings.py     AST CLS embedding extraction (v5)
│   │   ├── build_multiscale_dataset.py      Combine 15s/30s/1min/3min/full into X_multiscale
│   │   ├── gate1_report.py                  Cross-dataset feature alignment diagnostic (v2 era)
│   │   ├── verify_clip_features.py          Pre-flight check for clip feature quality (v2 era)
│   │   │
│   │   │  --- tonic detector (Tonic Detector section, not deployed) ---
│   │   ├── extract_tonic_candidates.py      Generate top-5 tonic candidates per recording
│   │   ├── train_tonic_detector.py          Learned re-ranker for tonic candidates (+2.1 pp)
│   │   ├── verify_tonic_detection.py        Auto-tonic vs expert diagnostic on 5 recordings
│   │   │
│   │   │  --- Phase 6: honest baseline + shared eval harness ---
│   │   ├── eval_harness.py                  Recording-aware k-fold CV, vote aggregation
│   │   ├── baseline_v1_cv.py                v1 honest re-baseline → 71.79% top-1
│   │   │
│   │   │  --- Phase 7: TDMS feature, expert pitch ---
│   │   ├── build_tdms.py                    TDMS extractor (Gulati 2016) + three distance fns
│   │   ├── extract_tdms_features.py         Walk 480 CompMusic recordings → X_tdms.npy
│   │   ├── train_tdms.py                    kNN + MLP variants → 86.67% LOO match
│   │   │
│   │   │  --- Phase 8 + 9: TDMS audio pipeline benchmarks ---
│   │   ├── build_tdms_from_audio.py         Pyin + tonic detection → TDMS, runtime variant
│   │   ├── eval_audio_ab.py                 v1 vs TDMS on identical audio (37.36% vs 8.32%)
│   │   ├── eval_audio_expert_tonic.py       Pyin + expert tonic (45.81%) — Phase 9a
│   │   └── eval_crepe_audio.py              CREPE + expert tonic (51.51%) — Phase 9b
│   │
│   ├── data/
│   │   ├── X.npy / y.npy                    CompMusic v1 training features (480 × 360)
│   │   ├── X_yt.npy / y_yt.npy              YouTube v1 training features (209 × 360)
│   │   ├── X_tdms.npy / y_tdms.npy          Phase 7 expert templates (480 × 14400)
│   │   ├── X_audio_clips.npy                v2 per-clip features (44,071 × 360)
│   │   ├── X_15s.npy / X_1min.npy / ...     v3 multi-scale features (gitignored)
│   │   ├── classes.json                     Raga names (40)
│   │   ├── youtube_videos.json              Video index for deduplication
│   │   ├── baseline_v1_cv_report.txt        Phase 6 artifact
│   │   ├── train_tdms_report.txt            Phase 7 artifact
│   │   ├── eval_audio_ab_report.txt         Phase 8 artifact
│   │   ├── eval_audio_expert_tonic_report.txt   Phase 9a artifact
│   │   └── eval_crepe_audio_report.txt      Phase 9b artifact
│   ├── models/
│   │   ├── raga_model_best.pt               v1 PyTorch
│   │   ├── raga_sklearn.pkl                 v1 deployed sklearn (loaded at startup from HF)
│   │   ├── scaler.pkl                       v1 feature scaler
│   │   ├── raga_model_best_v2.pt            v2 PyTorch (not deployed)
│   │   ├── raga_sklearn_v2.pkl / scaler_v2.pkl   v2 (not deployed)
│   │   ├── raga_model_best_v3.pt            v3 PyTorch (not deployed)
│   │   ├── raga_sklearn_v3.pkl / scaler_v3.pkl   v3 (not deployed)
│   │   ├── raga_cnn_v4.pt                   v4 (failed)
│   │   ├── raga_ast_head_v5.pt              v5 (failed)
│   │   └── tonic_detector_v1.pt             Tonic Detector section learned tonic re-ranker (not wired)
│   ├── requirements.txt                     Full local dependencies (includes crepe, TF)
│   ├── requirements-deploy.txt              Slim production deps (no PyTorch, no crepe, no TF)
│   ├── Dockerfile                           HF Spaces build (python:3.11-slim + ffmpeg)
│   └── .env                                 Supabase URL/key (gitignored)
└── frontend/
    ├── src/
    │   ├── App.jsx                          Main React component (single-file)
    │   ├── main.jsx                         Vite entry
    │   └── ragas.json                       40-raga knowledge base (arohanam, avarohanam, similar)
    ├── public/                              Static assets
    ├── index.html
    ├── vite.config.js
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

### v1 pipeline (the deployed model)

```bash
cd backend
source venv/bin/activate

# (Optional) Collect more YouTube training data
python src/download_youtube_data.py

# Extract features from CompMusic pitch files (produces X.npy / y.npy)
python src/preprocess.py

# (Optional) Slice raw audio recordings into 30s clips and extract features
# Requires the CompMusic audio dataset at ~/raga-data-audio/
python src/preprocess_audio_clips.py

# Train — produces both PyTorch model and sklearn model for deployment
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

### Honest re-baseline of v1 (Phase 6)

Once `data/X.npy` and `data/classes.json` exist, the honest 71.79% top-1 number for v1 can be reproduced with:

```bash
python src/baseline_v1_cv.py        # 5-fold CV with 5 seeds, ~5 seconds
cat data/baseline_v1_cv_report.txt
```

### TDMS pipeline (Phase 7 candidate, not deployed)

```bash
# Build the 480 TDMS templates from CompMusic expert pitch files
# Outputs data/X_tdms.npy (480 × 14400) and data/tdms_meta.json
python src/extract_tdms_features.py

# Evaluate 5 variants under the same harness as v1
# Headline: k-NN sym KL hits 86.67% LOO / 85.67% 5-fold
python src/train_tdms.py
cat data/train_tdms_report.txt
```

### Audio-pipeline benchmarks (Phase 8 + 9)

Phase 8 needs `~/raga-data-audio/` (the 7.5 GB CompMusic audio dataset). All three benchmarks share the same 60s middle window per recording.

```bash
# Phase 8: v1 vs TDMS on identical audio (12 min in ProcessPoolExecutor with 6 workers)
python src/eval_audio_ab.py --workers 6
cat data/eval_audio_ab_report.txt

# Phase 9a: pyin pitch + EXPERT tonic — isolates tonic-detection contribution (~13 min)
python src/eval_audio_expert_tonic.py --workers 6
cat data/eval_audio_expert_tonic_report.txt

# Phase 9b: CREPE pitch + EXPERT tonic — isolates pitch-extractor contribution (~18 min)
python src/eval_crepe_audio.py --model small
cat data/eval_crepe_audio_report.txt
```

### Deploying TDMS to production (not done yet, but here is the recipe)

These steps would replace the v1 model on the live API. They are NOT performed automatically — production cutover should only happen after Phase 10 lands the longer-window experiment and after a brief shadow-mode comparison.

```bash
# 1. Build the deployable index from AUDIO (not expert pitch — see Phase 8 cross-source results)
# This rebuilds the 480 TDMS templates using the runtime pyin pipeline.
python src/eval_audio_ab.py --workers 6
# (uses the cached X_tdms_audio.npy if present; pass --skip-extract to reuse)

# 2. Upload the audio templates + classes to Hugging Face Hub
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(path_or_fileobj='data/X_tdms_audio.npy', path_in_repo='X_tdms_audio.npy', repo_id='Smashgod23/raga-identifier')
api.upload_file(path_or_fileobj='data/y_tdms.npy', path_in_repo='y_tdms_audio.npy', repo_id='Smashgod23/raga-identifier')
"

# 3. Patch api/main.py to download the index on startup, swap the inference call to
#    src.build_tdms_from_audio.compute_tdms_from_audio() + a 1-NN with symmetric KL.
```

A clean A/B test before the cutover would route, say, 10% of requests to the new path and compare top-1 / top-5 / "did the user accept" on feedback corrections.

---

Built by Pratham Aithal
Rock Hill High School, Frisco, TX (PISD)
theprathamaithal@gmail.com
https://github.com/Smashgod23/raga-identifier
