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

The structural read is that pitch-class histograms throw away phrase order. A raga is partially defined by the sequence in which characteristic phrases appear (the pakad), and a histogram is a bag of pitches with no time information at all. Two ragas that share a swara set but differ in the canonical phrase order project to nearly identical pitch class distributions, so a histogram-based classifier cannot distinguish them no matter how much data it sees. v3 is not deployed. The 84.4% v1 model continues to serve traffic at smashgod23-raga-identifier-api.hf.space.

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

| Version | Approach | Per-recording test accuracy |
|---------|----------|------------------------------|
| v1 | 360-dim PCD features + MLP, full-recording only | 84.4% (deployed) |
| v2 | Same features + audio clip slicing (30s windows) | 36.77% per-clip |
| v3 | Multi-scale PCD features (15s/30s/1min/3min/full) + MLP | 53.62% |
| v4 | Log-Mel spectrograms + 2D CNN | 11.46% |
| v5 | AST audio embeddings + MLP head | 3.12% |

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

The learned tonic detector is not yet wired into the production inference path. The integration plan is to load `raga_tonic_detector.pkl` alongside the existing model and scaler in `api/main.py`, replace the heuristic argmax inside `_detect_tonic` with a forward pass of the re-ranker over the five candidates, and re-measure raga accuracy end to end. Expected lift on the live system is modest given the +2.1 pp tonic improvement, but the larger value is the diagnostic: it pins down where the actual bottleneck is.

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

**Wire the learned tonic detector into production**

The re-ranker described in the Tonic Detector section above sits at `models/raga_tonic_detector.pkl` and is not yet called from `api/main.py`. The integration is small (one inference call between the existing candidate generation and the existing feature extraction) but needs an end-to-end raga-accuracy re-measurement on the deployed v1 model before it ships.

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
│   │   ├── preprocess_melspec.py          Log-Mel spectrogram precompute (v4)
│   │   ├── preprocess_ast_embeddings.py   AST CLS embedding extraction (v5)
│   │   ├── build_multiscale_dataset.py    Combine scale-specific X arrays into X_multiscale
│   │   ├── combine_datasets.py            Stitch CompMusic + YouTube feature arrays
│   │   ├── extract_tonic_candidates.py    Generate top-5 tonic candidates per recording
│   │   ├── train.py                       v1 deployed model training (PyTorch + sklearn)
│   │   ├── train_v2.py                    v2 per-clip training (failed)
│   │   ├── train_v3.py                    v3 multi-scale training (53.62% per-recording vote)
│   │   ├── train_v4.py                    v4 Mel-spec CNN (failed)
│   │   ├── train_v5_ast.py                v5 AST embedding classifier (failed)
│   │   ├── train_tonic_detector.py        Learned re-ranker for tonic candidates
│   │   ├── predict.py                     Inference for live audio
│   │   ├── download_youtube_data.py       YouTube data collection
│   │   ├── gate1_report.py                Cross-dataset feature alignment diagnostic
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
