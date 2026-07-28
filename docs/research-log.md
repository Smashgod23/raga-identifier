# Raga Identifier: Research Log

Pratham Aithal, Rock Hill High School, Frisco, TX (PISD)
theprathamaithal@gmail.com
Live site: https://raga-identifier.vercel.app
Code: https://github.com/Smashgod23/raga-identifier

Last updated: 2026-07-28.

This file is the canonical copy of the research log I share with Prof. Vipul. The Google Doc he reads is a mirror of this file. When this file changes, the doc gets refreshed from it, so the doc link stays the same and the history lives in git.

Related canonical files: `leaderboard.md` holds the results tables, `raga-uncertainty-writeup.md` holds the open-set writeup.

---

## How to read this document

This is a running log of what I am changing and what the changes measure out to. I add a new dated entry every time I finish a piece of work, newest at the top of the log. The sections above the log get edited in place so they always show the current state rather than a history.

Everything in the results table is a measured number from a fixed evaluation protocol. If something is untested or still running, I say so rather than leaving it out.

## The project in one paragraph

The system takes an audio recording of Carnatic music and predicts which of 40 ragas is being performed. Input can be a microphone recording, an uploaded file, or a YouTube link. Training data is the CompMusic Carnatic Music Dataset (480 recordings with expert-annotated pitch contours and tonic values). The honest evaluation set is 194 YouTube concert recordings that no model has ever trained on, which is a much harder and more realistic test than a held-out split of the training corpus. The gap between those two numbers is the main story of the project so far.

## Current results

### Main table: novel real-world audio (n = 194 YouTube recordings, never trained on)

| # | Model / config | Top-1 (%) | Top-5 (%) | Status |
|---|---|---|---|---|
| 1 | v1: 360-D pitch-distribution features + MLP | 17.0 | - | Deployed (legacy) |
| 2 | TDMS (Gulati et al. 2016) + 1-NN, symmetric KL | 50.0 | 70-80 | Deployed |
| 3 | Bi-LSTM + attention, dual-pitch training, 3-seed ensemble | 64.4 | - | Research |
| 4 | + L = 1600 eval windows (about 70 s), 8-window average | 66.0 | 82.0 | Research |
| 5 | v2: + margin-gated blend tonic test-time augmentation | 69.6 | 85.6 | Research |
| 6 | v3: + learned tonic-frame selector (multitask frame head) | 69.6 | 87.1 | Research (best) |
| 7 | Deployed v3 config: 3 seeds, 8 windows, L = 1600, no TTA | 65.5 | 81.4 | Live in production |

Rows 5 and 6 are 5-fold cross-validated with the configuration chosen off-fold, so no knob was tuned on the evaluation set. Row 7 is the exact production path, which trades about 4 points of top-1 for latency on a CPU-only server.

### In-distribution reference (not comparable to the table above)

| Model | Top-1 (%) | Top-5 (%) | Protocol |
|---|---|---|---|
| TDMS + 1-NN | 75.62 | 91.17 | CMD 480, leave-one-out, 7 x 90 s queries |
| DeepSRGM-style bi-LSTM + attention (3 seeds) | 89.2 +/- 1.2 | 97.9 | 3-seed recording-aware CV on expert pitch |

### Open-set rejection: can the model tell when it is hearing a raga it was never trained on?

Protocol: 8 of 40 ragas held out (fixed seed, and deliberately including allied cousins like Kalyani and Sankarabharanam so the remaining classes stay confusable), 3 seeds retrained on the other 32, all 194 novel recordings scored. Mean top-similarity was 75.1% for seen ragas against 53.4% for unseen ones.

| Uncertainty score | AUROC | Notes |
|---|---|---|
| Max softmax probability | 75.3 | |
| Negative entropy | 78.9 | |
| Energy (logsumexp of logits) | 81.2 | Deployed. FPR at 95% TPR is about 86% |
| Deep k-NN distance in embedding space | 80.6 | |
| Energy + k-NN combined | 82.3 | Rejected, worse FPR at 95% TPR |

Because the false positive rate is high even at the best operating point, the deployed behavior is to communicate uncertainty rather than hard-reject. Recordings flagged by the energy threshold measure 5% top-1 against 72% for unflagged ones, so the flag is doing real work even though it is not a clean separator.

## What has been ruled out, with evidence

I keep this list so that ideas that sound good but already failed do not get retried. Each one was measured on the standard harness.

- Expanding the template index with auto-labeled YouTube data. 190 extra templates moved query top-1 from 50% to 43.8%. A cleaned 108-recording subset was neutral. Cause: label noise plus tonic-detection noise compounding.
- Demucs source separation to isolate the voice. Top-1 fell from 50% to 25%. Demucs is trained on Western music, it distorts gamakas, and it strips the tanpura drone that tonic detection depends on.
- CREPE instead of Melodia for pitch tracking. CREPE tracks Carnatic pitch worse than Melodia against expert ground truth. It is a monophonic model and these are polyphonic concert recordings.
- Fusing DeepSRGM with TDMS. The apparent +4 points came from tuning the fusion weight in-sample. At a fixed weight the gain is roughly zero.
- Adding YouTube contours to training, both raw and confidence-cleaned. Neutral, again label noise.
- Tempo and token-jitter augmentation. Neutral.
- Naive max-confidence tonic test-time augmentation. Actively harmful. The selector is miscalibrated across shifts, and clipping at the vocabulary edge fakes high confidence.
- Mel-spectrogram CNN trained from scratch (Phase 4). 11.46% per-recording. Train accuracy hit 86% while validation sat at chance. With 251k parameters and 384 training recordings the CNN memorized concert identity: tanpura tuning, microphone, room reverb, singer timbre.
- Frozen AST (Audio Spectrogram Transformer) embeddings + classifier head (Phase 5). 3.12% per-recording, barely above the 2.5% random baseline for 40 classes. AST's pretraining objective is closer to "what kind of acoustic event is this" than "what raga is this", so its embeddings cluster by concert rather than by raga.

## Where the remaining headroom is

Three independent methods (margin-gated TTA, a learned frame selector, and a confidence heuristic) all converge at 69.6% top-1 on novel audio, which reads as a plateau for the current model family rather than a tuning problem.

The one measured gap that is large and specific: an oracle that is allowed to pick the best tonic hypothesis per recording reaches about 79.9%. So roughly 10 points are sitting in tonic-hypothesis selection alone. The model often has the right answer available and picks the wrong frame of reference.

The other known limit is label quality. The 194 evaluation labels come from an automated search rather than expert verification, so the measured ceiling is a floor on true accuracy, and some fraction of the residual error is probably mislabeled evaluation data rather than model error.

---

## Log

### 2026-07-28: acting on the chromagram and spectrogram suggestion

Where the suggestion landed. Prof. Vipul suggested trying chromagrams or spectrograms. Two of those have already been tried and failed, which is worth stating up front so the record is honest: a mel-spectrogram CNN from scratch got 11.46%, and frozen AST spectrogram-transformer embeddings got 3.12%. Both failed the same way, by learning concert identity instead of raga identity. Raw spectrograms carry the singer's timbre, the room, and the microphone, and with only 384 training recordings that is the easiest signal for a model to latch onto.

Why the chromagram half of the suggestion is different. A chromagram fixes exactly the failure mode that killed the spectrogram attempts. Folding every octave onto one axis removes register, so a male and a female performer singing the same raga land in the same place. Normalizing to the tonic removes the performer's chosen Sa. What is left is pitch-class energy over time, with most of the timbre and room information gone.

There is also something a chromagram keeps that the current best model throws away. The current model runs Melodia, which commits to a single predominant pitch per frame and discards everything else. A chromagram is polyphonic: the tanpura drone, the violin, and the voice all stay visible in the same frame. That matters because the drone is a direct, continuous tonic cue, and tonic selection is precisely where the 10-point oracle gap sits.

What I built. A high-resolution constant-Q chromagram extractor (`src/preprocess_chroma.py`) for all 480 CompMusic recordings. Parameters were chosen to line up with the existing token pipeline so the two representations are directly comparable and can be combined later:

- Constant-Q transform, 60 bins per octave (20 cents per bin, fine enough to keep gamaka detail), 5 octaves starting at C2
- Folded to a 60-bin pitch-class profile per frame
- Hop of 704 samples at 16 kHz, giving 22.73 frames per second, which matches the roughly 22.5 Hz rate of the existing pitch-token sequences
- A separate per-frame log-energy channel, z-scored per recording so absolute loudness cannot act as a recording fingerprint
- Tonic normalization is a circular roll applied at load time, which means test-time tonic hypotheses are exact rotations rather than the lossy token shifts the current model uses

First sanity check, and it passed cleanly. On the first recording, averaging the chromagram over the whole performance puts the largest peak exactly on the annotated tonic bin, and the second largest peak 700 cents above it. That is Sa and Pa, which is exactly what a tanpura drone should produce. The representation is picking up the thing I built it to pick up.

Plan, with a gate so this does not become another dead end:

1. Extract chromagrams for all 480 recordings.
2. In-distribution gate (`src/chroma_gate.py`): a controlled swap of the proof-of-concept sequence model. Same recording-aware split, same window count, same bi-LSTM with attention pooling, same schedule. The only change is that the integer pitch-token embedding becomes a linear projection of the 61-dim chroma frame. If chroma lands near the 89.2% token baseline, the representation carries real raga information. If it collapses toward chance like Phase 4 did, chroma goes on the ruled-out list.
3. If it passes: re-download audio for the 194 evaluation recordings (only the pitch contours were archived, not the audio) and measure on the honest novel set.
4. Separately, and cheaper: use the chromagram drone peak as a tonic hypothesis scorer. This targets the 10-point oracle gap directly and does not require the chroma model to beat anything on its own.

What I expect. I do not expect the chroma model to beat the pitch-token model outright. The token model gets a clean melodic line; chroma gets a blurrier picture with more in it. What I think is more likely, and more useful, is that the two fail on different recordings, in which case combining them helps, and that the drone channel improves tonic selection.

### 2026-07-20: canonical results table

Built the leaderboard as a canonical file in the repository, with explicit admission rules: a number only goes in the table if it was measured on the standard harness, the configuration was chosen by cross-validation and never tuned on the evaluation set, and deployed configurations stay labeled separately from research ones. Any session that produces a new benchmark number now updates the table automatically.

### 2026-07-12 to 07-16: uncertainty and explainability, shipped live

This was the response to the problem Prof. Vipul posed: a raga identifier that knows when it does not know, and that can point at what it heard.

Open-set experiment: held out 8 of 40 ragas, retrained on the rest, and scored all 194 novel recordings. The full AUROC comparison is in the table above. Energy scoring won on the metric that matters for deployment and is now what runs in production.

Explainability: chunk-occlusion plus attention, with real timestamps mapped back through the token sequence (which skips silence, so the mapping is not a straight multiplication). The site now shows evidence segments phrased as "decided by what it heard at mm:ss".

The most useful thing that happened was a failure. The first live test after deployment misranked a Kalyani clip, and the uncertainty flag fired at the same time. The model was wrong and said so. That is the behavior the whole open-set piece was built for.

Two bugs surfaced in live testing and were fixed: microphone recordings arrive as WebM/Opus, which the Linux Essentia build cannot decode (now transcoded server-side with ffmpeg before analysis), and the server was segfaulting under load from a thread-pool clash between PyTorch and Essentia's OpenMP (fixed by pinning both to a single thread).

---

## Open questions and next steps

- Does the chromagram representation clear the in-distribution gate?
- Expert-verified labels for the evaluation set. The current labels come from automated search, and I believe some of the residual 30% error is label noise rather than model error. This is the single change most likely to make every other number more trustworthy.
- Explicit rest and gap tokens. The current tokenization drops silence entirely, which means phrase boundaries are invisible to the model, and phrase structure is a large part of what distinguishes allied ragas.
- A two-stage melakarta classifier: predict the parent scale first, then the raga within it.
- Finer pitch bins (25 cents instead of 50) for the token model.
- More clean training data, which remains the boring answer that would probably work.
