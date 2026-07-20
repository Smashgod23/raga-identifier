# Raga Identifier Leaderboard

Canonical results table. `leaderboard.xlsx` is generated from this file and must be kept in sync. Rules for adding a row: the number must be measured on the standard evaluation harness, config chosen by cross-validation (never tuned on the eval set), and deployed vs research configurations stay labeled separately.

## Main leaderboard (novel real-world audio, n=194 YouTube recordings, never trained on)

| # | Model / Config | Top-1 (%) | Top-5 (%) | Eval protocol | Status |
|---|---|---|---|---|---|
| 1 | v1: 360-D pitch-distribution features + MLP | 17.0 | - | 194 novel YouTube recordings | Deployed (/predict, legacy) |
| 2 | TDMS (Gulati et al. 2016) + 1-NN, symmetric KL | 50.0 | 70-80 | Same novel set | Deployed (/predict-tdms) |
| 3 | Bi-LSTM + attention, dual-pitch training, 3-seed ensemble | 64.4 | - | Same novel set | Research |
| 4 | + L=1600 eval windows (~70 s), 8-window average | 66.0 | 82.0 | Same novel set | Research |
| 5 | v2: + margin-gated BLEND tonic TTA (octave-fold) | 69.6 | 85.6 | 5-fold CV, config off-fold | Research |
| 6 | v3: + learned tonic-frame selector (multitask frame head) | 69.6 | 87.1 | 5-fold CV, config off-fold, L inside CV | Research (best) |
| 7 | Deployed v3 config: 3 seeds, 8 windows, L=1600, no TTA | 65.5 | 81.4 | Same novel set, exact production path | Deployed (/predict-v3, live) |

Note on deployment status: the live API is served from the Hugging Face Space repo, which is deployed separately from this GitHub repo. `/predict-v3` is live in production (verified end to end 2026-07-16) even though this repo's copy of `backend/api/main.py` has not yet been synced to include it.

## In-distribution reference (train-domain numbers, not comparable to the table above)

| Model | Top-1 (%) | Top-5 (%) | Protocol |
|---|---|---|---|
| v1 MLP (pitch-distribution features) | 84.4 | - | Held-out split on CompMusic features |
| TDMS + 1-NN | 75.62 | 91.17 | CMD 480 recordings, leave-one-out, 7x90s queries |
| DeepSRGM-style bi-LSTM + attention (3 seeds) | 89.2 +/- 1.2 | 97.9 | 3-seed recording-aware CV on expert pitch, CMD 480 |

## Open-set (unseen-raga rejection): AUROC separating seen vs unseen ragas

Protocol: 8/40 ragas held out (fixed rng(0) split, including Kalyani, Sankarabharanam, Bhairavi, Kapi), 3 seeds retrained on the remaining 32, all 194 novel recordings scored. Mean top-similarity: seen 75.1% vs unseen 53.4%.

| Uncertainty score | AUROC | Notes |
|---|---|---|
| Max softmax probability (MSP) | 75.3 | |
| Negative entropy | 78.9 | |
| Energy (logsumexp of logits) | 81.2 | Deployed. FPR@95TPR ~86% |
| Deep k-NN distance (embedding space) | 80.6 | |
| Energy + k-NN combination | 82.3 | Rejected: worse FPR@95TPR |

Deployed calibration: energy < 3.43 (10th percentile of the novel set) flags a recording; flagged recordings measure 5% top-1 vs 72% unflagged.

Last updated: 2026-07-20.
