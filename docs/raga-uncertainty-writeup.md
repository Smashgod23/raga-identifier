# Raga Identification with Uncertainty: Approach and Results

Pratham Aithal, July 2026. Live implementation: https://raga-identifier.vercel.app

## The problem

Given a test audio snippet, identify the raga and similar ragas with honest similarity scores. If the snippet is from a raga the model was never trained on, the scores should be low rather than confidently wrong. Where possible, explain which segments of the audio drove the decision.

## System summary

The classifier is a sequence model (DeepSRGM-style bidirectional LSTM with attention) over the tonic-normalized pitch contour: Essentia's Melodia extracts the melody, TonicIndianArtMusic finds Sa, and the contour becomes a sequence of 50-cent pitch-class tokens with silence removed. Three independently seeded models are trained on 480 CompMusic recordings across 40 ragas, using two pitch tracks per recording (the expert-corrected track and the Essentia track), which teaches invariance to pitch-extraction quality and was worth about ten points of real-world top-1 by itself. Inference averages the softmax of the three seeds over eight 70-second windows.

All numbers below are on 194 YouTube concert recordings the models never trained on, so they reflect real-world audio, not benchmark conditions. Labels for these recordings come from search results, so measured accuracy is a floor.

## 1. Ranked identification with similar ragas

The model returns a ranked list, not a single answer. Closed-set accuracy on the novel recordings: 65.5 percent top-1 and 81.4 percent top-5 in the deployed configuration, and 69.6 / 87.1 with cross-validated tonic-hypothesis correction in the research configuration. Since allied ragas (Kalyani and Purvikalyani, Kambhoji and Harikambhoji) share scale material and differ in phrase behavior, the ranked list is the honest output format: the true raga is in the top five ranks about six times more often than a wrong-only list would suggest from top-1 alone.

## 2. Unseen ragas get low scores

Protocol: hold out 8 of the 40 ragas entirely (fixed random pick, including Kalyani, Sankarabharanam, Bhairavi and Kapi, several of which have close cousins still in the training set, which is the hardest honest version of the test), retrain on the remaining 32, then score all 194 novel recordings.

Result: recordings of seen ragas score a mean top similarity of 75.1 percent; recordings of the held-out ragas score 53.4 percent. For rejection we compared max-softmax (AUROC 75.3), negative entropy (78.9), energy, i.e. logsumexp of the logits (81.2), a deep k-NN distance in embedding space (80.6), and an energy plus k-NN combination (82.3 but with worse FPR at 95 percent TPR). Energy was deployed: best FPR, within a point of the combination, and it costs nothing at inference.

The deployed threshold was calibrated on the 194 novel recordings: flagging the lowest-energy 10 percent separates the model's failures from its successes sharply. Flagged recordings are 5 percent top-1 (effectively "the model does not know this"), unflagged recordings are 72 percent. The live site shows this as a banner: "low similarity to every raga I know; this may be a raga outside my 40." In the very first live test after deployment, the model misranked a Kalyani alapana and the flag fired, which is exactly the designed behavior: when wrong, say so.

Caveat stated plainly: at a strict operating point that accepts 95 percent of seen-raga clips, most unseen-raga clips still pass (FPR at 95 percent TPR around 86 percent). With 35 unseen-raga recordings the confidence interval is wide. The separation is real and usable at the deployed operating point, but this is uncertainty communication, not a hard open-set classifier.

## 3. Explainable evidence segments

Two signals, both from the deployed model. Occlusion: the token sequence is split into twelve chunks, each chunk is masked in turn, and the drop in the predicted raga's probability measures how much evidence that chunk carried. Attention: the model's own attention weights over time. Because tokens skip unvoiced frames, every token carries its source timestamp, so reported segments are real audio times. The site displays the top segments as "decided by what it heard at 0:50-0:59, 1:27-1:35."

## What is deployed

The `/predict-v3` endpoint (FastAPI on Hugging Face Spaces, CPU) runs the full pipeline per upload or recording: melody extraction, three-seed ensemble, energy uncertainty with the calibrated threshold, and occlusion-based evidence segments. The frontend renders the ranked list, the unfamiliarity banner when the flag fires, and the evidence timestamps. If the sequence model is unavailable the API degrades to the older model rather than failing.

## Limitations and next steps

Top-1 on novel audio plateaus near 70 percent across three architecturally different attempts; the remaining errors are allied-raga confusions and label noise in the evaluation set. The open-set experiment used one 8-raga hold-out split; averaging over multiple splits would tighten the estimate. The energy threshold is calibrated on in-the-wild concert audio and may need recalibration for studio or vocal-only input. Promising next steps: expert-verified labels (both training and evaluation), phrase-gap tokens so the model sees phrase boundaries, and a multi-split open-set study.
