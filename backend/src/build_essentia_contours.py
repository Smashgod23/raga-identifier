"""
Build Essentia-pitch contours for DeepSRGM training (the DEPLOYABLE pipeline).

The DeepSRGM POC trained on expert .pitchSilIntrpPP contours -> 89.2% in-dist.
But inference uses Essentia Melodia pitch + Essentia tonic, not expert tracks, so
a deployable model must train on the SAME extractor. This runs Essentia Melodia +
TonicIndianArtMusic on each of the 480 CMD recordings and caches the tokenized
contours in the SAME format as deepsrgm_seqs.npz, so deepsrgm_poc.py can train on
them by pointing CACHE at this file.

Resumable: checkpoints every 25 recordings. ~30-80 min for 480 recordings.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import essentia.standard as es
import predict_essentia as pe
from eval_audio_ab import audio_path_for

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(DATA, "deepsrgm_essentia_seqs.npz")

# Match deepsrgm_poc tokenization, but downsample for the Essentia 344.5 Hz hop.
BIN, LO, HI = 50.0, -1200, 2400
DS = max(1, round((pe.SR / pe.MELODIA_HOP) / 22.5))   # -> 15, ~22.5 Hz tokens
MAX_ANALYZE_S = 1200.0                                  # cap decode/Melodia at 20 min


def tokenize(pitch, tonic):
    voiced = pitch[pitch > 0]
    if tonic <= 0 or len(voiced) < 200:
        return None
    cents = 1200.0 * np.log2(voiced / tonic)
    idx = np.round((np.clip(cents, LO, HI) - LO) / BIN).astype(np.int64)
    return idx[::DS].astype(np.int16)


def contour_for(path):
    audio = es.MonoLoader(filename=path, sampleRate=pe.SR)()
    if len(audio) > MAX_ANALYZE_S * pe.SR:                # middle window, bounds time
        mid = len(audio) // 2; half = int(MAX_ANALYZE_S * pe.SR // 2)
        audio = audio[mid - half: mid + half]
    tonic = pe.detect_tonic(audio)                        # Essentia TonicIndianArtMusic
    pitch, _ = es.PredominantPitchMelodia(frameSize=pe.MELODIA_FRAME, hopSize=pe.MELODIA_HOP)(audio)
    return tokenize(np.asarray(pitch, dtype=np.float64), tonic)


def save(seqs, labels):
    np.savez_compressed(OUT, **{f"s{i}": s for i, s in enumerate(seqs)},
                        labels=np.array(labels), n=len(seqs))


def main():
    meta = json.load(open(os.path.join(DATA, "tdms_meta.json")))
    classes = json.load(open(os.path.join(DATA, "classes.json")))
    name_to_label = {n: i for i, n in enumerate(classes)}
    seqs, labels = [], []
    ok = skip = 0
    for i, entry in enumerate(meta):
        rid = entry["recording_id"]; raga = entry["raga"]
        label = name_to_label.get(raga)
        path = audio_path_for(rid)
        if label is None or not os.path.exists(path):
            skip += 1; print(f"[{i+1}/{len(meta)}] skip ({'no label' if label is None else 'no mp3'}): {raga}", flush=True); continue
        try:
            toks = contour_for(path)
        except Exception as exc:
            skip += 1; print(f"[{i+1}/{len(meta)}] essentia failed: {exc!r}", flush=True); continue
        if toks is None:
            skip += 1; print(f"[{i+1}/{len(meta)}] no usable contour: {raga}", flush=True); continue
        seqs.append(toks); labels.append(label); ok += 1
        print(f"[{i+1}/{len(meta)}] {raga}: {len(toks)} tokens (ok {ok})", flush=True)
        if ok % 25 == 0:
            save(seqs, labels); print(f"  checkpoint: {ok} saved", flush=True)
    save(seqs, labels)
    print(f"\nDONE: {ok} contours, {skip} skipped -> {OUT}")


if __name__ == "__main__":
    main()
