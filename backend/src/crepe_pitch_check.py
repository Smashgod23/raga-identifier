"""
Decisive cheap test before committing to a CREPE re-extraction: on a few CMD
recordings (which have EXPERT .pitch ground truth), does CREPE track the pitch
better than Essentia Melodia? If CREPE agrees with the expert track more, it is
worth the full extraction; if polyphony confuses it, abandon it now.

Metric: on frames the expert marks voiced, fraction where the tracker is within
50 cents of expert (raw, and octave-folded). Higher = closer to ground truth.
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
DATASET = os.path.join(DATA, "RagaDataset", "Carnatic")
WIN = (120.0, 180.0)   # a 60s middle-ish window


def expert_path(recording_id):
    piece = os.path.basename(recording_id)
    return os.path.join(DATASET, "features", recording_id, f"{piece}.pitchSilIntrpPP")


def tonic_path(recording_id):
    piece = os.path.basename(recording_id)
    return os.path.join(DATASET, "features", recording_id, f"{piece}.tonicFine")


def agree(p, p_exp, tol=50.0, fold=False):
    m = (p > 0) & (p_exp > 0)
    if m.sum() == 0:
        return float("nan")
    c = 1200.0 * np.log2(p[m] / p_exp[m])
    if fold:
        c = (c + 600) % 1200 - 600
    return float(np.mean(np.abs(c) < tol))


def main():
    import torch
    import torchcrepe
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    meta = json.load(open(os.path.join(DATA, "tdms_meta.json")))
    picked = []
    for e in meta:
        rid = e["recording_id"]
        if os.path.exists(expert_path(rid)) and os.path.exists(audio_path_for(rid)):
            picked.append(e)
        if len(picked) >= 3:
            break

    for e in picked:
        rid = e["recording_id"]; raga = e["raga"]
        ep = np.loadtxt(expert_path(rid)); et, epi = ep[:, 0], ep[:, 1]
        mask = (et >= WIN[0]) & (et < WIN[1])
        et_w, exp_w = et[mask], epi[mask]
        if len(et_w) < 100:
            print(f"{raga}: too little expert pitch in window"); continue

        audio = es.MonoLoader(filename=audio_path_for(rid), sampleRate=pe.SR)()
        seg = audio[int(WIN[0] * pe.SR): int(WIN[1] * pe.SR)]

        # Essentia Melodia on the segment, aligned to expert timestamps
        epitch, _ = es.PredominantPitchMelodia(frameSize=pe.MELODIA_FRAME, hopSize=pe.MELODIA_HOP)(seg)
        et_ess = WIN[0] + np.arange(len(epitch)) * (pe.MELODIA_HOP / pe.SR)
        ess_on_exp = np.interp(et_w, et_ess, epitch, left=0, right=0)

        # CREPE on the segment (needs 16k), aligned to expert timestamps
        seg16 = es.Resample(inputSampleRate=pe.SR, outputSampleRate=16000)(seg)
        with torch.no_grad():
            f0, pd = torchcrepe.predict(torch.tensor(seg16)[None], 16000, hop_length=160,
                                        model="full", device=dev, batch_size=512,
                                        return_periodicity=True)
        f0 = f0[0].cpu().numpy(); pd = pd[0].cpu().numpy()
        f0 = np.where(pd > 0.5, f0, 0.0)   # voicing by periodicity
        ct = WIN[0] + np.arange(len(f0)) * (160 / 16000)
        crepe_on_exp = np.interp(et_w, ct, f0, left=0, right=0)

        print(f"\n{raga} ({rid.split('/')[0][:8]})  voiced expert frames: {(exp_w>0).sum()}")
        print(f"  Essentia vs expert: raw {agree(ess_on_exp, exp_w)*100:5.1f}%  octave-folded {agree(ess_on_exp, exp_w, fold=True)*100:5.1f}%")
        print(f"  CREPE    vs expert: raw {agree(crepe_on_exp, exp_w)*100:5.1f}%  octave-folded {agree(crepe_on_exp, exp_w, fold=True)*100:5.1f}%")


if __name__ == "__main__":
    main()
