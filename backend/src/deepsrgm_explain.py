"""
Explainable evidence segments for a DeepSRGM prediction (professor problem, pt 2).

Which parts of the clip made the model choose this raga? Two signals:
  1. OCCLUSION: split the (voiced) token sequence into ~contiguous chunks, PAD
     each chunk out in turn, and measure how much the predicted raga's ensemble
     probability drops. A big drop = that chunk carried the evidence.
  2. ATTENTION: the model's own attention weights over time, per chunk.

Tokens skip unvoiced frames, so token index is NOT wall-clock time; every token
carries the timestamp of its source pitch frame, and reported segments are real
audio times. Designed to be imported by api/main.py at deployment:
    ex = load_ensemble(); explain(ex, audio_path)  ->  dict payload
"""
import json
import os

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
BIN, LO, HI = 50.0, -1200, 2400
N_PITCH = int((HI - LO) / BIN) + 1
PAD = N_PITCH
VOCAB = N_PITCH + 1
MAX_L = 1600
N_CHUNKS = 12
TOP_SEGMENTS = 3


def tokenize_with_times(pitch_hz, tonic_hz, frame_period_s, ds):
    """Voiced pitch -> (tokens, per-token timestamps in seconds)."""
    idx = np.where(pitch_hz > 0)[0]
    if tonic_hz <= 0 or len(idx) < 200:
        return None, None
    cents = 1200.0 * np.log2(pitch_hz[idx] / tonic_hz)
    toks = np.round((np.clip(cents, LO, HI) - LO) / BIN).astype(np.int64)
    times = idx * frame_period_s
    return toks[::ds], times[::ds]


def contour_from_audio(audio_path):
    """Essentia pitch + tonic -> (tokens, times). Matches the training pipeline."""
    import essentia.standard as es
    import predict_essentia as pe
    audio = es.MonoLoader(filename=audio_path, sampleRate=pe.SR)()
    tonic = pe.detect_tonic(audio)
    pitch, _ = es.PredominantPitchMelodia(frameSize=pe.MELODIA_FRAME, hopSize=pe.MELODIA_HOP)(audio)
    ds = max(1, round((pe.SR / pe.MELODIA_HOP) / 22.5))
    toks, times = tokenize_with_times(np.asarray(pitch, dtype=np.float64), tonic,
                                      pe.MELODIA_HOP / pe.SR, ds)
    return toks, times, float(tonic)


def load_ensemble(tag="v3", seeds=(0, 1, 2), n_classes=40, with_frame_head=True):
    import torch
    import torch.nn as nn
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.emb = nn.Embedding(VOCAB, 96, padding_idx=PAD)
            s.lstm = nn.LSTM(96, 192, batch_first=True, bidirectional=True)
            s.attn = nn.Linear(384, 1)
            s.fc = nn.Sequential(nn.Linear(384, 192), nn.ReLU(), nn.Dropout(0.4),
                                 nn.Linear(192, n_classes))
            if with_frame_head:
                s.frame = nn.Linear(384, 1)
        def forward(s, x, return_attn=False):
            o, _ = s.lstm(s.emb(x))
            logits = s.attn(o).squeeze(-1).masked_fill(x == PAD, float("-inf"))
            a = torch.softmax(logits, dim=1)
            pooled = (o * a.unsqueeze(-1)).sum(dim=1)
            out = s.fc(pooled)
            return (out, a) if return_attn else out

    models = []
    for s in seeds:
        net = Net().to(dev)
        sd = __import__("torch").load(os.path.join(DATA, f"deepsrgm_{tag}_s{s}.pt"),
                                      map_location=dev)
        net.load_state_dict(sd, strict=False)  # frame head ignored when absent
        net.eval()
        models.append(net)
    return {"models": models, "dev": dev}


def explain(ens, toks, times, classes):
    """Occlusion + attention evidence for one clip. Returns payload dict."""
    import torch
    dev = ens["dev"]
    L = min(len(toks), MAX_L)
    # center crop for determinism
    st = max(0, (len(toks) - L) // 2)
    toks_w = toks[st:st + L]
    times_w = times[st:st + L]
    x0 = np.full(MAX_L, PAD, dtype=np.int64)
    x0[:L] = toks_w

    def ens_probs_attn(X):
        xb = torch.tensor(X).to(dev)
        ps, ats = [], []
        with torch.no_grad():
            for m in ens["models"]:
                out, a = m(xb, return_attn=True)
                ps.append(out.softmax(1).cpu().numpy())
                ats.append(a.cpu().numpy())
        return np.mean(ps, axis=0), np.mean(ats, axis=0)

    base_p, attn = ens_probs_attn(x0[None])
    base_p, attn = base_p[0], attn[0]
    pred = int(np.argmax(base_p))

    # chunk boundaries over the REAL (non-pad) tokens
    bounds = np.linspace(0, L, N_CHUNKS + 1).astype(int)
    occl = np.stack([x0] * N_CHUNKS)
    for c in range(N_CHUNKS):
        occl[c, bounds[c]:bounds[c + 1]] = PAD
    occ_p, _ = ens_probs_attn(occl)
    drops = base_p[pred] - occ_p[:, pred]           # evidence per chunk
    attn_per_chunk = np.array([attn[bounds[c]:bounds[c + 1]].sum() for c in range(N_CHUNKS)])

    segs = []
    for c in np.argsort(drops)[::-1][:TOP_SEGMENTS]:
        if drops[c] <= 0:
            continue
        t0 = float(times_w[bounds[c]])
        t1 = float(times_w[min(bounds[c + 1], L) - 1])
        segs.append({"start_s": round(t0, 1), "end_s": round(t1, 1),
                     "evidence": round(float(drops[c]), 4),
                     "attention": round(float(attn_per_chunk[c]), 4)})
    return {
        "predicted": classes[pred],
        "confidence": round(float(base_p[pred]) * 100, 1),
        "evidence_segments": segs,
        "attention_profile": [round(float(v), 4) for v in attn_per_chunk],
    }


if __name__ == "__main__":
    import sys
    classes = json.load(open(os.path.join(DATA, "classes.json")))
    ens = load_ensemble()
    if len(sys.argv) > 1:                     # explain a real audio file
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        toks, times, tonic = contour_from_audio(sys.argv[1])
        if toks is None:
            print("not enough pitched audio"); sys.exit(1)
        print(f"tonic {tonic:.1f} Hz, {len(toks)} tokens")
        print(json.dumps(explain(ens, toks, times, classes), indent=2, ensure_ascii=False))
    else:                                     # demo on a cached YouTube contour
        z = np.load(os.path.join(DATA, "deepsrgm_youtube_seqs.npz"), allow_pickle=True)
        toks = z["s0"].astype(np.int64)
        # cached contours lack timestamps; approximate with the nominal token rate
        times = np.arange(len(toks)) / 22.5
        lab = classes[int(z["labels"][0])]
        print(f"demo on cached contour (true: {lab}; times approximate)")
        print(json.dumps(explain(ens, toks, times, classes), indent=2, ensure_ascii=False))
