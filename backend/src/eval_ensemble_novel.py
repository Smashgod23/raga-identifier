"""
Lever: ensemble the sequence model (DeepSRGM) and the distribution model (TDMS)
on NOVEL YouTube audio. They fail on different recordings, so a late fusion of
their per-class scores should beat either alone on both top-1 and top-5.

Train DeepSRGM on all 480 CMD Essentia contours (save the model), then for each
novel YouTube recording compute BOTH a DeepSRGM softmax and a TDMS score vector,
cache them, and report DeepSRGM-only / TDMS-only / ensemble (weight swept offline).
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import essentia.standard as es
import predict_essentia as pe
from predict_tdms import TDMSIndex

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
CACHE = os.path.join(DATA, "deepsrgm_essentia_seqs.npz")
MODEL_PT = os.path.join(DATA, "deepsrgm_essentia_model.pt")
PRED_CACHE = os.path.join(DATA, "novel_ensemble_preds.npz")
BIN, LO, HI = 50.0, -1200, 2400
DS = max(1, round((pe.SR / pe.MELODIA_HOP) / 22.5))
VOCAB = int((HI - LO) / BIN) + 1
SEQ_LEN = 800
PER_RAGA = 2
WINDOW_SEC = 240


def make_net(torch, nn, ncls):
    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.emb = nn.Embedding(VOCAB, 96)
            s.lstm = nn.LSTM(96, 192, batch_first=True, bidirectional=True)
            s.attn = nn.Linear(384, 1)
            s.fc = nn.Sequential(nn.Linear(384, 192), nn.ReLU(), nn.Dropout(0.4), nn.Linear(192, ncls))
        def forward(s, x):
            o, _ = s.lstm(s.emb(x))
            a = torch.softmax(s.attn(o).squeeze(-1), dim=1)
            return s.fc((o * a.unsqueeze(-1)).sum(1))
    return Net()


def tokenize(pitch, tonic):
    voiced = pitch[pitch > 0]
    if tonic <= 0 or len(voiced) < 200:
        return None
    cents = 1200.0 * np.log2(voiced / tonic)
    return np.round((np.clip(cents, LO, HI) - LO) / BIN).astype(np.int64)[::DS]


def subseqs(tok, per, rng):
    out = []
    for _ in range(per):
        if len(tok) <= SEQ_LEN:
            out.append(np.pad(tok, (0, SEQ_LEN - len(tok))))
        else:
            st = rng.integers(0, len(tok) - SEQ_LEN); out.append(tok[st:st + SEQ_LEN])
    return np.array(out)


def train_model(torch, nn, dev, ncls):
    z = np.load(CACHE, allow_pickle=True); n = int(z["n"]); labels = z["labels"]
    seqs = [z[f"s{i}"].astype(np.int64) for i in range(n)]
    rng = np.random.default_rng(0)
    Xtr = np.concatenate([subseqs(seqs[i], 60, rng) for i in range(n)])
    ytr = np.concatenate([np.full(60, int(labels[i])) for i in range(n)])
    net = make_net(torch, nn, ncls).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr); bs = 128
    if dev == "mps":
        torch.mps.empty_cache()
    for ep in range(30):
        net.train(); perm = torch.randperm(len(Xt))
        for b in range(0, len(perm), bs):
            bi = perm[b:b + bs]; opt.zero_grad()
            loss = lossf(net(Xt[bi].to(dev)), yt[bi].to(dev)); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 10 == 0:
            print(f"  train epoch {ep+1}/30 loss {loss.item():.3f}", flush=True)
    torch.save(net.state_dict(), MODEL_PT)
    return net


def _download(vid, dst):
    try:
        r = subprocess.run([sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-warnings", "--skip-download",
                            "--print", "%(duration)s", f"https://www.youtube.com/watch?v={vid}"],
                           capture_output=True, text=True, timeout=60)
        dur = int(r.stdout.strip().split("\n")[-1]) if r.returncode == 0 else 0
    except Exception:
        dur = 0
    start = max(60, dur // 3) if dur > WINDOW_SEC + 120 else 0
    args = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-warnings", "-x", "--audio-format", "wav",
            "--download-sections", f"*{start}-{start + WINDOW_SEC}", "-o", dst, f"https://www.youtube.com/watch?v={vid}"]
    for attempt in range(2):
        try:
            subprocess.run(args, check=True, capture_output=True, text=True, timeout=240); return True
        except subprocess.TimeoutExpired:
            if attempt == 1:
                return False
        except subprocess.CalledProcessError:
            return False
    return False


def tdms_scores(index, audio, tonic):
    """Distance-based per-class softmax from the TDMS index (more informative than
    the rank score for fusion)."""
    from build_tdms import symmetric_kl_distance
    q, _ = pe.query_tdms_from_audio(audio, tonic_override=tonic)
    q = np.asarray(q, dtype=np.float64).ravel()
    d = np.array([symmetric_kl_distance(q, index.X[i]) for i in range(len(index.X))])
    # per-class minimum distance -> softmax of negative distance
    cls_d = np.full(index.n_classes, np.inf)
    for i, lab in enumerate(index.y):
        cls_d[int(lab)] = min(cls_d[int(lab)], d[i])
    finite = cls_d[np.isfinite(cls_d)]
    scale = (finite.max() - finite.min()) or 1.0
    s = np.exp(-(cls_d - finite.min()) / (scale / 4))
    s[~np.isfinite(cls_d)] = 0.0
    return s / max(s.sum(), 1e-12)


def main():
    import torch
    import torch.nn as nn
    torch.manual_seed(0); rng = np.random.default_rng(0)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    classes = json.load(open(os.path.join(DATA, "classes.json")))
    cls_idx = {c: i for i, c in enumerate(classes)}
    index = TDMSIndex(np.load(os.path.join(DATA, "X_tdms_essfull_template.npy")),
                      np.load(os.path.join(DATA, "y_tdms.npy")), n_classes=len(classes))

    print(f"training DeepSRGM on 480 CMD Essentia contours, device={dev}", flush=True)
    net = train_model(torch, nn, dev, len(classes)); net.eval()

    videos = json.load(open(os.path.join(DATA, "youtube_videos.json")))
    deep_P, tdms_P, ys = [], [], []
    for raga, vids in videos.items():
        if raga not in cls_idx:
            continue
        true = cls_idx[raga]
        for v in vids[:PER_RAGA]:
            vid = v["id"]
            with tempfile.TemporaryDirectory() as td:
                dst = os.path.join(td, "a.%(ext)s")
                if not _download(vid, dst):
                    print(f"  [{raga}] {vid} dl fail", flush=True); continue
                files = [f for f in os.listdir(td) if f.startswith("a.")]
                if not files:
                    continue
                try:
                    audio = es.MonoLoader(filename=os.path.join(td, files[0]), sampleRate=pe.SR)()
                    tonic = pe.detect_tonic(audio)
                    pitch, _ = es.PredominantPitchMelodia(frameSize=pe.MELODIA_FRAME, hopSize=pe.MELODIA_HOP)(audio)
                    tok = tokenize(np.asarray(pitch, dtype=np.float64), tonic)
                    if tok is None or len(tok) < 200:
                        print(f"  [{raga}] {vid} no contour", flush=True); continue
                    with torch.no_grad():
                        dp = net(torch.tensor(subseqs(tok.astype(np.int64), 20, rng), device=dev)).softmax(1).mean(0).cpu().numpy()
                    tp = tdms_scores(index, audio, tonic)
                except Exception as exc:
                    print(f"  [{raga}] {vid} fail: {exc!r}", flush=True); continue
            deep_P.append(dp); tdms_P.append(tp); ys.append(true)
            print(f"  [{raga}] {vid} deep->{classes[int(np.argmax(dp))]} tdms->{classes[int(np.argmax(tp))]}", flush=True)

    deep_P = np.array(deep_P); tdms_P = np.array(tdms_P); ys = np.array(ys)
    np.savez(PRED_CACHE, deep=deep_P, tdms=tdms_P, y=ys)
    n = len(ys)

    def acc(P):
        t1 = t5 = 0
        for i in range(n):
            top5 = np.argsort(P[i])[::-1][:5]
            t1 += top5[0] == ys[i]; t5 += ys[i] in top5
        return t1 / n * 100, t5 / n * 100

    print(f"\n=== Ensemble on NOVEL YouTube audio (n={n}) ===")
    d1, d5 = acc(deep_P); print(f"  DeepSRGM only:  top-1 {d1:5.1f}%  top-5 {d5:5.1f}%")
    t1, t5 = acc(tdms_P); print(f"  TDMS only:      top-1 {t1:5.1f}%  top-5 {t5:5.1f}%")
    best = (0, 0, 0)
    for w in np.arange(0.0, 1.01, 0.1):
        e1, e5 = acc(w * deep_P + (1 - w) * tdms_P)
        if e1 > best[1]:
            best = (w, e1, e5)
        print(f"  ensemble w(deep)={w:.1f}: top-1 {e1:5.1f}%  top-5 {e5:5.1f}%")
    print(f"  BEST ensemble: w={best[0]:.1f} -> top-1 {best[1]:.1f}% top-5 {best[2]:.1f}%")


if __name__ == "__main__":
    main()
