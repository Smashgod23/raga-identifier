"""
The number that actually matters: DeepSRGM (Essentia pitch) top-1 on NOVEL audio.

Train the deployable model on ALL 480 CompMusic Essentia contours, then download
YouTube recordings (novel to the model — it never saw them) , extract their
Essentia contours the same way, and classify. Labels come from youtube_videos.json
(auto-searched, so a few points of label noise are expected).
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

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
CACHE = os.path.join(DATA, "deepsrgm_essentia_seqs.npz")
BIN, LO, HI = 50.0, -1200, 2400
DS = max(1, round((pe.SR / pe.MELODIA_HOP) / 22.5))
SEQ_LEN = 800
PER_RAGA = 2
WINDOW_SEC = 240


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


def _probe(vid):
    try:
        r = subprocess.run([sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-warnings",
                            "--skip-download", "--print", "%(duration)s",
                            f"https://www.youtube.com/watch?v={vid}"], capture_output=True, text=True, timeout=60)
        return int(r.stdout.strip().split("\n")[-1]) if r.returncode == 0 else 0
    except Exception:
        return 0


def yt_contour(vid):
    dur = _probe(vid); start = max(60, dur // 3) if dur > WINDOW_SEC + 120 else 0
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "a.%(ext)s")
        args = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-warnings", "-x", "--audio-format", "wav",
                "--download-sections", f"*{start}-{start + WINDOW_SEC}", "-o", out, f"https://www.youtube.com/watch?v={vid}"]
        for attempt in range(2):
            try:
                subprocess.run(args, check=True, capture_output=True, text=True, timeout=240); break
            except subprocess.TimeoutExpired:
                if attempt == 1:
                    return None
            except subprocess.CalledProcessError:
                return None
        files = [f for f in os.listdir(td) if f.startswith("a.")]
        if not files:
            return None
        audio = es.MonoLoader(filename=os.path.join(td, files[0]), sampleRate=pe.SR)()
        tonic = pe.detect_tonic(audio)
        pitch, _ = es.PredominantPitchMelodia(frameSize=pe.MELODIA_FRAME, hopSize=pe.MELODIA_HOP)(audio)
    return tokenize(np.asarray(pitch, dtype=np.float64), tonic)


def main():
    import torch
    import torch.nn as nn
    torch.manual_seed(0); np.random.seed(0); rng = np.random.default_rng(0)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    classes = json.load(open(os.path.join(DATA, "classes.json")))
    cls_idx = {c: i for i, c in enumerate(classes)}
    VOCAB = int((HI - LO) / BIN) + 1

    z = np.load(CACHE, allow_pickle=True); n = int(z["n"]); labels = z["labels"]
    seqs = [z[f"s{i}"].astype(np.int64) for i in range(n)]
    print(f"training on all {n} CMD Essentia contours, device={dev}", flush=True)

    Xtr = np.concatenate([subseqs(seqs[i], 60, rng) for i in range(n)])
    ytr = np.concatenate([np.full(60, int(labels[i])) for i in range(n)])

    class Net(nn.Module):
        def __init__(self, vocab, ncls):
            super().__init__()
            self.emb = nn.Embedding(vocab, 96)
            self.lstm = nn.LSTM(96, 192, batch_first=True, bidirectional=True)
            self.attn = nn.Linear(384, 1)
            self.fc = nn.Sequential(nn.Linear(384, 192), nn.ReLU(), nn.Dropout(0.4), nn.Linear(192, ncls))
        def forward(self, x):
            o, _ = self.lstm(self.emb(x))
            a = torch.softmax(self.attn(o).squeeze(-1), dim=1)
            return self.fc((o * a.unsqueeze(-1)).sum(1))

    net = Net(VOCAB, len(classes)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    lossf = nn.CrossEntropyLoss()
    # Keep the full training set on CPU and move each batch to the device; the
    # whole 28800x800 tensor on MPS plus activations overruns unified memory.
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr); bs = 128
    if dev == "mps":
        torch.mps.empty_cache()
    for ep in range(30):
        net.train(); perm = torch.randperm(len(Xt))
        for b in range(0, len(perm), bs):
            bi = perm[b:b + bs]; opt.zero_grad()
            xb = Xt[bi].to(dev); yb = yt[bi].to(dev)
            loss = lossf(net(xb), yb); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1}/30 loss {loss.item():.3f}", flush=True)

    # novel YouTube eval
    videos = json.load(open(os.path.join(DATA, "youtube_videos.json")))
    net.eval(); t1 = t5 = n_eval = 0
    for raga, vids in videos.items():
        if raga not in cls_idx:
            continue
        true = cls_idx[raga]
        for v in vids[:PER_RAGA]:
            vid = v["id"]
            try:
                tok = yt_contour(vid)
            except Exception as exc:
                print(f"  [{raga}] {vid} extract failed: {exc!r}", flush=True); continue
            if tok is None or len(tok) < 200:
                print(f"  [{raga}] {vid} no contour", flush=True); continue
            with torch.no_grad():
                Xte = torch.tensor(subseqs(tok.astype(np.int64), 20, rng), device=dev)
                p = net(Xte).softmax(1).mean(0).cpu().numpy()
            top5 = np.argsort(p)[::-1][:5]; n_eval += 1
            t1 += top5[0] == true; t5 += true in top5
            print(f"  [{raga}] {vid} -> {classes[top5[0]]} {'OK' if top5[0]==true else 'X'} (top5 {'Y' if true in top5 else 'N'})", flush=True)

    print(f"\n=== DeepSRGM (Essentia pitch) on NOVEL YouTube audio ===")
    print(f"  top-1 {t1/max(n_eval,1)*100:5.1f}%   top-5 {t5/max(n_eval,1)*100:5.1f}%   (n={n_eval})")
    print(f"  (TDMS on novel: ~50% top-1; v1 on novel: ~17%)")


if __name__ == "__main__":
    main()
