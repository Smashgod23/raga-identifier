"""
One-time converter: tonic_detector_v1.pt (PyTorch) -> tonic_detector_v1.npz (numpy).

The deployed image deliberately omits PyTorch to keep the Docker image
under 1 GB. We instead extract the tiny MLP's weights, BatchNorm
running stats, and biases into a numpy archive and run the forward
pass in pure numpy at inference time.

Run from backend/:
  source venv/bin/activate
  python src/convert_tonic_detector.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

PT_PATH = "models/tonic_detector_v1.pt"
NPZ_PATH = "models/tonic_detector_v1.npz"


def main() -> None:
    if not os.path.exists(PT_PATH):
        print(f"missing {PT_PATH}", file=sys.stderr)
        sys.exit(1)

    sd = torch.load(PT_PATH, map_location="cpu", weights_only=True)
    # Architecture: net = Sequential(
    #   0: Linear(124, 64)
    #   1: BatchNorm1d(64)        + ReLU (2) + Dropout (3)
    #   4: Linear(64, 32)
    #   5: BatchNorm1d(32)        + ReLU (6) + Dropout (7)
    #   8: Linear(32, 1)
    # )
    out = {
        # Linear 0
        "fc0_W": sd["net.0.weight"].numpy().astype(np.float32),
        "fc0_b": sd["net.0.bias"].numpy().astype(np.float32),
        # BN 1
        "bn1_gamma": sd["net.1.weight"].numpy().astype(np.float32),
        "bn1_beta": sd["net.1.bias"].numpy().astype(np.float32),
        "bn1_mean": sd["net.1.running_mean"].numpy().astype(np.float32),
        "bn1_var": sd["net.1.running_var"].numpy().astype(np.float32),
        # Linear 4
        "fc4_W": sd["net.4.weight"].numpy().astype(np.float32),
        "fc4_b": sd["net.4.bias"].numpy().astype(np.float32),
        # BN 5
        "bn5_gamma": sd["net.5.weight"].numpy().astype(np.float32),
        "bn5_beta": sd["net.5.bias"].numpy().astype(np.float32),
        "bn5_mean": sd["net.5.running_mean"].numpy().astype(np.float32),
        "bn5_var": sd["net.5.running_var"].numpy().astype(np.float32),
        # Linear 8
        "fc8_W": sd["net.8.weight"].numpy().astype(np.float32),
        "fc8_b": sd["net.8.bias"].numpy().astype(np.float32),
        "bn_eps": np.float32(1e-5),  # PyTorch BatchNorm default
    }
    np.savez(NPZ_PATH, **out)
    sizes = {k: tuple(v.shape) for k, v in out.items() if isinstance(v, np.ndarray) and v.ndim}
    print(f"Wrote {NPZ_PATH} ({os.path.getsize(NPZ_PATH)} bytes)")
    for k, s in sizes.items():
        print(f"  {k}: {s}")


if __name__ == "__main__":
    main()
