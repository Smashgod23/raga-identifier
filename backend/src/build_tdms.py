"""
Time-Delayed Melody Surface (TDMS) feature extraction.

Implements the algorithm from:
  Gulati, Serra, Ganguli, Senturk, Serra. "Time-delayed melody surfaces for
  raga recognition." ISMIR 2016.

A TDMS is a 2D joint distribution of (pitch-class at time t, pitch-class
at time t+tau), built from the tonic-normalized predominant pitch contour.
It captures phrase order (which note follows which note), which a 1D
pitch-class histogram throws away. On the same CompMusic 480-recording
40-raga Carnatic dataset, the paper reports 86.7% top-1 with a k-NN
classifier using symmetric KL or Bhattacharyya distance.

Default hyperparameters match the paper:
  eta = 120 bins (10 cents per bin, octave-folded)
  tau = 0.3 seconds
  alpha = 0.75 (power compression)
  sigma = 2 bins (circular Gaussian smoothing)
  L1 normalization

The output is a (eta, eta) float32 matrix that sums to 1.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def cents_from_pitch(pitches_hz: np.ndarray, tonic_hz: float) -> np.ndarray:
    """Hz to cents above tonic, with negative values for sub-tonic pitches.

    Returns NaN for unvoiced frames (pitch <= 0). Callers must filter NaN.
    """
    out = np.full(pitches_hz.shape, np.nan, dtype=np.float64)
    voiced = pitches_hz > 0
    out[voiced] = 1200.0 * np.log2(pitches_hz[voiced] / tonic_hz)
    return out


def bin_indices(cents: np.ndarray, eta: int = 120) -> np.ndarray:
    """Octave-wrapped integer bin index for each cent value.

    B(x) = floor((eta * x / 1200) mod eta). Numpy's modulo wraps negatives
    correctly to [0, eta).
    """
    return np.floor((eta * cents / 1200.0) % eta).astype(np.int64)


def compute_tdms(
    timestamps: np.ndarray,
    pitches_hz: np.ndarray,
    tonic_hz: float,
    *,
    eta: int = 120,
    tau_seconds: float = 0.3,
    alpha: float = 0.75,
    sigma: float = 2.0,
) -> np.ndarray | None:
    """Build a TDMS from a pitch contour.

    Args:
        timestamps: 1D array of frame times in seconds.
        pitches_hz: 1D array of predominant-pitch values in Hz. Unvoiced
            frames are <= 0 and are excluded.
        tonic_hz: detected tonic frequency in Hz.
        eta: histogram size per dimension.
        tau_seconds: delay between the two pitch samples joined in each cell.
        alpha: power-compression exponent applied element-wise.
        sigma: circular Gaussian smoothing kernel std (in bins). Set to None
            or <= 0 to disable smoothing.

    Returns:
        (eta, eta) float32 matrix that sums to 1, or None if the contour is
        too short to produce any valid (t, t-tau) pair.
    """
    if len(timestamps) < 2 or len(pitches_hz) != len(timestamps):
        return None

    dt = float(np.median(np.diff(timestamps)))
    if dt <= 0:
        return None
    tau_frames = max(1, int(round(tau_seconds / dt)))
    if len(pitches_hz) <= tau_frames:
        return None

    cents = cents_from_pitch(pitches_hz, tonic_hz)
    bins = bin_indices(np.nan_to_num(cents, nan=0.0), eta=eta)
    voiced = ~np.isnan(cents)

    # A cell at (i, j) counts every t such that frame t is voiced with bin i
    # AND frame t-tau is voiced with bin j.
    bins_t = bins[tau_frames:]
    bins_t_minus_tau = bins[:-tau_frames]
    valid = voiced[tau_frames:] & voiced[:-tau_frames]
    if not valid.any():
        return None

    bi = bins_t[valid]
    bj = bins_t_minus_tau[valid]
    # np.add.at is the vectorized scatter-add. Equivalent to a loop over
    # (bi[k], bj[k]) incrementing a counter, but ~30x faster.
    surface = np.zeros((eta, eta), dtype=np.float64)
    np.add.at(surface, (bi, bj), 1.0)

    # Power compression. alpha < 1 lifts transitory regions relative to the
    # tall peaks at the steady svaras.
    surface = surface ** alpha

    # Circular Gaussian smoothing (cyclic in both dimensions because pitch
    # classes are cyclic at the octave).
    if sigma is not None and sigma > 0:
        surface = gaussian_filter(surface, sigma=sigma, mode="wrap")

    # L1 normalize so the matrix is a discrete probability distribution.
    total = surface.sum()
    if total <= 0:
        return None
    surface /= total

    return surface.astype(np.float32)


# ---------------------------------------------------------------------------
# Distance measures from the paper.
# ---------------------------------------------------------------------------

def frobenius_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def symmetric_kl_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Symmetric KL: D(a, b) + D(b, a), with eps to avoid log(0)."""
    ax = a + eps
    bx = b + eps
    return float(np.sum(ax * np.log(ax / bx)) + np.sum(bx * np.log(bx / ax)))


def bhattacharyya_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    bc = float(np.sum(np.sqrt(a * b)))
    return -float(np.log(bc + eps))
