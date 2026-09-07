"""W1 peak amplitude (no missingness).

Per-channel Wasserstein-1 between the distributions of detected peak
amplitudes. Peaks are found via scipy.signal.find_peaks per (sample, channel)
with user-injected kwargs (height/distance/prominence). Reports per-channel
array and channel-mean.

Inputs : real, fake of shape (N, T, C); peak_kwargs is a dict of kwargs for
         scipy.signal.find_peaks (or None for defaults); peak_kwargs_per_channel
         takes precedence and lets each channel use its own kwargs.
Returns: dict with 'per_channel' (list of length C) and 'mean' (float).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import wasserstein_distance

from utils.metric_utils import reduce_per_channel


def _gather_peak_amplitudes(X: np.ndarray, kwargs: dict) -> np.ndarray:
    """Concatenate peak-amplitude arrays across all samples for one channel."""
    amps = []
    for n in range(X.shape[0]):
        x = X[n]
        peaks, _ = find_peaks(x, **kwargs)
        if peaks.size:
            amps.append(x[peaks])
    return np.concatenate(amps) if amps else np.empty(0, dtype=float)


def w1_peak_amp(ori_data: np.ndarray, generated_data: np.ndarray,
                peak_kwargs: dict | None = None,
                peak_kwargs_per_channel: list[dict] | None = None) -> dict:
    real = np.asarray(ori_data, dtype=float)
    fake = np.asarray(generated_data, dtype=float)
    C = real.shape[-1]
    base = peak_kwargs or {}
    per = np.empty(C, dtype=float)
    for c in range(C):
        kw = (peak_kwargs_per_channel[c] if peak_kwargs_per_channel else base) or {}
        amps_r = _gather_peak_amplitudes(real[:, :, c], kw)
        amps_f = _gather_peak_amplitudes(fake[:, :, c], kw)
        if amps_r.size == 0 or amps_f.size == 0:
            per[c] = float("nan")
        else:
            per[c] = float(wasserstein_distance(amps_r, amps_f))
    return reduce_per_channel(per)
