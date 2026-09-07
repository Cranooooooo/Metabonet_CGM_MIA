"""W1 inter-peak interval (no missingness).

Per-channel Wasserstein-1 between the distributions of inter-peak intervals
(in samples). Peak indices are found via scipy.signal.find_peaks per
(sample, channel), then np.diff(peak_idx) gives the IPIs. Reports per-channel
array and channel-mean.

Inputs : real, fake of shape (N, T, C); peak_kwargs and
         peak_kwargs_per_channel as in w1_peak_amp; fs (Hz) is optional and
         only used to convert the score to seconds — the W1 itself is computed
         on whatever unit IPIs are given in (samples by default).
Returns: dict with 'per_channel' (list of length C) and 'mean' (float).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import wasserstein_distance

from utils.metric_utils import reduce_per_channel


def _gather_ipis(X: np.ndarray, kwargs: dict, fs: float) -> np.ndarray:
    ipis = []
    for n in range(X.shape[0]):
        peaks, _ = find_peaks(X[n], **kwargs)
        if peaks.size >= 2:
            ipis.append(np.diff(peaks).astype(float) / fs)
    return np.concatenate(ipis) if ipis else np.empty(0, dtype=float)


def w1_ipi(ori_data: np.ndarray, generated_data: np.ndarray,
           fs: float = 1.0,
           peak_kwargs: dict | None = None,
           peak_kwargs_per_channel: list[dict] | None = None) -> dict:
    real = np.asarray(ori_data, dtype=float)
    fake = np.asarray(generated_data, dtype=float)
    C = real.shape[-1]
    base = peak_kwargs or {}
    per = np.empty(C, dtype=float)
    for c in range(C):
        kw = (peak_kwargs_per_channel[c] if peak_kwargs_per_channel else base) or {}
        r_ipis = _gather_ipis(real[:, :, c], kw, fs)
        f_ipis = _gather_ipis(fake[:, :, c], kw, fs)
        if r_ipis.size == 0 or f_ipis.size == 0:
            per[c] = float("nan")
        else:
            per[c] = float(wasserstein_distance(r_ipis, f_ipis))
    return reduce_per_channel(per)
