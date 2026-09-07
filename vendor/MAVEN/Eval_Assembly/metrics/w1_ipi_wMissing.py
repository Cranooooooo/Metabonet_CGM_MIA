"""W1 inter-peak interval (with missingness — real-only mask).

Generated samples are assumed fully observed. Real-side IPIs are computed
only within contiguous observed segments per (sample, channel) — intervals
that would span a missing block are NOT taken (those would be mask-artifacts,
not real periods). Fake-side IPIs are computed across the full signal.

Falls back to the no-missing path when ori_mask is fully observed.

Inputs : real, fake of shape (N, T, C); ori_mask of shape (N, T, C);
         fs (Hz), peak_kwargs, peak_kwargs_per_channel as in the no-missing
         version.
Returns: dict with 'per_channel' (list of length C) and 'mean' (float).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import wasserstein_distance

from utils.mask_utils import is_fully_observed, per_sample_observed_indices
from utils.metric_utils import reduce_per_channel
from metrics.w1_ipi_woMissing import w1_ipi as _ipi_clean


def _gather_ipis_full(X: np.ndarray, kwargs: dict, fs: float) -> np.ndarray:
    ipis = []
    for n in range(X.shape[0]):
        peaks, _ = find_peaks(X[n], **kwargs)
        if peaks.size >= 2:
            ipis.append(np.diff(peaks).astype(float) / fs)
    return np.concatenate(ipis) if ipis else np.empty(0, dtype=float)


def _gather_ipis_masked(X: np.ndarray, M: np.ndarray, kwargs: dict, fs: float) -> np.ndarray:
    runs_per_sample = per_sample_observed_indices(M[..., None])
    ipis = []
    for n, per_c in enumerate(runs_per_sample):
        for run in per_c[0]:
            seg = X[n, run]
            if seg.size < 3:
                continue
            peaks, _ = find_peaks(seg, **kwargs)
            if peaks.size >= 2:
                global_idx = run[peaks]
                ipis.append(np.diff(global_idx).astype(float) / fs)
    return np.concatenate(ipis) if ipis else np.empty(0, dtype=float)


def w1_ipi_masked(ori_data: np.ndarray, generated_data: np.ndarray,
                  ori_mask: np.ndarray | None = None,
                  fs: float = 1.0,
                  peak_kwargs: dict | None = None,
                  peak_kwargs_per_channel: list[dict] | None = None) -> dict:
    if is_fully_observed(ori_mask):
        return _ipi_clean(ori_data, generated_data, fs=fs,
                          peak_kwargs=peak_kwargs,
                          peak_kwargs_per_channel=peak_kwargs_per_channel)

    real = np.asarray(ori_data, dtype=float)
    fake = np.asarray(generated_data, dtype=float)
    Mo = np.asarray(ori_mask).astype(bool)

    C = real.shape[-1]
    base = peak_kwargs or {}
    per = np.empty(C, dtype=float)
    for c in range(C):
        kw = (peak_kwargs_per_channel[c] if peak_kwargs_per_channel else base) or {}
        r_ipis = _gather_ipis_masked(real[:, :, c], Mo[:, :, c], kw, fs)
        f_ipis = _gather_ipis_full(fake[:, :, c], kw, fs)
        if r_ipis.size == 0 or f_ipis.size == 0:
            per[c] = float("nan")
        else:
            per[c] = float(wasserstein_distance(r_ipis, f_ipis))
    return reduce_per_channel(per)
