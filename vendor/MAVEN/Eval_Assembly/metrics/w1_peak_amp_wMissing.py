"""W1 peak amplitude (with missingness — real-only mask).

Generated samples are assumed fully observed. Per channel, real-side peaks
are detected only on contiguous observed segments (so a peak interior to a
missing-bordered region is discarded — those are mask discontinuities, not
real extrema). Fake-side peaks are detected on the full signal.

Falls back to the no-missing path when ori_mask is fully observed.

Inputs : real, fake of shape (N, T, C); ori_mask of shape (N, T, C);
         peak_kwargs / peak_kwargs_per_channel as in the no-missing version.
Returns: dict with 'per_channel' (list of length C) and 'mean' (float).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import wasserstein_distance

from utils.mask_utils import is_fully_observed, per_sample_observed_indices
from utils.metric_utils import reduce_per_channel
from metrics.w1_peak_amp_woMissing import w1_peak_amp as _peak_clean


def _gather_peak_amplitudes_full(X: np.ndarray, kwargs: dict) -> np.ndarray:
    """No-mask: scan each (N, T) sample's full series, collect all peak amps."""
    amps = []
    for n in range(X.shape[0]):
        seg = X[n]
        peaks, _ = find_peaks(seg, **kwargs)
        if peaks.size:
            amps.append(seg[peaks])
    return np.concatenate(amps) if amps else np.empty(0, dtype=float)


def _gather_peak_amplitudes_masked(X: np.ndarray, M: np.ndarray, kwargs: dict) -> np.ndarray:
    """Per-channel: detect peaks within each maximal observed run."""
    runs_per_sample = per_sample_observed_indices(M[..., None])
    amps = []
    for n, per_c in enumerate(runs_per_sample):
        for run in per_c[0]:
            seg = X[n, run]
            if seg.size < 3:
                continue
            peaks, _ = find_peaks(seg, **kwargs)
            if peaks.size:
                amps.append(seg[peaks])
    return np.concatenate(amps) if amps else np.empty(0, dtype=float)


def w1_peak_amp_masked(ori_data: np.ndarray, generated_data: np.ndarray,
                       ori_mask: np.ndarray | None = None,
                       peak_kwargs: dict | None = None,
                       peak_kwargs_per_channel: list[dict] | None = None) -> dict:
    if is_fully_observed(ori_mask):
        return _peak_clean(ori_data, generated_data,
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
        amps_r = _gather_peak_amplitudes_masked(real[:, :, c], Mo[:, :, c], kw)
        amps_f = _gather_peak_amplitudes_full(fake[:, :, c], kw)
        if amps_r.size == 0 or amps_f.size == 0:
            per[c] = float("nan")
        else:
            per[c] = float(wasserstein_distance(amps_r, amps_f))
    return reduce_per_channel(per)
