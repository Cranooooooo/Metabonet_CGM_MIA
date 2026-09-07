"""ACF loss (with missingness — real-only mask).

Generated samples are assumed fully observed. Per channel, the real-side ACF
at each lag l is computed pairwise-complete (only (t, t+l) pairs where both
are observed); fake's ACF uses the standard sample-mean estimator. Lags with
no valid pair on the real side yield NaN and are dropped from the L2 sum.

Falls back to the no-missing path when ori_mask is fully observed.

Inputs : real, fake of shape (N, T, C); ori_mask of shape (N, T, C).
Returns: dict with 'per_channel' (list of length C) and 'mean' (float).
"""
from __future__ import annotations

import numpy as np

from utils.mask_utils import is_fully_observed
from utils.metric_utils import reduce_per_channel
from metrics.acf_loss_woMissing import acf_loss as _acf_clean


def _masked_acf(X: np.ndarray, M: np.ndarray, max_lag: int) -> np.ndarray:
    """Pairwise-complete ACF at lags 0..max_lag-1, returns (max_lag, C)."""
    N, T, C = X.shape
    out = np.full((max_lag, C), np.nan, dtype=np.float64)
    for c in range(C):
        xc = X[:, :, c].astype(np.float64)
        mc = M[:, :, c].astype(bool)
        if mc.sum() < 2:
            continue
        mu = xc[mc].mean()
        xc_centered = np.where(mc, xc - mu, 0.0)
        var = (xc_centered[mc] ** 2).mean()
        if var < 1e-12:
            continue
        out[0, c] = 1.0
        for l in range(1, max_lag):
            if l >= T:
                continue
            both = mc[:, l:] & mc[:, :-l]
            n_pairs = int(both.sum())
            if n_pairs < 1:
                continue
            num = (xc_centered[:, l:] * xc_centered[:, :-l] * both).sum() / n_pairs
            out[l, c] = num / var
    return out


def acf_loss_masked(ori_data: np.ndarray, generated_data: np.ndarray,
                    ori_mask: np.ndarray | None = None,
                    max_lag: int = 64) -> dict:
    if is_fully_observed(ori_mask):
        return _acf_clean(ori_data, generated_data, max_lag=max_lag)

    real = np.asarray(ori_data, dtype=np.float64)
    fake = np.asarray(generated_data, dtype=np.float64)
    Mo = np.asarray(ori_mask).astype(bool)
    Mg = np.ones_like(fake, dtype=bool)

    L = max(1, min(max_lag, real.shape[1]))
    a_real = _masked_acf(real, Mo, L)
    a_fake = _masked_acf(fake, Mg, L)

    C = real.shape[-1]
    per = np.empty(C, dtype=float)
    for c in range(C):
        diff = a_real[:, c] - a_fake[:, c]
        valid = np.isfinite(diff)
        if not valid.any():
            per[c] = float("nan")
        else:
            per[c] = float(np.sqrt(np.sum(diff[valid] ** 2)))
    return reduce_per_channel(per)
