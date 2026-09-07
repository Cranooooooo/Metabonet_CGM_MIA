"""W1 marginal (no missingness).

Per-channel Wasserstein-1 distance between flattened real and synthetic
value distributions. Reports per-channel array and channel-mean.

Inputs : real, fake of shape (N, T, C).
Returns: dict with 'per_channel' (list of length C) and 'mean' (float).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

from utils.metric_utils import reduce_per_channel


def w1_marginal(ori_data: np.ndarray, generated_data: np.ndarray) -> dict:
    real = np.asarray(ori_data, dtype=float)
    fake = np.asarray(generated_data, dtype=float)
    C = real.shape[-1]
    per = np.empty(C, dtype=float)
    for c in range(C):
        r = real[..., c].reshape(-1)
        f = fake[..., c].reshape(-1)
        r = r[np.isfinite(r)]
        f = f[np.isfinite(f)]
        if r.size == 0 or f.size == 0:
            per[c] = float("nan")
        else:
            per[c] = float(wasserstein_distance(r, f))
    return reduce_per_channel(per)
