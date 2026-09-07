"""W1 marginal (with missingness — real-only mask).

Generated samples are assumed fully observed. Per channel, the real-side
empirical distribution uses only observed (and finite) values, while the
fake side uses every value. Falls back to the no-missing path when ori_mask
is fully observed.

Inputs : real, fake of shape (N, T, C); ori_mask of shape (N, T, C).
Returns: dict with 'per_channel' (list of length C) and 'mean' (float).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

from utils.mask_utils import is_fully_observed
from utils.metric_utils import reduce_per_channel
from metrics.w1_marginal_woMissing import w1_marginal as _w1_clean


def w1_marginal_masked(ori_data: np.ndarray, generated_data: np.ndarray,
                       ori_mask: np.ndarray | None = None) -> dict:
    if is_fully_observed(ori_mask):
        return _w1_clean(ori_data, generated_data)

    real = np.asarray(ori_data, dtype=float)
    fake = np.asarray(generated_data, dtype=float)
    Mo = np.asarray(ori_mask).astype(bool)

    C = real.shape[-1]
    per = np.empty(C, dtype=float)
    for c in range(C):
        r = real[..., c].reshape(-1)
        f = fake[..., c].reshape(-1)
        rm = Mo[..., c].reshape(-1)
        r = r[rm & np.isfinite(r)]
        f = f[np.isfinite(f)]
        if r.size == 0 or f.size == 0:
            per[c] = float("nan")
        else:
            per[c] = float(wasserstein_distance(r, f))
    return reduce_per_channel(per)
